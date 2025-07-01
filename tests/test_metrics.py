"""
Tests for evaluation metrics.
"""

import pytest
import torch
import numpy as np
from typing import List, Dict, Any
import editdistance

from src.evaluation.metrics import (
    DetectionEvaluator,
    RecognitionEvaluator,
    FingerspellingEvaluator,
    create_evaluator
)
from src.utils.types import TemporalSegment


class TestBasicMetrics:
    """Test cases for basic metric functions."""
    
    def test_edit_distance(self) -> None:
        """Test edit distance calculation using editdistance library."""
        # Same sequences
        assert editdistance.eval("hello", "hello") == 0
        
        # One substitution
        assert editdistance.eval("hello", "hallo") == 1
        
        # One insertion
        assert editdistance.eval("hello", "helloo") == 1
        
        # One deletion
        assert editdistance.eval("hello", "hell") == 1
        
        # Multiple operations
        assert editdistance.eval("kitten", "sitting") == 3
        
        # Empty strings
        assert editdistance.eval("", "") == 0
        assert editdistance.eval("hello", "") == 5
        assert editdistance.eval("", "world") == 5
    
    def test_calculate_iou(self) -> None:
        """Test IoU calculation between segments."""
        evaluator = DetectionEvaluator()
        
        # Perfect overlap
        seg1 = TemporalSegment(10, 20)
        seg2 = TemporalSegment(10, 20)
        assert evaluator.calculate_iou(seg1, seg2) == 1.0
        
        # Partial overlap
        seg1 = TemporalSegment(10, 20)
        seg2 = TemporalSegment(15, 25)
        expected_iou = 5 / 15  # intersection=5, union=15
        assert abs(evaluator.calculate_iou(seg1, seg2) - expected_iou) < 1e-6
        
        # No overlap
        seg1 = TemporalSegment(10, 20)
        seg2 = TemporalSegment(25, 35)
        assert evaluator.calculate_iou(seg1, seg2) == 0.0
        
        # One segment contains the other
        seg1 = TemporalSegment(10, 30)
        seg2 = TemporalSegment(15, 25)
        expected_iou = 10 / 20  # intersection=10, union=20
        assert abs(evaluator.calculate_iou(seg1, seg2) - expected_iou) < 1e-6


class TestAPMetrics:
    """Test cases for Average Precision metrics."""
    
    def create_mock_prediction_segments(self) -> List[TemporalSegment]:
        """Create mock prediction segments for testing."""
        return [
            TemporalSegment(10, 20),  # confidence would be tracked separately
            TemporalSegment(25, 35),
            TemporalSegment(40, 50)
        ]
    
    def create_mock_ground_truth_segments(self) -> List[TemporalSegment]:
        """Create mock ground truth segments for testing."""
        return [
            TemporalSegment(12, 22),  # Overlaps with first prediction
            TemporalSegment(60, 70)   # No overlap with predictions
        ]
    
    def test_calculate_ap_at_iou(self) -> None:
        """Test AP@IoU calculation."""
        evaluator = DetectionEvaluator()
        predictions = self.create_mock_prediction_segments()
        ground_truth = self.create_mock_ground_truth_segments()
        
        # Test with different IoU thresholds
        ap_low = evaluator.calculate_ap_iou(predictions, ground_truth, iou_threshold=0.1)
        ap_high = evaluator.calculate_ap_iou(predictions, ground_truth, iou_threshold=0.8)
        
        assert isinstance(ap_low, float)
        assert isinstance(ap_high, float)
        assert 0.0 <= ap_low <= 1.0
        assert 0.0 <= ap_high <= 1.0
        
        # Lower IoU threshold should generally give higher AP
        assert ap_low >= ap_high
    
    def test_recognition_accuracy(self) -> None:
        """Test recognition accuracy calculation."""
        evaluator = RecognitionEvaluator()
        
        # Perfect match
        acc = evaluator.calculate_accuracy('hello', 'hello')
        assert acc == 1.0
        
        # No match - check that accuracy is low but not necessarily 0.0
        acc = evaluator.calculate_accuracy('hello', 'world')
        assert acc < 0.5  # Should be low accuracy, not necessarily 0.0
        
        # Partial match
        acc = evaluator.calculate_accuracy('hello', 'hallo')
        assert 0.0 < acc < 1.0
    
    def test_empty_predictions(self) -> None:
        """Test metrics with empty predictions."""
        evaluator = DetectionEvaluator()
        ground_truth = self.create_mock_ground_truth_segments()
        
        ap_iou = evaluator.calculate_ap_iou([], ground_truth, iou_threshold=0.5)
        
        assert ap_iou == 0.0
    
    def test_empty_ground_truth(self) -> None:
        """Test metrics with empty ground truth."""
        evaluator = DetectionEvaluator()
        predictions = self.create_mock_prediction_segments()
        
        # When there's no ground truth, AP should be 1.0 if no predictions, 0.0 if predictions exist
        ap_iou = evaluator.calculate_ap_iou(predictions, [], iou_threshold=0.5)
        
        assert isinstance(ap_iou, float)
        assert ap_iou == 0.0  # Based on implementation


class TestFingerspellingEvaluator:
    """Test cases for the full fingerspelling evaluator."""
    
    def test_evaluator_creation(self) -> None:
        """Test creating evaluator instances."""
        evaluator = FingerspellingEvaluator()
        assert evaluator is not None
        
        # Test with custom thresholds
        evaluator = FingerspellingEvaluator(
            iou_thresholds=[0.1, 0.5],
            acc_thresholds=[0.5, 0.8]
        )
        assert evaluator is not None
    
    def test_simple_msa_calculation(self) -> None:
        """Test simple MSA calculation using the evaluator."""
        evaluator = FingerspellingEvaluator()
        
        # Create simple test data
        pred_segments = [TemporalSegment(0, 10), TemporalSegment(10, 20)]
        pred_recognitions = ['hello', 'world']
        gt_sequence = 'helloworld'
        video_length = 20
        
        msa = evaluator.calculate_msa(
            pred_segments, pred_recognitions, gt_sequence, video_length
        )
        
        assert isinstance(msa, float)
        assert 0.0 <= msa <= 1.0
    
    def test_recognition_decode(self) -> None:
        """Test recognition decoder."""
        evaluator = RecognitionEvaluator()
        
        # Create dummy log probabilities
        # Assuming 29 classes (26 letters + blank + start + end)
        seq_len = 5
        num_classes = 29
        log_probs = torch.randn(seq_len, num_classes)
        
        # This should not crash
        decoded = evaluator.decode_prediction(log_probs)
        assert isinstance(decoded, str)


class TestCreateEvaluator:
    """Test the evaluator factory function."""
    
    def test_create_evaluator_default(self) -> None:
        """Test creating evaluator with defaults."""
        evaluator = create_evaluator()
        assert isinstance(evaluator, FingerspellingEvaluator)
    
    def test_create_evaluator_custom_thresholds(self) -> None:
        """Test creating evaluator with custom thresholds."""
        evaluator = create_evaluator(
            iou_thresholds=[0.1, 0.5], 
            acc_thresholds=[0.0, 0.5]
        )
        assert isinstance(evaluator, FingerspellingEvaluator)


class TestMetricEdgeCases:
    """Test edge cases for metrics."""
    
    def test_invalid_segments(self) -> None:
        """Test handling of invalid temporal segments."""
        evaluator = DetectionEvaluator()
        
        # Test with very short segment
        seg1 = TemporalSegment(10, 11)  # Minimal length segment
        seg2 = TemporalSegment(10, 15)
        
        iou = evaluator.calculate_iou(seg1, seg2)
        assert isinstance(iou, float)
        assert 0.0 <= iou <= 1.0
        
        # Test with non-overlapping segments
        seg1 = TemporalSegment(10, 15)
        seg2 = TemporalSegment(20, 25)
        
        iou = evaluator.calculate_iou(seg1, seg2)
        assert iou == 0.0  # No overlap
    
    def test_unicode_sequences(self) -> None:
        """Test handling of unicode characters in sequences."""
        evaluator = RecognitionEvaluator()
        
        # Test accuracy with unicode
        acc = evaluator.calculate_accuracy('héllo', 'hello')
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0
        
        # Test edit distance with unicode
        distance = editdistance.eval('héllo', 'hello')
        assert distance == 1  # One character difference


if __name__ == "__main__":
    pytest.main([__file__])
