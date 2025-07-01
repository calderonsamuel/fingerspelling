"""
Tests for loss functions.
"""

import pytest
import torch
from unittest.mock import Mock

from src.training.losses import (
    FocalLoss,
    DetectionLoss,
    RecognitionLoss,
    LetterErrorRateLoss,
    PoseLoss,
    MultiTaskLoss
)


class TestFocalLoss:
    """Test cases for FocalLoss."""
    
    def test_focal_loss_creation(self) -> None:
        """Test focal loss creation."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        assert loss_fn.alpha == 0.25
        assert loss_fn.gamma == 2.0
    
    def test_focal_loss_forward(self) -> None:
        """Test focal loss forward pass."""
        loss_fn = FocalLoss()
        
        # Create mock inputs - FocalLoss expects binary targets same shape as predictions
        predictions = torch.sigmoid(torch.randn(4, 1, 10, requires_grad=True))  # (B, 1, T)
        targets = torch.randint(0, 2, (4, 1, 10)).float()  # Binary targets (B, 1, T)
        
        loss = loss_fn(predictions, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0  # Loss should be non-negative
    
    def test_focal_loss_gradients(self) -> None:
        """Test that focal loss produces gradients."""
        loss_fn = FocalLoss()
        
        # Create raw predictions that will be leaf tensors
        raw_predictions = torch.randn(2, 1, 5, requires_grad=True)  # (B, 1, T)
        predictions = torch.sigmoid(raw_predictions)  # Apply sigmoid
        targets = torch.randint(0, 2, (2, 1, 5)).float()  # Binary targets (B, 1, T)
        
        loss = loss_fn(predictions, targets)
        loss.backward()
        
        # Check gradients on the leaf tensor (raw_predictions)
        assert raw_predictions.grad is not None
        assert not torch.allclose(raw_predictions.grad, torch.zeros_like(raw_predictions.grad))


class TestDetectionLoss:
    """Test cases for DetectionLoss."""
    
    def test_detection_loss_creation(self) -> None:
        """Test detection loss creation."""
        loss_fn = DetectionLoss()
        assert hasattr(loss_fn, 'cls_weight')
        assert hasattr(loss_fn, 'reg_weight')
        assert hasattr(loss_fn, 'focal_loss')
    
    def test_detection_loss_forward(self) -> None:
        """Test detection loss forward pass."""
        loss_fn = DetectionLoss()
        
        # Mock detection predictions
        predictions = {
            'classification': torch.sigmoid(torch.randn(2, 1, 100)),  # (batch, 1, time)
            'regression': torch.randn(2, 2, 100),      # (batch, coords, time)
        }
        
        # Mock targets - match expected format from dataset
        targets = {
            'frame_labels': torch.randint(0, 2, (2, 100)).float(),  # (B, T)
            'segment_targets': torch.randn(2, 3, 3),  # (B, max_segments, 3)
        }
        
        losses = loss_fn(predictions, targets)
        
        assert 'classification_loss' in losses
        assert 'regression_loss' in losses
        assert 'total_detection_loss' in losses
        
        # Check that losses are tensors
        for loss_name, loss_value in losses.items():
            assert isinstance(loss_value, torch.Tensor)
            assert loss_value.item() >= 0


class TestRecognitionLoss:
    """Test cases for RecognitionLoss."""
    
    def test_recognition_loss_creation(self) -> None:
        """Test recognition loss creation."""
        loss_fn = RecognitionLoss()
        assert hasattr(loss_fn, 'ctc_loss')
    
    def test_recognition_loss_forward(self) -> None:
        """Test recognition loss forward pass."""
        loss_fn = RecognitionLoss()
        
        # Mock recognition predictions (log probabilities) - expect (B, T, C)
        predictions = torch.log_softmax(torch.randn(2, 50, 29), dim=-1)  # (batch, time, vocab)
        
        # Mock targets - match expected format from dataset
        targets = {
            'letter_targets': torch.randint(0, 28, (2, 10)),  # (B, max_length) - avoid blank token
            'target_lengths': torch.tensor([8, 6])  # (B,) - actual lengths
        }
        
        loss = loss_fn(predictions, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
    
    def test_recognition_loss_empty_targets(self) -> None:
        """Test recognition loss with minimal target sequences."""
        loss_fn = RecognitionLoss()
        
        predictions = torch.log_softmax(torch.randn(1, 30, 29), dim=-1)  # (B, T, C)
        targets = {
            'letter_targets': torch.randint(0, 28, (1, 2)),  # (B, small_length) - avoid blank
            'target_lengths': torch.tensor([2])  # At least length 2 for CTC
        }
        
        # Should handle minimal sequences gracefully
        loss = loss_fn(predictions, targets)
        assert isinstance(loss, torch.Tensor)


class TestLetterErrorRateLoss:
    """Test cases for LetterErrorRateLoss."""
    
    def test_ler_loss_creation(self) -> None:
        """Test LER loss creation."""
        loss_fn = LetterErrorRateLoss()
        assert hasattr(loss_fn, 'blank_idx')
    
    def test_ler_loss_forward(self) -> None:
        """Test LER loss forward pass."""
        loss_fn = LetterErrorRateLoss()
        
        # Mock predictions and targets
        predictions = torch.randn(2, 30, 29, requires_grad=True)  # (batch, time, vocab)
        targets = {
            'letter_targets': torch.randint(0, 29, (2, 10)),  # (B, max_length)
            'target_lengths': torch.tensor([8, 6])  # (B,) - actual lengths
        }
        segment_probs = torch.sigmoid(torch.randn(2, 30, requires_grad=True))  # (B, T)
        
        loss = loss_fn(predictions, targets, segment_probs)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= -10  # LER loss can be negative (accuracy-based)


class TestPoseLoss:
    """Test cases for PoseLoss."""
    
    def test_pose_loss_creation(self) -> None:
        """Test pose loss creation."""
        loss_fn = PoseLoss()
        assert hasattr(loss_fn, 'confidence_threshold')
    
    def test_pose_loss_forward(self) -> None:
        """Test pose loss forward pass."""
        loss_fn = PoseLoss()
        
        # Mock pose predictions - hand keypoint coordinates (21 keypoints * 2 coords = 42 features)
        predictions = torch.randn(4, 42)  # (batch*time, keypoint_coords)
        
        # Mock targets (pose keypoints coordinates)
        targets = torch.randn(4, 42)  # Same shape as predictions
        
        # Mock confidences - one confidence per keypoint (21 keypoints)
        confidences = torch.rand(4, 21)  # (batch*time, num_keypoints)
        
        loss = loss_fn(predictions, targets, confidences)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
    
    def test_pose_loss_no_targets(self) -> None:
        """Test pose loss when no confident keypoints are available."""
        loss_fn = PoseLoss(confidence_threshold=0.9)  # High threshold
        
        predictions = torch.randn(4, 42)  # (batch*time, keypoint_coords)
        targets = torch.randn(4, 42)     # (batch*time, keypoint_coords)
        confidences = torch.zeros(4, 21)  # No confident keypoints
        
        loss = loss_fn(predictions, targets, confidences)
        
        assert isinstance(loss, torch.Tensor)        
        assert loss.item() == 0.0  # Should be zero when no confident keypoints


class TestMultiTaskLoss:
    """Test cases for MultiTaskLoss."""
    
    def test_combined_loss_creation(self) -> None:
        """Test combined loss creation."""
        loss_fn = MultiTaskLoss()
        assert hasattr(loss_fn, 'detection_loss')
        assert hasattr(loss_fn, 'recognition_loss')
        assert hasattr(loss_fn, 'ler_loss')
        assert hasattr(loss_fn, 'pose_loss')
    
    def test_combined_loss_with_weights(self) -> None:
        """Test combined loss with custom weights."""
        loss_fn = MultiTaskLoss(
            detection_weight=2.0,
            recognition_weight=1.5,
            ler_weight=0.2,
            pose_weight=0.8
        )
        
        assert loss_fn.detection_weight == 2.0
        assert loss_fn.recognition_weight == 1.5
        assert loss_fn.ler_weight == 0.2
        assert loss_fn.pose_weight == 0.8
    
    def test_combined_loss_forward(self) -> None:
        """Test combined loss forward pass."""
        loss_fn = MultiTaskLoss()
        
        # Mock predictions
        predictions = {
            'detection': {
                'classification': torch.sigmoid(torch.randn(2, 1, 50)),
                'regression': torch.randn(2, 2, 50),
                'confidence': torch.sigmoid(torch.randn(2, 1, 50))
            },
            'recognition': torch.log_softmax(torch.randn(2, 50, 29), dim=-1),
        }
        
        # Mock targets - match expected format from dataset
        targets = {
            'detection_targets': {
                'frame_labels': torch.randint(0, 2, (2, 50)).float(),
                'segment_targets': torch.randn(2, 3, 3),
            },
            'recognition_targets': {
                'letter_targets': torch.randint(0, 29, (2, 10)),
                'target_lengths': torch.tensor([8, 6])
            }
        }
        
        losses = loss_fn(predictions, targets)
        
        # Check that all expected losses are returned
        assert 'classification_loss' in losses
        assert 'regression_loss' in losses
        assert 'total_detection_loss' in losses
        assert 'recognition_loss' in losses
        assert 'ler_loss' in losses
        assert 'pose_loss' in losses
        assert 'total_loss' in losses
        
        # Check that all losses are tensors
        for loss_name, loss_value in losses.items():
            assert isinstance(loss_value, torch.Tensor)
    
    def test_combined_loss_gradients(self) -> None:
        """Test that combined loss produces gradients."""
        loss_fn = MultiTaskLoss()
        
        # Create raw predictions (leaf tensors) that will have gradients
        raw_classification = torch.randn(1, 1, 10, requires_grad=True)
        raw_regression = torch.randn(1, 2, 10, requires_grad=True)
        raw_confidence = torch.randn(1, 1, 10, requires_grad=True)
        raw_recognition = torch.randn(1, 10, 29, requires_grad=True)
        
        # Create predictions from raw tensors
        predictions = {
            'detection': {
                'classification': torch.sigmoid(raw_classification),
                'regression': raw_regression,
                'confidence': torch.sigmoid(raw_confidence)
            },
            'recognition': torch.log_softmax(raw_recognition, dim=-1),
        }
        
        targets = {
            'detection_targets': {
                'frame_labels': torch.randint(0, 2, (1, 10)).float(),
                'segment_targets': torch.abs(torch.randn(1, 3, 3)) + 0.1,  # Ensure positive values
            },
            'recognition_targets': {
                'letter_targets': torch.randint(0, 29, (1, 5)),
                'target_lengths': torch.tensor([4])
            }
        }
        
        losses = loss_fn(predictions, targets)
        losses['total_loss'].backward()
        
        # Check that gradients were computed for the raw (leaf) tensors
        assert raw_classification.grad is not None
        assert raw_regression.grad is not None
        assert raw_confidence.grad is not None
        assert raw_recognition.grad is not None


if __name__ == "__main__":
    pytest.main([__file__])
