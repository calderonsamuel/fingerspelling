"""
Tests for data types and utilities.
"""

import pytest

from src.utils.types import (
    TemporalSegment, 
    DetectionTarget, 
    RecognitionTarget,
    MultiTaskTarget,
    CHAR_TO_IDX,
    IDX_TO_CHAR
)


class TestTemporalSegment:
    """Test cases for TemporalSegment."""
    
    def test_segment_creation(self):
        """Test basic segment creation."""
        segment = TemporalSegment(start_frame=10, end_frame=20)
        assert segment.start_frame == 10
        assert segment.end_frame == 20
        assert segment.length == 10
    
    def test_segment_validation(self):
        """Test segment validation."""
        with pytest.raises(ValueError):
            TemporalSegment(start_frame=20, end_frame=10)
    
    def test_iou_calculation(self):
        """Test IoU calculation between segments."""
        seg1 = TemporalSegment(start_frame=10, end_frame=20)
        seg2 = TemporalSegment(start_frame=15, end_frame=25)
        
        # Expected IoU: intersection=5, union=15, IoU=5/15=0.333...
        iou = seg1.iou(seg2)
        assert abs(iou - 5/15) < 1e-6
    
    def test_iou_no_overlap(self):
        """Test IoU with no overlap."""
        seg1 = TemporalSegment(start_frame=10, end_frame=20)
        seg2 = TemporalSegment(start_frame=25, end_frame=35)
        
        iou = seg1.iou(seg2)
        assert iou == 0.0
    
    def test_iou_complete_overlap(self):
        """Test IoU with complete overlap."""
        seg1 = TemporalSegment(start_frame=10, end_frame=20)
        seg2 = TemporalSegment(start_frame=10, end_frame=20)
        
        iou = seg1.iou(seg2)
        assert iou == 1.0


class TestMultiTaskTarget:
    """Test cases for MultiTaskTarget."""
    
    def test_target_creation(self):
        """Test creating multi-task targets."""
        # Detection target
        segment = TemporalSegment(start_frame=0, end_frame=10)
        detection_target = DetectionTarget(segments=[segment], video_length=10)
        
        # Recognition target
        recognition_target = RecognitionTarget(
            letter_sequence="hello",
            segment=segment
        )
        
        # Multi-task target
        multi_target = MultiTaskTarget(
            detection=detection_target,
            recognition=[recognition_target]
        )
        
        assert len(multi_target.detection.segments) == 1
        assert len(multi_target.recognition) == 1
        assert multi_target.recognition[0].letter_sequence == "hello"


class TestCharacterMapping:
    """Test character to index mapping."""
    
    def test_alphabet_mapping(self):
        """Test basic alphabet mapping."""
        assert CHAR_TO_IDX['a'] == 0
        assert CHAR_TO_IDX['z'] == 25
        assert IDX_TO_CHAR[0] == 'a'
        assert IDX_TO_CHAR[25] == 'z'
    
    def test_special_tokens(self):
        """Test special token mapping."""
        assert '<blank>' in CHAR_TO_IDX
        assert '<sp>' in CHAR_TO_IDX
        assert '<unk>' in CHAR_TO_IDX
    
    def test_bidirectional_mapping(self):
        """Test that character mapping is bidirectional."""
        for char, idx in CHAR_TO_IDX.items():
            assert IDX_TO_CHAR[idx] == char


if __name__ == "__main__":
    pytest.main([__file__])
