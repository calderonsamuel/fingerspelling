"""
Tests for dataset and data loading functionality.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path
from typing import Dict, Any

from src.data.dataset import (
    FingerspellingDataset,
    create_data_loaders
)
from src.utils.types import (
    SequenceInfo, 
    TemporalSegment, 
    DetectionTarget, 
    RecognitionTarget, 
    MultiTaskTarget
)


class TestFingerspellingDataset:
    """Test cases for FingerspellingDataset."""
    
    def create_mock_sequence(self, filename: str = "test_001") -> Dict[str, Any]:
        """Create a mock sequence for testing."""
        # Create mock frame data (numpy arrays like actual preprocessing would provide)
        mock_frames = np.random.randint(0, 255, (10, 108, 108, 3), dtype=np.uint8)  # (T, H, W, C)
        
        # Create mock sequence info
        mock_sequence_info = SequenceInfo(
            filename=f"{filename}.mp4",
            url="http://example.com/test.mp4",
            start_time="00:00:10",
            number_of_frames=10,
            width=108,
            height=108,
            label_proc="hello world",
            label_raw="hello world",
            label_notes="test sequence",
            partition="train",
            signer="test_signer"
        )
        
        # Create proper multi-task targets
        detection_target = DetectionTarget(
            segments=[TemporalSegment(start_frame=2, end_frame=8)],
            video_length=10
        )
        
        recognition_target = RecognitionTarget(
            letter_sequence="hello",
            segment=TemporalSegment(start_frame=2, end_frame=8)
        )
        
        multi_task_target = MultiTaskTarget(
            detection=detection_target,
            recognition=[recognition_target]
        )
        
        return {
            'sequence_id': filename,
            'frames': mock_frames,  # Actual frame data, not file paths
            'targets': multi_task_target,  # Proper MultiTaskTarget object
            'encoded_letters': [0, 1, 2, 3, 4],  # Add this field that dataset expects
            'sequence_info': mock_sequence_info,  # Add sequence info
            'sequence_length': 10
        }
    
    @patch('cv2.imread')
    def test_dataset_creation(self, mock_imread: Mock) -> None:
        """Test dataset creation."""
        # Mock image loading
        mock_imread.return_value = np.ones((108, 108, 3), dtype=np.uint8)
        
        sequences = [self.create_mock_sequence()]
        dataset = FingerspellingDataset(
            sequences=sequences,
            image_size=(108, 108),
            max_sequence_length=320
        )
        
        assert len(dataset) == 1
        assert dataset.image_size == (108, 108)
        assert dataset.max_sequence_length == 320
    
    @patch('cv2.imread')
    def test_dataset_getitem(self, mock_imread: Mock) -> None:
        """Test dataset item retrieval."""
        # Mock image loading
        mock_image = np.random.randint(0, 255, (108, 108, 3), dtype=np.uint8)
        mock_imread.return_value = mock_image
        
        sequences = [self.create_mock_sequence()]
        dataset = FingerspellingDataset(
            sequences=sequences,
            image_size=(108, 108),
            max_sequence_length=50
        )
        
        item = dataset[0]
        
        # Check return structure - match actual dataset output
        assert 'frames' in item
        assert 'detection_targets' in item
        assert 'recognition_targets' in item
        assert 'sequence_length' in item
        assert 'sequence_info' in item
        
        # Check frame tensor
        frames = item['frames']
        assert isinstance(frames, torch.Tensor)
        assert len(frames.shape) == 4  # (T, C, H, W)
        assert frames.shape[2:] == (108, 108)  # (H, W)
        
        # Check detection targets structure
        detection_targets = item['detection_targets']
        assert 'frame_labels' in detection_targets
        assert 'segment_targets' in detection_targets
        assert 'num_segments' in detection_targets
        
        # Check recognition targets structure
        recognition_targets = item['recognition_targets']
        assert 'letter_targets' in recognition_targets
        assert 'target_length' in recognition_targets
    
    def test_dataset_empty_sequences(self) -> None:
        """Test dataset with empty sequences."""
        dataset = FingerspellingDataset(
            sequences=[],
            image_size=(108, 108)
        )
        
        assert len(dataset) == 0
    
    def test_dataset_index_error(self) -> None:
        """Test dataset index out of bounds."""
        sequences = [self.create_mock_sequence()]
        dataset = FingerspellingDataset(
            sequences=sequences,
            image_size=(108, 108)
        )
        
        with pytest.raises(IndexError):
            _ = dataset[10]  # Index beyond dataset size


class TestDataLoaderCreation:
    """Test cases for data loader creation."""
    
    def create_mock_processed_data(self) -> Dict[str, Any]:
        """Create mock processed data."""
        return {
            'train': [
                {
                    'sequence_id': 'train_001',
                    'frames': np.random.randint(0, 255, (5, 108, 108, 3), dtype=np.uint8),  # Numpy arrays
                    'targets': {
                        'detection': {
                            'classification': torch.zeros(5, 1),
                            'regression': torch.zeros(5, 2),
                            'confidence': torch.zeros(5, 1)
                        },
                        'recognition': [0, 1, 2],
                        'pose': torch.zeros(5, 58)
                    },
                    'encoded_letters': [0, 1, 2],  # Add this field
                    'sequence_length': 5
                }
            ],
            'dev': [
                {
                    'sequence_id': 'dev_001',
                    'frames': np.random.randint(0, 255, (3, 108, 108, 3), dtype=np.uint8),  # Numpy arrays
                    'targets': {
                        'detection': {
                            'classification': torch.zeros(3, 1),
                            'regression': torch.zeros(3, 2),
                            'confidence': torch.zeros(3, 1)
                        },
                        'recognition': [1, 2],
                        'pose': torch.zeros(3, 58)
                    },
                    'encoded_letters': [1, 2],  # Add this field
                    'sequence_length': 3
                }
            ]
        }
    
    @patch('src.data.dataset.FingerspellingDataset')
    def test_create_data_loaders(self, mock_dataset_class: Mock) -> None:
        """Test data loader creation."""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=10)
        mock_dataset_class.return_value = mock_dataset
        
        processed_data = self.create_mock_processed_data()
        
        data_loaders = create_data_loaders(
            processed_data=processed_data,
            batch_size=4,
            num_workers=0,
            image_size=(108, 108)
        )
        
        # Check that loaders are created for each partition
        assert 'train' in data_loaders
        assert 'dev' in data_loaders
        
        # Verify dataset creation was called
        assert mock_dataset_class.call_count == 2
    
    @patch('src.data.dataset.FingerspellingDataset')
    def test_create_data_loaders_train_only(self, mock_dataset_class: Mock) -> None:
        """Test data loader creation with train data only."""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=5)
        mock_dataset_class.return_value = mock_dataset
        
        processed_data = {'train': self.create_mock_processed_data()['train']}
        
        data_loaders = create_data_loaders(
            processed_data=processed_data,
            batch_size=2
        )
        
        assert 'train' in data_loaders
        assert 'dev' not in data_loaders
        assert mock_dataset_class.call_count == 1
    
    def test_create_data_loaders_empty_data(self) -> None:
        """Test data loader creation with empty data."""
        data_loaders = create_data_loaders(
            processed_data={},
            batch_size=4
        )
        
        assert len(data_loaders) == 0
    
    def test_create_data_loaders_parameters(self) -> None:
        """Test data loader creation with different parameters."""
        processed_data = self.create_mock_processed_data()
        
        # Test different batch sizes
        with patch('src.data.dataset.FingerspellingDataset') as mock_dataset:
            mock_dataset.return_value.__len__ = Mock(return_value=1)
            
            loaders = create_data_loaders(
                processed_data=processed_data,
                batch_size=8,
                num_workers=2,
                image_size=(224, 224)
            )
            
            # Verify parameters were passed correctly
            mock_dataset.assert_called()
            call_args = mock_dataset.call_args_list[0]
            assert call_args[1]['image_size'] == (224, 224)


class TestSequenceInfoValidation:
    """Test cases for SequenceInfo validation."""
    
    def test_sequence_info_creation(self) -> None:
        """Test SequenceInfo creation with all required fields."""
        seq_info = SequenceInfo(
            filename="test.mp4",
            url="http://example.com/test.mp4",
            start_time="00:00:10",
            number_of_frames=300,
            width=640,
            height=480,
            label_proc="hello world",
            label_raw="hello world",
            label_notes="clear pronunciation",
            partition="train",
            signer="signer_001"
        )
        
        assert seq_info.filename == "test.mp4"
        assert seq_info.partition == "train"
        assert seq_info.number_of_frames == 300
    
    def test_sequence_info_missing_fields(self) -> None:
        """Test SequenceInfo with missing required fields."""
        with pytest.raises(TypeError):
            # Missing required fields should raise TypeError
            SequenceInfo(filename="test.mp4")


if __name__ == "__main__":
    pytest.main([__file__])
