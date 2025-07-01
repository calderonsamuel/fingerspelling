"""
PyTorch dataset classes for fingerspelling detection and recognition.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Any, Union
import cv2

from ..utils.types import CHAR_TO_IDX


class FingerspellingDataset(Dataset):
    """Dataset for fingerspelling detection and recognition."""
    
    def __init__(
        self,
        sequences: List[Dict],
        image_size: Tuple[int, int] = (108, 108),
        max_sequence_length: int = 320,
        augment: bool = False
    ):
        """
        Initialize dataset.
        
        Args:
            sequences: List of processed sequence dictionaries
            image_size: Target image size (height, width)
            max_sequence_length: Maximum sequence length for padding
            augment: Whether to apply data augmentation
        """
        self.sequences = sequences
        self.image_size = image_size
        self.max_sequence_length = max_sequence_length
        self.augment = augment
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
            - 'frames': Video frames tensor (T, C, H, W)
            - 'detection_targets': Detection segment targets
            - 'recognition_targets': Letter sequence targets
            - 'sequence_length': Actual sequence length
        """
        sequence_data = self.sequences[idx]
        frames = sequence_data['frames']  # (T, H, W, 3)
        targets = sequence_data['targets']  # MultiTaskTarget
        encoded_letters = sequence_data['encoded_letters']  # List[int]
        
        # Resize and normalize frames
        processed_frames = self._process_frames(frames)
        
        # Create detection targets
        detection_targets = self._create_detection_targets(targets.detection)
        
        # Create recognition targets
        recognition_targets = self._create_recognition_targets(encoded_letters)
        
        # Convert sequence_info to dictionary for PyTorch compatibility
        sequence_info_dict = {
            'filename': sequence_data['sequence_info'].filename,
            'partition': sequence_data['sequence_info'].partition,
            'signer': sequence_data['sequence_info'].signer,
            'label_proc': sequence_data['sequence_info'].label_proc
        }
        
        return {
            'frames': processed_frames,
            'detection_targets': detection_targets,
            'recognition_targets': recognition_targets,
            'sequence_length': torch.tensor(len(frames), dtype=torch.long),
            'sequence_info': sequence_info_dict
        }
    
    def _process_frames(self, frames: np.ndarray) -> torch.Tensor:
        """
        Process video frames.
        
        Args:
            frames: Input frames (T, H, W, 3)
            
        Returns:
            Processed frames tensor (T, 3, H, W)
        """
        processed = []
        
        for frame in frames:
            # Resize frame
            if frame.shape[:2] != self.image_size:
                frame = cv2.resize(frame, (self.image_size[1], self.image_size[0]))
            
            # Apply augmentation if enabled
            if self.augment:
                frame = self._augment_frame(frame)
            
            # Normalize to [0, 1]
            frame = frame.astype(np.float32) / 255.0
            
            # Convert to CHW format
            frame = frame.transpose(2, 0, 1)  # (H, W, 3) -> (3, H, W)
            
            processed.append(frame)
        
        # Pad or truncate to max_sequence_length
        processed_array = np.array(processed)  # (T, 3, H, W)
        
        if len(processed_array) > self.max_sequence_length:
            processed_array = processed_array[:self.max_sequence_length]
        elif len(processed_array) < self.max_sequence_length:
            # Pad with zeros
            padding_shape = (
                self.max_sequence_length - len(processed_array),
                3, self.image_size[0], self.image_size[1]
            )
            padding = np.zeros(padding_shape, dtype=np.float32)
            processed_array = np.concatenate([processed_array, padding], axis=0)
        
        return torch.from_numpy(processed_array)
    
    def _augment_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to a frame.
        
        Args:
            frame: Input frame (H, W, 3)
            
        Returns:
            Augmented frame
        """
        # Simple augmentation - can be extended
        
        # Random brightness adjustment
        if np.random.random() < 0.3:
            brightness_factor = np.random.uniform(0.8, 1.2)
            frame = np.clip(frame * brightness_factor, 0, 255).astype(np.uint8)
        
        # Random horizontal flip (with low probability for fingerspelling)
        if np.random.random() < 0.1:
            frame = cv2.flip(frame, 1)
        
        return frame
    
    def _create_detection_targets(self, detection_target) -> Dict[str, torch.Tensor]:
        """
        Create detection targets for training.
        
        Args:
            detection_target: DetectionTarget object
            
        Returns:
            Dictionary with detection targets
        """
        # For temporal detection, we create binary labels for each frame
        frame_labels = torch.zeros(self.max_sequence_length, dtype=torch.float32)
        
        for segment in detection_target.segments:
            start_idx = min(segment.start_frame, self.max_sequence_length - 1)
            end_idx = min(segment.end_frame, self.max_sequence_length)
            frame_labels[start_idx:end_idx] = 1.0
        
        # Also create segment-level targets for region-based detection
        segment_targets = []
        for segment in detection_target.segments:
            segment_targets.append([
                segment.start_frame / self.max_sequence_length,  # Normalized start
                segment.end_frame / self.max_sequence_length,    # Normalized end
                1.0  # Confidence/class (fingerspelling = 1)
            ])
        
        # Pad segment targets
        max_segments = 10  # Maximum number of segments per sequence
        while len(segment_targets) < max_segments:
            segment_targets.append([0.0, 0.0, 0.0])  # Padding
        
        segment_targets = segment_targets[:max_segments]  # Truncate if needed
        
        return {
            'frame_labels': frame_labels,
            'segment_targets': torch.tensor(segment_targets, dtype=torch.float32),
            'num_segments': torch.tensor(len(detection_target.segments), dtype=torch.long)
        }
    
    def _create_recognition_targets(self, encoded_letters: List[int]) -> Dict[str, torch.Tensor]:
        """
        Create recognition targets for CTC training.
        
        Args:
            encoded_letters: Encoded letter sequence
            
        Returns:
            Dictionary with recognition targets
        """
        # Pad or truncate letter sequence
        max_letters = 50  # Maximum number of letters
        padded_letters = encoded_letters[:max_letters]
        
        while len(padded_letters) < max_letters:
            padded_letters.append(CHAR_TO_IDX['<blank>'])  # Pad with blank token
        
        return {
            'letter_targets': torch.tensor(padded_letters, dtype=torch.long),
            'target_length': torch.tensor(len(encoded_letters), dtype=torch.long)
        }


def create_data_loaders(
    processed_data: Dict,
    batch_size: int = 4,
    num_workers: int = 0,
    image_size: Tuple[int, int] = (108, 108)
) -> Dict[str, DataLoader]:
    """
    Create data loaders for different partitions.
    
    Args:
        processed_data: Processed data dictionary
        batch_size: Batch size
        num_workers: Number of worker processes
        image_size: Target image size
        
    Returns:
        Dictionary of data loaders
    """
    data_loaders = {}
    
    for partition, sequences in processed_data.items():
        if not sequences:
            continue
        
        dataset = FingerspellingDataset(
            sequences=sequences,
            image_size=image_size,
            augment=(partition == 'train')
        )
        
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(partition == 'train'),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn
        )
        
        data_loaders[partition] = data_loader
    
    return data_loaders


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for variable-length sequences.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Batched data
    """
    # Stack tensors
    frames = torch.stack([item['frames'] for item in batch])
    sequence_lengths = torch.stack([item['sequence_length'] for item in batch])
    
    # Detection targets - stack tensors
    frame_labels = torch.stack([item['detection_targets']['frame_labels'] for item in batch])
    segment_targets = torch.stack([item['detection_targets']['segment_targets'] for item in batch])
    num_segments = torch.stack([item['detection_targets']['num_segments'] for item in batch])
    
    # Recognition targets - stack tensors
    letter_targets = torch.stack([item['recognition_targets']['letter_targets'] for item in batch])
    target_lengths = torch.stack([item['recognition_targets']['target_length'] for item in batch])
    
    # Collect sequence info as list (can't stack strings/dicts)
    sequence_info = [item['sequence_info'] for item in batch]
    
    return {
        'frames': frames,
        'sequence_lengths': sequence_lengths,
        'detection_targets': {
            'frame_labels': frame_labels,
            'segment_targets': segment_targets,
            'num_segments': num_segments
        },
        'recognition_targets': {
            'letter_targets': letter_targets,
            'target_lengths': target_lengths
        },
        'sequence_info': sequence_info
    }
