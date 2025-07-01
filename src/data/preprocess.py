"""
Data preprocessing for ChicagoFSWild dataset.
Converts the dataset format to be compatible with YOLO-based training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import cv2
from tqdm import tqdm
import json

from ..utils.types import (
    SequenceInfo, 
    TemporalSegment, 
    DetectionTarget, 
    RecognitionTarget, 
    MultiTaskTarget,
    CHAR_TO_IDX
)


class ChicagoFSWildProcessor:
    """Processor for ChicagoFSWild dataset."""
    
    def __init__(
        self, 
        dataset_root: Path,
        output_root: Path,
        chunk_size: int = 300,
        chunk_overlap: int = 75
    ):
        """
        Initialize the processor.
        
        Args:
            dataset_root: Path to ChicagoFSWild dataset root
            output_root: Path to output processed data
            chunk_size: Size of video chunks in frames (following paper)
            chunk_overlap: Overlap between chunks in frames
        """
        self.dataset_root = Path(dataset_root)
        self.output_root = Path(output_root)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Load the main CSV file
        self.csv_path = self.dataset_root / "ChicagoFSWild.csv"
        self.frames_root = self.dataset_root / "ChicagoFSWild-Frames"
        
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        if not self.frames_root.exists():
            raise FileNotFoundError(f"Frames directory not found: {self.frames_root}")
    
    def load_sequence_info(self) -> List[SequenceInfo]:
        """Load sequence information from CSV file."""
        df = pd.read_csv(self.csv_path)
        
        sequences = []
        for _, row in df.iterrows():
            sequences.append(SequenceInfo(
                filename=row['filename'],
                url=row['url'],
                start_time=row['start_time'],
                number_of_frames=row['number_of_frames'],
                width=row['width'],
                height=row['height'],
                label_proc=row['label_proc'],
                label_raw=row['label_raw'],
                label_notes=row['label_notes'] if pd.notna(row['label_notes']) else "",
                partition=row['partition'],
                signer=row['signer']
            ))
        
        return sequences
    
    def load_frames(self, sequence: SequenceInfo) -> Optional[np.ndarray]:
        """
        Load frames for a sequence.
        
        Args:
            sequence: Sequence information
            
        Returns:
            Array of shape (num_frames, height, width, 3) or None if loading fails
        """
        sequence_path = self.frames_root / sequence.filename
        
        if not sequence_path.exists():
            print(f"Warning: Sequence path not found: {sequence_path}")
            return None
        
        frame_files = sorted(sequence_path.glob("*.jpg"))
        if len(frame_files) != sequence.number_of_frames:
            print(f"Warning: Expected {sequence.number_of_frames} frames, "
                  f"found {len(frame_files)} for {sequence.filename}")
        
        frames = []
        for frame_file in frame_files:
            frame = cv2.imread(str(frame_file))
            if frame is None:
                print(f"Warning: Could not load frame {frame_file}")
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        return np.array(frames) if frames else None
    
    def create_video_chunks(
        self, 
        sequences: List[SequenceInfo]
    ) -> List[Tuple[List[SequenceInfo], int, int]]:
        """
        Create video chunks following the paper's approach.
        Groups sequences from the same video into chunks.
        
        Args:
            sequences: List of sequences
            
        Returns:
            List of (sequences_in_chunk, start_frame, end_frame) tuples
        """
        # Group sequences by video (based on URL)
        video_groups = {}
        for seq in sequences:
            if seq.url not in video_groups:
                video_groups[seq.url] = []
            video_groups[seq.url].append(seq)
        
        chunks = []
        for url, video_sequences in video_groups.items():
            # Sort sequences by start time
            video_sequences.sort(key=lambda x: x.start_time)
            
            # For simplicity, we'll create chunks based on sequence groups
            # In a full implementation, we'd need to handle the original video timing
            current_frame = 0
            
            while current_frame < sum(seq.number_of_frames for seq in video_sequences):
                chunk_end = min(
                    current_frame + self.chunk_size,
                    sum(seq.number_of_frames for seq in video_sequences)
                )
                
                # Find sequences that overlap with this chunk
                chunk_sequences = []
                seq_start = 0
                
                for seq in video_sequences:
                    seq_end = seq_start + seq.number_of_frames
                    
                    # Check if sequence overlaps with chunk
                    if (seq_start < chunk_end and seq_end > current_frame):
                        chunk_sequences.append(seq)
                    
                    seq_start = seq_end
                
                if chunk_sequences:
                    chunks.append((chunk_sequences, current_frame, chunk_end))
                
                current_frame += self.chunk_size - self.chunk_overlap
        
        return chunks
    
    def process_sequence_to_targets(self, sequence: SequenceInfo) -> MultiTaskTarget:
        """
        Convert a sequence to multi-task targets.
        
        Args:
            sequence: Sequence information
            
        Returns:
            Multi-task targets
        """
        # Detection target - the entire sequence is a fingerspelling segment
        detection_segment = TemporalSegment(
            start_frame=0,
            end_frame=sequence.number_of_frames
        )
        detection_target = DetectionTarget(
            segments=[detection_segment],
            video_length=sequence.number_of_frames
        )
        
        # Recognition target
        recognition_target = RecognitionTarget(
            letter_sequence=sequence.label_proc.lower(),
            segment=detection_segment
        )
        
        return MultiTaskTarget(
            detection=detection_target,
            recognition=[recognition_target],
            pose=None  # Will be added when pose estimation is implemented
        )
    
    def encode_letter_sequence(self, sequence: str) -> List[int]:
        """
        Encode letter sequence to indices for CTC training.
        
        Args:
            sequence: Letter sequence string
            
        Returns:
            List of character indices
        """
        encoded = []
        for char in sequence:
            if char == ' ':
                encoded.append(CHAR_TO_IDX['<sp>'])
            elif char in CHAR_TO_IDX:
                encoded.append(CHAR_TO_IDX[char])
            else:
                encoded.append(CHAR_TO_IDX['<unk>'])
        
        return encoded
    
    def process_dataset(self, subset_size: Optional[int] = None) -> Dict:
        """
        Process the entire dataset.
        
        Args:
            subset_size: If provided, only process this many sequences (for testing)
            
        Returns:
            Dictionary with processed data information
        """
        print("Loading sequence information...")
        sequences = self.load_sequence_info()
        
        if subset_size:
            sequences = sequences[:subset_size]
            print(f"Processing subset of {len(sequences)} sequences")
            
            # For subsets, create our own train/dev split (80/20)
            import random
            random.seed(42)  # For reproducibility
            sequences = sequences.copy()
            random.shuffle(sequences)
            
            split_idx = int(0.8 * len(sequences))
            train_sequences = sequences[:split_idx]
            dev_sequences = sequences[split_idx:]
            test_sequences = []
            
            # Override partition labels for subset
            for seq in train_sequences:
                seq.partition = 'train'
            for seq in dev_sequences:
                seq.partition = 'dev'
                
        else:
            # Use original partition labels for full dataset
            train_sequences = [s for s in sequences if s.partition == 'train']
            dev_sequences = [s for s in sequences if s.partition == 'dev']
            test_sequences = [s for s in sequences if s.partition == 'test']
        
        print(f"Train: {len(train_sequences)}, Dev: {len(dev_sequences)}, Test: {len(test_sequences)}")
        
        # Process each partition
        processed_data = {}
        for partition_name, partition_sequences in [
            ('train', train_sequences),
            ('dev', dev_sequences), 
            ('test', test_sequences)
        ]:
            if not partition_sequences:
                continue
                
            print(f"Processing {partition_name} partition...")
            partition_data = []
            
            for sequence in tqdm(partition_sequences):
                # Load frames
                frames = self.load_frames(sequence)
                if frames is None:
                    continue
                
                # Create targets
                targets = self.process_sequence_to_targets(sequence)
                
                # Encode letter sequence
                encoded_letters = self.encode_letter_sequence(sequence.label_proc.lower())
                
                partition_data.append({
                    'sequence_info': sequence,
                    'frames': frames,
                    'targets': targets,
                    'encoded_letters': encoded_letters
                })
            
            processed_data[partition_name] = partition_data
        
        # Save processed data info
        output_info = {
            'dataset_root': str(self.dataset_root),
            'output_root': str(self.output_root),
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'partitions': {
                name: len(data) for name, data in processed_data.items()
            }
        }
        
        self.output_root.mkdir(parents=True, exist_ok=True)
        with open(self.output_root / 'dataset_info.json', 'w') as f:
            json.dump(output_info, f, indent=2)
        
        return processed_data


def preprocess_data(
    dataset_root: str,
    output_root: str,
    subset_size: Optional[int] = None
) -> Dict:
    """
    Preprocess ChicagoFSWild dataset.
    
    Args:
        dataset_root: Path to dataset root
        output_root: Path to output directory
        subset_size: Size of subset for testing (None for full dataset)
        
    Returns:
        Processed data dictionary
    """
    processor = ChicagoFSWildProcessor(
        dataset_root=Path(dataset_root),
        output_root=Path(output_root)
    )
    
    return processor.process_dataset(subset_size=subset_size)


if __name__ == "__main__":
    # Test preprocessing with a small subset
    dataset_root = "dataset/ChicagoFSWild"
    output_root = "processed_data"
    
    processed_data = preprocess_data(
        dataset_root=dataset_root,
        output_root=output_root,
        subset_size=10  # Small subset for testing
    )
    
    print("Preprocessing completed!")
    for partition, data in processed_data.items():
        print(f"{partition}: {len(data)} sequences")
