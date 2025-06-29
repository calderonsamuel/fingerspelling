#!/usr/bin/env python3
"""
ChicagoFSWild to YOLO Format Converter

This script converts the ChicagoFSWild dataset to YOLO format for computer vision tasks.
It processes the CSV files and bounding box annotations to create YOLO-compatible
training data for hand detection and fingerspelling recognition.

Author: Generated for ChicagoFSWild Dataset Processing
Date: June 2025
"""

import os
import pandas as pd
import shutil
import yaml
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
import json
from collections import defaultdict
import time
import sys


class ChicagoFSWildYOLOConverter:
    """Converts ChicagoFSWild dataset to YOLO format."""
    
    def __init__(self, dataset_path: str, output_path: str):
        """
        Initialize the converter.
        
        Args:
            dataset_path: Path to ChicagoFSWild dataset directory
            output_path: Path where YOLO formatted dataset will be created
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        
        # Dataset file paths
        self.main_csv = self.dataset_path / "ChicagoFSWild.csv"
        self.hand_annotation_csv = self.dataset_path / "HandAnnotation.csv"
        self.bbox_dir = self.dataset_path / "BBox"
        self.frames_dir = self.dataset_path / "ChicagoFSWild-Frames"
        
        # YOLO class definitions
        self.classes = {
            1: "signing_hand",  # L=1 in original annotations
            2: "non_signing_hand"  # L=2 in original annotations
        }
        
        # Statistics
        self.stats = {
            "total_sequences": 0,
            "sequences_with_bbox": 0,
            "total_frames": 0,
            "frames_with_annotations": 0,
            "train_sequences": 0,
            "dev_sequences": 0,
            "test_sequences": 0
        }
        
        # Progress tracking
        self.start_time = None
        self.processed_sequences = 0
    
    def print_progress_bar(self, current: int, total: int, prefix: str = "Progress", 
                          suffix: str = "Complete", bar_length: int = 50):
        """Print a progress bar to the console."""
        if total == 0:
            return
            
        percent = (current / total) * 100
        filled_length = int(bar_length * current // total)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        # Calculate elapsed time and ETA
        if self.start_time:
            elapsed = time.time() - self.start_time
            if current > 0:
                eta = (elapsed / current) * (total - current)
                eta_str = f"ETA: {self.format_time(eta)}"
            else:
                eta_str = "ETA: --:--"
            time_str = f"Elapsed: {self.format_time(elapsed)}, {eta_str}"
        else:
            time_str = ""
        
        # Print progress bar
        sys.stdout.write(f'\r{prefix} |{bar}| {current}/{total} ({percent:.1f}%) {suffix} {time_str}')
        sys.stdout.flush()
        
        if current == total:
            print()  # New line when complete
    
    def format_time(self, seconds: float) -> str:
        """Format seconds into readable time string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    def setup_output_structure(self):
        """Create YOLO dataset directory structure."""
        print("Setting up output directory structure...")
        
        # Create main directories
        for split in ['train', 'val', 'test']:
            (self.output_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Create additional directories for metadata
        (self.output_path / 'metadata').mkdir(exist_ok=True)
        (self.output_path / 'scripts').mkdir(exist_ok=True)
    
    def load_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load the main dataset and hand annotation CSV files."""
        print("Loading dataset CSV files...")
        
        if not self.main_csv.exists():
            raise FileNotFoundError(f"Main CSV file not found: {self.main_csv}")
        
        if not self.hand_annotation_csv.exists():
            raise FileNotFoundError(f"Hand annotation CSV not found: {self.hand_annotation_csv}")
        
        main_df = pd.read_csv(self.main_csv)
        hand_df = pd.read_csv(self.hand_annotation_csv)
        
        print(f"Loaded {len(main_df)} sequences from main dataset")
        print(f"Loaded {len(hand_df)} sequences with hand annotations")
        
        return main_df, hand_df
    
    def convert_bbox_to_yolo(self, bbox: List[int], img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """
        Convert bounding box from (x0, y0, x1, y1) to YOLO format.
        
        Args:
            bbox: [x0, y0, x1, y1] format
            img_width: Image width
            img_height: Image height
            
        Returns:
            Tuple of (x_center, y_center, width, height) normalized to [0,1]
        """
        x0, y0, x1, y1 = bbox
        
        # Convert to center coordinates and normalize
        x_center = (x0 + x1) / 2.0 / img_width
        y_center = (y0 + y1) / 2.0 / img_height
        width = (x1 - x0) / img_width
        height = (y1 - y0) / img_height
        
        # Ensure values are within [0, 1]
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        return x_center, y_center, width, height
    
    def process_sequence_annotations(self, sequence_name: str, img_width: int, img_height: int) -> Dict[str, List[str]]:
        """
        Process bounding box annotations for a sequence.
        
        Args:
            sequence_name: Name of the sequence (e.g., 'misc_1/ryan_commerson_0369')
            img_width: Width of images in the sequence
            img_height: Height of images in the sequence
            
        Returns:
            Dictionary mapping frame numbers to YOLO annotation lines
        """
        sequence_bbox_dir = self.bbox_dir / sequence_name
        frame_annotations = {}
        
        if not sequence_bbox_dir.exists():
            return frame_annotations
        
        # Process each frame's bounding box file
        for bbox_file in sequence_bbox_dir.glob("*.txt"):
            frame_num = bbox_file.stem
            
            try:
                with open(bbox_file, 'r') as f:
                    lines = f.readlines()
                
                yolo_annotations = []
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split(',')
                    if len(parts) != 5:
                        continue
                    
                    x0, y0, x1, y1, label = map(int, parts)
                    
                    # Convert to YOLO format
                    x_center, y_center, width, height = self.convert_bbox_to_yolo(
                        [x0, y0, x1, y1], img_width, img_height
                    )
                    
                    # YOLO class index (0-based)
                    class_id = label - 1  # Convert from 1,2 to 0,1
                    
                    # YOLO annotation line
                    yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                    yolo_annotations.append(yolo_line)
                
                if yolo_annotations:
                    frame_annotations[frame_num] = yolo_annotations
                    
            except Exception as e:
                print(f"Error processing {bbox_file}: {e}")
        
        return frame_annotations
    
    def copy_and_annotate_sequence(self, sequence_info: dict, split: str) -> int:
        """
        Copy sequence frames and create YOLO annotations.
        
        Args:
            sequence_info: Dictionary with sequence information
            split: Dataset split ('train', 'val', 'test')
            
        Returns:
            Number of frames processed
        """
        sequence_name = sequence_info['filename']
        img_width = sequence_info['width']
        img_height = sequence_info['height']
        
        # Source directory for frames
        source_frames_dir = self.frames_dir / sequence_name
        
        if not source_frames_dir.exists():
            print(f"\nWarning: Frames directory not found for {sequence_name}")
            return 0
        
        # Get bounding box annotations for this sequence
        frame_annotations = self.process_sequence_annotations(sequence_name, img_width, img_height)
        
        # Get list of frame files
        frame_files = list(source_frames_dir.glob("*.jpg"))
        
        # Process each frame
        frames_processed = 0
        for frame_file in frame_files:
            frame_num = frame_file.stem
            
            # Create unique filename to avoid conflicts
            safe_sequence_name = sequence_name.replace('/', '_')
            new_frame_name = f"{safe_sequence_name}_{frame_num}.jpg"
            
            # Copy image
            dest_image_path = self.output_path / split / 'images' / new_frame_name
            shutil.copy2(frame_file, dest_image_path)
            
            # Create annotation file
            annotation_file = self.output_path / split / 'labels' / f"{safe_sequence_name}_{frame_num}.txt"
            
            if frame_num in frame_annotations:
                # Write YOLO annotations
                with open(annotation_file, 'w') as f:
                    for annotation_line in frame_annotations[frame_num]:
                        f.write(annotation_line + '\n')
                self.stats["frames_with_annotations"] += 1
            else:
                # Create empty annotation file for frames without hand annotations
                with open(annotation_file, 'w') as f:
                    pass
            
            frames_processed += 1
        
        return frames_processed
    
    def create_dataset_yaml(self):
        """Create YOLO dataset configuration file."""
        dataset_config = {
            'path': './',  # Use relative path for portability
            'train': 'train/images',
            'val': 'val/images', 
            'test': 'test/images',
            'nc': len(self.classes),
            'names': [self.classes[i+1] for i in range(len(self.classes))]
        }
        
        yaml_path = self.output_path / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"Created dataset configuration: {yaml_path}")
    
    def create_metadata_files(self, main_df: pd.DataFrame, hand_df: pd.DataFrame):
        """Create metadata files with dataset information."""
        metadata_dir = self.output_path / 'metadata'
        
        # Save original CSV files for reference
        main_df.to_csv(metadata_dir / 'original_dataset.csv', index=False)
        hand_df.to_csv(metadata_dir / 'hand_annotations.csv', index=False)
        
        # Create dataset statistics
        stats_file = metadata_dir / 'dataset_statistics.json'
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        # Create class mapping
        class_mapping = {
            'classes': self.classes,
            'class_names': [self.classes[i+1] for i in range(len(self.classes))],
            'num_classes': len(self.classes)
        }
        
        class_file = metadata_dir / 'class_mapping.json'
        with open(class_file, 'w') as f:
            json.dump(class_mapping, f, indent=2)
        
        print(f"Created metadata files in {metadata_dir}")
    
    def convert(self):
        """Main conversion process."""
        print("Starting ChicagoFSWild to YOLO conversion...")
        self.start_time = time.time()
        
        # Setup output structure
        print("Setting up output directory structure...")
        self.setup_output_structure()
        
        # Load datasets
        print("Loading dataset CSV files...")
        main_df, hand_df = self.load_datasets()
        
        # Update statistics
        self.stats["total_sequences"] = len(main_df)
        self.stats["sequences_with_bbox"] = len(hand_df)
        
        # Create set of sequences with bounding box annotations
        sequences_with_bbox = set(hand_df['filename'].values)
        
        # Filter to only sequences with bounding boxes for processing
        sequences_to_process = main_df[main_df['filename'].isin(sequences_with_bbox)]
        total_sequences_to_process = len(sequences_to_process)
        
        print(f"\nProcessing {total_sequences_to_process} sequences with bounding box annotations...")
        print(f"Total sequences in dataset: {len(main_df)}")
        print(f"Sequences with annotations: {total_sequences_to_process}")
        print()
        
        # Process sequences by partition
        partition_mapping = {'train': 'train', 'dev': 'val', 'test': 'test'}
        
        for idx, (_, row) in enumerate(sequences_to_process.iterrows()):
            sequence_name = row['filename']
            partition = row['partition']
            
            # Map partition to YOLO split
            split = partition_mapping.get(partition, 'train')
            
            # Process sequence
            frames_processed = self.copy_and_annotate_sequence(row.to_dict(), split)
            
            if frames_processed > 0:
                self.stats["total_frames"] += frames_processed
                self.stats[f"{partition}_sequences"] += 1
            
            # Update progress
            self.processed_sequences = idx + 1
            self.print_progress_bar(
                current=self.processed_sequences,
                total=total_sequences_to_process,
                prefix="Converting",
                suffix=f"| {sequence_name} ({frames_processed} frames) -> {split}"
            )
        
        print("\nCreating configuration files...")
        self.create_dataset_yaml()
        self.create_metadata_files(main_df, hand_df)
        
        # Print final statistics
        self.print_statistics()
    
    def print_statistics(self):
        """Print conversion statistics."""
        total_time = time.time() - self.start_time if self.start_time else 0
        
        print("\n" + "="*60)
        print("CONVERSION STATISTICS")
        print("="*60)
        print(f"Total processing time: {self.format_time(total_time)}")
        print(f"Total sequences in dataset: {self.stats['total_sequences']}")
        print(f"Sequences with bounding boxes: {self.stats['sequences_with_bbox']}")
        print(f"Total frames processed: {self.stats['total_frames']}")
        print(f"Frames with annotations: {self.stats['frames_with_annotations']}")
        
        # Processing rates
        if total_time > 0:
            seq_rate = self.processed_sequences / total_time
            frame_rate = self.stats['total_frames'] / total_time
            print(f"Processing rate: {seq_rate:.2f} sequences/sec, {frame_rate:.2f} frames/sec")
        
        print("\nSplit distribution:")
        print(f"  Train sequences: {self.stats['train_sequences']}")
        print(f"  Dev/Val sequences: {self.stats['dev_sequences']}")
        print(f"  Test sequences: {self.stats['test_sequences']}")
        print(f"\nOutput directory: {self.output_path}")
        print("="*60)


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Convert ChicagoFSWild dataset to YOLO format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_to_yolo.py --dataset ../dataset/ChicagoFSWild --output ./yolo_dataset
  python convert_to_yolo.py --dataset /path/to/ChicagoFSWild --output /path/to/output --all-sequences
        """
    )
    
    parser.add_argument(
        '--dataset', 
        type=str, 
        required=True,
        help='Path to ChicagoFSWild dataset directory'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        required=True,
        help='Output directory for YOLO formatted dataset'
    )
    
    args = parser.parse_args()
    
    # Validate input paths
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset directory does not exist: {dataset_path}")
        return 1
    
    if not (dataset_path / "ChicagoFSWild.csv").exists():
        print(f"Error: ChicagoFSWild.csv not found in {dataset_path}")
        return 1
    
    # Create converter and run conversion
    converter = ChicagoFSWildYOLOConverter(args.dataset, args.output)
    
    try:
        converter.convert()
        print("\nConversion completed successfully!")
        return 0
    except Exception as e:
        print(f"\nError during conversion: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
