"""
Simple inference script for fingerspelling detection and recognition.
Works with ChicagoFSWild frame directories.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
import yaml
from typing import Dict, Any
import json
import pandas as pd

from src.models.multitask_model import create_model
from src.utils.types import IDX_TO_CHAR


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_trained_model(checkpoint_path: str, config: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    """Load a trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")
    
    # Create model with same configuration as training
    model = create_model(
        backbone_model=config['model']['backbone'],
        use_pose=config['model']['use_pose'],
        freeze_backbone=config['model']['freeze_backbone']
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully. Best validation loss: {checkpoint.get('best_val_loss', 'N/A')}")
    return model


def load_frames_from_directory(frame_dir: str, max_frames: int = 200) -> np.ndarray:
    """
    Load frames from a directory containing numbered image files.
    
    Args:
        frame_dir: Path to directory containing frame images
        max_frames: Maximum number of frames to load
        
    Returns:
        Array of shape (T, H, W, 3)
    """
    frame_path = Path(frame_dir)
    if not frame_path.exists():
        raise ValueError(f"Frame directory does not exist: {frame_dir}")
    
    # Get all image files and sort them numerically
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(frame_path.glob(ext))
    
    # Sort by filename (assuming numbered frames)
    image_files.sort(key=lambda x: int(x.stem) if x.stem.isdigit() else 0)
    
    if len(image_files) == 0:
        raise ValueError(f"No image files found in {frame_dir}")
    
    # Load frames
    frames = []
    for i, img_file in enumerate(image_files[:max_frames]):
        frame = cv2.imread(str(img_file))
        if frame is not None:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    
    if len(frames) == 0:
        raise ValueError(f"Could not load any frames from {frame_dir}")
    
    return np.array(frames)


def preprocess_frames(frames: np.ndarray, image_size: tuple) -> torch.Tensor:
    """
    Preprocess video frames for inference.
    
    Args:
        frames: Array of shape (T, H, W, 3)
        image_size: Target image size (H, W)
        
    Returns:
        Preprocessed tensor of shape (1, T, 3, H, W)
    """
    processed_frames = []
    target_h, target_w = image_size
    
    for frame in frames:
        # Resize frame
        frame_resized = cv2.resize(frame, (target_w, target_h))
        
        # Convert to tensor and normalize
        frame_tensor = torch.from_numpy(frame_resized).float() / 255.0
        frame_tensor = frame_tensor.permute(2, 0, 1)  # (H, W, 3) -> (3, H, W)
        
        processed_frames.append(frame_tensor)
    
    # Stack into video tensor
    video_tensor = torch.stack(processed_frames, dim=0)  # (T, 3, H, W)
    
    # Add batch dimension
    return video_tensor.unsqueeze(0)  # (1, T, 3, H, W)


def decode_predictions(predictions: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Decode model predictions into human-readable format.
    
    Args:
        predictions: Raw model predictions
        
    Returns:
        Decoded predictions with segments and letter sequences
    """
    results = {
        'segments': [],
        'letter_sequences': [],
        'confidence_scores': []
    }
    
    # Detection predictions
    detection_preds = predictions['detection']
    classification = torch.sigmoid(detection_preds['classification']).squeeze()  # (T,)
    confidence = torch.sigmoid(detection_preds['confidence']).squeeze()  # (T,)
    
    # Find segments where classification > threshold
    threshold = 0.5
    positive_frames = (classification > threshold).cpu().numpy()
    
    # Group consecutive positive frames into segments
    segments = []
    start_frame = None
    
    for i, is_positive in enumerate(positive_frames):
        if is_positive and start_frame is None:
            start_frame = i
        elif not is_positive and start_frame is not None:
            segments.append((start_frame, i - 1, float(confidence[start_frame:i].mean())))
            start_frame = None
    
    # Handle case where video ends with positive frames
    if start_frame is not None:
        segments.append((start_frame, len(positive_frames) - 1, float(confidence[start_frame:].mean())))
    
    results['segments'] = segments
    
    # Recognition predictions
    recognition_preds = predictions['recognition']  # (1, T, num_classes)
    log_probs = recognition_preds.squeeze(0)  # (T, num_classes)
    
    # Simple greedy decoding
    pred_indices = torch.argmax(log_probs, dim=-1)  # (T,)
    
    # Remove blanks and consecutive duplicates
    decoded_chars = []
    prev_idx = None
    blank_idx = 28  # Assuming blank is last token
    
    for idx in pred_indices:
        idx = idx.item()
        if idx != blank_idx and idx != prev_idx:
            if idx < len(IDX_TO_CHAR):
                decoded_chars.append(IDX_TO_CHAR[idx])
        prev_idx = idx
    
    decoded_sequence = ''.join(decoded_chars)
    results['letter_sequences'] = [decoded_sequence]
    
    return results


def get_ground_truth(sequence_name: str, csv_path: str) -> str:
    """Get ground truth label for a sequence."""
    try:
        df = pd.read_csv(csv_path)
        row = df[df['filename'] == sequence_name]
        if len(row) > 0:
            return row.iloc[0]['label_proc']
        else:
            return "Unknown"
    except:
        return "Unknown"


def run_inference(
    model: torch.nn.Module,
    frame_dir: str,
    config: Dict[str, Any],
    device: torch.device
) -> Dict[str, Any]:
    """
    Run inference on a frame directory.
    
    Args:
        model: Trained model
        frame_dir: Path to frame directory
        config: Configuration
        device: Device for inference
        
    Returns:
        Inference results
    """
    print(f"Running inference on {frame_dir}")
    
    # Load frames from directory
    frames = load_frames_from_directory(frame_dir)
    print(f"Loaded {len(frames)} frames")
    
    # Preprocess frames
    image_size = tuple(config['data']['image_size'])
    video_tensor = preprocess_frames(frames, image_size)
    video_tensor = video_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        predictions = model(video_tensor)
    
    # Decode predictions
    results = decode_predictions(predictions)
    
    # Add metadata
    results['frame_directory'] = frame_dir
    results['num_frames'] = len(frames)
    results['frame_rate'] = 30  # Assume 30 FPS
    
    return results


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Run inference on fingerspelling frame directories')
    parser.add_argument('--frames', type=str, required=True,
                       help='Path to frame directory (e.g., dataset/ChicagoFSWild/ChicagoFSWild-Frames/aslized/elsie_stecker_0001)')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output', type=str, default='predictions.json',
                       help='Path to save predictions')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.frames).exists():
        print(f"ERROR: Frame directory not found: {args.frames}")
        return
    
    if not Path(args.checkpoint).exists():
        print(f"ERROR: Checkpoint file not found: {args.checkpoint}")
        return
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load trained model
    model = load_trained_model(args.checkpoint, config, device)
    
    # Get ground truth
    sequence_name = Path(args.frames).name
    parent_dir = Path(args.frames).parent.name
    full_sequence_name = f"{parent_dir}/{sequence_name}"
    csv_path = "dataset/ChicagoFSWild/ChicagoFSWild.csv"
    ground_truth = get_ground_truth(full_sequence_name, csv_path)
    
    # Run inference
    try:
        results = run_inference(model, args.frames, config, device)
        
        # Print results
        print("\n=== INFERENCE RESULTS ===")
        print(f"Sequence: {full_sequence_name}")
        print(f"Ground Truth: '{ground_truth}'")
        print(f"Frames: {results['num_frames']}")
        print(f"Detected segments: {len(results['segments'])}")
        
        for i, (start, end, conf) in enumerate(results['segments']):
            duration = (end - start + 1) / results['frame_rate']
            print(f"  Segment {i+1}: frames {start}-{end} ({duration:.2f}s, conf={conf:.3f})")
        
        predicted_sequence = results['letter_sequences'][0]
        print(f"Predicted sequence: '{predicted_sequence}'")
        
        # Compare with ground truth
        if ground_truth != "Unknown":
            print(f"Match: {'✓' if predicted_sequence.lower() == ground_truth.lower() else '✗'}")
        
        # Add ground truth to results
        results['ground_truth'] = ground_truth
        results['sequence_name'] = full_sequence_name
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
        
    except Exception as e:
        print(f"ERROR during inference: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
