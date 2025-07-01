"""
Live video inference for fingerspelling detection and recognition.
Supports webcam input and real-time video processing.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
import yaml
import time
from collections import deque
from typing import Dict, Any, Optional, Tuple
import threading
import queue

from src.models.multitask_model import create_model
from src.utils.types import IDX_TO_CHAR


class LiveInferenceEngine:
    """Real-time fingerspelling inference engine."""
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        device: torch.device,
        window_size: int = 30,
        overlap: int = 15
    ):
        """
        Initialize live inference engine.
        
        Args:
            checkpoint_path: Path to trained model
            config_path: Path to config file
            device: Device for inference
            window_size: Number of frames to process at once
            overlap: Frame overlap between windows
        """
        self.device = device
        self.window_size = window_size
        self.overlap = overlap
        
        # Load config and model
        self.config = self.load_config(config_path)
        self.model = self.load_model(checkpoint_path)
        
        # Frame buffer for temporal processing
        self.frame_buffer = deque(maxlen=window_size)
        self.image_size = tuple(self.config['data']['image_size'])
        
        # Results storage
        self.recent_predictions = deque(maxlen=10)
        self.current_segment = None
        self.segment_start_time = None
        
        # Display settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.font_thickness = 2
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_model(self, checkpoint_path: str) -> torch.nn.Module:
        """Load trained model."""
        print(f"Loading model from {checkpoint_path}")
        
        model = create_model(
            backbone_model=self.config['model']['backbone'],
            use_pose=self.config['model']['use_pose'],
            freeze_backbone=self.config['model']['freeze_backbone']
        )
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess single frame for model input."""
        # Resize to model input size
        target_h, target_w = self.image_size
        frame_resized = cv2.resize(frame, (target_w, target_h))
        
        # Convert to tensor and normalize
        frame_tensor = torch.from_numpy(frame_resized).float() / 255.0
        frame_tensor = frame_tensor.permute(2, 0, 1)  # (H, W, 3) -> (3, H, W)
        
        return frame_tensor
    
    def process_frame_window(self) -> Optional[Dict[str, Any]]:
        """Process current frame window and return predictions."""
        if len(self.frame_buffer) < self.window_size:
            return None
        
        # Stack frames into video tensor
        frames = list(self.frame_buffer)
        video_tensor = torch.stack(frames, dim=0).unsqueeze(0)  # (1, T, 3, H, W)
        video_tensor = video_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model(video_tensor)
        
        # Decode predictions
        return self.decode_predictions(predictions)
    
    def decode_predictions(self, predictions: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Decode model predictions."""
        # Detection predictions
        detection_preds = predictions['detection']
        classification = torch.sigmoid(detection_preds['classification']).squeeze()  # (T,)
        confidence = torch.sigmoid(detection_preds['confidence']).squeeze()  # (T,)
        
        # Find if fingerspelling is detected (use recent frames for stability)
        recent_frames = classification[-5:]  # Last 5 frames
        is_fingerspelling = (recent_frames > 0.5).float().mean() > 0.6  # 60% of recent frames
        avg_confidence = float(confidence[-5:].mean())
        
        # Recognition predictions
        recognition_preds = predictions['recognition']  # (1, T, num_classes)
        log_probs = recognition_preds.squeeze(0)  # (T, num_classes)
        
        # Simple greedy decoding for recent frames
        recent_log_probs = log_probs[-5:]  # Last 5 frames
        pred_indices = torch.argmax(recent_log_probs, dim=-1)
        
        # Decode characters
        decoded_chars = []
        prev_idx = None
        blank_idx = 28
        
        for idx in pred_indices:
            idx = idx.item()
            if idx != blank_idx and idx != prev_idx:
                if idx in IDX_TO_CHAR:
                    decoded_chars.append(IDX_TO_CHAR[idx])
            prev_idx = idx
        
        predicted_letters = ''.join(decoded_chars) if decoded_chars else ""
        
        return {
            'is_fingerspelling': is_fingerspelling,
            'confidence': avg_confidence,
            'predicted_letters': predicted_letters,
            'raw_classification': float(classification[-1]),  # Most recent frame
        }
    
    def update_segment_tracking(self, prediction: Dict[str, Any], timestamp: float):
        """Update segment tracking based on predictions."""
        is_fingerspelling = prediction['is_fingerspelling']
        
        if is_fingerspelling and self.current_segment is None:
            # Start new segment
            self.current_segment = {
                'start_time': timestamp,
                'letters': [],
                'max_confidence': prediction['confidence']
            }
            self.segment_start_time = timestamp
            
        elif is_fingerspelling and self.current_segment is not None:
            # Continue segment
            if prediction['predicted_letters']:
                self.current_segment['letters'].append(prediction['predicted_letters'])
            self.current_segment['max_confidence'] = max(
                self.current_segment['max_confidence'], 
                prediction['confidence']
            )
            
        elif not is_fingerspelling and self.current_segment is not None:
            # End segment
            duration = timestamp - self.current_segment['start_time']
            final_sequence = ''.join(set(self.current_segment['letters']))  # Remove duplicates
            
            completed_segment = {
                'start_time': self.current_segment['start_time'],
                'end_time': timestamp,
                'duration': duration,
                'sequence': final_sequence,
                'confidence': self.current_segment['max_confidence']
            }
            
            self.recent_predictions.append(completed_segment)
            self.current_segment = None
    
    def draw_overlay(self, frame: np.ndarray, prediction: Dict[str, Any], timestamp: float) -> np.ndarray:
        """Draw inference results on frame."""
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Status indicator
        status_color = (0, 255, 0) if prediction['is_fingerspelling'] else (0, 0, 255)
        status_text = "FINGERSPELLING" if prediction['is_fingerspelling'] else "NO DETECTION"
        
        cv2.rectangle(overlay, (10, 10), (300, 60), status_color, -1)
        cv2.putText(overlay, status_text, (20, 40), self.font, 0.8, (255, 255, 255), 2)
        
        # Confidence
        conf_text = f"Confidence: {prediction['confidence']:.2f}"
        cv2.putText(overlay, conf_text, (10, 80), self.font, 0.6, (255, 255, 255), 2)
        
        # Current prediction
        if prediction['predicted_letters']:
            letter_text = f"Letters: {prediction['predicted_letters']}"
            cv2.putText(overlay, letter_text, (10, 110), self.font, 0.6, (0, 255, 255), 2)
        
        # Current segment info
        if self.current_segment:
            duration = timestamp - self.current_segment['start_time']
            segment_text = f"Segment: {duration:.1f}s"
            cv2.putText(overlay, segment_text, (10, 140), self.font, 0.6, (255, 255, 0), 2)
        
        # Recent predictions
        y_offset = h - 150
        cv2.putText(overlay, "Recent Predictions:", (10, y_offset), self.font, 0.6, (255, 255, 255), 2)
        
        for i, segment in enumerate(list(self.recent_predictions)[-3:]):  # Last 3 predictions
            y_pos = y_offset + 30 + (i * 25)
            pred_text = f"{segment['sequence']} ({segment['duration']:.1f}s, {segment['confidence']:.2f})"
            cv2.putText(overlay, pred_text, (10, y_pos), self.font, 0.5, (0, 255, 255), 1)
        
        # Blend overlay
        alpha = 0.7
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    def run_webcam(self, camera_id: int = 0):
        """Run live inference on webcam."""
        print(f"Starting webcam inference (camera {camera_id})")
        print("Press 'q' to quit, 's' to save current predictions")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Could not open camera {camera_id}")
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = time.time()
                
                # Preprocess and add to buffer
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frame = self.preprocess_frame(frame_rgb)
                self.frame_buffer.append(processed_frame)
                
                # Process when buffer is full
                prediction = self.process_frame_window()
                if prediction:
                    self.update_segment_tracking(prediction, timestamp)
                    frame = self.draw_overlay(frame, prediction, timestamp)
                
                # Display
                cv2.imshow('Fingerspelling Detection', frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_predictions()
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def run_video_file(self, video_path: str):
        """Run inference on video file."""
        print(f"Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = frame_count / fps
                
                # Process frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frame = self.preprocess_frame(frame_rgb)
                self.frame_buffer.append(processed_frame)
                
                prediction = self.process_frame_window()
                if prediction:
                    self.update_segment_tracking(prediction, timestamp)
                    frame = self.draw_overlay(frame, prediction, timestamp)
                
                # Display
                cv2.imshow('Fingerspelling Detection', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                frame_count += 1
                
                # Progress
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}%")
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
        self.save_predictions()
    
    def save_predictions(self):
        """Save predictions to file."""
        import json
        
        results = {
            'timestamp': time.strftime('%Y-%m-%d_%H-%M-%S'),
            'segments': list(self.recent_predictions),
            'total_segments': len(self.recent_predictions)
        }
        
        filename = f"live_predictions_{results['timestamp']}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Predictions saved to {filename}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Live fingerspelling inference')
    parser.add_argument('--mode', choices=['webcam', 'video'], default='webcam',
                       help='Inference mode')
    parser.add_argument('--video', type=str, help='Video file path (for video mode)')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID (for webcam mode)')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                       help='Path to config file')
    parser.add_argument('--window-size', type=int, default=30,
                       help='Frame window size for processing')
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.mode == 'video' and not args.video:
        print("ERROR: --video required for video mode")
        return
    
    if not Path(args.checkpoint).exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        return
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create inference engine
    try:
        engine = LiveInferenceEngine(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            device=device,
            window_size=args.window_size
        )
        
        # Run inference
        if args.mode == 'webcam':
            engine.run_webcam(args.camera)
        else:
            engine.run_video_file(args.video)
            
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
