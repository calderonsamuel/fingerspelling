"""
Multi-task model for fingerspelling detection and recognition.
Based on YOLO architecture with custom heads for different tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from ultralytics import YOLO

from ..utils.types import CHAR_TO_IDX, NUM_POSE_KEYPOINTS


class TemporalDetectionHead(nn.Module):
    """Detection head for temporal segment detection."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        """
        Initialize detection head.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.conv1d = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(hidden_dim)
        
        # Classification head (fingerspelling vs non-fingerspelling)
        self.cls_head = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        
        # Regression head (segment boundaries)
        self.reg_head = nn.Conv1d(hidden_dim, 2, kernel_size=1)  # start, end offsets
        
        # Confidence head
        self.conf_head = nn.Conv1d(hidden_dim, 1, kernel_size=1)
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            features: Input features (B, C, T)
            
        Returns:
            Dictionary with detection outputs
        """
        x = F.relu(self.norm(self.conv1d(features)))
        
        # Classification output (probability of fingerspelling)
        cls_output = torch.sigmoid(self.cls_head(x))  # (B, 1, T)
        
        # Regression output (segment boundaries)
        reg_output = self.reg_head(x)  # (B, 2, T)
        
        # Confidence output
        conf_output = torch.sigmoid(self.conf_head(x))  # (B, 1, T)
        
        return {
            'classification': cls_output,
            'regression': reg_output,
            'confidence': conf_output
        }


class RecognitionHead(nn.Module):
    """Recognition head for letter sequence prediction using CTC."""
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 512,
        num_classes: int = len(CHAR_TO_IDX),
        num_layers: int = 2
    ):
        """
        Initialize recognition head.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: LSTM hidden dimension
            num_classes: Number of character classes
            num_layers: Number of LSTM layers
        """
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Bidirectional LSTM for sequence modeling
        self.lstm = nn.LSTM(
            hidden_dim, 
            hidden_dim // 2,  # Half size since bidirectional
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Output projection to character classes
        self.output_projection = nn.Linear(hidden_dim, num_classes)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: Input features (B, T, C)
            
        Returns:
            Character class logits (B, T, num_classes)
        """
        # Project input features
        x = self.input_projection(features)  # (B, T, hidden_dim)
        x = F.relu(x)
        x = self.dropout(x)
        
        # LSTM processing
        x, _ = self.lstm(x)  # (B, T, hidden_dim)
        x = self.dropout(x)
        
        # Output projection
        output = self.output_projection(x)  # (B, T, num_classes)
        
        return F.log_softmax(output, dim=-1)


class PoseEstimationHead(nn.Module):
    """Pose estimation head for auxiliary pose supervision."""
    
    def __init__(
        self, 
        input_channels: int, 
        num_keypoints: int = NUM_POSE_KEYPOINTS,
        hidden_channels: int = 256
    ):
        """
        Initialize pose estimation head.
        
        Args:
            input_channels: Input feature channels
            num_keypoints: Number of pose keypoints
            hidden_channels: Hidden layer channels
        """
        super().__init__()
        
        # Deconvolutional layers to upsample features
        self.deconv1 = nn.ConvTranspose2d(
            input_channels, hidden_channels, 
            kernel_size=4, stride=2, padding=1
        )
        self.norm1 = nn.BatchNorm2d(hidden_channels)
        
        self.deconv2 = nn.ConvTranspose2d(
            hidden_channels, hidden_channels // 2,
            kernel_size=4, stride=2, padding=1
        )
        self.norm2 = nn.BatchNorm2d(hidden_channels // 2)
        
        # Final layer to produce heatmaps
        self.final_conv = nn.Conv2d(
            hidden_channels // 2, num_keypoints,
            kernel_size=1, stride=1, padding=0
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: Input features (B*T, C, H, W)
            
        Returns:
            Pose heatmaps (B*T, num_keypoints, H', W')
        """
        x = F.relu(self.norm1(self.deconv1(features)))
        x = F.relu(self.norm2(self.deconv2(x)))
        x = self.final_conv(x)
        
        return x


class MultiTaskFingerspellingModel(nn.Module):
    """
    Multi-task model for fingerspelling detection and recognition.
    Uses YOLOv8 backbone with custom heads.
    """
    
    def __init__(
        self,
        backbone_model: str = "yolov8n.pt",
        detection_hidden_dim: int = 256,
        recognition_hidden_dim: int = 512,
        pose_hidden_channels: int = 256,
        use_pose: bool = True,
        freeze_backbone: bool = False
    ):
        """
        Initialize the multi-task model.
        
        Args:
            backbone_model: YOLO model name or path
            detection_hidden_dim: Hidden dimension for detection head
            recognition_hidden_dim: Hidden dimension for recognition head
            pose_hidden_channels: Hidden channels for pose head
            use_pose: Whether to use pose estimation head
            freeze_backbone: Whether to freeze backbone weights
        """
        super().__init__()
        
        # Load YOLO backbone - use a simpler CNN backbone for now
        # We'll create a simple backbone instead of using YOLO's complex structure
        self.backbone = self._create_simple_backbone()
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get feature dimensions from backbone
        self.feature_channels = self._get_feature_channels()
        
        # Feature aggregation for temporal modeling
        self.temporal_conv = nn.Conv1d(
            self.feature_channels, 
            detection_hidden_dim, 
            kernel_size=3, 
            padding=1
        )
        
        # Multi-task heads
        self.detection_head = TemporalDetectionHead(
            input_dim=detection_hidden_dim,
            hidden_dim=detection_hidden_dim
        )
        
        self.recognition_head = RecognitionHead(
            input_dim=detection_hidden_dim,
            hidden_dim=recognition_hidden_dim
        )
        
        self.use_pose = use_pose
        if use_pose:
            self.pose_head = PoseEstimationHead(
                input_channels=self.feature_channels,
                hidden_channels=pose_hidden_channels
            )
    
    def _get_feature_channels(self) -> int:
        """Get the number of feature channels from backbone."""
        # For our simple backbone, the output channels are 512
        return 512
    
    def extract_spatial_features(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial features from video frames.
        
        Args:
            frames: Input frames (B, T, 3, H, W)
            
        Returns:
            Spatial features (B, T, C, H', W')
        """
        B, T, C, H, W = frames.shape
        
        # Reshape for backbone processing
        frames_reshaped = frames.view(B * T, C, H, W)
        
        # Extract features using backbone
        with torch.set_grad_enabled(self.training):
            features = self.backbone(frames_reshaped)
            if isinstance(features, (list, tuple)):
                features = features[-1]  # Take last feature map
        
        # Reshape back to temporal format
        _, feat_C, feat_H, feat_W = features.shape
        features = features.view(B, T, feat_C, feat_H, feat_W)
        
        return features
    
    def aggregate_temporal_features(self, spatial_features: torch.Tensor) -> torch.Tensor:
        """
        Aggregate spatial features across time.
        
        Args:
            spatial_features: Spatial features (B, T, C, H, W)
            
        Returns:
            Temporal features (B, T, C')
        """
        B, T, C, H, W = spatial_features.shape
        
        # Global average pooling over spatial dimensions
        pooled_features = F.adaptive_avg_pool2d(
            spatial_features.view(-1, C, H, W), 
            (1, 1)
        ).view(B, T, C)
        
        # Apply temporal convolution
        # Convert to (B, C, T) for Conv1d
        pooled_features = pooled_features.transpose(1, 2)
        temporal_features = self.temporal_conv(pooled_features)
        
        return temporal_features  # (B, C', T)
    
    def forward(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            frames: Input video frames (B, T, 3, H, W)
            
        Returns:
            Dictionary with outputs from all heads
        """
        # Extract spatial features
        spatial_features = self.extract_spatial_features(frames)  # (B, T, C, H, W)
        
        # Aggregate temporal features
        temporal_features = self.aggregate_temporal_features(spatial_features)  # (B, C', T)
        
        # Detection head
        detection_output = self.detection_head(temporal_features)
        
        # Recognition head (needs (B, T, C) format)
        recognition_input = temporal_features.transpose(1, 2)  # (B, T, C')
        recognition_output = self.recognition_head(recognition_input)
        
        outputs = {
            'detection': detection_output,
            'recognition': recognition_output
        }
        
        # Pose estimation head (if enabled)
        if self.use_pose:
            B, T, C, H, W = spatial_features.shape
            pose_input = spatial_features.view(B * T, C, H, W)
            pose_output = self.pose_head(pose_input)
            outputs['pose'] = pose_output
        
        return outputs
    
    def _create_simple_backbone(self) -> nn.Module:
        """Create a simple CNN backbone for feature extraction."""
        return nn.Sequential(
            # First conv block
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Second conv block
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Third conv block
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Fourth conv block
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )


def create_model(
    backbone_model: str = "yolov8n.pt",
    use_pose: bool = True,
    freeze_backbone: bool = False
) -> MultiTaskFingerspellingModel:
    """
    Create a multi-task fingerspelling model.
    
    Args:
        backbone_model: YOLO backbone model
        use_pose: Whether to include pose estimation
        freeze_backbone: Whether to freeze backbone
        
    Returns:
        Multi-task model
    """
    return MultiTaskFingerspellingModel(
        backbone_model=backbone_model,
        use_pose=use_pose,
        freeze_backbone=freeze_backbone
    )
