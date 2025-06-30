"""
Loss functions for multi-task fingerspelling model.
Implements detection loss, recognition loss (CTC), letter error rate loss, and pose loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import editdistance

from ..utils.types import CHAR_TO_IDX, IDX_TO_CHAR


class DetectionLoss(nn.Module):
    """Loss for temporal segment detection."""
    
    def __init__(self, cls_weight: float = 1.0, reg_weight: float = 1.0):
        """
        Initialize detection loss.
        
        Args:
            cls_weight: Weight for classification loss
            reg_weight: Weight for regression loss
        """
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        
        # Use focal loss for classification to handle class imbalance
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    
    def forward(
        self, 
        predictions: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute detection loss.
        
        Args:
            predictions: Detection predictions
            targets: Detection targets
            
        Returns:
            Dictionary with loss components
        """
        # Classification loss (frame-level)
        cls_pred = predictions['classification']  # (B, 1, T)
        cls_target = targets['frame_labels'].unsqueeze(1)  # (B, 1, T)
        
        cls_loss = self.focal_loss(cls_pred, cls_target)
        
        # Regression loss (only for positive frames)
        reg_pred = predictions['regression']  # (B, 2, T)
        seg_targets = targets['segment_targets']  # (B, max_segments, 3)
        
        # Create regression targets for each frame
        reg_loss = torch.tensor(0.0, device=cls_pred.device)
        
        # Simplified regression loss - can be improved
        if seg_targets.sum() > 0:  # If there are valid segments
            # For now, use MSE loss on segment targets
            # This is a simplified approach - full implementation would use 
            # proper anchor-based regression like in YOLO
            reg_loss = F.mse_loss(reg_pred.mean(dim=2), seg_targets[..., :2].mean(dim=1))
        
        total_loss = self.cls_weight * cls_loss + self.reg_weight * reg_loss
        
        return {
            'classification_loss': cls_loss,
            'regression_loss': reg_loss,
            'total_detection_loss': total_loss
        }


class RecognitionLoss(nn.Module):
    """CTC loss for recognition task."""
    
    def __init__(self, blank_idx: int = CHAR_TO_IDX['<blank>']):
        """
        Initialize recognition loss.
        
        Args:
            blank_idx: Index of blank token for CTC
        """
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank_idx, reduction='mean', zero_infinity=True)
    
    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute CTC loss.
        
        Args:
            predictions: Recognition predictions (B, T, num_classes)
            targets: Recognition targets
            
        Returns:
            CTC loss
        """
        log_probs = predictions  # Already log-softmax from model
        target_sequences = targets['letter_targets']  # (B, max_length)
        target_lengths = targets['target_lengths']  # (B,)
        
        # Get input lengths (sequence length for each batch item)
        input_lengths = torch.full((log_probs.size(0),), log_probs.size(1), dtype=torch.long)
        
        # Remove padding from targets
        targets_list = []
        target_lengths_list = []
        
        for i in range(target_sequences.size(0)):
            target_len = target_lengths[i].item()
            target_seq = target_sequences[i, :target_len]
            targets_list.append(target_seq)
            target_lengths_list.append(target_len)
        
        # Concatenate targets for CTC
        targets_concat = torch.cat(targets_list)
        target_lengths_tensor = torch.tensor(target_lengths_list, dtype=torch.long)
        
        # Transpose predictions for CTC (T, B, C)
        log_probs = log_probs.transpose(0, 1)
        
        loss = self.ctc_loss(log_probs, targets_concat, input_lengths, target_lengths_tensor)
        
        return loss


class LetterErrorRateLoss(nn.Module):
    """Letter Error Rate (LER) loss using REINFORCE-like gradient estimation."""
    
    def __init__(self, blank_idx: int = CHAR_TO_IDX['<blank>']):
        """
        Initialize LER loss.
        
        Args:
            blank_idx: Index of blank token
        """
        super().__init__()
        self.blank_idx = blank_idx
    
    def decode_prediction(self, log_probs: torch.Tensor) -> str:
        """
        Decode CTC prediction to string.
        
        Args:
            log_probs: Log probabilities (T, num_classes)
            
        Returns:
            Decoded string
        """
        # Simple greedy decoding
        pred_indices = torch.argmax(log_probs, dim=-1)  # (T,)
        
        # Remove blanks and consecutive duplicates
        decoded = []
        prev_idx = None
        
        for idx in pred_indices:
            idx = idx.item()
            if idx != self.blank_idx and idx != prev_idx:
                if idx in IDX_TO_CHAR:
                    decoded.append(IDX_TO_CHAR[idx])
            prev_idx = idx
        
        return ''.join(decoded)
    
    def compute_accuracy(self, pred_str: str, target_str: str) -> float:
        """
        Compute letter accuracy between prediction and target.
        
        Args:
            pred_str: Predicted string
            target_str: Target string
            
        Returns:
            Accuracy (1 - normalized edit distance)
        """
        if len(target_str) == 0:
            return 1.0 if len(pred_str) == 0 else 0.0
        
        edit_dist = editdistance.eval(pred_str, target_str)
        accuracy = 1.0 - (edit_dist / len(target_str))
        return max(0.0, accuracy)  # Ensure non-negative
    
    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: Dict[str, torch.Tensor],
        segment_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute LER loss.
        
        Args:
            predictions: Recognition predictions (B, T, num_classes)
            targets: Recognition targets
            segment_probs: Segment probabilities from detection head (B, T)
            
        Returns:
            LER loss
        """
        batch_size = predictions.size(0)
        device = predictions.device
        
        total_loss = torch.tensor(0.0, device=device)
        
        for i in range(batch_size):
            # Decode prediction
            pred_log_probs = predictions[i]  # (T, num_classes)
            pred_str = self.decode_prediction(pred_log_probs)
            
            # Get target string
            target_seq = targets['letter_targets'][i]
            target_len = targets['target_lengths'][i].item()
            target_indices = target_seq[:target_len].cpu().numpy()
            
            target_str = ''.join([
                IDX_TO_CHAR[idx] for idx in target_indices 
                if idx in IDX_TO_CHAR and idx != self.blank_idx
            ])
            
            # Compute accuracy
            accuracy = self.compute_accuracy(pred_str, target_str)
            
            # REINFORCE-like gradient estimation
            # Use negative accuracy as reward (we want to minimize negative accuracy)
            reward = -accuracy
            
            # Get segment probability for this sample
            seg_prob = segment_probs[i].mean()  # Average over time
            
            # Loss is reward * log(probability)
            loss_i = reward * torch.log(seg_prob + 1e-8)
            total_loss += loss_i
        
        return total_loss / batch_size


class PoseLoss(nn.Module):
    """Loss for pose estimation task."""
    
    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize pose loss.
        
        Args:
            confidence_threshold: Threshold for pose confidence
        """
        super().__init__()
        self.confidence_threshold = confidence_threshold
    
    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        confidences: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pose loss.
        
        Args:
            predictions: Predicted pose heatmaps (B*T, num_keypoints, H, W)
            targets: Target pose heatmaps (B*T, num_keypoints, H, W)
            confidences: Confidence scores for each keypoint (B*T, num_keypoints)
            
        Returns:
            Pose loss
        """
        # Only compute loss for confident keypoints
        mask = confidences > self.confidence_threshold
        
        if mask.sum() == 0:
            return torch.tensor(0.0, device=predictions.device)
        
        # Compute MSE loss only for confident keypoints
        loss = F.mse_loss(predictions, targets, reduction='none')
        
        # Apply confidence mask
        mask_expanded = mask.unsqueeze(-1).unsqueeze(-1)  # (B*T, num_keypoints, 1, 1)
        masked_loss = loss * mask_expanded.float()
        
        return masked_loss.sum() / mask.sum()


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Initialize focal loss.
        
        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            predictions: Predicted probabilities (B, 1, T)
            targets: Target labels (B, 1, T)
            
        Returns:
            Focal loss
        """
        # Binary cross entropy
        bce_loss = F.binary_cross_entropy(predictions, targets, reduction='none')
        
        # Focal loss components
        pt = torch.where(targets == 1, predictions, 1 - predictions)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        
        focal_loss = focal_weight * bce_loss
        
        return focal_loss.mean()


class MultiTaskLoss(nn.Module):
    """Combined loss for multi-task learning."""
    
    def __init__(
        self,
        detection_weight: float = 1.0,
        recognition_weight: float = 1.0,
        ler_weight: float = 0.1,
        pose_weight: float = 0.5
    ):
        """
        Initialize multi-task loss.
        
        Args:
            detection_weight: Weight for detection loss
            recognition_weight: Weight for recognition loss
            ler_weight: Weight for letter error rate loss
            pose_weight: Weight for pose loss
        """
        super().__init__()
        
        self.detection_weight = detection_weight
        self.recognition_weight = recognition_weight
        self.ler_weight = ler_weight
        self.pose_weight = pose_weight
        
        self.detection_loss = DetectionLoss()
        self.recognition_loss = RecognitionLoss()
        self.ler_loss = LetterErrorRateLoss()
        self.pose_loss = PoseLoss()
    
    def forward(
        self, 
        predictions: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary with all loss components
        """
        losses = {}
        
        # Detection loss
        detection_losses = self.detection_loss(
            predictions['detection'], 
            targets['detection_targets']
        )
        losses.update(detection_losses)
        
        # Recognition loss
        recognition_loss = self.recognition_loss(
            predictions['recognition'],
            targets['recognition_targets']
        )
        losses['recognition_loss'] = recognition_loss
        
        # Letter Error Rate loss
        segment_probs = predictions['detection']['confidence'].squeeze(1)  # (B, T)
        ler_loss = self.ler_loss(
            predictions['recognition'],
            targets['recognition_targets'],
            segment_probs
        )
        losses['ler_loss'] = ler_loss
        
        # Pose loss (if available)
        pose_loss = torch.tensor(0.0, device=recognition_loss.device)
        if 'pose' in predictions and 'pose_targets' in targets:
            pose_loss = self.pose_loss(
                predictions['pose'],
                targets['pose_targets']['heatmaps'],
                targets['pose_targets']['confidences']
            )
        losses['pose_loss'] = pose_loss
        
        # Total loss
        total_loss = (
            self.detection_weight * losses['total_detection_loss'] +
            self.recognition_weight * recognition_loss +
            self.ler_weight * ler_loss +
            self.pose_weight * pose_loss
        )
        losses['total_loss'] = total_loss
        
        return losses
