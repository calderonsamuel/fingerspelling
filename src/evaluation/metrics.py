"""
Evaluation metrics for fingerspelling detection and recognition.
Implements AP@IoU, AP@Acc, and MSA metrics from the paper.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import editdistance

from ..utils.types import TemporalSegment, CHAR_TO_IDX, IDX_TO_CHAR


class DetectionEvaluator:
    """Evaluator for temporal segment detection."""
    
    def __init__(self, iou_thresholds: List[float] = [0.1, 0.3, 0.5]):
        """
        Initialize detection evaluator.
        
        Args:
            iou_thresholds: IoU thresholds for AP calculation
        """
        self.iou_thresholds = iou_thresholds
    
    def calculate_iou(self, pred_segment: TemporalSegment, gt_segment: TemporalSegment) -> float:
        """Calculate IoU between predicted and ground truth segments."""
        return pred_segment.iou(gt_segment)
    
    def match_segments(
        self, 
        pred_segments: List[TemporalSegment], 
        gt_segments: List[TemporalSegment],
        iou_threshold: float
    ) -> Tuple[List[int], List[bool]]:
        """
        Match predicted segments to ground truth segments.
        
        Args:
            pred_segments: Predicted segments sorted by confidence
            gt_segments: Ground truth segments
            iou_threshold: IoU threshold for matching
            
        Returns:
            Tuple of (matched_gt_indices, is_match_flags)
        """
        matched_gt_indices = []
        is_match_flags = []
        used_gt_indices = set()
        
        for pred_seg in pred_segments:
            best_iou = 0.0
            best_gt_idx = -1
            
            for gt_idx, gt_seg in enumerate(gt_segments):
                if gt_idx in used_gt_indices:
                    continue
                
                iou = self.calculate_iou(pred_seg, gt_seg)
                if iou > best_iou and iou > iou_threshold:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_gt_idx >= 0:
                matched_gt_indices.append(best_gt_idx)
                is_match_flags.append(True)
                used_gt_indices.add(best_gt_idx)
            else:
                matched_gt_indices.append(-1)
                is_match_flags.append(False)
        
        return matched_gt_indices, is_match_flags
    
    def calculate_ap_iou(
        self, 
        pred_segments: List[TemporalSegment], 
        gt_segments: List[TemporalSegment],
        iou_threshold: float
    ) -> float:
        """
        Calculate Average Precision at IoU threshold.
        
        Args:
            pred_segments: Predicted segments sorted by confidence
            gt_segments: Ground truth segments
            iou_threshold: IoU threshold
            
        Returns:
            Average Precision
        """
        if not gt_segments:
            return 1.0 if not pred_segments else 0.0
        
        if not pred_segments:
            return 0.0
        
        # Match segments
        matched_gt_indices, is_match_flags = self.match_segments(
            pred_segments, gt_segments, iou_threshold
        )
        
        # Calculate precision and recall at each threshold
        precisions = []
        recalls = []
        
        for i in range(len(pred_segments)):
            tp = sum(is_match_flags[:i+1])
            fp = (i + 1) - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / len(gt_segments) if len(gt_segments) > 0 else 0.0
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Calculate AP using trapezoidal rule
        ap = 0.0
        for i in range(len(recalls)):
            if i == 0:
                ap += precisions[i] * recalls[i]
            else:
                ap += precisions[i] * (recalls[i] - recalls[i-1])
        
        return ap


class RecognitionEvaluator:
    """Evaluator for letter sequence recognition."""
    
    def __init__(self, blank_idx: int = CHAR_TO_IDX['<blank>']):
        """
        Initialize recognition evaluator.
        
        Args:
            blank_idx: Index of blank token
        """
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
        pred_indices = torch.argmax(log_probs, dim=-1)
        
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
    
    def calculate_accuracy(self, pred_str: str, target_str: str) -> float:
        """
        Calculate letter accuracy (1 - normalized edit distance).
        
        Args:
            pred_str: Predicted string
            target_str: Target string
            
        Returns:
            Accuracy between 0 and 1
        """
        if len(target_str) == 0:
            return 1.0 if len(pred_str) == 0 else 0.0
        
        edit_dist = editdistance.eval(pred_str, target_str)
        accuracy = 1.0 - (edit_dist / len(target_str))
        return max(0.0, accuracy)


class FingerspellingEvaluator:
    """Combined evaluator for fingerspelling detection and recognition."""
    
    def __init__(
        self,
        iou_thresholds: List[float] = [0.1, 0.3, 0.5],
        acc_thresholds: List[float] = [0.0, 0.2, 0.4]
    ):
        """
        Initialize evaluator.
        
        Args:
            iou_thresholds: IoU thresholds for AP@IoU
            acc_thresholds: Accuracy thresholds for AP@Acc
        """
        self.iou_thresholds = iou_thresholds
        self.acc_thresholds = acc_thresholds
        
        self.detection_evaluator = DetectionEvaluator(iou_thresholds)
        self.recognition_evaluator = RecognitionEvaluator()
    
    def calculate_ap_acc(
        self,
        pred_segments: List[TemporalSegment],
        gt_segments: List[TemporalSegment],
        pred_recognitions: List[str],
        gt_recognitions: List[str],
        acc_threshold: float,
        iou_threshold: float = 0.0
    ) -> float:
        """
        Calculate Average Precision at Accuracy threshold.
        
        Args:
            pred_segments: Predicted segments sorted by confidence
            gt_segments: Ground truth segments
            pred_recognitions: Predicted letter sequences
            gt_recognitions: Ground truth letter sequences
            acc_threshold: Accuracy threshold
            iou_threshold: Minimum IoU for matching
            
        Returns:
            Average Precision at Accuracy threshold
        """
        if not gt_segments:
            return 1.0 if not pred_segments else 0.0
        
        if not pred_segments:
            return 0.0
        
        # Match segments based on both IoU and accuracy
        matched_gt_indices = []
        is_match_flags = []
        used_gt_indices = set()
        
        for i, pred_seg in enumerate(pred_segments):
            best_acc = 0.0
            best_gt_idx = -1
            
            for gt_idx, gt_seg in enumerate(gt_segments):
                if gt_idx in used_gt_indices:
                    continue
                
                # Check IoU threshold
                iou = self.detection_evaluator.calculate_iou(pred_seg, gt_seg)
                if iou <= iou_threshold:
                    continue
                
                # Check accuracy threshold
                if i < len(pred_recognitions) and gt_idx < len(gt_recognitions):
                    accuracy = self.recognition_evaluator.calculate_accuracy(
                        pred_recognitions[i], gt_recognitions[gt_idx]
                    )
                    
                    if accuracy > best_acc and accuracy > acc_threshold:
                        best_acc = accuracy
                        best_gt_idx = gt_idx
            
            if best_gt_idx >= 0:
                matched_gt_indices.append(best_gt_idx)
                is_match_flags.append(True)
                used_gt_indices.add(best_gt_idx)
            else:
                matched_gt_indices.append(-1)
                is_match_flags.append(False)
        
        # Calculate AP
        if not is_match_flags:
            return 0.0
        
        precisions = []
        recalls = []
        
        for i in range(len(pred_segments)):
            tp = sum(is_match_flags[:i+1])
            fp = (i + 1) - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / len(gt_segments) if len(gt_segments) > 0 else 0.0
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Calculate AP
        ap = 0.0
        for i in range(len(recalls)):
            if i == 0:
                ap += precisions[i] * recalls[i]
            else:
                ap += precisions[i] * (recalls[i] - recalls[i-1])
        
        return ap
    
    def calculate_msa(
        self,
        pred_segments: List[TemporalSegment],
        pred_recognitions: List[str],
        gt_sequence: str,
        video_length: int,
        score_thresholds: List[float] = None
    ) -> float:
        """
        Calculate Maximum Sequence Accuracy.
        
        Args:
            pred_segments: Predicted segments with confidence scores
            pred_recognitions: Predicted letter sequences
            gt_sequence: Ground truth full sequence
            video_length: Total video length in frames
            score_thresholds: Score thresholds to try
            
        Returns:
            Maximum Sequence Accuracy
        """
        if score_thresholds is None:
            score_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        max_accuracy = 0.0
        
        for threshold in score_thresholds:
            # Filter segments by threshold
            filtered_segments = []
            filtered_recognitions = []
            
            for i, segment in enumerate(pred_segments):
                if segment.confidence >= threshold:
                    filtered_segments.append(segment)
                    if i < len(pred_recognitions):
                        filtered_recognitions.append(pred_recognitions[i])
            
            # Apply non-maximum suppression
            nms_segments, nms_recognitions = self._apply_nms(
                filtered_segments, filtered_recognitions, iou_threshold=0.5
            )
            
            # Construct full sequence
            pred_full_sequence = self._construct_full_sequence(
                nms_segments, nms_recognitions, video_length
            )
            
            # Calculate accuracy
            accuracy = self.recognition_evaluator.calculate_accuracy(
                pred_full_sequence, gt_sequence
            )
            
            max_accuracy = max(max_accuracy, accuracy)
        
        return max_accuracy
    
    def _apply_nms(
        self, 
        segments: List[TemporalSegment], 
        recognitions: List[str],
        iou_threshold: float = 0.5
    ) -> Tuple[List[TemporalSegment], List[str]]:
        """Apply non-maximum suppression to segments."""
        if not segments:
            return [], []
        
        # Sort by confidence (descending)
        sorted_indices = sorted(
            range(len(segments)), 
            key=lambda i: segments[i].confidence, 
            reverse=True
        )
        
        keep_indices = []
        
        for i in sorted_indices:
            keep = True
            
            for j in keep_indices:
                iou = segments[i].iou(segments[j])
                if iou > iou_threshold:
                    keep = False
                    break
            
            if keep:
                keep_indices.append(i)
        
        nms_segments = [segments[i] for i in keep_indices]
        nms_recognitions = [recognitions[i] for i in keep_indices if i < len(recognitions)]
        
        return nms_segments, nms_recognitions
    
    def _construct_full_sequence(
        self, 
        segments: List[TemporalSegment], 
        recognitions: List[str],
        video_length: int
    ) -> str:
        """Construct full video sequence from segments."""
        if not segments:
            return ""
        
        # Sort segments by start time
        sorted_pairs = sorted(
            zip(segments, recognitions), 
            key=lambda x: x[0].start_frame
        )
        
        full_sequence = ""
        
        for i, (segment, recognition) in enumerate(sorted_pairs):
            full_sequence += recognition
            
            # Add separator if not the last segment
            if i < len(sorted_pairs) - 1:
                full_sequence += " "  # Space separator
        
        return full_sequence
    
    def evaluate_batch(
        self,
        batch_predictions: Dict,
        batch_targets: Dict
    ) -> Dict[str, float]:
        """
        Evaluate a batch of predictions.
        
        Args:
            batch_predictions: Batch predictions from model
            batch_targets: Batch ground truth targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        # This is a simplified version - full implementation would process
        # the batch predictions and targets to extract segments and recognitions
        
        metrics = {}
        
        # Calculate AP@IoU for different thresholds
        for iou_thresh in self.iou_thresholds:
            metrics[f'ap_iou_{iou_thresh}'] = 0.0  # Placeholder
        
        # Calculate AP@Acc for different thresholds
        for acc_thresh in self.acc_thresholds:
            metrics[f'ap_acc_{acc_thresh}'] = 0.0  # Placeholder
        
        # Calculate MSA
        metrics['msa'] = 0.0  # Placeholder
        
        return metrics


def create_evaluator(
    iou_thresholds: List[float] = [0.1, 0.3, 0.5],
    acc_thresholds: List[float] = [0.0, 0.2, 0.4]
) -> FingerspellingEvaluator:
    """
    Create a fingerspelling evaluator.
    
    Args:
        iou_thresholds: IoU thresholds for evaluation
        acc_thresholds: Accuracy thresholds for evaluation
        
    Returns:
        Fingerspelling evaluator
    """
    return FingerspellingEvaluator(
        iou_thresholds=iou_thresholds,
        acc_thresholds=acc_thresholds
    )
