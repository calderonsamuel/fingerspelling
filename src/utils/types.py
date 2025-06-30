"""
Core data types for fingerspelling detection and recognition.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np


@dataclass
class SequenceInfo:
    """Information about a fingerspelling sequence."""
    filename: str
    url: str
    start_time: str
    number_of_frames: int
    width: int
    height: int
    label_proc: str
    label_raw: str
    label_notes: str
    partition: str  # train/dev/test
    signer: str


@dataclass
class TemporalSegment:
    """A temporal segment with start and end frame indices."""
    start_frame: int
    end_frame: int
    confidence: float = 1.0
    
    def __post_init__(self) -> None:
        if self.start_frame >= self.end_frame:
            raise ValueError("Start frame must be less than end frame")
    
    @property
    def length(self) -> int:
        """Length of the segment in frames."""
        return self.end_frame - self.start_frame
    
    def iou(self, other: "TemporalSegment") -> float:
        """Calculate IoU (Intersection over Union) with another segment."""
        intersection_start = max(self.start_frame, other.start_frame)
        intersection_end = min(self.end_frame, other.end_frame)
        
        if intersection_start >= intersection_end:
            return 0.0
        
        intersection = intersection_end - intersection_start
        union = self.length + other.length - intersection
        
        return intersection / union if union > 0 else 0.0


@dataclass
class DetectionTarget:
    """Target for detection task."""
    segments: List[TemporalSegment]
    video_length: int  # Total number of frames in video


@dataclass
class RecognitionTarget:
    """Target for recognition task."""
    letter_sequence: str
    segment: TemporalSegment


@dataclass
class PoseTarget:
    """Target for pose estimation task."""
    keypoints: np.ndarray  # Shape: (num_frames, num_keypoints, 3) - x, y, confidence
    frame_indices: List[int]


@dataclass
class MultiTaskTarget:
    """Combined target for multi-task learning."""
    detection: DetectionTarget
    recognition: List[RecognitionTarget]
    pose: Optional[PoseTarget] = None


@dataclass
class ModelOutput:
    """Output from the multi-task model."""
    detection_segments: List[TemporalSegment]
    recognition_outputs: List[str]
    pose_heatmaps: Optional[np.ndarray] = None


@dataclass
class EvaluationMetrics:
    """Metrics for model evaluation."""
    ap_iou: Dict[float, float]  # AP@IoU for different IoU thresholds
    ap_acc: Dict[float, float]  # AP@Acc for different accuracy thresholds
    msa: float  # Maximum Sequence Accuracy
    
    
# Constants
ALPHABET = "abcdefghijklmnopqrstuvwxyz"
SPECIAL_TOKENS = {
    'blank': '<blank>',
    'space': '<sp>',
    'unknown': '<unk>'
}

# Character to index mapping for CTC
CHAR_TO_IDX = {char: idx for idx, char in enumerate(ALPHABET)}
CHAR_TO_IDX.update({
    SPECIAL_TOKENS['blank']: len(ALPHABET),
    SPECIAL_TOKENS['space']: len(ALPHABET) + 1,
    SPECIAL_TOKENS['unknown']: len(ALPHABET) + 2
})

IDX_TO_CHAR = {idx: char for char, idx in CHAR_TO_IDX.items()}

# Number of pose keypoints (OpenPose format)
NUM_POSE_KEYPOINTS = 58  # 15 body + 21 left hand + 21 right hand + 1 face
