"""
F.A.L.C.O.N. – Occlusion Detection Module
===========================================

Classifies per-person occlusion state based on YOLO keypoint visibility,
detection confidence, and tracking continuity.  The four-level enum maps
directly to visualization colour and radar-triggering logic.

COCO 17-keypoint layout reference
----------------------------------
 0  nose          5  left_shoulder   11 left_hip
 1  left_eye      6  right_shoulder  12 right_hip
 2  right_eye     7  left_elbow      13 left_knee
 3  left_ear      8  right_elbow     14 right_knee
 4  right_ear     9  left_wrist      15 left_ankle
                 10  right_wrist     16 right_ankle
"""

from __future__ import annotations

from enum import IntEnum
from typing import Dict, List, Optional

import numpy as np


# ── OcclusionState enum ─────────────────────────────────────────────

class OcclusionState(IntEnum):
    """Five-tier occlusion classification (includes recovery)."""
    VISIBLE = 0
    PARTIALLY_OCCLUDED = 1
    HEAVILY_OCCLUDED = 2
    LOST = 3
    RECOVERED = 4

    @property
    def label(self) -> str:
        _labels = {
            0: "VISIBLE",
            1: "PARTIAL",
            2: "HEAVY",
            3: "LOST",
            4: "RECOVERED",
        }
        return _labels[self.value]

    @property
    def colour_bgr(self) -> tuple:
        """OpenCV colour for visualization."""
        _colours = {
            0: (0, 255, 0),       # green
            1: (0, 255, 255),     # yellow
            2: (0, 0, 255),       # red
            3: (150, 150, 150),   # gray
            4: (255, 200, 0),     # cyan-ish blue – recovery indicator
        }
        return _colours[self.value]


# ── Body-region groupings ───────────────────────────────────────────

# Maps readable region names → COCO keypoint indices
BODY_REGIONS: Dict[str, List[int]] = {
    "head":       [0, 1, 2, 3, 4],
    "upper_body": [5, 6, 7, 8, 9, 10],
    "lower_body": [11, 12, 13, 14, 15, 16],
    "left_side":  [1, 3, 5, 7, 9, 11, 13, 15],
    "right_side": [2, 4, 6, 8, 10, 12, 14, 16],
}


# ── OcclusionAnalyzer ───────────────────────────────────────────────

class OcclusionAnalyzer:
    """
    Determines :class:`OcclusionState` for a tracked person using
    keypoint visibility counts and detection confidence.

    Parameters
    ----------
    kp_conf_threshold : float
        Minimum per-keypoint confidence to count as *visible*.
    visible_kp_min : int
        Keypoints needed for VISIBLE classification.
    partial_kp_min : int
        Keypoints needed for PARTIALLY_OCCLUDED (below this → HEAVILY).
    det_conf_high : float
        Detection confidence above which person is considered fully visible.
    det_conf_low : float
        Detection confidence below which person is considered partially occluded.
    lost_frame_limit : int
        Frames without detection before classifying as LOST.
    """

    def __init__(
        self,
        kp_conf_threshold: float = 0.4,
        visible_kp_min: int = 12,
        partial_kp_min: int = 5,
        det_conf_high: float = 0.6,
        det_conf_low: float = 0.4,
        lost_frame_limit: int = 90,
    ):
        self.kp_conf_threshold = kp_conf_threshold
        self.visible_kp_min = visible_kp_min
        self.partial_kp_min = partial_kp_min
        self.det_conf_high = det_conf_high
        self.det_conf_low = det_conf_low
        self.lost_frame_limit = lost_frame_limit

    # ── public API ───────────────────────────────────────────────────

    def analyze(
        self,
        keypoints: Optional[np.ndarray],
        bbox: Optional[np.ndarray],
        detection_confidence: float,
        frames_since_detection: int = 0,
    ) -> OcclusionState:
        """
        Classify occlusion level for one person.

        Parameters
        ----------
        keypoints : ndarray | None
            Shape (17, 3)  –  [x, y, conf] per keypoint, or *None* when
            only a predicted bbox is available.
        bbox : ndarray | None
            [x1, y1, x2, y2] (used for future region analysis).
        detection_confidence : float
            Overall detection confidence from YOLO (0-1).
        frames_since_detection : int
            How many frames since this person was last *detected* (0 = detected
            this frame).

        Returns
        -------
        OcclusionState
        """
        # LOST if we haven't seen the person for a long time
        if frames_since_detection >= self.lost_frame_limit:
            return OcclusionState.LOST

        # If no detection this frame, rely on frame gap
        if frames_since_detection > 0:
            # Still within tracking window → HEAVILY_OCCLUDED
            return OcclusionState.HEAVILY_OCCLUDED

        # Detection available this frame – use keypoints + confidence
        n_visible = self._count_visible_keypoints(keypoints)

        if n_visible >= self.visible_kp_min and detection_confidence > self.det_conf_high:
            return OcclusionState.VISIBLE

        if n_visible >= self.partial_kp_min or detection_confidence >= self.det_conf_low:
            return OcclusionState.PARTIALLY_OCCLUDED

        return OcclusionState.HEAVILY_OCCLUDED

    def region_visibility(
        self, keypoints: Optional[np.ndarray]
    ) -> Dict[str, float]:
        """
        Return per-body-region visibility ratio (0.0-1.0).

        Useful for downstream analysis (e.g., "upper body occluded while
        lower body visible").
        """
        result: Dict[str, float] = {}
        if keypoints is None or len(keypoints) == 0:
            return {name: 0.0 for name in BODY_REGIONS}

        for name, indices in BODY_REGIONS.items():
            visible = sum(
                1
                for idx in indices
                if idx < len(keypoints) and keypoints[idx, 2] >= self.kp_conf_threshold
            )
            result[name] = visible / max(len(indices), 1)
        return result

    # ── internal helpers ─────────────────────────────────────────────

    def _count_visible_keypoints(self, keypoints: Optional[np.ndarray]) -> int:
        """Count keypoints whose confidence exceeds the threshold."""
        if keypoints is None or len(keypoints) == 0:
            return 0
        return int(np.sum(keypoints[:, 2] >= self.kp_conf_threshold))
