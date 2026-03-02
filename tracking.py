"""
F.A.L.C.O.N. – IoU-Based Person Tracking Module
=================================================

Maintains persistent person IDs across video frames using Intersection-over-Union
(IoU) matching between predicted and detected bounding boxes.  Each tracked person
carries a short history of bounding boxes / keypoints, an occlusion state, and a
Kalman-filter predictor that is used when detection is temporarily lost.

Key classes
-----------
PersonTracker   – per-person state (dataclass)
TrackingManager – orchestrates matching, creation, and removal of tracks
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from occlusion import OcclusionAnalyzer, OcclusionState
from prediction import KalmanPredictor


# ── Utility ─────────────────────────────────────────────────────────

def iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    """Compute IoU between two [x1, y1, x2, y2] bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    if inter == 0.0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter + 1e-6)


# ── PersonTracker dataclass ─────────────────────────────────────────

@dataclass
class PersonTracker:
    """State for a single tracked person."""

    track_id: int
    bbox: np.ndarray                         # latest [x1, y1, x2, y2]
    keypoints: Optional[np.ndarray] = None   # (17, 3) or None
    detection_conf: float = 0.0

    # History buffers (last N frames)
    bbox_history: list = field(default_factory=list)
    keypoint_history: list = field(default_factory=list)
    max_history: int = 30

    # Occlusion & prediction
    occlusion_state: OcclusionState = OcclusionState.VISIBLE
    predictor: KalmanPredictor = field(default_factory=KalmanPredictor)
    prediction_confidence: float = 1.0

    # Bookkeeping
    frames_since_detection: int = 0
    total_frames_tracked: int = 0
    is_predicted: bool = False
    last_seen_time: float = field(default_factory=time.time)
    last_bbox_size: Optional[np.ndarray] = None  # [w, h] for predicted box

    # Recovery tracking
    is_recovered: bool = False
    recovery_frames_remaining: int = 0       # countdown from recovery_duration → 0
    was_occluded: bool = False                # True while in any occluded state

    # ── helpers ──────────────────────────────────────────────────────

    def push_history(self) -> None:
        """Append current bbox/keypoints to history rings."""
        self.bbox_history.append(self.bbox.copy())
        if len(self.bbox_history) > self.max_history:
            self.bbox_history.pop(0)
        if self.keypoints is not None:
            self.keypoint_history.append(self.keypoints.copy())
            if len(self.keypoint_history) > self.max_history:
                self.keypoint_history.pop(0)

    @property
    def center(self) -> np.ndarray:
        """Return [cx, cy] of the latest bbox."""
        return np.array([
            (self.bbox[0] + self.bbox[2]) / 2.0,
            (self.bbox[1] + self.bbox[3]) / 2.0,
        ])

    @property
    def bbox_wh(self) -> np.ndarray:
        """Return [w, h] of the latest bbox."""
        return np.array([
            self.bbox[2] - self.bbox[0],
            self.bbox[3] - self.bbox[1],
        ])


# ── TrackingManager ─────────────────────────────────────────────────

class TrackingManager:
    """
    Frame-level tracker that associates new detections with existing
    tracks via IoU matching, manages track lifecycles, and delegates
    occlusion classification.

    Parameters
    ----------
    iou_threshold : float
        Minimum IoU to associate a detection with a track (default 0.3).
    max_frames_lost : int
        Number of frames without detection before a track is removed (default 30).
    prediction_decay : float
        Multiplicative decay applied to prediction_confidence each missed frame.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_frames_lost: int = 20,
        prediction_decay: float = 0.90,
        recovery_duration: int = 20,
    ):
        self.iou_threshold = iou_threshold
        self.max_frames_lost = max_frames_lost
        self.prediction_decay = prediction_decay
        self.recovery_duration = recovery_duration   # frames to hold RECOVERED tag (~3 s)

        self._next_id: int = 1
        self.tracks: List[PersonTracker] = []
        self.occlusion_analyzer = OcclusionAnalyzer()

    # ── public API ───────────────────────────────────────────────────

    def update(
        self,
        det_boxes: np.ndarray,
        det_keypoints: Optional[np.ndarray] = None,
        det_confs: Optional[np.ndarray] = None,
    ) -> List[PersonTracker]:
        """
        Process one frame of detections and return all active tracks.

        Parameters
        ----------
        det_boxes : ndarray, shape (N, 4)
            Detected bounding boxes [x1, y1, x2, y2].
        det_keypoints : ndarray | None, shape (N, 17, 3)
            Detected keypoints with confidence per keypoint.
        det_confs : ndarray | None, shape (N,)
            Detection-level confidence scores.

        Returns
        -------
        list[PersonTracker]
            All active tracks (both detected and predicted).
        """
        if det_boxes is None or len(det_boxes) == 0:
            det_boxes = np.empty((0, 4))
        if det_confs is None:
            det_confs = np.ones(len(det_boxes))

        # 1. Predict step for every existing track
        self._predict_all()

        # 2. Match detections ↔ tracks
        matched, unmatched_dets, unmatched_tracks = self._match_detections(det_boxes)

        # 3. Update matched tracks
        for t_idx, d_idx in matched:
            track = self.tracks[t_idx]
            kps = det_keypoints[d_idx] if det_keypoints is not None else None
            self._update_track(track, det_boxes[d_idx], kps, det_confs[d_idx])

        # 4. Create new tracks for unmatched detections
        for d_idx in unmatched_dets:
            kps = det_keypoints[d_idx] if det_keypoints is not None else None
            self._create_track(det_boxes[d_idx], kps, det_confs[d_idx])

        # 5. Handle unmatched (lost) tracks – rely on prediction
        for t_idx in unmatched_tracks:
            track = self.tracks[t_idx]
            track.frames_since_detection += 1
            track.is_predicted = True
            track.prediction_confidence *= self.prediction_decay
            # Use predicted bbox from Kalman
            pred_center = track.predictor.get_state_position()
            if pred_center is not None and track.last_bbox_size is not None:
                w, h = track.last_bbox_size
                track.bbox = np.array([
                    pred_center[0] - w / 2,
                    pred_center[1] - h / 2,
                    pred_center[0] + w / 2,
                    pred_center[1] + h / 2,
                ])
            track.push_history()
            track.total_frames_tracked += 1

        # 6. Classify occlusion for every track + recovery bookkeeping
        for track in self.tracks:
            raw_state = self.occlusion_analyzer.analyze(
                keypoints=track.keypoints,
                bbox=track.bbox,
                detection_confidence=track.detection_conf,
                frames_since_detection=track.frames_since_detection,
            )

            # Mark was_occluded flag for future recovery detection
            if raw_state in (
                OcclusionState.PARTIALLY_OCCLUDED,
                OcclusionState.HEAVILY_OCCLUDED,
                OcclusionState.LOST,
            ):
                track.was_occluded = True

            # If currently in recovery window, override state
            if track.is_recovered and track.recovery_frames_remaining > 0:
                track.occlusion_state = OcclusionState.RECOVERED
                track.recovery_frames_remaining -= 1
                # Once countdown expires, transition to normal VISIBLE
                if track.recovery_frames_remaining <= 0:
                    track.is_recovered = False
                    track.was_occluded = False
                    track.occlusion_state = OcclusionState.VISIBLE
            else:
                track.occlusion_state = raw_state
                # If no longer occluded and not recovered, clear flag
                if raw_state == OcclusionState.VISIBLE:
                    track.was_occluded = False

        # 7. Remove stale tracks
        self._remove_stale_tracks()

        return list(self.tracks)

    # ── internal methods ─────────────────────────────────────────────

    def _predict_all(self) -> None:
        """Run Kalman predict step on every track."""
        for track in self.tracks:
            track.predictor.predict()

    def _match_detections(
        self, det_boxes: np.ndarray
    ) -> Tuple[list, list, list]:
        """
        Greedy IoU matching (highest IoU first).

        Returns (matched_pairs, unmatched_det_indices, unmatched_track_indices).
        """
        n_tracks = len(self.tracks)
        n_dets = len(det_boxes)

        if n_tracks == 0:
            return [], list(range(n_dets)), []
        if n_dets == 0:
            return [], [], list(range(n_tracks))

        # Build IoU cost matrix
        cost = np.zeros((n_tracks, n_dets), dtype=np.float32)
        for t in range(n_tracks):
            t_box = self.tracks[t].bbox
            for d in range(n_dets):
                cost[t, d] = iou(t_box, det_boxes[d])

        matched: list = []
        used_tracks: set = set()
        used_dets: set = set()

        # Greedy: pick highest IoU pair repeatedly
        while True:
            if cost.size == 0:
                break
            best = np.unravel_index(np.argmax(cost), cost.shape)
            best_iou = cost[best]
            if best_iou < self.iou_threshold:
                break
            t_idx, d_idx = int(best[0]), int(best[1])
            matched.append((t_idx, d_idx))
            used_tracks.add(t_idx)
            used_dets.add(d_idx)
            cost[t_idx, :] = -1
            cost[:, d_idx] = -1

        unmatched_dets = [d for d in range(n_dets) if d not in used_dets]
        unmatched_tracks = [t for t in range(n_tracks) if t not in used_tracks]
        return matched, unmatched_dets, unmatched_tracks

    def _update_track(
        self,
        track: PersonTracker,
        bbox: np.ndarray,
        keypoints: Optional[np.ndarray],
        conf: float,
    ) -> None:
        """Update an existing track with a matched detection."""
        # ── Recovery detection ───────────────────────────────────────
        # If the person was occluded and is now detected again → RECOVERED
        if track.was_occluded and track.frames_since_detection > 0:
            track.is_recovered = True
            track.recovery_frames_remaining = self.recovery_duration

        track.bbox = bbox.copy()
        track.keypoints = keypoints.copy() if keypoints is not None else None
        track.detection_conf = conf
        track.frames_since_detection = 0
        track.is_predicted = False
        track.prediction_confidence = 1.0
        track.last_seen_time = time.time()
        track.last_bbox_size = track.bbox_wh.copy()

        # Kalman update with measured center
        track.predictor.update(track.center)
        track.push_history()
        track.total_frames_tracked += 1

    def _create_track(
        self,
        bbox: np.ndarray,
        keypoints: Optional[np.ndarray],
        conf: float,
    ) -> PersonTracker:
        """Spawn a new track from an unmatched detection."""
        track = PersonTracker(
            track_id=self._next_id,
            bbox=bbox.copy(),
            keypoints=keypoints.copy() if keypoints is not None else None,
            detection_conf=conf,
        )
        track.last_bbox_size = track.bbox_wh.copy()
        self._next_id += 1

        # Initialise Kalman with first measured center
        track.predictor.init_state(track.center)
        track.push_history()
        track.total_frames_tracked = 1
        self.tracks.append(track)
        return track

    def _remove_stale_tracks(self) -> None:
        """Drop tracks that have been missing for too many frames."""
        self.tracks = [
            t for t in self.tracks
            if t.frames_since_detection <= self.max_frames_lost
        ]
