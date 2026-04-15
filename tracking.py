"""
F.A.L.C.O.N. – Appearance-Aware Person Tracking Module
========================================================

Maintains persistent person IDs across video frames using a hybrid
IoU + colour-histogram matching score.  Each tracked person carries a
short history of bounding boxes / keypoints, an HSV colour histogram for
re-identification, a relative skeleton for ghost-pose reconstruction,
an occlusion state, and a 7D Kalman-filter predictor.

Key classes
-----------
PersonTracker   – per-person state (dataclass)
TrackingManager – orchestrates matching, creation, and removal of tracks
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from occlusion import OcclusionAnalyzer, OcclusionState
from prediction import KalmanPredictor
from radar import CameraProjection, RadarPoint


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


def _compute_histogram(frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
    """Histogram computation disabled for Jetson CPU performance."""
    return None


def _compare_histograms(
    histA: Optional[np.ndarray],
    histB: Optional[np.ndarray],
) -> float:
    """Return histogram correlation in [0, 1].  0 if either is None."""
    if histA is None or histB is None:
        return 0.0
    score = cv2.compareHist(histA, histB, cv2.HISTCMP_CORREL)
    return max(0.0, score)


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

    # Appearance descriptor (HSV colour histogram)
    histogram: Optional[np.ndarray] = None

    # Ghost skeleton: relative keypoint offsets from bbox centre
    relative_skeleton: Optional[np.ndarray] = None  # (17, 3) offsets

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
    recovery_frames_remaining: int = 0
    was_occluded: bool = False

    # Kalman innovation (prediction-vs-measurement error)
    tracking_error: float = 0.0

    # Real depth from RealSense (metres), None when unavailable
    z_depth_meters: Optional[float] = None

    # Radar fusion state
    radar_position_3d: Optional[np.ndarray] = None   # [X, Y, Z] metres in radar frame
    radar_velocity: Optional[float] = None            # radial velocity m/s
    radar_confidence: float = 0.0                     # 0-1, decays when no radar match
    using_radar: bool = False                         # True when position driven by radar
    radar_frames_matched: int = 0                     # consecutive frames with radar match
    last_radar_match_frame: int = 0                   # frame counter at last radar match
    radar_track_id: Optional[int] = None               # linked TI radar track ID
    fusion_mode: str = "CAMERA_LOCKED"                # high-level handoff state
    handoff_age_frames: int = 0                       # frames spent in radar handoff
    recapture_confirm_frames: int = 0                 # stable camera frames after radar-only
    synthetic_pose: Optional[np.ndarray] = None        # radar-only generated COCO-17 pose
    last_real_keypoints: Optional[np.ndarray] = None   # latest camera-owned pose
    last_real_bbox: Optional[np.ndarray] = None        # latest camera-owned bbox
    last_fusion_event: str = ""

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

    def update_relative_skeleton(self) -> None:
        """Cache keypoint offsets relative to the bbox centre."""
        if self.keypoints is None:
            return
        cx, cy = self.center
        rel = self.keypoints.copy()
        rel[:, 0] -= cx
        rel[:, 1] -= cy
        # confidence column is kept as-is
        self.relative_skeleton = rel

    def reconstruct_ghost_keypoints(self) -> Optional[np.ndarray]:
        """Reconstruct keypoints from Kalman-predicted centre + stored offsets."""
        if self.relative_skeleton is None:
            return None
        pred_center = self.predictor.get_state_position()
        if pred_center is None:
            pred_center = self.center
        ghost = self.relative_skeleton.copy()
        ghost[:, 0] += pred_center[0]
        ghost[:, 1] += pred_center[1]
        return ghost

    def radar_position_2d(
        self, projection: CameraProjection,
    ) -> Optional[Tuple[int, int]]:
        """Project stored radar_position_3d to pixel coords.

        Returns ``None`` if no radar data is available.
        """
        if self.radar_position_3d is None:
            return None
        u, v = projection.project_3d_to_2d(self.radar_position_3d)
        if u == -1 and v == -1:
            return None
        return (u, v)


# ── TrackingManager ─────────────────────────────────────────────────

class TrackingManager:
    """
    Frame-level tracker that associates new detections with existing
    tracks via a weighted IoU + histogram-correlation score, manages
    track lifecycles, and delegates occlusion classification.

    Parameters
    ----------
    iou_threshold : float
        Minimum combined score to associate a detection with a track.
    max_frames_lost : int
        Frames without detection before a track is removed.
    prediction_decay : float
        Multiplicative decay applied to prediction_confidence each missed frame.
    iou_weight : float
        Weight for IoU component in the cost matrix (default 0.5).
    hist_weight : float
        Weight for histogram correlation component (default 0.5).
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_frames_lost: int = 20,
        prediction_decay: float = 0.90,
        recovery_duration: int = 20,
        iou_weight: float = 1.0,
        hist_weight: float = 0.0,
        radar: object = None,
        projection: Optional[CameraProjection] = None,
        radar_depth_tolerance: float = 0.8,
        radar_pixel_tolerance: float = 120.0,
        radar_handoff_state: OcclusionState = OcclusionState.HEAVILY_OCCLUDED,
        radar_confidence_decay: float = 0.85,
        radar_max_frames: int = 150,
    ):
        self.iou_threshold = iou_threshold
        self.max_frames_lost = max_frames_lost
        self.prediction_decay = prediction_decay
        self.recovery_duration = recovery_duration
        self.iou_weight = iou_weight
        self.hist_weight = hist_weight

        # Radar fusion
        self.radar = radar
        self.projection = projection
        self.radar_depth_tolerance = radar_depth_tolerance
        self.radar_pixel_tolerance = radar_pixel_tolerance
        self.radar_handoff_state = radar_handoff_state
        self.radar_confidence_decay = radar_confidence_decay
        self.radar_max_frames = radar_max_frames
        self._radar_point_count = 0  # latest frame's point count

        self._next_id: int = 1
        self.tracks: List[PersonTracker] = []
        self.occlusion_analyzer = OcclusionAnalyzer()
        self._depth_poll_counter: int = 0
        self._global_frame_counter: int = 0

    @property
    def max_tracking_error(self) -> float:
        """Return the largest tracking_error across all active tracks."""
        if not self.tracks:
            return 0.0
        return max(t.tracking_error for t in self.tracks)

    # ── public API ───────────────────────────────────────────────────

    def update(
        self,
        det_boxes: np.ndarray,
        det_keypoints: Optional[np.ndarray] = None,
        det_confs: Optional[np.ndarray] = None,
        frame: Optional[np.ndarray] = None,
        depth_frame: Optional[np.ndarray] = None,
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
        frame : ndarray | None
            Raw BGR frame used to compute appearance histograms.
        depth_frame : ndarray | None
            RealSense depth frame (uint16, millimetres).  ``None`` when
            running with a standard webcam.

        Returns
        -------
        list[PersonTracker]
            All active tracks (both detected and predicted).
        """
        if det_boxes is None or len(det_boxes) == 0:
            det_boxes = np.empty((0, 4))
        if det_confs is None:
            det_confs = np.ones(len(det_boxes))

        self._global_frame_counter += 1
        self._depth_poll_counter += 1
        _poll_depth = (self._depth_poll_counter % 3 == 0)

        # Pre-compute histograms for each detection
        det_hists: List[Optional[np.ndarray]] = []
        for d_idx in range(len(det_boxes)):
            if frame is not None:
                det_hists.append(_compute_histogram(frame, det_boxes[d_idx]))
            else:
                det_hists.append(None)

        # 1. Predict step for every existing track
        self._predict_all()

        # 2. Match detections ↔ tracks
        matched, unmatched_dets, unmatched_tracks = self._match_detections(
            det_boxes, det_hists,
        )

        # 3. Update matched tracks
        for t_idx, d_idx in matched:
            track = self.tracks[t_idx]
            kps = det_keypoints[d_idx] if det_keypoints is not None else None
            depth = self._query_depth(depth_frame, det_boxes[d_idx]) if _poll_depth else track.z_depth_meters
            self._update_track(track, det_boxes[d_idx], kps, det_confs[d_idx],
                               det_hists[d_idx], depth)

        # 4. Create new tracks for unmatched detections
        for d_idx in unmatched_dets:
            kps = det_keypoints[d_idx] if det_keypoints is not None else None
            depth = self._query_depth(depth_frame, det_boxes[d_idx]) if _poll_depth else None
            self._create_track(det_boxes[d_idx], kps, det_confs[d_idx],
                               det_hists[d_idx], depth)

        # 5. Handle unmatched (lost) tracks – rely on prediction
        for t_idx in unmatched_tracks:
            track = self.tracks[t_idx]
            track.frames_since_detection += 1
            track.is_predicted = True
            track.prediction_confidence *= self.prediction_decay

            # Use 7D Kalman predicted bbox (dynamic scale)
            pred_bbox = track.predictor.get_state_bbox()
            if pred_bbox is not None:
                track.bbox = pred_bbox
            else:
                pred_center = track.predictor.get_state_position()
                if pred_center is not None and track.last_bbox_size is not None:
                    w, h = track.last_bbox_size
                    track.bbox = np.array([
                        pred_center[0] - w / 2,
                        pred_center[1] - h / 2,
                        pred_center[0] + w / 2,
                        pred_center[1] + h / 2,
                    ])

            # Reconstruct ghost keypoints for occluded tracks
            ghost_kp = track.reconstruct_ghost_keypoints()
            if ghost_kp is not None:
                track.keypoints = ghost_kp

            track.push_history()
            track.total_frames_tracked += 1

        # 6. Classify occlusion for every track + recovery bookkeeping
        for track in self.tracks:
            # Query live depth at the track's current (possibly predicted) bbox
            current_depth = self._query_depth(depth_frame, track.bbox) if _poll_depth else track.z_depth_meters

            raw_state = self.occlusion_analyzer.analyze(
                keypoints=track.keypoints if not track.is_predicted else None,
                bbox=track.bbox,
                detection_confidence=track.detection_conf,
                frames_since_detection=track.frames_since_detection,
                current_depth=current_depth,
                predicted_depth=track.z_depth_meters,
            )

            if raw_state in (
                OcclusionState.PARTIALLY_OCCLUDED,
                OcclusionState.HEAVILY_OCCLUDED,
                OcclusionState.LOST,
            ):
                track.was_occluded = True

            if track.is_recovered and track.recovery_frames_remaining > 0:
                track.occlusion_state = OcclusionState.RECOVERED
                track.recovery_frames_remaining -= 1
                if track.recovery_frames_remaining <= 0:
                    track.is_recovered = False
                    track.was_occluded = False
                    track.occlusion_state = OcclusionState.VISIBLE
            else:
                track.occlusion_state = raw_state
                if raw_state == OcclusionState.VISIBLE:
                    track.was_occluded = False

        # 6b. Radar fusion – hand off occluded tracks to radar
        self._fuse_radar()

        # 7. Remove stale tracks
        self._remove_stale_tracks()

        return list(self.tracks)

    # ── internal methods ─────────────────────────────────────────────

    def _predict_all(self) -> None:
        """Run Kalman predict step on every track."""
        for track in self.tracks:
            track.predictor.predict()

    def _match_detections(
        self,
        det_boxes: np.ndarray,
        det_hists: List[Optional[np.ndarray]],
    ) -> Tuple[list, list, list]:
        """
        Greedy matching using weighted IoU + histogram correlation.

        Returns (matched_pairs, unmatched_det_indices, unmatched_track_indices).
        """
        n_tracks = len(self.tracks)
        n_dets = len(det_boxes)

        if n_tracks == 0:
            return [], list(range(n_dets)), []
        if n_dets == 0:
            return [], [], list(range(n_tracks))

        # Build combined cost matrix
        cost = np.zeros((n_tracks, n_dets), dtype=np.float32)
        for t in range(n_tracks):
            t_box = self.tracks[t].bbox
            t_hist = self.tracks[t].histogram
            for d in range(n_dets):
                iou_score = iou(t_box, det_boxes[d])
                hist_score = _compare_histograms(t_hist, det_hists[d])
                cost[t, d] = (self.iou_weight * iou_score
                              + self.hist_weight * hist_score)

        matched: list = []
        used_tracks: set = set()
        used_dets: set = set()

        # Greedy: pick highest combined score repeatedly
        while True:
            if cost.size == 0:
                break
            best = np.unravel_index(np.argmax(cost), cost.shape)
            best_score = cost[best]
            if best_score < self.iou_threshold:
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

    @staticmethod
    def _query_depth(
        depth_frame: Optional[np.ndarray],
        bbox: np.ndarray,
    ) -> Optional[float]:
        """Return median depth in metres from a central ROI of *bbox*.

        Samples a 10 %× 10 % patch around the bbox centre and takes the
        median of all non-zero pixels, making the reading robust to the
        zero-valued holes that RealSense depth frames typically contain.
        """
        if depth_frame is None:
            return None
        frame_h, frame_w = depth_frame.shape[:2]

        bx1, by1, bx2, by2 = bbox[:4]
        bw = bx2 - bx1
        bh = by2 - by1

        # 10 % of bbox dimensions, at least 2 px each side
        half_rw = max(int(bw * 0.05), 2)
        half_rh = max(int(bh * 0.05), 2)

        cx = int((bx1 + bx2) / 2)
        cy = int((by1 + by2) / 2)

        r_x1 = max(0, cx - half_rw)
        r_y1 = max(0, cy - half_rh)
        r_x2 = min(frame_w, cx + half_rw)
        r_y2 = min(frame_h, cy + half_rh)

        patch = depth_frame[r_y1:r_y2, r_x1:r_x2]
        valid = patch[patch > 0]
        if len(valid) == 0:
            return None
        return float(np.median(valid)) / 1000.0

    def _update_track(
        self,
        track: PersonTracker,
        bbox: np.ndarray,
        keypoints: Optional[np.ndarray],
        conf: float,
        histogram: Optional[np.ndarray] = None,
        depth: Optional[float] = None,
    ) -> None:
        """Update an existing track with a matched detection."""
        # ── Recovery detection ───────────────────────────────────────
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

        # Update appearance histogram (EMA blend for stability)
        if histogram is not None:
            if track.histogram is not None:
                track.histogram = 0.7 * track.histogram + 0.3 * histogram
                cv2.normalize(track.histogram, track.histogram)
            else:
                track.histogram = histogram

        # Update relative skeleton offsets
        track.update_relative_skeleton()
        if track.keypoints is not None:
            track.last_real_keypoints = track.keypoints.copy()
            track.last_real_bbox = track.bbox.copy()
            track.synthetic_pose = None

        # Store depth reading
        track.z_depth_meters = depth

        # Kalman update with full measurement [cx, cy, s, r]
        w, h = track.bbox_wh
        s = float(w * h)           # area (scale)
        r = float(w / h) if h > 0 else 1.0   # aspect ratio
        measurement = np.array([track.center[0], track.center[1], s, r])

        # Compute innovation (prediction error) before correction
        pred_pos = track.predictor.get_state_position()
        if pred_pos is not None:
            track.tracking_error = float(np.linalg.norm(
                measurement[:2] - pred_pos
            ))
        else:
            track.tracking_error = 0.0

        track.predictor.update(measurement)

        track.push_history()
        track.total_frames_tracked += 1

    def _create_track(
        self,
        bbox: np.ndarray,
        keypoints: Optional[np.ndarray],
        conf: float,
        histogram: Optional[np.ndarray] = None,
        depth: Optional[float] = None,
    ) -> PersonTracker:
        """Spawn a new track from an unmatched detection."""
        track = PersonTracker(
            track_id=self._next_id,
            bbox=bbox.copy(),
            keypoints=keypoints.copy() if keypoints is not None else None,
            detection_conf=conf,
            histogram=histogram,
            z_depth_meters=depth,
        )
        track.last_bbox_size = track.bbox_wh.copy()
        if track.keypoints is not None:
            track.last_real_keypoints = track.keypoints.copy()
            track.last_real_bbox = track.bbox.copy()
        self._next_id += 1

        # Cache relative skeleton
        track.update_relative_skeleton()

        # Initialise 7D Kalman with first measurement
        w, h = track.bbox_wh
        s = float(w * h)
        r = float(w / h) if h > 0 else 1.0
        track.predictor.init_state(track.center, scale=s, aspect=r)

        track.push_history()
        track.total_frames_tracked = 1
        self.tracks.append(track)
        return track

    # ── radar fusion ────────────────────────────────────────────────

    def _fuse_radar(self) -> None:
        """Match occluded tracks to radar point cloud and hand off position."""
        if self.radar is None or not self.radar.is_connected():
            self._radar_point_count = 0
            return

        radar_points: List[RadarPoint] = self.radar.get_point_cloud()
        self._radar_point_count = len(radar_points)

        if not radar_points:
            # Decay all radar-using tracks
            for track in self.tracks:
                if track.using_radar:
                    track.radar_confidence *= self.radar_confidence_decay
                    if track.radar_confidence < 0.1:
                        track.using_radar = False
            return

        # Track which radar points have already been claimed
        used_radar: set = set()

        # Pass 1: associate occluded/lost tracks with radar points
        for track in self.tracks:
            if track.occlusion_state not in (
                OcclusionState.HEAVILY_OCCLUDED,
                OcclusionState.LOST,
            ) and track.occlusion_state != self.radar_handoff_state:
                continue

            best_idx = -1
            best_score = float("inf")

            # Estimated 3D position for this track
            est_depth = track.z_depth_meters
            pred_center = track.predictor.get_state_position()
            if pred_center is None:
                pred_center = track.center

            for r_idx, rp in enumerate(radar_points):
                if r_idx in used_radar:
                    continue

                # Depth check
                if est_depth is not None:
                    depth_err = abs(rp.y - est_depth)
                    if depth_err > self.radar_depth_tolerance:
                        continue
                else:
                    depth_err = 0.0

                # 2D pixel proximity check
                pixel_err = 0.0
                if self.projection is not None:
                    u, v = self.projection.project_3d_to_2d(rp.position_3d)
                    if u != -1 and v != -1:
                        pixel_err = float(np.linalg.norm(
                            np.array([u, v]) - pred_center
                        ))
                        if pixel_err > self.radar_pixel_tolerance:
                            continue
                    else:
                        continue  # behind camera

                score = depth_err + pixel_err * 0.01
                if score < best_score:
                    best_score = score
                    best_idx = r_idx

            if best_idx >= 0:
                rp = radar_points[best_idx]
                used_radar.add(best_idx)

                track.radar_position_3d = rp.position_3d.copy()
                track.radar_velocity = rp.velocity
                track.radar_confidence = min(1.0, track.radar_confidence + 0.3)
                track.using_radar = True
                track.radar_frames_matched += 1
                track.last_radar_match_frame = self._global_frame_counter

                # Project radar position to 2D and feed Kalman as measurement
                if self.projection is not None:
                    u, v = self.projection.project_3d_to_2d(rp.position_3d)
                    if u != -1 and v != -1:
                        track.predictor.update(
                            np.array([float(u), float(v)])
                        )
                        # Update bbox from new Kalman state
                        pred_bbox = track.predictor.get_state_bbox()
                        if pred_bbox is not None:
                            track.bbox = pred_bbox
                        elif track.last_bbox_size is not None:
                            w, h = track.last_bbox_size
                            track.bbox = np.array([
                                u - w / 2, v - h / 2,
                                u + w / 2, v + h / 2,
                            ])
            else:
                # No radar match for this occluded track
                if track.using_radar:
                    track.radar_confidence *= self.radar_confidence_decay
                    if track.radar_confidence < 0.1:
                        track.using_radar = False
                        track.radar_frames_matched = 0

        # Pass 2: visible/partial tracks that were using radar — release them
        for track in self.tracks:
            if track.occlusion_state in (
                OcclusionState.VISIBLE,
                OcclusionState.PARTIALLY_OCCLUDED,
                OcclusionState.RECOVERED,
            ) and track.using_radar:
                track.using_radar = False
                track.radar_confidence *= 0.95

    @property
    def radar_fused_count(self) -> int:
        """Number of tracks currently using radar."""
        return sum(1 for t in self.tracks if t.using_radar)

    @property
    def radar_point_count(self) -> int:
        """Number of points in the latest radar frame."""
        return self._radar_point_count

    def _remove_stale_tracks(self) -> None:
        """Drop tracks that have been missing for too many frames.

        Radar-tracked targets are kept alive longer (up to
        ``radar_max_frames``) as long as radar confidence remains.
        """
        kept: List[PersonTracker] = []
        for t in self.tracks:
            if t.frames_since_detection <= self.max_frames_lost:
                kept.append(t)
            elif (t.using_radar and t.radar_confidence > 0.3
                  and t.frames_since_detection <= self.radar_max_frames):
                kept.append(t)
        self.tracks = kept

    # ── lightweight predict-only pass (for frame-skipping) ───────────

    def propagate_only(self) -> List[PersonTracker]:
        """Advance all tracks by one Kalman predict step without detection.

        Used on skipped frames to keep bounding boxes and ghost skeletons
        up-to-date while avoiding the cost of running the detector.
        """
        for track in self.tracks:
            track.predictor.predict()

            # Update bbox from Kalman state
            pred_bbox = track.predictor.get_state_bbox()
            if pred_bbox is not None:
                track.bbox = pred_bbox
            elif track.last_bbox_size is not None:
                pred_center = track.predictor.get_state_position()
                if pred_center is not None:
                    w, h = track.last_bbox_size
                    track.bbox = np.array([
                        pred_center[0] - w / 2,
                        pred_center[1] - h / 2,
                        pred_center[0] + w / 2,
                        pred_center[1] + h / 2,
                    ])

            # Reconstruct ghost keypoints so the skeleton stays current
            ghost_kp = track.reconstruct_ghost_keypoints()
            if ghost_kp is not None:
                track.keypoints = ghost_kp

            track.push_history()
            track.total_frames_tracked += 1

        return list(self.tracks)
