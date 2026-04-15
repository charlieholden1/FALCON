"""
Radar-camera fusion helpers for FALCON.

This module owns the single-person MVP handoff policy between camera tracking
and TI IWR6843 people-tracking radar tracks. It is intentionally independent of
the GUI so the state machine, synthetic pose behavior, event logging, and
calibration helpers can be tested without opening cameras or serial ports.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from occlusion import OcclusionState
from radar import CameraProjection, RadarFrame, RadarTrack


class FusionMode(str, Enum):
    """User-facing fusion state for a camera person track."""

    CAMERA_LOCKED = "CAMERA_LOCKED"
    CAMERA_DEGRADED = "CAMERA_DEGRADED"
    RADAR_ARMED = "RADAR_ARMED"
    RADAR_ONLY = "RADAR_ONLY"
    REACQUIRING = "REACQUIRING"
    CAMERA_RELOCKED = "CAMERA_RELOCKED"


@dataclass
class FusionConfig:
    """Thresholds for the single-person radar handoff MVP."""

    keypoint_conf_threshold: float = 0.35
    strong_keypoint_count: int = 9
    acceptable_keypoint_count: int = 5
    strong_detection_conf: float = 0.65
    acceptable_detection_conf: float = 0.45
    lost_frames_to_radar: int = 2
    radar_pixel_tolerance: float = 170.0
    radar_depth_tolerance_m: float = 0.9
    recapture_iou_threshold: float = 0.20
    recapture_center_tolerance_px: float = 150.0
    recapture_confirm_frames: int = 2
    synthetic_keypoint_conf: float = 0.38
    fallback_box_width_px: float = 110.0
    fallback_box_height_px: float = 240.0
    min_projected_box_width_px: float = 35.0
    min_projected_box_height_px: float = 90.0
    max_projected_box_scale: float = 2.2
    stale_radar_frame_s: float = 0.35
    radar_confidence_gain: float = 0.28
    radar_confidence_decay: float = 0.88


@dataclass
class FusionDebugSnapshot:
    """Small summary consumed by the GUI and tests."""

    mode_counts: Dict[str, int] = field(default_factory=dict)
    radar_track_count: int = 0
    linked_radar_ids: Dict[int, int] = field(default_factory=dict)
    last_event: str = ""
    event_log_path: str = ""

    def status_text(self) -> str:
        modes = ", ".join(
            f"{mode}:{count}" for mode, count in sorted(self.mode_counts.items())
        )
        links = ", ".join(
            f"P{person}->R{radar}" for person, radar in sorted(self.linked_radar_ids.items())
        )
        return f"{modes or 'idle'}" + (f" | {links}" if links else "")


@dataclass
class CalibrationSample:
    """One radar/camera correspondence for guided extrinsic calibration."""

    radar_xyz: List[float]
    image_uv: List[float]
    camera_track_id: int = -1
    radar_track_id: int = -1
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0
    source: str = "auto_pose"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "radar_xyz": [float(value) for value in self.radar_xyz],
            "image_uv": [float(value) for value in self.image_uv],
            "camera_track_id": int(self.camera_track_id),
            "radar_track_id": int(self.radar_track_id),
            "timestamp": float(self.timestamp),
            "confidence": float(self.confidence),
            "source": str(self.source),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CalibrationSample":
        return cls(
            radar_xyz=[float(value) for value in payload["radar_xyz"]],
            image_uv=[float(value) for value in payload["image_uv"]],
            camera_track_id=int(payload.get("camera_track_id", -1)),
            radar_track_id=int(payload.get("radar_track_id", -1)),
            timestamp=float(payload.get("timestamp", time.time())),
            confidence=float(payload.get("confidence", 1.0)),
            source=str(payload.get("source", "auto_pose")),
        )


@dataclass
class CalibrationSolveResult:
    """Outcome of a guided calibration solve."""

    ok: bool
    message: str
    params: Dict[str, float] = field(default_factory=dict)
    mean_error_px: float = 0.0
    median_error_px: float = 0.0
    sample_count: int = 0


class FusionEventLogger:
    """Low-overhead JSONL event logger for handoff/recapture diagnosis."""

    def __init__(self, path: Optional[Path | str] = None):
        self.path = Path(path) if path is not None else None
        self._handle: Optional[Any] = None
        self.last_event = ""
        if self.path is not None:
            self.open(self.path)

    def open(self, path: Path | str) -> None:
        self.close()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8", buffering=1)

    def close(self) -> None:
        if self._handle is None:
            return
        try:
            self._handle.flush()
            self._handle.close()
        finally:
            self._handle = None

    def write(self, event_type: str, payload: Dict[str, Any]) -> None:
        self.last_event = str(event_type)
        if self._handle is None:
            return
        record = {
            "type": str(event_type),
            "timestamp": time.time(),
            **_json_ready(payload),
        }
        self._handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


class SyntheticPoseGenerator:
    """Create radar-only COCO-17 pose estimates from the last real pose."""

    @staticmethod
    def generate(
        last_keypoints: Optional[np.ndarray],
        last_bbox: Optional[np.ndarray],
        target_bbox: np.ndarray,
        *,
        confidence: float = 0.38,
    ) -> np.ndarray:
        target = np.asarray(target_bbox, dtype=np.float64).reshape(4)
        if last_keypoints is not None and last_bbox is not None:
            src_bbox = np.asarray(last_bbox, dtype=np.float64).reshape(4)
            src_w = max(float(src_bbox[2] - src_bbox[0]), 1.0)
            src_h = max(float(src_bbox[3] - src_bbox[1]), 1.0)
            dst_w = max(float(target[2] - target[0]), 1.0)
            dst_h = max(float(target[3] - target[1]), 1.0)

            pose = np.asarray(last_keypoints, dtype=np.float64).copy()
            pose[:, 0] = target[0] + ((pose[:, 0] - src_bbox[0]) / src_w) * dst_w
            pose[:, 1] = target[1] + ((pose[:, 1] - src_bbox[1]) / src_h) * dst_h
            pose[:, 2] = np.where(pose[:, 2] > 0.0, confidence, 0.0)
            return pose.astype(np.float32)

        return SyntheticPoseGenerator._default_pose(target, confidence=confidence)

    @staticmethod
    def _default_pose(target_bbox: np.ndarray, *, confidence: float) -> np.ndarray:
        x1, y1, x2, y2 = [float(value) for value in target_bbox]
        w = max(x2 - x1, 1.0)
        h = max(y2 - y1, 1.0)
        cx = (x1 + x2) * 0.5
        points = [
            (cx, y1 + 0.08 * h),  # nose
            (cx - 0.10 * w, y1 + 0.06 * h),
            (cx + 0.10 * w, y1 + 0.06 * h),
            (cx - 0.18 * w, y1 + 0.10 * h),
            (cx + 0.18 * w, y1 + 0.10 * h),
            (cx - 0.30 * w, y1 + 0.28 * h),
            (cx + 0.30 * w, y1 + 0.28 * h),
            (cx - 0.38 * w, y1 + 0.48 * h),
            (cx + 0.38 * w, y1 + 0.48 * h),
            (cx - 0.35 * w, y1 + 0.66 * h),
            (cx + 0.35 * w, y1 + 0.66 * h),
            (cx - 0.22 * w, y1 + 0.58 * h),
            (cx + 0.22 * w, y1 + 0.58 * h),
            (cx - 0.24 * w, y1 + 0.78 * h),
            (cx + 0.24 * w, y1 + 0.78 * h),
            (cx - 0.22 * w, y2),
            (cx + 0.22 * w, y2),
        ]
        pose = np.zeros((17, 3), dtype=np.float32)
        for idx, (x, y) in enumerate(points):
            pose[idx] = [x, y, confidence]
        return pose


class RadarCameraFusionManager:
    """Single-person radar/camera handoff state machine."""

    def __init__(
        self,
        config: Optional[FusionConfig] = None,
        event_logger: Optional[FusionEventLogger] = None,
    ):
        self.config = config or FusionConfig()
        self.event_logger = event_logger or FusionEventLogger()
        self.enabled = True
        self._last_snapshot = FusionDebugSnapshot()

    @property
    def debug_snapshot(self) -> FusionDebugSnapshot:
        return self._last_snapshot

    def reset(self) -> None:
        self._last_snapshot = FusionDebugSnapshot(
            event_log_path=str(self.event_logger.path or "")
        )

    def update(
        self,
        tracks: Sequence[Any],
        radar_frame: Optional[RadarFrame],
        projection: Optional[CameraProjection],
        *,
        frame_shape: Optional[Tuple[int, int, int]] = None,
        gui_fps: float = 0.0,
        radar_fps: float = 0.0,
        now: Optional[float] = None,
    ) -> List[Any]:
        now = time.time() if now is None else float(now)
        frame_h, frame_w = _shape_hw(frame_shape)
        radar_tracks = _fresh_radar_tracks(radar_frame, now, self.config.stale_radar_frame_s)
        mode_counts: Dict[str, int] = {}
        linked: Dict[int, int] = {}

        for track in tracks:
            _ensure_fusion_attrs(track)
            previous_mode = getattr(track, "fusion_mode", FusionMode.CAMERA_LOCKED.value)

            if self._has_live_camera_detection(track):
                self._remember_real_pose(track)
                if getattr(track, "using_radar", False):
                    self._handle_recapture(track, radar_frame, projection, frame_w, frame_h, gui_fps, radar_fps)
                else:
                    mode = (
                        FusionMode.CAMERA_LOCKED
                        if self._camera_is_strong(track)
                        else FusionMode.CAMERA_DEGRADED
                    )
                    self._assign_mode(track, mode, radar_frame, gui_fps, radar_fps)
                    track.using_radar = False
                    track.radar_confidence *= self.config.radar_confidence_decay
            else:
                if self.enabled and projection is not None and radar_tracks:
                    match = self._select_radar_track(track, radar_tracks, projection)
                else:
                    match = None

                if match is not None:
                    should_takeover = (
                        getattr(track, "frames_since_detection", 0) >= self.config.lost_frames_to_radar
                        or getattr(track, "using_radar", False)
                        or getattr(track, "occlusion_state", None) in (
                            OcclusionState.HEAVILY_OCCLUDED,
                            OcclusionState.LOST,
                        )
                    )
                    if should_takeover:
                        self._apply_radar_only(
                            track,
                            match,
                            projection,
                            frame_w=frame_w,
                            frame_h=frame_h,
                            radar_frame=radar_frame,
                            gui_fps=gui_fps,
                            radar_fps=radar_fps,
                        )
                    else:
                        track.radar_track_id = int(match.track_id)
                        self._assign_mode(track, FusionMode.RADAR_ARMED, radar_frame, gui_fps, radar_fps)
                else:
                    if getattr(track, "using_radar", False):
                        track.radar_confidence *= self.config.radar_confidence_decay
                        if track.radar_confidence < 0.1:
                            track.using_radar = False
                            track.radar_track_id = None
                    self._assign_mode(track, FusionMode.CAMERA_DEGRADED, radar_frame, gui_fps, radar_fps)

            mode = str(getattr(track, "fusion_mode", FusionMode.CAMERA_LOCKED.value))
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
            radar_id = getattr(track, "radar_track_id", None)
            if radar_id is not None:
                linked[int(getattr(track, "track_id", -1))] = int(radar_id)

            if previous_mode != mode:
                setattr(track, "last_fusion_event", mode)

        self._last_snapshot = FusionDebugSnapshot(
            mode_counts=mode_counts,
            radar_track_count=len(radar_tracks),
            linked_radar_ids=linked,
            last_event=self.event_logger.last_event,
            event_log_path=str(self.event_logger.path or ""),
        )
        return list(tracks)

    def status_text(self) -> str:
        return self._last_snapshot.status_text()

    def _has_live_camera_detection(self, track: Any) -> bool:
        return (
            not bool(getattr(track, "is_predicted", False))
            and int(getattr(track, "frames_since_detection", 0)) == 0
        )

    def _camera_quality(self, track: Any) -> float:
        conf = float(getattr(track, "detection_conf", 0.0) or 0.0)
        keypoints = getattr(track, "keypoints", None)
        visible = _visible_keypoint_count(keypoints, self.config.keypoint_conf_threshold)
        return (0.55 * conf) + (0.45 * min(visible / 17.0, 1.0))

    def _camera_is_strong(self, track: Any) -> bool:
        conf = float(getattr(track, "detection_conf", 0.0) or 0.0)
        visible = _visible_keypoint_count(
            getattr(track, "keypoints", None),
            self.config.keypoint_conf_threshold,
        )
        return conf >= self.config.strong_detection_conf and visible >= self.config.strong_keypoint_count

    def _camera_is_acceptable(self, track: Any) -> bool:
        conf = float(getattr(track, "detection_conf", 0.0) or 0.0)
        visible = _visible_keypoint_count(
            getattr(track, "keypoints", None),
            self.config.keypoint_conf_threshold,
        )
        return conf >= self.config.acceptable_detection_conf and visible >= self.config.acceptable_keypoint_count

    def _remember_real_pose(self, track: Any) -> None:
        keypoints = getattr(track, "keypoints", None)
        bbox = getattr(track, "bbox", None)
        if keypoints is not None and bbox is not None and self._camera_is_acceptable(track):
            track.last_real_keypoints = np.asarray(keypoints, dtype=np.float32).copy()
            track.last_real_bbox = np.asarray(bbox, dtype=np.float32).copy()
            track.synthetic_pose = None

    def _select_radar_track(
        self,
        track: Any,
        radar_tracks: Sequence[RadarTrack],
        projection: CameraProjection,
    ) -> Optional[RadarTrack]:
        if not radar_tracks:
            return None

        linked_id = getattr(track, "radar_track_id", None)
        predicted_center = np.asarray(getattr(track, "center", _bbox_center(track.bbox)), dtype=np.float64)
        best: Optional[RadarTrack] = None
        best_score = float("inf")

        for radar_track in radar_tracks:
            u, v = projection.project_3d_to_2d(radar_track.position_3d)
            if u < 0 or v < 0:
                continue
            pixel_err = float(np.linalg.norm(np.array([u, v], dtype=np.float64) - predicted_center))
            depth_err = 0.0
            depth = getattr(track, "z_depth_meters", None)
            if depth is not None:
                depth_err = abs(float(radar_track.y) - float(depth))
                if depth_err > self.config.radar_depth_tolerance_m:
                    continue

            is_linked = linked_id is not None and int(linked_id) == int(radar_track.track_id)
            tolerance = self.config.radar_pixel_tolerance * (1.8 if is_linked else 1.0)
            if pixel_err > tolerance:
                continue

            score = pixel_err + (depth_err * 35.0)
            if is_linked:
                score -= self.config.radar_pixel_tolerance * 0.35
            score -= max(float(radar_track.confidence), 0.0) * 12.0
            if score < best_score:
                best = radar_track
                best_score = score

        return best

    def _apply_radar_only(
        self,
        track: Any,
        radar_track: RadarTrack,
        projection: CameraProjection,
        *,
        frame_w: int,
        frame_h: int,
        radar_frame: Optional[RadarFrame],
        gui_fps: float,
        radar_fps: float,
    ) -> None:
        old_bbox = np.asarray(getattr(track, "bbox"), dtype=np.float64).copy()
        target_bbox = project_radar_track_bbox(
            radar_track,
            projection,
            fallback_bbox=_first_array(getattr(track, "last_real_bbox", None), old_bbox),
            frame_w=frame_w,
            frame_h=frame_h,
            config=self.config,
        )
        if target_bbox is None:
            self._assign_mode(track, FusionMode.RADAR_ARMED, radar_frame, gui_fps, radar_fps)
            return

        track.bbox = target_bbox.astype(np.float32)
        track.last_bbox_size = np.array(
            [track.bbox[2] - track.bbox[0], track.bbox[3] - track.bbox[1]],
            dtype=np.float32,
        )
        track.keypoints = SyntheticPoseGenerator.generate(
            getattr(track, "last_real_keypoints", None),
            getattr(track, "last_real_bbox", None),
            track.bbox,
            confidence=self.config.synthetic_keypoint_conf,
        )
        track.synthetic_pose = track.keypoints.copy()
        track.using_radar = True
        track.radar_track_id = int(radar_track.track_id)
        track.radar_position_3d = radar_track.position_3d.copy()
        track.radar_velocity = float(np.linalg.norm(radar_track.velocity_3d))
        track.radar_confidence = min(
            1.0,
            max(float(getattr(track, "radar_confidence", 0.0)), 0.45)
            + self.config.radar_confidence_gain,
        )
        track.radar_frames_matched = int(getattr(track, "radar_frames_matched", 0)) + 1
        track.handoff_age_frames = int(getattr(track, "handoff_age_frames", 0)) + 1
        track.recapture_confirm_frames = 0

        measurement = _bbox_measurement(track.bbox)
        try:
            track.predictor.update(measurement)
        except Exception:
            pass

        if getattr(track, "bbox_history", None):
            track.bbox_history[-1] = track.bbox.copy()
        self._assign_mode(track, FusionMode.RADAR_ONLY, radar_frame, gui_fps, radar_fps)

    def _handle_recapture(
        self,
        track: Any,
        radar_frame: Optional[RadarFrame],
        projection: Optional[CameraProjection],
        frame_w: int,
        frame_h: int,
        gui_fps: float,
        radar_fps: float,
    ) -> None:
        strong = self._camera_is_strong(track)
        acceptable = self._camera_is_acceptable(track)
        aligned = False

        if projection is not None and radar_frame is not None:
            radar_track = _find_radar_track(radar_frame.tracks, getattr(track, "radar_track_id", None))
            if radar_track is not None:
                radar_bbox = project_radar_track_bbox(
                    radar_track,
                    projection,
                    fallback_bbox=_first_array(
                        getattr(track, "last_real_bbox", None),
                        getattr(track, "bbox", None),
                    ),
                    frame_w=frame_w,
                    frame_h=frame_h,
                    config=self.config,
                )
                if radar_bbox is not None:
                    aligned = (
                        _bbox_iou(np.asarray(track.bbox), radar_bbox) >= self.config.recapture_iou_threshold
                        or np.linalg.norm(_bbox_center(track.bbox) - _bbox_center(radar_bbox))
                        <= self.config.recapture_center_tolerance_px
                    )

        if strong or (acceptable and aligned):
            track.recapture_confirm_frames = int(getattr(track, "recapture_confirm_frames", 0)) + 1
        else:
            track.recapture_confirm_frames = 0

        if strong or track.recapture_confirm_frames >= self.config.recapture_confirm_frames:
            track.using_radar = False
            track.radar_track_id = None
            track.handoff_age_frames = 0
            track.radar_confidence = 0.0
            track.synthetic_pose = None
            self._assign_mode(track, FusionMode.CAMERA_RELOCKED, radar_frame, gui_fps, radar_fps)
        else:
            self._assign_mode(track, FusionMode.REACQUIRING, radar_frame, gui_fps, radar_fps)

    def _assign_mode(
        self,
        track: Any,
        mode: FusionMode,
        radar_frame: Optional[RadarFrame],
        gui_fps: float,
        radar_fps: float,
    ) -> None:
        old_mode = str(getattr(track, "fusion_mode", FusionMode.CAMERA_LOCKED.value))
        new_mode = mode.value
        track.fusion_mode = new_mode
        if old_mode == new_mode:
            return

        self.event_logger.write(
            new_mode.lower(),
            {
                "falcon_track_id": int(getattr(track, "track_id", -1)),
                "radar_track_id": getattr(track, "radar_track_id", None),
                "camera_bbox": _array_list(getattr(track, "bbox", None)),
                "fusion_confidence": float(getattr(track, "radar_confidence", 0.0)),
                "handoff_age_frames": int(getattr(track, "handoff_age_frames", 0)),
                "gui_fps": float(gui_fps),
                "radar_fps": float(radar_fps),
                "radar_frame_number": None
                if radar_frame is None
                else int(radar_frame.frame_number),
            },
        )


def project_radar_track_bbox(
    radar_track: RadarTrack,
    projection: CameraProjection,
    *,
    fallback_bbox: Optional[np.ndarray],
    frame_w: int = 0,
    frame_h: int = 0,
    config: Optional[FusionConfig] = None,
) -> Optional[np.ndarray]:
    config = config or FusionConfig()
    bbox_uv = None
    if radar_track.bbox is not None:
        projected = projection.project_box_3d(radar_track.bbox)
        if projected is not None and len(projected) == 4:
            bbox_uv = np.asarray(projected, dtype=np.float64)

    center_u, center_v = projection.project_3d_to_2d(radar_track.position_3d)
    if center_u < 0 or center_v < 0:
        return None

    fallback = None if fallback_bbox is None else np.asarray(fallback_bbox, dtype=np.float64).reshape(4)
    if bbox_uv is not None:
        bbox_uv = _sanitize_projected_bbox(
            bbox_uv,
            center=np.array([center_u, center_v], dtype=np.float64),
            fallback_bbox=fallback,
            config=config,
        )
    else:
        width = config.fallback_box_width_px
        height = config.fallback_box_height_px
        if fallback is not None:
            width = max(float(fallback[2] - fallback[0]), config.min_projected_box_width_px)
            height = max(float(fallback[3] - fallback[1]), config.min_projected_box_height_px)
        bbox_uv = np.array(
            [
                center_u - width * 0.5,
                center_v - height * 0.55,
                center_u + width * 0.5,
                center_v + height * 0.45,
            ],
            dtype=np.float64,
        )

    return _clamp_bbox(bbox_uv, frame_w=frame_w, frame_h=frame_h).astype(np.float32)


def capture_calibration_sample(
    camera_tracks: Sequence[Any],
    radar_frame: Optional[RadarFrame],
    *,
    image_uv_override: Optional[Sequence[float]] = None,
    allow_point_cloud_fallback: bool = False,
) -> CalibrationSample:
    camera_track = _best_camera_calibration_track(camera_tracks)
    if camera_track is None and image_uv_override is None:
        raise ValueError("No visible camera track is available for calibration.")
    if radar_frame is None:
        raise ValueError("No radar frame is available yet. Wait for the radar status to show frames/FPS, then try again.")

    radar_track = None
    radar_xyz = None
    radar_id = -1
    source_suffix = ""
    if radar_frame.tracks:
        radar_track = _best_radar_calibration_track(camera_track, radar_frame.tracks)
        radar_xyz = radar_track.position_3d
        radar_id = int(radar_track.track_id)
    elif not allow_point_cloud_fallback:
        point_count = len(getattr(radar_frame, "points", []) or [])
        presence = getattr(radar_frame, "presence", None)
        raise ValueError(
            "No radar person track is available for calibration "
            f"(frame {radar_frame.frame_number}, points={point_count}, presence={presence}). "
            "Move for a second or two until the radar status shows tracks=1, then capture. "
            "Point-cloud rough samples are available in Advanced if you need a coarse debug solve."
        )
    else:
        radar_xyz = _point_cloud_calibration_position(radar_frame)
        source_suffix = "_point_cloud"
        if radar_xyz is None:
            point_count = len(getattr(radar_frame, "points", []) or [])
            presence = getattr(radar_frame, "presence", None)
            raise ValueError(
                "No radar track is available for calibration "
                f"(frame {radar_frame.frame_number}, points={point_count}, presence={presence}). "
                "Use the people-tracking cfg/firmware, step into the scene to allocate a track, "
                "or capture again when point-cloud returns are visible."
            )

    if image_uv_override is not None:
        image_uv = np.asarray(image_uv_override, dtype=np.float64).reshape(2)
        source = "manual_click"
        visible = 0
        conf = max(
            0.85,
            float(getattr(camera_track, "detection_conf", 0.0) or 0.0)
            if camera_track is not None
            else 0.0,
        )
    else:
        assert camera_track is not None
        bbox = np.asarray(camera_track.bbox, dtype=np.float64)
        image_uv = _calibration_image_point(bbox, getattr(camera_track, "keypoints", None))
        source = "auto_pose"
        visible = _visible_keypoint_count(getattr(camera_track, "keypoints", None), 0.35)
        conf = float(getattr(camera_track, "detection_conf", 0.0) or 0.0)
    source += source_suffix
    return CalibrationSample(
        radar_xyz=[float(value) for value in radar_xyz],
        image_uv=[float(image_uv[0]), float(image_uv[1])],
        camera_track_id=int(getattr(camera_track, "track_id", -1)) if camera_track is not None else -1,
        radar_track_id=radar_id,
        confidence=max(conf, min(1.0, visible / 17.0)),
        source=source,
    )


def solve_calibration_samples(
    samples: Sequence[CalibrationSample],
    projection: CameraProjection,
    *,
    min_samples: int = 5,
    max_iterations: int = 25,
    max_median_error_px: float = 65.0,
) -> CalibrationSolveResult:
    if len(samples) < min_samples:
        return CalibrationSolveResult(
            ok=False,
            message=f"Need at least {min_samples} calibration samples; got {len(samples)}.",
            sample_count=len(samples),
        )

    base_params = projection.params
    keys = ["tx", "ty", "tz", "yaw_deg", "pitch_deg", "roll_deg"]
    x0 = np.array([base_params[key] for key in keys], dtype=np.float64)

    def build_projection(values: np.ndarray) -> CameraProjection:
        candidate = CameraProjection(K=projection.K.copy())
        params = dict(base_params)
        for key, value in zip(keys, values):
            params[key] = float(value)
        candidate.update(**params)
        return candidate

    def residuals(values: np.ndarray) -> np.ndarray:
        candidate = build_projection(values)
        out: List[float] = []
        for sample in samples:
            u, v = _project_float(candidate, np.asarray(sample.radar_xyz, dtype=np.float64))
            target = np.asarray(sample.image_uv, dtype=np.float64)
            if not np.isfinite(u) or not np.isfinite(v):
                out.extend([1000.0, 1000.0])
            else:
                weight = math.sqrt(max(float(sample.confidence), 0.05))
                out.extend([(float(u) - target[0]) * weight, (float(v) - target[1]) * weight])
        return np.asarray(out, dtype=np.float64)

    x = _coarse_calibration_start(x0, residuals)
    x = _refine_calibration_start(x, residuals, keys, max_iterations=max_iterations)

    if len(samples) >= min_samples + 2:
        initial_errors = _sample_pixel_errors(samples, build_projection(x))
        keep_indexes = _inlier_indexes(initial_errors, min_samples=min_samples)
        if len(keep_indexes) < len(samples):
            inlier_samples = [samples[index] for index in keep_indexes]

            def inlier_residuals(values: np.ndarray) -> np.ndarray:
                candidate = build_projection(values)
                out: List[float] = []
                for sample in inlier_samples:
                    u, v = _project_float(candidate, np.asarray(sample.radar_xyz, dtype=np.float64))
                    target = np.asarray(sample.image_uv, dtype=np.float64)
                    if not np.isfinite(u) or not np.isfinite(v):
                        out.extend([1000.0, 1000.0])
                    else:
                        weight = math.sqrt(max(float(sample.confidence), 0.05))
                        out.extend([(float(u) - target[0]) * weight, (float(v) - target[1]) * weight])
                return np.asarray(out, dtype=np.float64)

            x = _refine_calibration_start(x, inlier_residuals, keys, max_iterations=max_iterations)

    solved_projection = build_projection(x)
    pixel_errors = _sample_pixel_errors(samples, solved_projection)

    params = dict(base_params)
    for key, value in zip(keys, x):
        params[key] = float(value)
    mean_error = float(np.mean(pixel_errors)) if pixel_errors else 0.0
    median_error = float(np.median(pixel_errors)) if pixel_errors else 0.0
    near_bounds = [
        key
        for key in ("yaw_deg", "pitch_deg", "roll_deg")
        if abs(float(params.get(key, 0.0))) >= 88.0
    ]
    if median_error > max_median_error_px or near_bounds:
        reasons = []
        if median_error > max_median_error_px:
            reasons.append(
                f"median error {median_error:.1f}px is above {max_median_error_px:.0f}px"
            )
        if near_bounds:
            reasons.append(f"rotation hit limit ({', '.join(near_bounds)})")
        return CalibrationSolveResult(
            ok=False,
            message=(
                f"Rejected {len(samples)} samples: "
                + "; ".join(reasons)
                + ". Use radar-track samples only, move to more separated positions, then solve again."
            ),
            params=params,
            mean_error_px=mean_error,
            median_error_px=median_error,
            sample_count=len(samples),
        )
    return CalibrationSolveResult(
        ok=True,
        message=(
            f"Solved {len(samples)} samples: median error {median_error:.1f}px, "
            f"mean error {mean_error:.1f}px."
        ),
        params=params,
        mean_error_px=mean_error,
        median_error_px=median_error,
        sample_count=len(samples),
    )


def save_calibration_samples(samples: Sequence[CalibrationSample], path: Path | str) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [sample.to_dict() for sample in samples]
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def load_calibration_samples(path: Path | str) -> List[CalibrationSample]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [CalibrationSample.from_dict(item) for item in payload]


def _coarse_calibration_start(
    base: np.ndarray,
    residuals: Any,
) -> np.ndarray:
    """Find a reasonable pose seed before local optimization."""

    offsets = {
        0: [-0.30, 0.0, 0.30],          # tx
        1: [-0.20, 0.0, 0.20],          # ty
        2: [-0.20, 0.0, 0.20],          # tz
        3: [-25.0, -12.0, 0.0, 12.0, 25.0],  # yaw
        4: [-25.0, -12.0, 0.0, 12.0, 25.0],  # pitch
        5: [-12.0, 0.0, 12.0],         # roll
    }
    best = base.copy()
    best_score = float(residuals(best) @ residuals(best))
    for tx in offsets[0]:
        for ty in offsets[1]:
            for tz in offsets[2]:
                for yaw in offsets[3]:
                    for pitch in offsets[4]:
                        for roll in offsets[5]:
                            candidate = base + np.array([tx, ty, tz, yaw, pitch, roll], dtype=np.float64)
                            candidate = _clamp_calibration_values(candidate, [
                                "tx", "ty", "tz", "yaw_deg", "pitch_deg", "roll_deg"
                            ])
                            score = float(residuals(candidate) @ residuals(candidate))
                            if score < best_score:
                                best = candidate
                                best_score = score
    return best


def _refine_calibration_start(
    start: np.ndarray,
    residuals: Any,
    keys: Sequence[str],
    *,
    max_iterations: int,
) -> np.ndarray:
    x = start.copy()
    current = residuals(x)
    damping = 1e-2
    for _ in range(max_iterations):
        jac = np.zeros((current.size, x.size), dtype=np.float64)
        for col in range(x.size):
            step = 0.002 if keys[col].startswith("t") else 0.02
            shifted = x.copy()
            shifted[col] += step
            jac[:, col] = (residuals(shifted) - current) / step

        lhs = jac.T @ jac + damping * np.eye(x.size)
        rhs = -(jac.T @ current)
        try:
            delta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            break

        candidate_x = _clamp_calibration_values(x + delta, keys)
        candidate = residuals(candidate_x)
        if float(candidate @ candidate) < float(current @ current):
            x = candidate_x
            current = candidate
            damping = max(damping * 0.5, 1e-5)
            if float(np.linalg.norm(delta)) < 1e-5:
                break
        else:
            damping = min(damping * 5.0, 1e5)
    return x


def _clamp_calibration_values(values: np.ndarray, keys: Sequence[str]) -> np.ndarray:
    bounds = {
        "tx": (-2.0, 2.0),
        "ty": (-2.0, 2.0),
        "tz": (-2.0, 2.0),
        "yaw_deg": (-90.0, 90.0),
        "pitch_deg": (-90.0, 90.0),
        "roll_deg": (-90.0, 90.0),
    }
    out = np.asarray(values, dtype=np.float64).copy()
    for index, key in enumerate(keys):
        if key in bounds:
            low, high = bounds[key]
            out[index] = min(max(float(out[index]), low), high)
    return out


def _project_float(projection: CameraProjection, point_3d: np.ndarray) -> Tuple[float, float]:
    camera_point = projection.radar_to_camera(point_3d)
    if camera_point[2] <= 1e-6:
        return float("nan"), float("nan")
    projected = projection.K @ camera_point
    return float(projected[0] / projected[2]), float(projected[1] / projected[2])


def _sample_pixel_errors(
    samples: Sequence[CalibrationSample],
    projection: CameraProjection,
) -> List[float]:
    pixel_errors: List[float] = []
    for sample in samples:
        u, v = _project_float(projection, np.asarray(sample.radar_xyz, dtype=np.float64))
        if not np.isfinite(u) or not np.isfinite(v):
            pixel_errors.append(1000.0)
        else:
            pixel_errors.append(float(np.linalg.norm(np.array([u, v]) - np.asarray(sample.image_uv))))
    return pixel_errors


def _inlier_indexes(errors: Sequence[float], *, min_samples: int) -> List[int]:
    arr = np.asarray(errors, dtype=np.float64)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    threshold = median + max(40.0, 2.5 * mad)
    indexes = [index for index, error in enumerate(errors) if error <= threshold]
    if len(indexes) >= min_samples:
        return indexes
    ranked = sorted(range(len(errors)), key=lambda index: errors[index])
    return ranked[:min_samples]


def _fresh_radar_tracks(
    radar_frame: Optional[RadarFrame],
    now: float,
    stale_s: float,
) -> List[RadarTrack]:
    if radar_frame is None:
        return []
    timestamp = float(radar_frame.timestamp or radar_frame.source_timestamp or now)
    if now - timestamp > stale_s:
        return []
    return list(radar_frame.tracks)


def _ensure_fusion_attrs(track: Any) -> None:
    defaults = {
        "fusion_mode": FusionMode.CAMERA_LOCKED.value,
        "radar_track_id": None,
        "handoff_age_frames": 0,
        "recapture_confirm_frames": 0,
        "synthetic_pose": None,
        "last_real_keypoints": None,
        "last_real_bbox": None,
        "last_fusion_event": "",
    }
    for key, value in defaults.items():
        if not hasattr(track, key):
            setattr(track, key, value)


def _shape_hw(frame_shape: Optional[Tuple[int, ...]]) -> Tuple[int, int]:
    if frame_shape is None or len(frame_shape) < 2:
        return 0, 0
    return int(frame_shape[0]), int(frame_shape[1])


def _visible_keypoint_count(keypoints: Optional[np.ndarray], threshold: float) -> int:
    if keypoints is None:
        return 0
    arr = np.asarray(keypoints)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return 0
    return int(np.sum(arr[:, 2] >= threshold))


def _bbox_center(bbox: np.ndarray) -> np.ndarray:
    arr = np.asarray(bbox, dtype=np.float64).reshape(4)
    return np.array([(arr[0] + arr[2]) * 0.5, (arr[1] + arr[3]) * 0.5], dtype=np.float64)


def _bbox_measurement(bbox: np.ndarray) -> np.ndarray:
    arr = np.asarray(bbox, dtype=np.float64).reshape(4)
    width = max(float(arr[2] - arr[0]), 1.0)
    height = max(float(arr[3] - arr[1]), 1.0)
    return np.array([
        (arr[0] + arr[2]) * 0.5,
        (arr[1] + arr[3]) * 0.5,
        width * height,
        width / height,
    ], dtype=np.float64)


def _bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    a = np.asarray(box_a, dtype=np.float64).reshape(4)
    b = np.asarray(box_b, dtype=np.float64).reshape(4)
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return float(inter / max(area_a + area_b - inter, 1e-6))


def _sanitize_projected_bbox(
    bbox_uv: np.ndarray,
    *,
    center: np.ndarray,
    fallback_bbox: Optional[np.ndarray],
    config: FusionConfig,
) -> np.ndarray:
    x1, y1, x2, y2 = [float(value) for value in bbox_uv]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    if fallback_bbox is not None:
        fallback_w = max(float(fallback_bbox[2] - fallback_bbox[0]), config.min_projected_box_width_px)
        fallback_h = max(float(fallback_bbox[3] - fallback_bbox[1]), config.min_projected_box_height_px)
    else:
        fallback_w = config.fallback_box_width_px
        fallback_h = config.fallback_box_height_px

    too_small = width < config.min_projected_box_width_px or height < config.min_projected_box_height_px
    too_large = width > fallback_w * config.max_projected_box_scale or height > fallback_h * config.max_projected_box_scale
    if too_small or too_large:
        width = fallback_w
        height = fallback_h
        x1 = center[0] - width * 0.5
        x2 = center[0] + width * 0.5
        y1 = center[1] - height * 0.55
        y2 = center[1] + height * 0.45

    return np.array([x1, y1, x2, y2], dtype=np.float64)


def _clamp_bbox(bbox: np.ndarray, *, frame_w: int, frame_h: int) -> np.ndarray:
    out = np.asarray(bbox, dtype=np.float64).reshape(4).copy()
    if frame_w > 0:
        out[[0, 2]] = np.clip(out[[0, 2]], 0.0, float(frame_w - 1))
    if frame_h > 0:
        out[[1, 3]] = np.clip(out[[1, 3]], 0.0, float(frame_h - 1))
    if out[2] <= out[0]:
        out[2] = min(float(frame_w - 1) if frame_w > 0 else out[0] + 1.0, out[0] + 1.0)
    if out[3] <= out[1]:
        out[3] = min(float(frame_h - 1) if frame_h > 0 else out[1] + 1.0, out[1] + 1.0)
    return out


def _find_radar_track(
    radar_tracks: Sequence[RadarTrack],
    radar_track_id: Optional[int],
) -> Optional[RadarTrack]:
    if radar_track_id is None:
        return None
    for track in radar_tracks:
        if int(track.track_id) == int(radar_track_id):
            return track
    return None


def _point_cloud_calibration_position(radar_frame: RadarFrame) -> Optional[np.ndarray]:
    points = []
    for point in getattr(radar_frame, "points", []) or []:
        xyz = np.asarray(point.position_3d, dtype=np.float64)
        if not np.all(np.isfinite(xyz)):
            continue
        if abs(float(xyz[0])) > 5.0 or not (0.25 <= float(xyz[1]) <= 9.0) or not (-1.0 <= float(xyz[2]) <= 3.5):
            continue
        snr = max(float(getattr(point, "snr", 0.0) or 0.0), 0.0)
        points.append((xyz, snr))

    if len(points) < 3:
        return None

    arr = np.asarray([item[0] for item in points], dtype=np.float64)
    snr = np.asarray([item[1] for item in points], dtype=np.float64)
    if np.any(snr > 0.0):
        keep_count = max(3, min(len(points), int(round(len(points) * 0.6))))
        keep = np.argsort(snr)[-keep_count:]
        arr = arr[keep]

    return np.median(arr, axis=0)


def _best_camera_calibration_track(camera_tracks: Sequence[Any]) -> Optional[Any]:
    candidates = [
        track for track in camera_tracks
        if not bool(getattr(track, "is_predicted", False))
        and int(getattr(track, "frames_since_detection", 0)) == 0
        and getattr(track, "bbox", None) is not None
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda track: (
            float(getattr(track, "detection_conf", 0.0) or 0.0),
            _visible_keypoint_count(getattr(track, "keypoints", None), 0.35),
        ),
    )


def _best_radar_calibration_track(camera_track: Any, radar_tracks: Sequence[RadarTrack]) -> RadarTrack:
    linked_id = getattr(camera_track, "radar_track_id", None)
    linked = _find_radar_track(radar_tracks, linked_id)
    if linked is not None:
        return linked
    return max(radar_tracks, key=lambda track: float(track.confidence))


def _calibration_image_point(bbox: np.ndarray, keypoints: Optional[np.ndarray]) -> np.ndarray:
    if keypoints is not None:
        arr = np.asarray(keypoints, dtype=np.float64)
        torso_indices = [5, 6, 11, 12]
        valid = [
            arr[idx, :2]
            for idx in torso_indices
            if idx < len(arr) and arr[idx, 2] > 0.35
        ]
        if valid:
            return np.mean(np.asarray(valid, dtype=np.float64), axis=0)
    box = np.asarray(bbox, dtype=np.float64).reshape(4)
    return np.array([(box[0] + box[2]) * 0.5, box[1] + (box[3] - box[1]) * 0.45])


def _array_list(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    return [float(item) for item in arr]


def _first_array(*values: Any) -> Optional[np.ndarray]:
    for value in values:
        if value is not None:
            return np.asarray(value)
    return None


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _array_list(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value
