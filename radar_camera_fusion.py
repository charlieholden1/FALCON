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
from dataclasses import dataclass, field, replace
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

    single_person_mode: bool = True
    keypoint_conf_threshold: float = 0.35
    strong_keypoint_count: int = 9
    acceptable_keypoint_count: int = 5
    torso_keypoint_conf_threshold: float = 0.45
    min_visible_torso_keypoints: int = 2
    strong_detection_conf: float = 0.65
    acceptable_detection_conf: float = 0.45
    degraded_frames_to_arm: int = 2
    blocked_frames_to_radar: int = 2
    lost_frames_to_radar: int = 4
    radar_pixel_tolerance: float = 125.0
    radar_depth_tolerance_m: float = 0.9
    min_radar_confidence: float = 0.45
    min_radar_track_age_frames: int = 4
    min_radar_link_stable_frames: int = 2
    max_unlinked_radar_candidates: int = 1
    duplicate_track_distance_m: float = 0.75
    duplicate_track_distance_px: float = 110.0
    recapture_iou_threshold: float = 0.20
    recapture_center_tolerance_px: float = 120.0
    recapture_confirm_frames: int = 2
    synthetic_keypoint_conf: float = 0.38
    fallback_box_width_px: float = 110.0
    fallback_box_height_px: float = 240.0
    min_projected_box_width_px: float = 35.0
    min_projected_box_height_px: float = 90.0
    max_projected_box_scale: float = 2.2
    radar_bbox_blend: float = 0.65
    stale_radar_frame_s: float = 0.35
    radar_confidence_gain: float = 0.28
    radar_confidence_decay: float = 0.88


@dataclass
class FusionDebugSnapshot:
    """Small summary consumed by the GUI and tests."""

    mode_counts: Dict[str, int] = field(default_factory=dict)
    radar_track_count: int = 0
    radar_track_count_raw: int = 0
    duplicate_tracks_suppressed: int = 0
    ambiguous_radar_frames: int = 0
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
        radar_text = f"radar:{self.radar_track_count}/{self.radar_track_count_raw}"
        if self.duplicate_tracks_suppressed:
            radar_text += f" dedup:{self.duplicate_tracks_suppressed}"
        if self.ambiguous_radar_frames:
            radar_text += " ambiguous"
        suffix = f" | {radar_text}"
        if links:
            suffix += f" | {links}"
        return f"{modes or 'idle'}{suffix}"


MOUNT_GEOMETRY_PATH = Path("mount_geometry.json")


@dataclass
class MountGeometryPrior:
    """Physical camera-over-radar offset measured with a ruler.

    All values are in metres, expressed in the extrinsics convention used by
    CameraProjection (P_camera = R @ P_radar + t):
      tx_m  = lateral offset   (positive = camera to the right of radar)
      ty_m  = vertical offset  (positive = camera above radar)
      tz_m  = depth offset     (positive = camera behind radar / further from scene)
    Angles are deviations from perfect axis alignment.

    Current mount: camera 2.5 in (0.064 m) above, 1.5 in (0.038 m) behind radar.
    """

    tx_m: float = 0.0
    ty_m: float = 0.064       # 2.5 in above
    tz_m: float = 0.038       # 1.5 in behind
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    tx_tolerance_m: float = 0.10
    ty_tolerance_m: float = 0.02
    tz_tolerance_m: float = 0.02

    def to_dict(self) -> Dict[str, float]:
        return {
            "tx_m": self.tx_m, "ty_m": self.ty_m, "tz_m": self.tz_m,
            "yaw_deg": self.yaw_deg, "pitch_deg": self.pitch_deg, "roll_deg": self.roll_deg,
            "tx_tolerance_m": self.tx_tolerance_m,
            "ty_tolerance_m": self.ty_tolerance_m,
            "tz_tolerance_m": self.tz_tolerance_m,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MountGeometryPrior":
        fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: float(v) for k, v in d.items() if k in fields})

    def save(self, path: Path = MOUNT_GEOMETRY_PATH) -> Path:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path = MOUNT_GEOMETRY_PATH) -> "MountGeometryPrior":
        if not path.exists():
            return cls()
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return cls()


@dataclass
class CalibrationSample:
    """One radar/camera correspondence for guided extrinsic calibration.

    ``camera_xyz`` and ``depth_m`` are optional 3-D enrichments populated
    when a RealSense depth stream is available. They let the solver run a
    closed-form 3-D ↔ 3-D Umeyama fit instead of the under-constrained 2-D
    reprojection solve.
    """

    radar_xyz: List[float]
    image_uv: List[float]
    camera_track_id: int = -1
    radar_track_id: int = -1
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0
    source: str = "auto_pose"
    camera_xyz: Optional[List[float]] = None
    depth_m: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "radar_xyz": [float(value) for value in self.radar_xyz],
            "image_uv": [float(value) for value in self.image_uv],
            "camera_track_id": int(self.camera_track_id),
            "radar_track_id": int(self.radar_track_id),
            "timestamp": float(self.timestamp),
            "confidence": float(self.confidence),
            "source": str(self.source),
        }
        if self.camera_xyz is not None:
            payload["camera_xyz"] = [float(value) for value in self.camera_xyz]
        if self.depth_m is not None:
            payload["depth_m"] = float(self.depth_m)
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CalibrationSample":
        camera_xyz = payload.get("camera_xyz")
        return cls(
            radar_xyz=[float(value) for value in payload["radar_xyz"]],
            image_uv=[float(value) for value in payload["image_uv"]],
            camera_track_id=int(payload.get("camera_track_id", -1)),
            radar_track_id=int(payload.get("radar_track_id", -1)),
            timestamp=float(payload.get("timestamp", time.time())),
            confidence=float(payload.get("confidence", 1.0)),
            source=str(payload.get("source", "auto_pose")),
            camera_xyz=[float(v) for v in camera_xyz] if camera_xyz else None,
            depth_m=float(payload["depth_m"]) if "depth_m" in payload and payload["depth_m"] is not None else None,
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
    rmse_m: float = 0.0
    solve_type: str = ""


@dataclass
class AutoCalibrationConfig:
    """Thresholds for unattended radar-camera calibration."""

    min_samples_for_solve: int = 8
    max_samples: int = 25
    min_sample_spacing_m: float = 0.4
    min_sample_interval_s: float = 0.85
    required_coverage_cells: int = 5
    required_depth_bands: int = 2
    stationary_velocity_mps: float = 1.85
    stationary_pixel_speed: float = 260.0
    stable_link_frames: int = 10
    redetect_pixel_error_multiplier: float = 2.0
    redetect_trigger_seconds: float = 10.0
    max_rotation_near_bound_deg: float = 5.0
    solve_retry_timeout_s: float = 60.0
    min_detection_confidence: float = 0.70
    min_visible_keypoints: int = 12
    min_torso_keypoints: int = 4
    keypoint_conf_threshold: float = 0.35
    torso_keypoint_conf_threshold: float = 0.45
    min_radar_confidence: float = 0.65
    min_radar_age_frames: int = 8
    depth_min_m: float = 0.5
    depth_max_m: float = 6.5
    radar_depth_tolerance_m: float = 0.7
    image_border_px: float = 40.0
    stale_radar_frame_s: float = FusionConfig.stale_radar_frame_s
    max_median_error_3d_px: float = 25.0
    max_rmse_3d_m: float = 0.12
    max_median_error_2d_px: float = 45.0
    max_translation_near_bound_m: float = 0.05
    hold_window_s: float = 0.35
    min_hold_frames: int = 3
    max_hold_pixel_jitter: float = 95.0
    max_hold_radar_jitter_m: float = 0.60


class AutoCalibrationState(Enum):
    IDLE = "IDLE"
    COLLECTING = "COLLECTING"
    SOLVING = "SOLVING"
    MONITORING = "MONITORING"


@dataclass
class AutoCalibrationStatus:
    """Small status snapshot rendered by the GUI."""

    state: AutoCalibrationState = AutoCalibrationState.IDLE
    sample_count: int = 0
    coverage_cells_filled: int = 0
    depth_bands_filled: int = 0
    min_samples_for_solve: int = 8
    required_coverage_cells: int = 5
    required_depth_bands: int = 2
    last_solve_result: Optional[CalibrationSolveResult] = None
    last_event: str = "Auto calibration idle."
    operator_instruction: str = ""
    saved_path: str = ""
    record_path: str = ""

    def progress_text(self) -> str:
        if self.state == AutoCalibrationState.IDLE:
            return self.last_event or "Auto calibration idle."
        if self.state == AutoCalibrationState.MONITORING:
            return self.last_event or "Monitoring saved calibration."
        if self.state == AutoCalibrationState.SOLVING:
            return "Solving auto calibration..."
        prefix = (
            f"Collecting {self.sample_count}/{self.min_samples_for_solve} samples, "
            f"{self.coverage_cells_filled}/{self.required_coverage_cells} needed cells, "
            f"{self.depth_bands_filled}/{self.required_depth_bands} needed depth bands."
        )
        return f"{prefix} {self.last_event}" if self.last_event else prefix


class AutoCalibrationController:
    """Collect, solve, save, and monitor unattended calibration samples."""

    def __init__(
        self,
        projection: CameraProjection,
        event_logger: Optional[FusionEventLogger],
        config: Optional[AutoCalibrationConfig] = None,
    ):
        self.projection = projection
        self.event_logger = event_logger or FusionEventLogger()
        self.config = config or AutoCalibrationConfig()
        self.state = AutoCalibrationState.IDLE
        self.samples: List[CalibrationSample] = []
        self._last_sample_time = -float("inf")
        self._last_monitor_sample_time = -float("inf")
        self._collect_started_at = 0.0
        self._last_solve_sample_count = -1
        self._last_solve_result: Optional[CalibrationSolveResult] = None
        self._last_saved_path = ""
        self._last_event = "Auto calibration idle."
        self._radar_track_ages: Dict[int, int] = {}
        self._stable_pair: Optional[Tuple[int, int]] = None
        self._stable_pair_frames = 0
        self._camera_motion_history: Dict[int, List[Tuple[float, np.ndarray]]] = {}
        self._accepted_median_error_px = 0.0
        self._monitor_errors: List[Tuple[float, float]] = []
        self._monitor_bad_since: Optional[float] = None
        self._last_frame_shape: Optional[Tuple[int, ...]] = None
        self._solver_projection: Optional[CameraProjection] = None
        self._hold_samples: List[CalibrationSample] = []
        self._calibration_radar_id: Optional[int] = None
        self._record_path: Optional[Path] = None
        self._record_handle: Optional[Any] = None
        self._operator_instruction = "Press Auto Calibrate, then keep one person visible."
        self._last_candidate_debug: Dict[str, Any] = {}
        self._status = self._build_status()

    @property
    def status(self) -> AutoCalibrationStatus:
        return self._status

    def solve_now(self, now: Optional[float] = None) -> AutoCalibrationStatus:
        now = time.time() if now is None else float(now)
        if self.state == AutoCalibrationState.IDLE:
            self._last_event = "Start Auto Calibrate before solving auto samples."
        elif not self.samples:
            self._last_event = "No auto calibration samples collected yet."
        else:
            self._try_solve(now, force=False)
        self._status = self._build_status()
        return self._status

    def start(
        self,
        solver_projection: Optional[CameraProjection] = None,
        *,
        record_path: Optional[Path | str] = None,
    ) -> AutoCalibrationStatus:
        self.state = AutoCalibrationState.COLLECTING
        self.samples.clear()
        self._reset_runtime_history()
        self._open_record(record_path)
        self._solver_projection = (
            _copy_camera_projection(solver_projection)
            if solver_projection is not None
            else _copy_camera_projection(self.projection)
        )
        self._collect_started_at = time.time()
        self._last_solve_sample_count = -1
        self._last_solve_result = None
        self._last_saved_path = ""
        self._last_event = "Only one person in view. The first accepted radar track will be locked for this calibration."
        self._operator_instruction = "Walk slowly. Do not freeze completely; the radar needs motion to keep a track."
        self._record_event("start", {"solver_projection": self._working_projection().params})
        self._status = self._build_status()
        return self._status

    def stop(self) -> AutoCalibrationStatus:
        self.state = AutoCalibrationState.IDLE
        self.samples.clear()
        self._reset_runtime_history()
        self._solver_projection = None
        self._record_event("stop", {})
        self._close_record()
        self._last_event = "Auto calibration cancelled."
        self._operator_instruction = "Auto calibration is off."
        self._status = self._build_status()
        return self._status

    def update(
        self,
        camera_tracks: Sequence[Any],
        radar_frame: Optional[RadarFrame],
        depth_frame: Optional[np.ndarray],
        frame_shape: Optional[Tuple[int, ...]],
        now: Optional[float] = None,
    ) -> AutoCalibrationStatus:
        now = time.time() if now is None else float(now)
        self._last_frame_shape = tuple(frame_shape) if frame_shape is not None else None
        if self.state == AutoCalibrationState.IDLE:
            self._status = self._build_status()
            return self._status
        if self.state == AutoCalibrationState.MONITORING:
            self._update_monitoring(camera_tracks, radar_frame, depth_frame, frame_shape, now)
            self._status = self._build_status()
            return self._status

        self._update_collecting(camera_tracks, radar_frame, depth_frame, frame_shape, now)
        self._status = self._build_status()
        return self._status

    def _reset_runtime_history(self) -> None:
        self._last_sample_time = -float("inf")
        self._last_monitor_sample_time = -float("inf")
        self._radar_track_ages.clear()
        self._stable_pair = None
        self._stable_pair_frames = 0
        self._camera_motion_history.clear()
        self._monitor_errors.clear()
        self._monitor_bad_since = None
        self._hold_samples.clear()
        self._calibration_radar_id = None

    def _update_collecting(
        self,
        camera_tracks: Sequence[Any],
        radar_frame: Optional[RadarFrame],
        depth_frame: Optional[np.ndarray],
        frame_shape: Optional[Tuple[int, ...]],
        now: float,
    ) -> None:
        sample, reason = self._candidate_sample(
            camera_tracks,
            radar_frame,
            depth_frame,
            frame_shape,
            now,
            for_monitoring=False,
        )
        if sample is not None:
            if self._calibration_radar_id is None:
                self._calibration_radar_id = int(sample.radar_track_id)
            self.samples.append(sample)
            self._last_sample_time = now
            self._hold_samples.clear()
            cells = len(self._coverage_cells(frame_shape))
            bands = len(self._depth_bands())
            self._last_event = (
                f"Accepted sample {len(self.samples)} from radar R{self._calibration_radar_id} "
                f"({cells}/9 cells, {bands}/3 depth bands)."
            )
            self._operator_instruction = self._next_position_instruction(frame_shape)
            self._record_event(
                "sample_accepted",
                {
                    "sample": sample.to_dict(),
                    "sample_count": len(self.samples),
                    "coverage_cells": cells,
                    "depth_bands": bands,
                    "instruction": self._operator_instruction,
                },
            )
        elif reason:
            self._last_event = reason
            self._operator_instruction = self._instruction_for_reason(reason, frame_shape)
            self._record_event(
                "sample_rejected",
                {
                    "reason": reason,
                    "candidate": self._last_candidate_debug,
                    "sample_count": len(self.samples),
                    "coverage_cells": len(self._coverage_cells(frame_shape)),
                    "depth_bands": len(self._depth_bands()),
                    "instruction": self._operator_instruction,
                },
            )

        ready = self._ready_to_solve(frame_shape)
        forced = len(self.samples) >= self.config.max_samples
        timed_retry = (
            self._last_solve_result is not None
            and not self._last_solve_result.ok
            and now - self._collect_started_at >= self.config.solve_retry_timeout_s
        )
        new_samples_since_solve = len(self.samples) != self._last_solve_sample_count
        if self.samples and (forced or (ready and new_samples_since_solve) or (ready and timed_retry)):
            self._try_solve(now, force=forced or timed_retry)

    def _update_monitoring(
        self,
        camera_tracks: Sequence[Any],
        radar_frame: Optional[RadarFrame],
        depth_frame: Optional[np.ndarray],
        frame_shape: Optional[Tuple[int, ...]],
        now: float,
    ) -> None:
        sample, reason = self._candidate_sample(
            camera_tracks,
            radar_frame,
            depth_frame,
            frame_shape,
            now,
            for_monitoring=True,
        )
        if sample is None:
            if reason:
                self._last_event = f"Monitoring saved calibration. {reason}"
            return

        self._last_monitor_sample_time = now
        error = _sample_pixel_errors([sample], self.projection)[0]
        self._monitor_errors.append((now, error))
        cutoff = now - self.config.redetect_trigger_seconds
        self._monitor_errors = [
            item for item in self._monitor_errors
            if item[0] >= cutoff
        ]
        median = float(np.median([item[1] for item in self._monitor_errors]))
        threshold = max(
            10.0,
            self._accepted_median_error_px * self.config.redetect_pixel_error_multiplier,
        )
        if median > threshold:
            if self._monitor_bad_since is None:
                self._monitor_bad_since = now
            if now - self._monitor_bad_since >= self.config.redetect_trigger_seconds:
                self.state = AutoCalibrationState.COLLECTING
                self.samples.clear()
                self._reset_runtime_history()
                self._collect_started_at = now
                self._last_solve_sample_count = -1
                self._last_event = (
                    f"Live residual median {median:.1f}px exceeded {threshold:.1f}px; recalibrating."
                )
                return
        else:
            self._monitor_bad_since = None
        self._last_event = f"Monitoring saved calibration: live median {median:.1f}px."

    def _candidate_sample(
        self,
        camera_tracks: Sequence[Any],
        radar_frame: Optional[RadarFrame],
        depth_frame: Optional[np.ndarray],
        frame_shape: Optional[Tuple[int, ...]],
        now: float,
        *,
        for_monitoring: bool,
    ) -> Tuple[Optional[CalibrationSample], str]:
        frame_h, frame_w = _shape_hw(frame_shape)
        debug: Dict[str, Any] = {
            "now": float(now),
            "frame_shape": [] if frame_shape is None else [int(v) for v in frame_shape[:3]],
            "for_monitoring": bool(for_monitoring),
        }
        self._last_candidate_debug = debug
        live_tracks = [
            track for track in camera_tracks
            if not bool(getattr(track, "is_predicted", False))
            and int(getattr(track, "frames_since_detection", 0)) == 0
            and getattr(track, "bbox", None) is not None
        ]
        debug["live_camera_track_count"] = len(live_tracks)
        if len(live_tracks) != 1:
            self._reset_link_stability()
            return None, f"Need exactly one visible camera track; saw {len(live_tracks)}."
        camera_track = live_tracks[0]
        debug["camera_track"] = {
            "track_id": int(getattr(camera_track, "track_id", -1)),
            "detection_conf": float(getattr(camera_track, "detection_conf", 0.0) or 0.0),
            "frames_since_detection": int(getattr(camera_track, "frames_since_detection", 0)),
            "is_predicted": bool(getattr(camera_track, "is_predicted", False)),
            "using_radar": bool(getattr(camera_track, "using_radar", False)),
            "fusion_mode": str(getattr(camera_track, "fusion_mode", FusionMode.CAMERA_LOCKED.value)),
            "bbox": _array_list(getattr(camera_track, "bbox", None)),
        }
        occlusion = getattr(camera_track, "occlusion_state", None)
        if occlusion in (
            OcclusionState.PARTIALLY_OCCLUDED,
            OcclusionState.HEAVILY_OCCLUDED,
            OcclusionState.LOST,
        ):
            label = getattr(occlusion, "label", str(occlusion))
            debug["camera_track"]["occlusion_state"] = label
            return None, f"Camera track is {label}; wait until fully visible."
        if bool(getattr(camera_track, "using_radar", False)):
            return None, "Camera track is radar-assisted; wait until camera is fully locked."
        fusion_mode = str(getattr(camera_track, "fusion_mode", FusionMode.CAMERA_LOCKED.value))
        if fusion_mode not in (
            FusionMode.CAMERA_LOCKED.value,
            FusionMode.CAMERA_DEGRADED.value,
            FusionMode.CAMERA_RELOCKED.value,
        ):
            return None, f"Camera fusion mode is {fusion_mode}; wait for camera lock."
        conf = float(getattr(camera_track, "detection_conf", 0.0) or 0.0)
        if conf < self.config.min_detection_confidence:
            return None, f"Camera confidence {conf:.2f} < {self.config.min_detection_confidence:.2f}."
        visible = _visible_keypoint_count(
            getattr(camera_track, "keypoints", None),
            self.config.keypoint_conf_threshold,
        )
        debug["camera_track"]["visible_keypoints"] = visible
        if visible < self.config.min_visible_keypoints:
            return None, f"Need {self.config.min_visible_keypoints}+ visible keypoints; saw {visible}."
        torso_visible = _visible_torso_keypoint_count(
            getattr(camera_track, "keypoints", None),
            self.config.torso_keypoint_conf_threshold,
        )
        debug["camera_track"]["visible_torso_keypoints"] = torso_visible
        if torso_visible < self.config.min_torso_keypoints:
            return None, (
                f"Need all shoulder/hip keypoints visible; saw "
                f"{torso_visible}/{self.config.min_torso_keypoints}."
            )

        bbox = np.asarray(camera_track.bbox, dtype=np.float64)
        image_uv = _calibration_image_point(bbox, getattr(camera_track, "keypoints", None))
        debug["image_uv"] = [float(image_uv[0]), float(image_uv[1])]
        if frame_w > 0 and frame_h > 0:
            border = self.config.image_border_px
            if not (border <= image_uv[0] < frame_w - border and border <= image_uv[1] < frame_h - border):
                return None, "Torso point is too close to the image border."

        depth_m = _depth_meters_from_frame(depth_frame, image_uv)
        debug["depth_m"] = None if depth_m is None else float(depth_m)
        if depth_frame is not None:
            if depth_m is None:
                return None, "Depth missing at torso point."
            if not (self.config.depth_min_m <= depth_m <= self.config.depth_max_m):
                return None, f"Depth {depth_m:.2f}m outside {self.config.depth_min_m:.1f}-{self.config.depth_max_m:.1f}m."

        radar_tracks = _fresh_radar_tracks(radar_frame, now, self.config.stale_radar_frame_s)
        self._update_radar_track_ages(radar_tracks)
        debug["radar_track_count"] = len(radar_tracks)
        debug["radar_frame_number"] = None if radar_frame is None else int(radar_frame.frame_number)
        if not radar_tracks:
            self._reset_link_stability()
            return None, "Radar has no fresh track; walk slowly or shift your weight."
        if not for_monitoring and self._calibration_radar_id is not None:
            debug["locked_calibration_radar_id"] = int(self._calibration_radar_id)
            radar_tracks = [
                track for track in radar_tracks
                if int(track.track_id) == int(self._calibration_radar_id)
            ]
            if not radar_tracks:
                self._reset_link_stability()
                return None, (
                    f"Calibration is locked to radar R{self._calibration_radar_id}; "
                    "waiting for that same track. If the radar ID changed, cancel Auto "
                    "and start a fresh one-person run."
                )
        if depth_m is not None and all(
            abs(float(track.y) - float(depth_m)) > self.config.radar_depth_tolerance_m
            for track in radar_tracks
        ):
            nearest = min(abs(float(track.y) - float(depth_m)) for track in radar_tracks)
            return None, (
                f"Radar/depth mismatch {nearest:.2f}m "
                f"> {self.config.radar_depth_tolerance_m:.1f}m."
            )
        radar_track = self._select_calibration_radar_track(radar_tracks, depth_m)
        if radar_track is None:
            self._reset_link_stability()
            return None, "No radar track is stable/confident enough yet; keep moving slowly."
        debug["radar_track"] = {
            "track_id": int(radar_track.track_id),
            "position": [float(v) for v in radar_track.position_3d],
            "velocity": [float(v) for v in radar_track.velocity_3d],
            "confidence": float(getattr(radar_track, "confidence", 0.0) or 0.0),
            "age_frames": int(self._radar_track_ages.get(int(radar_track.track_id), 0)),
        }
        radar_conf = float(getattr(radar_track, "confidence", 0.0) or 0.0)
        if radar_conf < self.config.min_radar_confidence:
            return None, f"Radar confidence {radar_conf:.2f} < {self.config.min_radar_confidence:.2f}."
        radar_id = int(radar_track.track_id)
        radar_age = self._radar_track_ages.get(radar_id, 0)
        if radar_age < self.config.min_radar_age_frames:
            return None, f"Radar track age {radar_age}/{self.config.min_radar_age_frames} frames."

        camera_id = int(getattr(camera_track, "track_id", -1))
        stable_frames = self._update_link_stability(camera_id, radar_id)
        debug["stable_link_frames"] = stable_frames
        if stable_frames < self.config.stable_link_frames:
            return None, f"Waiting for stable camera-radar link {stable_frames}/{self.config.stable_link_frames}."

        radar_speed = float(np.linalg.norm(getattr(radar_track, "velocity_3d", np.zeros(3))))
        debug["radar_speed_mps"] = radar_speed
        if radar_speed > self.config.stationary_velocity_mps:
            return None, f"Move slower: radar speed {radar_speed:.2f} m/s."
        pixel_speed = self._camera_pixel_speed(camera_id, image_uv, now)
        debug["camera_pixel_speed"] = pixel_speed
        if pixel_speed > self.config.stationary_pixel_speed:
            return None, f"Move slower: camera motion {pixel_speed:.0f} px/s."
        if depth_m is not None and abs(depth_m - float(radar_track.y)) > self.config.radar_depth_tolerance_m:
            return None, (
                f"Radar/depth mismatch {abs(depth_m - float(radar_track.y)):.2f}m "
                f"> {self.config.radar_depth_tolerance_m:.1f}m."
            )

        sample_time = self._last_monitor_sample_time if for_monitoring else self._last_sample_time
        if now - sample_time < self.config.min_sample_interval_s:
            return None, "Holding sample rate limit."

        try:
            sample_frame = replace(radar_frame, tracks=[radar_track]) if radar_frame is not None else radar_frame
            sample = capture_calibration_sample([camera_track], sample_frame)
        except ValueError as exc:
            return None, str(exc)
        sample.timestamp = now
        sample.source = "auto_calibration"
        if depth_m is not None:
            projection = self.projection if for_monitoring else self._working_projection()
            camera_xyz = backproject_pixel(float(image_uv[0]), float(image_uv[1]), depth_m, projection)
            if camera_xyz is not None:
                sample.camera_xyz = [float(v) for v in camera_xyz]
                sample.depth_m = float(depth_m)

        stable_sample, stable_reason = self._stabilized_sample(sample, now)
        debug["hold_sample_count"] = len(self._hold_samples)
        if stable_sample is None:
            return None, stable_reason
        sample = stable_sample
        debug["stabilized_sample"] = sample.to_dict()

        if not for_monitoring:
            candidate_xyz = np.asarray(sample.radar_xyz, dtype=np.float64)
            for existing in self.samples:
                distance = float(np.linalg.norm(candidate_xyz - np.asarray(existing.radar_xyz, dtype=np.float64)))
                if distance < self.config.min_sample_spacing_m:
                    return None, f"Move to a new spot: nearest sample is {distance:.2f}m away."
        return sample, ""

    def _try_solve(self, now: float, *, force: bool) -> None:
        self.state = AutoCalibrationState.SOLVING
        self._last_solve_sample_count = len(self.samples)
        sample_snapshot = list(self.samples)
        sample_path = None
        if self._record_path is not None:
            sample_path = self._record_path.with_name(
                self._record_path.stem.replace("session", "samples") + ".json"
            )
            save_calibration_samples(sample_snapshot, sample_path)
        self._record_event(
            "solve_started",
            {
                "sample_count": len(sample_snapshot),
                "sample_path": "" if sample_path is None else str(sample_path),
                "force": bool(force),
            },
        )
        result, solve_samples, solve_label = self._solve_best_sample_set(sample_snapshot)
        accepted, reason = self._auto_accept_result(result)
        if accepted:
            self.projection.update(**result.params)
            path = self.projection.save()
            self.event_logger.write(
                "auto_calibration_saved",
                {
                    "path": str(path),
                    "sample_count": int(result.sample_count),
                    "median_error_px": float(result.median_error_px),
                    "mean_error_px": float(result.mean_error_px),
                    "rmse_m": float(result.rmse_m),
                    "solve_type": result.solve_type,
                    "solve_label": solve_label,
                },
            )
            self.samples.clear()
            self.state = AutoCalibrationState.MONITORING
            self._last_solve_result = result
            self._last_saved_path = str(path)
            self._accepted_median_error_px = max(float(result.median_error_px), 1.0)
            self._monitor_errors.clear()
            self._monitor_bad_since = None
            self._solver_projection = None
            if result.solve_type == "3d":
                self._last_event = (
                    f"Saved: median {result.median_error_px:.1f}px, rmse {result.rmse_m * 100.0:.1f}cm."
                )
            else:
                self._last_event = f"Saved: median {result.median_error_px:.1f}px via 2-D solve."
            self._operator_instruction = "Saved. Calibration is now monitoring for bumps or drift."
            self._record_event(
                "solve_accepted",
                {
                    "message": result.message,
                    "solve_label": solve_label,
                    "solve_sample_count": len(solve_samples),
                    "params": result.params,
                    "median_error_px": result.median_error_px,
                    "mean_error_px": result.mean_error_px,
                    "rmse_m": result.rmse_m,
                    "solve_type": result.solve_type,
                    "saved_path": str(path),
                    "sample_path": "" if sample_path is None else str(sample_path),
                },
            )
            return

        self.state = AutoCalibrationState.COLLECTING
        self._last_solve_result = CalibrationSolveResult(
            ok=False,
            message=reason,
            params=dict(result.params),
            mean_error_px=float(result.mean_error_px),
            median_error_px=float(result.median_error_px),
            sample_count=int(result.sample_count),
            rmse_m=float(result.rmse_m),
            solve_type=result.solve_type,
        )
        self._last_event = reason
        self._operator_instruction = self._instruction_for_reason(reason, self._last_frame_shape)
        self._record_event(
            "solve_rejected",
            {
                "message": result.message,
                "reason": reason,
                "solve_label": solve_label,
                "solve_sample_count": len(solve_samples),
                "params": result.params,
                "median_error_px": result.median_error_px,
                "mean_error_px": result.mean_error_px,
                "rmse_m": result.rmse_m,
                "solve_type": result.solve_type,
                "sample_path": "" if sample_path is None else str(sample_path),
            },
        )
        if force or now - self._collect_started_at >= self.config.solve_retry_timeout_s:
            drop_count = max(1, len(self.samples) // 2)
            del self.samples[:drop_count]
            self._collect_started_at = now
            self._last_solve_sample_count = -1
            self._last_event = f"{reason} Dropped oldest {drop_count} samples; keep walking."

    def _solve_best_sample_set(
        self,
        samples: Sequence[CalibrationSample],
    ) -> Tuple[CalibrationSolveResult, List[CalibrationSample], str]:
        by_radar: Dict[int, List[CalibrationSample]] = {}
        for sample in samples:
            by_radar.setdefault(int(sample.radar_track_id), []).append(sample)
        candidates: List[Tuple[str, List[CalibrationSample]]] = []
        if len(by_radar) <= 1:
            candidates.append(("all_samples", list(samples)))
        for radar_id, radar_samples in sorted(
            by_radar.items(),
            key=lambda item: (-len(item[1]), item[0]),
        ):
            if len(radar_samples) >= min(self.config.min_samples_for_solve, 5):
                candidates.append((f"radar_track_{radar_id}", radar_samples))
        if not candidates and samples:
            radar_id, radar_samples = max(by_radar.items(), key=lambda item: len(item[1]))
            candidates.append((f"radar_track_{radar_id}", list(radar_samples)))

        best_result: Optional[CalibrationSolveResult] = None
        best_samples: List[CalibrationSample] = []
        best_label = ""
        best_score = float("inf")
        prior = MountGeometryPrior.load()
        for label, candidate_samples in candidates:
            result = solve_calibration_constrained(candidate_samples, self._working_projection(), prior)
            if result.ok:
                score = float(result.median_error_px) - min(len(candidate_samples), 12) * 0.15
            else:
                penalty = 500.0
                if result.params:
                    penalty = 200.0
                score = penalty + float(result.median_error_px) + max(0, self.config.min_samples_for_solve - len(candidate_samples)) * 20.0
            if score < best_score:
                best_result = result
                best_samples = list(candidate_samples)
                best_label = label
                best_score = score

        if best_result is None:
            best_result = CalibrationSolveResult(
                ok=False,
                message="No calibration sample candidates were available.",
            )
        self._record_event(
            "solve_candidates",
            {
                "candidate_labels": [label for label, _items in candidates],
                "selected": best_label,
                "selected_sample_count": len(best_samples),
            },
        )
        return best_result, best_samples, best_label

    def _working_projection(self) -> CameraProjection:
        if self._solver_projection is None:
            self._solver_projection = _copy_camera_projection(self.projection)
        return self._solver_projection

    def _select_calibration_radar_track(
        self,
        radar_tracks: Sequence[RadarTrack],
        depth_m: Optional[float],
    ) -> Optional[RadarTrack]:
        candidates = []
        for track in radar_tracks:
            confidence = float(getattr(track, "confidence", 0.0) or 0.0)
            if confidence < self.config.min_radar_confidence:
                continue
            track_id = int(track.track_id)
            age = self._radar_track_ages.get(track_id, 0)
            if age < self.config.min_radar_age_frames:
                continue
            depth_err = abs(float(track.y) - float(depth_m)) if depth_m is not None else 0.0
            if depth_m is not None and depth_err > self.config.radar_depth_tolerance_m:
                continue
            score = (confidence * 2.0) + min(age / 60.0, 1.0) - (depth_err * 0.8)
            candidates.append((score, track))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def _stabilized_sample(
        self,
        sample: CalibrationSample,
        now: float,
    ) -> Tuple[Optional[CalibrationSample], str]:
        self._hold_samples.append(sample)
        cutoff = now - self.config.hold_window_s
        self._hold_samples = [
            item for item in self._hold_samples
            if float(item.timestamp) >= cutoff
            and int(item.camera_track_id) == int(sample.camera_track_id)
            and int(item.radar_track_id) == int(sample.radar_track_id)
        ]
        if len(self._hold_samples) < self.config.min_hold_frames:
            return (
                None,
                f"Hold still: stabilizing sample {len(self._hold_samples)}/{self.config.min_hold_frames} frames.",
            )

        image_arr = np.asarray([item.image_uv for item in self._hold_samples], dtype=np.float64)
        radar_arr = np.asarray([item.radar_xyz for item in self._hold_samples], dtype=np.float64)
        image_med = np.median(image_arr, axis=0)
        radar_med = np.median(radar_arr, axis=0)
        image_jitter = float(np.max(np.linalg.norm(image_arr - image_med, axis=1)))
        radar_jitter = float(np.max(np.linalg.norm(radar_arr - radar_med, axis=1)))
        if image_jitter > self.config.max_hold_pixel_jitter:
            return None, f"Hold still: image jitter {image_jitter:.0f}px."
        if radar_jitter > self.config.max_hold_radar_jitter_m:
            return None, f"Hold still: radar jitter {radar_jitter:.2f}m."

        stabilized = CalibrationSample(
            radar_xyz=[float(value) for value in radar_med],
            image_uv=[float(value) for value in image_med],
            camera_track_id=int(sample.camera_track_id),
            radar_track_id=int(sample.radar_track_id),
            timestamp=now,
            confidence=float(np.median([item.confidence for item in self._hold_samples])),
            source="auto_calibration_stabilized",
        )
        depth_values = [
            float(item.depth_m)
            for item in self._hold_samples
            if item.depth_m is not None and math.isfinite(float(item.depth_m))
        ]
        if depth_values:
            stabilized.depth_m = float(np.median(depth_values))
        camera_xyz_values = [
            item.camera_xyz
            for item in self._hold_samples
            if item.camera_xyz is not None and len(item.camera_xyz) >= 3
        ]
        if camera_xyz_values:
            cam_arr = np.asarray(camera_xyz_values, dtype=np.float64)
            stabilized.camera_xyz = [float(value) for value in np.median(cam_arr, axis=0)]
        return stabilized, ""

    def _instruction_for_reason(
        self,
        reason: str,
        frame_shape: Optional[Tuple[int, ...]],
    ) -> str:
        reason_lower = reason.lower()
        if "exactly one" in reason_lower:
            return "Only one person should be in view. Everyone else step out."
        if "no fresh track" in reason_lower:
            return "The radar lost you while static. Walk slowly or gently shift your weight."
        if "stable/confident" in reason_lower:
            return "Keep moving slowly so one radar track becomes dominant."
        if "occluded" in reason_lower or "fully visible" in reason_lower:
            return "Step into clear view. Keep your shoulders and hips visible."
        if "camera confidence" in reason_lower or "keypoints" in reason_lower:
            return "Face the camera/radar. Keep your torso, shoulders, and hips visible."
        if "move slower" in reason_lower:
            return "Move slower and smoother. Keep your torso visible."
        if "stand still" in reason_lower or "hold still" in reason_lower or "stabilizing" in reason_lower:
            return "Keep the motion smooth for a moment. Do not freeze completely."
        if "new spot" in reason_lower:
            return self._next_position_instruction(frame_shape)
        if "radar/depth mismatch" in reason_lower:
            return "Stand squarely in front of both sensors; avoid edge positions and reflective clutter."
        if "radar track age" in reason_lower or "stable camera-radar link" in reason_lower:
            return "Stay still in this spot while the radar track settles."
        return "Hold still in a clear, fully visible pose."

    def _next_position_instruction(self, frame_shape: Optional[Tuple[int, ...]]) -> str:
        cells = self._coverage_cells(frame_shape)
        bands = self._depth_bands()
        if len(self.samples) < self.config.min_samples_for_solve:
            needed = self.config.min_samples_for_solve - len(self.samples)
            base = f"Good. Walk slowly to a clearly different spot. Need {needed} more sample"
            return base + ("." if needed == 1 else "s.")
        if len(cells) < self.config.required_coverage_cells:
            return "Good. Move to a different left/center/right area of the image."
        if len(bands) < self.config.required_depth_bands:
            missing = {"near", "mid", "far"} - {
                {0: "near", 1: "mid", 2: "far"}[band]
                for band in bands
            }
            return f"Good. Move to the missing depth band: {', '.join(sorted(missing))}."
        return "Coverage is ready. Hold still while auto calibration solves."

    def _open_record(self, record_path: Optional[Path | str]) -> None:
        self._close_record()
        self._record_path = Path(record_path) if record_path is not None else None
        if self._record_path is None:
            return
        self._record_path.parent.mkdir(parents=True, exist_ok=True)
        self._record_handle = self._record_path.open("a", encoding="utf-8", buffering=1)

    def _close_record(self) -> None:
        if self._record_handle is None:
            return
        try:
            self._record_handle.flush()
            self._record_handle.close()
        finally:
            self._record_handle = None

    def _record_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        if self._record_handle is None:
            return
        record = {
            "type": str(event_type),
            "timestamp": time.time(),
            "state": self.state.value,
            **_json_ready(payload),
        }
        self._record_handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

    def _auto_accept_result(self, result: CalibrationSolveResult) -> Tuple[bool, str]:
        if not result.ok:
            return False, result.message
        near_limit = 90.0 - self.config.max_rotation_near_bound_deg
        near_bounds = [
            key for key in ("yaw_deg", "pitch_deg", "roll_deg")
            if abs(float(result.params.get(key, 0.0))) >= near_limit
        ]
        if near_bounds:
            return False, f"Rejected auto solve: rotation near limit ({', '.join(near_bounds)})."
        if result.solve_type == "2d":
            translation_limit = 2.0 - self.config.max_translation_near_bound_m
            near_translation_bounds = [
                key for key in ("tx", "ty", "tz")
                if abs(float(result.params.get(key, 0.0))) >= translation_limit
            ]
            if near_translation_bounds:
                return False, (
                    "Rejected auto solve: translation hit solver limit "
                    f"({', '.join(near_translation_bounds)})."
                )
        if result.solve_type == "3d":
            if result.median_error_px > self.config.max_median_error_3d_px:
                return False, (
                    f"Rejected auto solve: median {result.median_error_px:.1f}px "
                    f"> {self.config.max_median_error_3d_px:.0f}px."
                )
            if result.rmse_m > self.config.max_rmse_3d_m:
                return False, (
                    f"Rejected auto solve: 3-D rmse {result.rmse_m * 100.0:.1f}cm "
                    f"> {self.config.max_rmse_3d_m * 100.0:.0f}cm."
                )
        elif result.median_error_px > self.config.max_median_error_2d_px:
            return False, (
                f"Rejected auto solve: median {result.median_error_px:.1f}px "
                f"> {self.config.max_median_error_2d_px:.0f}px."
            )
        return True, result.message

    def _ready_to_solve(self, frame_shape: Optional[Tuple[int, ...]]) -> bool:
        return (
            len(self.samples) >= self.config.min_samples_for_solve
            and len(self._coverage_cells(frame_shape)) >= self.config.required_coverage_cells
            and len(self._depth_bands()) >= self.config.required_depth_bands
        )

    def _coverage_cells(self, frame_shape: Optional[Tuple[int, ...]]) -> set:
        frame_h, frame_w = _shape_hw(frame_shape)
        if frame_h <= 0 or frame_w <= 0:
            return set()
        cells = set()
        for sample in self.samples:
            u, v = [float(value) for value in sample.image_uv]
            col = min(2, max(0, int(u / max(frame_w / 3.0, 1.0))))
            row = min(2, max(0, int(v / max(frame_h / 3.0, 1.0))))
            cells.add((row, col))
        return cells

    def _depth_bands(self) -> set:
        bands = set()
        for sample in self.samples:
            depth = float(sample.radar_xyz[1])
            if depth < 1.5:
                bands.add(0)
            elif depth <= 3.0:
                bands.add(1)
            else:
                bands.add(2)
        return bands

    def _update_radar_track_ages(self, radar_tracks: Sequence[RadarTrack]) -> None:
        active_ids = {int(track.track_id) for track in radar_tracks}
        for track_id in list(self._radar_track_ages):
            if track_id not in active_ids:
                del self._radar_track_ages[track_id]
        for track_id in active_ids:
            self._radar_track_ages[track_id] = self._radar_track_ages.get(track_id, 0) + 1

    def _update_link_stability(self, camera_id: int, radar_id: int) -> int:
        pair = (int(camera_id), int(radar_id))
        if self._stable_pair == pair:
            self._stable_pair_frames += 1
        else:
            self._stable_pair = pair
            self._stable_pair_frames = 1
        return self._stable_pair_frames

    def _reset_link_stability(self) -> None:
        self._stable_pair = None
        self._stable_pair_frames = 0

    def _camera_pixel_speed(self, camera_id: int, image_uv: np.ndarray, now: float) -> float:
        history = self._camera_motion_history.setdefault(int(camera_id), [])
        point = np.asarray(image_uv, dtype=np.float64).reshape(2)
        history.append((now, point))
        cutoff = now - 0.5
        del history[:sum(1 for stamp, _ in history if stamp < cutoff)]
        if len(history) < 2:
            return 0.0
        dt = max(float(history[-1][0] - history[0][0]), 1e-6)
        return float(np.linalg.norm(history[-1][1] - history[0][1]) / dt)

    def _build_status(self) -> AutoCalibrationStatus:
        return AutoCalibrationStatus(
            state=self.state,
            sample_count=len(self.samples),
            coverage_cells_filled=len(self._coverage_cells(self._last_frame_shape)),
            depth_bands_filled=len(self._depth_bands()),
            min_samples_for_solve=self.config.min_samples_for_solve,
            required_coverage_cells=self.config.required_coverage_cells,
            required_depth_bands=self.config.required_depth_bands,
            last_solve_result=self._last_solve_result,
            last_event=self._last_event,
            operator_instruction=self._operator_instruction,
            saved_path=self._last_saved_path,
            record_path="" if self._record_path is None else str(self._record_path),
        )


class GuidedBodyCalibState(Enum):
    IDLE = "IDLE"
    COLLECTING = "COLLECTING"
    DONE = "DONE"


# Keep legacy aliases so existing imports and tests don't break.
CornerReflectorCalibState = GuidedBodyCalibState


_GUIDED_POSITIONS: List[str] = [
    "CENTER, ~2 m (6 ft) from sensor — face forward, stand still.",
    "YOUR LEFT ~1.5 m (5 ft), same depth ~2 m — face forward.",
    "YOUR RIGHT ~1.5 m (5 ft), same depth ~2 m — face forward.",
    "CENTER, move back to ~4 m (13 ft) from sensor.",
    "YOUR LEFT ~1.5 m (5 ft), same depth ~4 m.",
    "YOUR RIGHT ~1.5 m (5 ft), same depth ~4 m.",
]


@dataclass
class ReflectorSample:
    """One body-position correspondence for guided extrinsic calibration."""
    radar_xyz: List[float]
    image_uv: List[float]
    camera_xyz: Optional[List[float]] = None
    depth_m: Optional[float] = None
    snr_peak: float = 0.0


class GuidedBodyCalibController:
    """Guided step-by-step body calibration.

    The GUI instructs the operator where to stand.  For each position the
    operator stands still, clicks their torso centre in the video, and presses
    Capture Position.  The controller reads the radar track (world-frame) as
    the radar side and depth-backprojects the click for the camera side.

    After MIN_POSITIONS captures, Solve runs the Umeyama SVD
    (solve_calibration_3d) to compute the extrinsic transform.
    """

    MIN_POSITIONS = 4

    def __init__(self, projection: CameraProjection):
        self._projection = projection
        self.state = GuidedBodyCalibState.IDLE
        self.samples: List[ReflectorSample] = []
        self._step = 0
        self._status = "Idle."

    @property
    def current_instruction(self) -> str:
        if self.state != GuidedBodyCalibState.COLLECTING:
            return ""
        if self._step < len(_GUIDED_POSITIONS):
            return _GUIDED_POSITIONS[self._step]
        return "All positions done — press Solve."

    def start(self) -> str:
        self.samples.clear()
        self._step = 0
        self.state = GuidedBodyCalibState.COLLECTING
        self._status = (
            f"Position 1/{len(_GUIDED_POSITIONS)}: {_GUIDED_POSITIONS[0]} "
            "Click your torso centre in the video, then press Capture Position."
        )
        return self._status

    def capture_position(
        self,
        radar_frame: Optional[Any],
        depth_frame: Optional[Any],
        click_uv: Tuple[float, float],
    ) -> Tuple[bool, str]:
        """Record one person-position correspondence.

        Prefers a single radar track (world-frame); falls back to the
        highest-SNR raw point if no tracks are present.
        Returns (ok, status_message).
        """
        if self.state != GuidedBodyCalibState.COLLECTING:
            return False, "Call start() first."

        radar_xyz: Optional[List[float]] = None

        if radar_frame is not None and getattr(radar_frame, "tracks", None):
            tracks = radar_frame.tracks
            if len(tracks) > 1:
                self._status = (
                    f"Multiple radar tracks ({len(tracks)}) — "
                    "ensure only one person is in the room and press Capture again."
                )
                return False, self._status
            t = tracks[0]
            radar_xyz = [float(t.x), float(t.y), float(t.z)]
        elif radar_frame is not None and getattr(radar_frame, "points", None):
            best = max(radar_frame.points, key=lambda p: float(p.snr))
            if float(best.y) < 0.3:
                self._status = "Radar return in near-field (<0.3 m). Move further away."
                return False, self._status
            radar_xyz = [float(best.x), float(best.y), float(best.z)]

        if radar_xyz is None:
            self._status = (
                "No radar data. Ensure radar is running and you are "
                "standing within the radar FOV."
            )
            return False, self._status

        depth_m = _depth_meters_from_frame(depth_frame, click_uv)
        camera_xyz: Optional[List[float]] = None
        if depth_m is not None and depth_m > 0.1:
            bp = backproject_pixel(
                float(click_uv[0]), float(click_uv[1]), depth_m, self._projection
            )
            if bp is not None:
                camera_xyz = [float(v) for v in bp]

        self.samples.append(ReflectorSample(
            radar_xyz=radar_xyz,
            image_uv=list(click_uv),
            camera_xyz=camera_xyz,
            depth_m=float(depth_m) if depth_m is not None else None,
            snr_peak=0.0,
        ))

        self._step += 1
        n = len(self.samples)
        total = len(_GUIDED_POSITIONS)

        if self._step < total:
            next_instr = _GUIDED_POSITIONS[self._step]
            self._status = (
                f"Position {n} captured. "
                f"Position {n + 1}/{total}: {next_instr} "
                "Click torso, then Capture Position."
            )
        else:
            self.state = GuidedBodyCalibState.DONE
            self._status = f"All {n} positions captured. Press Solve."

        return True, self._status

    def solve(self) -> "CalibrationSolveResult":
        calib_samples = [
            CalibrationSample(
                radar_xyz=s.radar_xyz,
                image_uv=s.image_uv,
                camera_xyz=s.camera_xyz,
                depth_m=s.depth_m,
                source="guided_body",
                confidence=1.0,
            )
            for s in self.samples
        ]
        return solve_calibration_3d(calib_samples, self._projection)

    @property
    def status(self) -> str:
        return self._status

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def last_snr_peak(self) -> Optional[float]:
        return None


# Legacy alias so existing imports of CornerReflectorCalibController still work.
CornerReflectorCalibController = GuidedBodyCalibController


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
        self._radar_track_ages: Dict[int, int] = {}
        self._last_snapshot = FusionDebugSnapshot()

    @property
    def debug_snapshot(self) -> FusionDebugSnapshot:
        return self._last_snapshot

    def reset(self) -> None:
        self._radar_track_ages.clear()
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
        radar_tracks_raw = _fresh_radar_tracks(radar_frame, now, self.config.stale_radar_frame_s)
        self._update_radar_track_ages(radar_tracks_raw)
        radar_tracks, suppressed_duplicates = self._dedupe_radar_tracks(
            radar_tracks_raw,
            projection,
        )
        mode_counts: Dict[str, int] = {}
        linked: Dict[int, int] = {}
        ambiguous_frames = 0

        for track in tracks:
            _ensure_fusion_attrs(track)
            previous_mode = getattr(track, "fusion_mode", FusionMode.CAMERA_LOCKED.value)
            match, candidate_count = self._select_radar_track(track, radar_tracks, projection)
            track.radar_candidate_count = int(candidate_count)
            if candidate_count > 1:
                ambiguous_frames += 1

            if self._has_live_camera_detection(track):
                self._update_camera_state(track, matched_radar_track=match)
                blocked = self._camera_is_blocked(track)
                degraded = self._camera_is_degraded(track)
                if not degraded:
                    self._remember_real_pose(track)
                if getattr(track, "using_radar", False):
                    self._handle_recapture(track, radar_frame, projection, frame_w, frame_h, gui_fps, radar_fps)
                elif not self.enabled or projection is None or match is None:
                    self._assign_mode(
                        track,
                        FusionMode.CAMERA_LOCKED if self._camera_is_strong(track) else FusionMode.CAMERA_DEGRADED,
                        radar_frame,
                        gui_fps,
                        radar_fps,
                    )
                    track.using_radar = False
                    track.radar_confidence *= self.config.radar_confidence_decay
                else:
                    if blocked and self._ready_for_radar_takeover(track):
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
                        mode = FusionMode.RADAR_ARMED if blocked or degraded else FusionMode.CAMERA_LOCKED
                        self._assign_mode(track, mode, radar_frame, gui_fps, radar_fps)
            else:
                if match is not None:
                    self._update_radar_link_state(track, match)
                    should_takeover = (
                        getattr(track, "frames_since_detection", 0) >= self.config.lost_frames_to_radar
                        or getattr(track, "using_radar", False)
                        or self._camera_is_blocked(track)
                        or getattr(track, "occlusion_state", None) in (
                            OcclusionState.HEAVILY_OCCLUDED,
                            OcclusionState.LOST,
                        )
                    )
                    if should_takeover and self._ready_for_radar_takeover(track):
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
                    track.radar_link_stable_frames = 0
                    track.radar_candidate_count = 0
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
            radar_track_count_raw=len(radar_tracks_raw),
            duplicate_tracks_suppressed=suppressed_duplicates,
            ambiguous_radar_frames=ambiguous_frames,
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

    def _update_radar_track_ages(self, radar_tracks: Sequence[RadarTrack]) -> None:
        active_ids = {int(track.track_id) for track in radar_tracks}
        for track_id in list(self._radar_track_ages):
            if track_id not in active_ids:
                del self._radar_track_ages[track_id]
        for track_id in active_ids:
            self._radar_track_ages[track_id] = self._radar_track_ages.get(track_id, 0) + 1

    def _radar_track_rank(self, radar_track: RadarTrack) -> float:
        age = float(self._radar_track_ages.get(int(radar_track.track_id), 0))
        point_bonus = float(len(getattr(radar_track, "associated_point_indexes", []) or [])) * 0.05
        return (max(float(radar_track.confidence), 0.0) * 2.0) + min(age / 10.0, 1.5) + point_bonus

    def _tracks_look_duplicate(
        self,
        left: RadarTrack,
        right: RadarTrack,
        projection: Optional[CameraProjection],
    ) -> bool:
        left_xyz = np.asarray(left.position_3d, dtype=np.float64)
        right_xyz = np.asarray(right.position_3d, dtype=np.float64)
        if float(np.linalg.norm(left_xyz - right_xyz)) > self.config.duplicate_track_distance_m:
            return False
        if projection is None:
            return True
        lu, lv = projection.project_3d_to_2d(left_xyz)
        ru, rv = projection.project_3d_to_2d(right_xyz)
        if min(lu, lv, ru, rv) < 0:
            return True
        return float(np.linalg.norm(np.array([lu - ru, lv - rv], dtype=np.float64))) <= self.config.duplicate_track_distance_px

    def _dedupe_radar_tracks(
        self,
        radar_tracks: Sequence[RadarTrack],
        projection: Optional[CameraProjection],
    ) -> Tuple[List[RadarTrack], int]:
        if not self.config.single_person_mode or len(radar_tracks) <= 1:
            return list(radar_tracks), 0

        ranked = sorted(radar_tracks, key=self._radar_track_rank, reverse=True)
        kept: List[RadarTrack] = []
        suppressed = 0
        for radar_track in ranked:
            if any(self._tracks_look_duplicate(radar_track, existing, projection) for existing in kept):
                suppressed += 1
                continue
            kept.append(radar_track)
        kept.sort(key=lambda item: int(item.track_id))
        return kept, suppressed

    def _select_radar_track(
        self,
        track: Any,
        radar_tracks: Sequence[RadarTrack],
        projection: Optional[CameraProjection],
    ) -> Tuple[Optional[RadarTrack], int]:
        if not radar_tracks or projection is None:
            return None, 0

        linked_id = getattr(track, "radar_track_id", None)
        predicted_center = np.asarray(getattr(track, "center", _bbox_center(track.bbox)), dtype=np.float64)
        candidates: List[Tuple[float, RadarTrack]] = []

        for radar_track in radar_tracks:
            age = int(self._radar_track_ages.get(int(radar_track.track_id), 0))
            if age < self.config.min_radar_track_age_frames:
                continue
            if float(radar_track.confidence) < self.config.min_radar_confidence:
                continue
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
            score -= min(age * 2.0, 12.0)
            candidates.append((score, radar_track))

        if not candidates:
            return None, 0
        candidates.sort(key=lambda item: item[0])
        if (
            self.config.single_person_mode
            and linked_id is None
            and len(candidates) > self.config.max_unlinked_radar_candidates
        ):
            return None, len(candidates)
        return candidates[0][1], len(candidates)

    def _update_camera_state(
        self,
        track: Any,
        *,
        matched_radar_track: Optional[RadarTrack],
    ) -> None:
        degraded = self._camera_is_degraded(track)
        blocked = self._camera_is_blocked(track)
        if degraded:
            track.degraded_camera_frames = int(getattr(track, "degraded_camera_frames", 0)) + 1
        else:
            track.degraded_camera_frames = 0
        if blocked:
            track.blocked_camera_frames = int(getattr(track, "blocked_camera_frames", 0)) + 1
        else:
            track.blocked_camera_frames = 0

        if matched_radar_track is None:
            track.radar_link_stable_frames = 0
            track.radar_candidate_count = 0
            return

        track.radar_candidate_count = max(int(getattr(track, "radar_candidate_count", 0)), 1)
        self._update_radar_link_state(track, matched_radar_track)

    def _update_radar_link_state(
        self,
        track: Any,
        matched_radar_track: RadarTrack,
    ) -> None:
        current_id = getattr(track, "radar_track_id", None)
        next_id = int(matched_radar_track.track_id)
        if current_id is not None and int(current_id) == next_id:
            track.radar_link_stable_frames = int(getattr(track, "radar_link_stable_frames", 0)) + 1
        else:
            track.radar_link_stable_frames = 1

    def _camera_is_degraded(self, track: Any) -> bool:
        return not self._camera_is_strong(track)

    def _camera_is_blocked(self, track: Any) -> bool:
        occlusion = getattr(track, "occlusion_state", None)
        if occlusion in (OcclusionState.HEAVILY_OCCLUDED, OcclusionState.LOST):
            return True
        visible_torso = _visible_torso_keypoint_count(
            getattr(track, "keypoints", None),
            self.config.torso_keypoint_conf_threshold,
        )
        detection_conf = float(getattr(track, "detection_conf", 0.0) or 0.0)
        if visible_torso < self.config.min_visible_torso_keypoints and detection_conf < self.config.acceptable_detection_conf:
            return True
        return False

    def _ready_for_radar_takeover(self, track: Any) -> bool:
        if int(getattr(track, "radar_link_stable_frames", 0)) < self.config.min_radar_link_stable_frames:
            return False
        if int(getattr(track, "blocked_camera_frames", 0)) >= self.config.blocked_frames_to_radar:
            return True
        if int(getattr(track, "frames_since_detection", 0)) >= self.config.lost_frames_to_radar:
            return True
        return False

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

        blend = min(max(float(self.config.radar_bbox_blend), 0.0), 1.0)
        if np.all(np.isfinite(old_bbox)):
            blended_bbox = (old_bbox * (1.0 - blend)) + (np.asarray(target_bbox, dtype=np.float64) * blend)
        else:
            blended_bbox = np.asarray(target_bbox, dtype=np.float64)
        track.bbox = blended_bbox.astype(np.float32)
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
        track.degraded_camera_frames = 0
        track.blocked_camera_frames = 0

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
            track.radar_link_stable_frames = 0
            track.blocked_camera_frames = 0
            track.degraded_camera_frames = 0
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
            solve_type="2d",
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
    translation_near_bounds = [
        key
        for key in ("tx", "ty", "tz")
        if abs(float(params.get(key, 0.0))) >= 1.95
    ]
    if median_error > max_median_error_px or near_bounds or translation_near_bounds:
        reasons = []
        if median_error > max_median_error_px:
            reasons.append(
                f"median error {median_error:.1f}px is above {max_median_error_px:.0f}px"
            )
        if near_bounds:
            reasons.append(f"rotation hit limit ({', '.join(near_bounds)})")
        if translation_near_bounds:
            reasons.append(f"translation hit limit ({', '.join(translation_near_bounds)})")
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
            solve_type="2d",
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
        solve_type="2d",
    )


def backproject_pixel(
    u: float,
    v: float,
    depth_m: float,
    projection: CameraProjection,
) -> Optional[Tuple[float, float, float]]:
    """Back-project an image pixel into a camera-frame 3-D point.

    ``projection`` supplies fx, fy, cx, cy. ``depth_m`` is the RealSense
    depth reading in metres at that pixel; returns ``None`` when depth is
    missing, zero, or non-finite.
    """
    if depth_m is None:
        return None
    try:
        z = float(depth_m)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(z) or z <= 0.0:
        return None
    params = projection.params
    fx = float(params.get("fx", 0.0))
    fy = float(params.get("fy", 0.0))
    if fx <= 0.0 or fy <= 0.0:
        return None
    cx = float(params.get("cx", 0.0))
    cy = float(params.get("cy", 0.0))
    x = (float(u) - cx) * z / fx
    y = (float(v) - cy) * z / fy
    return (x, y, z)


def _depth_meters_from_frame(
    depth_frame: Optional[np.ndarray],
    image_uv: Sequence[float],
) -> Optional[float]:
    """Read RealSense-style uint16 millimetre depth at an image point."""
    if depth_frame is None:
        return None
    arr = np.asarray(depth_frame)
    if arr.size == 0 or arr.ndim < 2:
        return None
    h, w = arr.shape[:2]
    u, v = [float(value) for value in image_uv[:2]]
    if not (math.isfinite(u) and math.isfinite(v)):
        return None
    x = int(round(u))
    y = int(round(v))
    if x < 0 or x >= w or y < 0 or y >= h:
        return None
    raw = float(arr[y, x])
    if not math.isfinite(raw) or raw <= 0.0:
        return None
    return raw / 1000.0


def umeyama_rigid_transform(
    src: np.ndarray,
    dst: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Closed-form least-squares rigid transform (rotation + translation).

    Solves ``dst_i ≈ R @ src_i + t`` for ``R ∈ SO(3)`` and ``t ∈ R^3`` using
    Umeyama's SVD-based method. No scale. Returns ``(R, t, rmse)`` where rmse
    is the residual in input units.

    Inputs are Nx3 arrays. Requires N ≥ 3 non-degenerate correspondences.
    """
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    if src.shape != dst.shape or src.shape[0] < 3:
        raise ValueError("Umeyama needs matching Nx3 inputs with N >= 3.")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean

    # Cross-covariance; rows of src_c.T * dst_c produce the 3x3 H.
    H = src_c.T @ dst_c
    U, _s, Vt = np.linalg.svd(H)
    # Correct for reflection so det(R) = +1.
    reflect = np.eye(3)
    if np.linalg.det(Vt.T @ U.T) < 0:
        reflect[2, 2] = -1.0
    R = Vt.T @ reflect @ U.T
    t = dst_mean - R @ src_mean

    # Residual rmse in source/target units.
    projected = (R @ src.T).T + t
    rmse = float(np.sqrt(np.mean(np.sum((projected - dst) ** 2, axis=1))))
    return R, t, rmse


def _projection_from_rigid(
    R: np.ndarray,
    t: np.ndarray,
    base_projection: CameraProjection,
) -> CameraProjection:
    """Build a CameraProjection whose radar→camera transform equals (R, t)."""
    rt = np.eye(4, dtype=np.float64)
    rt[:3, :3] = R
    rt[:3, 3] = t
    return CameraProjection(K=base_projection.K.copy(), RT=rt)


def _copy_camera_projection(source: CameraProjection) -> CameraProjection:
    projection = CameraProjection(K=source.K.copy())
    projection.update(**source.params)
    return projection


def solve_calibration_3d(
    samples: Sequence[CalibrationSample],
    projection: CameraProjection,
    *,
    min_samples: int = 4,
    max_median_error_px: float = 40.0,
) -> CalibrationSolveResult:
    """Solve radar→camera extrinsics from 3-D ↔ 3-D correspondences.

    Uses Umeyama to recover the rigid transform in closed form (globally
    optimal, no initial guess needed). Reprojection error in pixels is then
    reported using the supplied intrinsics for UI feedback.

    Only samples with ``camera_xyz`` populated participate. Call
    ``solve_calibration_samples`` for the legacy 2-D reprojection fallback.
    """
    usable = [
        s for s in samples
        if s.camera_xyz is not None and len(s.camera_xyz) >= 3
    ]
    if len(usable) < min_samples:
        return CalibrationSolveResult(
            ok=False,
            message=(
                f"Need at least {min_samples} depth-backprojected samples for "
                f"the 3-D solve; got {len(usable)} of {len(samples)}."
            ),
            sample_count=len(usable),
            solve_type="3d",
        )

    src = np.asarray([s.radar_xyz for s in usable], dtype=np.float64)
    dst = np.asarray([s.camera_xyz for s in usable], dtype=np.float64)
    try:
        R, t, rmse_m = umeyama_rigid_transform(src, dst)
    except (ValueError, np.linalg.LinAlgError) as exc:
        return CalibrationSolveResult(
            ok=False,
            message=f"Umeyama solve failed: {exc}",
            sample_count=len(usable),
            solve_type="3d",
        )

    solved = _projection_from_rigid(R, t, projection)
    pixel_errors = _sample_pixel_errors(usable, solved)
    mean_error = float(np.mean(pixel_errors)) if pixel_errors else 0.0
    median_error = float(np.median(pixel_errors)) if pixel_errors else 0.0
    params = solved.params

    if median_error > max_median_error_px:
        return CalibrationSolveResult(
            ok=False,
            message=(
                f"3-D solve returned median pixel error {median_error:.1f}px "
                f"(>{max_median_error_px:.0f}px). Likely cause: depth/intrinsics "
                "mismatch or poorly separated samples. Retry with varied positions."
            ),
            params=params,
            mean_error_px=mean_error,
            median_error_px=median_error,
            sample_count=len(usable),
            rmse_m=rmse_m,
            solve_type="3d",
        )
    return CalibrationSolveResult(
        ok=True,
        message=(
            f"Umeyama solved {len(usable)} 3-D samples: rmse={rmse_m * 100.0:.1f}cm, "
            f"median reprojection {median_error:.1f}px, mean {mean_error:.1f}px."
        ),
        params=params,
        mean_error_px=mean_error,
        median_error_px=median_error,
        sample_count=len(usable),
        rmse_m=rmse_m,
        solve_type="3d",
    )


def solve_calibration_auto(
    samples: Sequence[CalibrationSample],
    projection: CameraProjection,
    *,
    min_samples_3d: int = 4,
    min_samples_2d: int = 5,
) -> CalibrationSolveResult:
    """Prefer the 3-D ↔ 3-D solve when enough depth samples exist.

    Falls back to the 2-D reprojection least-squares path only when there
    are not enough depth-backed samples. Once enough depth exists, a failed
    3-D solve means the correspondences are inconsistent; accepting a 2-D
    overfit in that case can save a projection that looks good in one small
    area but makes the radar box drift away elsewhere.
    """
    depth_count = sum(1 for s in samples if s.camera_xyz is not None)
    if depth_count >= min_samples_3d:
        result = solve_calibration_3d(
            samples,
            projection,
            min_samples=min_samples_3d,
        )
        if result.ok:
            return result
        result.message += (
            " Refusing 2-D fallback because depth-backed samples were available; "
            "collect cleaner one-person samples instead."
        )
        return result
    return solve_calibration_samples(
        samples,
        projection,
        min_samples=min_samples_2d,
    )


def solve_calibration_constrained(
    samples: Sequence[CalibrationSample],
    projection: CameraProjection,
    prior: MountGeometryPrior,
    *,
    min_samples_3d: int = 4,
    min_samples_2d: int = 5,
) -> CalibrationSolveResult:
    """Run solve_calibration_auto then clamp translation axes to prior bounds.

    The prior records ruler-measured camera-to-radar physical offsets.  Clamping
    prevents the solver from drifting to physically implausible translations while
    still allowing it to refine within the measured tolerance.
    """
    result = solve_calibration_auto(
        samples, projection,
        min_samples_3d=min_samples_3d,
        min_samples_2d=min_samples_2d,
    )
    if not result.ok:
        return result
    for axis, prior_attr, tol_attr in (
        ("tx", "tx_m", "tx_tolerance_m"),
        ("ty", "ty_m", "ty_tolerance_m"),
        ("tz", "tz_m", "tz_tolerance_m"),
    ):
        centre = getattr(prior, prior_attr)
        tol = getattr(prior, tol_attr)
        result.params[axis] = float(np.clip(result.params[axis], centre - tol, centre + tol))
    result.message += (
        f" [mount prior: ty={prior.ty_m:.3f}±{prior.ty_tolerance_m:.3f}"
        f" tz={prior.tz_m:.3f}±{prior.tz_tolerance_m:.3f} m]"
    )
    return result


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
        "degraded_camera_frames": 0,
        "blocked_camera_frames": 0,
        "radar_link_stable_frames": 0,
        "radar_candidate_count": 0,
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


def _visible_torso_keypoint_count(keypoints: Optional[np.ndarray], threshold: float) -> int:
    if keypoints is None:
        return 0
    arr = np.asarray(keypoints)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return 0
    torso_indices = [5, 6, 11, 12]
    return int(
        sum(
            1
            for idx in torso_indices
            if idx < len(arr) and float(arr[idx, 2]) >= threshold
        )
    )


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
