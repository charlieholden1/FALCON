"""
F.A.L.C.O.N. Vision System – Tkinter GUI Entry Point
======================================================

Wraps the existing vision pipeline (tracking, occlusion, prediction)
in a lightweight Tkinter GUI with:

* Model selection dropdown  (YOLO26 Nano / YOLO26 Large / MediaPipe Pose)
* Start / Stop controls
* Live toggle switches      (skeleton, Kalman predictions)
* Embedded OpenCV video feed rendered inside a Tkinter Label
* Background threading so the GUI never freezes
"""

from __future__ import annotations

import glob
import json
import logging
import os
import site
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Any, Dict, List, Optional


def _maybe_reexec_with_torch_libgomp() -> None:
    """Preload PyTorch's bundled libgomp before OpenCV/Torch are imported.

    Some Jetson/Linux Python environments fail with
    "cannot allocate memory in static TLS block" when torch imports its bundled
    OpenMP runtime after OpenCV/GStreamer libraries are already loaded. The
    reliable fix is to put torch's libgomp in LD_PRELOAD before process start,
    so the GUI re-execs itself once with that environment set.
    """

    if os.name != "posix" or os.environ.get("FALCON_LIBGOMP_PRELOADED") == "1":
        return
    if os.environ.get("FALCON_SKIP_LIBGOMP_PRELOAD") == "1":
        return

    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidate_roots = [
        Path.home() / ".local" / "lib" / version / "site-packages" / "torch.libs",
    ]
    try:
        candidate_roots.append(Path(site.getsitepackages()[0]) / "torch.libs")
    except Exception:
        pass

    libgomp_paths = []
    for root in candidate_roots:
        libgomp_paths.extend(sorted(root.glob("libgomp*.so*")))
    if not libgomp_paths:
        return

    libgomp = str(libgomp_paths[0])
    preload = os.environ.get("LD_PRELOAD", "")
    if libgomp not in preload.split(":"):
        os.environ["LD_PRELOAD"] = f"{libgomp}:{preload}" if preload else libgomp
    os.environ["FALCON_LIBGOMP_PRELOADED"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


if __name__ == "__main__":
    _maybe_reexec_with_torch_libgomp()

import cv2
import numpy as np
from PIL import Image, ImageTk

from occlusion import OcclusionState
from tracking import TrackingManager
from camera_stream import DualStreamCamera
from radar import CameraProjection, IWR6843Driver, MockRadar
from radar_camera_fusion import (
    AutoCalibrationController,
    AutoCalibrationState,
    CornerReflectorCalibController,
    CornerReflectorCalibState,
    FusionEventLogger,
    FusionMode,
    MountGeometryPrior,
    RadarCameraFusionManager,
    backproject_pixel,
    capture_calibration_sample,
    project_radar_track_bbox,
    save_calibration_samples,
    solve_calibration_auto,
    solve_calibration_constrained,
    solve_calibration_samples,
)
from camera_stream import get_depth_meters

logger = logging.getLogger("falcon.gui")

# Discover available cameras once at import time
_CAMERA_LIST = DualStreamCamera.discover_cameras()
print(
    "[FALCON] Cameras discovered: "
    + (
        ", ".join(c["label"] for c in _CAMERA_LIST)
        if _CAMERA_LIST else "none"
    )
)

# ── Model catalogue ─────────────────────────────────────────────────

MODEL_OPTIONS = {
    "Camera Only": {
        "backend": "none",
        "path": None,
        "description": "Open the live camera feed without pose detection",
    },
    "YOLO26 Nano (Default)": {
        "backend": "yolo",
        "path": "yolo26n-pose.pt",
        "description": "High speed, multi-person support",
    },
    "YOLO26 Nano (TensorRT)": {
        "backend": "yolo",
        "path": "yolo26n-pose.engine",
        "description": "Optimized for Jetson (Requires export)",
    },
    "YOLO26 Large": {
        "backend": "yolo",
        "path": "yolo26l-pose.pt",
        "description": "Higher accuracy, slower FPS, multi-person",
    },
    "MediaPipe Pose": {
        "backend": "mediapipe",
        "path": None,
        "description": "Single-person, high keypoint accuracy, CPU-friendly",
    },
}

# ── COCO 17-keypoint skeleton connections ────────────────────────────

SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # head
    (5, 6),                                  # shoulders
    (5, 7), (7, 9),                          # left arm
    (6, 8), (8, 10),                         # right arm
    (5, 11), (6, 12),                        # torso sides
    (11, 12),                                # hips
    (11, 13), (13, 15),                      # left leg
    (12, 14), (14, 16),                      # right leg
]

# MediaPipe 33-landmark index → COCO 17-keypoint index mapping
_MP_TO_COCO = {
    0: 0,    # nose
    2: 1,    # left_eye
    5: 2,    # right_eye
    7: 3,    # left_ear
    8: 4,    # right_ear
    11: 5,   # left_shoulder
    12: 6,   # right_shoulder
    13: 7,   # left_elbow
    14: 8,   # right_elbow
    15: 9,   # left_wrist
    16: 10,  # right_wrist
    23: 11,  # left_hip
    24: 12,  # right_hip
    25: 13,  # left_knee
    26: 14,  # right_knee
    27: 15,  # left_ankle
    28: 16,  # right_ankle
}

CONFIDENCE_THRESHOLD = 0.45
TRAIL_LENGTH = 10

def _gpu_resize(frame: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """Resize for display on CPU to avoid contending with YOLO on CUDA."""
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)


class GuiDiagnosticsRecorder:
    """Append lightweight GUI/radar/fusion snapshots to JSONL."""

    def __init__(self) -> None:
        self.path: Optional[Path] = None
        self._handle: Optional[Any] = None
        self.frame_count = 0
        self.flag_counts: Dict[str, int] = {}

    def open(self, path: Path | str) -> None:
        self.close()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8", buffering=1)
        self.frame_count = 0
        self.flag_counts = {}

    def write(self, record: Dict[str, Any]) -> None:
        if self._handle is None:
            return
        flags = [str(flag) for flag in record.get("flags", [])]
        for flag in flags:
            self.flag_counts[flag] = self.flag_counts.get(flag, 0) + 1
        self.frame_count += 1
        payload = dict(record)
        payload.setdefault("timestamp", time.time())
        self._handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")

    def close(self) -> None:
        if self._handle is None:
            return
        try:
            self._handle.write(
                json.dumps(
                    {
                        "type": "session_end",
                        "timestamp": time.time(),
                        "frame_count": int(self.frame_count),
                        "flag_counts": dict(sorted(self.flag_counts.items())),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            self._handle.flush()
            self._handle.close()
        finally:
            self._handle = None


class FusionViewMode:
    DEBUG = "Debug"
    PRODUCTION = "Production"


# ── MediaPipe helper ────────────────────────────────────────────────

def _mediapipe_to_coco17(landmarks_list, frame_w: int, frame_h: int) -> np.ndarray:
    """
    Convert a MediaPipe PoseLandmarker ``pose_landmarks`` list to a (17, 3)
    array in the same [x_pixel, y_pixel, confidence] format that YOLO pose
    produces.  Works with the new tasks-API NormalizedLandmark objects.
    """
    kp = np.zeros((17, 3), dtype=np.float32)
    for mp_idx, coco_idx in _MP_TO_COCO.items():
        if mp_idx >= len(landmarks_list):
            continue
        lm = landmarks_list[mp_idx]
        kp[coco_idx] = [lm.x * frame_w, lm.y * frame_h, lm.visibility]
    return kp


def _mediapipe_bbox_from_keypoints(kp: np.ndarray, frame_w: int, frame_h: int,
                                    pad: float = 0.05) -> np.ndarray:
    """
    Derive a bounding box [x1, y1, x2, y2] from visible COCO keypoints
    with a small padding factor.
    """
    visible = kp[kp[:, 2] > 0.3]
    if len(visible) == 0:
        return np.array([0, 0, frame_w, frame_h], dtype=np.float32)
    x_min, y_min = visible[:, 0].min(), visible[:, 1].min()
    x_max, y_max = visible[:, 0].max(), visible[:, 1].max()
    w, h = x_max - x_min, y_max - y_min
    pad_x, pad_y = w * pad, h * pad
    return np.array([
        max(0, x_min - pad_x),
        max(0, y_min - pad_y),
        min(frame_w, x_max + pad_x),
        min(frame_h, y_max + pad_y),
    ], dtype=np.float32)


# ── Visualisation helpers (ported from vision_test.py) ───────────────

def _draw_dashed_rect(img, pt1, pt2, colour, thickness=2, dash_len=10):
    """Draw a solid rectangle (replaces per-edge dash loops)."""
    cv2.rectangle(img, pt1, pt2, colour, thickness, cv2.LINE_AA)


def _draw_uncertainty_ellipse(img, center, sigma_x, sigma_y, colour):
    axes = (max(int(2 * sigma_x), 2), max(int(2 * sigma_y), 2))
    cv2.ellipse(img, center, axes, 0, 0, 360, colour, 1, cv2.LINE_AA)


def _draw_trail(img, history, colour, max_len=TRAIL_LENGTH):
    pts = history[-max_len:]
    if len(pts) < 2:
        return
    centers = np.array(
        [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in pts],
        dtype=np.int32,
    )
    cv2.polylines(img, [centers], isClosed=False, color=colour,
                  thickness=2, lineType=cv2.LINE_AA)


def _draw_skeleton(img, keypoints, colour, conf_thresh=0.4):
    """Draw COCO-format skeleton limbs on *img*."""
    if keypoints is None:
        return
    for (i, j) in SKELETON_CONNECTIONS:
        if i >= len(keypoints) or j >= len(keypoints):
            continue
        xi, yi, ci = keypoints[i]
        xj, yj, cj = keypoints[j]
        if ci > conf_thresh and cj > conf_thresh:
            cv2.line(img, (int(xi), int(yi)), (int(xj), int(yj)),
                     colour, 2, cv2.LINE_AA)
    # Draw keypoint dots
    for kp_idx in range(len(keypoints)):
        kx, ky, kc = keypoints[kp_idx]
        if kc > conf_thresh:
            cv2.circle(img, (int(kx), int(ky)), 4, colour, -1, cv2.LINE_AA)


def _draw_ghost_skeleton(img, keypoints, conf_thresh=0.3, dash_len=8):
    """Draw a solid thin skeleton for predicted (ghost) poses."""
    if keypoints is None:
        return
    ghost_colour = (160, 160, 130)
    for (i, j) in SKELETON_CONNECTIONS:
        if i >= len(keypoints) or j >= len(keypoints):
            continue
        xi, yi, ci = keypoints[i]
        xj, yj, cj = keypoints[j]
        if ci > conf_thresh and cj > conf_thresh:
            cv2.line(img, (int(xi), int(yi)), (int(xj), int(yj)),
                     ghost_colour, 1, cv2.LINE_AA)
    for kp_idx in range(len(keypoints)):
        kx, ky, kc = keypoints[kp_idx]
        if kc > conf_thresh:
            cv2.circle(img, (int(kx), int(ky)), 3, ghost_colour, 1, cv2.LINE_AA)


# ── Vision Pipeline (runs in background thread) ─────────────────────

class VisionPipeline:
    """Encapsulates model loading, webcam capture, detection, tracking,
    and frame annotation.  Designed to run in a background thread."""

    # Adaptive skip-rate parameters
    SKIP_MAX: int = 3            # cruise skip rate (low error)
    SKIP_MID: int = 2            # moderate movement
    SKIP_NONE: int = 1           # every frame (high error)
    ERROR_HIGH: float = 30.0     # px – triggers full-detect burst
    ERROR_LOW: float = 10.0      # px – allows relaxed skipping
    BURST_LENGTH: int = 8        # consecutive full-detect frames after spike

    def __init__(self, skip_frames: int = 2):
        self._webcam: Optional[DualStreamCamera] = None
        self.model = None
        self.mp_pose = None              # MediaPipe Pose instance
        self.backend: str = "yolo"       # "yolo" or "mediapipe"
        self._yolo_half = False
        self._yolo_device = "cpu"
        self._yolo_imgsz = 320
        self._yolo_is_engine = False
        self._timing = {
            "cam_ms": 0.0,
            "detect_ms": 0.0,
            "track_ms": 0.0,
            "fusion_ms": 0.0,
            "draw_ms": 0.0,
        }

        # Radar & projection 
        self._radar = None               # IWR6843Driver or MockRadar
        self._radar_backup = None        # stored ref when toggled off
        self._radar_init_thread: Optional[threading.Thread] = None
        self._radar_recovery_thread: Optional[threading.Thread] = None
        self._radar_recovery_in_progress = False
        self._radar_recovery_attempts = 0
        self._radar_status_message = "Radar: idle"
        self.projection: Optional[CameraProjection] = None
        self.radar_enabled = os.environ.get("FALCON_DISABLE_RADAR", "0") != "1"
        self.single_person_mode = True
        self.fusion_view_mode = FusionViewMode.PRODUCTION
        self.show_radar_overlay = True
        self.show_radar_points = True
        self.show_radar_tracks = True
        self.show_fusion_debug = True
        self._fusion_logger = FusionEventLogger()
        self.fusion = RadarCameraFusionManager(event_logger=self._fusion_logger)
        self._gui_diagnostics = GuiDiagnosticsRecorder()
        self.auto_calibration = AutoCalibrationController(
            self.ensure_projection(),
            self._fusion_logger,
        )
        self.corner_reflector_calib = CornerReflectorCalibController(self.ensure_projection())
        self._calibration_samples = []
        self._calibration_click_uv: Optional[tuple[float, float]] = None
        self.allow_point_cloud_calibration = False
        self._fusion_run_dir: Optional[Path] = None

        self.tracker = TrackingManager(
            iou_threshold=0.3,
            max_frames_lost=90,
            prediction_decay=0.90,
            recovery_duration=90,
        )

        # Adaptive frame-skipping state
        self.skip_frames = skip_frames
        self._skip_counter = 0
        self._burst_remaining = 0        # frames left in force-detect burst

        # Toggles (read by the render loop, written by the GUI)
        self.show_skeleton = True
        self.show_predictions = True

        # Thread control
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Latest annotated frame (shared with GUI)
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_tracks = []
        self._latest_radar_frame = None
        self._latest_depth_frame: Optional[np.ndarray] = None
        self._last_frame_shape = None
        self._last_radar_frame_number: Optional[int] = None
        self._last_radar_frame_seen_at = 0.0
        self._radar_stale_since: Optional[float] = None

        # FPS bookkeeping
        self._fps_value = 0.0
        self._fps_timer = time.time()
        self._frame_count = 0

    def ensure_projection(self) -> CameraProjection:
        if self.projection is None:
            self.projection = CameraProjection()
        return self.projection

    # ── model loading ────────────────────────────────────────────────

    def load_model(self, model_key: str) -> str:
        """
        Load the selected model.  Returns an empty string on success or
        an error message.
        """
        cfg = MODEL_OPTIONS[model_key]
        self.backend = cfg["backend"]
        self.model = None
        self.mp_pose = None

        if self.backend == "none":
            return ""

        if self.backend == "yolo":
            try:
                import torch
                from ultralytics import YOLO

                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                is_engine = cfg["path"].endswith(".engine")
                print(f"[FALCON] Loading YOLO model on {device.upper()}...")
                if device == "cuda":
                    torch.backends.cudnn.benchmark = True

                self.model = YOLO(cfg["path"], task="pose")
                self._yolo_device = device
                self._yolo_imgsz = 320 if device == "cuda" else 416
                self._yolo_is_engine = is_engine

                if is_engine:
                    # TensorRT engine – run FP16 inference; device is baked in
                    self._yolo_half = True
                elif device == "cuda":
                    self.model.to(device)
                    # Use FP16 on CUDA-capable GPUs (large speedup on Jetson)
                    self._yolo_half = True
                else:
                    self._yolo_half = False
                
                return ""
            except Exception as exc:
                return f"Failed to load YOLO model: {exc}"

        elif self.backend == "mediapipe":
            try:
                import mediapipe as mp
                from mediapipe.tasks.python import BaseOptions
                from mediapipe.tasks.python.vision import (
                    PoseLandmarker,
                    PoseLandmarkerOptions,
                    RunningMode,
                )

                # Check for GPU delegate availability (MediaPipe uses GPU delegate, not CUDA directly)
                # Note: On Linux/Jetson, MediaPipe Python GPU support can be tricky.
                delegate = BaseOptions.Delegate.CPU
                try:
                    # Attempt to use GPU delegate if available
                    delegate = BaseOptions.Delegate.GPU
                except AttributeError:
                    pass

                print(f"[FALCON] Loading MediaPipe model with delegate: {delegate}...")

                import os
                model_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "pose_landmarker_heavy.task",
                )
                if not os.path.exists(model_path):
                    return (
                        "MediaPipe model file not found.\n"
                        f"Expected at: {model_path}\n"
                        "Download from: https://storage.googleapis.com/"
                        "mediapipe-models/pose_landmarker/"
                        "pose_landmarker_heavy/float16/latest/"
                        "pose_landmarker_heavy.task"
                    )

                options = PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=model_path, delegate=delegate),
                    running_mode=RunningMode.IMAGE,
                    num_poses=1,
                    min_pose_detection_confidence=0.5,
                    min_pose_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self.mp_pose = PoseLandmarker.create_from_options(options)
                return ""
            except ImportError:
                return (
                    "MediaPipe is not installed.\n"
                    "Run:  pip install mediapipe"
                )
            except Exception as exc:
                return f"Failed to initialise MediaPipe: {exc}"

        return "Unknown backend"

    # ── webcam ───────────────────────────────────────────────────────

    def open_camera(self, cam_descriptor: dict) -> bool:
        """
        Open a DualStreamCamera based on a camera descriptor from
        ``DualStreamCamera.discover_cameras()``.
        """
        # Ensure any previous camera is released first
        self.close_camera()

        # Always try the selected camera first.  If it is a webcam and it
        # fails, fall back to any other webcam in the discovered list — this
        # lets the GUI auto-recover when the dropdown index is stale.
        # Stderr is suppressed during fallback attempts so GStreamer/V4L2
        # driver noise doesn't appear in the terminal.
        candidates = [cam_descriptor]
        if cam_descriptor.get("type") == "webcam":
            seen = {cam_descriptor.get("index")}
            for candidate in _CAMERA_LIST:
                if candidate.get("type") != "webcam":
                    continue
                if candidate.get("index") in seen:
                    continue
                candidates.append(candidate)
                seen.add(candidate.get("index"))

        for i, candidate in enumerate(candidates):
            is_rs = candidate["type"] == "realsense"
            idx = candidate.get("index", 0) or 0
            serial = candidate.get("serial")
            label = candidate["label"]
            print(f"[FALCON] Opening camera: {label}")

            is_fallback = i > 0
            if is_fallback and os.name == "posix":
                devnull_fd = os.open(os.devnull, os.O_WRONLY)
                saved_stderr = os.dup(2)
                os.dup2(devnull_fd, 2)

            try:
                self._webcam = DualStreamCamera(
                    webcam_index=idx,
                    use_realsense=is_rs,
                    realsense_serial=serial,
                )
            finally:
                if is_fallback and os.name == "posix":
                    os.dup2(saved_stderr, 2)
                    os.close(saved_stderr)
                    os.close(devnull_fd)

            if self._webcam.is_opened():
                return True
            self.close_camera()

        return False

    def close_camera(self):
        if self._webcam is not None:
            self._webcam.stop()
            self._webcam = None

    def _preferred_radar_cfg(self) -> str:
        fast_cfg = Path("iwr6843_people_tracking_20fps.cfg")
        if fast_cfg.exists():
            return str(fast_cfg)
        return "iwr6843_people_tracking.cfg"

    def _start_fusion_logging(self) -> None:
        root = Path("diagnostics_runs")
        root.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        suffix = f"{int((time.time() % 1.0) * 1000):03d}"
        self._fusion_run_dir = root / f"fusion_{timestamp}_{suffix}"
        self._fusion_run_dir.mkdir(parents=True, exist_ok=True)
        self._fusion_logger.open(self._fusion_run_dir / "fusion_events.jsonl")
        self._gui_diagnostics.open(self._fusion_run_dir / "gui_diagnostics.jsonl")
        self.fusion.reset()

    def _ensure_calibration_record_dir(self) -> Path:
        if self._fusion_run_dir is not None:
            return self._fusion_run_dir
        root = Path("diagnostics_runs")
        root.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        suffix = f"{int((time.time() % 1.0) * 1000):03d}"
        self._fusion_run_dir = root / f"auto_calibration_{timestamp}_{suffix}"
        self._fusion_run_dir.mkdir(parents=True, exist_ok=True)
        return self._fusion_run_dir

    def _stop_fusion_logging(self) -> None:
        self._fusion_logger.close()
        self._gui_diagnostics.close()

    # ── start / stop ─────────────────────────────────────────────────

    def _init_radar(self) -> None:
        """Initialize the projection immediately and radar asynchronously."""
        self._init_projection_from_camera()
        if not self.radar_enabled:
            self._radar_status_message = "Radar: disabled by FALCON_DISABLE_RADAR=1"
            return
        self._radar_status_message = "Radar: scanning..."
        self._radar_init_thread = threading.Thread(
            target=self._init_radar_worker,
            daemon=True,
            name="radar-init",
        )
        self._radar_init_thread.start()

    @staticmethod
    def _intrinsics_from_frame_size(width: float, height: float) -> np.ndarray:
        """Return a sane color-camera K guess when hardware intrinsics are absent."""
        width = max(float(width), 1.0)
        height = max(float(height), 1.0)
        focal = width / (2.0 * float(np.tan(np.deg2rad(75.0 * 0.5))))
        return np.array(
            [
                [focal, 0.0, width * 0.5],
                [0.0, focal, height * 0.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def _camera_intrinsics_from_open_camera(self) -> Optional[np.ndarray]:
        """Get RealSense intrinsics or estimate webcam intrinsics from frame size."""
        if self._webcam is None:
            return None

        # Prefer the factory-calibrated RealSense intrinsics exposed by
        # DualStreamCamera.get_color_intrinsics() (captured at pipeline
        # start). Falls back to the legacy direct _rs_pipeline probe if
        # something upstream missed them.
        try:
            intr = self._webcam.get_color_intrinsics()
        except Exception:
            intr = None
        if intr:
            logger.info("Using RealSense intrinsics for projection")
            return np.array([
                [float(intr["fx"]), 0.0, float(intr["cx"])],
                [0.0, float(intr["fy"]), float(intr["cy"])],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)

        width = height = 0.0
        cap = getattr(self._webcam, "_cap", None)
        if cap is not None:
            try:
                width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)
                height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)
            except Exception:
                width = height = 0.0

        if width <= 1.0 or height <= 1.0:
            shape = self._last_frame_shape
            if shape is not None and len(shape) >= 2:
                height = float(shape[0])
                width = float(shape[1])

        if width > 1.0 and height > 1.0:
            logger.info("Using webcam frame-size intrinsics guess %.0fx%.0f", width, height)
            return self._intrinsics_from_frame_size(width, height)
        return None

    def apply_camera_intrinsics_guess(self) -> bool:
        K = self._camera_intrinsics_from_open_camera()
        if K is None:
            return False
        self.ensure_projection().update(
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
        )
        return True

    def load_mount_geometry_prior(self) -> MountGeometryPrior:
        return MountGeometryPrior.load()

    def save_mount_geometry_prior(self, prior: MountGeometryPrior) -> None:
        prior.save()

    def seed_calibration_prior(self) -> bool:
        """Seed projection from the measured radar/camera mount geometry."""
        prior = MountGeometryPrior.load()
        seeded_intrinsics = self.apply_camera_intrinsics_guess()
        self.ensure_projection().update(
            tx=prior.tx_m,
            ty=prior.ty_m,
            tz=prior.tz_m,
            yaw_deg=prior.yaw_deg,
            pitch_deg=prior.pitch_deg,
            roll_deg=prior.roll_deg,
        )
        return seeded_intrinsics

    def _seeded_calibration_prior_projection(self) -> tuple[CameraProjection, bool]:
        """Return a non-live solver projection seeded from mount geometry."""
        prior = MountGeometryPrior.load()
        K = self._camera_intrinsics_from_open_camera()
        seeded_intrinsics = K is not None
        if K is None:
            K = self.ensure_projection().K.copy()
        projection = CameraProjection(K=K)
        projection.update(
            tx=prior.tx_m,
            ty=prior.ty_m,
            tz=prior.tz_m,
            yaw_deg=prior.yaw_deg,
            pitch_deg=prior.pitch_deg,
            roll_deg=prior.roll_deg,
        )
        return projection, seeded_intrinsics

    def _init_projection_from_camera(self) -> None:
        """Build the camera projection from the current camera intrinsics."""
        K = self._camera_intrinsics_from_open_camera()
        self.projection = CameraProjection(K=K)
        self.auto_calibration.projection = self.projection
        self.seed_calibration_prior()
        calib_path = Path(CameraProjection.DEFAULT_PATH)
        if calib_path.exists():
            try:
                self.projection.load(calib_path)
                logger.info("Loaded radar calibration from %s", calib_path)
            except Exception as exc:
                logger.warning("Failed to load radar calibration: %s", exc)

    def _discover_radar_ports(self) -> list:
        """Return likely TI radar UART ports, best-effort."""
        try:
            from serial.tools import list_ports
        except Exception:
            return []

        ports = list(list_ports.comports())
        if not ports:
            return []

        # Prefer stable XDS110 interface numbering when available:
        # if00 -> command UART, if03 -> logging/data UART.
        command_port = None
        data_port = None
        for port in ports:
            hwid = str(getattr(port, "hwid", "") or "").lower()
            if "0451:bef3" not in hwid and "xds110" not in hwid:
                continue
            if "if00" in hwid:
                command_port = port.device
            elif "if03" in hwid:
                data_port = port.device
        if command_port and data_port:
            return [command_port, data_port]

        scored = []
        for port in ports:
            text = " ".join(
                str(value or "")
                for value in (
                    port.device,
                    port.description,
                    port.manufacturer,
                    port.product,
                    port.interface,
                    port.hwid,
                )
            ).lower()
            score = 0
            if "texas instruments" in text or "ti " in text:
                score += 4
            if "xds110" in text:
                score += 5
            if "aux data port" in text or "application/user uart" in text:
                score += 4
            if "enhanced com port" in text or "xds110 class auxiliary data port" in text:
                score += 4
            if "acm" in port.device.lower() or "usb" in port.device.lower():
                score += 1
            if score > 0:
                scored.append((score, port.device))

        scored.sort(key=lambda item: (-item[0], item[1]))
        chosen = [device for _, device in scored[:2]]
        if len(chosen) >= 2:
            return chosen

        devices = sorted(p.device for p in ports if p.device)
        if len(devices) == 2:
            return devices
        return []

    def _init_radar_worker(self) -> None:
        try:
            ports = self._discover_radar_ports()
            if len(ports) < 2:
                self._radar = None
                self._radar_backup = None
                self._radar_status_message = "Radar: not found"
                logger.info("No TI radar serial pair detected")
                return

            # If discovery returned a stable XDS110 mapping, trust it first and
            # avoid the noisy reversed-order retry unless the first attempt fails
            # after opening successfully but never produces frames.
            attempted_pairs = [(ports[0], ports[1])]
            if not (ports[0].endswith("ttyACM0") and ports[1].endswith("ttyACM1")):
                attempted_pairs.append((ports[1], ports[0]))

            for config_port, data_port in attempted_pairs:
                logger.info("Trying radar ports config=%s data=%s", config_port, data_port)
                self._radar_status_message = f"Radar: opening {config_port} {data_port}"
                driver = IWR6843Driver(
                    config_port=config_port,
                    data_port=data_port,
                    config_path=self._preferred_radar_cfg(),
                )
                if not driver.open():
                    logger.warning("Radar open failed for config=%s data=%s", config_port, data_port)
                    continue

                if driver.wait_for_frame(timeout_s=6.0):
                    logger.info("IWR6843 radar connected on config=%s data=%s", config_port, data_port)
                    self._radar = driver
                    self._radar_backup = driver
                    self._radar_status_message = f"Radar: connected {config_port} {data_port}"
                    return

                logger.warning(
                    "Radar produced no frames on config=%s data=%s; trying next ordering",
                    config_port,
                    data_port,
                )
                diag = driver.diagnostics
                self._radar_status_message = (
                    f"Radar: no frames cfg={config_port} data={data_port} "
                    f"cmd={diag.get('last_command_error', '')} "
                    f"rx={diag.get('rx_bytes', 0)} magic={diag.get('magic_hits', 0)}"
                ).strip()
                driver.close()

            self._radar = None
            self._radar_backup = None
            if not self._radar_status_message.startswith("Radar: no frames"):
                self._radar_status_message = "Radar: no frames"
        finally:
            if threading.current_thread() is self._radar_init_thread:
                self._radar_init_thread = None

    def _schedule_radar_recovery(self, reason: str) -> None:
        if not self.radar_enabled:
            return
        if self._radar_recovery_in_progress:
            return
        if self._radar_init_thread is not None and self._radar_init_thread.is_alive():
            return
        self._radar_recovery_in_progress = True
        self._radar_recovery_attempts += 1
        self._radar_status_message = f"Radar: recovering ({reason})"
        self._fusion_logger.write(
            "radar_recovery_start",
            {
                "reason": str(reason),
                "attempt": int(self._radar_recovery_attempts),
            },
        )
        self._radar_recovery_thread = threading.Thread(
            target=self._recover_radar_worker,
            args=(reason,),
            daemon=True,
            name="radar-recovery",
        )
        self._radar_recovery_thread.start()

    def _recover_radar_worker(self, reason: str) -> None:
        try:
            driver = self._radar_backup if self._radar_backup is not None else self._radar
            if driver is not None:
                try:
                    driver.close()
                except Exception as exc:
                    logger.warning("Radar close during recovery failed: %s", exc)
            self._radar = None
            self._radar_backup = None
            self._latest_radar_frame = None
            self._last_radar_frame_number = None
            self._last_radar_frame_seen_at = 0.0
            self._radar_stale_since = None
            self._init_radar_worker()
            self._fusion_logger.write(
                "radar_recovery_done",
                {
                    "reason": str(reason),
                    "attempt": int(self._radar_recovery_attempts),
                    "status": str(self._radar_status_message),
                },
            )
        finally:
            self._radar_recovery_in_progress = False
            self._radar_recovery_thread = None

    def toggle_radar(self, enabled: bool) -> None:
        """Enable or disable radar fusion at runtime."""
        self.radar_enabled = enabled
        self.fusion.enabled = enabled
        self.tracker.radar = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._start_fusion_logging()

        # Init projection after camera is open; init radar unless safe mode disabled it.
        self._init_radar()

        self.tracker = TrackingManager(
            iou_threshold=0.3,
            max_frames_lost=90,
            prediction_decay=0.90,
            recovery_duration=90,
            radar=None,
            projection=self.projection,
        )
        self._skip_counter = 0
        self._burst_remaining = 0
        self._fps_timer = time.time()
        self._frame_count = 0
        self._latest_tracks = []
        self._latest_radar_frame = None
        self._latest_depth_frame = None
        if self._webcam is not None:
            self._webcam.start()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        self.auto_calibration.stop()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        # Close radar
        if self._radar is not None:
            self._radar.close()
            self._radar = None
            self._radar_backup = None
        self._stop_fusion_logging()
        self._radar_status_message = "Radar: idle"
        self.close_camera()
        with self._lock:
            self._latest_frame = None
            self._latest_tracks = []
            self._latest_radar_frame = None
            self._latest_depth_frame = None
        self._last_radar_frame_number = None
        self._last_radar_frame_seen_at = 0.0
        self._radar_stale_since = None

    @property
    def running(self) -> bool:
        return self._running

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    @property
    def fps(self) -> float:
        return self._fps_value

    def apply_projection_params(self, **params: float) -> None:
        projection = self.ensure_projection()
        projection.update(**params)

    def reset_projection(self) -> None:
        self.ensure_projection().reset()
        self.apply_camera_intrinsics_guess()

    def save_projection(self, path: Optional[str] = None) -> Path:
        projection = self.ensure_projection()
        return projection.save(path)

    def load_projection(self, path: Optional[str] = None) -> Path:
        projection = self.ensure_projection()
        return projection.load(path)

    def projection_params(self) -> dict:
        return self.ensure_projection().params

    def radar_points(self):
        radar = self._radar_backup if self._radar_backup is not None else self._radar
        if radar is None or not radar.is_connected():
            return []
        return radar.get_point_cloud()

    def radar_frame(self):
        radar = self._radar_backup if self._radar_backup is not None else self._radar
        if radar is None or not radar.is_connected():
            return None
        return radar.get_latest_frame()

    def radar_fps(self) -> float:
        radar = self._radar_backup if self._radar_backup is not None else self._radar
        if radar is None:
            return 0.0
        diagnostics = getattr(radar, "diagnostics", {})
        return float(diagnostics.get("fps", 0.0) or 0.0)

    def _radar_frame_age_s(self, radar_frame, *, now: Optional[float] = None) -> Optional[float]:
        if radar_frame is None:
            return None
        ref = time.time() if now is None else float(now)
        timestamp = float(
            getattr(radar_frame, "timestamp", 0.0)
            or getattr(radar_frame, "source_timestamp", 0.0)
            or 0.0
        )
        if timestamp <= 0.0:
            return None
        return max(0.0, ref - timestamp)

    def _update_radar_runtime_state(self, radar_frame, *, now: float) -> None:
        if radar_frame is not None:
            frame_number = int(getattr(radar_frame, "frame_number", -1))
            frame_age = self._radar_frame_age_s(radar_frame, now=now)
            if frame_age is None or frame_age <= max(self.fusion.config.stale_radar_frame_s, 0.6):
                if frame_number != self._last_radar_frame_number:
                    self._last_radar_frame_number = frame_number
                    self._last_radar_frame_seen_at = now
                self._radar_stale_since = None
                return

        driver_present = (self._radar_backup if self._radar_backup is not None else self._radar) is not None
        if not self.radar_enabled or not driver_present:
            self._radar_stale_since = None
            return

        if self._radar_stale_since is None:
            self._radar_stale_since = now
        stale_for = now - self._radar_stale_since
        if stale_for >= 2.5:
            self._schedule_radar_recovery("stale or missing frames")

    def _track_priority(self, track) -> float:
        bbox = np.asarray(getattr(track, "bbox", np.zeros(4, dtype=np.float32)), dtype=np.float64).reshape(4)
        area = max(float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])), 1.0)
        visible = 0.0
        keypoints = getattr(track, "keypoints", None)
        if keypoints is not None:
            visible = float(np.sum(np.asarray(keypoints)[:, 2] >= 0.35)) / 17.0
        return (
            float(getattr(track, "detection_conf", 0.0) or 0.0) * 1.6
            + visible * 0.6
            + min(float(getattr(track, "total_frames_tracked", 0)) / 20.0, 1.0) * 0.5
            + min(area / 30000.0, 1.0) * 0.15
            + (0.9 if bool(getattr(track, "using_radar", False)) else 0.0)
            + float(getattr(track, "radar_confidence", 0.0) or 0.0) * 0.5
            - float(getattr(track, "frames_since_detection", 0)) * 0.25
            - (0.25 if bool(getattr(track, "is_predicted", False)) else 0.0)
        )

    def _primary_track(self, tracks) -> Optional[Any]:
        if not tracks:
            return None
        return max(tracks, key=self._track_priority)

    def _filter_single_person_detections(self, det_boxes, det_confs, det_keypoints):
        if not self.single_person_mode or len(det_boxes) <= 1:
            return det_boxes, det_confs, det_keypoints

        reference_track = self._primary_track(getattr(self.tracker, "tracks", []))
        reference_center = None if reference_track is None else np.asarray(reference_track.center, dtype=np.float64)
        best_idx = 0
        best_score = -float("inf")
        for idx, bbox in enumerate(det_boxes):
            box = np.asarray(bbox, dtype=np.float64).reshape(4)
            width = max(float(box[2] - box[0]), 1.0)
            height = max(float(box[3] - box[1]), 1.0)
            area_score = min((width * height) / 30000.0, 1.0)
            visible_score = 0.0
            if det_keypoints is not None and idx < len(det_keypoints):
                visible_score = float(np.sum(np.asarray(det_keypoints[idx])[:, 2] >= 0.35)) / 17.0
            center_penalty = 0.0
            if reference_center is not None:
                center = np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=np.float64)
                center_penalty = min(float(np.linalg.norm(center - reference_center)) / 500.0, 1.0)
            score = (
                float(det_confs[idx]) * 1.8
                + visible_score * 0.7
                + area_score * 0.2
                - center_penalty * 0.8
            )
            if score > best_score:
                best_score = score
                best_idx = idx

        filtered_boxes = det_boxes[best_idx : best_idx + 1]
        filtered_confs = det_confs[best_idx : best_idx + 1]
        filtered_keypoints = None if det_keypoints is None else det_keypoints[best_idx : best_idx + 1]
        return filtered_boxes, filtered_confs, filtered_keypoints

    def _enforce_single_person_tracks(self, tracks):
        if not self.single_person_mode or len(tracks) <= 1:
            return tracks
        primary = self._primary_track(tracks)
        if primary is None:
            return tracks
        self.tracker.tracks = [primary]
        return [primary]

    def _visible_radar_tracks(self, radar_frame, tracks=None):
        if radar_frame is None or not radar_frame.tracks:
            return []
        if not self.single_person_mode or len(radar_frame.tracks) <= 1:
            return list(radar_frame.tracks)

        linked_id = None
        if tracks:
            primary = self._primary_track(tracks)
            linked_id = None if primary is None else getattr(primary, "radar_track_id", None)
        if linked_id is not None:
            for radar_track in radar_frame.tracks:
                if int(radar_track.track_id) == int(linked_id):
                    return [radar_track]
        best = max(
            radar_frame.tracks,
            key=lambda track: (
                float(getattr(track, "confidence", 0.0) or 0.0),
                len(getattr(track, "associated_point_indexes", []) or []),
            ),
        )
        return [best]

    def _diagnostic_flags(self, tracks, radar_frame, *, now: float) -> List[str]:
        flags: List[str] = []
        snapshot = self.fusion.debug_snapshot
        if radar_frame is None:
            flags.append("no_radar_frame")
        else:
            frame_age = self._radar_frame_age_s(radar_frame, now=now)
            if frame_age is not None and frame_age > self.fusion.config.stale_radar_frame_s:
                flags.append("radar_stale")
            if getattr(radar_frame, "presence", None) not in (None, 0) and len(radar_frame.tracks) == 0:
                flags.append("presence_without_track")
            if len(getattr(radar_frame, "points", []) or []) > 0 and len(radar_frame.tracks) == 0:
                flags.append("points_without_track")
            if len(radar_frame.tracks) > 1:
                flags.append("multiple_radar_tracks_raw")
        if snapshot.duplicate_tracks_suppressed:
            flags.append("duplicate_radar_tracks_suppressed")
        if snapshot.ambiguous_radar_frames:
            flags.append("ambiguous_radar_candidates")
        if self._radar_recovery_in_progress:
            flags.append("radar_recovering")
        primary = self._primary_track(tracks)
        if primary is not None:
            if (
                bool(getattr(primary, "using_radar", False))
                and int(getattr(primary, "frames_since_detection", 0)) == 0
                and float(getattr(primary, "detection_conf", 0.0) or 0.0) >= self.fusion.config.strong_detection_conf
                and getattr(primary, "occlusion_state", None) == OcclusionState.VISIBLE
            ):
                flags.append("radar_only_while_camera_healthy")
        return sorted(set(flags))

    def _record_gui_diagnostics(self, tracks, radar_frame, *, now: float) -> None:
        if self._gui_diagnostics.path is None:
            return
        primary = self._primary_track(tracks)
        snapshot = self.fusion.debug_snapshot
        visible_radar_tracks = self._visible_radar_tracks(radar_frame, tracks=tracks)
        frame_age = self._radar_frame_age_s(radar_frame, now=now)
        record: Dict[str, Any] = {
            "type": "frame",
            "timestamp": float(now),
            "gui_fps": float(self._fps_value),
            "radar_fps": float(self.radar_fps()),
            "fusion_view_mode": str(self.fusion_view_mode),
            "radar_status": str(self._radar_status_message),
            "radar_recovery_attempts": int(self._radar_recovery_attempts),
            "fusion": {
                "mode_counts": dict(snapshot.mode_counts),
                "last_event": str(snapshot.last_event),
                "radar_track_count_raw": int(snapshot.radar_track_count_raw),
                "radar_track_count_filtered": int(snapshot.radar_track_count),
                "duplicate_tracks_suppressed": int(snapshot.duplicate_tracks_suppressed),
                "ambiguous_radar_frames": int(snapshot.ambiguous_radar_frames),
            },
            "radar": {
                "frame_number": None if radar_frame is None else int(radar_frame.frame_number),
                "frame_age_s": None if frame_age is None else round(float(frame_age), 3),
                "point_count": 0 if radar_frame is None else len(radar_frame.points),
                "track_count_raw": 0 if radar_frame is None else len(radar_frame.tracks),
                "track_count_visible": len(visible_radar_tracks),
                "presence": None if radar_frame is None else getattr(radar_frame, "presence", None),
                "visible_track_ids": [int(track.track_id) for track in visible_radar_tracks],
            },
            "camera": {
                "track_count": len(tracks),
            },
            "flags": self._diagnostic_flags(tracks, radar_frame, now=now),
        }
        if primary is not None:
            bbox = np.asarray(primary.bbox, dtype=np.float64).reshape(4)
            record["camera"]["primary_track"] = {
                "track_id": int(getattr(primary, "track_id", -1)),
                "bbox": [round(float(value), 1) for value in bbox],
                "detection_conf": round(float(getattr(primary, "detection_conf", 0.0) or 0.0), 3),
                "frames_since_detection": int(getattr(primary, "frames_since_detection", 0)),
                "is_predicted": bool(getattr(primary, "is_predicted", False)),
                "occlusion_state": getattr(getattr(primary, "occlusion_state", None), "label", str(getattr(primary, "occlusion_state", ""))),
                "fusion_mode": str(getattr(primary, "fusion_mode", "")),
                "using_radar": bool(getattr(primary, "using_radar", False)),
                "radar_track_id": getattr(primary, "radar_track_id", None),
                "radar_confidence": round(float(getattr(primary, "radar_confidence", 0.0) or 0.0), 3),
                "degraded_camera_frames": int(getattr(primary, "degraded_camera_frames", 0)),
                "blocked_camera_frames": int(getattr(primary, "blocked_camera_frames", 0)),
                "radar_link_stable_frames": int(getattr(primary, "radar_link_stable_frames", 0)),
                "radar_candidate_count": int(getattr(primary, "radar_candidate_count", 0)),
            }
        self._gui_diagnostics.write(record)

    def capture_calibration_sample(self) -> str:
        with self._lock:
            tracks = list(self._latest_tracks)
            radar_frame = self._latest_radar_frame
            depth_frame = self._latest_depth_frame
        click_uv = self._calibration_click_uv
        sample = capture_calibration_sample(
            tracks,
            radar_frame,
            image_uv_override=click_uv,
            allow_point_cloud_fallback=self.allow_point_cloud_calibration,
        )
        # Enrich with camera-frame 3-D point from RealSense depth so the
        # solver can run the closed-form Umeyama 3-D <-> 3-D fit.
        u_val, v_val = float(sample.image_uv[0]), float(sample.image_uv[1])
        depth_m = get_depth_meters(depth_frame, int(round(u_val)), int(round(v_val)))
        if depth_m is not None and self.projection is not None:
            xyz = backproject_pixel(u_val, v_val, depth_m, self.projection)
            if xyz is not None:
                sample.camera_xyz = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
                sample.depth_m = float(depth_m)
        self._calibration_samples.append(sample)
        self._calibration_click_uv = None
        if self._fusion_run_dir is not None:
            save_calibration_samples(
                self._calibration_samples,
                self._fusion_run_dir / "calibration_samples.json",
            )
        depth_tag = (
            f"; depth={sample.depth_m:.2f}m"
            if sample.depth_m is not None else "; depth=n/a"
        )
        return (
            f"Captured sample {len(self._calibration_samples)} "
            f"({self._format_calibration_link(sample)}; {sample.source}{depth_tag})."
        )

    @staticmethod
    def _format_calibration_link(sample) -> str:
        camera_label = f"P{sample.camera_track_id}" if sample.camera_track_id >= 0 else "clicked point"
        radar_label = f"R{sample.radar_track_id}" if sample.radar_track_id >= 0 else "radar point cloud"
        return f"{camera_label} -> {radar_label}"

    def clear_calibration_samples(self) -> str:
        self._calibration_samples.clear()
        self._calibration_click_uv = None
        return "Cleared calibration samples."

    def solve_calibration_samples(self) -> str:
        seeded_intrinsics = self.apply_camera_intrinsics_guess()
        # solve_calibration_auto prefers the 3-D <-> 3-D Umeyama path when
        # enough depth-backprojected samples exist, and transparently falls
        # back to the legacy 2-D reprojection solve otherwise.
        result = solve_calibration_auto(
            self._calibration_samples,
            self.ensure_projection(),
        )
        if not result.ok:
            return result.message
        self.apply_projection_params(**result.params)
        prefix = "Seeded intrinsics from live camera. " if seeded_intrinsics else ""
        return prefix + result.message

    def auto_calibration_start(self) -> str:
        solver_projection, seeded_intrinsics = self._seeded_calibration_prior_projection()
        self.auto_calibration.projection = self.ensure_projection()
        record_path = self._ensure_calibration_record_dir() / "auto_calibration_session.jsonl"
        status = self.auto_calibration.start(
            solver_projection=solver_projection,
            record_path=record_path,
        )
        prefix = "Seeded intrinsics and mount prior. " if seeded_intrinsics else "Seeded mount prior. "
        return prefix + status.progress_text() + f" Recording: {record_path}"

    def auto_calibration_stop(self) -> str:
        return self.auto_calibration.stop().progress_text()

    def auto_calibration_solve_now(self) -> str:
        status = self.auto_calibration.solve_now()
        return status.progress_text()

    def auto_calibration_status(self):
        return self.auto_calibration.status

    # ── Corner Reflector Calibration ────────────────────────────────────────

    def corner_reflector_start(self) -> str:
        self.corner_reflector_calib._projection = self.ensure_projection()
        return self.corner_reflector_calib.start()

    def corner_reflector_capture(self) -> str:
        if self._calibration_click_uv is None:
            return "Click the reflector in the video first."
        radar_frame = self._latest_radar_frame
        depth_frame = self._latest_depth_frame
        ok, msg = self.corner_reflector_calib.capture_position(
            radar_frame, depth_frame, self._calibration_click_uv
        )
        if ok:
            self._calibration_click_uv = None
        return msg

    def corner_reflector_solve(self) -> str:
        result = self.corner_reflector_calib.solve()
        if result.ok:
            self.ensure_projection().update(**result.params)
        return result.message

    def corner_reflector_sample_count(self) -> int:
        return self.corner_reflector_calib.sample_count

    def corner_reflector_status(self) -> str:
        return self.corner_reflector_calib.status

    def corner_reflector_last_snr(self) -> Optional[float]:
        return self.corner_reflector_calib.last_snr_peak

    # ────────────────────────────────────────────────────────────────────────

    def calibration_sample_count(self) -> int:
        return len(self._calibration_samples)

    def set_calibration_click(self, u: float, v: float) -> str:
        self._calibration_click_uv = (float(u), float(v))
        return f"Calibration click set at ({u:.0f}, {v:.0f}). Capture Sample will use it once."

    def clear_calibration_click(self) -> None:
        self._calibration_click_uv = None

    def radar_status_text(self) -> str:
        radar = self._radar_backup if self._radar_backup is not None else self._radar
        if radar is None:
            return self._radar_status_message

        points = self.radar_points()
        frame = self.radar_frame()
        track_count = 0 if frame is None else len(frame.tracks)
        diagnostics = getattr(radar, "diagnostics", {})
        if isinstance(radar, MockRadar):
            return f"Radar: mock ({len(points)} pts, {track_count} tracks)"

        frames = diagnostics.get("frames", 0)
        fps = diagnostics.get("fps", 0.0)
        last_error = diagnostics.get("last_parse_error", "")
        packet_size = diagnostics.get("last_packet_size", 0)
        rx_bytes = diagnostics.get("rx_bytes", 0)
        magic_hits = diagnostics.get("magic_hits", 0)
        last_cmd_error = diagnostics.get("last_command_error", "")
        track_hint = ""
        if frame is not None and frame.tracks:
            first = frame.tracks[0]
            track_hint = f" R{first.track_id}=({first.x:.1f},{first.y:.1f},{first.z:.1f})"
        frame_age = self._radar_frame_age_s(frame)
        if last_cmd_error:
            return f"Radar cmd error: {last_cmd_error}"
        if self._radar_recovery_in_progress:
            return f"Radar: recovering attempt {self._radar_recovery_attempts}  {self._radar_status_message}"
        if frame_age is not None and frame_age > self.fusion.config.stale_radar_frame_s:
            return (
                f"Radar: stale {frame_age:.2f}s  {len(points)} pts {track_count} tracks{track_hint}  "
                f"frames {frames}  fps {fps:.1f}"
            )
        if last_error:
            return (
                f"Radar: {len(points)} pts {track_count} tracks{track_hint}  frames {frames}  "
                f"rx {rx_bytes}  magic {magic_hits}  pkt {packet_size}  parse {last_error}"
            )
        return (
            f"Radar: {len(points)} pts {track_count} tracks{track_hint}  frames {frames}  "
            f"rx {rx_bytes}  magic {magic_hits}  pkt {packet_size}  fps {fps:.1f}"
        )

    # ── main vision loop (background thread) ─────────────────────────

    def _loop(self):
        while self._running:
            loop_t0 = time.time()
            if self._webcam is None:
                time.sleep(0.01)
                continue

            frame, depth_frame = self._webcam.read()
            if frame is None:
                continue
            cam_t1 = time.time()

            h_frame, w_frame = frame.shape[:2]
            self._last_frame_shape = frame.shape
            self._skip_counter += 1

            # Decide whether to run the detector this frame
            run_detect = (
                self._burst_remaining > 0
                or self._skip_counter % self.skip_frames == 0
            )

            if run_detect:
                # ── full detection frame ─────────────────────────────
                det_t0 = time.time()
                det_boxes, det_confs, det_keypoints = self._detect(
                    frame, w_frame, h_frame,
                )
                det_t1 = time.time()
                tracks = self.tracker.update(
                    det_boxes, det_keypoints, det_confs,
                    frame=frame, depth_frame=depth_frame,
                )
                tracks = self._enforce_single_person_tracks(tracks)
                track_t1 = time.time()

                # ── adaptive skip-rate adjustment ────────────────────
                err = self.tracker.max_tracking_error
                if err >= self.ERROR_HIGH:
                    # Spike detected – force detection on every frame
                    self._burst_remaining = self.BURST_LENGTH
                    self.skip_frames = self.SKIP_NONE
                elif err <= self.ERROR_LOW:
                    self.skip_frames = self.SKIP_MAX
                else:
                    self.skip_frames = self.SKIP_MID

                if self._burst_remaining > 0:
                    self._burst_remaining -= 1
            else:
                # ── skipped frame: Kalman propagate only ─────────────
                det_t0 = det_t1 = time.time()
                tracks = self.tracker.propagate_only()
                tracks = self._enforce_single_person_tracks(tracks)
                track_t1 = time.time()

            # ── radar-camera fusion ─────────────────────────────────
            fusion_t0 = time.time()
            now = time.time()
            radar_frame = self.radar_frame() if self.radar_enabled else None
            self._update_radar_runtime_state(radar_frame, now=now)
            if self.radar_enabled and self.projection is not None:
                tracks = self.fusion.update(
                    tracks,
                    radar_frame,
                    self.projection,
                    frame_shape=frame.shape,
                    gui_fps=self._fps_value,
                    radar_fps=self.radar_fps(),
                    now=now,
                )
            else:
                tracks = self.fusion.update(
                    tracks,
                    None,
                    self.projection,
                    frame_shape=frame.shape,
                    gui_fps=self._fps_value,
                    radar_fps=0.0,
                    now=now,
                )
            tracks = self._enforce_single_person_tracks(tracks)
            self.auto_calibration.update(
                tracks,
                radar_frame,
                depth_frame,
                frame.shape,
                now=now,
            )
            self._record_gui_diagnostics(tracks, radar_frame, now=now)
            fusion_t1 = time.time()

            # ── annotate ─────────────────────────────────────────────
            draw_t0 = time.time()
            vis = frame.copy()
            self._annotate(vis, tracks)
            draw_t1 = time.time()

            # ── FPS ──────────────────────────────────────────────────
            self._frame_count += 1
            elapsed = time.time() - self._fps_timer
            if elapsed >= 0.5:
                self._fps_value = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_timer = time.time()

            self._timing["cam_ms"] = (cam_t1 - loop_t0) * 1000.0
            self._timing["detect_ms"] = (det_t1 - det_t0) * 1000.0
            self._timing["track_ms"] = (track_t1 - det_t1) * 1000.0
            self._timing["fusion_ms"] = (fusion_t1 - fusion_t0) * 1000.0
            self._timing["draw_ms"] = (draw_t1 - draw_t0) * 1000.0

            with self._lock:
                self._latest_frame = vis
                self._latest_tracks = list(tracks)
                self._latest_radar_frame = radar_frame
                self._latest_depth_frame = depth_frame

    # ── detection dispatch ───────────────────────────────────────────

    def _detect(self, frame, w, h):
        if self.backend == "yolo" and self.model is not None:
            return self._detect_yolo(frame)
        elif self.backend == "mediapipe" and self.mp_pose is not None:
            return self._detect_mediapipe(frame, w, h)
        return np.empty((0, 4)), np.empty(0), None

    def _detect_yolo(self, frame):
        import torch

        predict_kwargs = {
            "verbose": False,
            "half": self._yolo_half,
            "conf": CONFIDENCE_THRESHOLD,
            "classes": [0],
            "max_det": 6,
            "iou": 0.5,
        }
        if not self._yolo_is_engine:
            predict_kwargs["imgsz"] = self._yolo_imgsz
        if not self._yolo_is_engine:
            predict_kwargs["device"] = self._yolo_device

        with torch.inference_mode():
            results = self.model(frame, **predict_kwargs)
        result = results[0]

        det_boxes = np.empty((0, 4))
        det_confs = np.empty(0)
        det_keypoints = None

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes

            # Filter on-device – single boolean mask, no intermediate copies
            mask = boxes.conf >= CONFIDENCE_THRESHOLD

            filtered_boxes = boxes.xyxy[mask]
            if len(filtered_boxes) == 0:
                return det_boxes, det_confs, det_keypoints

            # Transfer only filtered, contiguous float32 tensors to CPU
            det_boxes = filtered_boxes.float().contiguous().cpu().numpy()
            det_confs = boxes.conf[mask].float().contiguous().cpu().numpy()

            if result.keypoints is not None and len(result.keypoints.data) > 0:
                det_keypoints = (
                    result.keypoints.data[mask]
                    .float().contiguous().cpu().numpy()
                )

        return self._filter_single_person_detections(det_boxes, det_confs, det_keypoints)

    def _detect_mediapipe(self, frame, w, h):
        import mediapipe as mp

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.mp_pose.detect(mp_image)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return np.empty((0, 4)), np.empty(0), None

        # result.pose_landmarks is list[list[NormalizedLandmark]]
        kp = _mediapipe_to_coco17(result.pose_landmarks[0], w, h)
        bbox = _mediapipe_bbox_from_keypoints(kp, w, h)

        # Compute average visibility as detection confidence
        avg_vis = float(kp[:, 2].mean())

        det_boxes = bbox.reshape(1, 4)
        det_confs = np.array([avg_vis])
        det_keypoints = kp.reshape(1, 17, 3)

        return self._filter_single_person_detections(det_boxes, det_confs, det_keypoints)

    # ── annotation ───────────────────────────────────────────────────

    def _annotate(self, vis, tracks):
        n_occluded = 0
        n_predicted = 0
        n_recovered = 0
        n_radar_only = 0
        n_depth = 0
        h_frame, w_frame = vis.shape[:2]
        debug_view = self.fusion_view_mode == FusionViewMode.DEBUG
        production_view = self.fusion_view_mode == FusionViewMode.PRODUCTION

        for t in tracks:
            fusion_mode = str(getattr(t, "fusion_mode", FusionMode.CAMERA_LOCKED.value))
            radar_only = fusion_mode == FusionMode.RADAR_ONLY.value or bool(getattr(t, "using_radar", False))
            colour = (0, 165, 255) if radar_only else t.occlusion_state.colour_bgr
            if production_view and radar_only:
                colour = (80, 220, 140)
            x1, y1, x2, y2 = t.bbox.astype(int)
            x1, x2 = max(0, min(x1, w_frame - 1)), max(0, min(x2, w_frame - 1))
            y1, y2 = max(0, min(y1, h_frame - 1)), max(0, min(y2, h_frame - 1))

            # Trajectory trail
            _draw_trail(vis, t.bbox_history, colour)

            if radar_only:
                if production_view:
                    if self.show_skeleton and t.keypoints is not None:
                        _draw_skeleton(vis, t.keypoints, colour, conf_thresh=0.15)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 1)
                    label = f"Person #{t.track_id}"
                else:
                    cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 3)
                    if x2 - x1 > 12 and y2 - y1 > 12:
                        cv2.rectangle(vis, (x1 + 4, y1 + 4), (x2 - 4, y2 - 4), colour, 1)
                    label = f"Person #{t.track_id} RADAR ONLY"
                    if self.show_skeleton and t.keypoints is not None:
                        _draw_skeleton(vis, t.keypoints, colour, conf_thresh=0.15)
                n_radar_only += 1

            elif t.is_predicted:
                if self.show_predictions:
                    _draw_dashed_rect(vis, (x1, y1), (x2, y2), colour, 2)
                    sig_x, sig_y = t.predictor.get_uncertainty()
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    _draw_uncertainty_ellipse(vis, (cx, cy), sig_x, sig_y, colour)
                    # Draw ghost skeleton for predicted tracks
                    if self.show_skeleton and t.keypoints is not None:
                        _draw_ghost_skeleton(vis, t.keypoints)
                label = f"Person #{t.track_id} (PRED {t.prediction_confidence:.0%})"
                n_predicted += 1

            elif t.is_recovered:
                cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 3)
                secs_left = t.recovery_frames_remaining / 30.0
                label = f"Person #{t.track_id} RECOVERED ({secs_left:.1f}s)"
                n_recovered += 1
                if self.show_skeleton and t.keypoints is not None:
                    _draw_skeleton(vis, t.keypoints, colour)

            else:
                cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)
                label = f"Person #{t.track_id}"
                if self.show_skeleton and t.keypoints is not None:
                    _draw_skeleton(vis, t.keypoints, colour)

            # Occlusion state label
            occ_label = t.occlusion_state.label
            if t.z_depth_meters is not None:
                occ_label += f"  [{t.z_depth_meters:.2f}m]"
                n_depth += 1
            if getattr(t, 'using_radar', False):
                if production_view:
                    occ_label += " [ESTIMATED]"
                else:
                    occ_label += (
                        f" [RADAR {t.radar_confidence:.0%}"
                        f" ID {getattr(t, 'radar_track_id', '-')}]"
                    )
            if fusion_mode not in (FusionMode.CAMERA_LOCKED.value, FusionMode.RADAR_ONLY.value):
                occ_label += f" [{fusion_mode}]"
            cv2.putText(vis, label, (x1, y1 - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)
            cv2.putText(vis, occ_label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)

            if t.occlusion_state in (
                OcclusionState.PARTIALLY_OCCLUDED,
                OcclusionState.HEAVILY_OCCLUDED,
                OcclusionState.LOST,
            ):
                n_occluded += 1

        radar_points = []
        radar_frame = self.radar_frame() if self.show_radar_overlay and self.projection is not None else None
        visible_radar_tracks = self._visible_radar_tracks(radar_frame, tracks=tracks)
        radar_track_count = len(visible_radar_tracks)
        radar_track_count_raw = 0 if radar_frame is None else len(radar_frame.tracks)
        radar_first_track = ""
        if visible_radar_tracks:
            first = visible_radar_tracks[0]
            radar_first_track = f" R{first.track_id}=({first.x:.1f},{first.y:.1f},{first.z:.1f})"
        if debug_view and self.show_radar_overlay and self.show_radar_points and self.projection is not None:
            radar_points = self.radar_points()
            self._draw_radar_overlay(vis, radar_points)
        if debug_view and self.show_radar_overlay and self.show_radar_tracks and self.projection is not None and radar_frame is not None:
            self._draw_radar_track_overlay(vis, visible_radar_tracks)

        fusion_status = self.fusion.status_text()
        self._draw_calibration_click(vis)

        # ── Info overlay (top-left) ─────────────────────────────────
        overlay_lines = [
            f"FPS: {self._fps_value:.1f}",
            f"Tracked: {len(tracks)}",
            f"View: {self.fusion_view_mode}",
            f"Occluded: {n_occluded}",
            f"Predicted: {n_predicted}",
            f"Radar Only: {n_radar_only}",
            f"Recovered: {n_recovered}",
            f"Depth: {n_depth}/{len(tracks)}",
            f"Radar Points: {len(radar_points)}",
            f"Radar Tracks: {radar_track_count}"
            + (f" raw:{radar_track_count_raw}" if radar_track_count_raw != radar_track_count else "")
            + radar_first_track,
            f"Timing ms C:{self._timing['cam_ms']:.0f} D:{self._timing['detect_ms']:.0f} T:{self._timing['track_ms']:.0f} F:{self._timing['fusion_ms']:.0f} R:{self._timing['draw_ms']:.0f}",
        ]
        if self.show_fusion_debug:
            overlay_lines.insert(-1, f"Fusion: {fusion_status}")
        for i, line in enumerate(overlay_lines):
            cv2.putText(vis, line, (10, 25 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA)

        # ── Status banner ───────────────────────────────────────────
        if n_occluded == 0:
            cv2.putText(vis, "STATE: ALL VISIBLE", (10, vis.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2, cv2.LINE_AA)

    def _draw_radar_overlay(self, vis, radar_points) -> None:
        if self.projection is None or not radar_points:
            return

        h_frame, w_frame = vis.shape[:2]
        for point, u, v in self.projection.project_points(radar_points):
            if u < 0 or v < 0 or u >= w_frame or v >= h_frame:
                continue
            snr_scale = max(0.2, min(point.snr / 20.0, 1.0))
            colour = (0, int(200 * snr_scale), 255)
            cv2.circle(vis, (u, v), 5, colour, -1, cv2.LINE_AA)
            cv2.circle(vis, (u, v), 10, colour, 1, cv2.LINE_AA)
            cv2.putText(
                vis,
                f"{point.y:.2f}m",
                (u + 8, v - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                colour,
                1,
                cv2.LINE_AA,
            )

    def _draw_radar_track_overlay(self, vis, radar_tracks) -> None:
        if self.projection is None or not radar_tracks:
            return
        h_frame, w_frame = vis.shape[:2]
        for radar_track in radar_tracks:
            bbox = project_radar_track_bbox(
                radar_track,
                self.projection,
                fallback_bbox=None,
                frame_w=w_frame,
                frame_h=h_frame,
            )
            u, v = self.projection.project_3d_to_2d(radar_track.position_3d)
            colour = (255, 180, 0)
            if bbox is not None:
                x1, y1, x2, y2 = bbox.astype(int)
                cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 1, cv2.LINE_AA)
            if 0 <= u < w_frame and 0 <= v < h_frame:
                cv2.drawMarker(
                    vis,
                    (u, v),
                    colour,
                    markerType=cv2.MARKER_CROSS,
                    markerSize=16,
                    thickness=1,
                    line_type=cv2.LINE_AA,
                )
                cv2.putText(
                    vis,
                    f"R{radar_track.track_id}",
                    (u + 8, v + 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    colour,
                    1,
                    cv2.LINE_AA,
                )

    def _draw_calibration_click(self, vis) -> None:
        if self._calibration_click_uv is None:
            return
        h_frame, w_frame = vis.shape[:2]
        u, v = self._calibration_click_uv
        if not (0 <= u < w_frame and 0 <= v < h_frame):
            return
        point = (int(round(u)), int(round(v)))
        colour = (255, 255, 0)
        cv2.drawMarker(
            vis,
            point,
            colour,
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=24,
            thickness=2,
            line_type=cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            "CAL",
            (point[0] + 10, point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            colour,
            2,
            cv2.LINE_AA,
        )

        # Draw a correction arrow from the current projected radar track
        # position to the click location so the operator can visually see
        # the offset that this sample will correct.
        radar_frame = self._latest_radar_frame
        projection = self.projection
        if radar_frame is None or projection is None or not radar_frame.tracks:
            return
        best_track = None
        best_dist = float("inf")
        for track in radar_frame.tracks:
            try:
                ru, rv = projection.project_3d_to_2d(track.position_3d)
            except Exception:
                continue
            dist = (ru - u) ** 2 + (rv - v) ** 2
            if dist < best_dist:
                best_dist = dist
                best_track = (track, int(ru), int(rv))
        if best_track is None:
            return
        track, ru, rv = best_track
        if not (0 <= ru < w_frame and 0 <= rv < h_frame):
            return
        cv2.arrowedLine(
            vis,
            (ru, rv),
            point,
            (0, 180, 255),
            2,
            cv2.LINE_AA,
            tipLength=0.18,
        )
        offset_px = float(np.hypot(u - ru, v - rv))
        cv2.putText(
            vis,
            f"R{track.track_id} -> click: {offset_px:.0f}px",
            (ru + 6, rv - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 180, 255),
            1,
            cv2.LINE_AA,
        )


class RadarCalibrationWindow:
    """Live radar-to-camera calibration controls."""

    PARAM_SPECS = [
        ("fx", "fx", 200.0, 2000.0, 5.0),
        ("fy", "fy", 200.0, 2000.0, 5.0),
        ("cx", "cx", 0.0, 1920.0, 2.0),
        ("cy", "cy", 0.0, 1080.0, 2.0),
        ("tx", "tx", -2.0, 2.0, 0.01),
        ("ty", "ty", -2.0, 2.0, 0.01),
        ("tz", "tz", -2.0, 2.0, 0.01),
        ("yaw_deg", "yaw", -90.0, 90.0, 0.1),
        ("pitch_deg", "pitch", -90.0, 90.0, 0.1),
        ("roll_deg", "roll", -90.0, 90.0, 0.1),
    ]

    def __init__(self, root: tk.Tk, pipeline: VisionPipeline):
        self.root = root
        self.pipeline = pipeline
        self.window = tk.Toplevel(root)
        self.window.title("Radar Calibration")
        self.window.configure(bg="#1e1e2e")
        self.window.geometry("780x640")
        self.window.resizable(False, True)
        style = ttk.Style(self.window)
        style.configure("CalibHero.TLabel", background="#1e1e2e", foreground="#f5e0dc",
                        font=("Segoe UI", 18, "bold"))
        style.configure("CalibInstruction.TLabel", background="#1e1e2e", foreground="#a6e3a1",
                        font=("Segoe UI", 14, "bold"))
        style.configure("CalibDetail.TLabel", background="#1e1e2e", foreground="#cdd6f4",
                        font=("Segoe UI", 11))

        self.status_var = tk.StringVar(
            value=f"Save path: {CameraProjection.DEFAULT_PATH}"
        )
        self.auto_status_var = tk.StringVar(
            value=self.pipeline.auto_calibration_status().progress_text()
        )
        self.auto_instruction_var = tk.StringVar(
            value="Press Auto Calibrate. Keep exactly one person visible and walk slowly."
        )
        self._vars = {}
        self._suspend_callbacks = False
        self._advanced_visible = False
        self._advanced_grid = None
        self._advanced_button = None
        self._last_auto_saved_path = ""
        self.allow_point_cloud_var = tk.BooleanVar(
            value=bool(self.pipeline.allow_point_cloud_calibration)
        )
        # Mount geometry prior entries (inches, driving m display).
        self._mount_in_vars: dict[str, tk.StringVar] = {}
        self._mount_m_vars: dict[str, tk.StringVar] = {}

        container = ttk.Frame(self.window)
        container.pack(fill="both", expand=True, padx=10, pady=10)

        ttk.Label(
            container,
            text="Radar To Camera Calibration",
            style="Header.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            container,
            textvariable=self.auto_instruction_var,
            style="CalibHero.TLabel",
            wraplength=720,
        ).pack(anchor="w", pady=(4, 8))
        ttk.Label(
            container,
            text=(
                "One person only. Shoulders and hips must be visible. "
                "Walk slowly and smoothly so the radar keeps a track, then move to a clearly different "
                "near / middle / far and left / center / right spot. "
                "Solve Calibration will solve auto samples when no manual samples are queued."
            ),
            style="CalibDetail.TLabel",
            wraplength=720,
        ).pack(anchor="w", pady=(0, 10))

        # ── Mount Geometry Prior panel ──────────────────────────────────────
        self._build_mount_prior_panel(container)

        # ── Guided Body Calibration panel ──────────────────────────────────
        self._reflector_status_var = tk.StringVar(value="Idle.")
        self._reflector_sample_var = tk.StringVar(value="Samples: 0/6")
        self._reflector_instr_var = tk.StringVar(value="")
        self._build_corner_reflector_panel(container)

        auto_row = ttk.Frame(container)
        auto_row.pack(fill="x", pady=(0, 6))
        ttk.Button(auto_row, text="Auto Calibrate", command=self._start_auto_calibration).pack(side="left")
        ttk.Button(auto_row, text="Cancel Auto", command=self._stop_auto_calibration).pack(
            side="left", padx=6
        )
        ttk.Label(
            container,
            textvariable=self.auto_status_var,
            style="CalibInstruction.TLabel",
            wraplength=720,
        ).pack(anchor="w", pady=(0, 8))

        sample_row = ttk.Frame(container)
        sample_row.pack(fill="x", pady=(0, 8))
        ttk.Button(sample_row, text="Capture Sample", command=self._capture_sample).pack(side="left")
        ttk.Button(sample_row, text="Solve Calibration", command=self._solve_samples).pack(
            side="left", padx=6
        )
        ttk.Button(sample_row, text="Clear Samples", command=self._clear_samples).pack(side="left")

        file_row = ttk.Frame(container)
        file_row.pack(fill="x", pady=(0, 10))
        ttk.Button(file_row, text="Save", command=self._save).pack(side="left")
        ttk.Button(file_row, text="Load Saved", command=self._load).pack(side="left", padx=6)
        ttk.Button(file_row, text="Reset Camera Guess", command=self._reset).pack(side="left")
        ttk.Button(file_row, text="Sync From Live", command=self.sync_from_pipeline).pack(
            side="left", padx=6
        )

        self._advanced_button = ttk.Button(
            container,
            text="Show Advanced Sliders",
            command=self._toggle_advanced,
        )
        self._advanced_button.pack(anchor="w", pady=(0, 8))

        grid = ttk.Frame(container)
        self._advanced_grid = grid
        grid.columnconfigure(2, weight=1)

        ttk.Checkbutton(
            grid,
            text="Allow rough point-cloud samples when no radar track exists",
            variable=self.allow_point_cloud_var,
            command=self._sync_options,
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        for row_idx, (key, label, low, high, resolution) in enumerate(self.PARAM_SPECS, start=1):
            ttk.Label(grid, text=label.upper(), width=9).grid(
                row=row_idx, column=0, sticky="w", padx=(0, 8), pady=3
            )
            var = tk.DoubleVar(value=0.0)
            self._vars[key] = var
            scale = tk.Scale(
                grid,
                from_=low,
                to=high,
                resolution=resolution,
                orient="horizontal",
                variable=var,
                length=380,
                bg="#1e1e2e",
                fg="#cdd6f4",
                troughcolor="#313244",
                highlightthickness=0,
                activebackground="#89b4fa",
            )
            scale.grid(row=row_idx, column=1, sticky="ew", padx=(0, 8), pady=2)
            entry = ttk.Entry(grid, textvariable=var, width=10)
            entry.grid(row=row_idx, column=2, sticky="w", pady=2)
            var.trace_add("write", self._on_change)

        self._status_label = ttk.Label(
            container,
            textvariable=self.status_var,
            style="Status.TLabel",
            wraplength=660,
        )
        self._status_label.pack(anchor="w")

        self.window.protocol("WM_DELETE_WINDOW", self.window.destroy)
        self._sync_options()
        self.sync_from_pipeline()
        self._refresh_auto_status()

    def _sync_options(self) -> None:
        self.pipeline.allow_point_cloud_calibration = self.allow_point_cloud_var.get()

    def _toggle_advanced(self) -> None:
        if self._advanced_grid is None or self._advanced_button is None:
            return
        if self._advanced_visible:
            self._advanced_grid.pack_forget()
            self._advanced_button.configure(text="Show Advanced Sliders")
            self._advanced_visible = False
            return

        self._advanced_grid.pack(
            fill="both",
            expand=True,
            pady=(0, 8),
            before=self._status_label,
        )
        self._advanced_button.configure(text="Hide Advanced Sliders")
        self._advanced_visible = True

    def sync_from_pipeline(self) -> None:
        params = self.pipeline.projection_params()
        self._suspend_callbacks = True
        try:
            for key, var in self._vars.items():
                if key in params:
                    var.set(params[key])
        finally:
            self._suspend_callbacks = False
        self.auto_status_var.set(
            self.pipeline.auto_calibration_status().progress_text()
        )
        self.auto_instruction_var.set(
            self.pipeline.auto_calibration_status().operator_instruction
            or "Press Auto Calibrate. Keep exactly one person visible and walk slowly."
        )

    def _refresh_auto_status(self) -> None:
        try:
            if not self.window.winfo_exists():
                return
        except tk.TclError:
            return
        status = self.pipeline.auto_calibration_status()
        self.auto_status_var.set(status.progress_text())
        self.auto_instruction_var.set(
            status.operator_instruction
            or "Press Auto Calibrate. Keep exactly one person visible and walk slowly."
        )
        if (
            status.state == AutoCalibrationState.MONITORING
            and status.saved_path
            and status.saved_path != self._last_auto_saved_path
        ):
            self._last_auto_saved_path = status.saved_path
            self.sync_from_pipeline()
            self.status_var.set(status.progress_text())
        self.window.after(500, self._refresh_auto_status)

    def _on_change(self, *_args) -> None:
        if self._suspend_callbacks:
            return
        params = {key: var.get() for key, var in self._vars.items()}
        self.pipeline.apply_projection_params(**params)
        self.status_var.set(
            "Advanced slider change applied live."
        )

    def _save(self) -> None:
        path = self.pipeline.save_projection()
        self.status_var.set(f"Saved radar calibration to {path}")

    def _load(self) -> None:
        try:
            path = self.pipeline.load_projection()
            self.sync_from_pipeline()
            self.status_var.set(f"Loaded radar calibration from {path}")
        except FileNotFoundError:
            self.status_var.set(
                f"No saved calibration at {CameraProjection.DEFAULT_PATH}"
            )
        except Exception as exc:
            self.status_var.set(f"Load failed: {exc}")

    def _reset(self) -> None:
        self.pipeline.reset_projection()
        self.sync_from_pipeline()
        self.status_var.set("Calibration reset to the live camera intrinsics guess.")

    def _build_mount_prior_panel(self, parent: ttk.Frame) -> None:
        """Build the Mount Geometry Prior input panel."""
        prior = self.pipeline.load_mount_geometry_prior()
        frame = ttk.LabelFrame(parent, text="Mount Geometry Prior (ruler measurements)")
        frame.pack(fill="x", pady=(0, 8))

        axes = [
            ("tx", "tx (lateral)",  prior.tx_m),
            ("ty", "ty (up)",       prior.ty_m),
            ("tz", "tz (fwd/back)", prior.tz_m),
        ]
        for row_idx, (key, label, value_m) in enumerate(axes):
            in_var = tk.StringVar(value=f"{value_m / 0.0254:.2f}")
            m_var  = tk.StringVar(value=f"{value_m:.4f} m")
            self._mount_in_vars[key] = in_var
            self._mount_m_vars[key]  = m_var

            ttk.Label(frame, text=label, width=16).grid(row=row_idx, column=0, sticky="w", padx=(6,2), pady=2)
            in_entry = ttk.Entry(frame, textvariable=in_var, width=8)
            in_entry.grid(row=row_idx, column=1, padx=2)
            ttk.Label(frame, text="in").grid(row=row_idx, column=2, sticky="w")
            ttk.Label(frame, textvariable=m_var, width=12).grid(row=row_idx, column=3, sticky="w", padx=(4,0))

            def _update_m(event=None, k=key, iv=in_var, mv=m_var):
                try:
                    mv.set(f"{float(iv.get()) * 0.0254:.4f} m")
                except ValueError:
                    pass
            in_entry.bind("<KeyRelease>", _update_m)
            in_entry.bind("<FocusOut>", _update_m)

        btn_row = ttk.Frame(frame)
        btn_row.grid(row=len(axes), column=0, columnspan=4, sticky="w", padx=6, pady=(4, 6))
        ttk.Button(btn_row, text="Apply Prior as Seed", command=self._apply_mount_prior).pack(side="left")
        ttk.Button(btn_row, text="Save Mount Geometry", command=self._save_mount_geometry).pack(side="left", padx=8)

    def _apply_mount_prior(self) -> None:
        prior = self._mount_prior_from_ui()
        self.pipeline.save_mount_geometry_prior(prior)
        self.pipeline.seed_calibration_prior()
        self.sync_from_pipeline()
        self.status_var.set(
            f"Mount prior applied: ty={prior.ty_m:.4f} m, tz={prior.tz_m:.4f} m"
        )

    def _save_mount_geometry(self) -> None:
        prior = self._mount_prior_from_ui()
        self.pipeline.save_mount_geometry_prior(prior)
        self.status_var.set(f"Mount geometry saved to mount_geometry.json")

    def _mount_prior_from_ui(self) -> MountGeometryPrior:
        loaded = self.pipeline.load_mount_geometry_prior()
        try:
            tx = float(self._mount_in_vars["tx"].get()) * 0.0254
        except (ValueError, KeyError):
            tx = loaded.tx_m
        try:
            ty = float(self._mount_in_vars["ty"].get()) * 0.0254
        except (ValueError, KeyError):
            ty = loaded.ty_m
        try:
            tz = float(self._mount_in_vars["tz"].get()) * 0.0254
        except (ValueError, KeyError):
            tz = loaded.tz_m
        return MountGeometryPrior(
            tx_m=tx, ty_m=ty, tz_m=tz,
            tx_tolerance_m=loaded.tx_tolerance_m,
            ty_tolerance_m=loaded.ty_tolerance_m,
            tz_tolerance_m=loaded.tz_tolerance_m,
        )

    def _build_corner_reflector_panel(self, parent: ttk.Frame) -> None:
        from radar_camera_fusion import _GUIDED_POSITIONS
        total = len(_GUIDED_POSITIONS)

        frame = ttk.LabelFrame(parent, text="Guided Body Calibration")
        frame.pack(fill="x", pady=(0, 8))

        ttk.Label(
            frame,
            text=(
                "Stand at each position shown below, click your torso centre "
                "in the video, then press Capture Position."
            ),
            wraplength=680,
        ).pack(anchor="w", padx=6, pady=(4, 2))

        # Big instruction label showing where to stand next.
        ttk.Label(
            frame,
            textvariable=self._reflector_instr_var,
            font=("TkDefaultFont", 11, "bold"),
            foreground="#1060c0",
            wraplength=680,
        ).pack(anchor="w", padx=8, pady=(2, 4))

        btn_row = ttk.Frame(frame)
        btn_row.pack(fill="x", padx=6, pady=(2, 4))
        ttk.Button(btn_row, text="Start", command=self._reflector_start).pack(side="left")
        self._reflector_capture_btn = ttk.Button(
            btn_row, text="Capture Position", command=self._reflector_capture, state="disabled"
        )
        self._reflector_capture_btn.pack(side="left", padx=6)
        self._reflector_solve_btn = ttk.Button(
            btn_row, text="Solve", command=self._reflector_solve, state="disabled"
        )
        self._reflector_solve_btn.pack(side="left")

        info_row = ttk.Frame(frame)
        info_row.pack(fill="x", padx=6, pady=(0, 2))
        ttk.Label(info_row, textvariable=self._reflector_sample_var, width=16).pack(side="left")

        ttk.Label(frame, textvariable=self._reflector_status_var, wraplength=680).pack(
            anchor="w", padx=6, pady=(0, 6)
        )
        self._refresh_reflector_status()

    def _reflector_start(self) -> None:
        from radar_camera_fusion import _GUIDED_POSITIONS
        msg = self.pipeline.corner_reflector_start()
        self._reflector_status_var.set(msg)
        self._reflector_capture_btn.config(state="normal")
        self._reflector_solve_btn.config(state="disabled")
        self._reflector_sample_var.set(f"Samples: 0/{len(_GUIDED_POSITIONS)}")
        self._reflector_instr_var.set(
            f"Step 1/{len(_GUIDED_POSITIONS)}: {_GUIDED_POSITIONS[0]}"
        )

    def _reflector_capture(self) -> None:
        from radar_camera_fusion import _GUIDED_POSITIONS, GuidedBodyCalibState
        msg = self.pipeline.corner_reflector_capture()
        self._reflector_status_var.set(msg)
        n = self.pipeline.corner_reflector_sample_count()
        total = len(_GUIDED_POSITIONS)
        self._reflector_sample_var.set(f"Samples: {n}/{total}")
        state = self.pipeline.corner_reflector_calib.state
        if state == GuidedBodyCalibState.DONE:
            self._reflector_solve_btn.config(state="normal")
            self._reflector_instr_var.set("All positions captured — press Solve.")
        elif n < total:
            self._reflector_instr_var.set(
                f"Step {n + 1}/{total}: {_GUIDED_POSITIONS[n]}"
            )
        # Enable Solve early once MIN_POSITIONS reached.
        if n >= self.pipeline.corner_reflector_calib.MIN_POSITIONS:
            self._reflector_solve_btn.config(state="normal")

    def _reflector_solve(self) -> None:
        msg = self.pipeline.corner_reflector_solve()
        self._reflector_status_var.set(msg)
        self._reflector_instr_var.set("")
        self.sync_from_pipeline()
        self.status_var.set(msg)

    def _refresh_reflector_status(self) -> None:
        if not self.window.winfo_exists():
            return
        self._reflector_status_var.set(self.pipeline.corner_reflector_status())
        self.window.after(500, self._refresh_reflector_status)

    def _start_auto_calibration(self) -> None:
        try:
            message = self.pipeline.auto_calibration_start()
            self.auto_status_var.set(self.pipeline.auto_calibration_status().progress_text())
            self.auto_instruction_var.set(
                self.pipeline.auto_calibration_status().operator_instruction
            )
            self.status_var.set(message)
        except Exception as exc:
            self.status_var.set(f"Auto calibration failed to start: {exc}")

    def _stop_auto_calibration(self) -> None:
        self.status_var.set(self.pipeline.auto_calibration_stop())
        self.auto_status_var.set(self.pipeline.auto_calibration_status().progress_text())
        self.auto_instruction_var.set(
            self.pipeline.auto_calibration_status().operator_instruction
        )

    def _capture_sample(self) -> None:
        try:
            message = self.pipeline.capture_calibration_sample()
            self.status_var.set(
                f"{message} Total samples: {self.pipeline.calibration_sample_count()}"
            )
        except Exception as exc:
            self.status_var.set(f"Sample failed: {exc}")

    def _solve_samples(self) -> None:
        try:
            if (
                self.pipeline.calibration_sample_count() == 0
                and self.pipeline.auto_calibration_status().sample_count > 0
            ):
                message = self.pipeline.auto_calibration_solve_now()
            else:
                message = self.pipeline.solve_calibration_samples()
            self.sync_from_pipeline()
            self.status_var.set(message)
        except Exception as exc:
            self.status_var.set(f"Solve failed: {exc}")

    def _clear_samples(self) -> None:
        self.status_var.set(self.pipeline.clear_calibration_samples())


# ── Tkinter GUI ─────────────────────────────────────────────────────

class FalconGUI:
    """
    Main application window.

    Layout
    ------
    ┌────────────────────────────────────────────────┐
    │  F.A.L.C.O.N. Vision System  (title bar)      │
    ├───────────────────────┬────────────────────────┤
    │                       │  Model: [▼ dropdown  ] │
    │                       │  [  Start  ] [ Stop  ] │
    │    Live Video Feed    │  ☑ Show Skeleton       │
    │   (Tkinter Label)     │                        │
    │                       │  ☑ Kalman Predictions  │
    │                       │                        │
    │                       │  FPS: --   Status: --  │
    └───────────────────────┴────────────────────────┘
    """

    REFRESH_MS = 33  # 30 FPS GUI refresh

    def __init__(self):
        self.pipeline = VisionPipeline()
        self.calibration_window: Optional[RadarCalibrationWindow] = None

        # ── Root window ─────────────────────────────────────────────
        self.root = tk.Tk()
        self.root.title("F.A.L.C.O.N. Vision System")
        self.root.configure(bg="#1e1e2e")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.minsize(960, 540)
        self.root.geometry('1280x720')
        self.root.resizable(False, False)

        # ── Style ───────────────────────────────────────────────────
        style = ttk.Style()
        style.theme_use("clam")

        BG = "#1e1e2e"
        FG = "#cdd6f4"
        ACCENT = "#89b4fa"
        BTN_BG = "#313244"
        BTN_ACTIVE = "#45475a"

        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=FG,
                        font=("Segoe UI", 10))
        style.configure("Header.TLabel", background=BG, foreground=ACCENT,
                        font=("Segoe UI", 14, "bold"))
        style.configure("Status.TLabel", background=BG, foreground="#a6adc8",
                        font=("Segoe UI", 9))
        style.configure("CalibHero.TLabel", background=BG, foreground="#f5e0dc",
                        font=("Segoe UI", 18, "bold"))
        style.configure("CalibInstruction.TLabel", background=BG, foreground="#a6e3a1",
                        font=("Segoe UI", 14, "bold"))
        style.configure("CalibDetail.TLabel", background=BG, foreground="#cdd6f4",
                        font=("Segoe UI", 11))
        style.configure("TButton", background=BTN_BG, foreground=FG,
                        font=("Segoe UI", 10, "bold"), padding=6)
        style.map("TButton",
                  background=[("active", BTN_ACTIVE)],
                  foreground=[("active", FG)])
        style.configure("TCheckbutton", background=BG, foreground=FG,
                        font=("Segoe UI", 10))
        style.map("TCheckbutton", background=[("active", BG)])
        style.configure("TCombobox", font=("Segoe UI", 10))

        # ── Layout: left = video, right = controls ──────────────────
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main_frame = ttk.Frame(self.root)
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # Video panel (left)
        video_frame = ttk.Frame(main_frame)
        video_frame.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)
        video_frame.rowconfigure(0, weight=1)
        video_frame.columnconfigure(0, weight=1)

        self.video_label = tk.Label(video_frame, bg="#11111b",
                                    text="No Video Feed",
                                    fg="#6c7086",
                                    font=("Segoe UI", 16))
        self.video_label.grid(row=0, column=0, sticky="nsew")
        self.video_label.bind("<Button-1>", self._on_video_click)

        # Control panel (right)
        ctrl = ttk.Frame(main_frame, width=280)
        ctrl.grid(row=0, column=1, sticky="ns", padx=(4, 8), pady=8)
        ctrl.columnconfigure(0, weight=1)
        # Prevent the control panel from shrinking
        ctrl.grid_propagate(True)

        # ── Title ───────────────────────────────────────────────────
        ttk.Label(ctrl, text="F.A.L.C.O.N.", style="Header.TLabel"
                  ).grid(row=0, column=0, pady=(0, 2), sticky="w")
        ttk.Label(ctrl, text="Vision Control Panel", style="Status.TLabel"
                  ).grid(row=1, column=0, pady=(0, 12), sticky="w")

        # ── Model selection ─────────────────────────────────────────
        ttk.Label(ctrl, text="Detection Model:").grid(
            row=2, column=0, sticky="w", pady=(0, 2))

        default_model = "Camera Only"
        self.model_var = tk.StringVar(value=default_model)
        model_cb = ttk.Combobox(
            ctrl, textvariable=self.model_var,
            values=list(MODEL_OPTIONS.keys()),
            state="readonly", width=28,
        )
        model_cb.grid(row=3, column=0, sticky="ew", pady=(0, 4))

        # Show model description beneath the dropdown
        self.model_desc_var = tk.StringVar(
            value=MODEL_OPTIONS[default_model]["description"])
        self.model_desc_label = ttk.Label(
            ctrl, textvariable=self.model_desc_var, style="Status.TLabel",
            wraplength=250)
        self.model_desc_label.grid(row=4, column=0, sticky="w", pady=(0, 10))
        model_cb.bind("<<ComboboxSelected>>", self._on_model_changed)

        # ── Camera selection ──────────────────────────────────────────
        ttk.Label(ctrl, text="Camera:").grid(
            row=5, column=0, sticky="w", pady=(0, 2))

        cam_labels = [c["label"] for c in _CAMERA_LIST]
        default_cam = "Intel RealSense (Auto)" if "Intel RealSense (Auto)" in cam_labels else (cam_labels[0] if cam_labels else "Webcam 0")
        self.cam_var = tk.StringVar(value=default_cam)
        cam_cb = ttk.Combobox(
            ctrl, textvariable=self.cam_var,
            values=cam_labels,
            state="readonly", width=28,
        )
        cam_cb.grid(row=6, column=0, sticky="ew", pady=(0, 12))

        # ── Start / Stop ────────────────────────────────────────────
        btn_frame = ttk.Frame(ctrl)
        btn_frame.grid(row=7, column=0, sticky="ew", pady=(0, 16))
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        self.start_btn = ttk.Button(btn_frame, text="▶  Start",
                                    command=self._on_start)
        self.start_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self.stop_btn = ttk.Button(btn_frame, text="■  Stop",
                                   command=self._on_stop, state="disabled")
        self.stop_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        # ── Toggle switches ─────────────────────────────────────────
        sep = ttk.Separator(ctrl, orient="horizontal")
        sep.grid(row=8, column=0, sticky="ew", pady=(0, 10))

        ttk.Label(ctrl, text="Fusion View:").grid(
            row=9, column=0, sticky="w", pady=(0, 2))
        self.fusion_view_var = tk.StringVar(value=FusionViewMode.PRODUCTION)
        fusion_view_cb = ttk.Combobox(
            ctrl,
            textvariable=self.fusion_view_var,
            values=[FusionViewMode.PRODUCTION, FusionViewMode.DEBUG],
            state="readonly",
            width=28,
        )
        fusion_view_cb.grid(row=10, column=0, sticky="ew", pady=(0, 10))
        fusion_view_cb.bind("<<ComboboxSelected>>", self._sync_toggles)

        ttk.Label(ctrl, text="Overlays", style="Header.TLabel"
                  ).grid(row=11, column=0, sticky="w", pady=(0, 6))

        self.skel_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="Show Skeleton",
                        variable=self.skel_var,
                        command=self._sync_toggles
                        ).grid(row=12, column=0, sticky="w", pady=2)

        self.pred_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="Show Kalman Predictions",
                        variable=self.pred_var,
                        command=self._sync_toggles
                        ).grid(row=13, column=0, sticky="w", pady=2)

        self.radar_overlay_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="Show Radar Overlay",
                        variable=self.radar_overlay_var,
                        command=self._sync_toggles
                        ).grid(row=14, column=0, sticky="w", pady=2)

        self.radar_points_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="Show Radar Points",
                        variable=self.radar_points_var,
                        command=self._sync_toggles
                        ).grid(row=15, column=0, sticky="w", pady=2)

        self.radar_tracks_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="Show Radar Tracks",
                        variable=self.radar_tracks_var,
                        command=self._sync_toggles
                        ).grid(row=16, column=0, sticky="w", pady=2)

        self.fusion_debug_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="Show Fusion Debug",
                        variable=self.fusion_debug_var,
                        command=self._sync_toggles
                        ).grid(row=17, column=0, sticky="w", pady=2)

        ttk.Button(ctrl, text="Radar Calibrator",
                   command=self._open_calibration_window
                   ).grid(row=18, column=0, sticky="ew", pady=(10, 0))

        # ── Status bar ──────────────────────────────────────────────
        sep2 = ttk.Separator(ctrl, orient="horizontal")
        sep2.grid(row=19, column=0, sticky="ew", pady=(16, 10))

        self.status_var = tk.StringVar(value="Status: Idle")
        ttk.Label(ctrl, textvariable=self.status_var,
                  style="Status.TLabel", wraplength=250
                  ).grid(row=20, column=0, sticky="w")

        self.fps_var = tk.StringVar(value="FPS: --")
        ttk.Label(ctrl, textvariable=self.fps_var,
                  style="Status.TLabel"
                  ).grid(row=21, column=0, sticky="w", pady=(4, 0))

        self.radar_var = tk.StringVar(value="Radar: --")
        ttk.Label(ctrl, textvariable=self.radar_var,
                  style="Status.TLabel", wraplength=250
                  ).grid(row=22, column=0, sticky="w", pady=(4, 0))

        # ── Photo image reference (prevent GC) ──────────────────────
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._display_transform = None

    # ── callbacks ────────────────────────────────────────────────────

    def _on_model_changed(self, _event=None):
        key = self.model_var.get()
        self.model_desc_var.set(MODEL_OPTIONS[key]["description"])

    def _sync_toggles(self):
        self.pipeline.fusion_view_mode = self.fusion_view_var.get()
        self.pipeline.show_skeleton = self.skel_var.get()
        self.pipeline.show_predictions = self.pred_var.get()
        self.pipeline.show_radar_overlay = self.radar_overlay_var.get()
        self.pipeline.show_radar_points = self.radar_points_var.get()
        self.pipeline.show_radar_tracks = self.radar_tracks_var.get()
        self.pipeline.show_fusion_debug = self.fusion_debug_var.get()

    def _open_calibration_window(self):
        if self.calibration_window is not None:
            try:
                if self.calibration_window.window.winfo_exists():
                    self.calibration_window.window.lift()
                    self.calibration_window.sync_from_pipeline()
                    return
            except tk.TclError:
                self.calibration_window = None
        self.calibration_window = RadarCalibrationWindow(self.root, self.pipeline)

    def _on_start(self):
        self.status_var.set("Status: Opening camera...")
        self.root.update_idletasks()

        # Open camera from dropdown selection
        cam_label = self.cam_var.get()
        cam_desc = next(
            (c for c in _CAMERA_LIST if c["label"] == cam_label),
            {"label": "Webcam 0", "type": "webcam", "index": 0, "serial": None},
        )

        if not self.pipeline.open_camera(cam_desc):
            self.status_var.set("Error: Cannot open camera")
            return

        self.status_var.set("Status: Loading model...")
        self.root.update_idletasks()

        # Load model after camera succeeds so bad camera indices fail cleanly
        err = self.pipeline.load_model(self.model_var.get())
        if err:
            print(f"[FALCON] {err}; continuing with camera-only feed")
            self.pipeline.backend = "none"
            self.pipeline.model = None
            self.pipeline.mp_pose = None

        # Sync toggle state
        self._sync_toggles()

        # Start pipeline
        self.pipeline.start()
        if self.calibration_window is not None:
            try:
                if self.calibration_window.window.winfo_exists():
                    self.calibration_window.sync_from_pipeline()
            except tk.TclError:
                self.calibration_window = None

        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        if err:
            self.status_var.set("Status: Running camera only")
        else:
            self.status_var.set("Status: Running")

        # Begin GUI refresh loop
        self._refresh()

    def _on_stop(self):
        self.pipeline.stop()
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_var.set("Status: Stopped")
        self.fps_var.set("FPS: --")
        self.radar_var.set("Radar: --")
        self.video_label.configure(image="", text="No Video Feed")
        self._photo = None
        self._display_transform = None

    def _on_close(self):
        self.pipeline.stop()
        self.root.destroy()

    def _on_video_click(self, event):
        transform = self._display_transform
        if transform is None or not self.pipeline.running:
            self.status_var.set("Status: Start the camera, then click the person center.")
            return

        scale, offset_x, offset_y, frame_w, frame_h = transform
        if scale <= 0:
            return
        u = (float(event.x) - offset_x) / scale
        v = (float(event.y) - offset_y) / scale
        if not (0.0 <= u < frame_w and 0.0 <= v < frame_h):
            self.status_var.set("Status: Calibration click ignored outside the image.")
            return

        message = self.pipeline.set_calibration_click(u, v)
        self.status_var.set(f"Status: {message}")

    # ── GUI refresh (polls the pipeline for new frames) ──────────────

    def _refresh(self):
        if not self.pipeline.running:
            return

        frame = self.pipeline.get_frame()
        if frame is not None:
            # Convert BGR → RGB → PIL → PhotoImage
            # Resize *before* heavier conversions (OpenCV resize is faster than PIL resize)
            lw = self.video_label.winfo_width()
            lh = self.video_label.winfo_height()
            display_frame = frame

            if lw > 1 and lh > 1:
                h, w = frame.shape[:2]  # frame is numpy array
                scale = min(lw / w, lh / h)
                new_w = max(int(w * scale), 1)
                new_h = max(int(h * scale), 1)
                offset_x = (lw - new_w) * 0.5
                offset_y = (lh - new_h) * 0.5
                self._display_transform = (scale, offset_x, offset_y, float(w), float(h))

                if new_w != w or new_h != h:
                    display_frame = _gpu_resize(frame, new_w, new_h)
            else:
                self._display_transform = None

            rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            
            self._photo = ImageTk.PhotoImage(image=pil_img)
            self.video_label.configure(image=self._photo, text="")

            # Update FPS readout
            self.fps_var.set(f"FPS: {self.pipeline.fps:.1f}")
            self.radar_var.set(self.pipeline.radar_status_text())

        self.root.after(self.REFRESH_MS, self._refresh)

    # ── run ──────────────────────────────────────────────────────────

    def run(self):
        self.root.mainloop()


# ── Entry point ──────────────────────────────────────────────────────

def main():
    app = FalconGUI()
    app.run()


if __name__ == "__main__":
    main()
