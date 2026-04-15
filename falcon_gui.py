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
import logging
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from occlusion import OcclusionState
from tracking import TrackingManager
from camera_stream import DualStreamCamera
from radar import CameraProjection, IWR6843Driver, MockRadar

logger = logging.getLogger("falcon.gui")

# Discover available cameras once at import time
_CAMERA_LIST = DualStreamCamera.discover_cameras()

# ── Model catalogue ─────────────────────────────────────────────────

MODEL_OPTIONS = {
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
            "draw_ms": 0.0,
        }

        # Radar & projection 
        self._radar = None               # IWR6843Driver or MockRadar
        self._radar_backup = None        # stored ref when toggled off
        self._radar_init_thread: Optional[threading.Thread] = None
        self._radar_status_message = "Radar: idle"
        self.projection: Optional[CameraProjection] = None
        self.radar_enabled = True
        self.show_radar_overlay = True

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
                
                self.mp_pose = None
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
                self.model = None
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

        is_rs = cam_descriptor["type"] == "realsense"
        idx = cam_descriptor.get("index", 0) or 0
        serial = cam_descriptor.get("serial")
        label = cam_descriptor["label"]
        print(f"[FALCON] Opening camera: {label}")

        self._webcam = DualStreamCamera(
            webcam_index=idx,
            use_realsense=is_rs,
            realsense_serial=serial,
        )
        return self._webcam.is_opened()

    def close_camera(self):
        if self._webcam is not None:
            self._webcam.stop()
            self._webcam = None

    # ── start / stop ─────────────────────────────────────────────────

    def _init_radar(self) -> None:
        """Initialize the projection immediately and radar asynchronously."""
        self._init_projection_from_camera()
        self._radar_status_message = "Radar: scanning..."
        self._radar_init_thread = threading.Thread(
            target=self._init_radar_worker,
            daemon=True,
            name="radar-init",
        )
        self._radar_init_thread.start()

    def _init_projection_from_camera(self) -> None:
        """Build the camera projection from the current camera intrinsics."""
        K = None
        if (self._webcam is not None
                and hasattr(self._webcam, '_rs_pipeline')
                and self._webcam._rs_pipeline is not None):
            try:
                import pyrealsense2 as rs
                profile = self._webcam._rs_pipeline.get_active_profile()
                intr = (profile.get_stream(rs.stream.color)
                        .as_video_stream_profile().get_intrinsics())
                K = np.array([
                    [intr.fx, 0.0, intr.ppx],
                    [0.0, intr.fy, intr.ppy],
                    [0.0, 0.0, 1.0],
                ], dtype=np.float64)
                logger.info("Using RealSense intrinsics for projection")
            except Exception:
                pass
        self.projection = CameraProjection(K=K)
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
            driver = IWR6843Driver(config_port=config_port, data_port=data_port)
            if not driver.open():
                logger.warning("Radar open failed for config=%s data=%s", config_port, data_port)
                continue

            if driver.wait_for_frame(timeout_s=6.0):
                logger.info("IWR6843 radar connected on config=%s data=%s", config_port, data_port)
                self._radar = driver
                self._radar_backup = driver
                self._radar_status_message = f"Radar: connected {config_port} {data_port}"
                if self.tracker is not None and self.radar_enabled:
                    self.tracker.radar = driver
                    self.tracker.projection = self.projection
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

    def toggle_radar(self, enabled: bool) -> None:
        """Enable or disable radar fusion at runtime."""
        self.radar_enabled = enabled
        if enabled and self._radar_backup is not None:
            self.tracker.radar = self._radar_backup
        else:
            self.tracker.radar = None

    def start(self):
        if self._running:
            return
        self._running = True

        # Init radar after camera is open
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
        if self._webcam is not None:
            self._webcam.start()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        # Close radar
        if self._radar is not None:
            self._radar.close()
            self._radar = None
            self._radar_backup = None
        self._radar_status_message = "Radar: idle"
        self.close_camera()
        with self._lock:
            self._latest_frame = None

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

    def radar_status_text(self) -> str:
        radar = self._radar_backup if self._radar_backup is not None else self._radar
        if radar is None:
            return self._radar_status_message

        points = self.radar_points()
        diagnostics = getattr(radar, "diagnostics", {})
        if isinstance(radar, MockRadar):
            return f"Radar: mock ({len(points)} pts)"

        frames = diagnostics.get("frames", 0)
        fps = diagnostics.get("fps", 0.0)
        last_error = diagnostics.get("last_parse_error", "")
        packet_size = diagnostics.get("last_packet_size", 0)
        rx_bytes = diagnostics.get("rx_bytes", 0)
        magic_hits = diagnostics.get("magic_hits", 0)
        last_cmd_error = diagnostics.get("last_command_error", "")
        if last_cmd_error:
            return f"Radar cmd error: {last_cmd_error}"
        if last_error:
            return (
                f"Radar: {len(points)} pts  frames {frames}  "
                f"rx {rx_bytes}  magic {magic_hits}  pkt {packet_size}  parse {last_error}"
            )
        return (
            f"Radar: {len(points)} pts  frames {frames}  "
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
                track_t1 = time.time()

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
            self._timing["draw_ms"] = (draw_t1 - draw_t0) * 1000.0

            with self._lock:
                self._latest_frame = vis

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

        return det_boxes, det_confs, det_keypoints

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

        return det_boxes, det_confs, det_keypoints

    # ── annotation ───────────────────────────────────────────────────

    def _annotate(self, vis, tracks):
        n_occluded = 0
        n_predicted = 0
        n_recovered = 0
        n_depth = 0
        h_frame, w_frame = vis.shape[:2]

        for t in tracks:
            colour = t.occlusion_state.colour_bgr
            x1, y1, x2, y2 = t.bbox.astype(int)

            # Trajectory trail
            _draw_trail(vis, t.bbox_history, colour)

            if t.is_predicted:
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
                occ_label += f' [RADAR {t.radar_confidence:.0%}]'
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
        if self.show_radar_overlay and self.projection is not None:
            radar_points = self.radar_points()
            self._draw_radar_overlay(vis, radar_points)

        # ── Info overlay (top-left) ─────────────────────────────────
        overlay_lines = [
            f"FPS: {self._fps_value:.1f}",
            f"Tracked: {len(tracks)}",
            f"Occluded: {n_occluded}",
            f"Predicted: {n_predicted}",
            f"Recovered: {n_recovered}",
            f"Depth: {n_depth}/{len(tracks)}",
            f"Radar Points: {len(radar_points)}",
            f"Radar Fusion: {sum(1 for t in tracks if getattr(t, 'using_radar', False))}/{len(tracks)}",
            f"Timing ms C:{self._timing['cam_ms']:.0f} D:{self._timing['detect_ms']:.0f} T:{self._timing['track_ms']:.0f} R:{self._timing['draw_ms']:.0f}",
        ]
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
        self.window.geometry("720x520")
        self.window.resizable(False, True)

        self.status_var = tk.StringVar(
            value=f"Save path: {CameraProjection.DEFAULT_PATH}"
        )
        self._vars = {}
        self._suspend_callbacks = False

        container = ttk.Frame(self.window)
        container.pack(fill="both", expand=True, padx=10, pady=10)

        ttk.Label(
            container,
            text="Radar To Camera Calibration",
            style="Header.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            container,
            text="Adjust the projected radar cloud until the dots sit on people in the camera view.",
            style="Status.TLabel",
            wraplength=660,
        ).pack(anchor="w", pady=(0, 10))

        grid = ttk.Frame(container)
        grid.pack(fill="both", expand=True)
        grid.columnconfigure(2, weight=1)

        for row_idx, (key, label, low, high, resolution) in enumerate(self.PARAM_SPECS):
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

        btn_row = ttk.Frame(container)
        btn_row.pack(fill="x", pady=(10, 6))
        ttk.Button(btn_row, text="Load Saved", command=self._load).pack(side="left")
        ttk.Button(btn_row, text="Save", command=self._save).pack(side="left", padx=6)
        ttk.Button(btn_row, text="Reset", command=self._reset).pack(side="left")
        ttk.Button(btn_row, text="Sync From Live", command=self.sync_from_pipeline).pack(
            side="left", padx=6
        )

        ttk.Label(
            container,
            textvariable=self.status_var,
            style="Status.TLabel",
            wraplength=660,
        ).pack(anchor="w")

        self.window.protocol("WM_DELETE_WINDOW", self.window.destroy)
        self.sync_from_pipeline()

    def sync_from_pipeline(self) -> None:
        params = self.pipeline.projection_params()
        self._suspend_callbacks = True
        try:
            for key, var in self._vars.items():
                if key in params:
                    var.set(params[key])
        finally:
            self._suspend_callbacks = False

    def _on_change(self, *_args) -> None:
        if self._suspend_callbacks:
            return
        params = {key: var.get() for key, var in self._vars.items()}
        self.pipeline.apply_projection_params(**params)
        self.status_var.set(
            "Live calibration applied. Tune until radar dots align with people."
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
        self.status_var.set("Calibration reset to defaults")


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

        default_model = (
            "YOLO26 Nano (TensorRT)"
            if Path("yolo26n-pose.engine").exists()
            else "YOLO26 Nano (Default)"
        )
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

        ttk.Label(ctrl, text="Overlays", style="Header.TLabel"
                  ).grid(row=9, column=0, sticky="w", pady=(0, 6))

        self.skel_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="Show Skeleton",
                        variable=self.skel_var,
                        command=self._sync_toggles
                        ).grid(row=10, column=0, sticky="w", pady=2)

        self.pred_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="Show Kalman Predictions",
                        variable=self.pred_var,
                        command=self._sync_toggles
                        ).grid(row=11, column=0, sticky="w", pady=2)

        self.radar_overlay_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="Show Radar Overlay",
                        variable=self.radar_overlay_var,
                        command=self._sync_toggles
                        ).grid(row=12, column=0, sticky="w", pady=2)

        ttk.Button(ctrl, text="Radar Calibrator",
                   command=self._open_calibration_window
                   ).grid(row=13, column=0, sticky="ew", pady=(10, 0))

        # ── Status bar ──────────────────────────────────────────────
        sep2 = ttk.Separator(ctrl, orient="horizontal")
        sep2.grid(row=14, column=0, sticky="ew", pady=(16, 10))

        self.status_var = tk.StringVar(value="Status: Idle")
        ttk.Label(ctrl, textvariable=self.status_var,
                  style="Status.TLabel", wraplength=250
                  ).grid(row=15, column=0, sticky="w")

        self.fps_var = tk.StringVar(value="FPS: --")
        ttk.Label(ctrl, textvariable=self.fps_var,
                  style="Status.TLabel"
                  ).grid(row=16, column=0, sticky="w", pady=(4, 0))

        self.radar_var = tk.StringVar(value="Radar: --")
        ttk.Label(ctrl, textvariable=self.radar_var,
                  style="Status.TLabel", wraplength=250
                  ).grid(row=17, column=0, sticky="w", pady=(4, 0))

        # ── Photo image reference (prevent GC) ──────────────────────
        self._photo: Optional[ImageTk.PhotoImage] = None

    # ── callbacks ────────────────────────────────────────────────────

    def _on_model_changed(self, _event=None):
        key = self.model_var.get()
        self.model_desc_var.set(MODEL_OPTIONS[key]["description"])

    def _sync_toggles(self):
        self.pipeline.show_skeleton = self.skel_var.get()
        self.pipeline.show_predictions = self.pred_var.get()
        self.pipeline.show_radar_overlay = self.radar_overlay_var.get()

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
            self.pipeline.close_camera()
            self.status_var.set(f"Error: {err}")
            return

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

    def _on_close(self):
        self.pipeline.stop()
        self.root.destroy()

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

            if lw > 1 and lh > 1:
                h, w = frame.shape[:2]  # frame is numpy array
                scale = min(lw / w, lh / h)
                new_w = max(int(w * scale), 1)
                new_h = max(int(h * scale), 1)
                
                if new_w != w or new_h != h:
                    frame = _gpu_resize(frame, new_w, new_h)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
