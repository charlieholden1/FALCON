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

import threading
import time
import tkinter as tk
from tkinter import ttk
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from occlusion import OcclusionState
from tracking import TrackingManager
from camera_stream import DualStreamCamera

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

# ── Hardware-accelerated resize helper ───────────────────────────────

try:
    _cuda_resize = cv2.cuda.resize  # type: ignore[attr-defined]
    _HAS_CUDA_RESIZE = True
except AttributeError:
    _HAS_CUDA_RESIZE = False


def _gpu_resize(frame: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """Resize using cv2.cuda when available, else CPU INTER_NEAREST."""
    if _HAS_CUDA_RESIZE:
        gpu_mat = cv2.cuda.GpuMat()
        gpu_mat.upload(frame)
        gpu_out = cv2.cuda.resize(gpu_mat, (new_w, new_h),
                                  interpolation=cv2.INTER_NEAREST)
        return gpu_out.download()
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

                self.model = YOLO(cfg["path"], task="pose")

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

    def start(self):
        if self._running:
            return
        self._running = True
        self.tracker = TrackingManager(
            iou_threshold=0.3,
            max_frames_lost=90,
            prediction_decay=0.90,
            recovery_duration=90,
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

    # ── main vision loop (background thread) ─────────────────────────

    def _loop(self):
        while self._running:
            if self._webcam is None:
                time.sleep(0.01)
                continue

            frame, depth_frame = self._webcam.read()
            if frame is None:
                continue

            h_frame, w_frame = frame.shape[:2]
            self._skip_counter += 1

            # Decide whether to run the detector this frame
            run_detect = (
                self._burst_remaining > 0
                or self._skip_counter % self.skip_frames == 0
            )

            if run_detect:
                # ── full detection frame ─────────────────────────────
                det_boxes, det_confs, det_keypoints = self._detect(
                    frame, w_frame, h_frame,
                )
                tracks = self.tracker.update(
                    det_boxes, det_keypoints, det_confs,
                    frame=frame, depth_frame=depth_frame,
                )

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
                tracks = self.tracker.propagate_only()

            # ── annotate ─────────────────────────────────────────────
            vis = frame.copy()
            self._annotate(vis, tracks)

            # ── FPS ──────────────────────────────────────────────────
            self._frame_count += 1
            elapsed = time.time() - self._fps_timer
            if elapsed >= 0.5:
                self._fps_value = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_timer = time.time()

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
        results = self.model(frame, verbose=False, half=self._yolo_half)
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

        # ── Info overlay (top-left) ─────────────────────────────────
        overlay_lines = [
            f"FPS: {self._fps_value:.1f}",
            f"Tracked: {len(tracks)}",
            f"Occluded: {n_occluded}",
            f"Predicted: {n_predicted}",
            f"Recovered: {n_recovered}",
            f"Depth: {n_depth}/{len(tracks)}",
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

    REFRESH_MS = 30  # ~33 FPS GUI refresh

    def __init__(self):
        self.pipeline = VisionPipeline()

        # ── Root window ─────────────────────────────────────────────
        self.root = tk.Tk()
        self.root.title("F.A.L.C.O.N. Vision System")
        self.root.configure(bg="#1e1e2e")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.minsize(960, 540)

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

        self.model_var = tk.StringVar(value="YOLO26 Nano (Default)")
        model_cb = ttk.Combobox(
            ctrl, textvariable=self.model_var,
            values=list(MODEL_OPTIONS.keys()),
            state="readonly", width=28,
        )
        model_cb.grid(row=3, column=0, sticky="ew", pady=(0, 4))

        # Show model description beneath the dropdown
        self.model_desc_var = tk.StringVar(
            value=MODEL_OPTIONS["YOLO26 Nano (Default)"]["description"])
        self.model_desc_label = ttk.Label(
            ctrl, textvariable=self.model_desc_var, style="Status.TLabel",
            wraplength=250)
        self.model_desc_label.grid(row=4, column=0, sticky="w", pady=(0, 10))
        model_cb.bind("<<ComboboxSelected>>", self._on_model_changed)

        # ── Camera selection ──────────────────────────────────────────
        ttk.Label(ctrl, text="Camera:").grid(
            row=5, column=0, sticky="w", pady=(0, 2))

        cam_labels = [c["label"] for c in _CAMERA_LIST]
        self.cam_var = tk.StringVar(value=cam_labels[0] if cam_labels else "Webcam 0")
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

        # ── Status bar ──────────────────────────────────────────────
        sep2 = ttk.Separator(ctrl, orient="horizontal")
        sep2.grid(row=13, column=0, sticky="ew", pady=(16, 10))

        self.status_var = tk.StringVar(value="Status: Idle")
        ttk.Label(ctrl, textvariable=self.status_var,
                  style="Status.TLabel", wraplength=250
                  ).grid(row=14, column=0, sticky="w")

        self.fps_var = tk.StringVar(value="FPS: --")
        ttk.Label(ctrl, textvariable=self.fps_var,
                  style="Status.TLabel"
                  ).grid(row=15, column=0, sticky="w", pady=(4, 0))

        # ── Photo image reference (prevent GC) ──────────────────────
        self._photo: Optional[ImageTk.PhotoImage] = None

    # ── callbacks ────────────────────────────────────────────────────

    def _on_model_changed(self, _event=None):
        key = self.model_var.get()
        self.model_desc_var.set(MODEL_OPTIONS[key]["description"])

    def _sync_toggles(self):
        self.pipeline.show_skeleton = self.skel_var.get()
        self.pipeline.show_predictions = self.pred_var.get()

    def _on_start(self):
        self.status_var.set("Status: Loading model...")
        self.root.update_idletasks()

        # Load model
        err = self.pipeline.load_model(self.model_var.get())
        if err:
            self.status_var.set(f"Error: {err}")
            return

        # Open camera from dropdown selection
        cam_label = self.cam_var.get()
        cam_desc = next(
            (c for c in _CAMERA_LIST if c["label"] == cam_label),
            {"label": "Webcam 0", "type": "webcam", "index": 0, "serial": None},
        )

        if not self.pipeline.open_camera(cam_desc):
            self.status_var.set("Error: Cannot open camera")
            return

        # Sync toggle state
        self._sync_toggles()

        # Start pipeline
        self.pipeline.start()

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

