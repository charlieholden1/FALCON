"""
F.A.L.C.O.N. Vision System – Tkinter GUI Entry Point
======================================================

Wraps the existing vision pipeline (tracking, occlusion, prediction,
radar) in a lightweight Tkinter GUI with:

* Model selection dropdown  (YOLO26 Nano / YOLO26 Large / MediaPipe Pose)
* Start / Stop controls
* Live toggle switches      (skeleton, radar, Kalman predictions)
* Embedded OpenCV video feed rendered inside a Tkinter Label
* Background threading so the GUI never freezes
"""

from __future__ import annotations

import sys
import threading
import time
import tkinter as tk
from tkinter import ttk
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from occlusion import OcclusionState
from radar import CameraProjection, MockRadar
from tracking import TrackingManager

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
    x1, y1 = pt1
    x2, y2 = pt2
    edges = [
        ((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1)),
    ]
    for (sx, sy), (ex, ey) in edges:
        dist = int(np.hypot(ex - sx, ey - sy))
        pts_x = np.linspace(sx, ex, max(dist // dash_len, 2)).astype(int)
        pts_y = np.linspace(sy, ey, max(dist // dash_len, 2)).astype(int)
        for i in range(0, len(pts_x) - 1, 2):
            cv2.line(img, (pts_x[i], pts_y[i]),
                     (pts_x[i + 1], pts_y[i + 1]), colour, thickness)


def _draw_uncertainty_ellipse(img, center, sigma_x, sigma_y, colour):
    axes = (max(int(2 * sigma_x), 2), max(int(2 * sigma_y), 2))
    cv2.ellipse(img, center, axes, 0, 0, 360, colour, 1, cv2.LINE_AA)


def _draw_trail(img, history, colour, max_len=TRAIL_LENGTH):
    pts = history[-max_len:]
    if len(pts) < 2:
        return
    for i in range(1, len(pts)):
        cx1 = int((pts[i - 1][0] + pts[i - 1][2]) / 2)
        cy1 = int((pts[i - 1][1] + pts[i - 1][3]) / 2)
        cx2 = int((pts[i][0] + pts[i][2]) / 2)
        cy2 = int((pts[i][1] + pts[i][3]) / 2)
        alpha = i / len(pts)
        fade = tuple(int(c * alpha) for c in colour)
        cv2.line(img, (cx1, cy1), (cx2, cy2), fade, 2, cv2.LINE_AA)


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


# ── Vision Pipeline (runs in background thread) ─────────────────────

class VisionPipeline:
    """
    Encapsulates model loading, webcam capture, detection, tracking,
    and frame annotation.  Designed to run in a background thread.
    """

    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.model = None
        self.mp_pose = None              # MediaPipe Pose instance
        self.backend: str = "yolo"       # "yolo" or "mediapipe"

        self.tracker = TrackingManager(
            iou_threshold=0.3,
            max_frames_lost=90,
            prediction_decay=0.90,
            recovery_duration=90,
        )
        self.mock_radar = MockRadar()
        self.cam_proj = CameraProjection()

        # Toggles (read by the render loop, written by the GUI)
        self.show_skeleton = True
        self.show_radar = True
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
                print(f"[FALCON] Loading YOLO model on {device.upper()}...")

                self.model = YOLO(cfg["path"])

                # Only move PyTorch models to device; TensorRT engines are device-bound by export
                if not cfg["path"].endswith(".engine"):
                    self.model.to(device)
                
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

    def open_camera(self, index: int = 0) -> bool:
        """
        Opens camera with optimized settings.
        Attempt 1: GStreamer (best for CSI cameras on Jetson)
        Attempt 2: V4L2 with MJPG (best for USB cameras)
        """
        print(f"[FALCON] Attempting to open camera {index}...")

        # 1. Try GStreamer pipeline (Standard for Jetson CSI cameras like IMX219)
        # This pipeline converts NVMM (hardware memory) to BGR for OpenCV
        gst_str = (
            f"nvarguscamerasrc sensor-id={index} ! "
            "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
            "nvvidconv flip-method=0 ! "
            "video/x-raw, width=640, height=480, format=BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! appsink drop=1"
        )
        
        try:
            # We only try GStreamer if it looks like we might be on a Jetson with CSI
            # But since user is having issues, we can try opening default first, 
            # and if that's slow/bad, checking capabilities.
            # actually, let's stick to the V4L2 fix first which is safer for index 0
            pass 
        except:
            pass

        # 2. Standard V4L2 (USB) with FORCE MJPG
        # This fixes the 8 FPS / 15 FPS limit on most USB cams
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        
        if not self.cap.isOpened():
            # Fallback to default backend if V4L2 fails
            self.cap = cv2.VideoCapture(index)

        if self.cap.isOpened():
            # Force MJPG - Critical for USB 2.0/3.0 bandwidth
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            # Logitech C270 specifc settings for 30FPS
            # 1 = Manual Exposure, 3 = Auto Exposure (V4L2 standard)
            # 0.25 is sometimes used in older bindings, but 1 is usually safer.
            try:
                # Disable auto exposure
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) 
                
                # Set manual exposure value (100-200 is typical for 30fps indoor)
                # Increasing to 450 to brighten image while trying to hold 30fps.
                # If FPS drops to 15, we hit the limit of 33ms shutter speed.
                self.cap.set(cv2.CAP_PROP_EXPOSURE, 450)
                
                # Also bump the Gain (digital amplification)
                # This helps brightness without slowing down the frame rate
                self.cap.set(cv2.CAP_PROP_GAIN, 128)
            except:
                pass
            
            # Logging actual obtained settings
            actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            mode = self.cap.get(cv2.CAP_PROP_FOURCC)
            print(f"[FALCON] Camera {index} active: {actual_w}x{actual_h} @ {actual_fps}FPS")
            
        return self.cap.isOpened()

    def close_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    # ── start / stop ─────────────────────────────────────────────────

    def start(self):
        if self._running:
            return
        self._running = True
        # Reset tracker state for a fresh session
        self.tracker = TrackingManager(
            iou_threshold=0.3,
            max_frames_lost=90,
            prediction_decay=0.90,
            recovery_duration=90,
        )
        self._fps_timer = time.time()
        self._frame_count = 0
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
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.01)
                continue

            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            h_frame, w_frame = frame.shape[:2]

            # ── detection ────────────────────────────────────────────
            det_boxes, det_confs, det_keypoints = self._detect(frame, w_frame, h_frame)

            # ── tracking ─────────────────────────────────────────────
            tracks = self.tracker.update(det_boxes, det_keypoints, det_confs)

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
        results = self.model(frame, verbose=False)
        result = results[0]

        det_boxes = np.empty((0, 4))
        det_confs = np.empty(0)
        det_keypoints = None

        if result.boxes is not None and len(result.boxes) > 0:
            # OPTIMIZATION: Process filtering on GPU to reduce transfer volume
            boxes = result.boxes
            # boxes.conf and boxes.xyxy keep data on device (cuda:0)
            
            # Create a boolean mask on GPU
            mask = boxes.conf >= CONFIDENCE_THRESHOLD
            
            # Apply mask on GPU tensors (much faster!)
            filtered_boxes = boxes.xyxy[mask]
            filtered_confs = boxes.conf[mask]

            # Only transfer the filtered results to CPU
            if len(filtered_boxes) > 0:
                det_boxes = filtered_boxes.cpu().numpy()
                det_confs = filtered_confs.cpu().numpy()

                if result.keypoints is not None:
                    # Filter keypoints on GPU too
                    kp_data = result.keypoints.data
                    if len(kp_data) > 0:
                        filtered_kps = kp_data[mask]
                        det_keypoints = filtered_kps.cpu().numpy()

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
        radar_active = False
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
            cv2.putText(vis, label, (x1, y1 - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)
            cv2.putText(vis, occ_label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)

            # ── Radar crosshair for heavily occluded / lost ──────
            if self.show_radar and t.occlusion_state in (
                OcclusionState.HEAVILY_OCCLUDED,
                OcclusionState.LOST,
            ):
                pt3d = self.mock_radar.get_target_3d()
                u, v = self.cam_proj.project_3d_to_2d(pt3d)
                if 0 <= u < w_frame and 0 <= v < h_frame:
                    radar_colour = (255, 0, 255)
                    cross_size = 20
                    cv2.line(vis, (u - cross_size, v), (u + cross_size, v),
                             radar_colour, 2, cv2.LINE_AA)
                    cv2.line(vis, (u, v - cross_size), (u, v + cross_size),
                             radar_colour, 2, cv2.LINE_AA)
                    cv2.circle(vis, (u, v), cross_size, radar_colour, 2, cv2.LINE_AA)
                    rw, rh = 40, 80
                    cv2.rectangle(vis, (u - rw, v - rh), (u + rw, v + rh),
                                  radar_colour, 2)
                    cv2.putText(vis, f"RADAR TARGET #{t.track_id}",
                                (u - rw, v - rh - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                radar_colour, 2, cv2.LINE_AA)

            if t.occlusion_state in (
                OcclusionState.PARTIALLY_OCCLUDED,
                OcclusionState.HEAVILY_OCCLUDED,
                OcclusionState.LOST,
            ):
                n_occluded += 1
                radar_active = True

        # ── Info overlay (top-left) ─────────────────────────────────
        overlay_lines = [
            f"FPS: {self._fps_value:.1f}",
            f"Tracked: {len(tracks)}",
            f"Occluded: {n_occluded}",
            f"Predicted: {n_predicted}",
            f"Recovered: {n_recovered}",
        ]
        for i, line in enumerate(overlay_lines):
            cv2.putText(vis, line, (10, 25 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA)

        # ── RADAR ACTIVE banner ─────────────────────────────────────
        if radar_active and self.show_radar:
            banner = "RADAR ACTIVE"
            banner_colour = (0, 0, 255)
            (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            bx = vis.shape[1] - tw - 20
            by = 40
            overlay = vis.copy()
            cv2.rectangle(overlay, (bx - 10, by - th - 10),
                          (bx + tw + 10, by + 10), banner_colour, -1)
            cv2.addWeighted(overlay, 0.5, vis, 0.5, 0, vis)
            cv2.putText(vis, banner, (bx, by),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 255, 255), 2, cv2.LINE_AA)
        elif not radar_active:
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
    │   (Tkinter Label)     │  ☑ Mock Radar          │
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

        # ── Webcam index ────────────────────────────────────────────
        ttk.Label(ctrl, text="Webcam Index:").grid(
            row=5, column=0, sticky="w", pady=(0, 2))

        self.cam_var = tk.StringVar(value="0")
        cam_entry = ttk.Entry(ctrl, textvariable=self.cam_var, width=6)
        cam_entry.grid(row=6, column=0, sticky="w", pady=(0, 12))

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

        self.radar_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="Enable Mock Radar",
                        variable=self.radar_var,
                        command=self._sync_toggles
                        ).grid(row=11, column=0, sticky="w", pady=2)

        self.pred_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="Show Kalman Predictions",
                        variable=self.pred_var,
                        command=self._sync_toggles
                        ).grid(row=12, column=0, sticky="w", pady=2)

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
        self.pipeline.show_radar = self.radar_var.get()
        self.pipeline.show_predictions = self.pred_var.get()

    def _on_start(self):
        self.status_var.set("Status: Loading model...")
        self.root.update_idletasks()

        # Load model
        err = self.pipeline.load_model(self.model_var.get())
        if err:
            self.status_var.set(f"Error: {err}")
            return

        # Open camera
        try:
            cam_idx = int(self.cam_var.get())
        except ValueError:
            cam_idx = 0

        if not self.pipeline.open_camera(cam_idx):
            self.status_var.set("Error: Cannot open webcam")
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
                
                # Use INTER_LINEAR or NEAREST for speed (LANCZOS/CUBIC is too slow)
                if new_w != w or new_h != h:
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

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
