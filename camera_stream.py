"""
F.A.L.C.O.N. – Dual-Stream Camera Module
==========================================

Provides a unified camera interface that attempts to use an Intel
RealSense depth camera (colour + aligned depth) and falls back to a
standard USB webcam when RealSense hardware is unavailable.

Public API
----------
DualStreamCamera
    read()  → (colour_bgr, depth_uint16 | None)
get_depth_meters(depth_frame, x, y) → float | None
"""

from __future__ import annotations

import queue
import sys
import threading
from typing import Optional, Tuple

import cv2
import numpy as np

# Try to import RealSense SDK
try:
    import pyrealsense2 as rs
    _HAS_REALSENSE = True
except ImportError:
    _HAS_REALSENSE = False


def get_depth_meters(
    depth_frame: Optional[np.ndarray],
    x: int,
    y: int,
) -> Optional[float]:
    """
    Query depth at pixel (x, y) from a RealSense depth frame.

    Parameters
    ----------
    depth_frame : ndarray | None
        Single-channel uint16 array where each pixel holds distance in
        **millimetres** from the sensor.
    x, y : int
        Pixel coordinates to query.

    Returns
    -------
    float | None
        Distance in metres, or ``None`` if the depth frame is missing
        or the sensor returned zero (invalid reading).
    """
    if depth_frame is None:
        return None
    h, w = depth_frame.shape[:2]
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    raw = int(depth_frame[y, x])
    if raw == 0:
        return None
    return raw / 1000.0


class DualStreamCamera:
    """
    Unified camera that tries Intel RealSense (colour + aligned depth)
    first and falls back to a threaded USB webcam capture.

    Parameters
    ----------
    webcam_index : int
        OpenCV camera index used when RealSense is unavailable.
    width, height : int
        Requested stream resolution.
    fps : int
        Requested frame rate.
    q_size : int
        Queue depth for the webcam fallback reader thread.
    """

    def __init__(
        self,
        webcam_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        q_size: int = 2,
    ):
        self._rs_pipeline = None
        self._rs_align = None
        self._using_realsense = False

        self._cap: Optional[cv2.VideoCapture] = None
        self._q: Optional[queue.Queue] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        self._q_size = q_size
        self._opened = False

        # Try RealSense first
        if _HAS_REALSENSE:
            try:
                self._init_realsense(width, height, fps)
                return
            except Exception as exc:
                print(f"[FALCON] RealSense not available: {exc}")

        # Fall back to webcam
        self._init_webcam(webcam_index, width, height, fps)

    # ── RealSense initialisation ────────────────────────────────────

    def _init_realsense(self, w: int, h: int, fps: int) -> None:
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        pipe.start(cfg)
        self._rs_pipeline = pipe
        self._rs_align = rs.align(rs.stream.color)
        self._using_realsense = True
        self._opened = True
        print(f"[FALCON] RealSense camera active: {w}x{h} @ {fps}FPS")

    # ── Webcam fallback ─────────────────────────────────────────────

    def _init_webcam(self, index: int, w: int, h: int, fps: int) -> None:
        if sys.platform == "linux":
            cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap = cv2.VideoCapture(index)
        elif sys.platform == "win32":
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(index)
        else:
            cap = cv2.VideoCapture(index)

        if cap.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_FPS, fps)
            try:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                cap.set(cv2.CAP_PROP_EXPOSURE, 320)
                cap.set(cv2.CAP_PROP_GAIN, 200)
            except Exception:
                pass
            actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"[FALCON] Webcam fallback active: {actual_w}x{actual_h} @ {actual_fps}FPS")

        self._cap = cap
        self._q = queue.Queue(maxsize=self._q_size)
        self._opened = cap.isOpened()

    # ── start / stop ────────────────────────────────────────────────

    def start(self) -> "DualStreamCamera":
        """Start capturing.  For webcam mode this launches the reader thread."""
        if self._using_realsense:
            # RealSense pipeline is already active from __init__
            return self
        self._running = True
        self._thread = threading.Thread(target=self._webcam_reader, daemon=True)
        self._thread.start()
        return self

    def _webcam_reader(self) -> None:
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                continue
            if self._q.full():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
            self._q.put(frame)

    def is_opened(self) -> bool:
        return self._opened

    @property
    def has_depth(self) -> bool:
        """True when the camera supplies native depth frames."""
        return self._using_realsense

    # ── read ────────────────────────────────────────────────────────

    def read(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Return ``(colour_bgr, depth_uint16)`` or ``(colour_bgr, None)``
        for the webcam fallback.

        *depth_uint16* is a single-channel uint16 array where each pixel
        holds the distance in **millimetres** from the sensor.
        """
        if self._using_realsense:
            return self._read_realsense()
        return self._read_webcam()

    def _read_realsense(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            frames = self._rs_pipeline.wait_for_frames(timeout_ms=100)
        except Exception:
            return None, None
        aligned = self._rs_align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame:
            return None, None
        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data()) if depth_frame else None
        return color, depth

    def _read_webcam(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            frame = self._q.get(timeout=0.02)
            return frame, None
        except queue.Empty:
            return None, None

    # ── cleanup ─────────────────────────────────────────────────────

    def stop(self) -> None:
        if self._using_realsense:
            if self._rs_pipeline is not None:
                try:
                    self._rs_pipeline.stop()
                except Exception:
                    pass
                self._rs_pipeline = None
        else:
            self._running = False
            if self._thread is not None:
                self._thread.join(timeout=2.0)
                self._thread = None
            if self._cap is not None:
                self._cap.release()
                self._cap = None
        self._opened = False
