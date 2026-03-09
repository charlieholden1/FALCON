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
    Unified camera that opens either an Intel RealSense depth camera or
    a standard USB webcam, depending on the *use_realsense* flag.

    Parameters
    ----------
    webcam_index : int
        OpenCV camera index (only used when *use_realsense* is False).
    width, height : int
        Requested stream resolution.
    fps : int
        Requested frame rate.
    q_size : int
        Queue depth for the webcam reader thread.
    use_realsense : bool
        If True, open an Intel RealSense device instead of a webcam.
    realsense_serial : str | None
        Optional serial number to target a specific RealSense device.
    """

    def __init__(
        self,
        webcam_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        q_size: int = 2,
        use_realsense: bool = False,
        realsense_serial: Optional[str] = None,
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

        if use_realsense and _HAS_REALSENSE:
            try:
                self._init_realsense(width, height, fps, realsense_serial)
                return
            except Exception as exc:
                print(f"[FALCON] RealSense failed to start: {exc}")

        # Webcam path (either by choice or RealSense fallback)
        self._init_webcam(webcam_index, width, height, fps)

    # ── device discovery ────────────────────────────────────────────

    @staticmethod
    def discover_cameras() -> list:
        """
        Return a list of available camera descriptors.

        Each entry is a dict with keys:
            label  : str  – human-readable name for the GUI
            type   : str  – ``'realsense'`` or ``'webcam'``
            index  : int | None – webcam index, or None for RealSense
            serial : str | None – RealSense serial, or None for webcams
        """
        cameras: list = []

        # Discover RealSense devices
        if _HAS_REALSENSE:
            try:
                ctx = rs.context()
                for dev in ctx.devices:
                    name = dev.get_info(rs.camera_info.name)
                    serial = dev.get_info(rs.camera_info.serial_number)
                    cameras.append({
                        "label": f"{name} (Depth)",
                        "type": "realsense",
                        "index": None,
                        "serial": serial,
                    })
            except Exception:
                pass

        # Offer webcam indices 0-9
        for i in range(10):
            cameras.append({
                "label": f"Webcam {i}",
                "type": "webcam",
                "index": i,
                "serial": None,
            })

        return cameras

    # ── RealSense initialisation ────────────────────────────────────

    # RealSense stream resolution defaults
    _RS_COLOR_W: int = 640
    _RS_COLOR_H: int = 480
    _RS_DEPTH_W: int = 640
    _RS_DEPTH_H: int = 480

    def _init_realsense(
        self, w: int, h: int, fps: int, serial: Optional[str] = None,
    ) -> None:
        pipe = rs.pipeline()
        cfg = rs.config()
        if serial:
            cfg.enable_device(serial)

        # High-res colour for better YOLO pose output
        cfg.enable_stream(
            rs.stream.color,
            self._RS_COLOR_W, self._RS_COLOR_H,
            rs.format.bgr8, fps,
        )
        # Standard-res depth (more reliable than matching colour res)
        cfg.enable_stream(
            rs.stream.depth,
            self._RS_DEPTH_W, self._RS_DEPTH_H,
            rs.format.z16, fps,
        )

        profile = pipe.start(cfg)
        self._rs_pipeline = pipe

        # Align depth → colour so pixel coordinates match despite
        # different native resolutions
        self._rs_align = rs.align(rs.stream.color)

        # ── Fix overexposure on the RGB sensor ──────────────────────
        device = profile.get_device()
        for sensor in device.query_sensors():
            # The colour sensor's name typically contains "RGB" or "Color"
            name = sensor.get_info(rs.camera_info.name)
            if "rgb" in name.lower() or "color" in name.lower():
                try:
                    if sensor.supports(rs.option.enable_auto_exposure):
                        sensor.set_option(rs.option.enable_auto_exposure, 1)
                        print("[FALCON] RealSense: auto-exposure enabled")
                except Exception:
                    # Auto-exposure failed – fall back to safe manual values
                    try:
                        sensor.set_option(rs.option.enable_auto_exposure, 0)
                        sensor.set_option(rs.option.exposure, 156)
                        sensor.set_option(rs.option.gain, 64)
                        print("[FALCON] RealSense: manual exposure set (exp=156, gain=64)")
                    except Exception as exc:
                        print(f"[FALCON] RealSense: could not set exposure: {exc}")
                break

        # Post-processing filters for cleaner depth maps
        self._rs_spatial = rs.spatial_filter()
        self._rs_spatial.set_option(rs.option.filter_magnitude, 2)
        self._rs_spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        self._rs_spatial.set_option(rs.option.filter_smooth_delta, 20)

        self._rs_temporal = rs.temporal_filter()
        self._rs_temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
        self._rs_temporal.set_option(rs.option.filter_smooth_delta, 20)

        self._rs_hole_fill = rs.hole_filling_filter()

        self._using_realsense = True
        self._opened = True
        label = f" (S/N {serial})" if serial else ""
        print(
            f"[FALCON] RealSense camera active{label}: "
            f"color {self._RS_COLOR_W}x{self._RS_COLOR_H}, "
            f"depth {self._RS_DEPTH_W}x{self._RS_DEPTH_H} @ {fps}FPS"
        )

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
            frames = self._rs_pipeline.wait_for_frames(timeout_ms=200)
        except Exception as exc:
            print(f"[FALCON] RealSense frame timeout: {exc}")
            return None, None
        try:
            aligned = self._rs_align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
        except Exception as exc:
            print(f"[FALCON] RealSense align error: {exc}")
            return None, None
        if not color_frame:
            return None, None

        # Apply post-processing filters to the depth frame
        if depth_frame:
            depth_frame = self._rs_spatial.process(depth_frame)
            depth_frame = self._rs_temporal.process(depth_frame)
            depth_frame = self._rs_hole_fill.process(depth_frame)

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
