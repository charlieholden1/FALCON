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
import os
import re
import sys
import threading
import time
from typing import Optional, Tuple

if sys.platform == "linux":
    import fcntl
    import struct

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
        # Populated during _init_realsense or _init_webcam so downstream
        # projection code can size the image plane correctly.
        self._color_intrinsics: Optional[dict] = None
        self._frame_size: Optional[Tuple[int, int]] = None

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

    _V4L2_COLOR_FORMATS = {
        "MJPG",
        "YUYV",
        "UYVY",
        "RGB3",
        "BGR3",
        "BA81",
        "RGGB",
        "GRBG",
        "GBRG",
        "BGGR",
    }
    _V4L2_DEPTH_FORMATS = {
        "Z16 ",
        "Y16 ",
        "GREY",
        "RW16",
        "INVZ",
        "INVI",
    }

    @staticmethod
    def _fourcc_to_str(value: int) -> str:
        return "".join(chr((value >> (8 * i)) & 0xFF) for i in range(4))

    @staticmethod
    def _linux_webcam_descriptors() -> list:
        """
        Discover Linux V4L2 nodes that are likely to work as color webcams.

        RealSense and UVC devices often publish multiple /dev/videoN nodes:
        color, depth/IR, and metadata. OpenCV emits noisy warnings when asked
        to open the wrong ones, so use lightweight V4L2 capability/format
        queries before the GUI builds its candidate list.
        """
        if sys.platform != "linux":
            return []

        video_dir = "/sys/class/video4linux"
        try:
            names = os.listdir(video_dir)
        except OSError:
            return []

        devices = []
        for name in names:
            match = re.fullmatch(r"video(\d+)", name)
            if match is None:
                continue
            index = int(match.group(1))
            path = f"/dev/video{index}"
            try:
                fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            except OSError:
                continue

            try:
                info = DualStreamCamera._query_v4l2_device(fd)
            except OSError:
                os.close(fd)
                continue
            finally:
                try:
                    os.close(fd)
                except OSError:
                    pass

            if not info["is_capture"] or info["is_metadata"]:
                continue

            formats = info["formats"]
            has_color = any(fmt in DualStreamCamera._V4L2_COLOR_FORMATS for fmt in formats)
            has_depth_only = formats and all(
                fmt in DualStreamCamera._V4L2_DEPTH_FORMATS for fmt in formats
            )
            if formats and not has_color and has_depth_only:
                continue

            devices.append({
                "label": f"Webcam {index}",
                "type": "webcam",
                "index": index,
                "serial": None,
                "_formats": formats,
                "_has_color": has_color,
            })

        devices.sort(key=lambda d: (not d["_has_color"], d["index"]))
        for device in devices:
            device.pop("_formats", None)
            device.pop("_has_color", None)

        # Silently probe each V4L2 candidate to drop nodes that pass the
        # capability filter but can't actually produce frames (e.g. RealSense
        # metadata streams, USB-audio capture interfaces, etc.).  Stderr is
        # redirected at the file-descriptor level so GStreamer/V4L2 driver
        # warnings don't appear in the terminal during discovery.
        verified: list = []
        for device in devices:
            if DualStreamCamera._probe_webcam_silent(device["index"]):
                verified.append(device)
        return verified

    @staticmethod
    def _query_v4l2_device(fd: int) -> dict:
        VIDIOC_QUERYCAP = 0x80685600
        VIDIOC_ENUM_FMT = 0xC0405602
        V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
        V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE = 9
        V4L2_CAP_VIDEO_CAPTURE = 0x00000001
        V4L2_CAP_VIDEO_CAPTURE_MPLANE = 0x00001000
        V4L2_CAP_META_CAPTURE = 0x00800000
        V4L2_CAP_DEVICE_CAPS = 0x80000000

        cap_buf = bytearray(104)
        fcntl.ioctl(fd, VIDIOC_QUERYCAP, cap_buf, True)
        _, _, _, _, capabilities, device_caps = struct.unpack_from(
            "16s32s32sIII", cap_buf,
        )
        caps = device_caps if capabilities & V4L2_CAP_DEVICE_CAPS else capabilities
        is_capture = bool(
            caps & (V4L2_CAP_VIDEO_CAPTURE | V4L2_CAP_VIDEO_CAPTURE_MPLANE)
        )
        is_metadata = bool(caps & V4L2_CAP_META_CAPTURE)

        formats = []
        for buffer_type in (
            V4L2_BUF_TYPE_VIDEO_CAPTURE,
            V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE,
        ):
            fmt_index = 0
            while True:
                fmt_buf = bytearray(64)
                struct.pack_into("II", fmt_buf, 0, fmt_index, buffer_type)
                try:
                    fcntl.ioctl(fd, VIDIOC_ENUM_FMT, fmt_buf, True)
                except OSError:
                    break
                pixelformat = struct.unpack_from("I", fmt_buf, 44)[0]
                formats.append(DualStreamCamera._fourcc_to_str(pixelformat))
                fmt_index += 1

        return {
            "is_capture": is_capture,
            "is_metadata": is_metadata,
            "formats": formats,
        }

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

        # Always expose a generic RealSense option when the SDK is present.
        # On Jetson/Linux the SDK device discovery can fail in some contexts
        # even though opening the pipeline directly still works.
        if _HAS_REALSENSE:
            cameras.append({
                "label": "Intel RealSense (Auto)",
                "type": "realsense",
                "index": None,
                "serial": None,
            })
            try:
                ctx = rs.context()
                for dev in ctx.devices:
                    name = dev.get_info(rs.camera_info.name)
                    serial = dev.get_info(rs.camera_info.serial_number)
                    entry = {
                        "label": f"{name} (Depth)",
                        "type": "realsense",
                        "index": None,
                        "serial": serial,
                    }
                    if entry not in cameras:
                        cameras.append(entry)
            except Exception:
                pass

        webcams = DualStreamCamera._linux_webcam_descriptors()
        if not webcams:
            # Avoid probing webcam indices during GUI startup because some
            # V4L2 stacks emit noisy driver errors or leave invalid handles
            # behind. Keep a few manual entries available as fallback.
            webcams = [
                {
                    "label": f"Webcam {i}",
                    "type": "webcam",
                    "index": i,
                    "serial": None,
                }
                for i in range(4)
            ]
        cameras.extend(webcams)

        return cameras

    @staticmethod
    def _open_webcam_capture(index: int) -> cv2.VideoCapture:
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
        return cap

    @staticmethod
    def _probe_webcam_index(index: int) -> bool:
        cap = DualStreamCamera._open_webcam_capture(index)
        try:
            if not cap.isOpened():
                return False
            ok, frame = cap.read()
            return bool(ok and frame is not None and frame.size > 0)
        finally:
            cap.release()

    @staticmethod
    def _probe_webcam_silent(index: int) -> bool:
        """Like _probe_webcam_index but redirects stderr to /dev/null first.

        Keeps GStreamer/V4L2 driver warnings out of the terminal when probing
        devices that pass the capability filter but can't produce frames.
        Only available on POSIX; falls back to the noisy version elsewhere.
        """
        if os.name != "posix":
            return DualStreamCamera._probe_webcam_index(index)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        saved_stderr = os.dup(2)
        try:
            os.dup2(devnull_fd, 2)
            return DualStreamCamera._probe_webcam_index(index)
        finally:
            os.dup2(saved_stderr, 2)
            os.close(saved_stderr)
            os.close(devnull_fd)

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

        # Capture factory-calibrated colour intrinsics so CameraProjection
        # can drop the hard-coded defaults (fx=fy=800, cx=640, cy=360).
        try:
            color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            intr = color_profile.get_intrinsics()
            self._color_intrinsics = {
                "fx": float(intr.fx),
                "fy": float(intr.fy),
                "cx": float(intr.ppx),
                "cy": float(intr.ppy),
                "width": int(intr.width),
                "height": int(intr.height),
                "model": str(intr.model).split(".")[-1],
                "coeffs": [float(c) for c in intr.coeffs],
            }
            self._frame_size = (int(intr.width), int(intr.height))
            print(
                f"[FALCON] RealSense intrinsics: "
                f"fx={intr.fx:.1f} fy={intr.fy:.1f} "
                f"cx={intr.ppx:.1f} cy={intr.ppy:.1f} "
                f"{intr.width}x{intr.height}"
            )
        except Exception as exc:
            print(f"[FALCON] Could not read RealSense intrinsics: {exc}")

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
        # These filters improve depth cosmetics, but they cost noticeable
        # CPU on Jetson. Keep them available but off by default.
        self._rs_enable_depth_postprocess = False
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
        cap = self._open_webcam_capture(index)

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
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                print(f"[FALCON] Webcam {index} opened but did not return frames")
                cap.release()
                cap = None
            else:
                self._frame_size = (int(actual_w), int(actual_h))
                print(
                    f"[FALCON] Webcam fallback active: "
                    f"{actual_w}x{actual_h} @ {actual_fps}FPS"
                )

        self._cap = cap
        self._q = queue.Queue(maxsize=self._q_size)
        self._opened = bool(cap is not None and cap.isOpened())

    # ── start / stop ────────────────────────────────────────────────

    def start(self) -> "DualStreamCamera":
        """Start capturing.  Launches a reader thread for both RS and webcam."""
        if not self._opened:
            return self
        self._running = True
        if self._using_realsense:
            self._q = queue.Queue(maxsize=self._q_size)
            self._thread = threading.Thread(
                target=self._realsense_reader, daemon=True,
            )
        else:
            self._thread = threading.Thread(
                target=self._webcam_reader, daemon=True,
            )
        self._thread.start()
        return self

    def _realsense_reader(self) -> None:
        """Continuously fetch RealSense frames into the shared queue."""
        while self._running:
            try:
                frames = self._rs_pipeline.wait_for_frames(timeout_ms=200)
            except Exception:
                continue
            try:
                aligned = self._rs_align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
            except Exception:
                continue
            if not color_frame:
                continue

            if depth_frame and self._rs_enable_depth_postprocess:
                depth_frame = self._rs_spatial.process(depth_frame)
                depth_frame = self._rs_temporal.process(depth_frame)
                depth_frame = self._rs_hole_fill.process(depth_frame)

            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data()) if depth_frame else None

            if self._q.full():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
            self._q.put((color, depth))

    def _webcam_reader(self) -> None:
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                break
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
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

    def get_color_intrinsics(self) -> Optional[dict]:
        """
        Return factory-calibrated colour intrinsics when a RealSense is in
        use, or ``None`` for the webcam fallback.

        Keys: fx, fy, cx, cy, width, height, model, coeffs.
        """
        return dict(self._color_intrinsics) if self._color_intrinsics else None

    def get_frame_size(self) -> Optional[Tuple[int, int]]:
        """Return (width, height) of the active colour stream when known."""
        return self._frame_size

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
            return self._q.get(timeout=0.02)
        except queue.Empty:
            return None, None

    def _read_webcam(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            frame = self._q.get(timeout=0.02)
            return frame, None
        except queue.Empty:
            return None, None

    # ── cleanup ─────────────────────────────────────────────────────

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._using_realsense:
            if self._rs_pipeline is not None:
                try:
                    self._rs_pipeline.stop()
                except Exception:
                    pass
                self._rs_pipeline = None
        else:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
        self._opened = False
