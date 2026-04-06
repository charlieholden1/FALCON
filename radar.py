"""
F.A.L.C.O.N. - Radar Sensor Module
==================================

Provides radar integration for sensor-fusion tracking and diagnostic tooling:

RadarPoint
    Dataclass for a single detected radar point (x, y, z, velocity, snr).

RadarFrame
    Normalized frame shape shared by live UART sessions and replay sessions.

IWR6843Driver
    Driver for the TI IWR6843 mmWave sensor. Communicates over the
    command UART and parses the logging UART packets produced by the
    flashed 3D people-tracking firmware.

ReplayRadarSource
    Loads TI Industrial Visualizer replay JSON and exposes it through the
    same frame-source contract used by the live driver.

MockRadar
    Simulates a simple 3-D radar target for testing without hardware.

CameraProjection
    Pinhole camera model with live-editable intrinsics/extrinsics and
    JSON save/load helpers for radar-to-camera calibration.
"""

from __future__ import annotations

import json
import logging
import math
import re
import struct
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import serial

    _HAS_SERIAL = True
except ImportError:
    _HAS_SERIAL = False

try:
    from serial.tools import list_ports
except Exception:
    list_ports = None

logger = logging.getLogger("falcon.radar")


MAGIC_WORD = b"\x02\x01\x04\x03\x06\x05\x08\x07"
HEADER_SIZE = 40
TLV_HEADER_SIZE = 8
TLV_DETECTED_OBJECTS = 1
TLV_SIDE_INFO = 7

HEALTH_HEALTHY = "Healthy"
HEALTH_CONFIG_FAILED = "Config Failed"
HEALTH_STREAMING_UNPARSED = "Streaming But Unparsed"
HEALTH_NO_DATA = "No Data"
HEALTH_REPLAY_MODE = "Replay Mode"

_MAGIC_HEADER_WORDS = (0x0102, 0x0304, 0x0506, 0x0708)
_HEADER_STRUCT = struct.Struct("<4H8I")
_TLV_STRUCT = struct.Struct("<2I")
_RAW_POINT_STRUCT = struct.Struct("<4f")
_COMPRESSED_UNIT_STRUCT = struct.Struct("<5f")
_COMPRESSED_POINT_STRUCT = struct.Struct("<bbHhH")

_RUNS_DIR = Path("diagnostics_runs")
_MAX_CLI_LOG_CHARS = 200000
_CLI_LOG_TAIL_CHARS = 4000
_SERIAL_OPEN_RETRIES = 6
_SERIAL_OPEN_DELAY_S = 0.5
_CLI_WAKE_ATTEMPTS = 3
_CLI_WAKE_TIMEOUT_S = 0.8
_CLI_POST_OPEN_SETTLE_S = 1.0


@dataclass
class RadarPoint:
    """A single radar-detected point in 3-D space."""

    x: float
    y: float
    z: float
    velocity: float
    snr: float = 0.0

    @property
    def position_3d(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    def __repr__(self) -> str:
        return (
            f"RadarPoint(x={self.x:+.3f}, y={self.y:.3f}, z={self.z:+.3f}, "
            f"vel={self.velocity:+.2f} m/s, snr={self.snr:.1f} dB)"
        )


@dataclass
class RadarFrame:
    """One parsed frame from the radar logging UART or replay."""

    frame_number: int
    subframe_number: int
    num_detected_obj: int
    num_tlvs: int
    points: List[RadarPoint]
    timestamp: float
    source_timestamp: float = 0.0


@dataclass
class RadarSessionState:
    """Current source/session status used by the diagnostic viewer."""

    mode: str
    connected: bool
    config_port: str = ""
    data_port: str = ""
    config_path: str = ""
    replay_path: str = ""
    health_verdict: str = HEALTH_NO_DATA
    health_reason: str = ""
    cli_log_tail: str = ""
    rx_bytes: int = 0
    magic_hits: int = 0
    frames: int = 0
    fps: float = 0.0
    last_packet_size: int = 0
    last_parse_error: str = ""
    last_command_error: str = ""
    connection_error: str = ""
    unknown_tlv_counts: Dict[int, int] = field(default_factory=dict)
    unknown_tlv_lengths: Dict[int, List[int]] = field(default_factory=dict)
    plots_updating: bool = False
    config_ok: bool = False
    probe_failed_stage: str = ""
    probe_log_tail: str = ""
    run_dir: str = ""
    report_path: str = ""


@dataclass
class SerialPortInfo:
    """Normalized serial-port metadata for UI listing and pairing."""

    device: str
    description: str = ""
    manufacturer: str = ""
    product: str = ""
    interface: str = ""
    hwid: str = ""
    role_hint: str = "unknown"
    score: int = 0
    group_key: str = ""

    def display_label(self) -> str:
        parts = [self.device]
        if self.description:
            parts.append(self.description)
        if self.interface:
            parts.append(f"[{self.interface}]")
        if self.role_hint != "unknown":
            parts.append(f"({self.role_hint} hint)")
        return " - ".join(parts[:2]) + (f" {parts[2]}" if len(parts) > 2 else "") + (
            f" {parts[3]}" if len(parts) > 3 else ""
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReplayCapture:
    """Loaded replay dataset plus timing information."""

    path: str
    cfg_lines: List[str]
    demo: str
    device: str
    frames: List[RadarFrame]
    schedule_s: List[float]


@dataclass
class ProbeStageResult:
    """One staged startup probe result."""

    stage_name: str
    ok: bool
    failed_command: Optional[str]
    rx_bytes: int
    magic_hits: int
    config_log_tail: str


class FrameSource(ABC):
    """Small contract shared by live, replay, and mock sources."""

    @abstractmethod
    def start(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def latest_frame(self) -> Optional[RadarFrame]:
        raise NotImplementedError

    @abstractmethod
    def session_state(self) -> RadarSessionState:
        raise NotImplementedError

    def note_frame_rendered(self, frame: Optional[RadarFrame]) -> None:
        del frame


def open_serial_with_retries(
    port: str,
    baudrate: int,
    *,
    timeout: float,
    attempts: int = _SERIAL_OPEN_RETRIES,
    retry_delay_s: float = _SERIAL_OPEN_DELAY_S,
) -> "serial.Serial":
    """Open a serial port with retries for transient USB bridge failures."""

    last_error: Optional[BaseException] = None
    for attempt in range(1, attempts + 1):
        try:
            return serial.Serial(port, baudrate, timeout=timeout)
        except (serial.SerialException, OSError) as exc:
            last_error = exc
            if attempt == attempts:
                break
            logger.warning(
                "Serial open failed for %s at %d baud (attempt %d/%d): %s",
                port,
                baudrate,
                attempt,
                attempts,
                exc,
            )
            time.sleep(retry_delay_s)

    assert last_error is not None
    if isinstance(last_error, serial.SerialException):
        raise last_error
    raise serial.SerialException(str(last_error))


def wake_cli(
    ser: "serial.Serial",
    *,
    attempts: int = _CLI_WAKE_ATTEMPTS,
    timeout_s: float = _CLI_WAKE_TIMEOUT_S,
) -> bytes:
    """Send blank lines until the radar CLI prompt starts responding."""

    response = bytearray()
    for _ in range(attempts):
        ser.write(b"\n")
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            chunk = ser.read(ser.in_waiting or 1)
            if chunk:
                response.extend(chunk)
                if (
                    b"mmwDemo:/>" in response
                    or b"Done" in response
                    or b"Ignored" in response
                    or b"Error" in response
                ):
                    return bytes(response)
            else:
                time.sleep(0.01)
    return bytes(response)


def _coerce_text(value: Any) -> str:
    return str(value or "")


def _extract_hwid_value(hwid: str, key: str) -> str:
    match = re.search(rf"{re.escape(key)}=([^\s]+)", hwid, re.IGNORECASE)
    return match.group(1) if match else ""


def _serial_group_key(info: SerialPortInfo) -> str:
    hwid = info.hwid or ""
    serial_id = _extract_hwid_value(hwid, "SER")
    location = _extract_hwid_value(hwid, "LOCATION")
    vid_pid = _extract_hwid_value(hwid, "VID:PID")
    if serial_id:
        return f"{vid_pid}:{serial_id}"
    if location:
        return f"{vid_pid}:{location.split(':', 1)[0]}"
    text = " ".join(
        part.lower()
        for part in (
            info.device,
            info.description,
            info.manufacturer,
            info.product,
        )
        if part
    )
    if "xds110" in text:
        return "xds110"
    if "cp2105" in text:
        return "cp2105"
    return ""


def _serial_role_hint(info: SerialPortInfo) -> str:
    text = " ".join(
        part.lower()
        for part in (
            info.device,
            info.description,
            info.manufacturer,
            info.product,
            info.interface,
            info.hwid,
        )
        if part
    )
    if "if00" in text:
        return "config"
    if "if03" in text:
        return "data"
    # On this project's CP2105-based adapter, the Enhanced interface is the
    # command CLI and the Standard interface is the high-rate data stream.
    if "enhanced com port" in text:
        return "config"
    if "standard com port" in text:
        return "data"
    if "application/user uart" in text:
        return "data"
    if "aux data port" in text:
        return "data"
    return "unknown"


def _serial_score(info: SerialPortInfo) -> int:
    text = " ".join(
        part.lower()
        for part in (
            info.device,
            info.description,
            info.manufacturer,
            info.product,
            info.interface,
            info.hwid,
        )
        if part
    )
    score = 0
    if "texas instruments" in text or "ti " in text or "xds110" in text:
        score += 8
    if "cp2105" in text:
        score += 6
    if "aux data port" in text or "application/user uart" in text:
        score += 4
    if "enhanced com port" in text or "standard com port" in text:
        score += 3
    if "acm" in info.device.lower() or "usb" in info.device.lower():
        score += 1
    return score


def serial_port_info_from_port(port: Any) -> SerialPortInfo:
    info = SerialPortInfo(
        device=_coerce_text(getattr(port, "device", "")),
        description=_coerce_text(getattr(port, "description", "")),
        manufacturer=_coerce_text(getattr(port, "manufacturer", "")),
        product=_coerce_text(getattr(port, "product", "")),
        interface=_coerce_text(getattr(port, "interface", "")),
        hwid=_coerce_text(getattr(port, "hwid", "")),
    )
    info.role_hint = _serial_role_hint(info)
    info.score = _serial_score(info)
    info.group_key = _serial_group_key(info)
    return info


def discover_serial_ports(port_objects: Optional[Iterable[Any]] = None) -> List[SerialPortInfo]:
    """Return normalized serial-port metadata for likely radar devices."""

    if port_objects is None:
        if list_ports is None:
            return []
        try:
            port_objects = list_ports.comports()
        except Exception:
            return []

    infos = [serial_port_info_from_port(port) for port in port_objects]
    infos.sort(key=lambda item: item.device)
    return infos


def suggest_serial_port_pairs(port_infos: Sequence[SerialPortInfo]) -> List[Tuple[str, str]]:
    """Suggest likely (config_port, data_port) pairs from enumerated ports."""

    pairs: List[Tuple[str, str]] = []
    seen: set = set()
    grouped: Dict[str, List[SerialPortInfo]] = defaultdict(list)

    for info in port_infos:
        if info.group_key:
            grouped[info.group_key].append(info)

    def add_pair(config_port: str, data_port: str) -> None:
        if not config_port or not data_port or config_port == data_port:
            return
        pair = (config_port, data_port)
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)

    for group_infos in grouped.values():
        if len(group_infos) < 2:
            continue
        config_candidates = [
            info for info in group_infos if info.role_hint == "config"
        ] or sorted(group_infos, key=lambda info: (info.score, info.device), reverse=True)
        data_candidates = [
            info for info in group_infos if info.role_hint == "data"
        ] or sorted(group_infos, key=lambda info: (info.score, info.device), reverse=True)
        add_pair(config_candidates[0].device, data_candidates[0].device)

    scored = sorted(port_infos, key=lambda info: (info.score, info.device), reverse=True)
    if len(scored) >= 2:
        config_port = ""
        data_port = ""
        for info in scored:
            if not config_port and info.role_hint == "config":
                config_port = info.device
            if not data_port and info.role_hint == "data":
                data_port = info.device
        if not config_port:
            config_port = scored[0].device
        if not data_port:
            for info in scored:
                if info.device != config_port:
                    data_port = info.device
                    break
        add_pair(config_port, data_port)

    return pairs


def evaluate_health_verdict(
    *,
    mode: str,
    config_ok: bool,
    last_command_error: str,
    connection_error: str,
    rx_bytes: int,
    magic_hits: int,
    frames: int,
    last_parse_error: str,
    data_opened_at: Optional[float],
    plots_updating: bool,
    now: Optional[float] = None,
) -> Tuple[str, str]:
    """Map low-level session signals onto the viewer's health states."""

    if mode == "replay":
        return HEALTH_REPLAY_MODE, "Replay is rendering; hardware viability is not being judged."

    if last_command_error:
        return HEALTH_CONFIG_FAILED, last_command_error.strip()

    if connection_error:
        return HEALTH_NO_DATA, connection_error.strip()

    if not config_ok:
        return HEALTH_NO_DATA, "Waiting to start the sensor."

    if magic_hits > 0 and frames >= 3 and plots_updating:
        return (
            HEALTH_HEALTHY,
            "Config succeeded, frames are parsing, and the live plots are updating.",
        )

    if (rx_bytes > 0 or magic_hits > 0) and (frames == 0 or bool(last_parse_error)):
        reason = "Bytes are arriving but frames are not parsing cleanly."
        if last_parse_error:
            reason += f" Last parse error: {last_parse_error}"
        return HEALTH_STREAMING_UNPARSED, reason

    if data_opened_at is None:
        return HEALTH_NO_DATA, "Logging UART has not been opened yet."

    elapsed = max(0.0, float(now if now is not None else time.time()) - data_opened_at)
    if elapsed < 5.0:
        if frames < 3:
            return HEALTH_NO_DATA, "Waiting for at least 3 parsed frames within the startup window."
        if not plots_updating:
            return HEALTH_NO_DATA, "Frames are parsing; waiting for the plots to update."

    if rx_bytes == 0:
        return HEALTH_NO_DATA, "Config succeeded, but no data bytes arrived within 5 seconds."

    if frames < 3:
        return HEALTH_NO_DATA, "Frames are arriving, but the viability threshold has not been met yet."

    if not plots_updating:
        return HEALTH_NO_DATA, "Frames parsed, but the plots are not updating yet."

    return HEALTH_NO_DATA, "Config succeeded, but the logging stream is still inconclusive."


def load_replay_capture(path: Union[str, Path]) -> ReplayCapture:
    """Load a TI Industrial Visualizer replay JSON into normalized frames."""

    replay_path = Path(path)
    payload = json.loads(replay_path.read_text(encoding="utf-8"))
    entries = payload.get("data", [])

    frames: List[RadarFrame] = []
    schedule_s: List[float] = []

    if entries:
        base_timestamp = float(entries[0].get("timestamp", 0.0))
    else:
        base_timestamp = 0.0

    for index, entry in enumerate(entries):
        frame_data = entry.get("frameData", {})
        points: List[RadarPoint] = []
        for raw_point in frame_data.get("pointCloud", []):
            if not isinstance(raw_point, (list, tuple)) or len(raw_point) < 4:
                continue
            snr = float(raw_point[4]) if len(raw_point) > 4 else 0.0
            points.append(
                RadarPoint(
                    x=float(raw_point[0]),
                    y=float(raw_point[1]),
                    z=float(raw_point[2]),
                    velocity=float(raw_point[3]),
                    snr=snr,
                )
            )

        source_timestamp = float(entry.get("timestamp", base_timestamp)) / 1000.0
        relative_schedule = max(0.0, (float(entry.get("timestamp", base_timestamp)) - base_timestamp) / 1000.0)
        schedule_s.append(relative_schedule)
        frames.append(
            RadarFrame(
                frame_number=int(frame_data.get("frameNum", index)),
                subframe_number=int(frame_data.get("subFrameNum", 0)),
                num_detected_obj=int(frame_data.get("numDetectedPoints", len(points))),
                num_tlvs=int(frame_data.get("numTLVs", 0)),
                points=points,
                timestamp=relative_schedule,
                source_timestamp=source_timestamp,
            )
        )

    cfg_lines = [str(line).rstrip("\n") for line in payload.get("cfg", [])]
    return ReplayCapture(
        path=str(replay_path),
        cfg_lines=cfg_lines,
        demo=str(payload.get("demo", "")),
        device=str(payload.get("device", "")),
        frames=frames,
        schedule_s=schedule_s,
    )


class IWR6843Driver(FrameSource):
    """Driver for the TI IWR6843ISK on the logging + command UARTs."""

    def __init__(
        self,
        config_port: str = "/dev/ttyACM0",
        data_port: str = "/dev/ttyACM1",
        config_path: str = "iwr6843_people_tracking.cfg",
        config_baud: int = 115200,
        data_baud: int = 921600,
    ):
        self._config_port_path = config_port
        self._data_port_path = data_port
        self._config_path = config_path
        self._config_baud = config_baud
        self._data_baud = data_baud

        self._config_serial: Optional[serial.Serial] = None  # type: ignore[name-defined]
        self._data_serial: Optional[serial.Serial] = None  # type: ignore[name-defined]

        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._connected = False

        self._frame_buffer: deque[RadarFrame] = deque(maxlen=5)
        self._lock = threading.Lock()

        self._frame_count = 0
        self._frame_timestamps: deque[float] = deque(maxlen=20)
        self._last_packet_size = 0
        self._last_parse_error = ""
        self._rx_bytes = 0
        self._magic_hits = 0
        self._last_command_error = ""
        self._connection_error = ""
        self._config_ok = False
        self._data_opened_at: Optional[float] = None

        self._last_rendered_at: Optional[float] = None
        self._last_rendered_frame_number: Optional[int] = None

        self._unknown_tlv_counts: Dict[int, int] = defaultdict(int)
        self._unknown_tlv_lengths: Dict[int, List[int]] = defaultdict(list)

        self._cli_log_text = ""
        self._run_dir: Optional[Path] = None
        self._report_path: Optional[Path] = None

        self._probe_failed_stage = ""
        self._probe_log_tail = ""

    def start(self) -> bool:
        return self.open()

    def stop(self) -> None:
        self.close()

    def latest_frame(self) -> Optional[RadarFrame]:
        return self.get_latest_frame()

    def session_state(self) -> RadarSessionState:
        latest_frame = self.get_latest_frame()
        plots_updating = self._plots_updating(latest_frame)
        verdict, reason = evaluate_health_verdict(
            mode="live",
            config_ok=self._config_ok,
            last_command_error=self._last_command_error,
            connection_error=self._connection_error,
            rx_bytes=self._rx_bytes,
            magic_hits=self._magic_hits,
            frames=self._frame_count,
            last_parse_error=self._last_parse_error,
            data_opened_at=self._data_opened_at,
            plots_updating=plots_updating,
        )
        return RadarSessionState(
            mode="live",
            connected=self._connected and self._running,
            config_port=self._config_port_path,
            data_port=self._data_port_path,
            config_path=self._config_path,
            health_verdict=verdict,
            health_reason=reason,
            cli_log_tail=self._cli_log_text[-_CLI_LOG_TAIL_CHARS:],
            rx_bytes=self._rx_bytes,
            magic_hits=self._magic_hits,
            frames=self._frame_count,
            fps=self.points_per_second,
            last_packet_size=self._last_packet_size,
            last_parse_error=self._last_parse_error,
            last_command_error=self._last_command_error,
            connection_error=self._connection_error,
            unknown_tlv_counts=dict(self._unknown_tlv_counts),
            unknown_tlv_lengths={key: list(value) for key, value in self._unknown_tlv_lengths.items()},
            plots_updating=plots_updating,
            config_ok=self._config_ok,
            probe_failed_stage=self._probe_failed_stage,
            probe_log_tail=self._probe_log_tail,
            run_dir=str(self._run_dir) if self._run_dir else "",
            report_path=str(self._report_path) if self._report_path else "",
        )

    def note_frame_rendered(self, frame: Optional[RadarFrame]) -> None:
        if frame is None:
            return
        self._last_rendered_at = time.time()
        self._last_rendered_frame_number = frame.frame_number

    def open(self) -> bool:
        """Configure the sensor and start data streaming."""

        if self._running:
            return True

        if not _HAS_SERIAL:
            logger.error("pyserial not installed - cannot open radar")
            self._connection_error = "pyserial not installed - cannot open radar"
            return False

        self._reset_session_metrics()
        self._start_run_artifacts()

        config_path = Path(self._config_path)
        if not config_path.exists():
            fallback = Path("iwr6843_config.cfg")
            if fallback.exists():
                logger.warning(
                    "Radar config %s not found; falling back to %s",
                    config_path,
                    fallback,
                )
                self._config_path = str(fallback)
            else:
                self._connection_error = f"No radar config file found at {config_path}"
                logger.error(self._connection_error)
                self._persist_session_artifacts()
                return False

        try:
            self._config_serial = open_serial_with_retries(
                self._config_port_path,
                self._config_baud,
                timeout=0.1,
            )
        except serial.SerialException as exc:
            self._connection_error = f"Failed to open config serial port: {exc}"
            logger.error(self._connection_error)
            self._cleanup_serial()
            self._persist_session_artifacts()
            return False

        try:
            self._data_serial = open_serial_with_retries(
                self._data_port_path,
                self._data_baud,
                timeout=0.05,
            )
        except serial.SerialException as exc:
            self._connection_error = f"Failed to open data serial port: {exc}"
            logger.error(self._connection_error)
            self._cleanup_serial()
            self._persist_session_artifacts()
            return False

        time.sleep(_CLI_POST_OPEN_SETTLE_S)
        try:
            if self._config_serial.in_waiting:
                self._config_serial.reset_input_buffer()
        except serial.SerialException:
            pass
        try:
            if self._data_serial.in_waiting:
                self._data_serial.reset_input_buffer()
        except serial.SerialException:
            pass

        self._poke_cli()

        if not self._send_config():
            logger.error("Failed to send configuration")
            self._cleanup_serial()
            self._persist_session_artifacts()
            return False

        self._config_ok = True

        self._data_opened_at = time.time()
        self._running = True
        self._connected = True
        self._thread = threading.Thread(
            target=self._reader_loop,
            daemon=True,
            name="radar-reader",
        )
        self._thread.start()
        logger.info(
            "IWR6843 radar started on %s / %s using %s",
            self._config_port_path,
            self._data_port_path,
            self._config_path,
        )
        self._persist_session_artifacts()
        return True

    def close(self) -> None:
        """Stop the sensor and close serial connections."""

        self._running = False

        if self._config_serial is not None and self._config_serial.is_open:
            try:
                self._config_serial.write(b"sensorStop\n")
                time.sleep(0.1)
            except serial.SerialException:
                pass

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        self._cleanup_serial()
        self._connected = False
        self._persist_session_artifacts()
        logger.info("IWR6843 radar stopped")

    def get_point_cloud(self) -> List[RadarPoint]:
        """Return the latest detected radar points."""
        with self._lock:
            if not self._frame_buffer:
                return []
            return list(self._frame_buffer[-1].points)

    def get_latest_frame(self) -> Optional[RadarFrame]:
        with self._lock:
            if not self._frame_buffer:
                return None
            frame = self._frame_buffer[-1]
            return RadarFrame(
                frame_number=frame.frame_number,
                subframe_number=frame.subframe_number,
                num_detected_obj=frame.num_detected_obj,
                num_tlvs=frame.num_tlvs,
                points=list(frame.points),
                timestamp=frame.timestamp,
                source_timestamp=frame.source_timestamp,
            )

    def get_frame_count(self) -> int:
        return self._frame_count

    def wait_for_frame(self, timeout_s: float = 3.0) -> bool:
        """Wait for at least one parsed frame to arrive."""
        deadline = time.time() + timeout_s
        start_count = self._frame_count
        while time.time() < deadline:
            if self._frame_count > start_count:
                return True
            time.sleep(0.05)
        return False

    def is_connected(self) -> bool:
        return self._connected and self._running

    @property
    def points_per_second(self) -> float:
        with self._lock:
            timestamps = list(self._frame_timestamps)
        if len(timestamps) < 2:
            return 0.0
        dt = timestamps[-1] - timestamps[0]
        if dt <= 0:
            return 0.0
        return (len(timestamps) - 1) / dt

    @property
    def diagnostics(self) -> Dict[str, Union[float, int, str, Dict[int, int]]]:
        state = self.session_state()
        return {
            "frames": state.frames,
            "fps": state.fps,
            "last_packet_size": state.last_packet_size,
            "last_parse_error": state.last_parse_error,
            "rx_bytes": state.rx_bytes,
            "magic_hits": state.magic_hits,
            "last_command_error": state.last_command_error,
            "connection_error": state.connection_error,
            "health_verdict": state.health_verdict,
            "health_reason": state.health_reason,
            "unknown_tlv_counts": state.unknown_tlv_counts,
        }

    def parse_packet_bytes(self, packet: bytes) -> Optional[RadarFrame]:
        return self._parse_packet(packet)

    def probe_startup(self) -> List[ProbeStageResult]:
        """Run staged startup probing against the configured cfg file."""

        from radar_diag import build_stage_commands, load_cfg_lines, run_cfg_once

        self.close()
        self._probe_failed_stage = ""
        self._probe_log_tail = ""

        lines = load_cfg_lines(self._config_path)
        results: List[ProbeStageResult] = []
        summary_lines: List[str] = []

        for stage_name, commands in build_stage_commands(lines):
            result = run_cfg_once(
                commands,
                self._config_port_path,
                self._data_port_path,
                self._config_baud,
                self._data_baud,
            )
            stage_result = ProbeStageResult(
                stage_name=stage_name,
                ok=result.ok,
                failed_command=result.failed_command,
                rx_bytes=len(result.data_bytes),
                magic_hits=result.data_bytes.count(MAGIC_WORD),
                config_log_tail=result.config_text[-1200:].strip(),
            )
            results.append(stage_result)
            summary_lines.append(
                f"[{stage_name}] {'PASS' if result.ok else 'FAIL'} "
                f"failed={result.failed_command or '-'} rx={stage_result.rx_bytes} "
                f"magic={stage_result.magic_hits}"
            )
            if stage_result.config_log_tail:
                summary_lines.append(stage_result.config_log_tail)
            if not result.ok:
                self._probe_failed_stage = stage_name
                break

        self._probe_log_tail = "\n".join(summary_lines)[-_CLI_LOG_TAIL_CHARS:]
        if self._probe_log_tail:
            self._append_cli_log("\n=== Probe Startup ===\n")
            self._append_cli_log(self._probe_log_tail + "\n")
        self._persist_session_artifacts()
        return results

    def _reset_session_metrics(self) -> None:
        with self._lock:
            self._frame_buffer.clear()
            self._frame_timestamps.clear()
        self._frame_count = 0
        self._last_packet_size = 0
        self._last_parse_error = ""
        self._rx_bytes = 0
        self._magic_hits = 0
        self._last_command_error = ""
        self._connection_error = ""
        self._config_ok = False
        self._data_opened_at = None
        self._last_rendered_at = None
        self._last_rendered_frame_number = None
        self._unknown_tlv_counts = defaultdict(int)
        self._unknown_tlv_lengths = defaultdict(list)
        self._cli_log_text = ""
        self._probe_failed_stage = ""
        self._probe_log_tail = ""

    def _start_run_artifacts(self) -> None:
        _RUNS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        suffix = f"{int((time.time() % 1.0) * 1000):03d}"
        self._run_dir = _RUNS_DIR / f"{timestamp}_{suffix}"
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._report_path = self._run_dir / "session_report.json"

    def _append_cli_log(self, text: Union[str, bytes]) -> None:
        if isinstance(text, bytes):
            chunk = text.decode("ascii", errors="replace")
        else:
            chunk = str(text)
        if not chunk:
            return
        self._cli_log_text += chunk
        if len(self._cli_log_text) > _MAX_CLI_LOG_CHARS:
            self._cli_log_text = self._cli_log_text[-_MAX_CLI_LOG_CHARS:]

    def _persist_session_artifacts(self) -> None:
        if self._run_dir is None or self._report_path is None:
            return
        try:
            cli_path = self._run_dir / "cli.log"
            cli_path.write_text(self._cli_log_text, encoding="utf-8")
            payload = {
                "saved_at": time.time(),
                "state": asdict(self.session_state()),
            }
            self._report_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Failed to persist radar diagnostics artifacts: %s", exc)

    def _plots_updating(self, latest_frame: Optional[RadarFrame]) -> bool:
        if latest_frame is None or self._last_rendered_at is None:
            return False
        if time.time() - self._last_rendered_at > 2.0:
            return False
        return self._last_rendered_frame_number == latest_frame.frame_number

    def _send_config(self) -> bool:
        """Read the .cfg file and send each command to the config port."""

        try:
            with open(self._config_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except OSError as exc:
            self._connection_error = f"Config file could not be opened: {exc}"
            logger.error(self._connection_error)
            return False

        has_sensor_start = False
        for line in lines:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            if line == "sensorStart":
                has_sensor_start = True
            if line in ("sensorStop", "flushCfg", "sensorStart"):
                ok = self._send_command_with_retries(line, attempts=3, retry_delay=0.4)
            else:
                ok = self._send_command_with_retries(line, attempts=2, retry_delay=0.25)
            if not ok:
                logger.warning("Command failed: %s", line)
                return False
            time.sleep(0.05)

        if not has_sensor_start:
            if not self._send_command_with_retries("sensorStart", attempts=3, retry_delay=0.4):
                return False
        return True

    def _poke_cli(self) -> None:
        if self._config_serial is None or not self._config_serial.is_open:
            return
        try:
            response = wake_cli(self._config_serial)
            self._append_cli_log(response)
        except serial.SerialException:
            pass

    def _send_command_with_retries(
        self,
        cmd: str,
        attempts: int = 2,
        retry_delay: float = 0.25,
    ) -> bool:
        for attempt in range(attempts):
            if self._send_command(cmd):
                return True
            if attempt < attempts - 1:
                time.sleep(retry_delay)
                self._poke_cli()
        return False

    def _send_command(self, cmd: str) -> bool:
        """Send a single CLI command and wait briefly for the response."""

        if self._config_serial is None or not self._config_serial.is_open:
            self._last_command_error = f"{cmd}: config serial port is not open"
            return False

        try:
            self._append_cli_log(f">> {cmd}\n")
            self._config_serial.write((cmd + "\n").encode("ascii"))
            deadline = time.time() + (5.0 if cmd == "sensorStart" else 2.0)
            response = b""
            saw_data = False
            last_data_time = time.time()
            while time.time() < deadline:
                chunk = self._config_serial.read(self._config_serial.in_waiting or 1)
                if chunk:
                    response += chunk
                    saw_data = True
                    last_data_time = time.time()
                    if b"Done" in response:
                        self._append_cli_log(response)
                        self._last_command_error = ""
                        return True
                    if b"Ignored" in response:
                        self._append_cli_log(response)
                        self._last_command_error = ""
                        return True
                    if b"Error" in response:
                        self._append_cli_log(response)
                        text = response.decode("ascii", errors="replace").strip()
                        self._last_command_error = f"{cmd}: {text}"
                        logger.warning("Sensor error for '%s': %s", cmd, text)
                        return False
                    if cmd == "sensorStart" and b"Init Calibration Status" in response:
                        self._append_cli_log(response)
                        self._last_command_error = ""
                        return True
                    if b"mmwDemo:/>" in response and saw_data:
                        self._append_cli_log(response)
                        self._last_command_error = ""
                        return True
                else:
                    if saw_data and cmd == "sensorStart" and (time.time() - last_data_time) > 0.25:
                        self._append_cli_log(response)
                        self._last_command_error = ""
                        return True
                    time.sleep(0.01)
            self._append_cli_log(response)
            text = response.decode("ascii", errors="replace").strip()
            self._last_command_error = (
                f"{cmd}: timeout waiting for CLI response"
                + (f" ({text})" if text else "")
            )
            logger.warning("Sensor timeout for '%s'", cmd)
            return False
        except serial.SerialException as exc:
            self._last_command_error = f"{cmd}: serial exception {exc}"
            logger.warning("Serial write error: %s", exc)
            return False

    def _cleanup_serial(self) -> None:
        if self._config_serial is not None:
            self._config_serial.close()
        if self._data_serial is not None:
            self._data_serial.close()
        self._config_serial = None
        self._data_serial = None

    def _reader_loop(self) -> None:
        """Background thread to parse UART packets from the radar."""

        buffer = bytearray()

        while self._running:
            try:
                if self._data_serial is None or not self._data_serial.is_open:
                    time.sleep(0.02)
                    continue

                chunk = self._data_serial.read(self._data_serial.in_waiting or 1)
                if not chunk:
                    continue
                self._rx_bytes += len(chunk)
                buffer.extend(chunk)

                while True:
                    start_idx = buffer.find(MAGIC_WORD)
                    if start_idx < 0:
                        if len(buffer) > HEADER_SIZE:
                            del buffer[:-HEADER_SIZE]
                        break
                    if start_idx > 0:
                        del buffer[:start_idx]
                    if len(buffer) < HEADER_SIZE:
                        break
                    self._magic_hits += 1

                    header = _HEADER_STRUCT.unpack_from(buffer, 0)
                    packet_len = int(header[5])
                    if packet_len < HEADER_SIZE:
                        self._last_parse_error = f"invalid packet len {packet_len}"
                        del buffer[: len(MAGIC_WORD)]
                        continue
                    if len(buffer) < packet_len:
                        break

                    packet = bytes(buffer[:packet_len])
                    del buffer[:packet_len]
                    self._last_packet_size = packet_len

                    frame = self._parse_packet(packet)
                    if frame is None:
                        continue

                    with self._lock:
                        self._frame_buffer.append(frame)
                        self._frame_count += 1
                        self._frame_timestamps.append(frame.timestamp)

            except Exception as exc:
                if self._running:
                    self._last_parse_error = str(exc)
                    logger.warning("Radar read error: %s", exc)
                time.sleep(0.1)

    def _parse_packet(self, packet: bytes) -> Optional[RadarFrame]:
        try:
            header = _HEADER_STRUCT.unpack_from(packet, 0)
        except struct.error as exc:
            self._last_parse_error = f"header unpack failed: {exc}"
            return None

        if tuple(header[:4]) != _MAGIC_HEADER_WORDS:
            self._last_parse_error = "header unpack failed: bad magic word"
            return None

        frame_number = int(header[7])
        num_detected_obj = int(header[9])
        num_tlvs = int(header[10])
        subframe_number = int(header[11])

        points: List[RadarPoint] = []
        offset = HEADER_SIZE

        for _ in range(num_tlvs):
            if offset + TLV_HEADER_SIZE > len(packet):
                self._last_parse_error = "truncated TLV header"
                break

            tlv_type, tlv_length = _TLV_STRUCT.unpack_from(packet, offset)
            offset += TLV_HEADER_SIZE

            if tlv_length < 0 or offset + tlv_length > len(packet):
                self._last_parse_error = f"bad TLV len {tlv_length} for type {tlv_type}"
                break

            payload = packet[offset : offset + tlv_length]
            offset += tlv_length

            parsed_points = self._parse_tlv_points(tlv_type, payload)
            if parsed_points:
                points = parsed_points
                continue

            if tlv_type not in (TLV_DETECTED_OBJECTS, TLV_SIDE_INFO):
                self._unknown_tlv_counts[int(tlv_type)] += 1
                lengths = self._unknown_tlv_lengths[int(tlv_type)]
                lengths.append(int(tlv_length))
                if len(lengths) > 10:
                    del lengths[:-10]

        return RadarFrame(
            frame_number=frame_number,
            subframe_number=subframe_number,
            num_detected_obj=num_detected_obj,
            num_tlvs=num_tlvs,
            points=points,
            timestamp=time.time(),
            source_timestamp=time.time(),
        )

    def _parse_tlv_points(
        self,
        tlv_type: int,
        payload: bytes,
    ) -> List[RadarPoint]:
        if tlv_type == TLV_DETECTED_OBJECTS and len(payload) % _RAW_POINT_STRUCT.size == 0:
            return self._parse_raw_cartesian_points(payload)

        compressed = self._parse_compressed_points(payload)
        if compressed is not None:
            return compressed

        return []

    def _parse_raw_cartesian_points(self, payload: bytes) -> List[RadarPoint]:
        points: List[RadarPoint] = []
        for offset in range(0, len(payload), _RAW_POINT_STRUCT.size):
            x, y, z, velocity = _RAW_POINT_STRUCT.unpack_from(payload, offset)
            points.append(
                RadarPoint(
                    x=float(x),
                    y=float(y),
                    z=float(z),
                    velocity=float(velocity),
                )
            )
        return points

    def _parse_compressed_points(self, payload: bytes) -> Optional[List[RadarPoint]]:
        if len(payload) < _COMPRESSED_UNIT_STRUCT.size:
            return None
        if (len(payload) - _COMPRESSED_UNIT_STRUCT.size) % _COMPRESSED_POINT_STRUCT.size != 0:
            return None

        units = _COMPRESSED_UNIT_STRUCT.unpack_from(payload, 0)
        azimuth_unit, elevation_unit, range_unit, doppler_unit, snr_unit = units

        if not self._looks_like_compressed_units(
            azimuth_unit,
            elevation_unit,
            range_unit,
            doppler_unit,
            snr_unit,
        ):
            return None

        points: List[RadarPoint] = []
        offset = _COMPRESSED_UNIT_STRUCT.size
        while offset + _COMPRESSED_POINT_STRUCT.size <= len(payload):
            azimuth_i, elevation_i, range_i, doppler_i, snr_i = (
                _COMPRESSED_POINT_STRUCT.unpack_from(payload, offset)
            )
            offset += _COMPRESSED_POINT_STRUCT.size

            azimuth = float(azimuth_i) * azimuth_unit
            elevation = float(elevation_i) * elevation_unit
            distance = float(range_i) * range_unit
            velocity = float(doppler_i) * doppler_unit
            snr = float(snr_i) * snr_unit

            x, y, z = self._spherical_to_cartesian(
                azimuth_deg=azimuth,
                elevation_deg=elevation,
                distance_m=distance,
            )
            points.append(RadarPoint(x=x, y=y, z=z, velocity=velocity, snr=snr))

        return points

    @staticmethod
    def _looks_like_compressed_units(
        azimuth_unit: float,
        elevation_unit: float,
        range_unit: float,
        doppler_unit: float,
        snr_unit: float,
    ) -> bool:
        return (
            np.isfinite(azimuth_unit)
            and np.isfinite(elevation_unit)
            and np.isfinite(range_unit)
            and np.isfinite(doppler_unit)
            and np.isfinite(snr_unit)
            and 0.0 < abs(azimuth_unit) <= 1.0
            and 0.0 < abs(elevation_unit) <= 1.0
            and 0.0 < range_unit <= 1.0
            and 0.0 < doppler_unit <= 5.0
            and 0.0 < snr_unit <= 20.0
        )

    @staticmethod
    def _spherical_to_cartesian(
        azimuth_deg: float,
        elevation_deg: float,
        distance_m: float,
    ) -> Tuple[float, float, float]:
        azimuth = math.radians(azimuth_deg)
        elevation = math.radians(elevation_deg)
        cos_el = math.cos(elevation)

        x = distance_m * math.sin(azimuth) * cos_el
        y = distance_m * math.cos(azimuth) * cos_el
        z = distance_m * math.sin(elevation)
        return float(x), float(y), float(z)


class ReplayRadarSource(FrameSource):
    """Frame source backed by TI replay JSON."""

    def __init__(self, replay_path: Union[str, Path]):
        self._replay_path = str(replay_path)
        self._capture: Optional[ReplayCapture] = None
        self._running = False
        self._start_monotonic: Optional[float] = None
        self._current_index = -1
        self._last_rendered_at: Optional[float] = None
        self._last_rendered_frame_number: Optional[int] = None

    def start(self) -> bool:
        self._capture = load_replay_capture(self._replay_path)
        self._running = True
        self._start_monotonic = time.monotonic()
        self._current_index = 0 if self._capture.frames else -1
        return True

    def stop(self) -> None:
        self._running = False

    def latest_frame(self) -> Optional[RadarFrame]:
        if not self._running or self._capture is None or not self._capture.frames:
            return None
        assert self._start_monotonic is not None
        elapsed = time.monotonic() - self._start_monotonic
        while (
            self._current_index + 1 < len(self._capture.frames)
            and self._capture.schedule_s[self._current_index + 1] <= elapsed
        ):
            self._current_index += 1
        frame = self._capture.frames[max(self._current_index, 0)]
        return RadarFrame(
            frame_number=frame.frame_number,
            subframe_number=frame.subframe_number,
            num_detected_obj=frame.num_detected_obj,
            num_tlvs=frame.num_tlvs,
            points=list(frame.points),
            timestamp=frame.timestamp,
            source_timestamp=frame.source_timestamp,
        )

    def session_state(self) -> RadarSessionState:
        frame = self.latest_frame()
        frames_seen = max(self._current_index + 1, 0)
        plots_updating = (
            frame is not None
            and self._last_rendered_at is not None
            and time.time() - self._last_rendered_at <= 2.0
            and self._last_rendered_frame_number == frame.frame_number
        )
        verdict, reason = evaluate_health_verdict(
            mode="replay",
            config_ok=True,
            last_command_error="",
            connection_error="",
            rx_bytes=0,
            magic_hits=0,
            frames=frames_seen,
            last_parse_error="",
            data_opened_at=time.time() if self._running else None,
            plots_updating=plots_updating,
        )
        return RadarSessionState(
            mode="replay",
            connected=self._running,
            replay_path=self._replay_path,
            config_path="",
            health_verdict=verdict,
            health_reason=reason,
            frames=frames_seen,
            fps=self._replay_fps(),
            plots_updating=plots_updating,
            config_ok=True,
        )

    def note_frame_rendered(self, frame: Optional[RadarFrame]) -> None:
        if frame is None:
            return
        self._last_rendered_at = time.time()
        self._last_rendered_frame_number = frame.frame_number

    def get_point_cloud(self) -> List[RadarPoint]:
        frame = self.latest_frame()
        return [] if frame is None else list(frame.points)

    def get_latest_frame(self) -> Optional[RadarFrame]:
        return self.latest_frame()

    def is_connected(self) -> bool:
        return self._running

    def _replay_fps(self) -> float:
        if self._capture is None or len(self._capture.schedule_s) < 2:
            return 0.0
        elapsed = self._capture.schedule_s[-1] - self._capture.schedule_s[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._capture.schedule_s) - 1) / elapsed


class MockRadar(FrameSource):
    """Simulate a 3-D radar return for a single person."""

    def __init__(
        self,
        amplitude_x: float = 1.0,
        frequency: float = 0.25,
        depth_y: float = 2.0,
        height_z: float = 0.0,
    ):
        self.amplitude_x = amplitude_x
        self.frequency = frequency
        self.depth_y = depth_y
        self.height_z = height_z
        self._t0 = time.time()
        self._frame_count = 0
        self._running = True
        self._last_rendered_at: Optional[float] = None
        self._last_rendered_frame_number: Optional[int] = None

    def start(self) -> bool:
        self._running = True
        return True

    def stop(self) -> None:
        self.close()

    def latest_frame(self) -> Optional[RadarFrame]:
        points = self.get_point_cloud()
        return RadarFrame(
            frame_number=self._frame_count,
            subframe_number=0,
            num_detected_obj=len(points),
            num_tlvs=1,
            points=points,
            timestamp=time.time(),
            source_timestamp=time.time(),
        )

    def session_state(self) -> RadarSessionState:
        latest_frame = self.latest_frame()
        plots_updating = (
            latest_frame is not None
            and self._last_rendered_at is not None
            and time.time() - self._last_rendered_at <= 2.0
            and self._last_rendered_frame_number == latest_frame.frame_number
        )
        verdict = HEALTH_HEALTHY if plots_updating else HEALTH_NO_DATA
        reason = (
            "Mock radar is producing synthetic frames."
            if plots_updating
            else "Mock radar is waiting for the plots to update."
        )
        return RadarSessionState(
            mode="mock",
            connected=self._running,
            health_verdict=verdict,
            health_reason=reason,
            frames=self._frame_count,
            config_ok=True,
            plots_updating=plots_updating,
        )

    def note_frame_rendered(self, frame: Optional[RadarFrame]) -> None:
        if frame is None:
            return
        self._last_rendered_at = time.time()
        self._last_rendered_frame_number = frame.frame_number

    def get_target_3d(self) -> np.ndarray:
        elapsed = time.time() - self._t0
        x = self.amplitude_x * math.sin(2.0 * math.pi * self.frequency * elapsed)
        return np.array([x, self.depth_y, self.height_z], dtype=np.float64)

    def is_connected(self) -> bool:
        return self._running

    def close(self) -> None:
        self._running = False

    def get_point_cloud(self) -> List[RadarPoint]:
        self._frame_count += 1
        target = self.get_target_3d()
        return [RadarPoint(x=target[0], y=target[1], z=target[2], velocity=0.0)]

    def get_latest_frame(self) -> Optional[RadarFrame]:
        return self.latest_frame()

    @property
    def diagnostics(self) -> Dict[str, Union[float, int, str]]:
        state = self.session_state()
        return {
            "frames": state.frames,
            "fps": 0.0,
            "last_packet_size": 0,
            "last_parse_error": "",
            "health_verdict": state.health_verdict,
            "health_reason": state.health_reason,
        }


class CameraProjection:
    """Pinhole camera model with editable radar-to-camera calibration."""

    DEFAULT_PATH = Path("radar_camera_calibration.json")
    _BASE_ALIGNMENT = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    def __init__(
        self,
        K: Optional[np.ndarray] = None,
        RT: Optional[np.ndarray] = None,
    ):
        if K is None:
            fx = fy = 800.0
            cx, cy = 640.0, 360.0
            K = np.array(
                [
                    [fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )

        self.K = np.asarray(K, dtype=np.float64).copy()
        self._yaw_deg = 0.0
        self._pitch_deg = 0.0
        self._roll_deg = 0.0
        self._translation = np.array([0.0, 0.10, 0.0], dtype=np.float64)

        if RT is not None:
            rt = np.asarray(RT, dtype=np.float64)
            self._translation = rt[:3, 3].copy()
            rotation = rt[:3, :3].copy()
            delta = self._BASE_ALIGNMENT.T @ rotation
            yaw, pitch, roll = self._matrix_to_euler_zyx(delta)
            self._yaw_deg = math.degrees(yaw)
            self._pitch_deg = math.degrees(pitch)
            self._roll_deg = math.degrees(roll)

        self._rebuild()

    @property
    def params(self) -> Dict[str, float]:
        return {
            "fx": float(self.K[0, 0]),
            "fy": float(self.K[1, 1]),
            "cx": float(self.K[0, 2]),
            "cy": float(self.K[1, 2]),
            "tx": float(self._translation[0]),
            "ty": float(self._translation[1]),
            "tz": float(self._translation[2]),
            "yaw_deg": float(self._yaw_deg),
            "pitch_deg": float(self._pitch_deg),
            "roll_deg": float(self._roll_deg),
        }

    def update(self, **params: float) -> None:
        for key, value in params.items():
            value = float(value)
            if key == "fx":
                self.K[0, 0] = value
            elif key == "fy":
                self.K[1, 1] = value
            elif key == "cx":
                self.K[0, 2] = value
            elif key == "cy":
                self.K[1, 2] = value
            elif key == "tx":
                self._translation[0] = value
            elif key == "ty":
                self._translation[1] = value
            elif key == "tz":
                self._translation[2] = value
            elif key == "yaw_deg":
                self._yaw_deg = value
            elif key == "pitch_deg":
                self._pitch_deg = value
            elif key == "roll_deg":
                self._roll_deg = value
        self._rebuild()

    def reset(self) -> None:
        self.K[0, 0] = 800.0
        self.K[1, 1] = 800.0
        self.K[0, 2] = 640.0
        self.K[1, 2] = 360.0
        self._translation[:] = np.array([0.0, 0.10, 0.0], dtype=np.float64)
        self._yaw_deg = 0.0
        self._pitch_deg = 0.0
        self._roll_deg = 0.0
        self._rebuild()

    def to_dict(self) -> Dict[str, object]:
        return {
            "intrinsics": {
                "fx": float(self.K[0, 0]),
                "fy": float(self.K[1, 1]),
                "cx": float(self.K[0, 2]),
                "cy": float(self.K[1, 2]),
            },
            "extrinsics": {
                "tx": float(self._translation[0]),
                "ty": float(self._translation[1]),
                "tz": float(self._translation[2]),
                "yaw_deg": float(self._yaw_deg),
                "pitch_deg": float(self._pitch_deg),
                "roll_deg": float(self._roll_deg),
            },
            "rt_matrix": self.RT.tolist(),
        }

    def save(self, path: Optional[Union[str, Path]] = None) -> Path:
        out_path = Path(path) if path is not None else self.DEFAULT_PATH
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return out_path

    def load(self, path: Optional[Union[str, Path]] = None) -> Path:
        in_path = Path(path) if path is not None else self.DEFAULT_PATH
        with open(in_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        intrinsics = payload.get("intrinsics", {})
        extrinsics = payload.get("extrinsics", {})
        self.update(
            fx=float(intrinsics.get("fx", self.K[0, 0])),
            fy=float(intrinsics.get("fy", self.K[1, 1])),
            cx=float(intrinsics.get("cx", self.K[0, 2])),
            cy=float(intrinsics.get("cy", self.K[1, 2])),
            tx=float(extrinsics.get("tx", self._translation[0])),
            ty=float(extrinsics.get("ty", self._translation[1])),
            tz=float(extrinsics.get("tz", self._translation[2])),
            yaw_deg=float(extrinsics.get("yaw_deg", self._yaw_deg)),
            pitch_deg=float(extrinsics.get("pitch_deg", self._pitch_deg)),
            roll_deg=float(extrinsics.get("roll_deg", self._roll_deg)),
        )
        return in_path

    def radar_to_camera(self, point_3d: np.ndarray) -> np.ndarray:
        point = np.asarray(point_3d, dtype=np.float64).reshape(3)
        return (self.R @ point) + self.t

    def project_3d_to_2d(self, point_3d: np.ndarray) -> Tuple[int, int]:
        camera_point = self.radar_to_camera(point_3d)
        if camera_point[2] <= 1e-6:
            return -1, -1
        projected = self.K @ camera_point
        u = projected[0] / projected[2]
        v = projected[1] / projected[2]
        return int(round(u)), int(round(v))

    def project_points(self, points: List[RadarPoint]) -> List[Tuple[RadarPoint, int, int]]:
        projected: List[Tuple[RadarPoint, int, int]] = []
        for point in points:
            u, v = self.project_3d_to_2d(point.position_3d)
            projected.append((point, u, v))
        return projected

    def _rebuild(self) -> None:
        delta = self._euler_zyx_matrix(
            yaw=math.radians(self._yaw_deg),
            pitch=math.radians(self._pitch_deg),
            roll=math.radians(self._roll_deg),
        )
        self.R = self._BASE_ALIGNMENT @ delta
        self.t = self._translation.copy()
        self.RT = np.eye(4, dtype=np.float64)
        self.RT[:3, :3] = self.R
        self.RT[:3, 3] = self.t
        self.P = self.K @ self.RT[:3, :]

    @staticmethod
    def _euler_zyx_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)

        rz = np.array(
            [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        ry = np.array(
            [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]],
            dtype=np.float64,
        )
        rx = np.array(
            [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]],
            dtype=np.float64,
        )
        return rz @ ry @ rx

    @staticmethod
    def _matrix_to_euler_zyx(rotation: np.ndarray) -> Tuple[float, float, float]:
        pitch = math.asin(-max(-1.0, min(1.0, rotation[2, 0])))
        if abs(math.cos(pitch)) < 1e-8:
            yaw = math.atan2(-rotation[0, 1], rotation[1, 1])
            roll = 0.0
        else:
            yaw = math.atan2(rotation[1, 0], rotation[0, 0])
            roll = math.atan2(rotation[2, 1], rotation[2, 2])
        return yaw, pitch, roll
