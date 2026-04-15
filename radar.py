"""
F.A.L.C.O.N. - Radar Sensor Module
==================================

Provides radar integration for sensor-fusion tracking and diagnostic tooling:

RadarPoint
    Dataclass for a single detected radar point (x, y, z, velocity, snr).

RadarFrame
    Normalized frame shape shared by live UART sessions and replay sessions,
    including tracked targets, 3-D extents, and fusion-ready metadata.

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

import copy
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
TLV_COMPRESSED_POINTS = 6
TLV_TRACK_LIST = 7
TLV_TRACK_INDEX = 8
TLV_TRACK_HEIGHT = 9
TLV_PRESENCE = 10
TLV_COMPRESSED_POINTS_LEGACY = 301
TLV_3D_PEOPLE_TRACK_LIST = 1010
TLV_3D_PEOPLE_TRACK_INDEX = 1011
TLV_3D_PEOPLE_TRACK_HEIGHT = 1012
TLV_3D_PEOPLE_COMPRESSED_POINTS = 1020
TLV_3D_PEOPLE_PRESENCE = 1021
TLV_COMPRESSED_POINT_TYPES = {
    TLV_COMPRESSED_POINTS,
    TLV_COMPRESSED_POINTS_LEGACY,
    TLV_3D_PEOPLE_COMPRESSED_POINTS,
}
TLV_TRACK_LIST_TYPES = {TLV_TRACK_LIST, TLV_3D_PEOPLE_TRACK_LIST}
TLV_TRACK_INDEX_TYPES = {TLV_TRACK_INDEX, TLV_3D_PEOPLE_TRACK_INDEX}
TLV_TRACK_HEIGHT_TYPES = {TLV_TRACK_HEIGHT, TLV_3D_PEOPLE_TRACK_HEIGHT}
TLV_PRESENCE_TYPES = {TLV_PRESENCE, TLV_3D_PEOPLE_PRESENCE}
TLV_SIDE_INFO_TYPES = {TLV_SIDE_INFO}
KNOWN_TLV_TYPES = (
    {TLV_DETECTED_OBJECTS}
    | TLV_COMPRESSED_POINT_TYPES
    | TLV_TRACK_LIST_TYPES
    | TLV_TRACK_INDEX_TYPES
    | TLV_TRACK_HEIGHT_TYPES
    | TLV_PRESENCE_TYPES
    | TLV_SIDE_INFO_TYPES
)

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
_TRACK_RECORD_SHORT_STRUCT = struct.Struct("<I15f")
_TRACK_RECORD_LONG_STRUCT = struct.Struct("<I27f")
_TRACK_HEIGHT_STRUCT = struct.Struct("<I2f")

_RUNS_DIR = Path("diagnostics_runs")
_MAX_CLI_LOG_CHARS = 200000
_CLI_LOG_TAIL_CHARS = 4000
_SERIAL_OPEN_RETRIES = 6
_SERIAL_OPEN_DELAY_S = 0.5
_CLI_WAKE_ATTEMPTS = 3
_CLI_WAKE_TIMEOUT_S = 0.8
_CLI_POST_OPEN_SETTLE_S = 1.0
PERSON_BOX_DEFAULT_WIDTH_M = 0.70
PERSON_BOX_DEFAULT_DEPTH_M = 0.70
PERSON_BOX_DEFAULT_HEIGHT_M = 1.80
PERSON_BOX_MIN_WIDTH_M = 0.45
PERSON_BOX_MIN_DEPTH_M = 0.45
PERSON_BOX_MIN_HEIGHT_M = 1.20
PERSON_BOX_MAX_WIDTH_M = 1.15
PERSON_BOX_MAX_DEPTH_M = 1.15
PERSON_BOX_MAX_HEIGHT_M = 2.30


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
class RadarBox3D:
    """Axis-aligned 3-D extent for tracks and configured scene volumes."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    label: str = ""
    kind: str = "track"
    track_id: Optional[int] = None

    @property
    def width(self) -> float:
        return float(self.x_max - self.x_min)

    @property
    def depth(self) -> float:
        return float(self.y_max - self.y_min)

    @property
    def height(self) -> float:
        return float(self.z_max - self.z_min)

    @property
    def center(self) -> Tuple[float, float, float]:
        return (
            float((self.x_min + self.x_max) * 0.5),
            float((self.y_min + self.y_max) * 0.5),
            float((self.z_min + self.z_max) * 0.5),
        )

    @property
    def center_3d(self) -> np.ndarray:
        return np.array(self.center, dtype=np.float64)

    def corners(self) -> List[np.ndarray]:
        return [
            np.array([x, y, z], dtype=np.float64)
            for x in (self.x_min, self.x_max)
            for y in (self.y_min, self.y_max)
            for z in (self.z_min, self.z_max)
        ]


@dataclass
class RadarTrack:
    """Tracked target reported by TI's people-tracking pipeline."""

    track_id: int
    x: float
    y: float
    z: float
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    ax: float = 0.0
    ay: float = 0.0
    az: float = 0.0
    state: int = 0
    confidence: float = 0.0
    bbox: Optional[RadarBox3D] = None
    height_range: Optional[Tuple[float, float]] = None
    associated_point_indexes: List[int] = field(default_factory=list)
    extra_values: List[float] = field(default_factory=list)
    raw_format: str = ""
    bbox_source: str = ""

    @property
    def position(self) -> Tuple[float, float, float]:
        return (float(self.x), float(self.y), float(self.z))

    @property
    def position_3d(self) -> np.ndarray:
        return np.array(self.position, dtype=np.float64)

    @property
    def velocity_3d(self) -> np.ndarray:
        return np.array([self.vx, self.vy, self.vz], dtype=np.float64)

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity_3d))


@dataclass
class RadarSceneMetadata:
    """Config-derived scene and calibration hints shared by live and replay."""

    coordinate_frame: str = "radar_sensor_xyz"
    sensor_height_m: float = 0.0
    azimuth_tilt_deg: float = 0.0
    elevation_tilt_deg: float = 0.0
    config_source: str = ""
    boundary_boxes: List[RadarBox3D] = field(default_factory=list)
    static_boundary_boxes: List[RadarBox3D] = field(default_factory=list)
    presence_boxes: List[RadarBox3D] = field(default_factory=list)

    def all_boxes(self) -> List[RadarBox3D]:
        return (
            list(self.static_boundary_boxes)
            + list(self.boundary_boxes)
            + list(self.presence_boxes)
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
    tracks: List[RadarTrack] = field(default_factory=list)
    track_indexes: List[int] = field(default_factory=list)
    presence: Optional[int] = None
    coordinate_frame: str = "radar_sensor_xyz"
    scene: RadarSceneMetadata = field(default_factory=RadarSceneMetadata)
    calibration: Dict[str, Any] = field(default_factory=dict)

    def fusion_ready_payload(self) -> Dict[str, Any]:
        """Return a camera-fusion-friendly snapshot of this frame."""

        return {
            "frame_number": int(self.frame_number),
            "subframe_number": int(self.subframe_number),
            "timestamp": float(self.timestamp),
            "source_timestamp": float(self.source_timestamp),
            "coordinate_frame": self.coordinate_frame,
            "points": [
                {
                    "x": float(point.x),
                    "y": float(point.y),
                    "z": float(point.z),
                    "velocity": float(point.velocity),
                    "snr": float(point.snr),
                }
                for point in self.points
            ],
            "tracks": [
                {
                    "track_id": int(track.track_id),
                    "position": [float(track.x), float(track.y), float(track.z)],
                    "velocity": [float(track.vx), float(track.vy), float(track.vz)],
                    "acceleration": [float(track.ax), float(track.ay), float(track.az)],
                    "confidence": float(track.confidence),
                    "state": int(track.state),
                    "bbox": None
                    if track.bbox is None
                    else {
                        "x_min": float(track.bbox.x_min),
                        "x_max": float(track.bbox.x_max),
                        "y_min": float(track.bbox.y_min),
                        "y_max": float(track.bbox.y_max),
                        "z_min": float(track.bbox.z_min),
                        "z_max": float(track.bbox.z_max),
                    },
                }
                for track in self.tracks
            ],
            "presence": None if self.presence is None else int(self.presence),
            "scene": asdict(self.scene),
            "calibration": dict(self.calibration),
        }

    def project_to_camera(self, projection: "CameraProjection") -> Dict[str, Any]:
        """Project points and track boxes into image space when calibration is available."""

        points = [
            {
                "xyz": [float(point.x), float(point.y), float(point.z)],
                "uv": list(projection.project_3d_to_2d(point.position_3d)),
                "velocity": float(point.velocity),
                "snr": float(point.snr),
            }
            for point in self.points
        ]
        tracks = []
        for track in self.tracks:
            bbox_2d = projection.project_box_3d(track.bbox) if track.bbox is not None else None
            tracks.append(
                {
                    "track_id": int(track.track_id),
                    "uv_center": list(projection.project_3d_to_2d(track.position_3d)),
                    "uv_bbox": list(bbox_2d) if bbox_2d is not None else None,
                }
            )
        return {"points": points, "tracks": tracks}


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
    tracks: int = 0
    presence: str = ""
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
    frame_debug_path: str = ""
    tracking_debug: Dict[str, Any] = field(default_factory=dict)


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
    scene: RadarSceneMetadata = field(default_factory=RadarSceneMetadata)


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


def clone_radar_frame(frame: Optional[RadarFrame]) -> Optional[RadarFrame]:
    if frame is None:
        return None
    return copy.deepcopy(frame)


def _numeric_stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"min": None, "max": None, "mean": None}
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
    }


def radar_frame_debug_payload(
    frame: RadarFrame,
    *,
    include_points: bool = True,
    max_points: int = 256,
) -> Dict[str, Any]:
    """Compact per-frame snapshot for tuning live people-tracking stability."""

    track_index_counts: Dict[str, int] = defaultdict(int)
    for value in frame.track_indexes:
        key = "unassigned" if int(value) >= 250 else str(int(value))
        track_index_counts[key] += 1

    point_payload = []
    if include_points:
        for point in frame.points[: max(0, int(max_points))]:
            point_payload.append(
                {
                    "x": float(point.x),
                    "y": float(point.y),
                    "z": float(point.z),
                    "velocity": float(point.velocity),
                    "snr": float(point.snr),
                }
            )

    tracks = []
    for track in frame.tracks:
        bbox = None
        if track.bbox is not None:
            bbox = {
                "x_min": float(track.bbox.x_min),
                "x_max": float(track.bbox.x_max),
                "y_min": float(track.bbox.y_min),
                "y_max": float(track.bbox.y_max),
                "z_min": float(track.bbox.z_min),
                "z_max": float(track.bbox.z_max),
                "width": float(track.bbox.width),
                "depth": float(track.bbox.depth),
                "height": float(track.bbox.height),
            }
        tracks.append(
            {
                "track_id": int(track.track_id),
                "position": [float(track.x), float(track.y), float(track.z)],
                "velocity": [float(track.vx), float(track.vy), float(track.vz)],
                "speed": float(track.speed),
                "confidence": float(track.confidence),
                "state": int(track.state),
                "associated_point_count": len(track.associated_point_indexes),
                "associated_point_indexes": list(track.associated_point_indexes[:64]),
                "bbox_source": track.bbox_source,
                "bbox": bbox,
                "raw_format": track.raw_format,
            }
        )

    return {
        "frame_number": int(frame.frame_number),
        "subframe_number": int(frame.subframe_number),
        "timestamp": float(frame.timestamp),
        "source_timestamp": float(frame.source_timestamp),
        "num_detected_obj": int(frame.num_detected_obj),
        "num_tlvs": int(frame.num_tlvs),
        "point_count": len(frame.points),
        "track_count": len(frame.tracks),
        "presence": None if frame.presence is None else int(frame.presence),
        "track_indexes_count": len(frame.track_indexes),
        "track_index_counts": dict(sorted(track_index_counts.items())),
        "point_stats": {
            "x": _numeric_stats([point.x for point in frame.points]),
            "y": _numeric_stats([point.y for point in frame.points]),
            "z": _numeric_stats([point.z for point in frame.points]),
            "velocity": _numeric_stats([point.velocity for point in frame.points]),
            "snr": _numeric_stats([point.snr for point in frame.points]),
        },
        "points_truncated": len(frame.points) > len(point_payload),
        "points": point_payload,
        "tracks": tracks,
    }


class TrackingStabilityMonitor:
    """Accumulates live-run stability stats without touching the render loop."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total_frames = 0
        self.frames_with_points = 0
        self.frames_with_tracks = 0
        self.frames_with_presence = 0
        self.frames_presence_without_tracks = 0
        self.frames_points_without_tracks = 0
        self.current_no_track_streak = 0
        self.max_no_track_streak = 0
        self.single_track_id_switches = 0
        self.track_gap_events = 0
        self.track_gap_frames = 0
        self._last_single_track_id: Optional[int] = None
        self._last_active_ids: List[int] = []
        self._point_counts: deque[int] = deque(maxlen=240)
        self._track_first_frame: Dict[int, int] = {}
        self._track_last_frame: Dict[int, int] = {}
        self._track_seen_frames: Dict[int, int] = defaultdict(int)

    def update(self, frame: RadarFrame) -> None:
        self.total_frames += 1
        point_count = len(frame.points)
        track_ids = sorted(int(track.track_id) for track in frame.tracks)
        presence_active = frame.presence not in (None, 0)

        self._point_counts.append(point_count)
        if point_count > 0:
            self.frames_with_points += 1
        if presence_active:
            self.frames_with_presence += 1
        if track_ids:
            self.frames_with_tracks += 1
            self.current_no_track_streak = 0
        else:
            self.current_no_track_streak += 1
            self.max_no_track_streak = max(self.max_no_track_streak, self.current_no_track_streak)
        if presence_active and not track_ids:
            self.frames_presence_without_tracks += 1
        if point_count > 0 and not track_ids:
            self.frames_points_without_tracks += 1

        frame_number = int(frame.frame_number)
        for track_id in track_ids:
            if track_id not in self._track_first_frame:
                self._track_first_frame[track_id] = frame_number
            last_frame = self._track_last_frame.get(track_id)
            if last_frame is not None and frame_number > last_frame + 1:
                self.track_gap_events += 1
                self.track_gap_frames += frame_number - last_frame - 1
            self._track_last_frame[track_id] = frame_number
            self._track_seen_frames[track_id] += 1

        if len(track_ids) == 1:
            track_id = track_ids[0]
            if self._last_single_track_id is not None and self._last_single_track_id != track_id:
                self.single_track_id_switches += 1
            self._last_single_track_id = track_id
        elif len(track_ids) > 1:
            self._last_single_track_id = None
        self._last_active_ids = track_ids

    def to_dict(self) -> Dict[str, Any]:
        point_counts = list(self._point_counts)
        total = max(self.total_frames, 1)
        frames_with_presence = max(self.frames_with_presence, 1)
        tracks: Dict[str, Dict[str, int]] = {}
        for track_id in sorted(self._track_seen_frames):
            tracks[str(track_id)] = {
                "first_frame": int(self._track_first_frame.get(track_id, 0)),
                "last_frame": int(self._track_last_frame.get(track_id, 0)),
                "seen_frames": int(self._track_seen_frames.get(track_id, 0)),
            }

        return {
            "total_frames": int(self.total_frames),
            "frames_with_points": int(self.frames_with_points),
            "frames_with_tracks": int(self.frames_with_tracks),
            "frames_with_presence": int(self.frames_with_presence),
            "frames_presence_without_tracks": int(self.frames_presence_without_tracks),
            "frames_points_without_tracks": int(self.frames_points_without_tracks),
            "track_coverage_pct": round(100.0 * self.frames_with_tracks / total, 2),
            "presence_track_agreement_pct": round(
                100.0 * (self.frames_with_presence - self.frames_presence_without_tracks) / frames_with_presence,
                2,
            ),
            "current_no_track_streak": int(self.current_no_track_streak),
            "max_no_track_streak": int(self.max_no_track_streak),
            "active_track_ids": list(self._last_active_ids),
            "unique_track_ids": sorted(int(track_id) for track_id in self._track_seen_frames),
            "single_track_id_switches": int(self.single_track_id_switches),
            "track_gap_events": int(self.track_gap_events),
            "track_gap_frames": int(self.track_gap_frames),
            "recent_point_count": {
                "min": min(point_counts) if point_counts else 0,
                "max": max(point_counts) if point_counts else 0,
                "mean": round(float(sum(point_counts) / len(point_counts)), 2) if point_counts else 0.0,
            },
            "tracks": tracks,
        }


def scene_metadata_from_cfg_lines(
    lines: Sequence[str],
    *,
    config_source: str = "",
) -> RadarSceneMetadata:
    """Extract scene volumes and sensor pose hints from cfg lines."""

    scene = RadarSceneMetadata(config_source=config_source)
    for raw_line in lines:
        line = str(raw_line).strip()
        if not line or line.startswith("%"):
            continue
        parts = line.split()
        cmd = parts[0]
        try:
            values = [float(value) for value in parts[1:]]
        except ValueError:
            continue

        if cmd == "sensorPosition" and len(values) >= 3:
            scene.sensor_height_m = float(values[0])
            scene.azimuth_tilt_deg = float(values[1])
            scene.elevation_tilt_deg = float(values[2])
            continue

        if cmd in {"boundaryBox", "staticBoundaryBox", "presenceBoundaryBox"} and len(values) >= 6:
            box = RadarBox3D(
                x_min=float(values[0]),
                x_max=float(values[1]),
                y_min=float(values[2]),
                y_max=float(values[3]),
                z_min=float(values[4]),
                z_max=float(values[5]),
                label=cmd,
                kind=cmd,
            )
            if cmd == "boundaryBox":
                scene.boundary_boxes.append(box)
            elif cmd == "staticBoundaryBox":
                scene.static_boundary_boxes.append(box)
            else:
                scene.presence_boxes.append(box)
    return scene


def scene_metadata_from_cfg_path(path: Union[str, Path]) -> RadarSceneMetadata:
    cfg_path = Path(path)
    try:
        lines = cfg_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return RadarSceneMetadata(config_source=str(cfg_path))
    return scene_metadata_from_cfg_lines(lines, config_source=str(cfg_path))


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


def _location_interface_number(hwid: str) -> str:
    location = _extract_hwid_value(hwid, "LOCATION")
    if not location:
        return ""
    return location.rsplit(":", 1)[-1].lower()


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
    location_interface = _location_interface_number(info.hwid)
    if ("xds110" in text or "0451:bef3" in text) and location_interface == "1.0":
        return "config"
    if ("xds110" in text or "0451:bef3" in text) and location_interface == "1.3":
        return "data"
    # On this project's CP2105-based adapter, the Enhanced interface is the
    # command CLI and the Standard interface is the high-rate data stream.
    if "enhanced com port" in text:
        return "config"
    if "standard com port" in text:
        return "data"
    if "application/user uart" in text:
        return "config"
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

    if magic_hits > 0 and frames >= 3:
        if plots_updating:
            return (
                HEALTH_HEALTHY,
                "Config succeeded, frames are parsing, and the live plots are updating.",
            )
        return (
            HEALTH_HEALTHY,
            "Config succeeded and frames are parsing. GUI plot rendering has not reported in this session.",
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
            return (
                HEALTH_HEALTHY,
                "Frames are parsing during the startup window; GUI plot rendering has not reported yet.",
            )

    if rx_bytes == 0:
        return HEALTH_NO_DATA, "Config succeeded, but no data bytes arrived within 5 seconds."

    if frames < 3:
        return HEALTH_NO_DATA, "Frames are arriving, but the viability threshold has not been met yet."

    if not plots_updating:
        return (
            HEALTH_HEALTHY,
            "Frames parsed successfully; GUI plot rendering has not reported yet.",
        )

    return HEALTH_NO_DATA, "Config succeeded, but the logging stream is still inconclusive."


def _calibration_from_scene(scene: RadarSceneMetadata) -> Dict[str, Any]:
    return {
        "coordinate_frame": scene.coordinate_frame,
        "sensor_pose": {
            "height_m": float(scene.sensor_height_m),
            "azimuth_tilt_deg": float(scene.azimuth_tilt_deg),
            "elevation_tilt_deg": float(scene.elevation_tilt_deg),
        },
        "camera_projection_hook": {
            "extrinsics_key": "radar_to_camera",
            "points_ready": True,
            "tracks_ready": True,
            "bboxes_ready": True,
        },
    }


def _track_is_plausible(track: RadarTrack) -> bool:
    values = [
        track.x,
        track.y,
        track.z,
        track.vx,
        track.vy,
        track.vz,
        track.ax,
        track.ay,
        track.az,
        track.confidence,
    ]
    if not all(np.isfinite(value) for value in values):
        return False
    if track.track_id < 0 or track.track_id > 4096:
        return False
    if max(abs(track.x), abs(track.y), abs(track.z)) > 100.0:
        return False
    if max(abs(track.vx), abs(track.vy), abs(track.vz)) > 50.0:
        return False
    if max(abs(track.ax), abs(track.ay), abs(track.az)) > 100.0:
        return False
    if abs(track.confidence) > 10.0:
        return False
    return True


def _parse_target_list_payload(payload: bytes) -> Optional[List[RadarTrack]]:
    candidates: List[Tuple[struct.Struct, str]] = [
        (_TRACK_RECORD_SHORT_STRUCT, "short"),
        (_TRACK_RECORD_LONG_STRUCT, "legacy"),
    ]

    for record_struct, record_format in candidates:
        if len(payload) == 0 or len(payload) % record_struct.size != 0:
            continue

        tracks: List[RadarTrack] = []
        valid = True
        for offset in range(0, len(payload), record_struct.size):
            values = record_struct.unpack_from(payload, offset)
            track_id = int(values[0])
            floats = [float(value) for value in values[1:]]

            if record_format == "short":
                track = RadarTrack(
                    track_id=track_id,
                    x=floats[0],
                    y=floats[1],
                    z=floats[2],
                    vx=floats[3],
                    vy=floats[4],
                    vz=floats[5],
                    ax=floats[6],
                    ay=floats[7],
                    az=floats[8],
                    state=int(round(floats[9])),
                    confidence=floats[10],
                    extra_values=floats[11:],
                    raw_format=record_format,
                )
            else:
                track = RadarTrack(
                    track_id=track_id,
                    x=floats[0],
                    y=floats[1],
                    z=floats[6],
                    vx=floats[2],
                    vy=floats[3],
                    vz=floats[7],
                    ax=floats[4],
                    ay=floats[5],
                    az=floats[8],
                    state=0,
                    confidence=max(0.0, floats[-1]),
                    extra_values=floats[9:],
                    raw_format=record_format,
                )
            if not _track_is_plausible(track):
                valid = False
                break
            tracks.append(track)

        if valid:
            return tracks
    return None


def _parse_height_payload(payload: bytes) -> Optional[Dict[int, Tuple[float, float]]]:
    if len(payload) == 0 or len(payload) % _TRACK_HEIGHT_STRUCT.size != 0:
        return None
    heights: Dict[int, Tuple[float, float]] = {}
    for offset in range(0, len(payload), _TRACK_HEIGHT_STRUCT.size):
        track_id, z_max, z_min = _TRACK_HEIGHT_STRUCT.unpack_from(payload, offset)
        if (
            track_id > 4096
            or not (np.isfinite(z_max) and np.isfinite(z_min))
            or abs(z_max) > 50.0
            or abs(z_min) > 50.0
        ):
            return None
        heights[int(track_id)] = (float(z_min), float(z_max))
    return heights


def _parse_target_index_payload(payload: bytes) -> Optional[List[int]]:
    if not payload:
        return []
    indexes = [int(value) for value in payload]
    if any(value < 0 or value > 255 for value in indexes):
        return None
    return indexes


def _clamped_extent_from_points(
    values: Sequence[float],
    center: float,
    *,
    min_size: float,
    max_size: float,
) -> Tuple[float, float]:
    """Build a stable human-scale extent from noisy associated radar points."""

    if values:
        arr = np.asarray(values, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
    else:
        finite = np.asarray([], dtype=np.float64)

    if finite.size >= 3:
        low, high = np.percentile(finite, [10.0, 90.0])
    elif finite.size:
        low = float(finite.min())
        high = float(finite.max())
    else:
        low = high = float(center)

    raw_size = max(float(high - low), float(min_size))
    size = min(max(raw_size, float(min_size)), float(max_size))
    midpoint = float((low + high) * 0.5) if finite.size else float(center)
    # Keep the visual box anchored to the tracker state; associated points can
    # be sparse or include clutter, especially with two people close together.
    midpoint = float((midpoint * 0.35) + (float(center) * 0.65))
    return midpoint - size * 0.5, midpoint + size * 0.5


def _clamp_height_range(
    z_center: float,
    height_range: Optional[Tuple[float, float]],
    associated_points: Sequence[RadarPoint],
) -> Tuple[float, float]:
    if height_range is not None:
        z_min, z_max = float(height_range[0]), float(height_range[1])
    elif associated_points:
        zs = np.asarray([point.z for point in associated_points], dtype=np.float64)
        finite = zs[np.isfinite(zs)]
        if finite.size >= 3:
            z_min, z_max = np.percentile(finite, [5.0, 95.0])
        elif finite.size:
            z_min = float(finite.min())
            z_max = float(finite.max())
        else:
            z_min = float(z_center) - PERSON_BOX_DEFAULT_HEIGHT_M * 0.5
            z_max = float(z_center) + PERSON_BOX_DEFAULT_HEIGHT_M * 0.5
    else:
        z_min = float(z_center) - PERSON_BOX_DEFAULT_HEIGHT_M * 0.5
        z_max = float(z_center) + PERSON_BOX_DEFAULT_HEIGHT_M * 0.5

    height = max(float(z_max - z_min), PERSON_BOX_MIN_HEIGHT_M)
    height = min(height, PERSON_BOX_MAX_HEIGHT_M)
    midpoint = float((z_min + z_max) * 0.5)
    if not np.isfinite(midpoint):
        midpoint = float(z_center)
    return midpoint - height * 0.5, midpoint + height * 0.5


def _build_track_bbox(
    track: RadarTrack,
    *,
    associated_points: Sequence[RadarPoint],
    height_range: Optional[Tuple[float, float]],
) -> Tuple[Optional[RadarBox3D], str]:
    x_center = float(track.x)
    y_center = float(track.y)
    z_center = float(track.z)

    z_min, z_max = _clamp_height_range(z_center, height_range, associated_points)

    if associated_points:
        xs = [point.x for point in associated_points]
        ys = [point.y for point in associated_points]
        x_min, x_max = _clamped_extent_from_points(
            xs,
            x_center,
            min_size=PERSON_BOX_MIN_WIDTH_M,
            max_size=PERSON_BOX_MAX_WIDTH_M,
        )
        y_min, y_max = _clamped_extent_from_points(
            ys,
            y_center,
            min_size=PERSON_BOX_MIN_DEPTH_M,
            max_size=PERSON_BOX_MAX_DEPTH_M,
        )
        return (
            RadarBox3D(
                x_min=float(x_min),
                x_max=float(x_max),
                y_min=float(y_min),
                y_max=float(y_max),
                z_min=float(z_min),
                z_max=float(z_max),
                label=f"ID {track.track_id}",
                kind="track",
                track_id=track.track_id,
            ),
            "associated_points_clamped",
        )

    half_width = PERSON_BOX_DEFAULT_WIDTH_M * 0.5
    half_depth = PERSON_BOX_DEFAULT_DEPTH_M * 0.5
    return (
        RadarBox3D(
            x_min=float(x_center - half_width),
            x_max=float(x_center + half_width),
            y_min=float(y_center - half_depth),
            y_max=float(y_center + half_depth),
            z_min=float(z_min),
            z_max=float(z_max),
            label=f"ID {track.track_id}",
            kind="track",
            track_id=track.track_id,
        ),
        "default_person_extent",
    )


def finalize_tracks(
    tracks: Sequence[RadarTrack],
    *,
    points_for_indexing: Sequence[RadarPoint],
    track_indexes: Sequence[int],
    heights_by_track: Optional[Dict[int, Tuple[float, float]]] = None,
) -> List[RadarTrack]:
    heights_by_track = heights_by_track or {}
    point_indexes_by_track: Dict[int, List[int]] = defaultdict(list)
    for point_index, track_id in enumerate(track_indexes):
        if track_id >= 250:
            continue
        point_indexes_by_track[int(track_id)].append(point_index)

    finalized: List[RadarTrack] = []
    for track in tracks:
        cloned = copy.deepcopy(track)
        associated_indexes = point_indexes_by_track.get(cloned.track_id, [])
        cloned.associated_point_indexes = list(associated_indexes)
        cloned.height_range = heights_by_track.get(cloned.track_id)

        associated_points = [
            points_for_indexing[index]
            for index in associated_indexes
            if 0 <= index < len(points_for_indexing)
        ]
        bbox, bbox_source = _build_track_bbox(
            cloned,
            associated_points=associated_points,
            height_range=cloned.height_range,
        )
        cloned.bbox = bbox
        cloned.bbox_source = bbox_source
        finalized.append(cloned)
    return finalized


def parse_replay_tracks(
    frame_data: Dict[str, Any],
    *,
    previous_points: Sequence[RadarPoint],
) -> Tuple[List[RadarTrack], List[int], Optional[int]]:
    raw_track_rows = frame_data.get("trackData", [])
    raw_height_rows = frame_data.get("heightData", [])
    track_indexes = [
        int(value)
        for value in frame_data.get("trackIndexes", [])
        if isinstance(value, (int, float))
    ]

    heights_by_track: Dict[int, Tuple[float, float]] = {}
    for row in raw_height_rows:
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            continue
        track_id = int(row[0])
        z_max = float(row[1])
        z_min = float(row[2])
        heights_by_track[track_id] = (z_min, z_max)

    tracks: List[RadarTrack] = []
    for row in raw_track_rows:
        if not isinstance(row, (list, tuple)) or len(row) < 12:
            continue
        floats = [float(value) for value in row]
        track = RadarTrack(
            track_id=int(round(floats[0])),
            x=floats[1],
            y=floats[2],
            z=floats[3],
            vx=floats[4],
            vy=floats[5],
            vz=floats[6],
            ax=floats[7],
            ay=floats[8],
            az=floats[9],
            state=int(round(floats[10])),
            confidence=floats[11],
            extra_values=floats[12:],
            raw_format="replay",
        )
        if _track_is_plausible(track):
            tracks.append(track)

    finalized = finalize_tracks(
        tracks,
        points_for_indexing=previous_points,
        track_indexes=track_indexes,
        heights_by_track=heights_by_track,
    )
    presence = None
    raw_presence = frame_data.get("presence")
    if isinstance(raw_presence, (int, float)):
        presence = int(raw_presence)
    return finalized, track_indexes, presence


def load_replay_capture(path: Union[str, Path]) -> ReplayCapture:
    """Load a TI Industrial Visualizer replay JSON into normalized frames."""

    replay_path = Path(path)
    payload = json.loads(replay_path.read_text(encoding="utf-8"))
    entries = payload.get("data", [])
    cfg_lines = [str(line).rstrip("\n") for line in payload.get("cfg", [])]
    scene = scene_metadata_from_cfg_lines(cfg_lines, config_source=str(replay_path))

    frames: List[RadarFrame] = []
    schedule_s: List[float] = []
    previous_points: List[RadarPoint] = []

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

        tracks, track_indexes, presence = parse_replay_tracks(
            frame_data,
            previous_points=previous_points,
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
                tracks=tracks,
                track_indexes=list(track_indexes),
                presence=presence,
                coordinate_frame=scene.coordinate_frame,
                scene=copy.deepcopy(scene),
                calibration=_calibration_from_scene(scene),
            )
        )
        previous_points = list(points)

    return ReplayCapture(
        path=str(replay_path),
        cfg_lines=cfg_lines,
        demo=str(payload.get("demo", "")),
        device=str(payload.get("device", "")),
        frames=frames,
        schedule_s=schedule_s,
        scene=scene,
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
        mount_geometry: Optional[Dict[str, Any]] = None,
    ):
        self._config_port_path = config_port
        self._data_port_path = data_port
        self._config_path = config_path
        self._config_baud = config_baud
        self._data_baud = data_baud
        # Optional mount overrides applied before sending each command.
        # Supported keys:
        #   height_m, azimuth_tilt_deg, elevation_tilt_deg
        #   boundary_box, static_boundary_box, presence_boundary_box
        #       (each a 6-tuple: x_min x_max y_min y_max z_min z_max)
        self._mount_geometry: Dict[str, Any] = dict(mount_geometry or {})

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
        self._frame_debug_path: Optional[Path] = None
        self._frame_debug_file: Optional[Any] = None
        self._last_frame_debug_flush = 0.0
        self._tracking_debug = TrackingStabilityMonitor()
        self._scene_metadata = RadarSceneMetadata(config_source=config_path)
        self._previous_points_for_indexes: List[RadarPoint] = []

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
            tracks=len(latest_frame.tracks) if latest_frame is not None else 0,
            presence=""
            if latest_frame is None or latest_frame.presence is None
            else str(latest_frame.presence),
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
            frame_debug_path=str(self._frame_debug_path) if self._frame_debug_path else "",
            tracking_debug=self._tracking_debug.to_dict(),
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
                self._close_frame_debug()
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
            self._close_frame_debug()
            return False

        self._scene_metadata = scene_metadata_from_cfg_path(self._config_path)

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
            self._close_frame_debug()
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
            self._close_frame_debug()
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
        self._close_frame_debug()
        logger.info("IWR6843 radar stopped")

    def get_point_cloud(self) -> List[RadarPoint]:
        """Return the latest detected radar points."""
        with self._lock:
            if not self._frame_buffer:
                return []
            return list(self._frame_buffer[-1].points)

    def get_tracks(self) -> List[RadarTrack]:
        """Return the latest people-tracking targets."""
        with self._lock:
            if not self._frame_buffer:
                return []
            return copy.deepcopy(self._frame_buffer[-1].tracks)

    def get_latest_frame(self) -> Optional[RadarFrame]:
        with self._lock:
            if not self._frame_buffer:
                return None
            frame = self._frame_buffer[-1]
            return clone_radar_frame(frame)

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
        latest_frame = self.get_latest_frame()
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
            "active_track_ids": []
            if latest_frame is None
            else [int(track.track_id) for track in latest_frame.tracks],
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
        self._previous_points_for_indexes = []
        self._probe_failed_stage = ""
        self._probe_log_tail = ""
        self._tracking_debug.reset()

    def _start_run_artifacts(self) -> None:
        self._close_frame_debug()
        _RUNS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        suffix = f"{int((time.time() % 1.0) * 1000):03d}"
        self._run_dir = _RUNS_DIR / f"{timestamp}_{suffix}"
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._report_path = self._run_dir / "session_report.json"
        self._frame_debug_path = self._run_dir / "frames.jsonl"
        self._frame_debug_file = self._frame_debug_path.open("a", encoding="utf-8", buffering=1)
        self._write_frame_debug_event(
            {
                "type": "session_start",
                "timestamp": time.time(),
                "config_port": self._config_port_path,
                "data_port": self._data_port_path,
                "config_path": self._config_path,
                "config_baud": self._config_baud,
                "data_baud": self._data_baud,
            }
        )

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

    def _write_frame_debug_event(self, payload: Dict[str, Any]) -> None:
        if self._frame_debug_file is None:
            return
        try:
            self._frame_debug_file.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
            now = time.time()
            if now - self._last_frame_debug_flush >= 1.0:
                self._frame_debug_file.flush()
                self._last_frame_debug_flush = now
        except Exception as exc:
            logger.warning("Failed to write radar frame debug event: %s", exc)
            self._close_frame_debug()

    def _record_frame_debug(self, frame: RadarFrame) -> None:
        self._tracking_debug.update(frame)
        payload = radar_frame_debug_payload(frame, include_points=True, max_points=256)
        payload["type"] = "frame"
        self._write_frame_debug_event(payload)

    def _close_frame_debug(self) -> None:
        if self._frame_debug_file is None:
            return
        try:
            self._frame_debug_file.flush()
            self._frame_debug_file.close()
        except Exception:
            pass
        finally:
            self._frame_debug_file = None

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
            line = self._apply_mount_geometry(line)
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

    def _apply_mount_geometry(self, line: str) -> str:
        """Rewrite a cfg command line using runtime mount geometry overrides."""
        if not self._mount_geometry:
            return line
        parts = line.split()
        if not parts:
            return line
        cmd = parts[0]
        geom = self._mount_geometry

        if cmd == "sensorPosition":
            height = geom.get("height_m")
            az = geom.get("azimuth_tilt_deg", 0.0)
            el = geom.get("elevation_tilt_deg", 0.0)
            if height is not None:
                return f"sensorPosition {float(height):.4f} {float(az):.4f} {float(el):.4f}"

        box_key_map = {
            "boundaryBox": "boundary_box",
            "staticBoundaryBox": "static_boundary_box",
            "presenceBoundaryBox": "presence_boundary_box",
        }
        key = box_key_map.get(cmd)
        if key and key in geom:
            box = geom[key]
            if len(box) >= 6:
                vals = " ".join(f"{float(v):.4f}" for v in box[:6])
                return f"{cmd} {vals}"
        return line

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
                    self._record_frame_debug(frame)

            except Exception as exc:
                if self._running:
                    self._last_parse_error = str(exc)
                    logger.warning("Radar read error: %s", exc)
                time.sleep(0.1)

    def _parse_packet(self, packet: bytes) -> Optional[RadarFrame]:
        parse_error = ""
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
        tracks: List[RadarTrack] = []
        heights_by_track: Dict[int, Tuple[float, float]] = {}
        track_indexes: List[int] = []
        presence: Optional[int] = None
        offset = HEADER_SIZE

        for _ in range(num_tlvs):
            if offset + TLV_HEADER_SIZE > len(packet):
                parse_error = "truncated TLV header"
                break

            tlv_type, tlv_length = _TLV_STRUCT.unpack_from(packet, offset)
            offset += TLV_HEADER_SIZE

            if tlv_length < 0 or offset + tlv_length > len(packet):
                parse_error = f"bad TLV len {tlv_length} for type {tlv_type}"
                break

            payload = packet[offset : offset + tlv_length]
            offset += tlv_length

            parsed_points = self._parse_tlv_points(tlv_type, payload)
            if parsed_points is not None:
                points = parsed_points
                continue

            parsed_tracks = self._parse_track_list_tlv(tlv_type, payload)
            if parsed_tracks is not None:
                tracks = parsed_tracks
                continue

            parsed_heights = self._parse_track_height_tlv(tlv_type, payload)
            if parsed_heights is not None:
                heights_by_track = parsed_heights
                continue

            parsed_indexes = self._parse_track_indexes_tlv(
                tlv_type,
                payload,
                point_count=len(points) or len(self._previous_points_for_indexes),
            )
            if parsed_indexes is not None:
                track_indexes = parsed_indexes
                continue

            parsed_presence = self._parse_presence_tlv(tlv_type, payload)
            if parsed_presence is not None:
                presence = parsed_presence
                continue

            if tlv_type not in TLV_SIDE_INFO_TYPES:
                self._unknown_tlv_counts[int(tlv_type)] += 1
                lengths = self._unknown_tlv_lengths[int(tlv_type)]
                lengths.append(int(tlv_length))
                if len(lengths) > 10:
                    del lengths[:-10]

        finalized_tracks = finalize_tracks(
            tracks,
            points_for_indexing=points or self._previous_points_for_indexes,
            track_indexes=track_indexes,
            heights_by_track=heights_by_track,
        )
        frame = RadarFrame(
            frame_number=frame_number,
            subframe_number=subframe_number,
            num_detected_obj=num_detected_obj,
            num_tlvs=num_tlvs,
            points=points,
            timestamp=time.time(),
            source_timestamp=time.time(),
            tracks=finalized_tracks,
            track_indexes=list(track_indexes),
            presence=presence,
            coordinate_frame=self._scene_metadata.coordinate_frame,
            scene=copy.deepcopy(self._scene_metadata),
            calibration=_calibration_from_scene(self._scene_metadata),
        )
        self._previous_points_for_indexes = list(points)
        self._last_parse_error = parse_error
        return frame

    def _parse_tlv_points(
        self,
        tlv_type: int,
        payload: bytes,
    ) -> Optional[List[RadarPoint]]:
        if tlv_type == TLV_DETECTED_OBJECTS and len(payload) % _RAW_POINT_STRUCT.size == 0:
            return self._parse_raw_cartesian_points(payload)

        if tlv_type not in (
            *TLV_COMPRESSED_POINT_TYPES,
            TLV_DETECTED_OBJECTS,
        ) and len(payload) >= _TRACK_RECORD_SHORT_STRUCT.size:
            return None

        compressed = self._parse_compressed_points(payload)
        if compressed is not None:
            return compressed

        return None

    def _parse_track_list_tlv(
        self,
        tlv_type: int,
        payload: bytes,
    ) -> Optional[List[RadarTrack]]:
        if tlv_type in TLV_TRACK_LIST_TYPES:
            return _parse_target_list_payload(payload)
        if tlv_type in KNOWN_TLV_TYPES:
            return None
        if len(payload) % _TRACK_RECORD_SHORT_STRUCT.size != 0 and len(payload) % _TRACK_RECORD_LONG_STRUCT.size != 0:
            return None
        return _parse_target_list_payload(payload)

    def _parse_track_height_tlv(
        self,
        tlv_type: int,
        payload: bytes,
    ) -> Optional[Dict[int, Tuple[float, float]]]:
        if tlv_type in TLV_TRACK_HEIGHT_TYPES:
            return _parse_height_payload(payload)
        if tlv_type in KNOWN_TLV_TYPES:
            return None
        if len(payload) % _TRACK_HEIGHT_STRUCT.size != 0:
            return None
        return _parse_height_payload(payload)

    def _parse_track_indexes_tlv(
        self,
        tlv_type: int,
        payload: bytes,
        *,
        point_count: int,
    ) -> Optional[List[int]]:
        expected_counts = {count for count in (point_count, len(self._previous_points_for_indexes)) if count > 0}
        if tlv_type in TLV_TRACK_INDEX_TYPES:
            return _parse_target_index_payload(payload)
        if tlv_type in KNOWN_TLV_TYPES:
            return None
        if not expected_counts or len(payload) not in expected_counts:
            return None
        return _parse_target_index_payload(payload)

    def _parse_presence_tlv(
        self,
        tlv_type: int,
        payload: bytes,
    ) -> Optional[int]:
        if len(payload) != 4:
            return None
        if tlv_type in TLV_PRESENCE_TYPES:
            return int(struct.unpack_from("<I", payload, 0)[0])
        if tlv_type in KNOWN_TLV_TYPES:
            return None
        value = struct.unpack_from("<I", payload, 0)[0]
        if value > 16:
            return None
        return int(value)

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
                azimuth_rad=azimuth,
                elevation_rad=elevation,
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
        azimuth_rad: float,
        elevation_rad: float,
        distance_m: float,
    ) -> Tuple[float, float, float]:
        cos_el = math.cos(elevation_rad)

        x = distance_m * math.sin(azimuth_rad) * cos_el
        y = distance_m * math.cos(azimuth_rad) * cos_el
        z = distance_m * math.sin(elevation_rad)
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
        return clone_radar_frame(frame)

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
            tracks=len(frame.tracks) if frame is not None else 0,
            presence="" if frame is None or frame.presence is None else str(frame.presence),
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

    def get_tracks(self) -> List[RadarTrack]:
        frame = self.latest_frame()
        return [] if frame is None else copy.deepcopy(frame.tracks)

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
        target = points[0] if points else RadarPoint(0.0, self.depth_y, self.height_z, 0.0)
        bbox = RadarBox3D(
            x_min=target.x - 0.3,
            x_max=target.x + 0.3,
            y_min=target.y - 0.3,
            y_max=target.y + 0.3,
            z_min=target.z,
            z_max=target.z + 1.7,
            label="ID 0",
            kind="track",
            track_id=0,
        )
        scene = RadarSceneMetadata()
        return RadarFrame(
            frame_number=self._frame_count,
            subframe_number=0,
            num_detected_obj=len(points),
            num_tlvs=1,
            points=points,
            timestamp=time.time(),
            source_timestamp=time.time(),
            tracks=[
                RadarTrack(
                    track_id=0,
                    x=target.x,
                    y=target.y,
                    z=target.z + 0.85,
                    bbox=bbox,
                    bbox_source="mock",
                )
            ],
            coordinate_frame=scene.coordinate_frame,
            scene=scene,
            calibration=_calibration_from_scene(scene),
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
            tracks=len(latest_frame.tracks) if latest_frame is not None else 0,
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

    def get_tracks(self) -> List[RadarTrack]:
        frame = self.latest_frame()
        return [] if frame is None else copy.deepcopy(frame.tracks)

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

    def project_box_3d(self, box: Optional[RadarBox3D]) -> Optional[Tuple[int, int, int, int]]:
        if box is None:
            return None
        projected = [
            self.project_3d_to_2d(corner)
            for corner in box.corners()
        ]
        valid = [(u, v) for u, v in projected if u >= 0 and v >= 0]
        if not valid:
            return None
        us = [u for u, _ in valid]
        vs = [v for _, v in valid]
        return (min(us), min(vs), max(us), max(vs))

    def project_tracks(self, tracks: List[RadarTrack]) -> List[Dict[str, Any]]:
        projected: List[Dict[str, Any]] = []
        for track in tracks:
            projected.append(
                {
                    "track_id": int(track.track_id),
                    "center_uv": list(self.project_3d_to_2d(track.position_3d)),
                    "bbox_uv": None
                    if track.bbox is None
                    else list(self.project_box_3d(track.bbox) or ()),
                }
            )
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
