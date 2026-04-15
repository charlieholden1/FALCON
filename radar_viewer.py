#!/usr/bin/env python3
"""
Primary radar viewer entrypoint.

On Ubuntu/Orin, this prefers a Qt + PyQtGraph frontend for smoother live
rendering and a more production-style engineering UI. The older Tk/Matplotlib
viewer is preserved in `radar_viewer_tk.py` as a compatibility fallback.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib import cm

from radar import (
    HEALTH_CONFIG_FAILED,
    HEALTH_HEALTHY,
    HEALTH_NO_DATA,
    HEALTH_REPLAY_MODE,
    HEALTH_STREAMING_UNPARSED,
    FrameSource,
    IWR6843Driver,
    RadarBox3D,
    RadarFrame,
    RadarSessionState,
    RadarTrack,
    ReplayRadarSource,
    SerialPortInfo,
    discover_serial_ports,
    radar_frame_debug_payload,
    suggest_serial_port_pairs,
)


DEFAULT_POLL_MS = 75
DEFAULT_RENDER_FPS = 18.0
DEFAULT_VIEW_DISTANCE = 16.0
DEFAULT_VIEW_ELEVATION = 18.0
DEFAULT_VIEW_AZIMUTH = -62.0
DEFAULT_VIEW_CENTER = np.array([0.0, 4.0, 1.2], dtype=np.float32)
DEFAULT_ROOM_WIDTH_M = 8.0
DEFAULT_ROOM_DEPTH_M = 8.0
DEFAULT_ROOM_HEIGHT_M = 3.2
ROOM_GUIDE_SPACING_M = 1.0
ROOM_MONO_LINE_COLOR = "#506777"
ROOM_MONO_GUIDE_COLOR = "#8da0ad"
ROOM_MONO_AXIS_COLOR = "#667b88"

_QT_IMPORT_ERROR: Optional[BaseException] = None
_HAS_QT = False


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _qt_runtime_dir() -> Path:
    return _repo_root() / ".qt-runtime" / "lib"


def _prepend_env_path(key: str, value: str) -> None:
    existing = os.environ.get(key, "")
    parts = [part for part in existing.split(os.pathsep) if part]
    if value not in parts:
        os.environ[key] = os.pathsep.join([value, *parts]) if parts else value


def _bootstrap_qt_runtime() -> None:
    runtime_dir = _qt_runtime_dir()
    if not runtime_dir.exists():
        return
    _prepend_env_path("LD_LIBRARY_PATH", str(runtime_dir))
    for lib_name in ("libdouble-conversion.so.3", "libxcb-cursor.so.0"):
        lib_path = runtime_dir / lib_name
        if lib_path.exists():
            ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)


try:
    _bootstrap_qt_runtime()
    from PySide6 import QtCore, QtGui, QtWidgets
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl

    _HAS_QT = True
except Exception as exc:  # pragma: no cover - depends on runtime availability
    _QT_IMPORT_ERROR = exc


def _default_cfg_path() -> str:
    # The people-tracking config is the only fusion-ready default. The OOB
    # demo cfg (iwr6843_config.cfg) is intentionally NOT in the fallback
    # chain because it ships without an onboard tracker and with static
    # clutter removal, so fused tracking silently starves of track TLVs.
    preferred = Path("iwr6843_people_tracking.cfg")
    if preferred.exists():
        return str(preferred)
    return ""


def _rgba_tuple(color_hex: str, alpha: float = 1.0) -> Tuple[float, float, float, float]:
    color_hex = color_hex.lstrip("#")
    return (
        int(color_hex[0:2], 16) / 255.0,
        int(color_hex[2:4], 16) / 255.0,
        int(color_hex[4:6], 16) / 255.0,
        float(alpha),
    )


def _color_brush(color_hex: str, alpha: float = 1.0):
    assert _HAS_QT
    return pg.mkBrush(*_rgba_tuple(color_hex, alpha))


def _color_pen(
    color_hex: str,
    *,
    width: float = 1.5,
    alpha: float = 1.0,
    style: Optional[int] = None,
):
    assert _HAS_QT
    kwargs: Dict[str, Any] = {"color": _rgba_tuple(color_hex, alpha), "width": width}
    if style is not None:
        kwargs["style"] = style
    return pg.mkPen(**kwargs)


def _track_color_hex(track_id: int) -> str:
    palette = (
        "#cf5c36",
        "#1d7a8c",
        "#5a9367",
        "#2c6e99",
        "#9c6644",
        "#7b8c69",
        "#a23b72",
        "#f4a259",
    )
    return palette[track_id % len(palette)]


def _health_accent_color(verdict: str) -> str:
    mapping = {
        HEALTH_HEALTHY: "#2a9d8f",
        HEALTH_CONFIG_FAILED: "#c44536",
        HEALTH_STREAMING_UNPARSED: "#f4a259",
        HEALTH_NO_DATA: "#577590",
        HEALTH_REPLAY_MODE: "#2c6e99",
    }
    return mapping.get(verdict, "#577590")


def _points_to_array(points: Sequence[Any]) -> np.ndarray:
    if not points:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(
        [[float(point.x), float(point.y), float(point.z)] for point in points],
        dtype=np.float32,
    )


def _color_array(points: Sequence[Any], mode: str) -> np.ndarray:
    if not points:
        return np.empty((0, 4), dtype=np.float32)
    if mode == "snr":
        values = np.asarray([float(point.snr) for point in points], dtype=np.float32)
        vmin = float(values.min())
        vmax = float(values.max())
        if vmax <= vmin:
            vmax = vmin + 1.0
        normalized = (values - vmin) / (vmax - vmin)
        cmap = cm.get_cmap("viridis")
    else:
        values = np.asarray([float(point.velocity) for point in points], dtype=np.float32)
        vmax = max(1.0, float(np.max(np.abs(values))))
        normalized = (values + vmax) / (2.0 * vmax)
        cmap = cm.get_cmap("coolwarm")
    colors = np.asarray(cmap(normalized), dtype=np.float32)
    colors[:, 3] = 0.88
    return colors


def _box_segments(box: RadarBox3D) -> np.ndarray:
    x0, x1 = box.x_min, box.x_max
    y0, y1 = box.y_min, box.y_max
    z0, z1 = box.z_min, box.z_max
    corners = np.asarray(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float32,
    )
    edge_indices = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    segments = []
    for a, b in edge_indices:
        segments.extend([corners[a], corners[b]])
    return np.asarray(segments, dtype=np.float32)


def _track_summaries(tracks: Sequence[RadarTrack]) -> str:
    if not tracks:
        return "No active tracked people."
    lines = []
    for track in tracks:
        lines.append(
            f"ID {track.track_id}: pos=({track.x:+.2f}, {track.y:.2f}, {track.z:+.2f}) "
            f"vel=({track.vx:+.2f}, {track.vy:+.2f}, {track.vz:+.2f}) "
            f"conf={track.confidence:.2f} box={track.bbox_source or '-'}"
        )
    return "\n".join(lines)


def _tracking_debug_text(state: RadarSessionState) -> str:
    debug = state.tracking_debug or {}
    if not debug:
        return "No tracking debug data yet. Connect live mode or load replay frames first."

    lines = [
        f"Frame debug file: {state.frame_debug_path or '-'}",
        f"Frames observed: {debug.get('total_frames', 0)}",
        f"Track coverage: {debug.get('track_coverage_pct', 0.0)}%",
        f"Presence/track agreement: {debug.get('presence_track_agreement_pct', 0.0)}%",
        f"Presence but no track frames: {debug.get('frames_presence_without_tracks', 0)}",
        f"Points but no track frames: {debug.get('frames_points_without_tracks', 0)}",
        f"Current no-track streak: {debug.get('current_no_track_streak', 0)} frames",
        f"Max no-track streak: {debug.get('max_no_track_streak', 0)} frames",
        f"Active track IDs: {debug.get('active_track_ids', [])}",
        f"Unique track IDs: {debug.get('unique_track_ids', [])}",
        f"Single-person ID switches: {debug.get('single_track_id_switches', 0)}",
        f"Track gap events: {debug.get('track_gap_events', 0)}",
        f"Recent point count: {debug.get('recent_point_count', {})}",
    ]
    tracks = debug.get("tracks") or {}
    if tracks:
        lines.append("")
        lines.append("Per-ID continuity:")
        for track_id, info in sorted(tracks.items(), key=lambda item: int(item[0])):
            lines.append(
                f"  ID {track_id}: seen={info.get('seen_frames', 0)} "
                f"first={info.get('first_frame', 0)} last={info.get('last_frame', 0)}"
            )
    return "\n".join(lines)


def _default_room_box() -> RadarBox3D:
    half_width = DEFAULT_ROOM_WIDTH_M * 0.5
    return RadarBox3D(
        x_min=-half_width,
        x_max=half_width,
        y_min=0.0,
        y_max=DEFAULT_ROOM_DEPTH_M,
        z_min=0.0,
        z_max=DEFAULT_ROOM_HEIGHT_M,
        label="default_room",
        kind="environment",
    )


def _merge_boxes(boxes: Sequence[RadarBox3D]) -> Optional[RadarBox3D]:
    if not boxes:
        return None
    return RadarBox3D(
        x_min=min(box.x_min for box in boxes),
        x_max=max(box.x_max for box in boxes),
        y_min=min(box.y_min for box in boxes),
        y_max=max(box.y_max for box in boxes),
        z_min=min(box.z_min for box in boxes),
        z_max=max(box.z_max for box in boxes),
        label="merged_scene",
        kind="environment",
    )


def _fixed_room_box_for_scene(frame: RadarFrame) -> RadarBox3D:
    scene_box = _merge_boxes(frame.scene.all_boxes())
    if scene_box is None:
        return _default_room_box()

    width = max(DEFAULT_ROOM_WIDTH_M, scene_box.width + 1.6)
    depth = max(DEFAULT_ROOM_DEPTH_M, scene_box.depth + 1.6)
    height = max(DEFAULT_ROOM_HEIGHT_M, scene_box.height + 0.9)
    center_x = (scene_box.x_min + scene_box.x_max) * 0.5
    y_min = min(0.0, scene_box.y_min - 0.4)
    z_min = min(0.0, scene_box.z_min - 0.2)
    return RadarBox3D(
        x_min=center_x - width * 0.5,
        x_max=center_x + width * 0.5,
        y_min=y_min,
        y_max=y_min + depth,
        z_min=z_min,
        z_max=z_min + height,
        label="scene_room",
        kind="environment",
    )


def _room_camera_defaults(box: RadarBox3D) -> Tuple[float, float, float, np.ndarray]:
    center = np.array(
        [
            float((box.x_min + box.x_max) * 0.5),
            float((box.y_min + box.y_max) * 0.5),
            float(box.z_min + box.height * 0.38),
        ],
        dtype=np.float32,
    )
    diagonal = float(np.linalg.norm([box.width, box.depth * 0.95, box.height * 1.3]))
    distance = max(DEFAULT_VIEW_DISTANCE, diagonal * 1.35)
    return distance, DEFAULT_VIEW_ELEVATION, DEFAULT_VIEW_AZIMUTH, center


def _box_mesh_geometry(box: RadarBox3D) -> Tuple[np.ndarray, np.ndarray]:
    x0, x1 = box.x_min, box.x_max
    y0, y1 = box.y_min, box.y_max
    z0, z1 = box.z_min, box.z_max
    vertices = np.asarray(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [
            [0, 1, 2], [0, 2, 3],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int32,
    )
    return vertices, faces


def _room_guide_segments(box: RadarBox3D, spacing: float = ROOM_GUIDE_SPACING_M) -> np.ndarray:
    spacing = max(0.25, float(spacing))
    segments: List[np.ndarray] = []

    def add_segment(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> None:
        segments.extend([np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)])

    x_values = np.arange(box.x_min, box.x_max + spacing * 0.5, spacing)
    y_values = np.arange(box.y_min, box.y_max + spacing * 0.5, spacing)
    z_values = np.arange(box.z_min, box.z_max + spacing * 0.5, spacing)

    for x_value in x_values:
        add_segment((x_value, box.y_min, box.z_min), (x_value, box.y_max, box.z_min))
        add_segment((x_value, box.y_max, box.z_min), (x_value, box.y_max, box.z_max))
    for y_value in y_values:
        add_segment((box.x_min, y_value, box.z_min), (box.x_max, y_value, box.z_min))
        add_segment((box.x_min, y_value, box.z_min), (box.x_min, y_value, box.z_max))
    for z_value in z_values:
        add_segment((box.x_min, box.y_max, z_value), (box.x_max, box.y_max, z_value))
        add_segment((box.x_min, box.y_min, z_value), (box.x_min, box.y_max, z_value))

    return np.asarray(segments, dtype=np.float32) if segments else np.empty((0, 3), dtype=np.float32)


def _sensor_reference_segments(
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    *,
    x_length: float = 0.75,
    y_length: float = 1.0,
    z_length: float = 0.8,
) -> np.ndarray:
    ox, oy, oz = (float(origin[0]), float(origin[1]), float(origin[2]))
    return np.asarray(
        [
            [ox, oy, oz], [ox + x_length, oy, oz],
            [ox, oy, oz], [ox, oy + y_length, oz],
            [ox, oy, oz], [ox, oy, oz + z_length],
        ],
        dtype=np.float32,
    )


def _summary_label(source_mode: str, frame: RadarFrame, color_mode: str) -> str:
    return (
        f"{source_mode.capitalize()}  |  Frame {frame.frame_number}  |  "
        f"{len(frame.points)} points  |  {len(frame.tracks)} tracks  |  "
        f"Color {color_mode.upper()}"
    )


def _run_tk_viewer(args: argparse.Namespace) -> None:
    import radar_viewer_tk as tk_viewer

    root = tk_viewer.tk.Tk()
    tk_viewer.RadarViewerApp(
        root,
        mode=args.mode,
        replay_path=args.replay,
        cfg_path=args.cfg,
        config_port=args.config_port,
        data_port=args.data_port,
        config_baud=args.config_baud,
        data_baud=args.data_baud,
    )
    root.mainloop()


if _HAS_QT:
    pg.setConfigOptions(antialias=True, background="#f4f7fa", foreground="#213547")

    class AsyncBridge(QtCore.QObject):
        finished = QtCore.Signal(object, object, str)


    class StatusCard(QtWidgets.QFrame):
        def __init__(self, title: str, accent: str):
            super().__init__()
            self.setObjectName("StatusCard")
            self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

            layout = QtWidgets.QHBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)

            self._accent_bar = QtWidgets.QFrame()
            self._accent_bar.setFixedWidth(7)
            self._accent_bar.setStyleSheet(
                f"background:{accent}; border-top-left-radius:14px; border-bottom-left-radius:14px;"
            )
            layout.addWidget(self._accent_bar)

            content = QtWidgets.QWidget()
            content_layout = QtWidgets.QVBoxLayout(content)
            content_layout.setContentsMargins(14, 12, 14, 12)
            content_layout.setSpacing(4)

            self.title_label = QtWidgets.QLabel(title.upper())
            self.title_label.setObjectName("CardTitle")
            self.value_label = QtWidgets.QLabel("--")
            self.value_label.setObjectName("CardValue")
            self.value_label.setWordWrap(True)
            content_layout.addWidget(self.title_label)
            content_layout.addWidget(self.value_label)
            layout.addWidget(content)

        def set_value(self, text: str) -> None:
            self.value_label.setText(text)

        def set_accent(self, color: str) -> None:
            self._accent_bar.setStyleSheet(
                f"background:{color}; border-top-left-radius:14px; border-bottom-left-radius:14px;"
            )


    class QtRadarViewerApp(QtWidgets.QMainWindow):
        def __init__(self, args: argparse.Namespace):
            super().__init__()
            self.args = args
            self._source: Optional[FrameSource] = None
            self._port_infos: List[SerialPortInfo] = []
            self._busy = False
            self._last_drawn_key: Optional[Tuple[str, int]] = None
            self._last_cli_text = ""
            self._last_tracking_debug_text = ""
            self._last_frame_debug_text_at = 0.0
            self._last_render_at = 0.0
            self._render_interval_s = 1.0 / max(float(args.render_fps), 1.0)
            self._track_histories: Dict[int, deque[Tuple[float, float, float]]] = defaultdict(
                lambda: deque(maxlen=36)
            )
            self._track_last_seen: Dict[int, int] = {}
            self._dynamic_gl_items: List[Any] = []
            self._room_gl_items: List[Any] = []
            self._room_box = _default_room_box()
            self._room_locked = False
            self._room_camera = _room_camera_defaults(self._room_box)
            self._async_done: Optional[Callable[[object, Optional[BaseException]], None]] = None

            self.setWindowTitle("FALCON Radar Console")
            self.resize(1680, 980)
            self.setMinimumSize(1320, 820)
            QtWidgets.QApplication.instance().setStyle("Fusion")
            QtWidgets.QApplication.instance().setFont(QtGui.QFont("Noto Sans", 10))
            self.setStyleSheet(self._build_stylesheet())

            self._async_bridge = AsyncBridge()
            self._async_bridge.finished.connect(self._handle_async_finished)

            self._build_ui()
            self.refresh_ports()
            self._update_mode_controls()

            self._poll_timer = QtCore.QTimer(self)
            self._poll_timer.setInterval(max(25, int(args.poll_ms)))
            self._poll_timer.timeout.connect(self._poll_source)
            self._poll_timer.start()

        def _build_stylesheet(self) -> str:
            return """
            QMainWindow {
                background: #eef2f5;
                color: #1f3344;
            }
            QWidget {
                color: #1f3344;
            }
            QFrame#Panel, QFrame#HeaderPanel, QFrame#ControlPanel, QFrame#StatusCard {
                background: #fbfcfd;
                border: 1px solid #d8e0e6;
                border-radius: 14px;
            }
            QLabel#AppTitle {
                font-size: 24px;
                font-weight: 700;
                color: #173042;
            }
            QLabel#AppSubtitle {
                font-size: 11px;
                color: #577590;
            }
            QLabel#SectionTitle {
                font-size: 14px;
                font-weight: 700;
                color: #173042;
            }
            QLabel#SectionCaption {
                color: #577590;
                font-size: 11px;
            }
            QLabel#CardTitle {
                font-size: 10px;
                letter-spacing: 0.08em;
                color: #5f7486;
            }
            QLabel#CardValue {
                font-size: 18px;
                font-weight: 700;
                color: #173042;
            }
            QLabel#BannerLabel {
                background: #edf4f8;
                border: 1px solid #d4e0e8;
                border-radius: 10px;
                padding: 8px 12px;
                color: #25455d;
            }
            QGroupBox {
                border: 1px solid #d8e0e6;
                border-radius: 14px;
                margin-top: 16px;
                padding-top: 12px;
                background: #ffffff;
                font-weight: 700;
                color: #25455d;
            }
            QGroupBox::title {
                left: 12px;
                padding: 0 4px;
            }
            QPushButton {
                background: #1d5876;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 8px 12px;
                font-weight: 700;
            }
            QPushButton:hover {
                background: #246c90;
            }
            QPushButton:disabled {
                background: #a8b8c4;
                color: #eef3f7;
            }
            QLineEdit, QPlainTextEdit, QComboBox, QSpinBox {
                background: #ffffff;
                border: 1px solid #ccd8e0;
                border-radius: 10px;
                padding: 6px 8px;
            }
            QPlainTextEdit {
                selection-background-color: #8fbcd4;
            }
            QTabWidget::pane {
                border: 1px solid #d8e0e6;
                border-radius: 14px;
                background: #ffffff;
                top: -1px;
            }
            QTabBar::tab {
                background: #e6edf2;
                border: 1px solid #d8e0e6;
                padding: 8px 14px;
                margin-right: 4px;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                color: #335065;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                color: #173042;
            }
            """

        def _build_ui(self) -> None:
            root = QtWidgets.QWidget()
            self.setCentralWidget(root)
            root_layout = QtWidgets.QVBoxLayout(root)
            root_layout.setContentsMargins(14, 14, 14, 14)
            root_layout.setSpacing(12)

            header = QtWidgets.QFrame()
            header.setObjectName("HeaderPanel")
            header_layout = QtWidgets.QHBoxLayout(header)
            header_layout.setContentsMargins(18, 16, 18, 16)
            header_layout.setSpacing(16)
            title_col = QtWidgets.QVBoxLayout()
            title = QtWidgets.QLabel("FALCON Radar Console")
            title.setObjectName("AppTitle")
            subtitle = QtWidgets.QLabel(
                "Low-latency IWR6843 live diagnostics, replay analysis, and tracked-person visualization for Ubuntu/Orin."
            )
            subtitle.setObjectName("AppSubtitle")
            subtitle.setWordWrap(True)
            title_col.addWidget(title)
            title_col.addWidget(subtitle)
            header_layout.addLayout(title_col, 1)

            self.backend_badge = QtWidgets.QLabel("Qt + PyQtGraph + OpenGL")
            self.backend_badge.setObjectName("BannerLabel")
            header_layout.addWidget(self.backend_badge, 0, QtCore.Qt.AlignmentFlag.AlignRight)
            root_layout.addWidget(header)

            controls = QtWidgets.QFrame()
            controls.setObjectName("ControlPanel")
            controls_layout = QtWidgets.QHBoxLayout(controls)
            controls_layout.setContentsMargins(14, 14, 14, 14)
            controls_layout.setSpacing(12)
            controls_layout.addWidget(self._build_session_group(), 1)
            controls_layout.addWidget(self._build_live_group(), 2)
            controls_layout.addWidget(self._build_replay_group(), 2)
            controls_layout.addWidget(self._build_actions_group(), 1)
            root_layout.addWidget(controls)

            cards_row = QtWidgets.QHBoxLayout()
            cards_row.setSpacing(10)
            self.cards = {
                "health": StatusCard("Health", "#2c6e99"),
                "frames": StatusCard("Frames", "#577590"),
                "fps": StatusCard("FPS", "#2a9d8f"),
                "tracks": StatusCard("Tracks", "#cf5c36"),
                "rx": StatusCard("RX Bytes", "#7a8f99"),
                "magic": StatusCard("Magic Hits", "#5a9367"),
                "presence": StatusCard("Presence", "#f4a259"),
            }
            for card in self.cards.values():
                cards_row.addWidget(card, 1)
            root_layout.addLayout(cards_row)

            self.status_banner = QtWidgets.QLabel("Idle")
            self.status_banner.setObjectName("BannerLabel")
            root_layout.addWidget(self.status_banner)

            splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
            splitter.setChildrenCollapsible(False)
            splitter.addWidget(self._build_visual_panel())
            splitter.addWidget(self._build_diagnostics_panel())
            splitter.setStretchFactor(0, 4)
            splitter.setStretchFactor(1, 2)
            root_layout.addWidget(splitter, 1)

        def _build_session_group(self) -> QtWidgets.QGroupBox:
            group = QtWidgets.QGroupBox("Session")
            layout = QtWidgets.QGridLayout(group)
            layout.setHorizontalSpacing(10)
            layout.setVerticalSpacing(8)

            self.mode_combo = QtWidgets.QComboBox()
            self.mode_combo.addItem("Live", "live")
            self.mode_combo.addItem("Replay", "replay")
            mode_index = 0 if self.args.mode == "live" else 1
            self.mode_combo.setCurrentIndex(mode_index)
            self.mode_combo.currentIndexChanged.connect(self._update_mode_controls)

            self.color_combo = QtWidgets.QComboBox()
            self.color_combo.addItems(["velocity", "snr"])

            self.poll_spin = QtWidgets.QSpinBox()
            self.poll_spin.setRange(25, 1000)
            self.poll_spin.setValue(int(self.args.poll_ms))
            self.poll_spin.setSuffix(" ms")
            self.poll_spin.valueChanged.connect(self._set_poll_interval)

            self.render_fps_spin = QtWidgets.QDoubleSpinBox()
            self.render_fps_spin.setRange(5.0, 60.0)
            self.render_fps_spin.setValue(float(self.args.render_fps))
            self.render_fps_spin.setSingleStep(1.0)
            self.render_fps_spin.setDecimals(1)
            self.render_fps_spin.setSuffix(" fps")
            self.render_fps_spin.valueChanged.connect(self._set_render_rate)

            layout.addWidget(QtWidgets.QLabel("Mode"), 0, 0)
            layout.addWidget(self.mode_combo, 0, 1)
            layout.addWidget(QtWidgets.QLabel("Color"), 1, 0)
            layout.addWidget(self.color_combo, 1, 1)
            layout.addWidget(QtWidgets.QLabel("Poll"), 2, 0)
            layout.addWidget(self.poll_spin, 2, 1)
            layout.addWidget(QtWidgets.QLabel("Render"), 3, 0)
            layout.addWidget(self.render_fps_spin, 3, 1)
            return group

        def _build_live_group(self) -> QtWidgets.QGroupBox:
            group = QtWidgets.QGroupBox("Live Setup")
            layout = QtWidgets.QGridLayout(group)
            layout.setHorizontalSpacing(10)
            layout.setVerticalSpacing(8)

            self.config_port_combo = QtWidgets.QComboBox()
            self.config_port_combo.setEditable(True)
            self.data_port_combo = QtWidgets.QComboBox()
            self.data_port_combo.setEditable(True)

            self.config_baud_spin = QtWidgets.QSpinBox()
            self.config_baud_spin.setRange(9600, 3000000)
            self.config_baud_spin.setValue(int(self.args.config_baud))
            self.config_baud_spin.setSingleStep(9600)
            self.data_baud_spin = QtWidgets.QSpinBox()
            self.data_baud_spin.setRange(9600, 6000000)
            self.data_baud_spin.setValue(int(self.args.data_baud))
            self.data_baud_spin.setSingleStep(115200)

            self.cfg_line = QtWidgets.QLineEdit(self.args.cfg)
            cfg_browse = QtWidgets.QPushButton("Browse")
            cfg_browse.clicked.connect(self._browse_cfg)
            self.refresh_ports_button = QtWidgets.QPushButton("Refresh")
            self.refresh_ports_button.clicked.connect(self.refresh_ports)

            layout.addWidget(QtWidgets.QLabel("Config Port"), 0, 0)
            layout.addWidget(self.config_port_combo, 0, 1)
            layout.addWidget(QtWidgets.QLabel("Data Port"), 0, 2)
            layout.addWidget(self.data_port_combo, 0, 3)
            layout.addWidget(QtWidgets.QLabel("Config Baud"), 1, 0)
            layout.addWidget(self.config_baud_spin, 1, 1)
            layout.addWidget(QtWidgets.QLabel("Data Baud"), 1, 2)
            layout.addWidget(self.data_baud_spin, 1, 3)
            layout.addWidget(QtWidgets.QLabel("Cfg"), 2, 0)
            layout.addWidget(self.cfg_line, 2, 1, 1, 2)
            layout.addWidget(cfg_browse, 2, 3)
            layout.addWidget(QtWidgets.QLabel("Suggested"), 3, 0)
            self.suggestion_label = QtWidgets.QLabel("Suggested pair: none")
            self.suggestion_label.setWordWrap(True)
            layout.addWidget(self.suggestion_label, 3, 1, 1, 2)
            layout.addWidget(self.refresh_ports_button, 3, 3)
            return group

        def _build_replay_group(self) -> QtWidgets.QGroupBox:
            group = QtWidgets.QGroupBox("Replay")
            layout = QtWidgets.QGridLayout(group)
            layout.setHorizontalSpacing(10)
            layout.setVerticalSpacing(8)

            self.replay_line = QtWidgets.QLineEdit(self.args.replay)
            replay_browse = QtWidgets.QPushButton("Browse")
            replay_browse.clicked.connect(self._browse_replay)

            tips = QtWidgets.QLabel(
                "Replay mode uses the same normalized frame model as live mode, so tracks, scene boxes, and later camera-fusion hooks behave consistently."
            )
            tips.setObjectName("SectionCaption")
            tips.setWordWrap(True)

            layout.addWidget(QtWidgets.QLabel("Replay JSON"), 0, 0)
            layout.addWidget(self.replay_line, 0, 1)
            layout.addWidget(replay_browse, 0, 2)
            layout.addWidget(tips, 1, 0, 1, 3)
            return group

        def _build_actions_group(self) -> QtWidgets.QGroupBox:
            group = QtWidgets.QGroupBox("Actions")
            layout = QtWidgets.QVBoxLayout(group)
            layout.setSpacing(8)

            self.connect_button = QtWidgets.QPushButton("Connect")
            self.connect_button.clicked.connect(self._connect)
            self.disconnect_button = QtWidgets.QPushButton("Disconnect")
            self.disconnect_button.clicked.connect(self._disconnect)
            self.probe_button = QtWidgets.QPushButton("Probe Startup")
            self.probe_button.clicked.connect(self._probe_startup)
            self.reset_view_button = QtWidgets.QPushButton("Reset 3D View")
            self.reset_view_button.clicked.connect(self._reset_3d_view)

            layout.addWidget(self.connect_button)
            layout.addWidget(self.disconnect_button)
            layout.addWidget(self.probe_button)
            layout.addWidget(self.reset_view_button)
            layout.addStretch(1)
            return group

        def _build_visual_panel(self) -> QtWidgets.QWidget:
            panel = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(panel)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(10)

            plot_panel = QtWidgets.QFrame()
            plot_panel.setObjectName("Panel")
            plot_layout = QtWidgets.QVBoxLayout(plot_panel)
            plot_layout.setContentsMargins(14, 14, 14, 14)
            plot_layout.setSpacing(10)

            title = QtWidgets.QLabel("3D Room View")
            title.setObjectName("SectionTitle")
            caption = QtWidgets.QLabel(
                "Fixed room-scale OpenGL scene with tracked people rendered as 3D boxes and raw radar returns living in the same stable environment."
            )
            caption.setObjectName("SectionCaption")
            caption.setWordWrap(True)
            self.render_summary = QtWidgets.QLabel("No frames yet")
            self.render_summary.setObjectName("BannerLabel")
            self.room_summary = QtWidgets.QLabel("Room: 8.0m wide x 8.0m deep x 3.2m high")
            self.room_summary.setObjectName("SectionCaption")

            plot_layout.addWidget(title)
            plot_layout.addWidget(caption)
            plot_layout.addWidget(self.render_summary)
            plot_layout.addWidget(self.room_summary)

            self.gl_view = gl.GLViewWidget()
            self.gl_view.setBackgroundColor(pg.mkColor("#edf2f6"))
            self.gl_view.setCameraPosition(
                distance=self._room_camera[0],
                elevation=self._room_camera[1],
                azimuth=self._room_camera[2],
            )
            self.gl_view.opts["fov"] = 55
            self.gl_view.opts["center"] = pg.Vector(*self._room_camera[3])

            self.point_item = gl.GLScatterPlotItem(
                pos=np.empty((0, 3), dtype=np.float32),
                color=np.empty((0, 4), dtype=np.float32),
                size=0.075,
                pxMode=False,
            )
            self.track_item = gl.GLScatterPlotItem(
                pos=np.empty((0, 3), dtype=np.float32),
                color=np.empty((0, 4), dtype=np.float32),
                size=0.18,
                pxMode=False,
            )
            self.sensor_item = gl.GLScatterPlotItem(
                pos=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
                color=np.asarray([_rgba_tuple("#173042", 1.0)], dtype=np.float32),
                size=0.22,
                pxMode=False,
            )
            self.gl_view.addItem(self.point_item)
            self.gl_view.addItem(self.track_item)
            self.gl_view.addItem(self.sensor_item)
            self.sensor_label_item = gl.GLTextItem(
                pos=np.asarray([0.08, 0.12, 0.18], dtype=np.float32),
                color=QtGui.QColor("#173042"),
                text="Radar",
                font=QtGui.QFont("Noto Sans", 12, QtGui.QFont.Weight.DemiBold),
            )
            self.gl_view.addItem(self.sensor_label_item)
            self._rebuild_room_environment()
            plot_layout.addWidget(self.gl_view, 1)

            layout.addWidget(plot_panel)
            return panel

        def _build_diagnostics_panel(self) -> QtWidgets.QWidget:
            panel = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(panel)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)

            tabs = QtWidgets.QTabWidget()
            tabs.addTab(self._build_session_tab(), "Session")
            tabs.addTab(self._build_debug_tab(), "Debug")
            tabs.addTab(self._build_cli_tab(), "CLI")
            tabs.addTab(self._build_ports_tab(), "Ports")
            tabs.addTab(self._build_notes_tab(), "Notes")
            layout.addWidget(tabs)
            return panel

        def _build_session_tab(self) -> QtWidgets.QWidget:
            page = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(page)
            layout.setContentsMargins(14, 14, 14, 14)
            layout.setSpacing(12)

            form_frame = QtWidgets.QFrame()
            form_frame.setObjectName("Panel")
            form_layout = QtWidgets.QFormLayout(form_frame)
            form_layout.setContentsMargins(14, 14, 14, 14)
            form_layout.setSpacing(10)

            self.session_labels: Dict[str, QtWidgets.QLabel] = {}
            for key, label_text in (
                ("reason", "Reason"),
                ("parse_error", "Parse Error"),
                ("command_error", "Command Error"),
                ("connection_error", "Connection Error"),
                ("unknown_tlvs", "Unknown TLVs"),
                ("probe", "Probe Result"),
                ("run_dir", "Run Dir"),
                ("config", "Config / Replay"),
                ("ports", "Ports"),
            ):
                label = QtWidgets.QLabel("-")
                label.setWordWrap(True)
                self.session_labels[key] = label
                form_layout.addRow(f"{label_text}:", label)
            layout.addWidget(form_frame)

            tracks_frame = QtWidgets.QFrame()
            tracks_frame.setObjectName("Panel")
            tracks_layout = QtWidgets.QVBoxLayout(tracks_frame)
            tracks_layout.setContentsMargins(14, 14, 14, 14)
            tracks_layout.setSpacing(8)
            title = QtWidgets.QLabel("Track Summary")
            title.setObjectName("SectionTitle")
            self.track_summary = QtWidgets.QPlainTextEdit()
            self.track_summary.setReadOnly(True)
            tracks_layout.addWidget(title)
            tracks_layout.addWidget(self.track_summary, 1)
            layout.addWidget(tracks_frame, 1)
            return page

        def _build_debug_tab(self) -> QtWidgets.QWidget:
            page = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(page)
            layout.setContentsMargins(14, 14, 14, 14)
            layout.setSpacing(12)

            summary_frame = QtWidgets.QFrame()
            summary_frame.setObjectName("Panel")
            summary_layout = QtWidgets.QVBoxLayout(summary_frame)
            summary_layout.setContentsMargins(14, 14, 14, 14)
            summary_layout.setSpacing(8)
            summary_title = QtWidgets.QLabel("Tracking Stability")
            summary_title.setObjectName("SectionTitle")
            self.tracking_debug_text = QtWidgets.QPlainTextEdit()
            self.tracking_debug_text.setReadOnly(True)
            summary_layout.addWidget(summary_title)
            summary_layout.addWidget(self.tracking_debug_text, 1)
            layout.addWidget(summary_frame, 1)

            frame_frame = QtWidgets.QFrame()
            frame_frame.setObjectName("Panel")
            frame_layout = QtWidgets.QVBoxLayout(frame_frame)
            frame_layout.setContentsMargins(14, 14, 14, 14)
            frame_layout.setSpacing(8)
            frame_title = QtWidgets.QLabel("Latest Parsed Frame")
            frame_title.setObjectName("SectionTitle")
            self.latest_frame_debug_text = QtWidgets.QPlainTextEdit()
            self.latest_frame_debug_text.setReadOnly(True)
            frame_layout.addWidget(frame_title)
            frame_layout.addWidget(self.latest_frame_debug_text, 1)
            layout.addWidget(frame_frame, 1)
            return page

        def _build_cli_tab(self) -> QtWidgets.QWidget:
            page = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(page)
            layout.setContentsMargins(14, 14, 14, 14)
            title = QtWidgets.QLabel("CLI Log Tail")
            title.setObjectName("SectionTitle")
            self.cli_text = QtWidgets.QPlainTextEdit()
            self.cli_text.setReadOnly(True)
            layout.addWidget(title)
            layout.addWidget(self.cli_text, 1)
            return page

        def _build_ports_tab(self) -> QtWidgets.QWidget:
            page = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(page)
            layout.setContentsMargins(14, 14, 14, 14)
            title = QtWidgets.QLabel("Detected Ports")
            title.setObjectName("SectionTitle")
            self.port_text = QtWidgets.QPlainTextEdit()
            self.port_text.setReadOnly(True)
            layout.addWidget(title)
            layout.addWidget(self.port_text, 1)
            return page

        def _build_notes_tab(self) -> QtWidgets.QWidget:
            page = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(page)
            layout.setContentsMargins(14, 14, 14, 14)
            note_frame = QtWidgets.QFrame()
            note_frame.setObjectName("Panel")
            note_layout = QtWidgets.QVBoxLayout(note_frame)
            note_layout.setContentsMargins(14, 14, 14, 14)
            note_layout.setSpacing(8)
            title = QtWidgets.QLabel("Operating Notes")
            title.setObjectName("SectionTitle")
            body = QtWidgets.QLabel(
                "Live mode owns sensor startup and keeps serial I/O off the UI thread.\n\n"
                "Replay mode feeds the same normalized frame model into the renderer, so tracked people, 3-D extents, and later camera-fusion hooks stay aligned.\n\n"
                "The room view is intentionally fixed once a session starts, so you can reason about motion in a stable space instead of watching the world reframe itself.\n\n"
                "If Qt ever fails on another machine, the older Tk fallback remains available with `--backend tk`."
            )
            body.setWordWrap(True)
            note_layout.addWidget(title)
            note_layout.addWidget(body)
            layout.addWidget(note_frame)
            layout.addStretch(1)
            return page

        def _set_poll_interval(self, value: int) -> None:
            self._poll_timer.setInterval(max(25, int(value)))

        def _set_render_rate(self, value: float) -> None:
            self._render_interval_s = 1.0 / max(float(value), 1.0)

        def refresh_ports(self) -> None:
            self._port_infos = discover_serial_ports()
            pairs = suggest_serial_port_pairs(self._port_infos)
            devices = [info.device for info in self._port_infos]

            current_config = self.config_port_combo.currentText()
            current_data = self.data_port_combo.currentText()

            self.config_port_combo.clear()
            self.data_port_combo.clear()
            self.config_port_combo.addItems(devices)
            self.data_port_combo.addItems(devices)

            if pairs:
                suggested_config, suggested_data = pairs[0]
                self.suggestion_label.setText(
                    f"Suggested pair: config={suggested_config}, data={suggested_data}"
                )
                if not current_config:
                    self.config_port_combo.setEditText(suggested_config)
                else:
                    self.config_port_combo.setEditText(current_config)
                if not current_data:
                    self.data_port_combo.setEditText(suggested_data)
                else:
                    self.data_port_combo.setEditText(current_data)
            else:
                self.suggestion_label.setText("Suggested pair: none")
                if current_config:
                    self.config_port_combo.setEditText(current_config)
                elif devices:
                    self.config_port_combo.setEditText(devices[0])
                if current_data:
                    self.data_port_combo.setEditText(current_data)
                elif len(devices) > 1:
                    self.data_port_combo.setEditText(devices[1])

            lines = []
            for info in self._port_infos:
                lines.append(info.display_label())
                if info.manufacturer or info.product or info.hwid:
                    lines.append(
                        f"  manufacturer={info.manufacturer or '-'} | product={info.product or '-'} | hwid={info.hwid or '-'}"
                    )
            self.port_text.setPlainText("\n".join(lines) if lines else "No serial ports found.")

        def _browse_cfg(self) -> None:
            chosen, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Choose radar cfg",
                str(_repo_root()),
                "Config files (*.cfg);;All files (*.*)",
            )
            if chosen:
                self.cfg_line.setText(chosen)

        def _browse_replay(self) -> None:
            chosen, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Choose replay JSON",
                str(_repo_root()),
                "JSON files (*.json);;All files (*.*)",
            )
            if chosen:
                self.replay_line.setText(chosen)

        def _current_mode(self) -> str:
            return str(self.mode_combo.currentData())

        def _run_async(
            self,
            label: str,
            worker: Callable[[], object],
            done: Optional[Callable[[object, Optional[BaseException]], None]] = None,
        ) -> None:
            if self._busy:
                return
            self._busy = True
            self._async_done = done
            self.status_banner.setText(label)
            self._update_mode_controls()

            def target() -> None:
                result: object = None
                error: Optional[BaseException] = None
                try:
                    result = worker()
                except BaseException as exc:  # pragma: no cover - async UI path
                    error = exc
                self._async_bridge.finished.emit(result, error, label)

            threading.Thread(target=target, daemon=True, name="radar-viewer-worker").start()

        def _handle_async_finished(
            self,
            result: object,
            error: Optional[BaseException],
            label: str,
        ) -> None:
            self._busy = False
            done = self._async_done
            self._async_done = None
            if error is not None:
                self.status_banner.setText(f"{label} failed")
                QtWidgets.QMessageBox.critical(self, "Radar Viewer", str(error))
            if done is not None:
                done(result, error)
            self._update_mode_controls()

        def _connect(self) -> None:
            mode = self._current_mode()
            cfg_path = self.cfg_line.text().strip()
            replay_path = self.replay_line.text().strip()
            config_port = self.config_port_combo.currentText().strip()
            data_port = self.data_port_combo.currentText().strip()

            if mode == "live" and not cfg_path:
                QtWidgets.QMessageBox.warning(self, "Radar Viewer", "Choose a cfg file before connecting.")
                return
            if mode == "live" and (not config_port or not data_port):
                QtWidgets.QMessageBox.warning(self, "Radar Viewer", "Choose both config and data ports before connecting.")
                return
            if mode == "replay" and not replay_path:
                QtWidgets.QMessageBox.warning(self, "Radar Viewer", "Choose a replay JSON before connecting.")
                return

            if mode == "live":
                source: FrameSource = IWR6843Driver(
                    config_port=config_port,
                    data_port=data_port,
                    config_path=cfg_path,
                    config_baud=int(self.config_baud_spin.value()),
                    data_baud=int(self.data_baud_spin.value()),
                )
            else:
                source = ReplayRadarSource(replay_path)

            self._source = source
            self._last_drawn_key = None
            self._track_histories.clear()
            self._track_last_seen.clear()
            self._room_locked = False
            self._set_room_box(_default_room_box())
            self._clear_dynamic_render_items()

            def worker() -> bool:
                return bool(source.start())

            def done(result: object, error: Optional[BaseException]) -> None:
                if error is None:
                    self.status_banner.setText("Connected" if result else "Connect finished with errors")
                self._refresh_state_panel()

            self._run_async("Connecting...", worker, done)

        def _disconnect(self) -> None:
            source = self._source
            if source is None:
                return

            def worker() -> None:
                source.stop()

            def done(result: object, error: Optional[BaseException]) -> None:
                del result
                if error is None:
                    self.status_banner.setText("Disconnected")
                self._track_histories.clear()
                self._track_last_seen.clear()
                self._last_drawn_key = None
                self._room_locked = False
                self._set_room_box(_default_room_box())
                self._clear_dynamic_render_items()
                self._reset_plots()
                self._refresh_state_panel()

            self._run_async("Disconnecting...", worker, done)

        def _probe_startup(self) -> None:
            if self._current_mode() != "live":
                QtWidgets.QMessageBox.information(self, "Radar Viewer", "Probe Startup is only available in live mode.")
                return

            if isinstance(self._source, IWR6843Driver):
                source = self._source
            else:
                source = IWR6843Driver(
                    config_port=self.config_port_combo.currentText().strip(),
                    data_port=self.data_port_combo.currentText().strip(),
                    config_path=self.cfg_line.text().strip(),
                    config_baud=int(self.config_baud_spin.value()),
                    data_baud=int(self.data_baud_spin.value()),
                )
                self._source = source

            self._last_drawn_key = None

            def worker() -> object:
                return source.probe_startup()

            def done(result: object, error: Optional[BaseException]) -> None:
                del result
                if error is None:
                    self.status_banner.setText("Probe complete")
                self._refresh_state_panel()

            self._run_async("Running staged probe...", worker, done)

        def _current_state(self) -> Optional[RadarSessionState]:
            if self._source is None:
                return None
            try:
                return self._source.session_state()
            except Exception as exc:  # pragma: no cover - defensive UI path
                self.status_banner.setText(f"State error: {exc}")
                return None

        def _refresh_state_panel(self) -> None:
            self._apply_state(self._current_state())

        def _apply_state(self, state: Optional[RadarSessionState]) -> None:
            if state is None:
                return

            self.cards["health"].set_value(state.health_verdict or "Idle")
            self.cards["health"].set_accent(_health_accent_color(state.health_verdict))
            self.cards["frames"].set_value(str(state.frames))
            self.cards["fps"].set_value(f"{state.fps:.1f}")
            self.cards["tracks"].set_value(str(state.tracks))
            self.cards["rx"].set_value(str(state.rx_bytes))
            self.cards["magic"].set_value(str(state.magic_hits))
            self.cards["presence"].set_value(state.presence or "-")

            self.session_labels["reason"].setText(state.health_reason or "-")
            self.session_labels["parse_error"].setText(state.last_parse_error or "-")
            self.session_labels["command_error"].setText(state.last_command_error or "-")
            self.session_labels["connection_error"].setText(state.connection_error or "-")
            unknown = ", ".join(
                f"type {key}: {count}"
                for key, count in sorted(state.unknown_tlv_counts.items())
            )
            self.session_labels["unknown_tlvs"].setText(unknown or "-")
            probe_text = state.probe_failed_stage or (
                state.probe_log_tail.splitlines()[0] if state.probe_log_tail else "-"
            )
            self.session_labels["probe"].setText(probe_text)
            self.session_labels["run_dir"].setText(state.run_dir or "-")
            self.session_labels["config"].setText(state.config_path or state.replay_path or "-")
            self.session_labels["ports"].setText(
                f"config={state.config_port or '-'} | data={state.data_port or '-'}"
            )

            tracking_text = _tracking_debug_text(state)
            if tracking_text != self._last_tracking_debug_text:
                self._last_tracking_debug_text = tracking_text
                self.tracking_debug_text.setPlainText(tracking_text)

            if state.cli_log_tail != self._last_cli_text:
                self._last_cli_text = state.cli_log_tail
                self.cli_text.setPlainText(state.cli_log_tail or "(no CLI output yet)")
                cursor = self.cli_text.textCursor()
                cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
                self.cli_text.setTextCursor(cursor)

        def _poll_source(self) -> None:
            state = self._current_state()
            if state is not None:
                self._apply_state(state)
                if self._source is not None:
                    try:
                        frame = self._source.latest_frame()
                    except Exception as exc:  # pragma: no cover - defensive UI path
                        self.status_banner.setText(f"Frame error: {exc}")
                        frame = None
                    if frame is not None:
                        key = (state.mode, frame.frame_number)
                        now = time.monotonic()
                        if key != self._last_drawn_key and (now - self._last_render_at) >= self._render_interval_s:
                            self._render_frame(state.mode, frame)
                            self._source.note_frame_rendered(frame)
                            self._last_drawn_key = key
                            self._last_render_at = now
                    elif state.health_verdict == HEALTH_REPLAY_MODE:
                        self._clear_dynamic_render_items()
                        self._reset_plots()
                        self._last_drawn_key = None

        def _render_frame(self, source_mode: str, frame: RadarFrame) -> None:
            if not self._room_locked:
                self._set_room_box(_fixed_room_box_for_scene(frame))
                self._room_locked = True
            self._update_track_history(frame)
            self.render_summary.setText(_summary_label(source_mode, frame, self.color_combo.currentText()))
            self.track_summary.setPlainText(_track_summaries(frame.tracks))
            self._update_latest_frame_debug(frame)

            point_positions = _points_to_array(frame.points)
            point_colors = _color_array(frame.points, self.color_combo.currentText())
            self.point_item.setData(
                pos=point_positions,
                color=point_colors,
                size=0.075,
                pxMode=False,
            )

            if frame.tracks:
                track_positions = np.asarray(
                    [[track.x, track.y, track.z] for track in frame.tracks],
                    dtype=np.float32,
                )
                track_colors = np.asarray(
                    [_rgba_tuple(_track_color_hex(track.track_id), 0.98) for track in frame.tracks],
                    dtype=np.float32,
                )
            else:
                track_positions = np.empty((0, 3), dtype=np.float32)
                track_colors = np.empty((0, 4), dtype=np.float32)
            self.track_item.setData(pos=track_positions, color=track_colors, size=0.18, pxMode=False)
            sensor_height = float(frame.scene.sensor_height_m)
            self.sensor_item.setData(
                pos=np.asarray([[0.0, 0.0, sensor_height]], dtype=np.float32),
                color=np.asarray([_rgba_tuple("#173042", 1.0)], dtype=np.float32),
                size=0.22,
                pxMode=False,
            )
            self.sensor_label_item.setData(
                pos=np.asarray([0.1, 0.14, sensor_height + 0.16], dtype=np.float32),
                text="Radar",
            )

            self._clear_dynamic_render_items()
            self._draw_scene_overlays(frame)
            self._draw_tracks(frame)

        def _update_latest_frame_debug(self, frame: RadarFrame) -> None:
            now = time.monotonic()
            if now - self._last_frame_debug_text_at < 0.5:
                return
            self._last_frame_debug_text_at = now
            payload = radar_frame_debug_payload(frame, include_points=True, max_points=48)
            self.latest_frame_debug_text.setPlainText(json.dumps(payload, indent=2, sort_keys=True))

        def _update_track_history(self, frame: RadarFrame) -> None:
            for track in frame.tracks:
                self._track_histories[track.track_id].append((track.x, track.y, track.z))
                self._track_last_seen[track.track_id] = frame.frame_number

            stale_ids = [
                track_id
                for track_id, last_frame in self._track_last_seen.items()
                if frame.frame_number - last_frame > 40
            ]
            for track_id in stale_ids:
                self._track_last_seen.pop(track_id, None)
                self._track_histories.pop(track_id, None)

        def _draw_scene_overlays(self, frame: RadarFrame) -> None:
            for box in frame.scene.static_boundary_boxes:
                self._add_box_outline(box, "#7b8c99", width=1.2, alpha=0.45)
            for box in frame.scene.boundary_boxes:
                self._add_box_outline(box, "#2c6e99", width=1.35, alpha=0.55)
            for box in frame.scene.presence_boxes:
                self._add_box_outline(box, "#f4a259", width=1.35, alpha=0.62)

        def _draw_tracks(self, frame: RadarFrame) -> None:
            for track in frame.tracks:
                color = _track_color_hex(track.track_id)
                if track.bbox is not None:
                    self._add_track_box(track.bbox, color)

                history = list(self._track_histories.get(track.track_id, ()))
                if len(history) >= 2:
                    history_arr = np.asarray(history, dtype=np.float32)
                    self._add_gl_line(
                        history_arr,
                        color,
                        width=2.2,
                        alpha=0.55,
                        mode="line_strip",
                    )

                self._add_track_labels(track)

        def _add_track_labels(self, track: RadarTrack) -> None:
            color = _track_color_hex(track.track_id)
            label_item = gl.GLTextItem(
                pos=np.asarray(
                    [
                        track.x + 0.08,
                        track.y + 0.08,
                        (track.bbox.z_max + 0.08) if track.bbox is not None else (track.z + 0.28),
                    ],
                    dtype=np.float32,
                ),
                color=QtGui.QColor(color),
                text=f"ID {track.track_id}",
                font=QtGui.QFont("Noto Sans", 12, QtGui.QFont.Weight.DemiBold),
            )
            self.gl_view.addItem(label_item)
            self._dynamic_gl_items.append(label_item)

        def _add_track_box(self, box: RadarBox3D, color: str) -> None:
            vertices, faces = _box_mesh_geometry(box)
            mesh = gl.GLMeshItem(
                meshdata=gl.MeshData(vertexes=vertices, faces=faces),
                color=_rgba_tuple(color, 0.16),
                edgeColor=_rgba_tuple(color, 0.95),
                drawEdges=True,
                drawFaces=True,
                smooth=False,
                shader=None,
                glOptions="translucent",
            )
            self.gl_view.addItem(mesh)
            self._dynamic_gl_items.append(mesh)
            self._add_gl_line(_box_segments(box), color, width=1.8, alpha=0.95, mode="lines")

        def _add_box_outline(
            self,
            box: RadarBox3D,
            color: str,
            *,
            width: float,
            alpha: float,
        ) -> None:
            self._add_gl_line(_box_segments(box), color, width=width, alpha=alpha, mode="lines")

        def _add_gl_line(
            self,
            positions: np.ndarray,
            color_hex: str,
            *,
            width: float,
            alpha: float,
            mode: str,
        ) -> None:
            item = gl.GLLinePlotItem(
                pos=np.asarray(positions, dtype=np.float32),
                color=np.asarray(_rgba_tuple(color_hex, alpha), dtype=np.float32),
                width=float(width),
                antialias=True,
                mode=mode,
            )
            self.gl_view.addItem(item)
            self._dynamic_gl_items.append(item)

        def _clear_dynamic_render_items(self) -> None:
            for item in self._dynamic_gl_items:
                try:
                    self.gl_view.removeItem(item)
                except Exception:
                    pass
            self._dynamic_gl_items.clear()

        def _reset_plots(self) -> None:
            self.point_item.setData(
                pos=np.empty((0, 3), dtype=np.float32),
                color=np.empty((0, 4), dtype=np.float32),
                size=0.075,
                pxMode=False,
            )
            self.track_item.setData(
                pos=np.empty((0, 3), dtype=np.float32),
                color=np.empty((0, 4), dtype=np.float32),
                size=0.18,
                pxMode=False,
            )
            self.sensor_item.setData(
                pos=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
                color=np.asarray([_rgba_tuple("#173042", 1.0)], dtype=np.float32),
                size=0.22,
                pxMode=False,
            )
            self.sensor_label_item.setData(pos=np.asarray([0.1, 0.14, 0.16], dtype=np.float32), text="Radar")
            self.render_summary.setText("No frames yet")
            self.room_summary.setText(
                f"Room: {self._room_box.width:.1f}m wide x {self._room_box.depth:.1f}m deep x {self._room_box.height:.1f}m high"
            )
            self.track_summary.setPlainText("No active tracked people.")
            self.latest_frame_debug_text.setPlainText("No parsed frame yet.")

        def _reset_3d_view(self) -> None:
            self.gl_view.setCameraPosition(
                distance=self._room_camera[0],
                elevation=self._room_camera[1],
                azimuth=self._room_camera[2],
            )
            self.gl_view.opts["center"] = pg.Vector(*self._room_camera[3])
            self.gl_view.update()

        def _set_room_box(self, room_box: RadarBox3D) -> None:
            self._room_box = room_box
            self._room_camera = _room_camera_defaults(room_box)
            self.room_summary.setText(
                f"Room: {room_box.width:.1f}m wide x {room_box.depth:.1f}m deep x {room_box.height:.1f}m high"
            )
            self._rebuild_room_environment()
            self._reset_3d_view()

        def _rebuild_room_environment(self) -> None:
            for item in self._room_gl_items:
                try:
                    self.gl_view.removeItem(item)
                except Exception:
                    pass
            self._room_gl_items.clear()

            floor_vertices = np.asarray(
                [
                    [self._room_box.x_min, self._room_box.y_min, self._room_box.z_min],
                    [self._room_box.x_max, self._room_box.y_min, self._room_box.z_min],
                    [self._room_box.x_max, self._room_box.y_max, self._room_box.z_min],
                    [self._room_box.x_min, self._room_box.y_max, self._room_box.z_min],
                ],
                dtype=np.float32,
            )
            floor_faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
            floor_item = gl.GLMeshItem(
                meshdata=gl.MeshData(vertexes=floor_vertices, faces=floor_faces),
                color=_rgba_tuple("#dde7ee", 0.35),
                drawEdges=False,
                drawFaces=True,
                smooth=False,
                shader=None,
                glOptions="translucent",
            )
            self.gl_view.addItem(floor_item)
            self._room_gl_items.append(floor_item)

            guide_segments = _room_guide_segments(self._room_box)
            if len(guide_segments):
                guide_item = gl.GLLinePlotItem(
                    pos=guide_segments,
                    color=np.asarray(_rgba_tuple(ROOM_MONO_GUIDE_COLOR, 0.24), dtype=np.float32),
                    width=1.0,
                    antialias=True,
                    mode="lines",
                )
                self.gl_view.addItem(guide_item)
                self._room_gl_items.append(guide_item)

            outline_item = gl.GLLinePlotItem(
                pos=_box_segments(self._room_box),
                color=np.asarray(_rgba_tuple(ROOM_MONO_LINE_COLOR, 0.72), dtype=np.float32),
                width=1.7,
                antialias=True,
                mode="lines",
            )
            self.gl_view.addItem(outline_item)
            self._room_gl_items.append(outline_item)

            axis_item = gl.GLLinePlotItem(
                pos=_sensor_reference_segments(),
                color=np.asarray(_rgba_tuple(ROOM_MONO_AXIS_COLOR, 0.88), dtype=np.float32),
                width=2.1,
                antialias=True,
                mode="lines",
            )
            self.gl_view.addItem(axis_item)
            self._room_gl_items.append(axis_item)

            for label, position in (
                ("X", np.asarray([0.82, 0.0, 0.02], dtype=np.float32)),
                ("Y", np.asarray([0.0, 1.08, 0.02], dtype=np.float32)),
                ("Z", np.asarray([0.0, 0.0, 0.88], dtype=np.float32)),
            ):
                text_item = gl.GLTextItem(
                    pos=position,
                    color=QtGui.QColor(ROOM_MONO_AXIS_COLOR),
                    text=label,
                    font=QtGui.QFont("Noto Sans", 10, QtGui.QFont.Weight.DemiBold),
                )
                self.gl_view.addItem(text_item)
                self._room_gl_items.append(text_item)

        def _update_mode_controls(self) -> None:
            live_enabled = self._current_mode() == "live"
            replay_enabled = not live_enabled

            for widget in (
                self.config_port_combo,
                self.data_port_combo,
                self.config_baud_spin,
                self.data_baud_spin,
                self.cfg_line,
                self.refresh_ports_button,
            ):
                widget.setEnabled(live_enabled and not self._busy)

            self.replay_line.setEnabled(replay_enabled and not self._busy)
            self.connect_button.setEnabled(not self._busy)
            self.disconnect_button.setEnabled(self._source is not None and not self._busy)
            self.probe_button.setEnabled(live_enabled and not self._busy)
            self.reset_view_button.setEnabled(not self._busy)
            self.mode_combo.setEnabled(not self._busy)
            self.color_combo.setEnabled(not self._busy)
            self.poll_spin.setEnabled(not self._busy)
            self.render_fps_spin.setEnabled(not self._busy)

        def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # pragma: no cover - UI lifecycle
            try:
                if self._source is not None:
                    self._source.stop()
            finally:
                super().closeEvent(event)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("live", "replay"), default="live")
    parser.add_argument("--replay", default="")
    parser.add_argument("--cfg", default=_default_cfg_path())
    parser.add_argument("--config-port", default="")
    parser.add_argument("--data-port", default="")
    parser.add_argument("--config-baud", type=int, default=115200)
    parser.add_argument("--data-baud", type=int, default=921600)
    parser.add_argument("--backend", choices=("auto", "qt", "tk"), default="auto")
    parser.add_argument("--poll-ms", type=int, default=DEFAULT_POLL_MS)
    parser.add_argument("--render-fps", type=float, default=DEFAULT_RENDER_FPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.backend == "tk":
        _run_tk_viewer(args)
        return

    if not _HAS_QT:
        if args.backend == "qt":
            raise RuntimeError(
                "Qt backend requested but unavailable. "
                f"Original import error: {_QT_IMPORT_ERROR}"
            )
        _run_tk_viewer(args)
        return

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = QtRadarViewerApp(args)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
