#!/usr/bin/env python3
"""
Standalone diagnostic viewer for the TI IWR6843ISK radar.

The app owns sensor startup in live mode, loads TI replay JSON in replay mode,
and renders a TI-Visualizer-style 3-D radar view plus orthographic support
views while surfacing CLI logs, health verdicts, and low-level UART diagnostics.
"""

from __future__ import annotations

import argparse
import threading
import tkinter as tk
from collections import defaultdict, deque
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Optional

import matplotlib

matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from radar import (
    HEALTH_REPLAY_MODE,
    FrameSource,
    IWR6843Driver,
    RadarBox3D,
    RadarFrame,
    RadarSessionState,
    RadarTrack,
    ReplayRadarSource,
    SerialPortInfo,
    discover_serial_ports,
    suggest_serial_port_pairs,
)


class RadarViewerApp:
    def __init__(
        self,
        root: tk.Tk,
        *,
        mode: str,
        replay_path: str,
        cfg_path: str,
        config_port: str,
        data_port: str,
        config_baud: int,
        data_baud: int,
    ):
        self.root = root
        self.root.title("IWR6843 Diagnostic Viewer")
        self.root.geometry("1580x980")

        self._source: Optional[FrameSource] = None
        self._port_infos: list[SerialPortInfo] = []
        self._busy = False
        self._last_drawn_key: Optional[tuple[str, int]] = None
        self._last_cli_text = ""
        self._view_elev = 18.0
        self._view_azim = -68.0
        self._track_histories: dict[int, deque[tuple[float, float, float]]] = defaultdict(
            lambda: deque(maxlen=25)
        )
        self._track_last_seen: dict[int, int] = {}

        self.mode_var = tk.StringVar(value=mode)
        self.config_port_var = tk.StringVar(value=config_port)
        self.data_port_var = tk.StringVar(value=data_port)
        self.cfg_path_var = tk.StringVar(value=cfg_path)
        self.replay_path_var = tk.StringVar(value=replay_path)
        self.config_baud_var = tk.IntVar(value=config_baud)
        self.data_baud_var = tk.IntVar(value=data_baud)
        self.color_mode_var = tk.StringVar(value="velocity")
        self.suggestion_var = tk.StringVar(value="Suggested pair: none")
        self.status_banner_var = tk.StringVar(value="Idle")

        self.health_var = tk.StringVar(value="")
        self.reason_var = tk.StringVar(value="")
        self.frames_var = tk.StringVar(value="0")
        self.tracks_var = tk.StringVar(value="0")
        self.presence_var = tk.StringVar(value="")
        self.fps_var = tk.StringVar(value="0.00")
        self.rx_var = tk.StringVar(value="0")
        self.magic_var = tk.StringVar(value="0")
        self.packet_var = tk.StringVar(value="0")
        self.config_ok_var = tk.StringVar(value="False")
        self.plots_var = tk.StringVar(value="False")
        self.parse_error_var = tk.StringVar(value="")
        self.command_error_var = tk.StringVar(value="")
        self.connection_error_var = tk.StringVar(value="")
        self.unknown_tlvs_var = tk.StringVar(value="")
        self.run_dir_var = tk.StringVar(value="")
        self.probe_var = tk.StringVar(value="")

        self._build_layout()
        self.refresh_ports()
        self._update_mode_controls()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(200, self._poll_source)

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        controls = ttk.Frame(self.root, padding=12)
        controls.grid(row=0, column=0, sticky="ew")
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(2, weight=1)
        controls.columnconfigure(3, weight=1)

        mode_box = ttk.LabelFrame(controls, text="Mode")
        mode_box.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        ttk.Radiobutton(mode_box, text="Live", value="live", variable=self.mode_var, command=self._update_mode_controls).grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Radiobutton(mode_box, text="Replay", value="replay", variable=self.mode_var, command=self._update_mode_controls).grid(row=1, column=0, sticky="w", padx=8, pady=(0, 8))

        live_box = ttk.LabelFrame(controls, text="Live Setup")
        live_box.grid(row=0, column=1, sticky="nsew", padx=8)
        live_box.columnconfigure(1, weight=1)
        ttk.Label(live_box, text="Config Port").grid(row=0, column=0, sticky="w", padx=8, pady=(8, 4))
        self.config_port_combo = ttk.Combobox(live_box, textvariable=self.config_port_var, state="readonly")
        self.config_port_combo.grid(row=0, column=1, sticky="ew", padx=8, pady=(8, 4))
        ttk.Label(live_box, text="Data Port").grid(row=1, column=0, sticky="w", padx=8, pady=4)
        self.data_port_combo = ttk.Combobox(live_box, textvariable=self.data_port_var, state="readonly")
        self.data_port_combo.grid(row=1, column=1, sticky="ew", padx=8, pady=4)
        ttk.Label(live_box, text="Config Baud").grid(row=2, column=0, sticky="w", padx=8, pady=4)
        self.config_baud_entry = ttk.Entry(live_box, textvariable=self.config_baud_var, width=10)
        self.config_baud_entry.grid(row=2, column=1, sticky="w", padx=8, pady=4)
        ttk.Label(live_box, text="Data Baud").grid(row=3, column=0, sticky="w", padx=8, pady=4)
        self.data_baud_entry = ttk.Entry(live_box, textvariable=self.data_baud_var, width=10)
        self.data_baud_entry.grid(row=3, column=1, sticky="w", padx=8, pady=4)
        ttk.Label(live_box, text="Cfg File").grid(row=4, column=0, sticky="w", padx=8, pady=4)
        self.cfg_entry = ttk.Entry(live_box, textvariable=self.cfg_path_var)
        self.cfg_entry.grid(row=4, column=1, sticky="ew", padx=8, pady=4)
        self.cfg_button = ttk.Button(live_box, text="Browse", command=self._browse_cfg)
        self.cfg_button.grid(row=4, column=2, sticky="ew", padx=(0, 8), pady=4)
        self.refresh_ports_button = ttk.Button(live_box, text="Refresh Ports", command=self.refresh_ports)
        self.refresh_ports_button.grid(row=5, column=1, sticky="w", padx=8, pady=(4, 8))

        replay_box = ttk.LabelFrame(controls, text="Replay Setup")
        replay_box.grid(row=0, column=2, sticky="nsew", padx=8)
        replay_box.columnconfigure(0, weight=1)
        ttk.Label(replay_box, text="Replay JSON").grid(row=0, column=0, sticky="w", padx=8, pady=(8, 4))
        self.replay_entry = ttk.Entry(replay_box, textvariable=self.replay_path_var)
        self.replay_entry.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        self.replay_button = ttk.Button(replay_box, text="Browse", command=self._browse_replay)
        self.replay_button.grid(row=1, column=1, sticky="ew", padx=(0, 8), pady=4)
        ttk.Label(replay_box, textvariable=self.suggestion_var, wraplength=320).grid(row=2, column=0, columnspan=2, sticky="w", padx=8, pady=(4, 8))

        action_box = ttk.LabelFrame(controls, text="Actions")
        action_box.grid(row=0, column=3, sticky="nsew", padx=(8, 0))
        action_box.columnconfigure(0, weight=1)
        self.connect_button = ttk.Button(action_box, text="Connect", command=self._connect)
        self.connect_button.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        self.disconnect_button = ttk.Button(action_box, text="Disconnect", command=self._disconnect)
        self.disconnect_button.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        self.probe_button = ttk.Button(action_box, text="Probe Startup", command=self._probe_startup)
        self.probe_button.grid(row=2, column=0, sticky="ew", padx=8, pady=4)
        ttk.Label(action_box, text="Color By").grid(row=3, column=0, sticky="w", padx=8, pady=(8, 2))
        self.color_mode_combo = ttk.Combobox(
            action_box,
            textvariable=self.color_mode_var,
            state="readonly",
            values=("velocity", "snr"),
        )
        self.color_mode_combo.grid(row=4, column=0, sticky="ew", padx=8, pady=2)
        self.reset_view_button = ttk.Button(action_box, text="Reset 3D View", command=self._reset_3d_view)
        self.reset_view_button.grid(row=5, column=0, sticky="ew", padx=8, pady=(4, 4))
        ttk.Label(action_box, textvariable=self.status_banner_var, wraplength=260).grid(row=6, column=0, sticky="w", padx=8, pady=(6, 8))

        body = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        body.grid(row=1, column=0, sticky="nsew")

        left = ttk.Frame(body, padding=(12, 6, 6, 12))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)
        body.add(left, weight=3)

        right = ttk.Frame(body, padding=(6, 6, 12, 12))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        right.rowconfigure(3, weight=2)
        body.add(right, weight=2)

        status_box = ttk.LabelFrame(left, text="Session Status")
        status_box.grid(row=0, column=0, sticky="ew")
        for col in range(4):
            status_box.columnconfigure(col, weight=1)

        self._add_status_pair(status_box, 0, 0, "Health", self.health_var)
        self._add_status_pair(status_box, 0, 1, "Frames", self.frames_var)
        self._add_status_pair(status_box, 0, 2, "Tracks", self.tracks_var)
        self._add_status_pair(status_box, 0, 3, "Presence", self.presence_var)

        self._add_status_pair(status_box, 1, 0, "RX Bytes", self.rx_var)
        self._add_status_pair(status_box, 1, 1, "Magic Hits", self.magic_var)
        self._add_status_pair(status_box, 1, 2, "FPS", self.fps_var)
        self._add_status_pair(status_box, 1, 3, "Config OK", self.config_ok_var)

        self._add_status_pair(status_box, 2, 0, "Last Packet", self.packet_var)
        self._add_status_pair(status_box, 2, 1, "Plots Updating", self.plots_var)

        self._add_status_pair(status_box, 3, 0, "Reason", self.reason_var, span=4)
        self._add_status_pair(status_box, 4, 0, "Parse Error", self.parse_error_var, span=4)
        self._add_status_pair(status_box, 5, 0, "Command Error", self.command_error_var, span=4)
        self._add_status_pair(status_box, 6, 0, "Connection Error", self.connection_error_var, span=4)
        self._add_status_pair(status_box, 7, 0, "Unknown TLVs", self.unknown_tlvs_var, span=4)
        self._add_status_pair(status_box, 8, 0, "Probe Result", self.probe_var, span=4)
        self._add_status_pair(status_box, 9, 0, "Run Dir", self.run_dir_var, span=4)

        plot_box = ttk.LabelFrame(left, text="3D Radar View")
        plot_box.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        plot_box.columnconfigure(0, weight=1)
        plot_box.rowconfigure(0, weight=1)

        self.figure = Figure(figsize=(9.8, 7.0), dpi=100)
        grid = self.figure.add_gridspec(2, 2, width_ratios=[2.3, 1.0], height_ratios=[1.0, 1.0])
        self.main3d_ax = self.figure.add_subplot(grid[:, 0], projection="3d")
        self.top_ax = self.figure.add_subplot(grid[0, 1])
        self.side_ax = self.figure.add_subplot(grid[1, 1])
        self.figure.tight_layout(pad=2.0)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_box)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._draw_empty_plots()

        port_box = ttk.LabelFrame(right, text="Detected Ports")
        port_box.grid(row=0, column=0, sticky="nsew")
        port_box.columnconfigure(0, weight=1)
        port_box.rowconfigure(0, weight=1)
        self.port_text = tk.Text(port_box, height=12, wrap="word")
        self.port_text.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.port_text.configure(state="disabled")

        cli_box = ttk.LabelFrame(right, text="CLI Log Tail")
        cli_box.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        cli_box.columnconfigure(0, weight=1)
        cli_box.rowconfigure(0, weight=1)
        self.cli_text = tk.Text(cli_box, wrap="word")
        self.cli_text.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.cli_text.configure(state="disabled")

        help_box = ttk.LabelFrame(right, text="Health Criteria")
        help_box.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(
            help_box,
            text=(
                "Healthy requires: cfg success, magic hits, at least 3 parsed frames "
                "within 5 seconds, live 3D updates, and rendered plots."
            ),
            wraplength=420,
        ).grid(row=0, column=0, sticky="w", padx=8, pady=8)

        notes_box = ttk.LabelFrame(right, text="Notes")
        notes_box.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
        notes_box.columnconfigure(0, weight=1)
        notes_box.rowconfigure(0, weight=1)
        self.notes_text = tk.Text(notes_box, wrap="word")
        self.notes_text.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.notes_text.insert(
            "1.0",
            "Use live mode to prove board health and replay mode to compare against saved TI captures.\n"
            "Drag the 3D plot to rotate it. Points can be colored by velocity or SNR, and tracked people render with IDs, boxes, and short motion trails.\n"
            "The Probe Startup action reruns the cfg in staged blocks and reports the first failing stage.",
        )
        self.notes_text.configure(state="disabled")

    def _add_status_pair(
        self,
        parent: ttk.LabelFrame,
        row: int,
        column: int,
        label: str,
        value_var: tk.StringVar,
        span: int = 1,
    ) -> None:
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=column, columnspan=span, sticky="ew", padx=8, pady=4)
        frame.columnconfigure(1, weight=1)
        ttk.Label(frame, text=label).grid(row=0, column=0, sticky="w")
        ttk.Label(frame, textvariable=value_var, wraplength=760).grid(row=0, column=1, sticky="w", padx=(8, 0))

    def _set_text_widget(self, widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _draw_empty_plots(self) -> None:
        self.main3d_ax.clear()
        self.top_ax.clear()
        self.side_ax.clear()
        self.main3d_ax.set_title("3D Point Cloud")
        self.main3d_ax.set_xlabel("X (m)")
        self.main3d_ax.set_ylabel("Y (m)")
        self.main3d_ax.set_zlabel("Z (m)")
        self.main3d_ax.set_xlim(-4.0, 4.0)
        self.main3d_ax.set_ylim(0.0, 8.0)
        self.main3d_ax.set_zlim(-0.2, 3.5)
        self.main3d_ax.grid(True, alpha=0.2)
        self.top_ax.set_title("Top Down (X/Y)")
        self.top_ax.set_xlabel("X (m)")
        self.top_ax.set_ylabel("Y (m)")
        self.top_ax.grid(True, alpha=0.3)
        self.top_ax.set_xlim(-4.0, 4.0)
        self.top_ax.set_ylim(0.0, 8.0)

        self.side_ax.clear()
        self.side_ax.set_title("Side View (Y/Z)")
        self.side_ax.set_xlabel("Y (m)")
        self.side_ax.set_ylabel("Z (m)")
        self.side_ax.grid(True, alpha=0.3)
        self.side_ax.set_xlim(0.0, 8.0)
        self.side_ax.set_ylim(-1.0, 3.5)
        self._reset_3d_view()
        self.canvas.draw_idle()

    def _draw_frame(self, source_mode: str, frame: RadarFrame) -> None:
        points = frame.points
        tracks = frame.tracks
        self._update_track_history(frame)
        if hasattr(self, "main3d_ax"):
            self._view_elev = float(getattr(self.main3d_ax, "elev", self._view_elev))
            self._view_azim = float(getattr(self.main3d_ax, "azim", self._view_azim))

        self.main3d_ax.clear()
        self.top_ax.clear()
        self.side_ax.clear()

        self.main3d_ax.set_title("3D Point Cloud")
        self.main3d_ax.set_xlabel("X (m)")
        self.main3d_ax.set_ylabel("Y (m)")
        self.main3d_ax.set_zlabel("Z (m)")
        self.main3d_ax.grid(True, alpha=0.2)

        self.top_ax.set_title("Top Down (X/Y)")
        self.top_ax.set_xlabel("X (m)")
        self.top_ax.set_ylabel("Y (m)")
        self.top_ax.grid(True, alpha=0.3)

        self.side_ax.set_title("Side View (Y/Z)")
        self.side_ax.set_xlabel("Y (m)")
        self.side_ax.set_ylabel("Z (m)")
        self.side_ax.grid(True, alpha=0.3)

        color_mode = self.color_mode_var.get().strip().lower()
        if points:
            xs = [point.x for point in points]
            ys = [point.y for point in points]
            zs = [point.z for point in points]
            metric = [point.snr for point in points] if color_mode == "snr" else [point.velocity for point in points]
            if color_mode == "snr":
                vmin = min(metric)
                vmax = max(metric) if metric else 1.0
                cmap = "viridis"
            else:
                vmax = max(1.0, max(abs(value) for value in metric))
                vmin = -vmax
                cmap = "coolwarm"

            self.main3d_ax.scatter(xs, ys, zs, c=metric, cmap=cmap, vmin=vmin, vmax=vmax, s=18, alpha=0.78, depthshade=True)
            self.top_ax.scatter(xs, ys, c=metric, cmap=cmap, vmin=vmin, vmax=vmax, s=22, alpha=0.88)
            self.side_ax.scatter(ys, zs, c=metric, cmap=cmap, vmin=vmin, vmax=vmax, s=22, alpha=0.88)

        self._draw_scene_boxes(frame)
        self._draw_tracks(frame)
        self._apply_frame_limits(frame)

        if hasattr(self.main3d_ax, "set_box_aspect"):
            self.main3d_ax.set_box_aspect((1.0, 1.4, 0.8))
        self.main3d_ax.view_init(elev=self._view_elev, azim=self._view_azim)

        label = (
            f"{source_mode.capitalize()} frame {frame.frame_number} | "
            f"points={len(points)} | tracks={len(tracks)} | color={color_mode}"
        )
        self.figure.suptitle(label)
        self.figure.tight_layout(rect=(0, 0, 1, 0.97))
        self.canvas.draw_idle()

    def _update_track_history(self, frame: RadarFrame) -> None:
        seen_ids = set()
        for track in frame.tracks:
            self._track_histories[track.track_id].append((track.x, track.y, track.z))
            self._track_last_seen[track.track_id] = frame.frame_number
            seen_ids.add(track.track_id)
        stale_ids = [
            track_id
            for track_id, last_frame in self._track_last_seen.items()
            if frame.frame_number - last_frame > 40
        ]
        for track_id in stale_ids:
            self._track_last_seen.pop(track_id, None)
            self._track_histories.pop(track_id, None)

    def _draw_scene_boxes(self, frame: RadarFrame) -> None:
        sensor_height = frame.scene.sensor_height_m
        self.main3d_ax.scatter([0.0], [0.0], [sensor_height], c="black", s=30, marker="^")
        self.main3d_ax.text(0.0, 0.0, sensor_height + 0.1, "Radar", color="black")
        for box in frame.scene.static_boundary_boxes:
            self._plot_box(box, color="#7f8c8d", linestyle=":")
        for box in frame.scene.boundary_boxes:
            self._plot_box(box, color="#1f77b4", linestyle="--")
        for box in frame.scene.presence_boxes:
            self._plot_box(box, color="#ff7f0e", linestyle="-.")

    def _draw_tracks(self, frame: RadarFrame) -> None:
        for track in frame.tracks:
            color = self._track_color(track)
            self.main3d_ax.scatter([track.x], [track.y], [track.z], c=[color], s=64, marker="o", edgecolors="black", linewidths=0.6)
            self.top_ax.scatter([track.x], [track.y], c=[color], s=56, marker="o", edgecolors="black", linewidths=0.6)
            self.side_ax.scatter([track.y], [track.z], c=[color], s=56, marker="o", edgecolors="black", linewidths=0.6)
            self.main3d_ax.text(track.x, track.y, track.z + 0.12, f"ID {track.track_id}", color=color)
            self.top_ax.text(track.x, track.y + 0.08, f"{track.track_id}", color=color, fontsize=9)
            self.side_ax.text(track.y, track.z + 0.08, f"{track.track_id}", color=color, fontsize=9)
            if track.bbox is not None:
                self._plot_box(track.bbox, color=color, linestyle="-")

            history = list(self._track_histories.get(track.track_id, ()))
            if len(history) >= 2:
                xs = [p[0] for p in history]
                ys = [p[1] for p in history]
                zs = [p[2] for p in history]
                self.main3d_ax.plot(xs, ys, zs, color=color, alpha=0.55, linewidth=1.6)
                self.top_ax.plot(xs, ys, color=color, alpha=0.55, linewidth=1.4)
                self.side_ax.plot(ys, zs, color=color, alpha=0.55, linewidth=1.4)

    def _plot_box(self, box: RadarBox3D, *, color: str, linestyle: str) -> None:
        x0, x1 = box.x_min, box.x_max
        y0, y1 = box.y_min, box.y_max
        z0, z1 = box.z_min, box.z_max
        corners = [
            (x0, y0, z0),
            (x1, y0, z0),
            (x1, y1, z0),
            (x0, y1, z0),
            (x0, y0, z1),
            (x1, y0, z1),
            (x1, y1, z1),
            (x0, y1, z1),
        ]
        edges = (
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        )
        for a, b in edges:
            xa, ya, za = corners[a]
            xb, yb, zb = corners[b]
            self.main3d_ax.plot([xa, xb], [ya, yb], [za, zb], color=color, linestyle=linestyle, linewidth=1.2, alpha=0.9)

        self.top_ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=color, linestyle=linestyle, linewidth=1.1, alpha=0.85)
        self.side_ax.plot([y0, y1, y1, y0, y0], [z0, z0, z1, z1, z0], color=color, linestyle=linestyle, linewidth=1.1, alpha=0.85)

    def _track_color(self, track: RadarTrack) -> str:
        palette = (
            "#d62728",
            "#2ca02c",
            "#ff7f0e",
            "#1f77b4",
            "#8c564b",
            "#17becf",
            "#e377c2",
            "#bcbd22",
        )
        return palette[track.track_id % len(palette)]

    def _apply_frame_limits(self, frame: RadarFrame) -> None:
        xs: list[float] = [point.x for point in frame.points]
        ys: list[float] = [point.y for point in frame.points]
        zs: list[float] = [point.z for point in frame.points]

        for track in frame.tracks:
            xs.append(track.x)
            ys.append(track.y)
            zs.append(track.z)
            if track.bbox is not None:
                xs.extend([track.bbox.x_min, track.bbox.x_max])
                ys.extend([track.bbox.y_min, track.bbox.y_max])
                zs.extend([track.bbox.z_min, track.bbox.z_max])

        for box in frame.scene.all_boxes():
            xs.extend([box.x_min, box.x_max])
            ys.extend([box.y_min, box.y_max])
            zs.extend([box.z_min, box.z_max])

        if not xs or not ys or not zs:
            self.main3d_ax.set_xlim(-4.0, 4.0)
            self.main3d_ax.set_ylim(0.0, 8.0)
            self.main3d_ax.set_zlim(-0.2, 3.5)
            self.top_ax.set_xlim(-4.0, 4.0)
            self.top_ax.set_ylim(0.0, 8.0)
            self.side_ax.set_xlim(0.0, 8.0)
            self.side_ax.set_ylim(-1.0, 3.5)
            return

        x_margin = max(0.8, (max(xs) - min(xs)) * 0.2)
        y_margin = max(0.8, (max(ys) - min(ys)) * 0.2)
        z_margin = max(0.5, (max(zs) - min(zs)) * 0.2)
        self.main3d_ax.set_xlim(min(xs) - x_margin, max(xs) + x_margin)
        self.main3d_ax.set_ylim(min(0.0, min(ys) - y_margin), max(8.0, max(ys) + y_margin))
        self.main3d_ax.set_zlim(min(-0.2, min(zs) - z_margin), max(3.5, max(zs) + z_margin))
        self.top_ax.set_xlim(min(xs) - x_margin, max(xs) + x_margin)
        self.top_ax.set_ylim(min(0.0, min(ys) - y_margin), max(8.0, max(ys) + y_margin))
        self.side_ax.set_xlim(min(0.0, min(ys) - y_margin), max(8.0, max(ys) + y_margin))
        self.side_ax.set_ylim(min(-1.0, min(zs) - z_margin), max(3.5, max(zs) + z_margin))

    def _reset_3d_view(self) -> None:
        self._view_elev = 18.0
        self._view_azim = -68.0
        if hasattr(self, "main3d_ax"):
            self.main3d_ax.view_init(elev=self._view_elev, azim=self._view_azim)
        if hasattr(self, "canvas"):
            self.canvas.draw_idle()

    def refresh_ports(self) -> None:
        self._port_infos = discover_serial_ports()
        pairs = suggest_serial_port_pairs(self._port_infos)

        devices = [info.device for info in self._port_infos]
        self.config_port_combo["values"] = devices
        self.data_port_combo["values"] = devices

        if pairs:
            suggested_config, suggested_data = pairs[0]
            self.suggestion_var.set(
                f"Suggested pair: config={suggested_config}, data={suggested_data}"
            )
            if not self.config_port_var.get():
                self.config_port_var.set(suggested_config)
            if not self.data_port_var.get():
                self.data_port_var.set(suggested_data)
        else:
            self.suggestion_var.set("Suggested pair: none")

        if not self.config_port_var.get() and devices:
            self.config_port_var.set(devices[0])
        if not self.data_port_var.get() and len(devices) > 1:
            self.data_port_var.set(devices[1])

        lines = []
        for info in self._port_infos:
            lines.append(info.display_label())
            if info.manufacturer or info.product or info.hwid:
                lines.append(
                    f"  manufacturer={info.manufacturer or '-'} | product={info.product or '-'} | hwid={info.hwid or '-'}"
                )
        self._set_text_widget(self.port_text, "\n".join(lines) if lines else "No serial ports found.")

    def _browse_cfg(self) -> None:
        chosen = filedialog.askopenfilename(
            title="Choose radar cfg",
            filetypes=[("Config files", "*.cfg"), ("All files", "*.*")],
        )
        if chosen:
            self.cfg_path_var.set(chosen)

    def _browse_replay(self) -> None:
        chosen = filedialog.askopenfilename(
            title="Choose replay JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if chosen:
            self.replay_path_var.set(chosen)

    def _run_async(
        self,
        label: str,
        worker: Callable[[], object],
        done: Optional[Callable[[object, Optional[BaseException]], None]] = None,
    ) -> None:
        if self._busy:
            return

        self._busy = True
        self.status_banner_var.set(label)
        self._update_mode_controls()

        def target() -> None:
            result: object = None
            error: Optional[BaseException] = None
            try:
                result = worker()
            except BaseException as exc:  # pragma: no cover - surfaced in UI
                error = exc

            def finish() -> None:
                self._busy = False
                if error is not None:
                    self.status_banner_var.set(f"{label} failed")
                    messagebox.showerror("Radar Viewer", str(error))
                if done is not None:
                    done(result, error)
                self._update_mode_controls()

            self.root.after(0, finish)

        threading.Thread(target=target, daemon=True).start()

    def _connect(self) -> None:
        mode = self.mode_var.get()
        cfg_path = self.cfg_path_var.get().strip()
        replay_path = self.replay_path_var.get().strip()
        config_port = self.config_port_var.get().strip()
        data_port = self.data_port_var.get().strip()
        try:
            config_baud = int(self.config_baud_var.get())
            data_baud = int(self.data_baud_var.get())
        except (TypeError, ValueError):
            messagebox.showwarning("Radar Viewer", "Config baud and data baud must be integers.")
            return

        if mode == "live" and not cfg_path:
            messagebox.showwarning("Radar Viewer", "Choose a cfg file before connecting.")
            return
        if mode == "live" and (not config_port or not data_port):
            messagebox.showwarning("Radar Viewer", "Choose both config and data ports before connecting.")
            return
        if mode == "replay" and not replay_path:
            messagebox.showwarning("Radar Viewer", "Choose a replay JSON before connecting.")
            return

        if mode == "live":
            source: FrameSource = IWR6843Driver(
                config_port=config_port,
                data_port=data_port,
                config_path=cfg_path,
                config_baud=config_baud,
                data_baud=data_baud,
            )
        else:
            source = ReplayRadarSource(replay_path)

        self._source = source
        self._last_drawn_key = None
        self._track_histories.clear()
        self._track_last_seen.clear()

        def worker() -> bool:
            return bool(source.start())

        def done(result: object, error: Optional[BaseException]) -> None:
            if error is None:
                if result:
                    self.status_banner_var.set("Connected")
                else:
                    self.status_banner_var.set("Connect finished with errors")
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
                self.status_banner_var.set("Disconnected")
            self._track_histories.clear()
            self._track_last_seen.clear()
            self._refresh_state_panel()

        self._run_async("Disconnecting...", worker, done)

    def _probe_startup(self) -> None:
        if self.mode_var.get() != "live":
            messagebox.showinfo("Radar Viewer", "Probe Startup is only available in live mode.")
            return

        if isinstance(self._source, IWR6843Driver):
            source = self._source
        else:
            try:
                config_baud = int(self.config_baud_var.get())
                data_baud = int(self.data_baud_var.get())
            except (TypeError, ValueError):
                messagebox.showwarning("Radar Viewer", "Config baud and data baud must be integers.")
                return
            source = IWR6843Driver(
                config_port=self.config_port_var.get().strip(),
                data_port=self.data_port_var.get().strip(),
                config_path=self.cfg_path_var.get().strip(),
                config_baud=config_baud,
                data_baud=data_baud,
            )
            self._source = source

        self._last_drawn_key = None

        def worker() -> object:
            return source.probe_startup()

        def done(result: object, error: Optional[BaseException]) -> None:
            del result
            if error is None:
                self.status_banner_var.set("Probe complete")
            self._refresh_state_panel()

        self._run_async("Running staged probe...", worker, done)

    def _refresh_state_panel(self) -> None:
        state = self._current_state()
        self._apply_state(state)

    def _current_state(self) -> Optional[RadarSessionState]:
        if self._source is None:
            return None
        try:
            return self._source.session_state()
        except Exception as exc:  # pragma: no cover - defensive UI path
            self.status_banner_var.set(f"State error: {exc}")
            return None

    def _apply_state(self, state: Optional[RadarSessionState]) -> None:
        if state is None:
            return

        self.health_var.set(state.health_verdict)
        self.reason_var.set(state.health_reason)
        self.frames_var.set(str(state.frames))
        self.tracks_var.set(str(state.tracks))
        self.presence_var.set(state.presence or "-")
        self.fps_var.set(f"{state.fps:.2f}")
        self.rx_var.set(str(state.rx_bytes))
        self.magic_var.set(str(state.magic_hits))
        self.packet_var.set(str(state.last_packet_size))
        self.config_ok_var.set(str(state.config_ok))
        self.plots_var.set(str(state.plots_updating))
        self.parse_error_var.set(state.last_parse_error)
        self.command_error_var.set(state.last_command_error)
        self.connection_error_var.set(state.connection_error)
        unknown = ", ".join(
            f"type {key}: {count}"
            for key, count in sorted(state.unknown_tlv_counts.items())
        )
        self.unknown_tlvs_var.set(unknown)
        self.run_dir_var.set(state.run_dir)
        self.probe_var.set(
            state.probe_failed_stage or state.probe_log_tail.splitlines()[0] if state.probe_log_tail else ""
        )

        if state.cli_log_tail != self._last_cli_text:
            self._last_cli_text = state.cli_log_tail
            self._set_text_widget(self.cli_text, state.cli_log_tail or "(no CLI output yet)")

    def _poll_source(self) -> None:
        state = self._current_state()
        if state is not None:
            self._apply_state(state)
            if self._source is not None:
                try:
                    frame = self._source.latest_frame()
                except Exception as exc:  # pragma: no cover - defensive UI path
                    self.status_banner_var.set(f"Frame error: {exc}")
                    frame = None
                if frame is not None:
                    source_mode = state.mode
                    key = (source_mode, frame.frame_number)
                    if key != self._last_drawn_key:
                        self._draw_frame(source_mode, frame)
                        self._source.note_frame_rendered(frame)
                        self._last_drawn_key = key
                elif state.health_verdict == HEALTH_REPLAY_MODE:
                    self._draw_empty_plots()
                    self._last_drawn_key = None
        self.root.after(200, self._poll_source)

    def _update_mode_controls(self) -> None:
        live_enabled = self.mode_var.get() == "live"
        replay_enabled = not live_enabled

        self.config_port_combo.configure(state="readonly" if live_enabled and not self._busy else "disabled")
        self.data_port_combo.configure(state="readonly" if live_enabled and not self._busy else "disabled")
        self.config_baud_entry.configure(state="normal" if live_enabled and not self._busy else "disabled")
        self.data_baud_entry.configure(state="normal" if live_enabled and not self._busy else "disabled")
        self.cfg_entry.configure(state="normal" if live_enabled and not self._busy else "disabled")
        self.cfg_button.configure(state="normal" if live_enabled and not self._busy else "disabled")
        self.refresh_ports_button.configure(state="normal" if live_enabled and not self._busy else "disabled")

        self.replay_entry.configure(state="normal" if replay_enabled and not self._busy else "disabled")
        self.replay_button.configure(state="normal" if replay_enabled and not self._busy else "disabled")

        self.connect_button.configure(state="normal" if not self._busy else "disabled")
        self.disconnect_button.configure(
            state="normal" if self._source is not None and not self._busy else "disabled"
        )
        self.probe_button.configure(
            state="normal" if live_enabled and not self._busy else "disabled"
        )
        self.color_mode_combo.configure(state="readonly" if not self._busy else "disabled")
        self.reset_view_button.configure(state="normal" if not self._busy else "disabled")

    def _on_close(self) -> None:
        try:
            if self._source is not None:
                self._source.stop()
        finally:
            self.root.destroy()


def _default_cfg_path() -> str:
    preferred = Path("iwr6843_people_tracking.cfg")
    if preferred.exists():
        return str(preferred)
    fallback = Path("iwr6843_config.cfg")
    return str(fallback) if fallback.exists() else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("live", "replay"), default="live")
    parser.add_argument("--replay", default="")
    parser.add_argument("--cfg", default=_default_cfg_path())
    parser.add_argument("--config-port", default="")
    parser.add_argument("--data-port", default="")
    parser.add_argument("--config-baud", type=int, default=115200)
    parser.add_argument("--data-baud", type=int, default=921600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    RadarViewerApp(
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


if __name__ == "__main__":
    main()
