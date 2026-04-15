#!/usr/bin/env python3
"""Radar debug capture and analysis helpers.

Use this when the 3-D viewer looks unstable and we need evidence from the
actual UART stream instead of eyeballing the OpenGL scene.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from radar import IWR6843Driver


def _default_cfg_path() -> str:
    preferred = Path("iwr6843_people_tracking_20fps.cfg")
    if preferred.exists():
        return str(preferred)
    preferred = Path("iwr6843_people_tracking.cfg")
    if preferred.exists():
        return str(preferred)
    fallback = Path("iwr6843_config.cfg")
    return str(fallback)


def _cfg_key_lines(cfg_path: str) -> Dict[str, str]:
    wanted = {
        "sensorPosition",
        "gatingParam",
        "allocationParam",
        "maxAcceleration",
        "frameCfg",
        "trackingCfg",
    }
    out: Dict[str, str] = {}
    path = Path(cfg_path)
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        key = line.split()[0]
        if key in wanted:
            out[key] = line
    return out


def _latest_frames_path() -> Optional[Path]:
    candidates = list(Path("diagnostics_runs").glob("*/frames.jsonl"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _resolve_frames_path(path_text: str) -> Path:
    if path_text == "latest":
        latest = _latest_frames_path()
        if latest is None:
            raise FileNotFoundError("No diagnostics_runs/*/frames.jsonl file exists yet.")
        return latest

    path = Path(path_text)
    if path.is_dir():
        path = path / "frames.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Frame debug file not found: {path}")
    return path


def iter_frame_records(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if payload.get("type") == "frame":
                yield payload


def analyze_frame_records(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    total_frames = 0
    frames_with_points = 0
    frames_with_tracks = 0
    frames_with_presence = 0
    frames_presence_without_tracks = 0
    frames_points_without_tracks = 0
    current_no_track_streak = 0
    max_no_track_streak = 0
    single_track_id_switches = 0
    last_single_track_id: Optional[int] = None
    timestamps: List[float] = []
    point_counts: List[int] = []
    bbox_widths: List[float] = []
    bbox_depths: List[float] = []
    bbox_heights: List[float] = []
    track_seen_frames: Dict[int, int] = defaultdict(int)
    track_first_frame: Dict[int, int] = {}
    track_last_frame: Dict[int, int] = {}
    track_gap_events = 0
    track_gap_frames = 0

    for record in records:
        total_frames += 1
        timestamps.append(float(record.get("timestamp") or 0.0))
        point_count = int(record.get("point_count") or 0)
        point_counts.append(point_count)
        tracks = record.get("tracks") or []
        track_ids = sorted(int(track.get("track_id")) for track in tracks if "track_id" in track)
        presence_active = record.get("presence") not in (None, 0, "0", "")

        if point_count > 0:
            frames_with_points += 1
        if track_ids:
            frames_with_tracks += 1
            current_no_track_streak = 0
        else:
            current_no_track_streak += 1
            max_no_track_streak = max(max_no_track_streak, current_no_track_streak)
        if presence_active:
            frames_with_presence += 1
        if presence_active and not track_ids:
            frames_presence_without_tracks += 1
        if point_count > 0 and not track_ids:
            frames_points_without_tracks += 1

        frame_number = int(record.get("frame_number") or total_frames)
        for track_id in track_ids:
            if track_id not in track_first_frame:
                track_first_frame[track_id] = frame_number
            last_frame = track_last_frame.get(track_id)
            if last_frame is not None and frame_number > last_frame + 1:
                track_gap_events += 1
                track_gap_frames += frame_number - last_frame - 1
            track_last_frame[track_id] = frame_number
            track_seen_frames[track_id] += 1

        if len(track_ids) == 1:
            track_id = track_ids[0]
            if last_single_track_id is not None and last_single_track_id != track_id:
                single_track_id_switches += 1
            last_single_track_id = track_id
        elif len(track_ids) > 1:
            last_single_track_id = None

        for track in tracks:
            bbox = track.get("bbox") or {}
            for key, bucket in (
                ("width", bbox_widths),
                ("depth", bbox_depths),
                ("height", bbox_heights),
            ):
                if bbox.get(key) is not None:
                    bucket.append(float(bbox[key]))

    duration_s = 0.0
    if len(timestamps) >= 2 and timestamps[-1] >= timestamps[0]:
        duration_s = timestamps[-1] - timestamps[0]

    total = max(total_frames, 1)
    presence_total = max(frames_with_presence, 1)
    tracks = {
        str(track_id): {
            "first_frame": int(track_first_frame.get(track_id, 0)),
            "last_frame": int(track_last_frame.get(track_id, 0)),
            "seen_frames": int(track_seen_frames.get(track_id, 0)),
        }
        for track_id in sorted(track_seen_frames)
    }

    return {
        "total_frames": total_frames,
        "duration_s": round(duration_s, 3),
        "approx_fps": round((total_frames - 1) / duration_s, 2) if duration_s > 0.0 and total_frames > 1 else 0.0,
        "frames_with_points": frames_with_points,
        "frames_with_tracks": frames_with_tracks,
        "frames_with_presence": frames_with_presence,
        "frames_presence_without_tracks": frames_presence_without_tracks,
        "frames_points_without_tracks": frames_points_without_tracks,
        "track_coverage_pct": round(100.0 * frames_with_tracks / total, 2),
        "presence_track_agreement_pct": round(
            100.0 * (frames_with_presence - frames_presence_without_tracks) / presence_total,
            2,
        ),
        "max_no_track_streak": max_no_track_streak,
        "unique_track_ids": sorted(track_seen_frames),
        "single_track_id_switches": single_track_id_switches,
        "track_gap_events": track_gap_events,
        "track_gap_frames": track_gap_frames,
        "point_count": _series_summary(point_counts),
        "bbox_width_m": _series_summary(bbox_widths),
        "bbox_depth_m": _series_summary(bbox_depths),
        "bbox_height_m": _series_summary(bbox_heights),
        "tracks": tracks,
    }


def _series_summary(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": round(float(min(values)), 3),
        "max": round(float(max(values)), 3),
        "mean": round(float(sum(values) / len(values)), 3),
    }


def print_analysis(summary: Dict[str, Any], *, source: Path) -> None:
    print(f"Source: {source}")
    print(f"Frames: {summary['total_frames']} over {summary['duration_s']}s @ ~{summary['approx_fps']} fps")
    print(f"Track coverage: {summary['track_coverage_pct']}%")
    print(f"Presence/track agreement: {summary['presence_track_agreement_pct']}%")
    print(f"Presence but no track frames: {summary['frames_presence_without_tracks']}")
    print(f"Points but no track frames: {summary['frames_points_without_tracks']}")
    print(f"Max no-track streak: {summary['max_no_track_streak']} frames")
    print(f"Unique track IDs: {summary['unique_track_ids']}")
    print(f"Single-person ID switches: {summary['single_track_id_switches']}")
    print(f"Track gap events: {summary['track_gap_events']} ({summary['track_gap_frames']} missing frames)")
    print(f"Point count/frame: {summary['point_count']}")
    print(f"Box width/depth/height m: {summary['bbox_width_m']} / {summary['bbox_depth_m']} / {summary['bbox_height_m']}")


def analyze_command(args: argparse.Namespace) -> int:
    frames_path = _resolve_frames_path(args.path)
    summary = analyze_frame_records(iter_frame_records(frames_path))
    print_analysis(summary, source=frames_path)
    if args.write_json:
        output_path = frames_path.with_name("tracking_analysis.json")
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote: {output_path}")
    return 0


def live_capture_command(args: argparse.Namespace) -> int:
    cfg_lines = _cfg_key_lines(args.cfg)
    print(f"Using cfg: {args.cfg}")
    for key in ("sensorPosition", "allocationParam", "gatingParam", "maxAcceleration", "frameCfg", "trackingCfg"):
        if key in cfg_lines:
            print(f"  {cfg_lines[key]}")

    driver = IWR6843Driver(
        config_port=args.config_port,
        data_port=args.data_port,
        config_path=args.cfg,
        config_baud=args.config_baud,
        data_baud=args.data_baud,
    )
    if not driver.start():
        state = driver.session_state()
        print(f"Capture failed: {state.health_verdict} - {state.health_reason}")
        print(f"Command error: {state.last_command_error or '-'}")
        print(f"Connection error: {state.connection_error or '-'}")
        return 2

    print(f"Capturing {args.seconds:.1f}s of live radar debug data...")
    interrupted = False
    try:
        end_time = time.time() + float(args.seconds)
        while time.time() < end_time:
            time.sleep(min(1.0, max(0.05, end_time - time.time())))
            state = driver.session_state()
            print(
                f"frames={state.frames} fps={state.fps:.1f} tracks={state.tracks} "
                f"presence={state.presence or '-'} health={state.health_verdict}"
            )
    except KeyboardInterrupt:
        interrupted = True
        print("\nCapture interrupted; saving and analyzing the frames captured so far...")
    finally:
        state = driver.session_state()
        frames_path = Path(state.frame_debug_path) if state.frame_debug_path else None
        driver.stop()

    if frames_path is None or not frames_path.exists():
        print("Capture finished, but no frames.jsonl file was written.")
        return 1

    summary = analyze_frame_records(iter_frame_records(frames_path))
    output_path = frames_path.with_name("tracking_analysis.json")
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print_analysis(summary, source=frames_path)
    print(f"Wrote: {output_path}")
    return 130 if interrupted else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze or capture radar tracking debug timelines.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Analyze a diagnostics run frames.jsonl file.")
    analyze.add_argument("path", nargs="?", default="latest", help="Run directory, frames.jsonl path, or 'latest'.")
    analyze.add_argument("--write-json", action="store_true", help="Write tracking_analysis.json next to frames.jsonl.")
    analyze.set_defaults(func=analyze_command)

    capture = subparsers.add_parser("live-capture", help="Run the radar headlessly and save a debug timeline.")
    capture.add_argument("--seconds", type=float, default=20.0)
    capture.add_argument("--cfg", default=_default_cfg_path())
    capture.add_argument("--config-port", default="/dev/ttyACM0")
    capture.add_argument("--data-port", default="/dev/ttyACM1")
    capture.add_argument("--config-baud", type=int, default=115200)
    capture.add_argument("--data-baud", type=int, default=921600)
    capture.set_defaults(func=live_capture_command)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
