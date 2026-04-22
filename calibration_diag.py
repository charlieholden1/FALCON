#!/usr/bin/env python3
"""Radar-camera calibration diagnostic CLI.

Measures how far off the saved calibration is from a set of captured
calibration samples, re-solves from those samples to show the best
achievable accuracy, and flags likely causes (physical mount, radar
sensorPosition, intrinsics, sample coverage).

Typical use:

    python3 calibration_diag.py report \\
        --samples latest \\
        --calibration radar_camera_calibration.json \\
        --cfg iwr6843_people_tracking_20fps.cfg

``--samples latest`` picks the newest
``diagnostics_runs/<run>/auto_calibration_samples.json``.

Subcommands:
    eval      - accuracy of a calibration file against samples (before).
    resolve   - re-solve from samples, print delta vs current file (after).
    report    - full before/after + coverage + physical/cfg sanity.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from radar import CameraProjection
from radar_camera_fusion import (
    CalibrationSample,
    _sample_pixel_errors,
    load_calibration_samples,
    solve_calibration_auto,
)


# Physical expectations from measured mount geometry (April 2026).
# Camera is 2.5 in (0.064 m) above radar, 1.5 in (0.038 m) behind radar (closer to wall).
EXPECTED = {
    "tx_m": (0.0, 0.10),           # (target, tolerance)
    "ty_m": (0.064, 0.03),         # 2.5 in above — measured
    "tz_m": (0.038, 0.03),         # 1.5 in behind — measured (camera-Z = radar-Y; camera further from scene)
    "yaw_deg": (0.0, 8.0),
    "pitch_deg": (0.0, 8.0),
    "roll_deg": (0.0, 5.0),
    "sensor_height_m": (1.88, 0.10),
    "sensor_tilt_deg": (8.0, 5.0),
    # RealSense D435 @ 640x480 factory-typical.
    "fx_640x480": (605.0, 20.0),
    "fy_640x480": (605.0, 20.0),
    "cx_640x480": (320.0, 15.0),
    "cy_640x480": (240.0, 15.0),
    # RealSense D435 @ 848x480 factory-typical.
    "fx_848x480": (630.0, 25.0),
    "fy_848x480": (630.0, 25.0),
    "cx_848x480": (424.0, 15.0),
    "cy_848x480": (240.0, 15.0),
}

FRAME_W_DEFAULT = 640
FRAME_H_DEFAULT = 480


# ---------------------------------------------------------------- paths --


def _latest_samples_path() -> Optional[Path]:
    candidates = list(Path("diagnostics_runs").glob("*/auto_calibration_samples.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _resolve_samples_path(text: str) -> Path:
    if text == "latest":
        latest = _latest_samples_path()
        if latest is None:
            raise FileNotFoundError(
                "No diagnostics_runs/*/auto_calibration_samples.json found. "
                "Run Auto Calibrate once (even if it rejects) to populate samples."
            )
        return latest
    p = Path(text)
    if p.is_dir():
        p = p / "auto_calibration_samples.json"
    if not p.exists():
        raise FileNotFoundError(f"Samples file not found: {p}")
    return p


def _load_projection(path: Path) -> CameraProjection:
    proj = CameraProjection()
    proj.load(path)
    return proj


# ---------------------------------------------------------- sample stats --


@dataclass
class SampleDiag:
    index: int
    radar_track_id: int
    u: float
    v: float
    radar_xyz: Tuple[float, float, float]
    cam_xyz: Optional[Tuple[float, float, float]]
    depth_m: Optional[float]
    pixel_error_px: float
    residual_m: Optional[float]        # 3-D ||proj·radar - cam_xyz||
    depth_mismatch_m: Optional[float]  # |depth - radar.y|
    image_cell: Tuple[int, int]
    depth_band: str


def _cell(u: float, v: float, w: int, h: int) -> Tuple[int, int]:
    col = min(2, max(0, int(u / max(w / 3.0, 1.0))))
    row = min(2, max(0, int(v / max(h / 3.0, 1.0))))
    return (row, col)


def _band(y: float) -> str:
    if y < 1.5:
        return "near"
    if y <= 3.0:
        return "mid"
    return "far"


def _radar_to_camera_xyz(projection: CameraProjection, radar_xyz: Sequence[float]) -> np.ndarray:
    return projection.radar_to_camera(np.asarray(radar_xyz, dtype=np.float64))


def analyze_samples(
    samples: Sequence[CalibrationSample],
    projection: CameraProjection,
    frame_w: int,
    frame_h: int,
) -> List[SampleDiag]:
    px_errs = _sample_pixel_errors(samples, projection)
    out: List[SampleDiag] = []
    for idx, (sample, err) in enumerate(zip(samples, px_errs)):
        radar_xyz = tuple(float(v) for v in sample.radar_xyz)
        cam_xyz = tuple(float(v) for v in sample.camera_xyz) if sample.camera_xyz else None
        residual_m: Optional[float] = None
        if cam_xyz is not None:
            projected_cam = _radar_to_camera_xyz(projection, radar_xyz)
            residual_m = float(np.linalg.norm(projected_cam - np.asarray(cam_xyz)))
        depth_mismatch: Optional[float] = None
        if sample.depth_m is not None:
            depth_mismatch = float(abs(float(sample.depth_m) - float(radar_xyz[1])))
        out.append(
            SampleDiag(
                index=idx,
                radar_track_id=int(sample.radar_track_id),
                u=float(sample.image_uv[0]),
                v=float(sample.image_uv[1]),
                radar_xyz=radar_xyz,
                cam_xyz=cam_xyz,
                depth_m=float(sample.depth_m) if sample.depth_m is not None else None,
                pixel_error_px=float(err),
                residual_m=residual_m,
                depth_mismatch_m=depth_mismatch,
                image_cell=_cell(float(sample.image_uv[0]), float(sample.image_uv[1]), frame_w, frame_h),
                depth_band=_band(float(radar_xyz[1])),
            )
        )
    return out


def aggregate(diags: Sequence[SampleDiag]) -> Dict[str, Any]:
    if not diags:
        return {
            "count": 0,
            "mean_px": 0.0,
            "median_px": 0.0,
            "p90_px": 0.0,
            "max_px": 0.0,
            "rmse_m": 0.0,
            "median_depth_mismatch_m": 0.0,
            "cells_filled": 0,
            "bands_filled": 0,
            "unique_radar_ids": [],
        }
    px = np.array([d.pixel_error_px for d in diags], dtype=np.float64)
    res = [d.residual_m for d in diags if d.residual_m is not None]
    dm = [d.depth_mismatch_m for d in diags if d.depth_mismatch_m is not None]
    cells = {d.image_cell for d in diags}
    bands = {d.depth_band for d in diags}
    return {
        "count": len(diags),
        "mean_px": float(np.mean(px)),
        "median_px": float(np.median(px)),
        "p90_px": float(np.percentile(px, 90)),
        "max_px": float(np.max(px)),
        "rmse_m": float(np.sqrt(np.mean(np.array(res) ** 2))) if res else 0.0,
        "residual_samples": len(res),
        "median_depth_mismatch_m": float(np.median(dm)) if dm else 0.0,
        "cells_filled": len(cells),
        "bands_filled": len(bands),
        "unique_radar_ids": sorted({d.radar_track_id for d in diags}),
    }


def flag_outliers(diags: Sequence[SampleDiag]) -> List[int]:
    if len(diags) < 4:
        return []
    errs = np.array([d.pixel_error_px for d in diags], dtype=np.float64)
    med = float(np.median(errs))
    mad = float(np.median(np.abs(errs - med)))
    threshold = med + max(30.0, 2.5 * mad)
    return [int(d.index) for d in diags if d.pixel_error_px > threshold]


# --------------------------------------------------------- physical check --


def _check(value: float, target: float, tol: float, label: str, unit: str) -> Tuple[str, str]:
    diff = value - target
    if abs(diff) <= tol:
        return "OK", f"  [OK]   {label:<18} = {value:8.3f}{unit}  (target {target:+.2f}{unit}, tol {tol:.2f})"
    return "WARN", (
        f"  [WARN] {label:<18} = {value:8.3f}{unit}  "
        f"(target {target:+.2f}{unit} ±{tol:.2f}, off by {diff:+.3f}{unit})"
    )


def physical_sanity(params: Dict[str, float], frame_w: int, frame_h: int) -> List[str]:
    lines: List[str] = ["Physical sanity vs agent.md + RealSense factory expectations:"]
    warns: List[str] = []

    def record(status: str, line: str, warn_hint: Optional[str] = None) -> None:
        lines.append(line)
        if status == "WARN" and warn_hint is not None:
            warns.append(warn_hint)

    # Extrinsics.
    tx_t, tx_tol = EXPECTED["tx_m"]
    ty_t, ty_tol = EXPECTED["ty_m"]
    tz_t, tz_tol = EXPECTED["tz_m"]
    yaw_t, yaw_tol = EXPECTED["yaw_deg"]
    pitch_t, pitch_tol = EXPECTED["pitch_deg"]
    roll_t, roll_tol = EXPECTED["roll_deg"]

    s, l = _check(params["tx"], tx_t, tx_tol, "tx (lateral)", "m")
    record(s, l, "tx off: camera not horizontally centered with radar, or radar mount offset — re-measure the rack.")
    s, l = _check(params["ty"], ty_t, ty_tol, "ty (camera-down)", "m")
    record(s, l, "ty off: camera-radar vertical offset wrong; expected 2.5 in (0.064 m) camera above radar.")
    s, l = _check(params["tz"], tz_t, tz_tol, "tz (depth)", "m")
    record(s, l, "tz off: expected 1.5 in (0.038 m) — camera is behind radar (closer to wall). Check mount plate.")
    s, l = _check(params["yaw_deg"], yaw_t, yaw_tol, "yaw_deg", "°")
    record(s, l, "yaw large: sensors not aimed same direction (rotate one about vertical axis).")
    s, l = _check(params["pitch_deg"], pitch_t, pitch_tol, "pitch_deg", "°")
    record(s, l, (
        "pitch large: likely cause is radar sensorPosition tilt mismatch — "
        "either the cfg's tilt value does not match the real board tilt, "
        "or the board is bowing. Remeasure board angle with an inclinometer "
        "and update `sensorPosition 1.88 0 <tilt_deg>` in the cfg."
    ))
    s, l = _check(params["roll_deg"], roll_t, roll_tol, "roll_deg", "°")
    record(s, l, "roll large: radar board or camera is rotated around line-of-sight axis; level the mount.")

    # Intrinsics — checked for 640×480 and 848×480 RealSense profiles.
    if frame_w == 640 and frame_h == 480:
        res_label = "640x480"
        intrinsic_keys = [("fx","fx_640x480"),("fy","fy_640x480"),("cx","cx_640x480"),("cy","cy_640x480")]
    elif frame_w == 848 and frame_h == 480:
        res_label = "848x480"
        intrinsic_keys = [("fx","fx_848x480"),("fy","fy_848x480"),("cx","cx_848x480"),("cy","cy_848x480")]
    else:
        res_label = None
        intrinsic_keys = []

    if intrinsic_keys:
        lines.append(f"Intrinsics vs RealSense D435 factory-typical at {res_label}:")
        for key, tgt_key in intrinsic_keys:
            tgt, tol = EXPECTED[tgt_key]
            s, l = _check(params[key], tgt, tol, key, "")
            record(s, l, f"{key} far from factory-typical: intrinsics may not match the current RealSense stream resolution.")
        # fx ≈ fy cross-check.
        if abs(params["fx"] - params["fy"]) > 8.0:
            lines.append(
                f"  [WARN] fx-fy gap  = {params['fx'] - params['fy']:+.2f}px — "
                "unusually anisotropic intrinsics; check RealSense intrinsics source."
            )
            warns.append("fx/fy mismatch: intrinsics likely from the wrong stream profile.")
    else:
        lines.append(
            f"Intrinsics checks skipped: frame size {frame_w}x{frame_h} "
            "is not a known baseline (640x480 or 848x480)."
        )

    if warns:
        lines.append("")
        lines.append("Likely physical/setup causes:")
        for w in warns:
            lines.append(f"  - {w}")
    return lines


# ------------------------------------------------------------- cfg check --


def parse_cfg_tracker(cfg_path: Path) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    if not cfg_path.exists():
        return out
    _KEYS = {
        "sensorPosition", "allocationParam", "gatingParam", "maxAcceleration",
        "frameCfg", "fovCfg", "compRangeBiasAndRxChanPhase",
        "boundaryBox", "staticBoundaryBox", "presenceBoundaryBox",
    }
    for raw in cfg_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        parts = line.split()
        if parts and parts[0] in _KEYS:
            out[parts[0]] = parts[1:]
    return out


def cfg_sanity(cfg_path: Path) -> List[str]:
    lines: List[str] = [f"Radar cfg sanity ({cfg_path}):"]
    parsed = parse_cfg_tracker(cfg_path)
    if not parsed:
        lines.append("  [WARN] cfg not found or empty — cannot check tracker geometry.")
        return lines

    sp = parsed.get("sensorPosition")
    if sp is None or len(sp) < 3:
        lines.append("  [WARN] sensorPosition missing. Add `sensorPosition 1.88 0 8` per current mount.")
    else:
        try:
            h = float(sp[0]); az = float(sp[1]); tilt = float(sp[2])
        except ValueError:
            lines.append(f"  [WARN] sensorPosition values unparseable: {sp}")
        else:
            target_h, tol_h = EXPECTED["sensor_height_m"]
            if abs(h - target_h) <= tol_h:
                lines.append(f"  [OK]   sensorPosition height = {h:.2f}m (target {target_h:.2f})")
            else:
                lines.append(
                    f"  [WARN] sensorPosition height = {h:.2f}m, expected {target_h:.2f}±{tol_h:.2f}. "
                    f"Tracker world-Z will be off — update the first value to the actual floor-to-radar-centre metres."
                )
            if abs(az) > 3.0:
                lines.append(f"  [WARN] sensorPosition azimuth = {az:.1f}°, expected ~0 (mount pointing straight forward).")
            else:
                lines.append(f"  [OK]   sensorPosition azimuth = {az:.1f}°")
            tilt_t, tilt_tol = EXPECTED["sensor_tilt_deg"]
            if abs(tilt - tilt_t) <= tilt_tol:
                lines.append(f"  [OK]   sensorPosition tilt   = {tilt:.1f}° (target {tilt_t:.0f}°±{tilt_tol:.0f}°)")
            else:
                lines.append(
                    f"  [WARN] sensorPosition tilt   = {tilt:.1f}°, expected {tilt_t:.0f}°±{tilt_tol:.0f}°. "
                    f"Physically measure board elevation angle with a protractor/inclinometer. "
                    f"Wrong tilt shows up as a large pitch_deg in calibration."
                )

    # Range bias.
    rb = parsed.get("compRangeBiasAndRxChanPhase")
    if rb is None:
        lines.append("  [WARN] compRangeBiasAndRxChanPhase missing — add line from range_bias_calibration.py output.")
    else:
        try:
            bias = float(rb[0])
        except (ValueError, IndexError):
            lines.append(f"  [WARN] compRangeBiasAndRxChanPhase unparseable: {rb}")
        else:
            if bias == 0.0:
                lines.append(
                    "  [WARN] Range bias is 0.0 m — run range_bias_calibration.py with a corner "
                    "reflector to measure the actual hardware range offset."
                )
            else:
                lines.append(f"  [OK]   range bias = {bias:+.5f} m")

    # Elevation FOV.
    fov = parsed.get("fovCfg")
    if fov is not None and len(fov) >= 3:
        try:
            elev = float(fov[2])
        except ValueError:
            pass
        else:
            if elev > 22.0:
                lines.append(
                    f"  [WARN] fovCfg elevation = ±{elev:.1f}° (> 22°). On a single-floor deployment "
                    "this is wider than needed and may admit ceiling multipath. Consider ±20°."
                )
            else:
                lines.append(f"  [OK]   fovCfg elevation = ±{elev:.1f}°")

    for key in ("allocationParam", "gatingParam", "maxAcceleration", "frameCfg"):
        if key in parsed:
            lines.append(f"  [info] {key} {' '.join(parsed[key])}")
    return lines


# --------------------------------------------------------------- room check --


def room_sanity(
    cfg_path: Path,
    room_depth_m: Optional[float],
    room_half_width_m: Optional[float],
) -> List[str]:
    """Cross-check boundary box extents against known room dimensions.

    Args:
        cfg_path: Radar cfg file.
        room_depth_m: Distance from sensor wall to far wall (metres).
        room_half_width_m: Half the room width (metres).

    Returns a list of warning strings; empty = all OK.
    Skips checks for which the corresponding argument is None.
    """
    lines: List[str] = [f"Room sanity ({cfg_path}):"]
    parsed = parse_cfg_tracker(cfg_path)
    if not parsed:
        lines.append("  [WARN] cfg not found or empty — cannot check boundary geometry.")
        return lines

    def _box(key: str) -> Optional[List[float]]:
        vals = parsed.get(key)
        if vals is None or len(vals) < 6:
            return None
        try:
            return [float(v) for v in vals[:6]]
        except ValueError:
            return None

    bb = _box("boundaryBox")
    sbb = _box("staticBoundaryBox")
    pbb = _box("presenceBoundaryBox")

    for label, box in (("boundaryBox", bb), ("staticBoundaryBox", sbb), ("presenceBoundaryBox", pbb)):
        if box is None:
            lines.append(f"  [info] {label} not found in cfg")
            continue
        x_min, x_max, y_min, y_max, z_min, z_max = box
        ok = True
        if room_depth_m is not None and y_max > room_depth_m:
            lines.append(
                f"  [WARN] {label} max-Y {y_max:.1f} m > room depth {room_depth_m:.1f} m. "
                "Back-wall reflections may create ghost tracks. Set max-Y to room_depth − 0.5 m."
            )
            ok = False
        if room_half_width_m is not None and max(abs(x_min), abs(x_max)) > room_half_width_m:
            lines.append(
                f"  [WARN] {label} lateral extent ±{max(abs(x_min), abs(x_max)):.1f} m "
                f"> room half-width {room_half_width_m:.1f} m. Side-wall returns may enter tracking zone."
            )
            ok = False
        if label == "staticBoundaryBox" and room_depth_m is not None and y_max > room_depth_m - 0.3:
            lines.append(
                f"  [WARN] staticBoundaryBox max-Y {y_max:.1f} m is within 0.3 m of far wall — "
                "static clutter suppression may clip legitimate far-wall targets."
            )
            ok = False
        if ok:
            lines.append(f"  [OK]   {label} fits within room dimensions")
    return lines


# --------------------------------------------------------------- reports --


def print_sample_table(diags: Sequence[SampleDiag]) -> None:
    print(f"  {'idx':>3}  {'track':>5}  {'u':>6}  {'v':>6}  {'cell':>5}  {'band':>4}  "
          f"{'err_px':>7}  {'res_m':>6}  {'|depth-y|':>9}")
    for d in diags:
        cell = f"{d.image_cell[0]},{d.image_cell[1]}"
        res = f"{d.residual_m:.3f}" if d.residual_m is not None else "  -  "
        dm = f"{d.depth_mismatch_m:.3f}" if d.depth_mismatch_m is not None else "  -  "
        print(
            f"  {d.index:>3}  {d.radar_track_id:>5}  {d.u:>6.0f}  {d.v:>6.0f}  "
            f"{cell:>5}  {d.depth_band:>4}  {d.pixel_error_px:>7.1f}  {res:>6}  {dm:>9}"
        )


def print_aggregate(title: str, agg: Dict[str, Any]) -> None:
    print(f"{title}")
    print(f"  samples               : {agg['count']} "
          f"(with depth/3-D: {agg.get('residual_samples', 0)})")
    print(f"  pixel error px        : median {agg['median_px']:.1f}, mean {agg['mean_px']:.1f}, "
          f"p90 {agg['p90_px']:.1f}, max {agg['max_px']:.1f}")
    if agg.get('residual_samples', 0) > 0:
        print(f"  3-D residual rmse_m   : {agg['rmse_m']:.3f}m "
              f"({agg['rmse_m'] * 100.0:.1f} cm)")
        print(f"  |depth - radar.y| med : {agg['median_depth_mismatch_m']:.3f}m")
    print(f"  coverage              : {agg['cells_filled']}/9 cells, "
          f"{agg['bands_filled']}/3 depth bands")
    print(f"  radar track ids seen  : {agg['unique_radar_ids']}")


def verdict_line(label: str, agg: Dict[str, Any]) -> str:
    median = agg["median_px"]
    if median <= 25 and agg["rmse_m"] <= 0.12:
        tag = "[GOOD]"
    elif median <= 45:
        tag = "[OK-ish]"
    else:
        tag = "[BAD]"
    return f"  {tag} {label}: median {median:.1f}px, rmse {agg['rmse_m']*100.0:.1f}cm"


def print_param_diff(before: Dict[str, float], after: Dict[str, float]) -> None:
    print("Parameter diff (before ⇒ after):")
    for key in ("tx", "ty", "tz", "yaw_deg", "pitch_deg", "roll_deg", "fx", "fy", "cx", "cy"):
        b = before.get(key, 0.0); a = after.get(key, 0.0)
        unit = "°" if key.endswith("_deg") else ("m" if key in ("tx", "ty", "tz") else "px")
        if abs(a - b) < 1e-6:
            continue
        print(f"  {key:<9} {b:+9.3f}{unit}  ⇒  {a:+9.3f}{unit}   (Δ {a - b:+8.3f}{unit})")


def diagnose_reasons(before_agg: Dict[str, Any], after_agg: Dict[str, Any], outliers: int) -> List[str]:
    msgs: List[str] = ["Interpretation:"]
    b_med = before_agg["median_px"]
    a_med = after_agg["median_px"]
    b_rmse = before_agg["rmse_m"]
    a_rmse = after_agg["rmse_m"]

    if before_agg["count"] < 6:
        msgs.append("  - Too few samples (< 6) to draw strong conclusions. Collect more positions.")
    if before_agg["cells_filled"] < 5 or before_agg["bands_filled"] < 2:
        msgs.append(
            "  - Poor spatial coverage. Many calibration errors come from flooding one area. "
            "Walk the full left/center/right × near/mid/far grid."
        )
    if a_med < b_med * 0.6 and b_med > 25:
        msgs.append(
            f"  - Re-solving drops median error {b_med:.1f}→{a_med:.1f}px. "
            "The saved calibration is the main problem, not the samples. "
            "Run Auto Calibrate to completion (or hit Solve Calibration on a good sample set)."
        )
    elif a_med >= 25 and b_med >= 25:
        msgs.append(
            f"  - Even a fresh solve yields median {a_med:.1f}px. "
            "The samples themselves are inconsistent. Likely causes, in order: "
            "(1) radar sensorPosition tilt in the cfg does not match the physical mount; "
            "(2) samples were collected while the person was moving faster than the "
            "radar-camera timing can handle; "
            "(3) depth at torso was occluded (RealSense holes) causing bad 3-D samples."
        )
    if after_agg["rmse_m"] > 0.20 and after_agg.get("residual_samples", 0) >= 4:
        msgs.append(
            f"  - 3-D rmse {a_rmse*100:.1f}cm is large. This is independent of the image pixel size, "
            "so it usually means the radar and camera disagree in 3-D space — almost always "
            "a cfg/mount problem (sensorPosition wrong) rather than an intrinsics problem."
        )
    if before_agg.get("median_depth_mismatch_m", 0.0) > 0.5:
        msgs.append(
            f"  - |depth - radar.y| median {before_agg['median_depth_mismatch_m']:.2f}m is high. "
            "RealSense depth and radar forward-range do not agree: check the person was "
            "standing (not reaching) and that the radar tilt is correct."
        )
    if outliers > 0:
        msgs.append(
            f"  - {outliers} outlier sample(s) dominate the median. "
            "Consider re-running auto calibrate in a cleaner single-person scene."
        )

    if len(msgs) == 1:
        msgs.append("  - Calibration looks good. If fusion still drifts, check handoff logs or radar tracker config.")
    return msgs


# ------------------------------------------------------------- commands --


def cmd_eval(args: argparse.Namespace) -> int:
    samples_path = _resolve_samples_path(args.samples)
    calib_path = Path(args.calibration)
    samples = load_calibration_samples(samples_path)
    projection = _load_projection(calib_path)

    diags = analyze_samples(samples, projection, args.frame_w, args.frame_h)
    agg = aggregate(diags)

    print(f"Calibration:  {calib_path}")
    print(f"Samples:      {samples_path}  (n={len(samples)})")
    print(f"Frame size:   {args.frame_w}x{args.frame_h}")
    print()
    print_aggregate("Accuracy (current calibration):", agg)
    print()
    if args.per_sample:
        print("Per-sample breakdown:")
        print_sample_table(diags)
        print()
    outliers = flag_outliers(diags)
    if outliers:
        print(f"Outlier indexes (error far above median): {outliers}")
    print(verdict_line("current", agg))

    if args.write_json:
        out = samples_path.with_name("calibration_eval.json")
        out.write_text(json.dumps({
            "calibration": str(calib_path),
            "samples_path": str(samples_path),
            "aggregate": agg,
            "per_sample": [d.__dict__ for d in diags],
            "outlier_indexes": outliers,
        }, indent=2, default=_jsonable), encoding="utf-8")
        print(f"Wrote: {out}")
    return 0


def cmd_resolve(args: argparse.Namespace) -> int:
    samples_path = _resolve_samples_path(args.samples)
    calib_path = Path(args.calibration)
    samples = load_calibration_samples(samples_path)
    projection_before = _load_projection(calib_path)
    before_params = dict(projection_before.params)

    before_agg = aggregate(analyze_samples(samples, projection_before, args.frame_w, args.frame_h))

    projection_solve = _copy_projection(projection_before)
    result = solve_calibration_auto(samples, projection_solve)

    print(f"Calibration:  {calib_path}")
    print(f"Samples:      {samples_path}  (n={len(samples)})")
    print()
    print_aggregate("Before (current calibration):", before_agg)
    print()
    print(f"Solver: type={result.solve_type}, ok={result.ok}")
    print(f"  message: {result.message}")

    if not result.params:
        print("Solver produced no parameters; nothing to compare.")
        return 1

    projection_after = _copy_projection(projection_before)
    projection_after.update(**{k: float(v) for k, v in result.params.items() if k in (
        "tx", "ty", "tz", "yaw_deg", "pitch_deg", "roll_deg", "fx", "fy", "cx", "cy",
    )})
    after_agg = aggregate(analyze_samples(samples, projection_after, args.frame_w, args.frame_h))
    print()
    print_aggregate("After (re-solved from same samples):", after_agg)
    print()
    print_param_diff(before_params, dict(projection_after.params))
    print()
    print(verdict_line("before", before_agg))
    print(verdict_line("after ", after_agg))

    if args.save_to:
        projection_after.save(Path(args.save_to))
        print(f"Wrote candidate calibration: {args.save_to}")
    return 0 if result.ok else 2


def cmd_report(args: argparse.Namespace) -> int:
    samples_path = _resolve_samples_path(args.samples)
    calib_path = Path(args.calibration)
    cfg_path = Path(args.cfg)

    samples = load_calibration_samples(samples_path)
    projection_before = _load_projection(calib_path)
    before_params = dict(projection_before.params)

    before_diags = analyze_samples(samples, projection_before, args.frame_w, args.frame_h)
    before_agg = aggregate(before_diags)
    outliers = flag_outliers(before_diags)

    projection_solve = _copy_projection(projection_before)
    result = solve_calibration_auto(samples, projection_solve)
    projection_after = _copy_projection(projection_before)
    if result.params:
        projection_after.update(**{k: float(v) for k, v in result.params.items() if k in (
            "tx", "ty", "tz", "yaw_deg", "pitch_deg", "roll_deg", "fx", "fy", "cx", "cy",
        )})
    after_diags = analyze_samples(samples, projection_after, args.frame_w, args.frame_h)
    after_agg = aggregate(after_diags)

    print("=" * 72)
    print(" Radar-Camera Calibration Report")
    print("=" * 72)
    print(f"Calibration:  {calib_path}")
    print(f"Samples:      {samples_path}  (n={len(samples)})")
    print(f"Cfg:          {cfg_path}")
    print(f"Frame size:   {args.frame_w}x{args.frame_h}")
    print()
    print_aggregate("BEFORE - using current calibration:", before_agg)
    print()
    print(f"Re-solve: type={result.solve_type}, ok={result.ok}")
    print(f"  message: {result.message}")
    print()
    print_aggregate("AFTER - same samples re-solved:", after_agg)
    print()
    print_param_diff(before_params, dict(projection_after.params))
    print()
    print(verdict_line("before", before_agg))
    print(verdict_line("after ", after_agg))
    print()
    if outliers:
        print(f"Outlier samples (likely bad single-frame captures): {outliers}")
        print()
    print_sample_table(before_diags)
    print()
    for line in physical_sanity(dict(projection_after.params), args.frame_w, args.frame_h):
        print(line)
    print()
    for line in cfg_sanity(cfg_path):
        print(line)
    print()
    room_depth = getattr(args, "room_depth", None)
    _rw = getattr(args, "room_half_width", None)
    room_half_width = (_rw / 2.0) if _rw is not None else None
    if room_depth is not None or room_half_width is not None:
        for line in room_sanity(cfg_path, room_depth, room_half_width):
            print(line)
        print()
    for line in diagnose_reasons(before_agg, after_agg, len(outliers)):
        print(line)

    if args.write_json:
        out = samples_path.with_name("calibration_report.json")
        payload = {
            "calibration": str(calib_path),
            "samples_path": str(samples_path),
            "cfg": str(cfg_path),
            "frame_wh": [args.frame_w, args.frame_h],
            "before": {"params": before_params, "aggregate": before_agg,
                       "per_sample": [d.__dict__ for d in before_diags]},
            "after": {"params": dict(projection_after.params), "aggregate": after_agg,
                      "per_sample": [d.__dict__ for d in after_diags],
                      "solver": {"ok": result.ok, "type": result.solve_type,
                                 "message": result.message}},
            "outliers": outliers,
        }
        out.write_text(json.dumps(payload, indent=2, default=_jsonable), encoding="utf-8")
        print(f"\nWrote: {out}")
    return 0


# ---------------------------------------------------------------- utils --


def _copy_projection(src: CameraProjection) -> CameraProjection:
    copy_p = CameraProjection(K=src.K.copy())
    copy_p.update(**src.params)
    return copy_p


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    raise TypeError(f"not JSON serializable: {type(obj)}")


# -------------------------------------------------------------- argparse --


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Radar-camera calibration accuracy & diagnostics."
    )
    subs = parser.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--samples", default="latest",
                        help="Path to auto_calibration_samples.json, a diagnostics_runs/* dir, or 'latest'.")
        sp.add_argument("--calibration", default="radar_camera_calibration.json",
                        help="Calibration JSON file to evaluate.")
        sp.add_argument("--frame-w", type=int, default=FRAME_W_DEFAULT, dest="frame_w")
        sp.add_argument("--frame-h", type=int, default=FRAME_H_DEFAULT, dest="frame_h")
        sp.add_argument("--write-json", action="store_true",
                        help="Also write a JSON next to the samples file.")

    ev = subs.add_parser("eval", help="Evaluate a calibration's pixel/3-D error against samples.")
    add_common(ev)
    ev.add_argument("--per-sample", action="store_true", help="Print per-sample error table.")
    ev.set_defaults(func=cmd_eval)

    rs = subs.add_parser("resolve", help="Re-solve from samples, show before vs after.")
    add_common(rs)
    rs.add_argument("--save-to", default=None,
                    help="If set, save re-solved calibration to this path (does not overwrite current).")
    rs.set_defaults(func=cmd_resolve)

    rp = subs.add_parser("report", help="Full accuracy + physical/cfg diagnostic.")
    add_common(rp)
    rp.add_argument("--cfg", default="iwr6843_people_tracking_20fps.cfg",
                    help="Radar cfg file to sanity-check against.")
    rp.add_argument("--room-depth", type=float, default=None, dest="room_depth",
                    help="Room depth in metres (sensor wall to far wall). Enables room_sanity check.")
    rp.add_argument("--room-width", type=float, default=None, dest="room_half_width",
                    metavar="ROOM_WIDTH",
                    help="Full room width in metres. Half-width used for lateral boundary check.")
    rp.set_defaults(func=cmd_report)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
