#!/usr/bin/env python3
"""Range bias calibration for IWR6843 mmWave radar.

Measures the systematic range offset of the radar by comparing known
corner-reflector distances to reported distances.  On completion it
patches compRangeBiasAndRxChanPhase in both cfg files.

Usage:
    python3 range_bias_calibration.py \\
        --known-distances 1.0 2.0 3.0 5.0 \\
        --config-port /dev/ttyUSB0 \\
        --data-port /dev/ttyUSB1 \\
        [--frames 40] [--dry-run]

Procedure:
  1. Ensure the radar is connected and powered; clear the FOV of people.
  2. For each distance the tool prompts for, place a corner reflector at that
     SLANT distance from the radar face along the boresight direction.
     (Tape from the radar board face to the centre of the reflector.)
  3. Hold still for ~2 s once the reflector is in place, then press Enter.
  4. After all distances the tool patches both cfg files automatically.

Typical range bias for IWR6843ISK: 0.00 – 0.10 m.
"""

from __future__ import annotations

import argparse
import math
import tempfile
import textwrap
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from radar import IWR6843Driver


# Cfg files (relative to this script) to patch after calibration.
_CFG_FILES = [
    "iwr6843_people_tracking.cfg",
    "iwr6843_people_tracking_20fps.cfg",
]

# Source cfg to derive the calibration-only cfg from.
_SOURCE_CFG = "iwr6843_people_tracking_20fps.cfg"

# Velocity below which a point is treated as stationary (m/s).
_VEL_THRESH = 0.5

# Half-window around known distance for point association (m).
_ASSOC_WINDOW = 0.8


# --------------------------------------------------------------------------- #
# Temporary calibration cfg construction                                       #
# --------------------------------------------------------------------------- #


def _build_cal_cfg(source_path: Path) -> str:
    """Return a modified cfg string with static point output enabled.

    Changes made vs. source cfg:
      - staticRACfarCfg: second-to-last field set to 1 (staticOutputEn=1)
      - boundaryBox / staticBoundaryBox / presenceBoundaryBox: widened to
        cover any reasonable corner reflector placement
      - compRangeBiasAndRxChanPhase: bias reset to 0 for absolute measurement
    """
    lines = source_path.read_text(encoding="utf-8").splitlines()
    out: List[str] = []
    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            out.append(line)
            continue
        parts = stripped.split()
        key = parts[0]
        if key == "staticRACfarCfg" and len(parts) >= 15:
            # Enable static output (parts[-2] = staticOutputEn).
            new_parts = parts[:]
            new_parts[-2] = "1"
            out.append(" ".join(new_parts))
        elif key in ("boundaryBox", "staticBoundaryBox", "presenceBoundaryBox"):
            # Permissive box: x ±8 m, y 0.1–9.9 m, z −3–6 m (world frame).
            out.append(f"{key} -8 8 0.1 9.9 -3 6")
        elif key == "compRangeBiasAndRxChanPhase":
            # Reset bias to 0 so we measure absolute offset.
            out.append("compRangeBiasAndRxChanPhase 0 " + " ".join(parts[2:]))
        else:
            out.append(line)
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #
# Point collection                                                             #
# --------------------------------------------------------------------------- #


def _collect_at_distance(
    driver: IWR6843Driver,
    known_m: float,
    num_frames: int,
) -> Optional[float]:
    """Capture up to num_frames and return median slant range near known_m.

    Filters for near-stationary points (|velocity| < _VEL_THRESH) within
    ±_ASSOC_WINDOW of the known distance.  Returns None if nothing found.
    """
    collected: List[float] = []
    prev_count = driver.get_frame_count()
    consumed = 0
    deadline = time.time() + num_frames / 20.0 + 5.0  # 5 s safety margin

    print(f"  Capturing {num_frames} frames...", end="", flush=True)

    while consumed < num_frames and time.time() < deadline:
        time.sleep(0.04)
        cur = driver.get_frame_count()
        if cur <= prev_count:
            continue
        prev_count = cur
        consumed += 1
        frame = driver.get_latest_frame()
        if frame is None:
            continue
        for pt in frame.points:
            r = math.sqrt(pt.x**2 + pt.y**2 + pt.z**2)
            if abs(r - known_m) <= _ASSOC_WINDOW and abs(pt.velocity) <= _VEL_THRESH:
                collected.append(r)

    print(f" {len(collected)} static point samples from {consumed} frames")
    return float(np.median(collected)) if collected else None


# --------------------------------------------------------------------------- #
# Bias computation                                                             #
# --------------------------------------------------------------------------- #


def compute_bias(
    known: List[float],
    measured: List[float],
) -> Tuple[float, float, float]:
    """Fit measured = slope * known + intercept; return (mean_bias, slope, intercept).

    mean_bias = mean(measured − known) is the primary calibration value.
    slope and intercept are returned for diagnostic purposes.
    """
    arr_k = np.array(known, dtype=np.float64)
    arr_m = np.array(measured, dtype=np.float64)
    mean_bias = float(np.mean(arr_m - arr_k))

    if len(known) < 2:
        return mean_bias, 1.0, mean_bias

    A = np.column_stack([arr_k, np.ones_like(arr_k)])
    coeffs, _, _, _ = np.linalg.lstsq(A, arr_m, rcond=None)
    slope, intercept = float(coeffs[0]), float(coeffs[1])
    return mean_bias, slope, intercept


# --------------------------------------------------------------------------- #
# Cfg patching                                                                 #
# --------------------------------------------------------------------------- #


def patch_cfg_file(cfg_path: Path, bias_m: float, dry_run: bool) -> bool:
    """Update compRangeBiasAndRxChanPhase bias in cfg_path."""
    if not cfg_path.exists():
        print(f"  [skip] {cfg_path} not found")
        return False

    text = cfg_path.read_text(encoding="utf-8")
    patched_lines: List[str] = []
    found = False

    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("compRangeBiasAndRxChanPhase") and not stripped.startswith("%"):
            parts = stripped.split()
            if len(parts) >= 3:
                rest = " ".join(parts[2:])
                new_line = f"compRangeBiasAndRxChanPhase {bias_m:.5f} {rest}\n"
                patched_lines.append(new_line)
                found = True
                continue
        patched_lines.append(line)

    if not found:
        print(f"  [skip] compRangeBiasAndRxChanPhase not found in {cfg_path}")
        return False

    if dry_run:
        new_line = next(
            l.rstrip() for l in patched_lines if "compRangeBiasAndRxChanPhase" in l
        )
        print(f"  [dry-run] {cfg_path}  →  {new_line}")
        return True

    cfg_path.write_text("".join(patched_lines), encoding="utf-8")
    print(f"  Patched: {cfg_path}")
    return True


# --------------------------------------------------------------------------- #
# Main calibration flow                                                        #
# --------------------------------------------------------------------------- #


def run_calibration(args: argparse.Namespace) -> int:
    script_dir = Path(__file__).resolve().parent
    source_cfg = script_dir / _SOURCE_CFG

    if not source_cfg.exists():
        print(f"Error: source cfg not found: {source_cfg}")
        return 1

    known_distances: List[float] = list(args.known_distances)
    if not known_distances:
        print("Error: no known distances provided.")
        return 1

    # Build a temporary calibration cfg with static output enabled.
    cal_cfg_text = _build_cal_cfg(source_cfg)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".cfg",
        delete=False,
        encoding="utf-8",
        prefix="range_bias_cal_",
    ) as tmp:
        tmp.write(cal_cfg_text)
        tmp_path = Path(tmp.name)

    print("Range bias calibration — IWR6843ISK")
    print(f"Distances to measure: {known_distances} m")
    print(f"Temporary cfg: {tmp_path}")
    print()

    driver = IWR6843Driver(
        config_port=args.config_port,
        data_port=args.data_port,
        config_path=str(tmp_path),
        config_baud=args.config_baud,
        data_baud=args.data_baud,
    )

    pairs: List[Tuple[float, float]] = []
    try:
        print("Connecting to radar...")
        if not driver.start():
            state = driver.session_state()
            print(f"Failed to start: {state.health_verdict} — {state.health_reason}")
            return 2
        print("Connected. Waiting for first frame...")
        if not driver.wait_for_frame(timeout_s=6.0):
            print("Timeout: no radar frames received.")
            return 2
        print("Radar producing frames.\n")

        for d_known in known_distances:
            print(f"--- {d_known:.1f} m ---")
            input(
                f"  Place corner reflector at {d_known:.2f} m slant distance\n"
                f"  from radar face along boresight, then hold still.\n"
                f"  Press Enter when ready..."
            )
            time.sleep(1.5)  # let radar stabilise after user movement

            measured = _collect_at_distance(driver, d_known, args.frames)
            if measured is None:
                print(
                    f"  [WARN] No static detections near {d_known:.2f} m. "
                    "Skipping. (Check reflector is within boresight FOV.)"
                )
                continue
            delta = measured - d_known
            print(f"  Known: {d_known:.3f} m  Measured: {measured:.3f} m  Δ: {delta:+.4f} m")
            pairs.append((d_known, measured))
    finally:
        driver.stop()
        tmp_path.unlink(missing_ok=True)

    if not pairs:
        print("\nNo valid measurements. Calibration aborted.")
        return 1

    known_vals = [p[0] for p in pairs]
    measured_vals = [p[1] for p in pairs]
    mean_bias, slope, intercept = compute_bias(known_vals, measured_vals)

    print(f"\n{'=' * 60}")
    print(f"Results: {len(pairs)} valid measurement(s)")
    for k, m in pairs:
        print(f"  {k:.2f} m  →  {m:.3f} m  (Δ {m - k:+.4f} m)")
    print(f"\nLinear fit: measured = {slope:.5f} × known + {intercept:+.5f} m")
    print(f"Mean bias (Δ):  {mean_bias:+.5f} m")

    if abs(slope - 1.0) > 0.02:
        print(
            f"\n[WARN] Scale factor {slope:.4f} deviates from 1.0 by "
            f"{abs(slope - 1.0) * 100:.1f}%.  This is unusual — verify that "
            "the reflector was on boresight and not at an oblique angle."
        )

    print(f"\nPatching cfg files (bias = {mean_bias:+.5f} m):")
    any_patched = False
    for name in _CFG_FILES:
        if patch_cfg_file(script_dir / name, mean_bias, args.dry_run):
            any_patched = True

    if not any_patched:
        print("No cfg files patched.")
        return 1

    if not args.dry_run:
        print(
            f"\nDone. Restart radar with the updated cfg to apply the correction."
        )

    return 0


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Calibrate IWR6843 range bias using a corner reflector.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Typical range bias for IWR6843ISK is 0.00 – 0.10 m.
            Run after any hardware change that may affect propagation delay.
        """),
    )
    p.add_argument(
        "--known-distances",
        nargs="+",
        type=float,
        required=True,
        metavar="D",
        help="Slant distances in metres (e.g. 1.0 2.0 3.0 5.0).",
    )
    p.add_argument("--config-port", default="/dev/ttyUSB0",
                   help="CLI/config UART (default /dev/ttyUSB0)")
    p.add_argument("--data-port", default="/dev/ttyUSB1",
                   help="Data UART (default /dev/ttyUSB1)")
    p.add_argument("--config-baud", type=int, default=115200)
    p.add_argument("--data-baud", type=int, default=921600)
    p.add_argument(
        "--frames",
        type=int,
        default=40,
        help="Frames to capture per distance (default 40 ≈ 2 s @ 20 fps).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print bias without modifying cfg files.",
    )
    return p


def main() -> int:
    return run_calibration(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
