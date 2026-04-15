#!/usr/bin/env python3
"""
UART diagnostic and staged config probe for the TI mmWave radar.

The tool can either send a cfg file verbatim or run staged startup probes that
isolate whether RF config, detection config, geometry config, tracker config,
or presence config is causing `sensorStart` to fail.
"""

from __future__ import annotations

import argparse
import string
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import serial

from radar import _CLI_POST_OPEN_SETTLE_S, open_serial_with_retries, wake_cli

MAGIC = b"\x02\x01\x04\x03\x06\x05\x08\x07"

RF_COMMANDS = {
    "dfeDataOutputMode",
    "channelCfg",
    "adcCfg",
    "adcbufCfg",
    "profileCfg",
    "chirpCfg",
    "frameCfg",
}
DETECTION_COMMANDS = {
    "dynamicRACfarCfg",
    "staticRACfarCfg",
    "dynamicRangeAngleCfg",
    "dynamic2DAngleCfg",
    "staticRangeAngleCfg",
    "fineMotionCfg",
    "fovCfg",
}
GEOMETRY_COMMANDS = {
    "antGeometry0",
    "antGeometry1",
    "antPhaseRot",
    "compRangeBiasAndRxChanPhase",
}
TRACKER_COMMANDS = {
    "staticBoundaryBox",
    "boundaryBox",
    "sensorPosition",
    "gatingParam",
    "stateParam",
    "allocationParam",
    "maxAcceleration",
    "trackingCfg",
}
PRESENCE_COMMANDS = {"presenceBoundaryBox"}


@dataclass
class RunResult:
    ok: bool
    failed_command: Optional[str]
    config_text: str
    data_bytes: bytes


def load_cfg_lines(cfg_path: str) -> List[str]:
    lines = Path(cfg_path).read_text(encoding="utf-8", errors="replace").splitlines()
    out: List[str] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        out.append(line)
    return out


def printable_preview(data: bytes, limit: int = 200) -> str:
    chars = []
    for b in data[:limit]:
        ch = chr(b)
        if ch in string.printable and ch not in "\x0b\x0c":
            chars.append(ch)
        else:
            chars.append(".")
    return "".join(chars)


def drain_serial(ser: serial.Serial, duration_s: float) -> bytes:
    end = time.time() + duration_s
    buf = bytearray()
    while time.time() < end:
        chunk = ser.read(ser.in_waiting or 1)
        if not chunk:
            time.sleep(0.01)
            continue
        buf.extend(chunk)
    return bytes(buf)


def read_command_response(
    cfg_ser: serial.Serial,
    data_ser: Optional[serial.Serial],
    cmd: str,
    timeout_s: float,
) -> Tuple[bytes, bytes]:
    cfg_buf = bytearray()
    data_buf = bytearray()
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        cfg_chunk = cfg_ser.read(cfg_ser.in_waiting or 1)
        if cfg_chunk:
            cfg_buf.extend(cfg_chunk)
            if b"Done" in cfg_buf or b"Ignored" in cfg_buf or b"Error" in cfg_buf:
                break
        if data_ser is not None:
            data_chunk = data_ser.read(data_ser.in_waiting or 1)
            if data_chunk:
                data_buf.extend(data_chunk)
        if not cfg_chunk:
            time.sleep(0.01)

    # Give the firmware a short follow-up window for late prints after sensorStart.
    if cmd == "sensorStart":
        cfg_buf.extend(drain_serial(cfg_ser, 1.0))
        if data_ser is not None:
            data_buf.extend(drain_serial(data_ser, 1.0))
    return bytes(cfg_buf), bytes(data_buf)


def run_cfg_once(
    commands: Sequence[str],
    config_port: str,
    data_port: Optional[str],
    config_baud: int,
    data_baud: int,
) -> RunResult:
    cfg_ser = open_serial_with_retries(config_port, config_baud, timeout=0.1)
    failed_command: Optional[str] = None
    config_logs = bytearray()
    data_logs = bytearray()
    ok = True
    data_ser: Optional[serial.Serial] = None

    try:
        time.sleep(_CLI_POST_OPEN_SETTLE_S)
        cfg_ser.reset_input_buffer()
        if data_port:
            data_ser = open_serial_with_retries(data_port, data_baud, timeout=0.05)
            time.sleep(0.2)
            data_ser.reset_input_buffer()
        wake_cli(cfg_ser)

        for line in commands:
            print(f"\n>> {line}")
            cfg_ser.write((line + "\n").encode("ascii"))
            cfg_text, data_text = read_command_response(
                cfg_ser,
                data_ser,
                line,
                3.5 if line == "sensorStart" else 1.5,
            )
            config_logs.extend(cfg_text)
            data_logs.extend(data_text)
            text = cfg_text.decode("ascii", errors="replace").strip()
            print(text if text else "(timeout/no response)")

            if b"Error" in cfg_text or (line == "sensorStart" and not cfg_text):
                ok = False
                failed_command = line
                break
            time.sleep(0.05)
        if data_ser is not None and ok:
            data_logs.extend(drain_serial(data_ser, 2.0))
    finally:
        # Drain any last bytes before closing.
        if data_ser is not None:
            try:
                data_logs.extend(drain_serial(data_ser, 0.2))
            except Exception:
                pass
            data_ser.close()
        config_logs.extend(drain_serial(cfg_ser, 0.2))
        cfg_ser.close()

    return RunResult(
        ok=ok,
        failed_command=failed_command,
        config_text=config_logs.decode("ascii", errors="replace"),
        data_bytes=bytes(data_logs),
    )


def summarize_data(data: bytes) -> None:
    print("\nData summary")
    print(f"rx_bytes: {len(data)}")
    print(f"magic_hits: {data.count(MAGIC)}")
    if data:
        print(f"ascii_preview: {printable_preview(data)}")
        print(f"first_hex: {data[:128].hex()}")
    else:
        print("ascii_preview: (none)")
        print("first_hex: (none)")


def build_stage_commands(lines: Sequence[str]) -> List[Tuple[str, List[str]]]:
    groups = {
        "rf_core": [],
        "detection": [],
        "geometry": [],
        "tracker": [],
        "presence": [],
    }

    for line in lines:
        cmd = line.split()[0]
        if cmd in ("sensorStop", "flushCfg", "sensorStart"):
            continue
        if cmd in RF_COMMANDS:
            groups["rf_core"].append(line)
        elif cmd in DETECTION_COMMANDS:
            groups["detection"].append(line)
        elif cmd in GEOMETRY_COMMANDS:
            groups["geometry"].append(line)
        elif cmd in TRACKER_COMMANDS:
            groups["tracker"].append(line)
        elif cmd in PRESENCE_COMMANDS:
            groups["presence"].append(line)
        else:
            groups["tracker"].append(line)

    cumulative: List[str] = []
    out: List[Tuple[str, List[str]]] = []
    for stage_name in ("rf_core", "detection", "geometry", "tracker", "presence"):
        stage_lines = groups[stage_name]
        if not stage_lines:
            continue
        cumulative.extend(stage_lines)
        commands = ["sensorStop", "flushCfg", *cumulative, "sensorStart"]
        out.append((stage_name, commands))
    return out


def run_staged_probe(
    lines: Sequence[str],
    config_port: str,
    data_port: Optional[str],
    config_baud: int,
    data_baud: int,
) -> None:
    print("\nRunning staged startup probe...")
    for stage_name, commands in build_stage_commands(lines):
        print(f"\n===== Stage: {stage_name} =====")
        result = run_cfg_once(commands, config_port, data_port, config_baud, data_baud)
        print(f"\nStage result: {'PASS' if result.ok else 'FAIL'}")
        if result.failed_command:
            print(f"failed_command: {result.failed_command}")
        if result.config_text.strip():
            print("\nConfig log tail")
            tail = result.config_text[-1200:].strip()
            print(tail if tail else "(none)")
        summarize_data(result.data_bytes)
        if not result.ok:
            print(
                "\nProbe conclusion: the failure first appears in this stage, "
                "so the offending config is very likely in this block."
            )
            return
    print("\nProbe conclusion: every stage passed. If frames are still missing, the issue is likely UART parsing or port selection.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-port", default="/dev/ttyACM0")
    parser.add_argument("--data-port", default="/dev/ttyACM1")
    parser.add_argument("--skip-data-open", action="store_true")
    parser.add_argument("--cfg", default="iwr6843_people_tracking.cfg")
    parser.add_argument("--config-baud", type=int, default=115200)
    parser.add_argument("--data-baud", type=int, default=921600)
    parser.add_argument("--probe-stages", action="store_true")
    args = parser.parse_args()

    print(f"Using config port {args.config_port}")
    print(f"Using data port   {args.data_port if not args.skip_data_open else '(skipped)'}")
    print(f"Using cfg         {args.cfg}")

    lines = load_cfg_lines(args.cfg)
    data_port = None if args.skip_data_open else args.data_port
    if args.probe_stages:
        run_staged_probe(
            lines,
            args.config_port,
            data_port,
            args.config_baud,
            args.data_baud,
        )
        return

    result = run_cfg_once(
        lines,
        args.config_port,
        data_port,
        args.config_baud,
        args.data_baud,
    )
    print(f"\nConfig result: {'ok' if result.ok else 'FAILED'}")
    if result.failed_command:
        print(f"failed_command: {result.failed_command}")
    print("\nConfig follow-up")
    followup = result.config_text.strip()
    print(followup[-1200:] if followup else "(none)")
    summarize_data(result.data_bytes)


if __name__ == "__main__":
    main()
