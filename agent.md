# Agent Memory

This file is working memory for future coding agents in the FALCON repo. Keep it
short, factual, and update it when hardware assumptions, configs, or recovery
steps change.

## Project Goal

FALCON is a camera plus TI mmWave radar fusion pipeline. The current MVP is
single-person tracking: camera pose/YOLO owns identity while visible; radar
hands off position when the person is obscured; camera recaptures identity when
the person is visible again.

Target performance is 20+ FPS on an NVIDIA Orin Nano.

## Hardware

- Radar: TI IWR6843ISK with mmWaveICBOOST carrier.
- Current IWR6843ISK board orientation: antenna patch/sensor area on the
  right side of the board, USB/connector edge on the bottom.
- Standalone IWR6843ISK currently enumerates as Silicon Labs CP2105:
  - `/dev/ttyUSB0`: Enhanced Com Port, works as config/CLI UART.
  - `/dev/ttyUSB1`: Standard Com Port, works as data/logging UART.
- Radar mount: center of radar board is about 74 inches / 1.88 m from floor.
- Camera mount: camera is about 76.5 inches / 1.944 m from floor (2.5 in above radar).
- Current radar tilt: 8 degrees elevation tilt toward the ground (corrected April 2026 —
  sensor was previously mounted sideways; orientation and tilt were both fixed).
- Current tracker config should use:

```cfg
sensorPosition 1.88 0 8
```

## Mount Geometry (camera relative to radar)

Measured physical offsets used to seed and constrain the calibration solver:

- tx = 0 m (no lateral offset)
- ty = +0.064 m (camera is 2.5 in above radar; camera-Y is down, so positive ty)
- tz = +0.038 m (camera is 1.5 in behind radar / closer to wall; camera-Z = radar-Y depth, camera further from scene → +tz)

These are stored in `mount_geometry.json` and loaded by the Radar Calibrator UI.

## Important Files

- `falcon_gui.py`: Tkinter GUI, camera pipeline, radar startup, radar-camera fusion display, calibration UI.
- `radar.py`: IWR6843 serial driver, parser, diagnostics, projection model.
- `radar_camera_fusion.py`: camera/radar handoff state machine and calibration solver.
- `radar_debug_tools.py`: CLI capture/analyze tool for headless radar debugging.
- `iwr6843_people_tracking_20fps.cfg`: preferred radar config for GUI/fusion.
- `iwr6843_people_tracking.cfg`: conservative fallback.
- `iwr6843_config.cfg`: OOB-style point-cloud config; not the final config for people handoff.

## Current Radar Config Assumptions

Preferred config is `iwr6843_people_tracking_20fps.cfg`.

Important tracker lines:

```cfg
sensorPosition 1.88 0 8
gatingParam 3 2 2 2 4
stateParam 1 3 12 500 5 6000
allocationParam 15 100 0.05 5 1.0 20
maxAcceleration 0.1 0.1 0.1
fovCfg -1 70.0 20.0
boundaryBox -4 4 0 8 0 3
staticBoundaryBox -3 3 0.5 7.5 0 3
presenceBoundaryBox -3 3 0.5 7.5 0 3
```

Reasoning:

- Earlier config used `sensorPosition 2 0 15`, which did not match the physical mount.
- April 2026: sensor was physically sideways; fixing the mount orientation and remeasuring
  gave `sensorPosition 1.88 0 8` (74 in height, 8° down-tilt, azimuth 0).
- **April 20 2026 diagnostic**: `allocationParam snrThre=60` was filtering 99.9% of
  body returns. Median body-return SNR measured ~7.6 dB; only 5/5408 points (0.1%)
  met SNR≥60. Tracks were never allocated despite presence=1 and 13–51 points/frame.
  Fix: `snrThre` lowered 60→15, `snrThreObscured` lowered 200→50. Validated at
  98.5% track coverage with person, 0% false-positive tracks in empty room.
- `fovCfg ±15°` was tried for single-floor multipath reduction but cut off upper-body
  returns at close range (<4m). At 3m depth, torso centre is only 0.8° inside the ±15°
  cone. Reverted to ±20°. Do not lower elevation FOV below 20° for this mount.
- `boundaryBox y_min` raised from 0 to 0.5 m to exclude near-field wall clutter
  (mounting surface returns cluster at world-frame Y=0–0.4 m).

## Radar Recovery Rules

The GUI does not flash or erase radar firmware. It only opens UARTs and sends CLI config lines.

If radar stops responding:

1. Close all FALCON/radar processes.

```bash
pkill -f falcon_gui.py
pkill -f radar_debug_tools.py
pkill -f radar_viewer.py
```

2. Fully power-cycle radar: unplug USB/power, wait 10 seconds, reconnect.
3. Confirm serial ports:

```bash
python3 -m serial.tools.list_ports -v
```

4. Run CLI-only radar capture with GUI closed.

```bash
python3 radar_debug_tools.py live-capture \
  --seconds 25 \
  --cfg iwr6843_people_tracking_20fps.cfg \
  --config-port /dev/ttyUSB0 \
  --data-port /dev/ttyUSB1 \
  --config-baud 115200 \
  --data-baud 921600
```

For mmWaveICBOOST/XDS110, the ports may instead be `/dev/ttyACM0` and
`/dev/ttyACM1`.

If `sensorStop: timeout waiting for CLI response` appears, likely causes are:

- wrong CLI UART,
- a process still owns the port,
- board left in flashing/SOP boot mode,
- firmware app did not boot,
- radar app wedged and needs full power-cycle.

On Windows/TeraTerm, COM3 may be the binary data port. Use the XDS110
Application/User UART or Enhanced COM Port at 115200 8-N-1, no flow control.
The binary data UART will not show readable CLI text.

## GUI Safe Mode

Use this to run camera GUI without opening radar UARTs:

```bash
FALCON_DISABLE_RADAR=1 python3 falcon_gui.py
```

This is useful while recovering firmware/ports or proving the camera side is not
touching the radar.

## Calibration Workflow

Three calibration paths exist in the Radar Calibrator window, from most to
least accurate:

### Path 1 — Guided Body Calibration (recommended)

1. Open Radar Calibrator → **Guided Body Calibration** panel.
2. Click **Start**. A blue instruction appears: "Step 1/6: CENTER, ~2 m …"
3. Stand at the described position, stand still.
4. **Click your torso centre** in the video feed.
5. Click **Capture Position**. Status shows if the capture succeeded or why
   it failed (no radar track, multiple people, etc.).
6. Follow the next instruction. Six positions total (3 lateral × 2 depths).
7. **Solve** becomes active after 4 captures (full 6 is better).
8. Click **Solve** — if result is ok, Advanced Sliders auto-update and
   `radar_camera_calibration.json` is saved.
9. Spot-check: stand at 2 new positions and confirm the radar dot is within
   ~20 px of torso centre.

Tips:
- Ensure exactly one person is in the room — multiple radar tracks cause a
  rejection.
- Stand still for 1–2 seconds before clicking so the radar track is stable.
- The solver prefers depth-backprojected 3D correspondences; if the depth
  camera is off, it falls back to 2D reprojection (less accurate).

### Path 2 — Auto Calibrate (walk-through, hands-free)

1. Click **Auto Calibrate**.
2. Walk slowly and smoothly through 6–8 varied spots (left/center/right,
   near/mid/far). Do not freeze; this radar loses static returns.
3. Auto-solves when enough coverage is reached; saves automatically.
4. If coverage is complete but not solved, press **Solve Calibration**.

### Path 3 — Manual single-sample fallback

1. Wait for `Radar Tracks: 1` in the overlay.
2. Click torso centre in video.
3. Press **Capture Sample**.
4. Repeat for 7–9 varied positions, then **Solve Calibration**.

Do not use raw point-cloud-only samples for final calibration.

## Radar Debug Interpretation

- `Tracked: 1` in the GUI overlay is camera tracking, not radar tracking.
- Use `Radar Tracks: N` for radar person-track count.
- `Radar Points: >0` and `Radar Tracks: 0` means the radar sees returns but the TI tracker has not allocated a person target.
- `presence=1` with `tracks=0` usually points to tracker allocation/config/FOV issues, not UART failure.
- `frames=0` plus `sensorStop timeout` means CLI/config UART is not responding.
- `Device or resource busy` means another process owns the serial port.

## Known Parser Detail

The people-counting compressed point cloud encodes azimuth/elevation units as
radians. Do not convert those angles as if they are degrees.

## Testing Commands

Run focused tests:

```bash
python3 tests/test_auto_calibration.py
python3 tests/test_radar_diagnostics.py
python3 tests/test_radar_camera_fusion.py
```

Run all tests:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

Avoid `python3 -m unittest tests.test_...` in this environment because an
installed third-party `tests` package may be imported instead.

## User Preferences

- The user wants complete, practical integrations, not partial plans.
- Give step-by-step test instructions and specify exactly what output to report.
- Ask clarifying questions only when the answer cannot be inferred or tested.
