# Radar Tracking Debug Workflow

Use this when the room view is streaming but person tracking feels unstable.
The goal is to capture what the radar parser actually saw frame-by-frame, then
measure stability instead of guessing from the 3D display.

## GUI Workflow

1. Restart the viewer so the debug recorder is active.

   ```bash
   python3 radar_viewer.py --backend qt --config-port /dev/ttyACM0 --data-port /dev/ttyACM1 --config-baud 115200 --data-baud 921600
   ```

2. Connect live mode and run a short controlled test.

3. Watch the new `Debug` tab.

   It shows track coverage, presence/track agreement, no-track streaks, ID
   switches, point counts, and the latest parsed frame payload.

4. After the run, copy the `Frame debug file` path from the Debug tab.

5. Analyze it from the terminal.

   ```bash
   python3 radar_debug_tools.py analyze diagnostics_runs/<run_id>/frames.jsonl --write-json
   ```

## Headless Capture Workflow

Use this when you want a clean capture without the GUI render loop.

```bash
python3 radar_debug_tools.py live-capture \
  --seconds 20 \
  --cfg iwr6843_people_tracking.cfg \
  --config-port /dev/ttyACM0 \
  --data-port /dev/ttyACM1 \
  --config-baud 115200 \
  --data-baud 921600
```

This writes:

- `diagnostics_runs/<run_id>/frames.jsonl`
- `diagnostics_runs/<run_id>/tracking_analysis.json`
- `diagnostics_runs/<run_id>/session_report.json`
- `diagnostics_runs/<run_id>/cli.log`

## What To Look For

- `Track coverage` should be high when a person is clearly inside the detection volume.
- `Presence/track agreement` should be high if presence is detecting a person and the tracker is successfully locking.
- `Presence but no track frames` means the radar sees occupancy but tracker association is failing.
- `Points but no track frames` means point-cloud detection exists but the people tracker is not forming a target.
- `Single-person ID switches` should stay near zero during a one-person test.
- `Max no-track streak` tells us how long tracking disappears at once.

## Controlled Test Pattern

For a first stability pass, use one person and a quiet scene.

1. Stand still at center, around 2 m from the radar, for 10 seconds.
2. Step left and right slowly inside the boundary box for 10 seconds.
3. Walk toward and away from the radar for 10 seconds.
4. Repeat with the person near the edge of the configured boundary box.

Those four captures tell us whether the problem is config boundary geometry,
point-cloud quality, tracker thresholds, or visualization/rendering.
