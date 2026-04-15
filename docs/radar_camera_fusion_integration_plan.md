# Radar-Camera Fusion Integration Plan

Date: April 15, 2026

## Goal

Integrate the IWR6843 people-tracking radar into the FALCON camera GUI so a
single person can remain visually tracked when the camera can no longer see
them. During camera occlusion, the GUI should show a YOLO-like box and a
synthetic pose driven by the radar estimate, then recapture the person with the
camera as soon as it is reliable again while preserving the same FALCON person
ID.

Targets:

- 20+ FPS camera GUI update rate on Orin Nano.
- 20+ FPS radar update path, or the nearest stable radar profile if RF/tracker
  quality regresses.
- Single-person MVP first.
- Radar runs continuously in the background to avoid handoff startup delay.
- Calibration is simple enough to repeat after remounting.
- Debug mode can always show camera detections, radar tracks, projected radar
  boxes, and handoff state together.
- Fusion events are logged for diagnosis.

## Current Codebase Findings

The best starting point is the existing `falcon_gui.py`, not a rewrite.

Reasons:

- It already owns camera selection, model loading, display, start/stop, and GUI
  refresh.
- It already supports RealSense color/depth fallback through `DualStreamCamera`.
- It already uses a persistent `TrackingManager` with Kalman prediction and
  recovered/occluded states.
- It already has initial radar startup, point overlay, and a manual projection
  calibration window.
- `radar.py` already normalizes live radar into `RadarFrame`, `RadarTrack`,
  `RadarPoint`, scene metadata, and camera projection helpers.

What needs to change:

- The current fusion path in `tracking.py` is mostly point-cloud based. For
  handoff and identity preservation, the MVP should use radar `RadarTrack`
  objects first, with raw points as debug overlay only.
- The fusion state machine should be explicit instead of being folded into
  occlusion and Kalman prediction logic.
- Synthetic pose should be an intentional output mode, not just a generic ghost
  prediction.
- Calibration should move from manual slider tuning to a repeatable guided
  workflow with saved samples and numeric reprojection error.

## Decisions

### 1. Keep `falcon_gui.py` For The MVP

We will integrate into the existing GUI first. If profiling later proves that
Tkinter/PIL display conversion is the bottleneck, we can build a leaner Qt or
OpenCV display shell after the fusion behavior is correct.

Decision:

- MVP entrypoint remains `falcon_gui.py`.
- Fusion logic moves into a new standalone module so a future GUI can reuse it.

### 2. Use RealSense Color As The Preferred Camera Path

For runtime, we prefer the RealSense RGB stream at 640x480 and 30 FPS. It gives
us color, stable intrinsics, and optional depth for calibration/debug.

Decision:

- Runtime tracking uses color frames.
- Depth is optional during runtime and should be polled sparingly.
- Calibration uses RealSense intrinsics automatically when available.
- If RealSense depth costs too much runtime CPU, depth stays enabled only for
  calibration and debug sessions.

### 3. Use YOLO Nano TensorRT By Default On Orin

For the target Orin Nano, the performance path should be:

- `yolo26n-pose.engine` when present.
- `yolo26n-pose.pt` on CUDA as fallback.
- MediaPipe only as a fallback/debug mode, not the primary fusion path.

Decision:

- Keep adaptive detector skipping.
- Keep Kalman propagation on skipped camera frames.
- Keep fusion O(1) for the single-person MVP.

### 4. Radar Runs Continuously

Radar should start with the pipeline and keep streaming even while the camera is
healthy. Handoff should consume the latest already-running radar track.

Decision:

- No "start radar after occlusion" behavior.
- Radar thread remains independent from camera thread.
- The camera loop reads a copy of the latest radar frame without blocking.

### 5. Radar Tracks Are Primary, Points Are Debug

The people-tracking firmware already produces radar tracks with track IDs,
positions, heights, and boxes. These are much better for handoff than raw point
cloud clusters.

Decision:

- Handoff association uses `RadarFrame.tracks`.
- Point cloud overlay remains available in debug.
- If no radar track exists but presence/points exist, we can show "radar
  presence only" but should not drive identity from raw points in the MVP.

### 6. Camera ID Is The Main Identity

For the MVP, FALCON's camera tracker ID remains the user-facing identity. Radar
track IDs are linked to the active FALCON track during occlusion.

Decision:

- Display label stays `Person #N`.
- During radar-only mode, label becomes `Person #N RADAR ONLY`.
- A linked `radar_track_id` is stored internally for diagnosis and association.

### 7. Handoff Uses A State Machine

Instead of one threshold, handoff should use a small state machine:

- `CAMERA_LOCKED`: camera detection is confident.
- `CAMERA_DEGRADED`: detection exists but keypoints/confidence are weak.
- `RADAR_ARMED`: camera has degraded/lost target but radar has a plausible
  linked track.
- `RADAR_ONLY`: bbox and synthetic pose are driven by radar projection.
- `REACQUIRING`: camera detection overlaps the projected radar estimate.
- `CAMERA_RELOCKED`: camera owns the track again, same ID retained.

Trigger policy:

- Enter `CAMERA_DEGRADED` when pose confidence/keypoint count drops or the
  tracker reports partial/heavy occlusion.
- Enter `RADAR_ONLY` when the camera misses the person for 2-3 camera frames
  and a radar track projects near the last known person position.
- Recapture when a new camera detection overlaps the radar-projected bbox or
  lands near the predicted center.
- Require either one very strong detection or two consecutive acceptable
  detections before fully releasing radar. This keeps recapture quick without
  bouncing between modes.

### 8. Synthetic Pose Reuses The Last Real Pose

During radar-only tracking, we will preserve the most recent reliable COCO-17
pose shape and transform it to the radar-projected person box.

Decision:

- Store last reliable `relative_skeleton` from camera detection.
- In radar-only mode, project radar track center and/or radar 3D box to image
  space.
- Translate and scale the stored skeleton into the projected radar box.
- Mark generated keypoints with synthetic confidence.
- Draw it differently from normal pose, but close enough that downstream code
  sees a continuous pose estimate.

### 9. Calibration Is Guided Sample Collection

Manual sliders are useful for debug but too fragile as the primary setup.

Preferred calibration workflow:

1. Use RealSense color intrinsics automatically.
2. If RealSense intrinsics are unavailable, seed a color-camera intrinsic guess
   from the live frame size before solving.
3. Ask the user to stand at several positions in the radar/camera overlap.
4. On each sample, collect:
   - current radar `RadarTrack` 3D center/box,
   - a clicked torso-center image point or the camera person bbox/keypoints,
   - optional RealSense depth at torso/feet,
   - timestamp and confidence.
5. Solve radar-to-camera extrinsics from the collected correspondences.
6. Save `radar_camera_calibration.json`.
7. Show reprojection error in pixels and a pass/fail hint.

Simple MVP calibration:

- Minimum 5 good samples.
- Recommended 7-9 samples:
  - center near,
  - center mid,
  - center far,
  - left mid,
  - right mid,
  - left far,
  - right far,
  - optional near-left and near-right.

Fallback:

- If RealSense depth is unavailable, use pixel-only optimization against the
  radar 3D points and clicked camera torso points.
- Prefer real radar-track samples for final calibration. Point-cloud samples
  are rough debug samples and must be explicitly enabled from Advanced.
- Reject solves with high reprojection error or extrinsics pushed to rotation
  limits instead of applying a misleading calibration.
- Keep the existing manual sliders as an expert override.

### 10. 20+ FPS Requires A Radar CFG Change

The current `iwr6843_people_tracking.cfg` uses:

```text
frameCfg 0 2 96 0 55.00 1 0
trackingCfg 1 2 800 30 46 96 55
```

A 55 ms frame period is about 18.2 FPS. To support 20+ FPS, we need a validated
20 FPS profile, starting with 50 ms:

```text
frameCfg 0 2 96 0 50.00 1 0
trackingCfg 1 2 800 30 46 96 50
```

Decision:

- Create a separate `iwr6843_people_tracking_20fps.cfg` rather than mutating
  the known-good cfg first.
- Validate radar tracking stability before lowering below 50 ms.
- If 50 ms is unstable, keep camera GUI 20+ FPS and document radar update rate
  as the limiting sensor rate.

Mount-specific tracker setting:

```text
sensorPosition 1.85 0 0
```

This matches the current mount: radar center at 73 in / 1.85 m, level, with no
tilt. Camera center is about 75 in / 1.91 m and is handled by camera calibration.

Tracker allocation tuning:

```text
allocationParam 60 200 0.1 5 1.5 2
gatingParam 3 2 2 2 12
maxAcceleration 1 1 1
```

This uses the firmware's people-counting allocation defaults. The earlier
`pointsThre=20` was too strict for captures averaging roughly 18 points/frame,
so presence could be active while no track was allocated.

## Proposed File Changes

### New: `radar_camera_fusion.py`

Owns fusion behavior independent of GUI:

- `FusionMode` enum.
- `FusionTrackState` dataclass.
- `RadarCameraFusionManager`.
- `SyntheticPoseGenerator`.
- `RadarCameraAssociation`.
- `FusionEventLogger`.

Inputs per camera frame:

- frame timestamp,
- camera detections/tracks,
- latest `RadarFrame`,
- calibration/projection,
- runtime config.

Outputs:

- updated visible bbox,
- synthetic or real keypoints,
- fusion mode,
- radar link metadata,
- debug overlay payload,
- JSONL event record.

### Update: `radar.py`

Small API additions:

- `get_tracks() -> List[RadarTrack]`
- `get_latest_frame()` already exists and should remain the main API.
- Include radar FPS, frame number, active radar track IDs in diagnostics.

### Update: `tracking.py`

Refactor current radar fusion:

- Remove point-first `_fuse_radar` from core tracker or gate it behind the new
  fusion manager.
- Add fields to `PersonTracker`:
  - `fusion_mode`,
  - `radar_track_id`,
  - `handoff_age_frames`,
  - `synthetic_pose`,
  - `last_real_keypoints`,
  - `last_real_bbox`.
- Keep Kalman prediction but allow radar projection to act as a measurement.

### Update: `falcon_gui.py`

GUI/runtime integration:

- Start radar continuously.
- Read latest radar frame once per camera loop.
- Pass camera tracks and radar frame into `RadarCameraFusionManager`.
- Draw radar-only bbox with a clear denoter.
- Add debug toggles:
  - show radar points,
  - show radar tracks,
  - show projected radar boxes,
  - always show both camera and radar.
- Add fusion status lines:
  - mode,
  - radar FPS,
  - linked radar ID,
  - handoff frames,
  - recapture confidence.

### Update: Calibration UI

Extend `RadarCalibrationWindow`:

- `Capture Sample`
- `Clear Samples`
- `Solve Calibration`
- `Save Calibration`
- live sample count,
- median reprojection error,
- pass/fail quality label.

Keep existing sliders as expert fine-tuning.

### New: `iwr6843_people_tracking_20fps.cfg`

Copy known-good people-tracking config and adjust frame period/tracking period
to 50 ms. Keep the original cfg as fallback.

### New Tests

Add `tests/test_radar_camera_fusion.py`:

- camera-locked remains camera-owned when detections are healthy,
- occlusion enters radar-only when radar track is plausible,
- synthetic pose preserves keypoint shape while moving to radar bbox,
- recapture keeps the same FALCON ID,
- bad radar projection does not steal identity,
- event logger writes handoff/recapture records,
- calibration solver rejects too few samples.

## Implementation Phases

### Phase 1: Baseline And Sensor Rate

1. Add `iwr6843_people_tracking_20fps.cfg`.
2. Run headless radar capture for 30 seconds.
3. Confirm radar FPS, track coverage, and no new parser errors.
4. If stable, make the 20 FPS cfg the fusion default.

Success criteria:

- radar parsed FPS is >= 20.0 or documented stable maximum,
- stable single-person radar track ID,
- no unknown TLV regression.

### Phase 2: Fusion Core

1. Add `radar_camera_fusion.py`.
2. Implement single-person state machine.
3. Implement camera/radar association using:
   - projected radar bbox center distance,
   - last camera bbox center distance,
   - optional depth/range agreement,
   - radar track continuity.
4. Add synthetic pose generation from last real skeleton.
5. Add unit tests with fake camera and radar frames.

Success criteria:

- tests prove handoff and recapture preserve one person ID,
- synthetic pose output is valid COCO-17 shape,
- fusion logic has no GUI dependency.

### Phase 3: GUI Integration

1. Wire fusion manager into `VisionPipeline._loop`.
2. Replace point-only handoff with radar track handoff.
3. Draw radar-only bbox and synthetic skeleton.
4. Add debug overlays and status text.
5. Add fusion JSONL logging under `diagnostics_runs/<run_id>/fusion_events.jsonl`.

Success criteria:

- the GUI can run camera-only if radar is absent,
- the GUI can run radar debug overlays when radar is present,
- radar-only mode appears without blocking the UI thread.

### Phase 4: Calibration Wizard

1. Use RealSense intrinsics automatically.
2. Add sample capture from live camera/radar pair.
3. Add solver and reprojection error.
4. Save/load calibration.
5. Keep manual sliders for final tuning.

Success criteria:

- calibration can be completed without editing JSON by hand,
- saved calibration reloads on next run,
- projected radar boxes land near the visible person.

### Phase 5: Performance Pass

1. Profile camera, detection, tracking, fusion, drawing, and GUI refresh.
2. Keep display resize and PIL conversion under control.
3. Use TensorRT YOLO Nano on Orin.
4. Make radar/fusion logging low overhead.
5. Add a visible warning when GUI FPS drops below 20 for more than 2 seconds.

Success criteria:

- GUI FPS >= 20 in normal tracking,
- handoff latency <= 150 ms after camera loss when radar track is already live,
- recapture latency <= 2 stable camera detections.

## Step-By-Step Test Plan When Done

### A. Unit Tests

Run:

```bash
python3 -m pytest tests/test_radar_camera_fusion.py tests/test_radar_diagnostics.py tests/test_radar_viewer_backend.py
```

Expected:

- all tests pass,
- no parser regressions,
- synthetic pose and recapture tests pass.

### B. Radar 20 FPS Validation

Run:

```bash
python3 radar_debug_tools.py live-capture \
  --seconds 30 \
  --cfg iwr6843_people_tracking_20fps.cfg \
  --config-port /dev/ttyACM0 \
  --data-port /dev/ttyACM1 \
  --config-baud 115200 \
  --data-baud 921600
```

Expected:

- parsed FPS near or above 20,
- healthy radar state,
- stable single-person track coverage.

If this fails, repeat with the original cfg and record the stable radar FPS.

### C. Calibration Test

1. Mount camera and radar rigidly in the 3D-printed mount.
2. Start the GUI with RealSense color selected.
3. Open `Radar Calibrator`.
4. Enable debug overlay:
   - radar points,
   - radar tracks,
   - projected radar boxes,
   - always show both.
5. Stand at 7-9 positions across the scene.
6. Press `Capture Sample` at each position only when both camera and radar are
   stable.
7. Press `Solve Calibration`.
8. Save the calibration.
9. Restart the GUI and confirm calibration reloads automatically.

Expected:

- median reprojection error is acceptable for the room,
- projected radar boxes sit close to the visible person,
- saved `radar_camera_calibration.json` is used on restart.

### D. Handoff Test

1. Start with one person fully visible for 5 seconds.
2. Confirm label is `Person #1` and fusion mode is `CAMERA_LOCKED`.
3. Move an opaque blocker between camera and person while keeping radar line of
   sight.
4. Confirm mode transitions through degraded/armed into `RADAR_ONLY`.
5. Confirm the displayed bbox remains YOLO-like and labeled `RADAR ONLY`.
6. Confirm the synthetic skeleton continues from the last real pose shape.
7. Remove the blocker.
8. Confirm camera recaptures quickly and the label remains `Person #1`.

Expected:

- no new person ID is created during recapture,
- radar-only mode starts within 2-3 missed camera frames,
- camera relock occurs after one strong detection or two acceptable detections.

### E. Debug Log Review

After a handoff test, inspect:

```bash
ls -lt diagnostics_runs/*/fusion_events.jsonl | head
python3 radar_debug_tools.py analyze latest --write-json
```

Expected fusion events:

- `camera_locked`
- `camera_degraded`
- `radar_armed`
- `radar_only`
- `camera_relocked`

Each event should include:

- timestamp,
- FALCON track ID,
- radar track ID,
- projected radar bbox,
- camera bbox if present,
- fusion confidence,
- GUI FPS,
- radar FPS.

### F. Performance Test On Orin Nano

1. Select `YOLO26 Nano (TensorRT)`.
2. Use 640x480 color stream.
3. Run for 5 minutes.
4. Confirm status overlay:
   - GUI FPS >= 20,
   - radar FPS >= 20 if the 20 FPS cfg is stable,
   - detection time stays within budget,
   - no growing frame delay.
5. Repeat one occlusion/recapture every minute.

Expected:

- no UI freezes,
- no radar serial reconnects,
- no runaway diagnostics file size,
- stable ID after repeated occlusion cycles.

## MVP Acceptance Criteria

The MVP is complete when:

- one person can be tracked normally by camera,
- the same person can be handed to radar-only mode during camera occlusion,
- a YOLO-like bbox and synthetic pose remain visible during radar-only mode,
- camera recapture keeps the same FALCON ID,
- calibration can be done from the GUI and saved,
- debug overlays can show both radar and camera simultaneously,
- fusion events are logged,
- Orin Nano GUI FPS remains at or above 20 in the normal test setup.
