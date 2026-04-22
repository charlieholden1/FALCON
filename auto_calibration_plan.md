# Auto-Calibration Plan — FALCON Radar ↔ Camera

This plan replaces the current manual "click torso → Capture Sample × 7–9 → Solve"
workflow with a background auto-calibration routine that collects, filters,
solves, and validates samples on its own. The manual path stays available as a
fallback.

Audience: the next coding agent who will implement this. Read [agent.md](agent.md)
first for hardware constants, the radar recovery rules, and the current
calibration rationale.

---

## 1. Why auto-calibrate

Current pain points:

- Manual click is slow (≈60 s/sample with walking), error-prone at the edges
  of the FOV, and gates every deployment on a human in the scene.
- Torso click accuracy varies by ±30 px; that dominates reprojection error at
  close range.
- Guided flow does not use RealSense depth for the click — only for the
  post-hoc sample enrichment — so we throw away a free 3-D constraint.
- Recalibration after a bump or radar re-seating requires the same 5+ minute
  dance every time.

Goal: a "stand in the FOV and walk around for ~30 s" pathway that captures
10–20 well-distributed samples, rejects junk automatically, solves with the
3-D Umeyama path when RealSense is streaming, and saves only when the
residuals are below thresholds. No clicking required.

---

## 2. What we already have (do not re-build)

Referenced files + symbols (use these verbatim, don't duplicate them):

- [radar_camera_fusion.py:84-132](radar_camera_fusion.py#L84-L132) `CalibrationSample` with `camera_xyz`/`depth_m`.
- [radar_camera_fusion.py:621-689](radar_camera_fusion.py#L621-L689) `capture_calibration_sample` — builds a sample from the best visible camera track + best radar track (or click override). Keep this as the per-sample builder.
- [radar_camera_fusion.py:692-799](radar_camera_fusion.py#L692-L799) `solve_calibration_samples` — 2-D reprojection LM solve, 5+ samples, 65 px median gate.
- [radar_camera_fusion.py:884-954](radar_camera_fusion.py#L884-L954) `solve_calibration_3d` — closed-form Umeyama on `camera_xyz`, 4+ samples, 40 px median gate.
- [radar_camera_fusion.py:957-983](radar_camera_fusion.py#L957-L983) `solve_calibration_auto` — prefers 3-D then falls back to 2-D.
- [radar_camera_fusion.py:1307-1319](radar_camera_fusion.py#L1307-L1319) `_calibration_image_point` — torso midpoint from COCO-17 (indices 5,6,11,12).
- [falcon_gui.py:828-899](falcon_gui.py#L828-L899) pipeline calibration handles (`capture_calibration_sample`, `solve_calibration_samples`, `set_calibration_click`).
- [falcon_gui.py:1376-1583](falcon_gui.py#L1376-L1583) `RadarCalibrationWindow` — existing UI to extend.
- [radar.py:2665-2821](radar.py#L2665-L2821) `CameraProjection` (K + extrinsics + save/load, `DEFAULT_PATH=radar_camera_calibration.json`).

Reuse these. Do not rewrite the solvers or the sample schema.

---

## 3. High-level approach

Add an **AutoCalibrationController** that runs alongside the fusion loop.
It drives three phases:

1. **Seed** — apply prior extrinsics from hardware truth + RealSense intrinsics.
2. **Collect** — watch every camera/radar frame; emit a `CalibrationSample`
   only when strict gates pass. Enforce spatial coverage so the solve is
   well-conditioned.
3. **Solve & commit** — once enough diverse samples exist, run
   `solve_calibration_auto`. If the residuals pass, write to
   `CameraProjection.DEFAULT_PATH`. Otherwise keep collecting.

The controller is non-destructive: nothing happens to live projection until a
solve is accepted. Manual capture/solve remain wired to the same sample list.

---

## 4. Seeding (initial pose prior)

Before sampling, derive a reasonable starting extrinsic so the 2-D LM solve
converges even if the 3-D path is unavailable:

- From [agent.md](agent.md): radar center at **1.85 m** above floor, camera at
  **1.91 m**, both assumed level and co-located horizontally.
- Radar axes (per the parser): X = left/right, Y = forward/depth, Z = up.
  Camera axes: X right, Y down, Z forward. `CameraProjection._BASE_ALIGNMENT`
  already handles the axis swap ([radar.py:2669-2676](radar.py#L2669-L2676)).
- Seed extrinsics: `tx=0.0, ty=+0.06 (camera is 6 cm above radar), tz=0.0,
  yaw=0, pitch=0, roll=0`. Small ty is positive because camera Y is down.
- Intrinsics: reuse `apply_camera_intrinsics_guess()` — this already grabs
  factory RealSense intrinsics.

Expose this as `VisionPipeline.seed_calibration_prior()` called once on
pipeline start and before each new auto-calibration run.

---

## 5. Collection gates (the hard part)

For each live frame (run inside the vision loop or a fused-frame callback),
compute a candidate sample; accept only if **all** the following pass.

Per-frame gates:

| Gate | Threshold | Rationale |
| --- | --- | --- |
| Exactly one camera track with `is_predicted=False` and `frames_since_detection==0` | required | Avoid ambiguous associations in multi-person scenes. |
| `detection_conf ≥ 0.60` | FusionConfig `strong_detection_conf` is 0.65 — relax 0.05 for speed | Weeds out low-quality YOLO outputs. |
| ≥ 9 visible torso/limb keypoints with conf ≥ 0.35 | matches `strong_keypoint_count` | Torso midpoint needs shoulders + hips; 9 keeps us honest. |
| Exactly one fresh radar track (stale cutoff = `FusionConfig.stale_radar_frame_s`, 0.35 s) | required | Linked ID ambiguity otherwise. |
| Radar track confidence ≥ 0.5 and age ≥ 5 frames | required | TI tracker sometimes emits one-frame ghosts. |
| Radar-camera ID pairing stable across ≥ 8 consecutive frames (same `camera_track_id` ↔ same `radar_track_id`) | debounce | Rejects transient mis-associations. |
| `|radar_velocity| ≤ 0.35 m/s` and camera bbox centroid velocity ≤ 40 px/s | person is near-stationary | Minimises radar-camera timing skew. Averaged over a 0.5 s window. |
| Depth at torso pixel is finite and within `[0.5 m, 8.0 m]` | required when RealSense is up | Needed for the 3-D Umeyama path. |
| `|depth_m - radar.y| ≤ 0.7 m` | cross-check | Radar forward should roughly match RealSense depth. |
| Torso pixel is ≥ 40 px inside the image border | required | Avoids clipped keypoints. |

Spatial coverage gates (over the sample set so far):

- Divide the usable FOV into a 3 × 3 grid in image space (u/v thirds) and
  into three depth bands (`y < 1.5 m`, `1.5–3 m`, `> 3 m`).
- Require ≥ 1 sample in ≥ 6 of the 9 image cells and ≥ 1 in each depth band
  before allowing a solve.
- Enforce a minimum 3-D separation of 0.4 m between any two accepted
  samples' radar positions — stops the person from "flooding" one corner.

Rate limiting:

- Maximum **one accepted sample per 0.75 s** to avoid correlated noise.
- Maximum **25 samples** total before forcing a solve+reset.

---

## 6. Solve & commit loop

Once collection gates say "enough + diverse" (minimum 8 samples, coverage
satisfied):

1. Snapshot the sample list.
2. Call `solve_calibration_auto(samples, projection)`.
3. Accept only if:
   - 3-D path: `median_error_px ≤ 25` and `rmse_m ≤ 0.12` (tighter than the
     function's own 40 px default — we can afford to be picky in auto mode).
   - 2-D path (no depth): `median_error_px ≤ 45`.
   - No rotation parameter within 5° of its ±90° bound.
4. On accept: `projection.update(**result.params)`, `projection.save()`, log
   `event_type="auto_calibration_saved"` via `FusionEventLogger`, clear
   samples, transition controller to **Monitoring** state.
5. On reject: keep samples, relax nothing, continue collecting until either
   improvement or a 60 s timeout — then discard the oldest 50 % of samples
   and keep going. This implements a simple RANSAC-ish retry without
   dropping good points.

Monitoring state:

- Keep watching residuals using new candidate samples (without updating the
  projection). If live median pixel error on fresh samples grows past 2×
  the solve-time median for > 10 s, automatically re-enter **Collect**.
  This handles tripod bumps.

---

## 7. Public API (what to add)

```python
# radar_camera_fusion.py
@dataclass
class AutoCalibrationConfig:
    min_samples_for_solve: int = 8
    max_samples: int = 25
    min_sample_spacing_m: float = 0.4
    min_sample_interval_s: float = 0.75
    required_coverage_cells: int = 6
    stationary_velocity_mps: float = 0.35
    stationary_pixel_speed: float = 40.0
    stable_link_frames: int = 8
    redetect_pixel_error_multiplier: float = 2.0
    redetect_trigger_seconds: float = 10.0
    max_rotation_near_bound_deg: float = 5.0
    solve_retry_timeout_s: float = 60.0

class AutoCalibrationState(Enum):
    IDLE = "IDLE"
    COLLECTING = "COLLECTING"
    SOLVING = "SOLVING"
    MONITORING = "MONITORING"

class AutoCalibrationController:
    def __init__(self, projection, event_logger, config=None): ...
    def start(self): ...         # IDLE -> COLLECTING, clears samples
    def stop(self): ...           # -> IDLE, discard in-flight samples
    def update(self,
               camera_tracks, radar_frame, depth_frame,
               frame_shape, now) -> AutoCalibrationStatus: ...
    @property
    def status(self) -> AutoCalibrationStatus: ...
```

`AutoCalibrationStatus` is a small dataclass the GUI renders: state, sample
count, coverage-cells-filled, last-solve-result, last-event string.

Integration points in `falcon_gui.py`:

- Construct one `AutoCalibrationController` on `VisionPipeline.__init__`.
- Call `controller.update(...)` inside `_loop` right after `self.fusion.update(...)`.
  The controller reuses the same tracks, radar frame, depth frame, and frame
  shape already in scope — no extra locking.
- Expose `pipeline.auto_calibration_start()`, `...stop()`, `...status()` for
  the GUI.
- Extend `RadarCalibrationWindow`: a new "Auto Calibrate" button at the top
  of the existing sample row, a live progress label
  ("Collecting 6/8 samples, 7/9 cells covered"), and a "Cancel" button that
  calls `stop()`. Advanced sliders and manual Capture/Solve stay as-is.

---

## 8. Tests to add (mirror existing style in `tests/`)

- `tests/test_auto_calibration.py`:
  - Feed synthetic camera+radar frame sequences built from a known
    ground-truth projection; assert the controller recovers parameters to
    within 2 cm / 0.5°.
  - Gate-rejection tests: moving person, single-cell flooding, stale radar
    frame, mismatched radar/depth — none should emit samples.
  - Coverage test: verify the 3×3 + 3-depth-band gate blocks premature solves.
  - Commit-and-monitor: after accept, simulate a ground-truth rotation and
    verify the controller re-enters COLLECTING within the timeout.
- Extend `tests/test_radar_camera_fusion.py` with one integration test that
  runs `AutoCalibrationController` against the existing synthetic radar
  fixtures to confirm no regression of the manual path.

All tests must run via the repo-standard command:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

Avoid `python3 -m unittest tests.test_*` per [agent.md](agent.md).

---

## 9. Operator workflow (what the user sees)

1. Launch GUI, start the camera + radar as normal. Intrinsics auto-seed.
2. Open Radar Calibrator → click **Auto Calibrate**.
3. Walk slowly through the FOV: stand still for ~1 s in each of 6–8 spots
   covering left/center/right × near/mid/far. Status label updates live.
4. Controller auto-solves when coverage + sample count gates pass. On
   accept it saves to `radar_camera_calibration.json` and the label reads
   "Saved: median Xpx, rmse Ycm".
5. On reject the label says which gate failed ("median 52 px > 25 — keep
   walking"). The user can keep moving or press **Cancel** to revert.
6. Manual Capture/Solve is still there for edge cases (no RealSense, only
   one radar track location reachable, etc.).

---

## 10. Implementation order (for the next agent)

Do this top-to-bottom. Each step is independently testable.

1. Add `AutoCalibrationConfig`, `AutoCalibrationState`, `AutoCalibrationStatus`
   dataclasses in `radar_camera_fusion.py`. Add a unit test that their
   defaults match this plan.
2. Add `AutoCalibrationController` skeleton with pure `update()` logic —
   no GUI wiring. Drive it from synthetic frame data in unit tests.
3. Implement the per-frame gates (Section 5). Unit-test each gate in
   isolation using hand-crafted `RadarFrame`/track fixtures.
4. Implement the spatial coverage tracker (3×3 image grid + 3 depth bands)
   with its own unit test.
5. Implement the solve+commit+monitor state machine. Unit-test
   accept/reject/retry paths with a known-good projection.
6. Wire into `VisionPipeline`: construction, `_loop` update call,
   `auto_calibration_start/stop/status` public methods,
   `seed_calibration_prior`.
7. Extend `RadarCalibrationWindow` with the button + progress label +
   cancel. Do not remove manual Capture/Solve.
8. Hand-test on hardware per [agent.md](agent.md) "Calibration Workflow"
   section, with one addition: run the auto path first, then spot-check
   reprojection by having the user walk to 3 fresh positions and verifying
   `Radar Tracks: 1 R0=(...)` overlays the torso within ~25 px.
9. Update [agent.md](agent.md): replace the "Calibration Workflow" section
   with the auto path as primary, manual as fallback.

---

## 11. Out of scope (explicitly)

- Moving-person calibration (radar/camera time sync bias). Hard problem,
  deferred until single-person handoff MVP ships.
- Multi-person auto-calibration. We require exactly one camera track +
  exactly one radar track — multi-person wrecks association.
- Intrinsic calibration. We trust RealSense factory values.
- Radar firmware flashing / CLI config changes. Controller assumes the
  preferred `iwr6843_people_tracking_20fps.cfg` is already loaded.

---

## 12. Risk notes

- **Radar-camera timing skew** at walking speed is the single biggest
  error source. The `stationary_*` gates address it, but if residuals
  plateau above the accept threshold, revisit by adding a radar-frame
  timestamp offset parameter to the solver.
- **Single-person requirement** is strict. Document it in the GUI label:
  "Only one person in view during Auto Calibrate."
- **Floor echoes / ghost tracks** from IWR6843 can masquerade as people.
  The `radar_track_age ≥ 5 frames` + `confidence ≥ 0.5` gates should
  filter most, but watch for `presence=1, tracks=1` false positives
  during testing.
- **RealSense depth holes** on dark clothing kill the 3-D Umeyama path.
  The 2-D fallback catches this, but the accept thresholds differ — make
  sure the UI tells the user which path fired.
