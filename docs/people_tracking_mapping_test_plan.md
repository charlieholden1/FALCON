# People-Tracking Physical Mapping Test Plan

## Goal

Measure how well the radar's 3D plot matches the real world for **people tracking**, and use the results to fine tune the physical mapping.

This plan is **not** trying to prove micron-level precision. The target is practical human tracking quality:

- stable person position in the room
- believable person height / 3D box placement
- useful walking-path accuracy
- consistent enough mapping to support later camera fusion

## Scope

This plan evaluates:

- 3D point cloud placement versus real-world position
- tracked-person box placement versus real-world person location
- repeatability while a person stands still
- path accuracy while a person walks
- separation quality for two people
- gross height / box realism

This plan does **not** try to validate:

- limb-level body shape
- precise skeletal pose
- sub-centimeter metrology

## Relevant Repo Context

The current viewer and frame model already support:

- fixed room rendering in [radar_viewer.py](/home/ionat/FALCON/radar_viewer.py)
- scene metadata and boundary boxes in [radar.py](/home/ionat/FALCON/radar.py)
- tracked people and 3D boxes in [radar.py](/home/ionat/FALCON/radar.py)

The mapping tests should be run using the same cfg and mounting arrangement intended for the real deployment.

## Practical Accuracy Targets

For indoor people tracking, use these as initial pass/fail targets:

### Minimum usable

- static XY position median error: `<= 0.30 m`
- static XY 95th percentile error: `<= 0.50 m`
- static Z / box-height error: `<= 0.35 m`
- walking-path median XY error: `<= 0.40 m`
- walking-path 95th percentile XY error: `<= 0.60 m`
- single-person detection persistence in active zone: `>= 95%`
- track ID continuity during unobstructed walking: `>= 90%`

### Good target

- static XY position median error: `<= 0.20 m`
- static XY 95th percentile error: `<= 0.35 m`
- static Z / box-height error: `<= 0.25 m`
- walking-path median XY error: `<= 0.30 m`
- walking-path 95th percentile XY error: `<= 0.50 m`
- two-person separation success at `>= 0.75 m` spacing: `>= 90%`

These values are intentionally person-scale. If the system is within roughly `20-30 cm` most of the time, it is usually good enough for room-scale people tracking and a reasonable starting point for radar-camera fusion.

## Test Equipment

- mounted IWR6843 sensor in final or near-final placement
- the exact cfg to be used in deployment
- the current 3D room viewer
- tape measure or laser distance meter
- floor tape
- masking tape or chalk for floor markers
- plumb line or weighted string
- notebook, spreadsheet, or CSV log
- optional phone video for later timing review

## Coordinate Assumptions To Confirm First

Before collecting data, confirm the mapping convention the team will use:

- radar origin: sensor location
- `x`: left/right across the room
- `y`: forward distance away from the sensor
- `z`: height

Also record:

- sensor height above floor
- sensor tilt
- sensor yaw relative to the room
- cfg file used
- whether scene boxes in cfg match the physical room

If these are wrong, the rest of the test data will be misleading.

## Test Area Setup

Create a simple floor grid in the active tracking region:

- `x` markers every `0.5 m` or `1.0 m`
- `y` markers every `0.5 m` or `1.0 m`
- at least `3` depth rows
- at least `3` lateral columns

Recommended indoor grid:

- `x = -2.0, -1.0, 0.0, 1.0, 2.0 m`
- `y = 1.0, 2.5, 4.0, 5.5 m`

Mark a known standing point at each grid intersection.

For height checks:

- use one or two people of known approximate height
- optionally place a tall box / tripod / pole of known height as a non-human reference

## Data To Record At Each Test Point

For each condition, log:

- timestamp
- test ID
- real-world `x, y, z`
- measured track center `x, y, z`
- measured box extents
- number of detections / track dropouts
- notes on multipath, occlusion, or instability

Recommended derived metrics:

- `error_x = measured_x - truth_x`
- `error_y = measured_y - truth_y`
- `error_z = measured_z - truth_z`
- `xy_error = sqrt(error_x^2 + error_y^2)`
- `xyz_error = sqrt(error_x^2 + error_y^2 + error_z^2)`

## Phase 1: Mounting And Zero-Point Sanity Check

Purpose: catch obvious coordinate mistakes before detailed testing.

Steps:

1. Measure and record sensor height from floor.
2. Confirm the viewer room orientation looks sensible.
3. Stand one person centered in front of the radar at about `y = 2.0 m`.
4. Verify the track appears:
   - in front of the sensor, not behind it
   - roughly centered if the person is centered
   - at a believable height
5. Move `0.5 m` left and right and verify the sign of `x`.
6. Move `0.5 m` closer and farther and verify the sign of `y`.

Pass condition:

- axis directions are correct
- no obvious room rotation or mirror inversion
- person box appears in the right general location

If this fails, do not continue. Fix mounting / cfg geometry first.

## Phase 2: Static Single-Person Grid Test

Purpose: measure absolute position accuracy and repeatability.

Procedure:

1. Place one person at each marked floor location.
2. At each location, stand naturally for `10 seconds`.
3. Record:
   - median tracked center over the interval
   - standard deviation of tracked center over the interval
   - median box height
4. Repeat each point `2-3` times if possible.

Suggested truth model:

- `truth_x`, `truth_y`: floor marker center
- `truth_z`: approximate torso center or chest center, not top of head

Why:

The tracker box center is usually closer to body center than to the top of the head, so comparing `z` to full body height can be misleading.

Evaluate:

- median XY error per location
- 95th percentile XY error across all locations
- bias by region of room
- jitter while standing still

Look for patterns:

- all points shifted equally: mounting offset
- error increases with distance: range scaling / detection geometry issue
- left side different from right side: yaw or rotation issue

## Phase 3: Static Rotation / Pose Test

Purpose: check whether body orientation changes the mapped position too much.

At `3-4` representative locations:

1. Stand facing the radar for `10 seconds`.
2. Turn `90` degrees for `10 seconds`.
3. Turn away from radar for `10 seconds`.
4. Stand with arms at sides, then slightly wider stance.

Evaluate:

- how much the tracked center drifts with pose
- whether the 3D box becomes unrealistically large or shifted

Target:

- pose-dependent center drift should ideally stay `<= 0.20-0.30 m`

## Phase 4: Walking Path Test

Purpose: validate real tracking performance rather than only static snapshots.

Prepare `3-4` simple walking paths:

- straight toward radar
- straight away from radar
- left-to-right across field of view
- diagonal path

Procedure:

1. Mark the path on the floor with tape.
2. Walk at natural indoor speed.
3. Repeat each path `3-5` times.
4. Optionally record video from above or from the side for timing review.

Evaluate:

- median XY error to the planned path
- 95th percentile XY error
- track continuity
- box stability
- lag or overshoot at turns

Also note:

- does the path curve in radar space when it should be straight?
- does the track jump at certain depths or angles?

## Phase 5: Two-Person Separation Test

Purpose: confirm the radar is good enough for people tracking, not just single blobs.

Test conditions:

- two people side by side
- one behind the other with partial depth separation
- crossing paths

Recommended spacings:

- `0.5 m`
- `0.75 m`
- `1.0 m`
- `1.5 m`

Evaluate:

- when two distinct people become two distinct tracks
- track swaps during crossing
- merged boxes
- ID continuity

Target:

- reliable separation at `>= 0.75 m`
- acceptable behavior at `1.0 m` or wider

## Phase 6: Height And Box Realism Test

Purpose: check whether the 3D boxes are believable and useful.

Procedure:

1. Use at least `2` people with different heights if possible.
2. Stand each person at near, mid, and far range.
3. Record:
   - box bottom relative to floor
   - box top relative to person body
   - box width / depth realism

Evaluate:

- does the box start near floor level?
- is the box center visually centered on the person?
- does box height roughly track actual person height?

For people tracking, the box only needs to be **roughly human-realistic**. If the top is within about `20-35 cm` and the center is stable, that is often good enough.

## Phase 7: Edge And Failure Cases

Purpose: find where mapping quality breaks down.

Try:

- person near the left edge of coverage
- person near the right edge of coverage
- person near the far edge of coverage
- person partially crouching
- person near reflective objects
- person entering / leaving the scene

Record where:

- tracks disappear
- boxes inflate
- point cloud becomes sparse
- mapping becomes biased

This helps define the true reliable zone for deployment.

## Suggested Logging Table

Use a sheet with columns like:

| Test ID | Person | Truth X | Truth Y | Truth Z | Measured X | Measured Y | Measured Z | XY Error | Box Height | Dropouts | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|

For walking tests:

| Trial | Path Type | Mean XY Error | P95 XY Error | ID Swaps | Dropouts | Notes |
|---|---|---:|---:|---:|---:|---|

## How To Fine Tune Based On Results

### If all points are shifted the same way

Likely issue:

- wrong sensor position in the physical model
- wrong mounting origin assumption

Action:

- remeasure sensor location
- verify sensor height
- verify the room origin convention used by the team

### If left/right is consistently skewed

Likely issue:

- yaw misalignment

Action:

- physically realign the sensor
- verify any angle / orientation cfg assumptions

### If near points look good but far points drift

Likely issue:

- tracking quality limit
- scene geometry mismatch
- weak coverage at range

Action:

- tighten the reliable operating zone
- retest with a cleaner environment
- verify range-related cfg choices

### If height is consistently wrong

Likely issue:

- wrong sensor height
- wrong tilt
- wrong interpretation of box center versus body height

Action:

- remeasure mounting height
- verify sensor tilt
- compare box center and box top separately

### If tracks jump or swap during crossings

Likely issue:

- tracker limitation rather than pure geometry error

Action:

- measure this separately from static localization accuracy
- keep a realistic performance envelope for multi-person situations

## Final Pass / Fail Recommendation

For the current use case, I would consider the mapping **good enough to proceed** if all of these are true:

- static XY median error is `<= 0.20-0.30 m`
- static XY 95th percentile error is `<= 0.35-0.50 m`
- walking median XY error is `<= 0.30-0.40 m`
- track IDs are mostly stable in single-person motion
- two people at `0.75-1.0 m` separation are usually distinguishable
- 3D boxes are visually believable and centered well enough for operator trust

## Recommended Next Step After This Plan

Once this mapping accuracy is acceptable:

1. freeze the physical mounting
2. freeze the cfg used for deployment
3. record the accepted room transform assumptions
4. move on to radar-camera extrinsic calibration

Do **not** start camera fusion until the radar-only mapping is stable and repeatable.
