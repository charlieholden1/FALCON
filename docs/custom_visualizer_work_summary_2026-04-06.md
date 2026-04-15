# Custom Radar Visualizer Work Summary

Date: April 6, 2026

## Why We Built It

The project originally depended on TI's tooling and the IWR6843ISK + mmWaveICBOOST path. After the mmWaveICBOOST stopped being usable and TI's Industrial Visualizer was no longer a reliable way to inspect the board, the goal shifted to building a **custom viewer and diagnostic path** that could answer two practical questions:

1. Is the IWR6843ISK still working?
2. Is the problem TI's visualizer / host tooling rather than the radar itself?

The evidence from serial CLI testing showed the radar was still alive and accepting configuration, so the custom visualizer became the main path forward.

## What We Built

### 1. Standalone diagnostic viewer

We created a standalone radar viewer with:

- live mode
- replay mode
- explicit connect / disconnect flow
- sensor startup ownership
- tracked-person rendering
- diagnostic state panels

This made it possible to inspect radar output without depending on TI's Industrial Visualizer.

### 2. Reusable radar source / parser layer

We expanded the radar backend so it no longer behaved like a one-off script. The backend now supports:

- a `FrameSource` style interface
- live UART streaming
- replay loading from saved JSON
- normalized `RadarFrame` objects
- session-state reporting
- health verdicts such as `Healthy`, `Config Failed`, `No Data`, and `Replay Mode`

This gave the viewer a stable foundation instead of mixing parsing, UI, and serial startup logic together.

### 3. Better serial robustness

We spent time hardening the CLI and startup path because the CP2105 bridge and reopen behavior were inconsistent across sessions and across Linux vs Windows. Improvements included:

- better port discovery and suggested config/data pairing
- stronger serial wakeup / retry logic
- multiple startup strategies for live mode
- cleaner handling of CLI logs and timeouts

This helped separate real radar failures from serial / host-side issues.

### 4. Replay support

We made replay captures go through the same normalized frame model as live data. That means:

- the same viewer can render live or replay
- the same tracked-target code path works in both modes
- the same scene metadata and future fusion hooks stay aligned

Replay mode became a very useful tool for UI development and parser validation even when hardware access was awkward.

### 5. TI people-tracking decode support

The parser was extended beyond raw point clouds so the viewer could show higher-level people-tracking information more like TI's visualizer. We added support for:

- tracked targets
- target IDs
- target heights
- target indexes
- presence information
- 3D track boxes / extents

That moved the viewer from being only a packet monitor into something that is actually useful for people tracking.

### 6. Fusion-ready data model

We refactored the normalized radar frame model so later radar-camera fusion is easier. The frame structure now carries:

- timestamps
- point cloud data
- stable track IDs
- scene metadata
- calibration hooks
- camera projection helpers

We did **not** fully implement camera fusion yet, but the data model is now prepared for it.

## UI Evolution

### First phase: diagnostic viewer

The first usable viewer focused on proving the radar path worked:

- simple controls
- session state
- CLI log tail
- point cloud rendering
- replay loading

This phase answered the viability question and gave us a base to iterate from.

### Second phase: better 3D tracking visualization

We then upgraded the viewer to render:

- live 3D point cloud
- tracked people
- 3D boxes
- trajectories
- scene metadata

This brought the viewer closer to TI's people-tracking visualizer behavior.

### Third phase: Linux / Orin-focused frontend upgrade

Because the target platform is Ubuntu on an NVIDIA Orin Nano, we moved beyond the earlier Tkinter / Matplotlib approach and upgraded the primary viewer path to:

- `PySide6`
- `PyQtGraph`
- OpenGL rendering

This was done to improve:

- live rendering smoothness
- UI polish
- long-run usability
- responsiveness on Linux / Orin

The older Tk viewer was preserved as a fallback.

### Fourth phase: fixed 3D room-style environment

The viewer was then refined to behave more like TI's 3D room / cube visualizer. We changed it so that:

- the 2D plots were removed
- the main view is now one 3D room-style environment
- the world no longer keeps auto-rescaling during runtime
- the camera stays stable unless the user changes it
- people appear as 3D bounded boxes
- radar points appear in the same fixed room

This made the viewer feel much more like a real room-scale tracking tool instead of a generic scatter plot.

### Fifth phase: visual polish cleanup

We also cleaned up visual details such as:

- more production-style panel layout
- monochrome room guides
- removal of the rainbow default OpenGL axis
- more coherent room styling

This pushed the viewer closer to the intended industrial / engineering-console look.

## Hardware / Host Findings From Today

We learned several important things while building and testing:

- the IWR6843ISK itself appears viable
- the CLI/config path was confirmed working through manual serial interaction
- the CP2105 port roles needed care
- Windows proved to be the known-good host path for the serial setup
- Linux had more friction around serial behavior and GUI/runtime dependencies

An important turning point was confirming that the setup worked on Windows. That strongly suggested the radar and serial hardware were viable, and that the remaining issues were on the host/tooling side rather than the sensor being dead.

## Supporting Artifacts Added

Over the course of the work, we added or improved:

- the main viewer entrypoint in [radar_viewer.py](/home/ionat/FALCON/radar_viewer.py)
- the Tk fallback in [radar_viewer_tk.py](/home/ionat/FALCON/radar_viewer_tk.py)
- the richer parser / source layer in [radar.py](/home/ionat/FALCON/radar.py)
- the diagnostic startup helper in [radar_diag.py](/home/ionat/FALCON/radar_diag.py)
- viewer/backend tests in [tests/test_radar_viewer_backend.py](/home/ionat/FALCON/tests/test_radar_viewer_backend.py)
- parser and health tests in [tests/test_radar_diagnostics.py](/home/ionat/FALCON/tests/test_radar_diagnostics.py)
- Linux viewer dependency list in [requirements-radar-viewer.txt](/home/ionat/FALCON/requirements-radar-viewer.txt)
- Qt runtime notes in [.qt-runtime/README.md](/home/ionat/FALCON/.qt-runtime/README.md)
- the mapping validation plan in [docs/people_tracking_mapping_test_plan.md](/home/ionat/FALCON/docs/people_tracking_mapping_test_plan.md)

## Current State At End Of Day

By the end of today's work, the custom visualizer had become:

- independent from TI's Industrial Visualizer
- capable of live or replay rendering
- capable of tracked-person visualization
- structured for later radar-camera fusion
- tuned toward Ubuntu / Orin Nano performance
- visually closer to a TI-style 3D room viewer

The viewer is no longer just a diagnostic experiment. It is now a real custom radar visualization tool with a clear architecture and a path forward.

## What Still Remains

The main follow-on items after today are:

- fine tuning physical mapping against real-world positions
- validating people-tracking accuracy in the room
- freezing the mounting and cfg assumptions
- calibrating radar to camera once radar-only mapping is good enough
- further visual polish if desired

## Bottom Line

Today we went from "the TI tooling path is unreliable and we need to know if the board is even usable" to having a **custom 3D people-tracking radar visualizer** with diagnostics, replay support, tracked-person boxes, fixed room rendering, and a cleaner foundation for future radar-camera fusion.
