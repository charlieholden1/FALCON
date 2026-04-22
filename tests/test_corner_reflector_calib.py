"""Unit tests for GuidedBodyCalibController (formerly CornerReflectorCalibController)."""

import math
import sys
import unittest
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from radar import CameraProjection
from radar_camera_fusion import (
    GuidedBodyCalibController,
    GuidedBodyCalibState,
    # Legacy aliases still importable.
    CornerReflectorCalibController,
    CornerReflectorCalibState,
)


# ── Minimal radar-frame / point / track stubs ─────────────────────────────────

@dataclass
class _Point:
    x: float = 0.0
    y: float = 3.0
    z: float = 0.0
    velocity: float = 0.0
    snr: float = 30.0


@dataclass
class _Track:
    track_id: int = 0
    x: float = 0.0
    y: float = 3.0
    z: float = 0.0


@dataclass
class _RadarFrame:
    points: List[_Point] = field(default_factory=list)
    tracks: List[_Track] = field(default_factory=list)


def _proj() -> CameraProjection:
    K = np.array([[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]], dtype=np.float64)
    return CameraProjection(K=K)


# ── Tests ─────────────────────────────────────────────────────────────────────

class CaptureBehaviourTests(unittest.TestCase):

    def test_capture_uses_track_position(self):
        ctrl = GuidedBodyCalibController(_proj())
        ctrl.start()
        frame = _RadarFrame(
            points=[_Point(x=0.1, y=2.0, snr=10.0)],
            tracks=[_Track(x=0.5, y=3.0, z=0.1)],
        )
        ok, _ = ctrl.capture_position(frame, None, (320.0, 240.0))
        self.assertTrue(ok)
        self.assertAlmostEqual(ctrl.samples[0].radar_xyz[0], 0.5)
        self.assertAlmostEqual(ctrl.samples[0].radar_xyz[1], 3.0)

    def test_capture_falls_back_to_points_when_no_tracks(self):
        ctrl = GuidedBodyCalibController(_proj())
        ctrl.start()
        points = [
            _Point(x=0.1, y=2.0, snr=10.0),
            _Point(x=0.5, y=3.0, snr=40.0),
            _Point(x=-0.2, y=4.0, snr=8.0),
        ]
        frame = _RadarFrame(points=points, tracks=[])
        ok, _ = ctrl.capture_position(frame, None, (320.0, 240.0))
        self.assertTrue(ok)
        # Falls back to highest-SNR point.
        self.assertAlmostEqual(ctrl.samples[0].radar_xyz[0], 0.5)
        self.assertAlmostEqual(ctrl.samples[0].radar_xyz[1], 3.0)

    def test_capture_rejects_multiple_tracks(self):
        ctrl = GuidedBodyCalibController(_proj())
        ctrl.start()
        frame = _RadarFrame(
            tracks=[_Track(x=0.0, y=2.0), _Track(x=1.5, y=3.0)],
        )
        ok, msg = ctrl.capture_position(frame, None, (320.0, 240.0))
        self.assertFalse(ok)
        self.assertIn("multiple", msg.lower())

    def test_capture_rejects_near_field(self):
        ctrl = GuidedBodyCalibController(_proj())
        ctrl.start()
        frame = _RadarFrame(points=[_Point(y=0.1, snr=50.0), _Point(y=0.12, snr=5.0)])
        ok, msg = ctrl.capture_position(frame, None, (320.0, 240.0))
        self.assertFalse(ok)
        self.assertIn("near-field", msg.lower())

    def test_capture_rejects_no_radar_data(self):
        ctrl = GuidedBodyCalibController(_proj())
        ctrl.start()
        ok, _ = ctrl.capture_position(None, None, (320.0, 240.0))
        self.assertFalse(ok)

    def test_state_transitions_to_done_after_all_positions(self):
        from radar_camera_fusion import _GUIDED_POSITIONS
        ctrl = GuidedBodyCalibController(_proj())
        ctrl.start()
        self.assertEqual(ctrl.state, GuidedBodyCalibState.COLLECTING)
        for i in range(len(_GUIDED_POSITIONS)):
            frame = _RadarFrame(tracks=[_Track(x=float(i) * 0.3, y=2.0 + i * 0.5)])
            ok, _ = ctrl.capture_position(frame, None, (320.0 + i * 10, 240.0))
            self.assertTrue(ok, f"position {i+1} capture failed unexpectedly")
        self.assertEqual(ctrl.state, GuidedBodyCalibState.DONE)
        self.assertEqual(ctrl.sample_count, len(_GUIDED_POSITIONS))

    def test_capture_rejected_when_not_in_collecting_state(self):
        ctrl = GuidedBodyCalibController(_proj())
        # State is IDLE, not COLLECTING.
        frame = _RadarFrame(tracks=[_Track(y=3.0)])
        ok, _ = ctrl.capture_position(frame, None, (320.0, 240.0))
        self.assertFalse(ok)

    def test_min_positions_less_than_total_positions(self):
        from radar_camera_fusion import _GUIDED_POSITIONS
        self.assertLess(
            GuidedBodyCalibController.MIN_POSITIONS,
            len(_GUIDED_POSITIONS),
            "MIN_POSITIONS should allow early solve before all 6 steps",
        )

    def test_legacy_alias_works(self):
        """CornerReflectorCalibController and CornerReflectorCalibState still importable."""
        ctrl = CornerReflectorCalibController(_proj())
        ctrl.start()
        self.assertEqual(ctrl.state, CornerReflectorCalibState.COLLECTING)


class SolveTests(unittest.TestCase):

    def _make_ctrl_with_samples(self, n: int = 6) -> GuidedBodyCalibController:
        """Pre-load controller with consistent synthetic samples (near-zero reprojection error)."""
        proj = _proj()
        ctrl = GuidedBodyCalibController(proj)
        from radar_camera_fusion import ReflectorSample

        for i in range(n):
            angle = i * (2 * math.pi / n)
            rx, ry, rz = 1.5 * math.cos(angle), 3.0 + i * 0.5, 0.3 * math.sin(angle)
            cx, cy, cz = rx, -rz, ry
            u = float(proj.K[0, 0]) * cx / cz + float(proj.K[0, 2])
            v = float(proj.K[1, 1]) * cy / cz + float(proj.K[1, 2])
            ctrl.samples.append(ReflectorSample(
                radar_xyz=[rx, ry, rz],
                image_uv=[u, v],
                camera_xyz=[cx, cy, cz],
                depth_m=float(ry),
                snr_peak=0.0,
            ))
        ctrl.state = GuidedBodyCalibState.DONE
        return ctrl

    def test_solve_returns_ok_with_valid_camera_xyz(self):
        ctrl = self._make_ctrl_with_samples(6)
        result = ctrl.solve()
        self.assertTrue(result.ok, f"solve failed: {result.message}")

    def test_solve_returns_not_ok_without_camera_xyz(self):
        proj = _proj()
        ctrl = GuidedBodyCalibController(proj)
        from radar_camera_fusion import ReflectorSample
        for i in range(5):
            ctrl.samples.append(ReflectorSample(
                radar_xyz=[float(i), 3.0, 0.0],
                image_uv=[320.0, 240.0],
            ))
        ctrl.state = GuidedBodyCalibState.DONE
        result = ctrl.solve()
        self.assertFalse(result.ok)

    def test_start_clears_samples(self):
        ctrl = self._make_ctrl_with_samples(4)
        self.assertEqual(ctrl.sample_count, 4)
        ctrl.start()
        self.assertEqual(ctrl.sample_count, 0)
        self.assertEqual(ctrl.state, GuidedBodyCalibState.COLLECTING)


if __name__ == "__main__":
    unittest.main()
