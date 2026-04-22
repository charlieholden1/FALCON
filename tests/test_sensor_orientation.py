"""
Sensor orientation tests for the IWR6843 radar.

Ground truth (TI Detection Layer Tuning Guide §1.2, Figure 2):
  +x = right, as viewed when facing forward (same perspective as camera observer)
  +y = forward / boresight depth
  +z = up
  Azimuth φ is the angle from +y toward +x, so positive φ = target to the right.

These tests verify the full chain:
  raw azimuth angle
    → IWR6843Driver._spherical_to_cartesian
    → CameraProjection.project_3d_to_2d
    → pixel (u, v)

No hardware required.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from radar import CameraProjection, IWR6843Driver


def _proj_640x480() -> CameraProjection:
    """Standard synthetic 640×480 camera, no extrinsic rotation."""
    K = np.array([[500.0, 0.0, 320.0],
                  [0.0, 500.0, 240.0],
                  [0.0, 0.0,   1.0]], dtype=np.float64)
    return CameraProjection(K=K)


class SphericalToCartesianTests(unittest.TestCase):
    """Unit tests for IWR6843Driver._spherical_to_cartesian sign conventions."""

    def test_positive_azimuth_gives_positive_x(self):
        x, y, z = IWR6843Driver._spherical_to_cartesian(
            azimuth_rad=0.3, elevation_rad=0.0, distance_m=3.0
        )
        self.assertGreater(x, 0.0,
            "positive azimuth must yield positive x (right of boresight)")

    def test_negative_azimuth_gives_negative_x(self):
        x, y, z = IWR6843Driver._spherical_to_cartesian(
            azimuth_rad=-0.3, elevation_rad=0.0, distance_m=3.0
        )
        self.assertLess(x, 0.0,
            "negative azimuth must yield negative x (left of boresight)")

    def test_boresight_zero_azimuth(self):
        x, y, z = IWR6843Driver._spherical_to_cartesian(
            azimuth_rad=0.0, elevation_rad=0.0, distance_m=3.0
        )
        self.assertAlmostEqual(x, 0.0, places=9,
            msg="boresight (azimuth=0) must give x=0")
        self.assertAlmostEqual(y, 3.0, places=6,
            msg="boresight must give y=distance")

    def test_depth_y_always_positive(self):
        for az in [-0.5, 0.0, 0.5]:
            x, y, z = IWR6843Driver._spherical_to_cartesian(az, 0.0, 2.0)
            self.assertGreater(y, 0.0,
                f"y must be positive for azimuth={az} (sensor faces forward)")


class ProjectionOrientationTests(unittest.TestCase):
    """Verify CameraProjection.project_3d_to_2d lateral and vertical direction."""

    def test_right_radar_point_projects_right(self):
        proj = _proj_640x480()
        u_right, _ = proj.project_3d_to_2d(np.array([1.0, 3.0, 0.0]))
        u_center, _ = proj.project_3d_to_2d(np.array([0.0, 3.0, 0.0]))
        self.assertGreater(u_right, u_center,
            "radar x>0 (right) must project to u > image centre")

    def test_left_radar_point_projects_left(self):
        proj = _proj_640x480()
        u_left, _ = proj.project_3d_to_2d(np.array([-1.0, 3.0, 0.0]))
        u_center, _ = proj.project_3d_to_2d(np.array([0.0, 3.0, 0.0]))
        self.assertLess(u_left, u_center,
            "radar x<0 (left) must project to u < image centre")

    def test_rightward_motion_increases_u_monotonically(self):
        proj = _proj_640x480()
        xs = [-1.5, -0.5, 0.0, 0.5, 1.5]
        us = [proj.project_3d_to_2d(np.array([x, 3.0, 0.0]))[0] for x in xs]
        for i in range(len(us) - 1):
            self.assertLess(us[i], us[i + 1],
                f"u must increase as x increases "
                f"(x={xs[i]}→{xs[i+1]}, u={us[i]}→{us[i+1]})")

    def test_upward_point_projects_higher_on_screen(self):
        proj = _proj_640x480()
        _, v_up = proj.project_3d_to_2d(np.array([0.0, 3.0, 0.5]))
        _, v_mid = proj.project_3d_to_2d(np.array([0.0, 3.0, 0.0]))
        self.assertLess(v_up, v_mid,
            "radar z>0 (above boresight) must project to smaller v (higher on screen)")


class EndToEndAzimuthOrientationTests(unittest.TestCase):
    """Full pipeline: azimuth angle → cartesian → pixel."""

    def test_positive_azimuth_maps_right_of_image_centre(self):
        x, y, z = IWR6843Driver._spherical_to_cartesian(
            azimuth_rad=0.3, elevation_rad=0.0, distance_m=3.0
        )
        u, _ = _proj_640x480().project_3d_to_2d(np.array([x, y, z]))
        self.assertGreater(u, 320,
            "positive azimuth (right of boresight) must yield u > cx=320")

    def test_negative_azimuth_maps_left_of_image_centre(self):
        x, y, z = IWR6843Driver._spherical_to_cartesian(
            azimuth_rad=-0.3, elevation_rad=0.0, distance_m=3.0
        )
        u, _ = _proj_640x480().project_3d_to_2d(np.array([x, y, z]))
        self.assertLess(u, 320,
            "negative azimuth (left of boresight) must yield u < cx=320")


class TiltCompensationTests(unittest.TestCase):
    """Tests for IWR6843Driver._apply_elevation_tilt() (sensor-frame → world-frame)."""

    def test_zero_tilt_leaves_point_unchanged(self):
        x, y, z = IWR6843Driver._apply_elevation_tilt(0.0, 3.0, 0.0, 0.0)
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, 3.0)
        self.assertAlmostEqual(z, 0.0)

    def test_zero_tilt_with_nonzero_z(self):
        x, y, z = IWR6843Driver._apply_elevation_tilt(1.0, 2.0, 0.5, 0.0)
        self.assertAlmostEqual(x, 1.0)
        self.assertAlmostEqual(y, 2.0)
        self.assertAlmostEqual(z, 0.5)

    def test_8deg_tilt_forward_point_world_z_below_sensor(self):
        # Sensor tilted 8° downward: boresight dips toward floor, so a point directly
        # in front (sensor-frame y=3, z=0) is below sensor level in world frame.
        # z' = -y*sin(8°) ≈ -0.42 m
        x, y, z = IWR6843Driver._apply_elevation_tilt(0.0, 3.0, 0.0, 8.0)
        self.assertLess(z, 0.0, "8° down-tilt on forward point must produce negative world-z (below sensor)")

    def test_8deg_tilt_preserves_range(self):
        x, y, z = IWR6843Driver._apply_elevation_tilt(0.0, 3.0, 0.0, 8.0)
        self.assertAlmostEqual(y**2 + z**2, 9.0, places=5, msg="range must be preserved by rotation")

    def test_x_axis_unchanged_by_elevation_tilt(self):
        # Elevation tilt is rotation about X; x must not change.
        x, y, z = IWR6843Driver._apply_elevation_tilt(1.5, 2.0, 0.3, 15.0)
        self.assertAlmostEqual(x, 1.5, places=9, msg="x must be invariant under elevation tilt rotation")

    def test_90deg_tilt_maps_forward_to_down(self):
        # 90° tilt: y' = y*cos(90°)+z*sin(90°) = z=0; z' = -y*sin(90°)+z*cos(90°) = -y=-3.
        # A point 3 m in front of sensor is now 3 m below sensor level in world frame.
        x, y, z = IWR6843Driver._apply_elevation_tilt(0.0, 3.0, 0.0, 90.0)
        self.assertAlmostEqual(y, 0.0, places=5)
        self.assertAlmostEqual(z, -3.0, places=5)


if __name__ == "__main__":
    unittest.main()
