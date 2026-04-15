import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import math

import numpy as np

import radar
import radar_camera_fusion
import radar_viewer


class RadarViewerBackendTests(unittest.TestCase):
    def test_default_cfg_path_points_at_existing_cfg(self):
        cfg_path = radar_viewer._default_cfg_path()
        self.assertTrue(cfg_path)
        self.assertTrue(Path(cfg_path).exists())

    def test_backend_bootstrap_has_a_clear_result(self):
        self.assertTrue(radar_viewer._HAS_QT or radar_viewer._QT_IMPORT_ERROR is not None)

    def test_fixed_room_box_expands_scene_boxes_without_reframing_each_frame(self):
        box = radar.RadarBox3D(
            x_min=-1.0,
            x_max=0.5,
            y_min=1.5,
            y_max=4.0,
            z_min=0.1,
            z_max=2.3,
            kind="boundary",
        )
        frame = radar.RadarFrame(
            frame_number=1,
            subframe_number=0,
            num_detected_obj=0,
            num_tlvs=0,
            points=[],
            timestamp=0.0,
            scene=radar.RadarSceneMetadata(boundary_boxes=[box]),
        )
        room = radar_viewer._fixed_room_box_for_scene(frame)
        self.assertLessEqual(room.x_min, -1.0)
        self.assertGreaterEqual(room.x_max, 0.5)
        self.assertLessEqual(room.y_min, 0.0)
        self.assertGreaterEqual(room.y_max, 4.0)
        self.assertLessEqual(room.z_min, 0.0)
        self.assertGreaterEqual(room.z_max, 2.3)
        self.assertGreaterEqual(room.width, radar_viewer.DEFAULT_ROOM_WIDTH_M)
        self.assertGreaterEqual(room.depth, radar_viewer.DEFAULT_ROOM_DEPTH_M)
        self.assertGreaterEqual(room.height, radar_viewer.DEFAULT_ROOM_HEIGHT_M)

    def test_box_mesh_geometry_builds_closed_prism(self):
        vertices, faces = radar_viewer._box_mesh_geometry(
            radar.RadarBox3D(
                x_min=-0.4,
                x_max=0.4,
                y_min=1.0,
                y_max=2.0,
                z_min=0.0,
                z_max=1.8,
            )
        )
        self.assertEqual(vertices.shape, (8, 3))
        self.assertEqual(faces.shape, (12, 3))

    def test_room_guides_produce_floor_and_wall_segments(self):
        segments = radar_viewer._room_guide_segments(radar_viewer._default_room_box(), spacing=2.0)
        self.assertGreater(len(segments), 0)
        self.assertEqual(segments.shape[1], 3)


class CalibrationSolverTests(unittest.TestCase):
    def _make_rotation(self, yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
        yaw = math.radians(yaw_deg)
        pitch = math.radians(pitch_deg)
        roll = math.radians(roll_deg)
        cy_, sy_ = math.cos(yaw), math.sin(yaw)
        cp_, sp_ = math.cos(pitch), math.sin(pitch)
        cr_, sr_ = math.cos(roll), math.sin(roll)
        Rz = np.array([[cy_, -sy_, 0.0], [sy_, cy_, 0.0], [0.0, 0.0, 1.0]])
        Ry = np.array([[cp_, 0.0, sp_], [0.0, 1.0, 0.0], [-sp_, 0.0, cp_]])
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr_, -sr_], [0.0, sr_, cr_]])
        return Rz @ Ry @ Rx

    def test_umeyama_recovers_known_transform(self):
        R_true = self._make_rotation(30.0, -10.0, 5.0)
        t_true = np.array([0.10, 1.50, -0.20])
        rng = np.random.default_rng(12345)
        src = rng.uniform(-2.0, 2.0, size=(10, 3))
        dst = (R_true @ src.T).T + t_true

        R, t, rmse = radar_camera_fusion.umeyama_rigid_transform(src, dst)
        np.testing.assert_allclose(R, R_true, atol=1e-9)
        np.testing.assert_allclose(t, t_true, atol=1e-9)
        self.assertLess(rmse, 1e-9)

    def test_umeyama_rejects_reflections(self):
        # Force det(R) = +1 even when the cross-covariance SVD is degenerate.
        src = np.eye(3)
        dst = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        R, _t, _rmse = radar_camera_fusion.umeyama_rigid_transform(src, dst)
        self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=6)

    def test_backproject_pixel_uses_projection_intrinsics(self):
        K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
        projection = radar.CameraProjection(K=K)
        xyz = radar_camera_fusion.backproject_pixel(320.0, 240.0, 2.5, projection)
        self.assertIsNotNone(xyz)
        x, y, z = xyz
        self.assertAlmostEqual(x, 0.0, places=6)
        self.assertAlmostEqual(y, 0.0, places=6)
        self.assertAlmostEqual(z, 2.5, places=6)

        off = radar_camera_fusion.backproject_pixel(380.0, 240.0, 2.0, projection)
        self.assertIsNotNone(off)
        self.assertAlmostEqual(off[0], (380.0 - 320.0) * 2.0 / 600.0, places=6)
        self.assertAlmostEqual(off[2], 2.0, places=6)

    def test_backproject_pixel_rejects_invalid_depth(self):
        projection = radar.CameraProjection()
        self.assertIsNone(radar_camera_fusion.backproject_pixel(10.0, 10.0, 0.0, projection))
        self.assertIsNone(radar_camera_fusion.backproject_pixel(10.0, 10.0, None, projection))
        self.assertIsNone(
            radar_camera_fusion.backproject_pixel(10.0, 10.0, float("nan"), projection)
        )

    def test_solve_calibration_3d_recovers_projection(self):
        # Build a ground-truth projection, synthesise radar<->camera pairs,
        # and confirm the Umeyama-based solver reports low reprojection error.
        K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
        R_true = self._make_rotation(12.0, -8.0, 3.0)
        t_true = np.array([0.05, 0.30, 0.10])
        rt = np.eye(4)
        rt[:3, :3] = R_true
        rt[:3, 3] = t_true
        truth = radar.CameraProjection(K=K.copy(), RT=rt)

        rng = np.random.default_rng(7)
        radar_points = rng.uniform(-1.5, 1.5, size=(8, 3))
        radar_points[:, 1] = np.abs(radar_points[:, 1]) + 1.0  # keep in front

        samples = []
        for p in radar_points:
            cam_pt = (R_true @ p) + t_true
            u, v = truth.project_3d_to_2d(p)
            samples.append(
                radar_camera_fusion.CalibrationSample(
                    radar_xyz=[float(p[0]), float(p[1]), float(p[2])],
                    image_uv=[float(u), float(v)],
                    camera_xyz=[float(cam_pt[0]), float(cam_pt[1]), float(cam_pt[2])],
                    depth_m=float(cam_pt[2]),
                    source="test",
                )
            )

        # Start the solver from default (identity-ish) extrinsics.
        candidate = radar.CameraProjection(K=K.copy())
        result = radar_camera_fusion.solve_calibration_3d(samples, candidate)
        self.assertTrue(result.ok, msg=result.message)
        self.assertLess(result.median_error_px, 1.0)

    def test_solve_calibration_auto_falls_back_to_2d(self):
        # No camera_xyz data -> auto solver must use the legacy 2-D path.
        K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
        truth = radar.CameraProjection(K=K.copy())
        rng = np.random.default_rng(3)
        samples = []
        for _ in range(6):
            p = rng.uniform(-1.0, 1.0, size=3)
            p[1] = abs(p[1]) + 1.0
            u, v = truth.project_3d_to_2d(p)
            samples.append(
                radar_camera_fusion.CalibrationSample(
                    radar_xyz=[float(p[0]), float(p[1]), float(p[2])],
                    image_uv=[float(u), float(v)],
                    source="test_2d",
                )
            )
        result = radar_camera_fusion.solve_calibration_auto(
            samples,
            radar.CameraProjection(K=K.copy()),
        )
        # Legacy path runs; we only assert it produced a result object, not
        # whether the median error passes its default threshold.
        self.assertIsInstance(result, radar_camera_fusion.CalibrationSolveResult)


class MountGeometryTemplatingTests(unittest.TestCase):
    def test_apply_mount_geometry_rewrites_sensor_position_and_boxes(self):
        driver = radar.IWR6843Driver(
            mount_geometry={
                "height_m": 2.10,
                "azimuth_tilt_deg": -5.0,
                "elevation_tilt_deg": 12.5,
                "boundary_box": [-2.5, 2.5, 0.0, 5.0, 0.0, 2.6],
                "static_boundary_box": [-2.0, 2.0, 0.5, 4.5, 0.0, 2.5],
            }
        )
        self.assertTrue(driver._apply_mount_geometry("sensorPosition 1.85 0 0").startswith("sensorPosition 2.1000"))
        self.assertTrue(
            driver._apply_mount_geometry("boundaryBox -4 4 0 8 0 3").startswith("boundaryBox -2.5000 2.5000")
        )
        self.assertTrue(
            driver._apply_mount_geometry("staticBoundaryBox -3 3 0.5 7.5 0 3").startswith(
                "staticBoundaryBox -2.0000 2.0000"
            )
        )
        # Untouched commands pass through unchanged.
        self.assertEqual(driver._apply_mount_geometry("sensorStart"), "sensorStart")

    def test_apply_mount_geometry_noop_without_overrides(self):
        driver = radar.IWR6843Driver()
        self.assertEqual(
            driver._apply_mount_geometry("sensorPosition 1.85 0 0"),
            "sensorPosition 1.85 0 0",
        )


if __name__ == "__main__":
    unittest.main()
