import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import radar
from radar_camera_fusion import (
    AutoCalibrationConfig,
    AutoCalibrationController,
    AutoCalibrationState,
    CalibrationSample,
    FusionEventLogger,
)


FRAME_SHAPE = (480, 640, 3)
K = np.array(
    [
        [600.0, 0.0, 320.0],
        [0.0, 600.0, 240.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def keypoints_at(u: float, v: float, confidence: float = 0.92) -> np.ndarray:
    offsets = [
        (0.0, -70.0),
        (-12.0, -75.0),
        (12.0, -75.0),
        (-22.0, -66.0),
        (22.0, -66.0),
        (-26.0, -20.0),
        (26.0, -20.0),
        (-36.0, 18.0),
        (36.0, 18.0),
        (-32.0, 58.0),
        (32.0, 58.0),
        (-18.0, 20.0),
        (18.0, 20.0),
        (-22.0, 70.0),
        (22.0, 70.0),
        (-20.0, 110.0),
        (20.0, 110.0),
    ]
    out = np.zeros((17, 3), dtype=np.float32)
    for idx, (dx, dy) in enumerate(offsets):
        out[idx] = [float(u + dx), float(v + dy), confidence]
    # Keep the calibration torso midpoint exactly on the projected target.
    for idx in (5, 6, 11, 12):
        out[idx, 0] = float(u)
        out[idx, 1] = float(v)
    return out


def camera_track_at(u: float, v: float, track_id: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        track_id=track_id,
        bbox=np.array([u - 45.0, v - 95.0, u + 45.0, v + 115.0], dtype=np.float32),
        keypoints=keypoints_at(u, v),
        detection_conf=0.94,
        is_predicted=False,
        frames_since_detection=0,
    )


def radar_frame_at(point, *, now: float, track_id: int = 7, confidence: float = 0.9, speed: float = 0.0):
    x, y, z = [float(value) for value in point]
    return radar.RadarFrame(
        frame_number=int(now * 1000.0),
        subframe_number=0,
        num_detected_obj=1,
        num_tlvs=1,
        points=[],
        timestamp=now,
        tracks=[
            radar.RadarTrack(
                track_id=track_id,
                x=x,
                y=y,
                z=z,
                vx=speed,
                confidence=confidence,
            )
        ],
        presence=1,
    )


def depth_frame_for(u: float, v: float, depth_m: float) -> np.ndarray:
    frame = np.zeros(FRAME_SHAPE[:2], dtype=np.uint16)
    x = int(round(u))
    y = int(round(v))
    frame[y, x] = int(round(depth_m * 1000.0))
    return frame


def synthetic_observation(truth: radar.CameraProjection, point, *, now: float):
    u, v = truth.project_3d_to_2d(np.asarray(point, dtype=np.float64))
    camera_xyz = truth.radar_to_camera(np.asarray(point, dtype=np.float64))
    return (
        camera_track_at(float(u), float(v)),
        radar_frame_at(point, now=now),
        depth_frame_for(float(u), float(v), float(camera_xyz[2])),
    )


class AutoCalibrationTests(unittest.TestCase):
    def test_config_defaults_match_plan(self):
        cfg = AutoCalibrationConfig()
        self.assertEqual(cfg.min_samples_for_solve, 8)
        self.assertEqual(cfg.max_samples, 25)
        self.assertEqual(cfg.required_coverage_cells, 5)
        self.assertEqual(cfg.required_depth_bands, 2)
        self.assertEqual(cfg.stable_link_frames, 10)
        self.assertAlmostEqual(cfg.stationary_velocity_mps, 1.85)
        self.assertAlmostEqual(cfg.stationary_pixel_speed, 260.0)
        self.assertEqual(cfg.min_visible_keypoints, 12)
        self.assertEqual(cfg.min_torso_keypoints, 4)
        self.assertAlmostEqual(cfg.min_sample_interval_s, 0.85)
        self.assertAlmostEqual(cfg.max_median_error_3d_px, 25.0)
        self.assertAlmostEqual(cfg.max_rmse_3d_m, 0.12)
        self.assertAlmostEqual(cfg.max_translation_near_bound_m, 0.05)

    def test_start_uses_solver_projection_without_mutating_live_projection(self):
        live = radar.CameraProjection(K=K.copy())
        live.update(tx=0.80, ty=-0.40, tz=0.25, yaw_deg=12.0)
        solver = radar.CameraProjection(K=K.copy())
        solver.update(tx=0.0, ty=0.06, tz=0.0, yaw_deg=0.0)
        controller = AutoCalibrationController(live, FusionEventLogger())

        controller.start(solver_projection=solver)

        self.assertAlmostEqual(live.params["tx"], 0.80)
        self.assertAlmostEqual(live.params["ty"], -0.40)
        self.assertAlmostEqual(live.params["yaw_deg"], 12.0)

    def test_controller_recovers_and_saves_synthetic_projection(self):
        truth = radar.CameraProjection(K=K.copy())
        truth.update(
            tx=0.04,
            ty=0.06,
            tz=0.02,
            yaw_deg=2.0,
            pitch_deg=-1.0,
            roll_deg=1.0,
        )
        candidate = radar.CameraProjection(K=K.copy())
        cfg = AutoCalibrationConfig()
        points = [
            [-0.42, 1.20, 0.36],
            [0.00, 1.28, -0.05],
            [0.42, 1.22, -0.30],
            [-0.72, 2.20, -0.18],
            [0.00, 2.15, 0.50],
            [0.72, 2.25, 0.02],
            [-1.10, 3.60, 0.08],
            [0.00, 3.55, -0.44],
            [1.10, 3.65, 0.28],
            [-0.20, 1.50, -0.50],
            [0.60, 1.80, 0.45],
        ]

        original_path = radar.CameraProjection.DEFAULT_PATH
        with tempfile.TemporaryDirectory() as tmp:
            radar.CameraProjection.DEFAULT_PATH = Path(tmp) / "radar_camera_calibration.json"
            try:
                controller = AutoCalibrationController(
                    candidate,
                    FusionEventLogger(),
                    cfg,
                )
                controller.start()
                now = 100.0
                status = controller.status
                for point in points:
                    for _ in range(20):
                        track, frame, depth = synthetic_observation(truth, point, now=now)
                        status = controller.update([track], frame, depth, FRAME_SHAPE, now=now)
                        now += 0.1
                    if status.state == AutoCalibrationState.MONITORING:
                        break
                self.assertEqual(status.state, AutoCalibrationState.MONITORING, status.progress_text())
                self.assertTrue(radar.CameraProjection.DEFAULT_PATH.exists())
                self.assertIsNotNone(status.last_solve_result)
                self.assertTrue(status.last_solve_result.ok, status.progress_text())
                self.assertEqual(status.last_solve_result.solve_type, "3d")

                solved = candidate.params
                expected = truth.params
                for key in ("tx", "ty", "tz"):
                    self.assertAlmostEqual(solved[key], expected[key], delta=0.02)
                for key in ("yaw_deg", "pitch_deg", "roll_deg"):
                    self.assertAlmostEqual(solved[key], expected[key], delta=0.5)
            finally:
                radar.CameraProjection.DEFAULT_PATH = original_path

    def test_fast_moving_person_gate_rejects_sample(self):
        cfg = AutoCalibrationConfig(
            min_radar_age_frames=1,
            stable_link_frames=1,
            min_sample_interval_s=0.75,
        )
        controller = AutoCalibrationController(
            radar.CameraProjection(K=K.copy()),
            FusionEventLogger(),
            cfg,
        )
        controller.start()
        controller._last_sample_time = 1.0

        frame = radar_frame_at([0.0, 2.0, 0.0], now=1.1)
        depth = depth_frame_for(320.0, 240.0, 2.0)
        controller.update([camera_track_at(320.0, 240.0)], frame, depth, FRAME_SHAPE, now=1.1)

        frame = radar_frame_at([0.0, 2.0, 0.0], now=1.2)
        depth = depth_frame_for(520.0, 240.0, 2.0)
        status = controller.update([camera_track_at(520.0, 240.0)], frame, depth, FRAME_SHAPE, now=1.2)

        self.assertEqual(controller.samples, [])
        self.assertIn("Move slower", status.last_event)

    def test_coverage_blocks_premature_solve(self):
        controller = AutoCalibrationController(
            radar.CameraProjection(K=K.copy()),
            FusionEventLogger(),
            AutoCalibrationConfig(),
        )
        controller.start()
        controller.samples = [
            CalibrationSample(
                radar_xyz=[0.0, 2.0 + index * 0.45, 0.0],
                image_uv=[320.0, 240.0],
                camera_xyz=[0.0, 0.0, 2.0 + index * 0.45],
                depth_m=2.0 + index * 0.45,
            )
            for index in range(8)
        ]

        self.assertFalse(controller._ready_to_solve(FRAME_SHAPE))
        self.assertEqual(len(controller._coverage_cells(FRAME_SHAPE)), 1)

    def test_stale_radar_frame_rejects_sample(self):
        cfg = AutoCalibrationConfig(min_radar_age_frames=1, stable_link_frames=1)
        controller = AutoCalibrationController(
            radar.CameraProjection(K=K.copy()),
            FusionEventLogger(),
            cfg,
        )
        controller.start()
        frame = radar_frame_at([0.0, 2.0, 0.0], now=1.0)
        depth = depth_frame_for(320.0, 240.0, 2.0)

        status = controller.update([camera_track_at(320.0, 240.0)], frame, depth, FRAME_SHAPE, now=2.0)

        self.assertEqual(controller.samples, [])
        self.assertIn("no fresh track", status.last_event)

    def test_radar_depth_mismatch_rejects_sample(self):
        cfg = AutoCalibrationConfig(min_radar_age_frames=1, stable_link_frames=1)
        controller = AutoCalibrationController(
            radar.CameraProjection(K=K.copy()),
            FusionEventLogger(),
            cfg,
        )
        controller.start()
        frame = radar_frame_at([0.0, 2.0, 0.0], now=1.0)
        depth = depth_frame_for(320.0, 240.0, 4.0)

        status = controller.update([camera_track_at(320.0, 240.0)], frame, depth, FRAME_SHAPE, now=1.0)

        self.assertEqual(controller.samples, [])
        self.assertIn("Radar/depth mismatch", status.last_event)

    def test_multiple_radar_tracks_selects_best_confident_track(self):
        cfg = AutoCalibrationConfig(
            min_radar_age_frames=1,
            stable_link_frames=1,
            min_sample_interval_s=0.0,
            min_hold_frames=1,
        )
        controller = AutoCalibrationController(
            radar.CameraProjection(K=K.copy()),
            FusionEventLogger(),
            cfg,
        )
        controller.start()
        frame = radar_frame_at([0.0, 2.0, 0.0], now=1.0, track_id=7, confidence=0.7)
        frame.tracks.append(
            radar.RadarTrack(
                track_id=9,
                x=0.4,
                y=2.1,
                z=0.0,
                confidence=0.95,
            )
        )

        controller.update([camera_track_at(320.0, 240.0)], frame, None, FRAME_SHAPE, now=1.0)

        self.assertEqual(len(controller.samples), 1)
        self.assertEqual(controller.samples[0].radar_track_id, 9)

    def test_auto_calibration_locks_to_first_accepted_radar_track(self):
        cfg = AutoCalibrationConfig(
            min_radar_age_frames=1,
            stable_link_frames=1,
            min_sample_interval_s=0.0,
            min_hold_frames=1,
        )
        controller = AutoCalibrationController(
            radar.CameraProjection(K=K.copy()),
            FusionEventLogger(),
            cfg,
        )
        controller.start()

        first = radar_frame_at([0.0, 2.0, 0.0], now=1.0, track_id=9)
        controller.update([camera_track_at(320.0, 240.0)], first, None, FRAME_SHAPE, now=1.0)

        second = radar_frame_at([0.8, 2.8, 0.0], now=2.0, track_id=7)
        status = controller.update([camera_track_at(440.0, 240.0)], second, None, FRAME_SHAPE, now=2.0)

        self.assertEqual(len(controller.samples), 1)
        self.assertEqual(controller.samples[0].radar_track_id, 9)
        self.assertIn("locked to radar R9", status.last_event)

    def test_occluded_track_rejects_sample(self):
        from occlusion import OcclusionState

        cfg = AutoCalibrationConfig(min_radar_age_frames=1, stable_link_frames=1)
        controller = AutoCalibrationController(
            radar.CameraProjection(K=K.copy()),
            FusionEventLogger(),
            cfg,
        )
        controller.start()
        track = camera_track_at(320.0, 240.0)
        track.occlusion_state = OcclusionState.PARTIALLY_OCCLUDED
        frame = radar_frame_at([0.0, 2.0, 0.0], now=1.0)
        depth = depth_frame_for(320.0, 240.0, 2.0)

        status = controller.update([track], frame, depth, FRAME_SHAPE, now=1.0)

        self.assertEqual(controller.samples, [])
        self.assertIn("wait until fully visible", status.last_event)

    def test_records_calibration_session_jsonl(self):
        cfg = AutoCalibrationConfig(
            min_radar_age_frames=1,
            stable_link_frames=1,
            min_sample_interval_s=0.0,
            min_hold_frames=1,
        )
        controller = AutoCalibrationController(
            radar.CameraProjection(K=K.copy()),
            FusionEventLogger(),
            cfg,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "auto_calibration_session.jsonl"
            controller.start(record_path=path)
            frame = radar_frame_at([0.0, 2.0, 0.0], now=1.0)
            controller.update([camera_track_at(320.0, 240.0)], frame, None, FRAME_SHAPE, now=1.0)
            controller.stop()

            text = path.read_text(encoding="utf-8")
            self.assertIn('"type":"start"', text)
            self.assertIn('"type":"sample_accepted"', text)
            self.assertIn('"type":"stop"', text)

    def test_solve_now_uses_coherent_radar_track_cluster(self):
        truth = radar.CameraProjection(K=K.copy())
        truth.update(tx=0.05, ty=0.04, tz=0.02, yaw_deg=3.0)
        controller = AutoCalibrationController(
            radar.CameraProjection(K=K.copy()),
            FusionEventLogger(),
            AutoCalibrationConfig(),
        )
        good_points = [
            [-0.8, 1.5, 0.2],
            [-0.4, 2.0, 0.0],
            [0.0, 2.5, 0.4],
            [0.4, 3.0, -0.1],
            [0.8, 3.5, 0.3],
            [-0.2, 4.0, 0.1],
        ]
        for point in good_points:
            u, v = truth.project_3d_to_2d(np.asarray(point, dtype=np.float64))
            controller.samples.append(
                CalibrationSample(
                    radar_xyz=point,
                    image_uv=[float(u), float(v)],
                    radar_track_id=11,
                    source="test_good",
                )
            )
        for index, point in enumerate(good_points):
            controller.samples.append(
                CalibrationSample(
                    radar_xyz=[point[0] + 1.5, point[1], point[2]],
                    image_uv=[80.0 + index * 15.0, 420.0],
                    radar_track_id=22,
                    source="test_bad",
                )
            )

        original_path = radar.CameraProjection.DEFAULT_PATH
        with tempfile.TemporaryDirectory() as tmp:
            radar.CameraProjection.DEFAULT_PATH = Path(tmp) / "radar_camera_calibration.json"
            try:
                controller.state = AutoCalibrationState.COLLECTING
                status = controller.solve_now(now=10.0)
                self.assertEqual(status.state, AutoCalibrationState.MONITORING, status.progress_text())
                self.assertIsNotNone(status.last_solve_result)
                self.assertTrue(status.last_solve_result.ok, status.last_solve_result.message)
                self.assertEqual(status.last_solve_result.sample_count, len(good_points))
            finally:
                radar.CameraProjection.DEFAULT_PATH = original_path


if __name__ == "__main__":
    unittest.main()
