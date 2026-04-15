import tempfile
import time
import unittest
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import radar
from occlusion import OcclusionState
from radar_camera_fusion import (
    CalibrationSample,
    FusionConfig,
    FusionEventLogger,
    FusionMode,
    RadarCameraFusionManager,
    SyntheticPoseGenerator,
    capture_calibration_sample,
    solve_calibration_samples,
)
from tracking import PersonTracker


def keypoints_for_bbox(bbox, confidence=0.9):
    x1, y1, x2, y2 = [float(value) for value in bbox]
    w = x2 - x1
    h = y2 - y1
    coords = [
        (0.50, 0.08),
        (0.42, 0.06),
        (0.58, 0.06),
        (0.35, 0.10),
        (0.65, 0.10),
        (0.30, 0.28),
        (0.70, 0.28),
        (0.22, 0.47),
        (0.78, 0.47),
        (0.24, 0.65),
        (0.76, 0.65),
        (0.38, 0.58),
        (0.62, 0.58),
        (0.35, 0.78),
        (0.65, 0.78),
        (0.34, 0.96),
        (0.66, 0.96),
    ]
    out = np.zeros((17, 3), dtype=np.float32)
    for idx, (nx, ny) in enumerate(coords):
        out[idx] = [x1 + nx * w, y1 + ny * h, confidence]
    return out


def make_camera_track():
    bbox = np.array([590.0, 300.0, 690.0, 450.0], dtype=np.float32)
    track = PersonTracker(
        track_id=1,
        bbox=bbox.copy(),
        keypoints=keypoints_for_bbox(bbox),
        detection_conf=0.92,
    )
    track.last_bbox_size = track.bbox_wh.copy()
    track.last_real_bbox = bbox.copy()
    track.last_real_keypoints = track.keypoints.copy()
    track.predictor.init_state(track.center, scale=float(track.bbox_wh[0] * track.bbox_wh[1]), aspect=float(track.bbox_wh[0] / track.bbox_wh[1]))
    return track


def make_radar_frame(track_id=7, x=0.0, y=2.0, z=0.0):
    bbox = radar.RadarBox3D(
        x_min=x - 0.35,
        x_max=x + 0.35,
        y_min=y - 0.35,
        y_max=y + 0.35,
        z_min=0.0,
        z_max=1.75,
        track_id=track_id,
    )
    return radar.RadarFrame(
        frame_number=10,
        subframe_number=0,
        num_detected_obj=1,
        num_tlvs=2,
        points=[],
        timestamp=time.time(),
        tracks=[
            radar.RadarTrack(
                track_id=track_id,
                x=x,
                y=y,
                z=z,
                confidence=0.8,
                bbox=bbox,
            )
        ],
        presence=1,
    )


def make_point_cloud_frame():
    return radar.RadarFrame(
        frame_number=11,
        subframe_number=0,
        num_detected_obj=4,
        num_tlvs=2,
        points=[
            radar.RadarPoint(x=-0.20, y=2.00, z=0.85, velocity=0.05, snr=12.0),
            radar.RadarPoint(x=0.00, y=2.10, z=0.95, velocity=0.03, snr=18.0),
            radar.RadarPoint(x=0.15, y=1.95, z=0.90, velocity=-0.02, snr=14.0),
            radar.RadarPoint(x=2.50, y=7.50, z=0.20, velocity=0.00, snr=2.0),
        ],
        timestamp=time.time(),
        tracks=[],
        presence=1,
    )


class SyntheticPoseTests(unittest.TestCase):
    def test_pose_shape_is_scaled_into_target_bbox(self):
        src_bbox = np.array([100.0, 100.0, 200.0, 300.0], dtype=np.float32)
        dst_bbox = np.array([300.0, 120.0, 420.0, 360.0], dtype=np.float32)
        src_pose = keypoints_for_bbox(src_bbox, confidence=0.95)

        pose = SyntheticPoseGenerator.generate(src_pose, src_bbox, dst_bbox, confidence=0.4)

        self.assertEqual(pose.shape, (17, 3))
        self.assertAlmostEqual(float(pose[0, 0]), 360.0, places=3)
        self.assertGreaterEqual(float(pose[:, 0].min()), float(dst_bbox[0]))
        self.assertLessEqual(float(pose[:, 0].max()), float(dst_bbox[2]))
        self.assertTrue(np.allclose(pose[:, 2], 0.4))


class FusionManagerTests(unittest.TestCase):
    def test_camera_locked_when_detection_is_healthy(self):
        manager = RadarCameraFusionManager()
        track = make_camera_track()

        manager.update([track], make_radar_frame(), radar.CameraProjection(), frame_shape=(480, 640, 3))

        self.assertEqual(track.fusion_mode, FusionMode.CAMERA_LOCKED.value)
        self.assertFalse(track.using_radar)

    def test_occlusion_enters_radar_only_and_preserves_person_id(self):
        manager = RadarCameraFusionManager()
        projection = radar.CameraProjection()
        track = make_camera_track()
        radar_frame = make_radar_frame(track_id=7)

        manager.update([track], radar_frame, projection, frame_shape=(480, 1280, 3))
        track.frames_since_detection = 3
        track.is_predicted = True
        track.occlusion_state = OcclusionState.HEAVILY_OCCLUDED
        track.detection_conf = 0.0

        manager.update([track], radar_frame, projection, frame_shape=(480, 1280, 3))

        self.assertEqual(track.track_id, 1)
        self.assertEqual(track.radar_track_id, 7)
        self.assertEqual(track.fusion_mode, FusionMode.RADAR_ONLY.value)
        self.assertTrue(track.using_radar)
        self.assertIsNotNone(track.synthetic_pose)
        self.assertEqual(track.keypoints.shape, (17, 3))

    def test_bad_radar_projection_does_not_steal_track(self):
        manager = RadarCameraFusionManager(config=FusionConfig(radar_pixel_tolerance=80.0))
        track = make_camera_track()
        track.frames_since_detection = 3
        track.is_predicted = True
        track.occlusion_state = OcclusionState.HEAVILY_OCCLUDED

        manager.update(
            [track],
            make_radar_frame(track_id=8, x=3.0, y=2.0, z=0.0),
            radar.CameraProjection(),
            frame_shape=(480, 1280, 3),
        )

        self.assertFalse(track.using_radar)
        self.assertIsNone(track.radar_track_id)

    def test_camera_recapture_releases_radar_and_keeps_id(self):
        manager = RadarCameraFusionManager()
        projection = radar.CameraProjection()
        track = make_camera_track()
        radar_frame = make_radar_frame(track_id=7)
        track.frames_since_detection = 3
        track.is_predicted = True
        track.occlusion_state = OcclusionState.HEAVILY_OCCLUDED
        manager.update([track], radar_frame, projection, frame_shape=(480, 1280, 3))

        track.bbox = np.array([592.0, 302.0, 692.0, 452.0], dtype=np.float32)
        track.keypoints = keypoints_for_bbox(track.bbox, confidence=0.95)
        track.detection_conf = 0.93
        track.frames_since_detection = 0
        track.is_predicted = False
        track.occlusion_state = OcclusionState.VISIBLE
        manager.update([track], radar_frame, projection, frame_shape=(480, 1280, 3))

        self.assertEqual(track.track_id, 1)
        self.assertFalse(track.using_radar)
        self.assertIsNone(track.radar_track_id)
        self.assertEqual(track.fusion_mode, FusionMode.CAMERA_RELOCKED.value)

    def test_event_logger_writes_handoff_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fusion_events.jsonl"
            logger = FusionEventLogger(path)
            manager = RadarCameraFusionManager(event_logger=logger)
            track = make_camera_track()
            track.frames_since_detection = 3
            track.is_predicted = True
            track.occlusion_state = OcclusionState.HEAVILY_OCCLUDED

            manager.update(
                [track],
                make_radar_frame(track_id=7),
                radar.CameraProjection(),
                frame_shape=(480, 1280, 3),
            )
            logger.close()

            text = path.read_text(encoding="utf-8")
            self.assertIn("radar_only", text)
            self.assertIn('"falcon_track_id":1', text)


class CalibrationTests(unittest.TestCase):
    def test_calibration_sample_requires_visible_camera_and_radar(self):
        track = make_camera_track()
        sample = capture_calibration_sample([track], make_radar_frame(track_id=5))

        self.assertEqual(sample.camera_track_id, 1)
        self.assertEqual(sample.radar_track_id, 5)
        self.assertEqual(len(sample.radar_xyz), 3)
        self.assertEqual(len(sample.image_uv), 2)
        self.assertEqual(sample.source, "auto_pose")

    def test_calibration_sample_accepts_manual_click_without_camera_track(self):
        sample = capture_calibration_sample(
            [],
            make_radar_frame(track_id=5),
            image_uv_override=[123.0, 234.0],
        )

        self.assertEqual(sample.camera_track_id, -1)
        self.assertEqual(sample.radar_track_id, 5)
        self.assertEqual(sample.image_uv, [123.0, 234.0])
        self.assertEqual(sample.source, "manual_click")

    def test_calibration_sample_can_fall_back_to_point_cloud(self):
        sample = capture_calibration_sample(
            [],
            make_point_cloud_frame(),
            image_uv_override=[123.0, 234.0],
            allow_point_cloud_fallback=True,
        )

        self.assertEqual(sample.camera_track_id, -1)
        self.assertEqual(sample.radar_track_id, -1)
        self.assertEqual(sample.image_uv, [123.0, 234.0])
        self.assertEqual(sample.source, "manual_click_point_cloud")
        self.assertAlmostEqual(sample.radar_xyz[1], 2.0, places=1)

    def test_calibration_sample_rejects_point_cloud_by_default(self):
        with self.assertRaisesRegex(ValueError, "No radar person track"):
            capture_calibration_sample(
                [],
                make_point_cloud_frame(),
                image_uv_override=[123.0, 234.0],
            )

    def test_calibration_solver_rejects_too_few_samples(self):
        result = solve_calibration_samples(
            [CalibrationSample(radar_xyz=[0.0, 2.0, 0.0], image_uv=[640.0, 360.0])],
            radar.CameraProjection(),
        )

        self.assertFalse(result.ok)
        self.assertIn("Need at least", result.message)

    def test_calibration_solver_recovers_synthetic_mount_offset(self):
        truth = radar.CameraProjection()
        truth.update(
            fx=420.0,
            fy=420.0,
            cx=320.0,
            cy=240.0,
            tx=0.08,
            ty=0.02,
            tz=-0.06,
            yaw_deg=8.0,
            pitch_deg=-4.0,
            roll_deg=2.0,
        )
        start = radar.CameraProjection()
        start.update(fx=420.0, fy=420.0, cx=320.0, cy=240.0)
        radar_points = [
            [-0.7, 1.5, 0.9],
            [-0.3, 2.0, 0.9],
            [0.0, 2.5, 0.9],
            [0.4, 3.0, 0.9],
            [0.7, 2.2, 0.9],
            [-0.5, 3.4, 0.9],
            [0.2, 1.7, 0.9],
            [0.0, 3.8, 0.9],
        ]
        samples = []
        for point in radar_points:
            u, v = truth.project_3d_to_2d(np.asarray(point, dtype=np.float64))
            samples.append(CalibrationSample(radar_xyz=point, image_uv=[u, v]))

        result = solve_calibration_samples(samples, start)

        self.assertTrue(result.ok, result.message)
        self.assertLess(result.median_error_px, 5.0)


if __name__ == "__main__":
    unittest.main()
