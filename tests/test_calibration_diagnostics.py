"""Tests for calibration_diag.py.

These use synthetic samples built from a known ground-truth projection so
we can assert exact behaviour of the before/after error accounting and the
physical/cfg sanity checks.
"""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import radar  # noqa: E402
from radar_camera_fusion import CalibrationSample, save_calibration_samples  # noqa: E402

import calibration_diag  # noqa: E402


FRAME_W = 640
FRAME_H = 480
K = np.array(
    [[605.0, 0.0, 320.0],
     [0.0, 605.0, 240.0],
     [0.0, 0.0, 1.0]],
    dtype=np.float64,
)


def make_truth_projection() -> radar.CameraProjection:
    proj = radar.CameraProjection(K=K.copy())
    proj.update(tx=0.0, ty=0.06, tz=0.0, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0)
    return proj


def make_samples_from_truth(truth: radar.CameraProjection, points):
    samples = []
    for idx, point in enumerate(points):
        arr = np.asarray(point, dtype=np.float64)
        u, v = truth.project_3d_to_2d(arr)
        cam_xyz = truth.radar_to_camera(arr)
        samples.append(
            CalibrationSample(
                radar_xyz=[float(x) for x in arr],
                image_uv=[float(u), float(v)],
                camera_track_id=1,
                radar_track_id=2,
                timestamp=1_000.0 + idx,
                confidence=0.95,
                source="test",
                camera_xyz=[float(v2) for v2 in cam_xyz],
                depth_m=float(cam_xyz[2]),
            )
        )
    return samples


POSITIONS = [
    (-1.4, 1.2, 0.6),
    (1.2, 1.4, 0.5),
    (0.0, 2.5, -0.3),
    (-1.0, 3.0, 0.7),
    (1.1, 2.9, -0.4),
    (-1.5, 4.5, 0.5),
    (1.5, 4.5, -0.5),
    (0.0, 4.0, 0.2),
    (0.3, 1.8, -0.6),
]


class AnalyzeSamplesTests(unittest.TestCase):
    def test_perfect_calibration_gives_near_zero_error(self):
        truth = make_truth_projection()
        samples = make_samples_from_truth(truth, POSITIONS)
        diags = calibration_diag.analyze_samples(samples, truth, FRAME_W, FRAME_H)
        agg = calibration_diag.aggregate(diags)
        # project_3d_to_2d rounds to int pixels, so sub-pixel residual is expected.
        self.assertLess(agg["median_px"], 1.0)
        self.assertLess(agg["mean_px"], 1.0)
        self.assertLess(agg["rmse_m"], 1e-6)
        self.assertGreaterEqual(agg["cells_filled"], 5)
        self.assertEqual(agg["bands_filled"], 3)

    def test_wrong_calibration_yields_measurable_error(self):
        truth = make_truth_projection()
        samples = make_samples_from_truth(truth, POSITIONS)
        wrong = make_truth_projection()
        wrong.update(pitch_deg=10.0, ty=0.40)

        diags = calibration_diag.analyze_samples(samples, wrong, FRAME_W, FRAME_H)
        agg = calibration_diag.aggregate(diags)
        self.assertGreater(agg["median_px"], 30.0)
        self.assertGreater(agg["rmse_m"], 0.05)

    def test_outlier_detection_flags_injected_bad_sample(self):
        truth = make_truth_projection()
        samples = make_samples_from_truth(truth, POSITIONS)
        # Corrupt one sample's image_uv far from where it should reproject.
        samples[3].image_uv = [samples[3].image_uv[0] + 200.0,
                               samples[3].image_uv[1] + 200.0]
        diags = calibration_diag.analyze_samples(samples, truth, FRAME_W, FRAME_H)
        outliers = calibration_diag.flag_outliers(diags)
        self.assertIn(3, outliers)

    def test_cell_and_band_binning(self):
        # Top-left: u < w/3, v < h/3, near: y<1.5
        cell = calibration_diag._cell(100.0, 100.0, FRAME_W, FRAME_H)
        self.assertEqual(cell, (0, 0))
        # Center mid: u in middle third, v in middle third, mid: 1.5<=y<=3
        cell = calibration_diag._cell(FRAME_W / 2, FRAME_H / 2, FRAME_W, FRAME_H)
        self.assertEqual(cell, (1, 1))
        self.assertEqual(calibration_diag._band(1.0), "near")
        self.assertEqual(calibration_diag._band(2.0), "mid")
        self.assertEqual(calibration_diag._band(4.0), "far")


class PhysicalSanityTests(unittest.TestCase):
    def test_good_params_reports_all_ok(self):
        params = {"tx": 0.0, "ty": 0.064, "tz": 0.038,
                  "yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0,
                  "fx": 605.0, "fy": 605.0, "cx": 320.0, "cy": 240.0}
        lines = calibration_diag.physical_sanity(params, FRAME_W, FRAME_H)
        joined = "\n".join(lines)
        self.assertNotIn("[WARN]", joined)
        self.assertIn("[OK]", joined)

    def test_bad_pitch_flags_sensor_position_cause(self):
        params = {"tx": 0.0, "ty": 0.06, "tz": 0.0,
                  "yaw_deg": 0.0, "pitch_deg": 22.0, "roll_deg": 0.0,
                  "fx": 605.0, "fy": 605.0, "cx": 320.0, "cy": 240.0}
        lines = calibration_diag.physical_sanity(params, FRAME_W, FRAME_H)
        joined = "\n".join(lines)
        self.assertIn("[WARN]", joined)
        self.assertIn("sensorPosition", joined)

    def test_bad_tx_flags_lateral_offset(self):
        params = {"tx": 0.5, "ty": 0.06, "tz": 0.0,
                  "yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0,
                  "fx": 605.0, "fy": 605.0, "cx": 320.0, "cy": 240.0}
        lines = calibration_diag.physical_sanity(params, FRAME_W, FRAME_H)
        self.assertTrue(any("tx" in line and "WARN" in line for line in lines))


class CfgSanityTests(unittest.TestCase):
    def _write_cfg(self, text: str) -> Path:
        tmp = Path(tempfile.mkstemp(suffix=".cfg")[1])
        tmp.write_text(text, encoding="utf-8")
        self.addCleanup(tmp.unlink, missing_ok=True)
        return tmp

    def test_expected_cfg_reports_ok(self):
        cfg = self._write_cfg(
            "sensorPosition 1.88 0 8\n"
            "allocationParam 60 200 0.1 5 1.5 2\n"
            "fovCfg -1 70.0 15.0\n"
            "compRangeBiasAndRxChanPhase 0.06 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0\n"
        )
        lines = calibration_diag.cfg_sanity(cfg)
        joined = "\n".join(lines)
        self.assertIn("[OK]", joined)
        self.assertNotIn("[WARN]", joined)

    def test_wrong_tilt_flagged(self):
        cfg = self._write_cfg("sensorPosition 1.85 0 15\n")
        lines = calibration_diag.cfg_sanity(cfg)
        joined = "\n".join(lines)
        self.assertIn("tilt", joined)
        self.assertIn("[WARN]", joined)
        self.assertIn("8", joined)

    def test_missing_cfg_warns(self):
        lines = calibration_diag.cfg_sanity(Path("this_does_not_exist_xyz.cfg"))
        self.assertIn("[WARN]", "\n".join(lines))

    def test_range_bias_zero_flagged(self):
        cfg = self._write_cfg(
            "sensorPosition 1.85 0 30\n"
            "compRangeBiasAndRxChanPhase 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0\n"
        )
        lines = calibration_diag.cfg_sanity(cfg)
        joined = "\n".join(lines)
        self.assertIn("[WARN]", joined)
        self.assertIn("range bias", joined.lower())

    def test_nonzero_range_bias_ok(self):
        cfg = self._write_cfg(
            "sensorPosition 1.85 0 30\n"
            "compRangeBiasAndRxChanPhase 0.06 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0\n"
        )
        lines = calibration_diag.cfg_sanity(cfg)
        joined = "\n".join(lines)
        self.assertNotIn("Range bias is 0", joined)
        self.assertIn("[OK]", joined)

    def test_elevation_fov_above_22_flagged(self):
        cfg = self._write_cfg(
            "sensorPosition 1.85 0 30\n"
            "fovCfg -1 70.0 25.0\n"
        )
        lines = calibration_diag.cfg_sanity(cfg)
        joined = "\n".join(lines)
        self.assertIn("[WARN]", joined)
        self.assertIn("elevation", joined.lower())

    def test_elevation_fov_15_ok(self):
        cfg = self._write_cfg(
            "sensorPosition 1.85 0 30\n"
            "fovCfg -1 70.0 15.0\n"
            "compRangeBiasAndRxChanPhase 0.06 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0\n"
        )
        lines = calibration_diag.cfg_sanity(cfg)
        # The fovCfg line should be [OK], not [WARN].
        self.assertFalse(
            any("[WARN]" in l and "fovCfg" in l for l in lines),
            msg="fovCfg at ±15° should not produce a warning",
        )
        self.assertTrue(
            any("[OK]" in l and "fovCfg" in l for l in lines),
            msg="fovCfg at ±15° should produce an OK line",
        )


class RoomSanityTests(unittest.TestCase):
    def _write_cfg(self, text: str) -> Path:
        tmp = Path(tempfile.mkstemp(suffix=".cfg")[1])
        tmp.write_text(text, encoding="utf-8")
        self.addCleanup(tmp.unlink, missing_ok=True)
        return tmp

    def _good_cfg(self) -> Path:
        return self._write_cfg(
            "boundaryBox -4 4 0 5.5 0 3\n"
            "staticBoundaryBox -3 3 0.5 5.0 0 3\n"
            "presenceBoundaryBox -3 3 0.5 5.5 0 3\n"
        )

    def test_clean_config_no_warnings(self):
        cfg = self._good_cfg()
        lines = calibration_diag.room_sanity(cfg, room_depth_m=6.0, room_half_width_m=4.0)
        joined = "\n".join(lines)
        self.assertNotIn("[WARN]", joined)

    def test_boundary_extends_beyond_far_wall(self):
        cfg = self._write_cfg(
            "boundaryBox -4 4 0 9.0 0 3\n"
            "staticBoundaryBox -3 3 0.5 8.5 0 3\n"
            "presenceBoundaryBox -3 3 0.5 9.0 0 3\n"
        )
        lines = calibration_diag.room_sanity(cfg, room_depth_m=6.0, room_half_width_m=4.0)
        joined = "\n".join(lines)
        self.assertIn("[WARN]", joined)
        self.assertIn("far wall", joined.lower())

    def test_lateral_extent_beyond_side_walls(self):
        cfg = self._write_cfg("boundaryBox -6 6 0 5.5 0 3\n")
        lines = calibration_diag.room_sanity(cfg, room_depth_m=6.0, room_half_width_m=4.0)
        joined = "\n".join(lines)
        self.assertIn("[WARN]", joined)
        self.assertIn("half-width", joined.lower())

    def test_none_args_skips_checks(self):
        # With both None, no depth or width checks should fire.
        cfg = self._write_cfg("boundaryBox -4 4 0 99.0 0 3\n")
        lines = calibration_diag.room_sanity(cfg, room_depth_m=None, room_half_width_m=None)
        joined = "\n".join(lines)
        # No boundary-related warnings without reference dimensions.
        self.assertNotIn("far wall", joined.lower())
        self.assertNotIn("half-width", joined.lower())

    def test_missing_cfg_warns(self):
        lines = calibration_diag.room_sanity(
            Path("nonexistent_xyz.cfg"), room_depth_m=6.0, room_half_width_m=4.0
        )
        self.assertIn("[WARN]", "\n".join(lines))


class EndToEndCommandTests(unittest.TestCase):
    def _write_samples(self) -> Path:
        truth = make_truth_projection()
        samples = make_samples_from_truth(truth, POSITIONS)
        tmpdir = Path(tempfile.mkdtemp())
        self.addCleanup(_rm_rf, tmpdir)
        samples_path = tmpdir / "auto_calibration_samples.json"
        save_calibration_samples(samples, samples_path)
        return samples_path

    def _write_calibration(self, **overrides) -> Path:
        proj = make_truth_projection()
        proj.update(**overrides)
        tmpdir = Path(tempfile.mkdtemp())
        self.addCleanup(_rm_rf, tmpdir)
        calib_path = tmpdir / "calibration.json"
        proj.save(calib_path)
        return calib_path

    def test_report_runs_and_includes_sections(self):
        samples_path = self._write_samples()
        calib_path = self._write_calibration(pitch_deg=18.0)
        cfg_path = Path(tempfile.mkstemp(suffix=".cfg")[1])
        cfg_path.write_text("sensorPosition 1.85 0 15\n", encoding="utf-8")
        self.addCleanup(cfg_path.unlink, missing_ok=True)

        args = SimpleNamespace(
            samples=str(samples_path),
            calibration=str(calib_path),
            cfg=str(cfg_path),
            frame_w=FRAME_W,
            frame_h=FRAME_H,
            write_json=True,
        )

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = calibration_diag.cmd_report(args)
        out = buf.getvalue()

        self.assertEqual(rc, 0)
        self.assertIn("BEFORE", out)
        self.assertIn("AFTER", out)
        self.assertIn("Physical sanity", out)
        self.assertIn("Radar cfg sanity", out)
        self.assertIn("Interpretation", out)
        # The re-solve should substantially reduce the median pixel error
        # because the samples are noise-free and we only perturbed pitch.
        self.assertIn("before", out)
        self.assertIn("after ", out)

        report_json = samples_path.with_name("calibration_report.json")
        self.assertTrue(report_json.exists())
        payload = json.loads(report_json.read_text(encoding="utf-8"))
        self.assertGreater(payload["before"]["aggregate"]["median_px"], 20.0)
        self.assertLess(payload["after"]["aggregate"]["median_px"], 2.0)

    def test_eval_command_prints_aggregate(self):
        samples_path = self._write_samples()
        calib_path = self._write_calibration()
        args = SimpleNamespace(
            samples=str(samples_path),
            calibration=str(calib_path),
            frame_w=FRAME_W,
            frame_h=FRAME_H,
            write_json=False,
            per_sample=True,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = calibration_diag.cmd_eval(args)
        out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("Accuracy", out)
        self.assertIn("median", out)
        self.assertIn("Per-sample breakdown", out)


def _rm_rf(path: Path) -> None:
    if not path.exists():
        return
    if path.is_file():
        path.unlink(missing_ok=True)
        return
    for child in path.iterdir():
        _rm_rf(child)
    path.rmdir()


if __name__ == "__main__":
    unittest.main()
