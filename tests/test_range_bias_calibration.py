"""Tests for range_bias_calibration.py."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import range_bias_calibration as rbc


SAMPLE_CFG = """\
sensorStop
flushCfg
dfeDataOutputMode 1
channelCfg 15 7 0
adcCfg 2 1
profileCfg 0 60.75 30.00 25.00 59.10 657930 0 54.71 1 96 2950.00 2 1 36
chirpCfg 0 0 0 0 0 0 0 1
chirpCfg 1 1 0 0 0 0 0 2
chirpCfg 2 2 0 0 0 0 0 4
frameCfg 0 2 96 0 50.00 1 0
dynamicRACfarCfg -1 4 4 2 2 8 12 4 8 5.00 8.00 0.40 1 1
staticRACfarCfg -1 6 2 2 2 8 8 6 4 8.00 15.00 0.30 0 0
antGeometry0 0 -1 -2 -3 -2 -3 -4 -5 -4 -5 -6 -7
antGeometry1 -1 -1 -1 -1 0 0 0 0 -1 -1 -1 -1
antPhaseRot 1 1 1 1 1 1 1 1 1 1 1 1
fovCfg -1 70.0 15.0
compRangeBiasAndRxChanPhase 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0
staticBoundaryBox -3 3 0.5 5.5 0 3
boundaryBox -4 4 0 6.0 0 3
sensorPosition 1.85 0 30
gatingParam 3 2 2 2 12
stateParam 3 3 12 500 5 6000
allocationParam 60 200 0.1 5 1.5 2
maxAcceleration 1 1 1
trackingCfg 1 2 800 30 46 96 50
presenceBoundaryBox -3 3 0.5 5.5 0 3
sensorStart
"""


class ComputeBiasTests(unittest.TestCase):
    def test_single_pair_returns_exact_bias(self):
        bias, slope, intercept = rbc.compute_bias([2.0], [2.05])
        self.assertAlmostEqual(bias, 0.05, places=6)
        self.assertAlmostEqual(slope, 1.0, places=6)

    def test_multiple_pairs_computes_mean_bias(self):
        known = [1.0, 2.0, 3.0, 5.0]
        # All measurements shifted by exactly 0.04 m.
        measured = [k + 0.04 for k in known]
        bias, slope, intercept = rbc.compute_bias(known, measured)
        self.assertAlmostEqual(bias, 0.04, places=5)

    def test_returns_bias_within_tolerance(self):
        known = [1.0, 2.0, 3.0]
        # Small random offsets around 0.06 m bias.
        measured = [1.058, 2.063, 3.059]
        bias, _, _ = rbc.compute_bias(known, measured)
        self.assertAlmostEqual(bias, 0.06, delta=0.005)

    def test_scale_factor_detected_when_slope_off(self):
        # slope = 1.03 > 1.02 threshold — caller should warn
        known = [1.0, 2.0, 3.0, 4.0]
        measured = [k * 1.03 for k in known]
        _, slope, _ = rbc.compute_bias(known, measured)
        self.assertGreater(abs(slope - 1.0), 0.02)


class BuildCalCfgTests(unittest.TestCase):
    def _write_cfg(self, text: str) -> Path:
        tmp = Path(tempfile.mkstemp(suffix=".cfg")[1])
        tmp.write_text(text, encoding="utf-8")
        self.addCleanup(tmp.unlink, missing_ok=True)
        return tmp

    def test_static_output_enabled(self):
        path = self._write_cfg(SAMPLE_CFG)
        result = rbc._build_cal_cfg(path)
        # staticRACfarCfg second-to-last field must be "1" (staticOutputEn).
        for line in result.splitlines():
            stripped = line.strip()
            if stripped.startswith("staticRACfarCfg") and not stripped.startswith("%"):
                parts = stripped.split()
                self.assertEqual(
                    parts[-2], "1",
                    msg="staticOutputEn should be 1 in calibration cfg",
                )
                self.assertEqual(
                    parts[-1], "0",
                    msg="staticClutterRemovalEn should remain 0",
                )
                return
        self.fail("staticRACfarCfg line not found in calibration cfg")

    def test_boundary_boxes_widened(self):
        path = self._write_cfg(SAMPLE_CFG)
        result = rbc._build_cal_cfg(path)
        for line in result.splitlines():
            stripped = line.strip()
            for key in ("boundaryBox", "staticBoundaryBox", "presenceBoundaryBox"):
                if stripped.startswith(key) and not stripped.startswith("%"):
                    parts = stripped.split()
                    # x_min = -8, x_max = 8, y_min = 0.1, y_max = 9.9
                    self.assertAlmostEqual(float(parts[1]), -8.0, places=5)
                    self.assertAlmostEqual(float(parts[2]),  8.0, places=5)
                    self.assertAlmostEqual(float(parts[4]),  9.9, places=5)

    def test_range_bias_reset_to_zero(self):
        cfg_with_bias = SAMPLE_CFG.replace(
            "compRangeBiasAndRxChanPhase 0 1 0",
            "compRangeBiasAndRxChanPhase 0.07 1 0",
        )
        path = self._write_cfg(cfg_with_bias)
        result = rbc._build_cal_cfg(path)
        for line in result.splitlines():
            stripped = line.strip()
            if stripped.startswith("compRangeBiasAndRxChanPhase") and not stripped.startswith("%"):
                bias = float(stripped.split()[1])
                self.assertAlmostEqual(bias, 0.0, places=6,
                                       msg="compRangeBiasAndRxChanPhase bias must be reset to 0")
                return
        self.fail("compRangeBiasAndRxChanPhase not found in calibration cfg")


class PatchCfgFileTests(unittest.TestCase):
    def _write_cfg(self, text: str) -> Path:
        tmp = Path(tempfile.mkstemp(suffix=".cfg")[1])
        tmp.write_text(text, encoding="utf-8")
        self.addCleanup(tmp.unlink, missing_ok=True)
        return tmp

    def test_bias_value_updated_other_lines_unchanged(self):
        path = self._write_cfg(SAMPLE_CFG)
        ok = rbc.patch_cfg_file(path, 0.06, dry_run=False)
        self.assertTrue(ok)
        text = path.read_text(encoding="utf-8")
        # New bias present.
        self.assertIn("compRangeBiasAndRxChanPhase 0.06000", text)
        # Phase values (1 0 1 0 ...) preserved.
        self.assertIn("1 0 1 0 1 0 1 0", text)
        # Unrelated lines unchanged.
        self.assertIn("sensorPosition 1.85 0 30", text)
        self.assertIn("trackingCfg 1 2 800 30 46 96 50", text)

    def test_dry_run_does_not_modify_file(self):
        path = self._write_cfg(SAMPLE_CFG)
        original = path.read_text(encoding="utf-8")
        rbc.patch_cfg_file(path, 0.06, dry_run=True)
        self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_missing_cfg_returns_false(self):
        ok = rbc.patch_cfg_file(Path("/nonexistent/file.cfg"), 0.05, dry_run=False)
        self.assertFalse(ok)

    def test_missing_key_returns_false(self):
        path = self._write_cfg("sensorPosition 1.85 0 30\n")
        ok = rbc.patch_cfg_file(path, 0.05, dry_run=False)
        self.assertFalse(ok)

    def test_negative_bias_formatted_correctly(self):
        path = self._write_cfg(SAMPLE_CFG)
        rbc.patch_cfg_file(path, -0.02, dry_run=False)
        text = path.read_text(encoding="utf-8")
        self.assertIn("compRangeBiasAndRxChanPhase -0.02000", text)


if __name__ == "__main__":
    unittest.main()
