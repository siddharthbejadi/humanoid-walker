"""CPU regressions for measured numerical reset bounds and exact state guards."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reset_validation import validate_reset_repeatability


class ResetValidationTests(unittest.TestCase):
    def setUp(self):
        self.reference = dict(
            proprio=np.zeros((5, 45), np.float32),
            teacher_state=np.zeros((5, 98), np.float32),
            qpos=np.zeros((5, 26), np.float32), qvel=np.zeros((5, 25), np.float32),
            ctrl=np.zeros((5, 19), np.float32), time=np.zeros(5, np.float32),
            step_count=np.zeros(5, np.int32), pose_ids=np.arange(5, dtype=np.int32),
            state_rng=np.arange(10, dtype=np.uint32).reshape(5, 2),
            bridge_rng=np.asarray([123, 456], np.uint32))
        self.repeated = {key: value.copy() for key, value in self.reference.items()}
        self.retained = {key: self.reference[key].copy() for key in ("proprio", "teacher_state")}

    def check(self):
        return validate_reset_repeatability(
            self.reference, self.repeated, self.retained, np.arange(5))

    def test_identical_reset_passes(self):
        self.assertTrue(self.check()["passed"])

    def test_measured_gpu_differences_pass_scoped_bounds(self):
        self.repeated["proprio"][0, 19] = 2.1457672e-5
        self.repeated["proprio"][0, 28] = 1.7523766e-5
        self.repeated["proprio"][0, 37] = 3.8743019e-6
        self.repeated["teacher_state"][0, 86] = 1.0505319e-6
        self.repeated["teacher_state"][0, 90] = 4.3958426e-7
        report = self.check()
        self.assertTrue(report["passed"])
        self.assertGreater(report["observations"]["proprio"]["imu_accel"]["max_abs"], 1e-6)

    def test_wrong_pose_rejected_even_when_both_resets_match(self):
        self.reference["pose_ids"][1] = self.repeated["pose_ids"][1] = 4
        self.assertFalse(self.check()["passed"])

    def test_exact_reset_state_changes_rejected(self):
        for name in ("qpos", "qvel", "ctrl", "state_rng", "bridge_rng"):
            with self.subTest(name=name):
                saved = self.repeated[name].copy()
                self.repeated[name].flat[0] += 1
                self.assertFalse(self.check()["passed"])
                self.repeated[name] = saved

    def test_tiny_control_change_is_still_rejected(self):
        self.repeated["ctrl"][0, 0] = 1e-8
        self.assertFalse(self.check()["passed"])

    def test_excessive_acceleration_rejected(self):
        for channel in (19, 21, 28, 30):
            with self.subTest(channel=channel):
                self.repeated["proprio"].fill(0)
                self.repeated["proprio"][0, channel] = 1.01e-3
                self.assertFalse(self.check()["passed"])

    def test_other_observation_channels_keep_1e6_limit(self):
        for key, channels in (("proprio", (0, 18, 22, 27, 31, 34, 36)),
                              ("teacher_state", (0, 19, 38, 57, 76, 82, 84, 88))):
            for channel in channels:
                with self.subTest(key=key, channel=channel):
                    self.repeated[key].fill(0)
                    self.repeated[key][0, channel] = 2e-6
                    self.assertFalse(self.check()["passed"])
            self.repeated[key].fill(0)

    def test_excessive_contact_difference_rejected(self):
        for key, channel in (("proprio", 37), ("teacher_state", 86), ("teacher_state", 90)):
            with self.subTest(key=key, channel=channel):
                self.repeated[key][0, channel] = 1.01e-4 if key == "proprio" else 1.01e-5
                self.assertFalse(self.check()["passed"])
                self.repeated[key].fill(0)

    def test_filtered_pressure_bound_matches_normalization_gain(self):
        self.repeated["proprio"][0, 37] = 1.10434e-5
        self.repeated["teacher_state"][0, 90] = 1.13622e-6
        self.assertTrue(self.check()["passed"])

    def test_fresh_gpu_process_acceleration_measurement(self):
        self.repeated["proprio"][0, 19] = 1.20163e-4
        self.repeated["proprio"][0, 28] = 1.25886e-4
        self.assertTrue(self.check()["passed"])

    def test_retained_buffer_mutation_rejected_even_inside_numeric_bound(self):
        self.retained["proprio"][0, 19] = 1e-8
        self.assertFalse(self.check()["passed"])

    def test_nonfinite_rejected(self):
        for key in ("proprio", "teacher_state", "qpos", "ctrl"):
            for bad in (np.nan, np.inf):
                with self.subTest(key=key, bad=bad):
                    self.repeated[key].flat[0] = bad
                    self.assertFalse(self.check()["passed"])
                    self.repeated[key].flat[0] = 0

    def test_wrong_shape_or_missing_field_rejected(self):
        self.repeated["proprio"] = np.zeros((5, 44))
        self.assertFalse(self.check()["passed"])
        self.repeated["proprio"] = self.reference["proprio"].copy()
        del self.repeated["bridge_rng"]
        self.assertFalse(self.check()["passed"])

    def test_nonzero_reset_clock_rejected_even_when_both_match(self):
        for field in ("time", "step_count"):
            with self.subTest(field=field):
                self.reference[field][0] = self.repeated[field][0] = 1
                self.assertFalse(self.check()["passed"])
                self.reference[field][0] = self.repeated[field][0] = 0


if __name__ == "__main__":
    unittest.main()
