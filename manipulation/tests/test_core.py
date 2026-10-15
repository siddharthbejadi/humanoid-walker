"""Checks for the claims the README makes about the geometry, the gate, the filter and the robot driver.   python -m unittest discover -s tests"""
import sys, time, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from armcopy.pose import arm_vectors, forward_fractions, RS, RE, RW
from armcopy.filters import OneEuro


def person(shoulder=(300, 200), elbow=(300, 290), wrist=(300, 370), left_shoulder=(380, 200), score=0.3):
    kp = np.zeros((17, 3), np.float32)
    for i, p in ((RS, shoulder), (RE, elbow), (RW, wrist), (5, left_shoulder)): kp[i] = (*p, score)
    return kp


class Gate(unittest.TestCase):
    def test_accepts_a_plausible_hanging_arm(self):
        self.assertIsNotNone(arm_vectors(person(), size=(640, 480)))

    def test_rejects_low_confidence(self):
        self.assertIsNone(arm_vectors(person(score=0.01), size=(640, 480)))

    def test_rejects_keypoints_on_the_image_border(self):             # a cut-off body gives junk on the edge
        self.assertIsNone(arm_vectors(person(elbow=(300, 470)), size=(640, 480)))

    def test_rejects_a_person_too_close_to_the_lens(self):
        self.assertIsNone(arm_vectors(person(shoulder=(100, 200), left_shoulder=(400, 200)), size=(640, 480)))

    def test_rejects_unlevel_shoulders(self):
        self.assertIsNone(arm_vectors(person(left_shoulder=(380, 300)), size=(640, 480)))

    def test_an_outstretched_wrist_at_the_border_is_allowed(self):    # only shoulders/elbow are edge-checked
        self.assertIsNotNone(arm_vectors(person(elbow=(200, 205), wrist=(20, 210)), size=(640, 480)))


class DepthCue(unittest.TestCase):
    def test_arm_in_the_image_plane_has_no_forward_component(self):
        fu, ff = forward_fractions(person())
        self.assertEqual((fu, ff), (0.0, 0.0))

    def test_arm_pointing_at_the_camera_looks_short(self):
        fu, _ = forward_fractions(person(elbow=(300, 215)))           # upper arm projected to 15 px instead of ~90
        self.assertGreater(fu, 0.8)


class Filter(unittest.TestCase):
    def test_removes_most_jitter_at_a_bounded_lag_cost(self):
        rng = np.random.default_rng(0); t = np.arange(300) / 15.0; clean = 0.8 * np.sin(2 * np.pi * 0.15 * t); noisy = clean + rng.normal(0, 0.05, t.size)
        f = OneEuro(0.7); out = np.array([f(x, tt) for x, tt in zip(noisy, t)])
        self.assertLess(np.abs(np.diff(out)).mean(), 0.5 * np.abs(np.diff(noisy)).mean())
        self.assertLess(np.abs(out - clean).mean(), 0.15)                 # the price of smoothing is lag: it must stay small next to the 0.8 rad swing


class Geometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from armcopy.sim import Humanoid, ArmIK3
        cls.robot = Humanoid(32, 32); cls.ik = ArmIK3(cls.robot)

    def test_ik_recovers_a_forward_raise(self):
        d = self.ik._dirs(np.array([1.2, self.ik.REST[1], 0.0])); u, f = d[:3], d[3:]
        self.ik.q = self.ik.REST.copy()
        q, (eu, ef) = self.ik.fit(u[:2], f[:2], float(abs(u[2])), float(abs(f[2])))
        self.assertLess(eu, 10); self.assertLess(ef, 10); self.assertAlmostEqual(q[0], 1.2, delta=0.15)

    def test_the_robot_renders_from_the_repo_layout(self):
        self.assertEqual(self.robot.render([0.3, 0.2, 0.1, 0.0]).shape, (32, 32, 3))


class RobotDriver(unittest.TestCase):
    def test_pulses_are_clamped_and_slew_limited_in_dry_run(self):
        from armcopy.robot import RightArm
        a = RightArm(live=False, max_us_per_s=200)
        try:
            a.set_angles(1.5, 1.5, 1.5, 1.39); time.sleep(0.5)
            for n, v in a.j.items():
                self.assertTrue(min(v["min_us"], v["max_us"]) - 1 <= a.cur[n] <= max(v["min_us"], v["max_us"]) + 1, n)
                self.assertLessEqual(abs(a.cur[n] - v["home_us"]), 200 * 0.6 + 2, n)       # 0.5 s at 200 us/s plus one tick
        finally:
            a.stop = True


if __name__ == "__main__":
    unittest.main()
