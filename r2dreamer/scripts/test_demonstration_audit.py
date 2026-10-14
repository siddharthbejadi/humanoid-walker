"""Synthetic CPU safety/trajectory checks; no simulator or walking experiment."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from demonstration_audit import validate_trajectory_contract
from envs.mjx_teacher_env import MJXTeacherEnv
from imitation_data import align_episode, validate_arrays


def trajectory_fixture(lengths=(5, 3, 7), horizon=7):
    episodes, steps = len(lengths), max(lengths)
    low, high = np.full(19, -.8, np.float32), np.full(19, 1.2, np.float32)
    center, scale = (low + high) / 2., (high - low) / 2.
    standing = np.full(19, .03, np.float32)
    contract = dict(control=dict(action_mode="full_command", actuator_ctrlrange_rad=np.stack((low, high), axis=1).tolist(),
        command_center_rad=center.tolist(), action_scale_rad=scale.tolist(), standing_ctrl_rad=standing.tolist(),
        control_dt_seconds=.02, action_repeat=10, slew_rad_per_second=8.),
        task=dict(episode_horizon_control_steps=horizon))
    arrays = dict(proprio=np.zeros((episodes, steps + 1, 45), np.float32),
        action=np.zeros((episodes, steps, 19), np.float32), desired_ctrl=np.zeros((episodes, steps, 19), np.float32),
        applied_ctrl=np.zeros((episodes, steps, 19), np.float32), state_ctrl=np.zeros((episodes, steps + 1, 19), np.float32),
        state_time=np.zeros((episodes, steps + 1), np.float32), state_step_count=np.zeros((episodes, steps + 1), np.int32),
        state_state_rng=np.zeros((episodes, steps + 1, 2), np.uint32), state_bridge_rng=np.zeros((episodes, steps + 1, 2), np.uint32),
        reward=np.zeros((episodes, steps), np.float32), is_terminal=np.zeros((episodes, steps), bool),
        valid=np.zeros((episodes, steps), bool), present=np.zeros((episodes, steps), bool), split=np.arange(episodes) % 3,
        **{"cause_" + name: np.zeros((episodes, steps), bool) for name in
           ("physical_fall", "wall_hit", "stall_out", "goal_reached", "time_limit")})
    for episode, length in enumerate(lengths):
        arrays["state_ctrl"][episode, 0] = standing
        arrays["state_state_rng"][episode, :length + 1] = [episode + 100, 200]
        arrays["state_bridge_rng"][episode, :length + 1, 0] = np.arange(length + 1) + 300
        arrays["present"][episode, :length] = arrays["valid"][episode, :length] = True
        for time in range(length):
            desired = center + .4
            if time == 0:
                desired = desired.copy()
                desired[0] = 2.  # Clipped labels are auditable, but not safe BC supervision.
                arrays["valid"][episode, time] = False
            action = np.clip((desired - center) / scale, -1., 1.)
            target = np.clip(center + scale * action, low, high)
            current = np.clip(target, arrays["state_ctrl"][episode, time] - .16,
                              arrays["state_ctrl"][episode, time] + .16)
            arrays["desired_ctrl"][episode, time], arrays["action"][episode, time] = desired, action
            arrays["state_ctrl"][episode, time + 1] = arrays["applied_ctrl"][episode, time] = current
            arrays["state_step_count"][episode, time + 1] = time + 1
            arrays["state_time"][episode, time + 1] = (time + 1) * .02
        arrays["proprio"][episode, :length + 1, :19] = np.clip(
            2. * (arrays["state_ctrl"][episode, :length + 1] - low) / (high - low) - 1., -1., 1.)
        if length == horizon:
            arrays["cause_time_limit"][episode, length - 1] = True
    if episodes > 1:
        arrays["is_terminal"][1, lengths[1] - 1] = arrays["cause_physical_fall"][1, lengths[1] - 1] = True
        arrays["cause_time_limit"][1] = False
        arrays["valid"][1] = False
    return arrays, contract


class DemonstrationAuditTests(unittest.TestCase):
    def setUp(self):
        self.arrays, self.contract = trajectory_fixture()

    def test_clipped_targets_slew_padding_and_terminal_identity_are_consistent(self):
        report = validate_trajectory_contract(self.arrays, self.contract)
        self.assertEqual(report["observed_transitions"], 15)
        self.assertLess(report["max_slew_error_rad"], 2e-6)
        self.assertIn("not physics or walking", report["scope"])
        json.dumps(report, allow_nan=False)

    def test_action_mapping_and_applied_state_disagreements_are_rejected(self):
        for name in ("action", "applied_ctrl"):
            arrays = deepcopy(self.arrays)
            arrays[name][0, 1, 0] += .03
            with self.subTest(field=name), self.assertRaisesRegex(ValueError, "mapping and slew"):
                validate_trajectory_contract(arrays, self.contract)

    def test_coherently_changed_state_applied_and_observation_still_fails_slew(self):
        arrays = deepcopy(self.arrays)
        arrays["state_ctrl"][0, 1, 0] += .02
        arrays["applied_ctrl"][0, 0, 0] = arrays["state_ctrl"][0, 1, 0]
        arrays["proprio"][0, 1, 0] += .02
        with self.assertRaisesRegex(ValueError, "mapping and slew"):
            validate_trajectory_contract(arrays, self.contract)

    def test_reset_control_and_observation_commands_are_not_inferred_from_targets(self):
        for name in ("state_ctrl", "proprio"):
            arrays = deepcopy(self.arrays)
            arrays[name][0, 0, 0] += .02
            with self.subTest(field=name), self.assertRaises(ValueError):
                validate_trajectory_contract(arrays, self.contract)

    def test_internal_reset_skipped_step_and_rng_identity_changes_are_rejected(self):
        for name in ("state_step_count", "state_state_rng"):
            arrays = deepcopy(self.arrays)
            arrays[name][0, 2] = 0
            with self.subTest(field=name), self.assertRaisesRegex(ValueError, "reset|identity"):
                validate_trajectory_contract(arrays, self.contract)

    def test_clock_zero_substep_double_step_and_backward_time_are_rejected(self):
        for value in (.002, .04, -.02):
            arrays = deepcopy(self.arrays)
            arrays["state_time"][0, 1] = value
            with self.subTest(time=value), self.assertRaisesRegex(ValueError, "clock"):
                validate_trajectory_contract(arrays, self.contract)
        arrays = deepcopy(self.arrays)
        arrays["state_time"][0, 0] = .01
        with self.assertRaisesRegex(ValueError, "clock"):
            validate_trajectory_contract(arrays, self.contract)

    def test_float32_two_millisecond_accumulation_is_not_a_false_clock_failure(self):
        arrays, contract = trajectory_fixture((1000,), horizon=1000)
        time = np.float32(0.)
        for control_step in range(1000):
            for _ in range(10):
                time = np.float32(time + np.float32(.002))
            arrays["state_time"][0, control_step + 1] = time
        report = validate_trajectory_contract(arrays, contract)
        self.assertLess(report["max_clock_increment_error_s"], report["time_tolerance_max_s"])

    def test_terminal_causes_and_time_limit_match_the_final_observed_boundary(self):
        for name in ("is_terminal", "cause_physical_fall", "cause_time_limit"):
            arrays = deepcopy(self.arrays)
            arrays[name][0, 0] = True
            with self.subTest(field=name), self.assertRaises(ValueError):
                validate_trajectory_contract(arrays, self.contract)
        arrays = deepcopy(self.arrays)
        arrays["cause_time_limit"][2, -1] = False
        with self.assertRaisesRegex(ValueError, "time-limit"):
            validate_trajectory_contract(arrays, self.contract)

    def test_terminal_padding_and_nonbinary_causes_are_rejected(self):
        arrays = deepcopy(self.arrays)
        arrays["is_terminal"][1, -1] = True
        with self.assertRaisesRegex(ValueError, "final observed"):
            validate_arrays(arrays)
        arrays = deepcopy(self.arrays)
        arrays["cause_wall_hit"] = arrays["cause_wall_hit"].astype(np.float32)
        arrays["cause_wall_hit"][0, 0] = 2.
        with self.assertRaisesRegex(ValueError, "binary"):
            validate_trajectory_contract(arrays, self.contract)

    def test_missing_malformed_floating_identity_and_nonfinite_arrays_fail_closed(self):
        variants = []
        missing = deepcopy(self.arrays)
        del missing["state_step_count"]
        variants.append(missing)
        malformed = deepcopy(self.arrays)
        malformed["state_ctrl"] = malformed["state_ctrl"][..., :18]
        variants.append(malformed)
        floating = deepcopy(self.arrays)
        floating["state_step_count"] = floating["state_step_count"].astype(np.float32)
        variants.append(floating)
        nonfinite = deepcopy(self.arrays)
        nonfinite["desired_ctrl"][0, 0, 0] = np.nan
        variants.append(nonfinite)
        for arrays in variants:
            with self.assertRaises(ValueError):
                validate_trajectory_contract(arrays, self.contract)

    def test_next_float_outside_action_limits_is_not_a_valid_expert_label(self):
        for limit in (np.nextafter(np.float32(1.), np.float32(np.inf)),
                      np.nextafter(np.float32(-1.), np.float32(-np.inf))):
            arrays = deepcopy(self.arrays)
            arrays["action"][0, 0, 0] = limit
            with self.assertRaisesRegex(ValueError, "normalized full motor"):
                validate_arrays(arrays)

    def test_window_bounds_keep_terminal_state_but_reject_fractional_and_empty_starts(self):
        data = align_episode(self.arrays, 1, start=3, length=1)
        self.assertTrue(data["is_terminal"][0, 0])
        self.assertFalse(data["expert_mask"].any())
        for kwargs in ({"episode": True}, {"episode": 3}, {"episode": 1, "start": 4},
                       {"episode": 0, "start": 1.5}, {"episode": 0, "length": 2.5},
                       {"episode": 0, "length": 2, "burnin": 2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                align_episode(self.arrays, **kwargs)


class FullCommandBoundaryTests(unittest.TestCase):
    def boundary(self, full_command):
        env = MJXTeacherEnv.__new__(MJXTeacherEnv)
        env.full_command, env.env_num, env.device, env.rng = full_command, 2, torch.device("cpu"), 123
        calls = []

        class PhysicsBoundary(Exception):
            pass

        def split(*unused):
            calls.append("rng_or_physics_reached")
            raise PhysicsBoundary

        env.jax = NS(random=NS(split=split))
        return env, calls, PhysicsBoundary

    def test_invalid_full_commands_fail_before_rng_or_physics(self):
        for value in (float("nan"), float("inf"), -float("inf"), 1.00001, -1.00001):
            env, calls, _ = self.boundary(True)
            action = torch.zeros(2, 19)
            action[0, 0] = value
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, "Full-command motor action must be finite and in"):
                env.step(action, torch.ones(2, dtype=torch.bool))
            self.assertEqual(calls, [])
            self.assertEqual(env.rng, 123)

    def test_all_finite_in_range_commands_and_legacy_inputs_keep_the_old_path(self):
        for full, value in ((True, -1.), (True, 0.), (True, 1.), (False, float("nan")), (False, 2.)):
            env, calls, boundary = self.boundary(full)
            with self.subTest(full=full, value=value), self.assertRaises(boundary):
                env.step(torch.full((2, 19), value), torch.ones(2, dtype=torch.bool))
            self.assertEqual(calls, ["rng_or_physics_reached"])


if __name__ == "__main__":
    unittest.main()
