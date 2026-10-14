"""Script-free command, observation and reward regressions without MJX-Warp.

FULL_COMMAND_TEST_JAX=1 enables CPU jit/vmap and exact actuation/reset prefixes.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from full_command import motor_targets_from_actions, normalize_motor_targets, validate_script_free_settings

ENV_PATH = ROOT / "envs/mjx_walking/env.py"
BRIDGE_PATH = ROOT / "envs/mjx_teacher_env.py"
TREE = ast.parse(ENV_PATH.read_text(encoding="utf-8"))
ENV = next(node for node in TREE.body if isinstance(node, ast.ClassDef) and node.name == "MJXWalkingEnv")


def method(name):
    return next(node for node in ENV.body if isinstance(node, ast.FunctionDef) and node.name == name)


def namespace(array_module=np):
    helpers = [node for node in TREE.body if isinstance(node, ast.FunctionDef)
               and node.name in {"autonomous_walk_reward_terms", "foot_swing_qualification", "alternating_step_events"}]
    result = {"jnp": array_module, "np": np, "Path": Path,
              "motor_targets_from_actions": motor_targets_from_actions,
              "validate_script_free_settings": validate_script_free_settings}
    exec(compile(ast.Module(body=helpers, type_ignores=[]), str(ENV_PATH), "exec"), result)
    return result


def reward(**overrides):
    values = dict(velocity=np.float32(.08), lateral_velocity=np.float32(0.), tilt=np.float32(0.),
        height_error=np.float32(0.), load=np.asarray([.7, .3]), foot_speed_sq=np.zeros(2),
        action=np.zeros(19), action_delta=np.zeros(19), fallen=np.bool_(False), dt=.02, target_vx=.08)
    values.update(overrides)
    return namespace()["autonomous_walk_reward_terms"](**values)


class FullCommandTests(unittest.TestCase):
    def test_absolute_control_range_roundtrip(self):
        low = np.linspace(-1.5, -.2, 19)
        high = np.linspace(.3, 1.5, 19)
        center, scale = (low + high) / 2., (high - low) / 2.
        for target, expected in ((low, -1.), (center, 0.), (high, 1.)):
            action = normalize_motor_targets(target, center, scale)
            np.testing.assert_allclose(action, expected)
            np.testing.assert_allclose(motor_targets_from_actions(action, center, scale), target)

    def test_out_of_range_actions_are_clipped_before_target_mapping(self):
        np.testing.assert_array_equal(motor_targets_from_actions(np.asarray([-2., 2., np.nan]),
            np.zeros(3), np.ones(3)), [-1., 1., 0.])

    def test_full_mode_rejects_every_reference_before_physics(self):
        initialize = method("__init__")

        class PhysicsSentinel(Exception):
            pass

        def build(*args, **kwargs):
            raise PhysicsSentinel

        ns = namespace()
        ns["build_mjx_scene"] = build
        exec(compile(ast.Module(body=[initialize], type_ignores=[]), str(ENV_PATH), "exec"), ns)
        settings = dict(full_command=True, rsi_stratified=False, reward_profile="autonomous_walk_v1",
                        reset_joint_noise_rad=.005, reset_velocity_noise=.005)
        with self.assertRaises(PhysicsSentinel):
            ns["__init__"](SimpleNamespace(), "unused.xml", **settings)
        for key in ("scripted_gait", "scripted_phase_pool", "pose_rsi_json", "gait_reference_csv", "imitation_csv"):
            with self.subTest(reference=key), self.assertRaisesRegex(ValueError, "null gait"):
                ns["__init__"](SimpleNamespace(), "unused.xml", **settings, **{key: "missing_poison_asset.npz"})
        for key, value in (("stabilizer", [1., .04, -1., .25]), ("gait_reference_signs", [1.] * 19),
                           ("phase_obs", True), ("phase_control", .5), ("gait_residual", True),
                           ("gait_curriculum", True), ("footsteps", {})):
            with self.subTest(controller=key), self.assertRaises(ValueError):
                ns["__init__"](SimpleNamespace(), "unused.xml", **settings, **{key: value})
        with self.assertRaises(ValueError):
            ns["__init__"](SimpleNamespace(), "unused.xml", **{**settings, "rsi_stratified": True})

    def test_noise_bounds_reject_before_physics(self):
        ns = namespace()
        ns["build_mjx_scene"] = lambda *args, **kwargs: self.fail("Physics should not be constructed")
        exec(compile(ast.Module(body=[method("__init__")], type_ignores=[]), str(ENV_PATH), "exec"), ns)
        for field, value in (("reset_joint_noise_rad", .031), ("reset_velocity_noise", .051),
                             ("reset_joint_noise_rad", float("nan")), ("reset_velocity_noise", -.001)):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                ns["__init__"](SimpleNamespace(), "unused.xml", full_command=True,
                    rsi_stratified=False, reward_profile="autonomous_walk_v1", **{field: value})

    def test_sensor_observation_has_no_phase_or_navigation_remapping(self):
        ns = namespace()
        exec(compile(ast.Module(body=[method("_observation")], type_ignores=[]), str(ENV_PATH), "exec"), ns)
        sensors = {name: np.full(3, i + 1., dtype=np.float32) for i, name in enumerate(
            ("imu_accel", "imu_gyro", "imu_mag", "onboard_accel", "onboard_gyro"))}
        env = SimpleNamespace(full_command=True, phase_rate_range=0., phase_obs=False,
            task="walking", reward_profile="autonomous_walk_v1", target_forward_speed=.08,
            _sensor=lambda data, name: sensors[name])
        obs = ns["_observation"](env, None, np.zeros(19), np.zeros(8), np.int32(999),
                                  phase=object(), nav=object())
        self.assertEqual(obs.shape, (45,))
        np.testing.assert_array_equal(obs[28:31], sensors["onboard_accel"])
        np.testing.assert_array_equal(obs[31:34], sensors["onboard_gyro"])
        np.testing.assert_allclose(obs[34:37], [.08, 0., 0.])

    def test_bridge_returns_no_teacher_state_for_student(self):
        bridge = ast.parse(BRIDGE_PATH.read_text(encoding="utf-8"))
        cls = next(node for node in bridge.body if isinstance(node, ast.ClassDef) and node.name == "MJXTeacherEnv")
        step = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "step")
        result_index = next(i for i, node in enumerate(step.body) if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "result" for target in node.targets))
        ns = dict(self=SimpleNamespace(full_command=True), packed=np.zeros((2, 45)), reward=np.zeros(2),
                  first=np.ones(2, bool), last=np.zeros(2, bool), terminal=np.zeros(2, bool))
        exec(compile(ast.Module(body=step.body[result_index:result_index + 2], type_ignores=[]), str(BRIDGE_PATH), "exec"), ns)
        self.assertNotIn("teacher_state", ns["result"])
        self.assertEqual(ns["result"]["proprio"].shape, (2, 45))

    def test_reward_prefers_supported_forward_motion_to_standing(self):
        moving, _, failed = reward()
        standing, _, _ = reward(velocity=np.float32(0.))
        self.assertGreater(float(moving), float(standing))
        self.assertFalse(failed)
        backward, _, _ = reward(velocity=np.float32(-.08))
        self.assertLessEqual(float(backward), float(standing))

    def test_reward_fall_and_bad_height_stop_income(self):
        for values in (dict(fallen=np.bool_(True)), dict(height_error=np.float32(-.11)),
                       dict(velocity=np.float32(np.nan))):
            with self.subTest(values=values):
                value, terms, failed = reward(**values)
                self.assertTrue(failed)
                self.assertLessEqual(float(value), -3.)
                self.assertEqual(float(terms["reward_forward"]), 0.)

    def test_step_clearance_does_not_become_a_reward_lookup(self):
        _, terms, _ = reward()
        self.assertFalse(any("track" in key or "dot" in key or "reference" in key for key in terms))
        gate = namespace()["foot_swing_qualification"](clearance=np.asarray([0., .015]),
            load=np.asarray([.8, 0.]), stable_support=True, foot_speed_sq=np.zeros(2), step_clearance_m=.015)
        np.testing.assert_array_equal(gate, [False, True])

    def test_autonomous_branch_counts_ordered_alternation(self):
        branch = next(node for node in ast.walk(method("step")) if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare) and any(isinstance(value, ast.Constant)
                and value.value == "autonomous_walk_v1" for value in node.test.comparators))
        code = compile(ast.Module(body=branch.body, type_ignores=[]), str(ENV_PATH), "exec")
        for sequence, expected in (("LLLRRR", 1), ("LRLR", 3), ("RRRR", 0)):
            ns = {**namespace(), "self": SimpleNamespace(control_dt=.02, target_forward_speed=.08, standing_height=.293478),
                "state": SimpleNamespace(info={"step_count": np.int32(1), "last_action": np.zeros(19)}),
                "data": SimpleNamespace(qvel=np.zeros(25)), "action": np.zeros(19),
                "velocity": np.float32(.08), "tilt": np.float32(.01), "root_height": np.float32(.293478),
                "load": np.asarray([.7, .3]), "foot_speed_sq": np.zeros(2), "fallen": np.bool_(False),
                "last_step_side": np.int32(-1), "alternating_steps": np.float32(0.)}
            for side in sequence:
                ns["touched"] = np.asarray([side == "L", side == "R"])
                exec(code, ns)
            with self.subTest(sequence=sequence):
                self.assertEqual(float(ns["alternating_steps"]), expected)

    def test_bridge_uses_ordered_counter_delta_for_autonomous_profile(self):
        import torch

        tree = ast.parse(BRIDGE_PATH.read_text(encoding="utf-8"))
        bridge = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MJXTeacherEnv")
        step = next(node for node in bridge.body if isinstance(node, ast.FunctionDef) and node.name == "step")
        branch = next(node for node in step.body if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Attribute)
            and node.test.left.attr == "reward_profile")
        delta = next(node for node in bridge.body if isinstance(node, ast.FunctionDef)
                     and node.name == "_standing_seconds_delta")
        ns = {"torch": torch}
        exec(compile(ast.Module(body=[delta], type_ignores=[]), str(BRIDGE_PATH), "exec"), ns)
        delta_fn = ns["_standing_seconds_delta"].__func__
        ns.update(self=SimpleNamespace(raw=SimpleNamespace(reward_profile="autonomous_walk_v1"),
            state=SimpleNamespace(info={"alternating_steps": torch.tensor([0., 2.])}),
            _standing_seconds_delta=delta_fn), old_state=SimpleNamespace(info={"alternating_steps": torch.tensor([3., 1.])}),
            first=torch.tensor([True, False]), current_scrub=torch.zeros(2),
            gpu=lambda value: torch.as_tensor(value).float(), episode_logs={"alternating_steps": torch.tensor([99., 99.])})
        exec(compile(ast.Module(body=[branch], type_ignores=[]), str(BRIDGE_PATH), "exec"), ns)
        np.testing.assert_array_equal(ns["episode_logs"]["alternating_steps"].numpy(), [0., 1.])

    def test_evaluation_protocol_declares_student_semantics_and_noise(self):
        tree = ast.parse(BRIDGE_PATH.read_text(encoding="utf-8"))
        bridge = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MJXTeacherEnv")
        fn = next(node for node in bridge.body if isinstance(node, ast.FunctionDef) and node.name == "evaluation_protocol")
        ns = {"np": np}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(BRIDGE_PATH), "exec"), ns)
        env = SimpleNamespace(full_command=True, env_num=4, _evaluation_seed=11000,
            _repeatable_evaluation=True, _balanced_evaluation=False, horizon=1000,
            raw=SimpleNamespace(control_dt=.02, step_clearance_m=.015, rsi_stratified=False,
                full_command=True, controller_identity="autonomous_full_command_v1",
                observation_semantics="sensor_command_imu_fsr_45_v1", reset_joint_noise_rad=.005,
                reset_velocity_noise=.005, target_forward_speed=.08, phase_obs=False,
                stabilizer=None, gait_curriculum=False, gait_reference_kind="none"))
        protocol = ns["evaluation_protocol"](env)
        self.assertTrue(protocol["full_command"])
        self.assertEqual(protocol["observation_keys"], ["proprio"])
        self.assertEqual(protocol["reset_joint_noise_rad"], .005)
        self.assertEqual(protocol["reset_velocity_noise"], .005)
        self.assertFalse(protocol["stabilizer_enabled"])
        self.assertFalse(protocol["phase_observations"])
        self.assertFalse(protocol["gait_curriculum"])

    @unittest.skipUnless(os.environ.get("FULL_COMMAND_TEST_JAX") == "1", "Optional CPU JAX checks")
    def test_cpu_jax_exact_actuation_and_bounded_reset_prefixes(self):
        os.environ["JAX_PLATFORMS"] = "cpu"
        import jax
        import jax.numpy as jnp

        ns = namespace(jnp)
        ns["jax"] = jax
        source_step = method("step")
        prefix = []
        for node in source_step.body:
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "replace"):
                break
            prefix.append(node)
        prefix.append(ast.Return(ast.Tuple([ast.Name("target", ast.Load()), ast.Name("ctrl", ast.Load())], ast.Load())))
        fn = ast.FunctionDef(name="control_prefix", args=source_step.args, body=prefix,
                             decorator_list=[], returns=None)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(ENV_PATH), "exec"), ns)

        def poisoned(*args, **kwargs):
            raise AssertionError("Student attempted to call a reference")

        env = SimpleNamespace(command_center=jnp.zeros(19), command_scale=jnp.full(19, 1.5708),
            phase_rate_range=0., gait_residual=False, stabilizer=None, low=jnp.full(19, -1.5708),
            high=jnp.full(19, 1.5708), control_slew_rad_s=8., control_dt=.02,
            _reference_at=poisoned, _residual_base=poisoned)
        state = SimpleNamespace(pipeline_state=SimpleNamespace(ctrl=jnp.zeros(19)),
            info={"gait_phase": jnp.float32(0.), "stab_prev": jnp.float32(0.), "stab_filt": jnp.float32(0.)})
        target, applied = jax.jit(lambda action: ns["control_prefix"](env, state, action))(jnp.ones(19))
        np.testing.assert_allclose(np.asarray(target), 1.5708, atol=1e-6)
        np.testing.assert_allclose(np.asarray(applied), .16, atol=1e-6)

        reset = method("reset")
        prefix = []
        for node in reset.body:
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "ctrl" for target in node.targets):
                break
            prefix.append(node)
        prefix.append(ast.Return(ast.Tuple([ast.Name("qpos", ast.Load()), ast.Name("qvel", ast.Load())], ast.Load())))
        fn = ast.FunctionDef(name="reset_noise_prefix", args=reset.args, body=prefix,
                             decorator_list=[], returns=None)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(ENV_PATH), "exec"), ns)
        initial = jnp.zeros(26).at[2].set(.293478).at[3].set(1.)
        reset_env = SimpleNamespace(full_command=True, initial_qpos=initial,
            act_qpos=jnp.arange(7, 26), act_dof=jnp.arange(6, 25), m=SimpleNamespace(nv=25),
            reset_joint_noise_rad=.005, reset_velocity_noise=.005)
        qpos, qvel = jax.jit(jax.vmap(lambda key: ns["reset_noise_prefix"](reset_env, key)))(
            jax.random.split(jax.random.PRNGKey(1), 4))
        np.testing.assert_array_equal(np.asarray(qpos[:, :7]), np.tile(np.asarray(initial[:7]), (4, 1)))
        np.testing.assert_array_equal(np.asarray(qvel[:, :6]), np.zeros((4, 6)))
        self.assertLessEqual(float(jnp.max(jnp.abs(qpos[:, 7:]))), .005)
        self.assertLessEqual(float(jnp.max(jnp.abs(qvel[:, 6:]))), .005)


if __name__ == "__main__":
    unittest.main()
