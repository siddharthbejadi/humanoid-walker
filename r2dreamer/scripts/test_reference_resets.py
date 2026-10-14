"""Reference reset regressions without loading Brax or the MJX-Warp environment.

Run: python scripts/test_reference_resets.py
Optional CPU JAX checks: REFERENCE_RESET_TEST_JAX=1 python scripts/test_reference_resets.py
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from full_command import motor_targets_from_actions

ENV_SOURCE = ROOT / "envs/mjx_walking/env.py"
SCENE_SOURCE = ROOT / "envs/mjx_walking/scene.py"
NAMES = {"reference_sample_phase", "residual_gait_target", "stabilizer_reset_state",
         "footstep_target_terms", "rsi_injection_flags", "build_scripted_rsi_pool",
         "alternating_step_events"}


def script_free_archive_only(root=ROOT):
    manifest = root.parent / "SOURCE_MANIFEST.json"
    if not manifest.is_file():
        return False
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    return isinstance(metadata, dict) and metadata.get("bundle") in {
        "r2dreamer_script_free_next_20261012", "r2dreamer_script_free_next_20261013_v2"}


def load_source(array_module):
    trees = [ast.parse(path.read_text(encoding="utf-8")) for path in (ENV_SOURCE, SCENE_SOURCE)]
    helpers = [node for tree in trees for node in tree.body
               if isinstance(node, ast.FunctionDef) and node.name in NAMES]
    if {node.name for node in helpers} != NAMES:
        raise RuntimeError("Reference reset helpers are missing")
    environment = next(node for node in trees[0].body
                       if isinstance(node, ast.ClassDef) and node.name == "MJXWalkingEnv")
    methods = [node for node in environment.body if isinstance(node, ast.FunctionDef)
               and node.name in {"_reference_at", "_residual_base", "_gait_index"}]
    harness = ast.ClassDef(name="ReferenceHarness", bases=[], keywords=[], body=methods, decorator_list=[])
    namespace = {"jnp": array_module, "np": np, "Path": Path}
    module = ast.fix_missing_locations(ast.Module(body=helpers + [harness], type_ignores=[]))
    exec(compile(module, str(ENV_SOURCE), "exec"), namespace)
    return namespace, environment


FUNCTIONS, ENVIRONMENT = load_source(np)


def fixture(namespace=FUNCTIONS, array_module=np):
    env = namespace["ReferenceHarness"]()
    env.gait_reference = array_module.asarray(np.tile(np.arange(400)[:, None] / 1000., (1, 19)))
    env.gait_reference_sample_offset = -1.
    env.control_dt = .02
    env.gait_cycle_start, env.gait_period = .08, .04
    env.INJECT_HOLD_S, env.INJECT_BLEND_S = 2., 1.
    return env


def rotation(roll):
    c, s = np.cos(roll), np.sin(roll)
    return np.asarray([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def source_expression(expression, namespace):
    return eval(compile(ast.Expression(expression), str(ENV_SOURCE), "eval"), namespace)


class ReferenceResetTests(unittest.TestCase):
    def test_scripted_pool_speed_and_velocity_guard_precedes_physics(self):
        constructor = next(node for node in ENVIRONMENT.body
                           if isinstance(node, ast.FunctionDef) and node.name == "__init__")

        class PhysicsSentinel(Exception):
            pass

        def stop_before_physics(*args, **kwargs):
            raise PhysicsSentinel

        namespace = {"np": np, "Path": Path, "build_mjx_scene": stop_before_physics}
        exec(compile(ast.Module(body=[constructor], type_ignores=[]), str(ENV_SOURCE), "exec"), namespace)
        initialize = namespace["__init__"]
        pool = "unused_phase_pool.npz"
        with self.assertRaises(PhysicsSentinel):
            initialize(SimpleNamespace(), "unused.xml", step_clearance_m=.015,
                       scripted_phase_pool=pool, gait_reference_speed=1., rsi_reference_velocity=True)
        for speed in (.5, 2., float("nan"), float("inf")):
            with self.subTest(speed=speed), self.assertRaisesRegex(ValueError, "Scripted phase pools require"):
                initialize(SimpleNamespace(), "unused.xml", scripted_phase_pool=pool,
                           gait_reference_speed=speed, rsi_reference_velocity=True)
        with self.assertRaisesRegex(ValueError, "recorded reference velocities"):
            initialize(SimpleNamespace(), "unused.xml", scripted_phase_pool=pool,
                       gait_reference_speed=1., rsi_reference_velocity=False)
        with self.assertRaises(PhysicsSentinel):
            initialize(SimpleNamespace(), "unused.xml", scripted_phase_pool=None,
                       gait_reference_speed=.5, rsi_reference_velocity=False)

    def test_scripted_phase_ids_never_activate_manual_injection(self):
        flags = FUNCTIONS["rsi_injection_flags"](np.asarray([0, 100, 125, 156]),
            manual_pool=False, injected_pose_min_id=8)
        np.testing.assert_array_equal(flags, [False] * 4)

    def test_authored_manual_injection_remains_explicit(self):
        flags = FUNCTIONS["rsi_injection_flags"](np.asarray([0, 1, 7, 8, 9]),
            manual_pool=True, injected_pose_min_id=8)
        np.testing.assert_array_equal(flags, [False, False, False, True, True])

    @unittest.skipIf(script_free_archive_only(), "Script-free archive excludes the unused residual RSI phase pool")
    def test_shipped_scripted_pool_has_standing_id_zero_and_separate_phase_ids(self):
        model = SimpleNamespace(nq=26, nv=25, nu=19,
                                actuator_ctrlrange=np.tile([-2., 2.], (19, 1)))
        pool = FUNCTIONS["build_scripted_rsi_pool"](model,
            ROOT / "envs/mjx_walking/assets/humanlike_phase_pool.npz")
        self.assertEqual(int(pool["pose_ids"][0]), 0)
        self.assertEqual(int(pool["counts"][0]), 1)
        self.assertEqual(int(pool["counts"][3]), 57)
        self.assertTrue(np.all(pool["pose_ids"][1:] >= 100))
        self.assertFalse(FUNCTIONS["rsi_injection_flags"](pool["pose_ids"],
            manual_pool=False, injected_pose_min_id=8).any())

    def test_scripted_sample_clock_continues_from_saved_phase(self):
        env = fixture()
        for saved_phase in (0, 5, 120, 205, 285):
            with self.subTest(phase=saved_phase):
                np.testing.assert_allclose(env._reference_at(np.float32(saved_phase + 1)), saved_phase / 1000.)
        np.testing.assert_array_equal(env._reference_at(np.float32(0.)), np.zeros(19))

    def test_vendor_clock_is_not_shifted(self):
        env = fixture()
        env.gait_reference_sample_offset = 0.
        np.testing.assert_allclose(env._reference_at(np.float32(121.)), .121)

    def test_ordinary_phase_reset_never_holds_saved_stabilized_ctrl(self):
        env = fixture()
        info = {"phase_offset": np.int32(120), "inject_ctrl": np.full(19, .5), "injected": np.float32(0.)}
        for local_step in (1, 50, 125):
            with self.subTest(step=local_step):
                target = env._residual_base(info, np.int32(local_step), np.float32(121.))
                np.testing.assert_allclose(target, .120)

    def test_manual_hold_and_blend_still_require_injected_flag(self):
        env = fixture()
        info = {"phase_offset": np.int32(120), "inject_ctrl": np.full(19, .5), "injected": np.float32(1.)}
        np.testing.assert_allclose(env._residual_base(info, np.int32(1), np.float32(121.)), .5)
        np.testing.assert_allclose(env._residual_base(info, np.int32(125), np.float32(121.)), .31)
        np.testing.assert_allclose(env._residual_base(info, np.int32(150), np.float32(121.)), .12)

    def test_reset_stabilizer_history_starts_at_actual_roll(self):
        roll, filtered = FUNCTIONS["stabilizer_reset_state"](rotation(.2))
        self.assertAlmostEqual(float(roll), .2)
        self.assertEqual(float(filtered), 0.)
        first_filtered = .8 * filtered + .2 * (.2 - roll) / .02
        self.assertAlmostEqual(float(first_filtered), 0.)

    def test_alternation_counts_ordered_changes(self):
        for sequence, expected in (("LLLRRR", 1), ("LRLR", 3), ("LLLL", 0), ("RRRR", 0)):
            last, count = np.int32(-1), 0
            for side in sequence:
                last, changed = FUNCTIONS["alternating_step_events"](
                    np.asarray([side == "L", side == "R"]), last)
                count += int(changed)
            with self.subTest(sequence=sequence):
                self.assertEqual(count, expected)

    def test_simultaneous_events_and_reset_do_not_create_alternation(self):
        fn = FUNCTIONS["alternating_step_events"]
        last, changed = fn(np.asarray([True, True]), np.int32(-1))
        self.assertEqual(int(last), -1)
        self.assertFalse(changed)
        last, changed = fn(np.asarray([True, False]), last)
        self.assertEqual(int(last), 0)
        self.assertFalse(changed)
        last, changed = fn(np.asarray([True, True]), last)
        self.assertEqual(int(last), 0)
        self.assertFalse(changed)
        last, changed = fn(np.asarray([False, False]), np.int32(-1))
        self.assertEqual(int(last), -1)
        self.assertFalse(changed)
        _, changed = fn(np.asarray([False, True]), last)
        self.assertFalse(changed)

    def test_actual_reset_storage_uses_explicit_pool_flag_and_fresh_history(self):
        reset = next(node for node in ENVIRONMENT.body if isinstance(node, ast.FunctionDef) and node.name == "reset")
        pool_assignment = [node for node in ast.walk(reset) if isinstance(node, ast.Assign)
                           and any(isinstance(target, ast.Name) and target.id == "pool_injected"
                                   for target in node.targets)][-1]
        namespace = {**FUNCTIONS, "self": SimpleNamespace(rsi_injected=np.asarray([False, False]), base_id=0),
                     "use_reference": np.bool_(True), "index": np.int32(1), "use_forced_pose": False,
                     "data": SimpleNamespace(xmat=np.asarray([rotation(.2)]))}
        namespace["pool_injected"] = source_expression(pool_assignment.value, namespace)
        self.assertFalse(namespace["pool_injected"])
        history_assignment = next(node for node in reset.body if isinstance(node, ast.Assign)
                                  and isinstance(node.value, ast.Call)
                                  and isinstance(node.value.func, ast.Name)
                                  and node.value.func.id == "stabilizer_reset_state")
        namespace["stab_prev"], namespace["stab_filt"] = source_expression(history_assignment.value, namespace)
        info = next(node.value for node in reset.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "info" for target in node.targets))
        keywords = {keyword.arg: keyword.value for keyword in info.keywords}
        self.assertEqual(float(source_expression(keywords["injected"], namespace)), 0.)
        self.assertAlmostEqual(float(source_expression(keywords["stab_prev"], namespace)), .2)
        self.assertEqual(float(source_expression(keywords["stab_filt"], namespace)), 0.)

    def test_ordinary_phase_start_receives_no_forced_dot(self):
        step = next(node for node in ENVIRONMENT.body if isinstance(node, ast.FunctionDef) and node.name == "step")
        dots_call = next(node for node in ast.walk(step) if isinstance(node, ast.Call)
                         and isinstance(node.func, ast.Name) and node.func.id == "footstep_target_terms")
        forcing = next(keyword.value for keyword in dots_call.keywords if keyword.arg == "force_side")
        namespace = {**FUNCTIONS, "state": SimpleNamespace(info={"injected": 0., "step_count": 0, "start_forward": 0.}),
                     "self": SimpleNamespace(standing_heel_forward=[0., 0.]), "heel_f": np.asarray([0., .05])}
        forced_side = source_expression(forcing, namespace)
        self.assertEqual(int(forced_side), -1)
        hit = FUNCTIONS["footstep_target_terms"](
            heel_forward=np.asarray([0., .05]), foot_lateral=np.asarray([.05, -.05]),
            landed=np.zeros(2, bool), alive=np.bool_(True), dot_index=np.int32(0), prev_dist=.05,
            anchor_forward=np.zeros(2), anchor_lateral=np.asarray([.05, -.05]),
            start_side=np.int32(-1), step=.05, radius=.025, force_side=forced_side)
        self.assertEqual(float(hit[0]), 0.)
        self.assertEqual(int(hit[2]), 0)

    @unittest.skipUnless(os.environ.get("REFERENCE_RESET_TEST_JAX") == "1", "Optional CPU JAX checks")
    def test_cpu_jax_helpers_and_single_controller_application(self):
        os.environ["JAX_PLATFORMS"] = "cpu"
        import jax
        import jax.numpy as jnp

        namespace, environment = load_source(jnp)
        env = fixture(namespace, jnp)
        keys = {"phase_offset": jnp.asarray([120, 120]), "inject_ctrl": jnp.full((2, 19), .5),
                "injected": jnp.asarray([0., 1.])}
        batched = jax.jit(jax.vmap(lambda info: env._residual_base(info, jnp.int32(1), jnp.float32(121.))))
        target = batched(keys)
        np.testing.assert_allclose(np.asarray(target[0]), .12, atol=1e-6)
        np.testing.assert_allclose(np.asarray(target[1]), .5, atol=1e-6)
        histories = jax.jit(jax.vmap(namespace["stabilizer_reset_state"]))(jnp.asarray([rotation(.02), rotation(-.02)]))
        np.testing.assert_allclose(np.asarray(histories[0]), [.02, -.02], atol=1e-6)
        np.testing.assert_array_equal(np.asarray(histories[1]), [0., 0.])
        next_side, changed = jax.jit(jax.vmap(namespace["alternating_step_events"]))(jnp.asarray(
            [[True, False], [False, True], [True, True]]), jnp.asarray([-1, 0, 0]))
        np.testing.assert_array_equal(np.asarray(next_side), [0, 1, 0])
        np.testing.assert_array_equal(np.asarray(changed), [False, True, False])

        step = next(node for node in environment.body if isinstance(node, ast.FunctionDef) and node.name == "step")
        prefix = []
        for node in step.body:
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "replace"):
                break
            prefix.append(node)
        prefix.append(ast.Return(value=ast.Tuple(elts=[ast.Name("target", ast.Load()), ast.Name("ctrl", ast.Load())], ctx=ast.Load())))
        function = ast.FunctionDef(name="controller_prefix", args=step.args, body=prefix,
                                  decorator_list=[], returns=None, type_comment=None)
        module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        namespace["jax"] = jax
        namespace["motor_targets_from_actions"] = motor_targets_from_actions
        exec(compile(module, str(ENV_SOURCE), "exec"), namespace)
        env.standing_ctrl, env.phase_rate_range, env.gait_residual = jnp.zeros(19), 0., True
        env.full_command = False
        env.stabilizer, env.base_id = (1., .04, -1., .25), 0
        env.action_scale, env.low, env.high = jnp.full(19, .1), jnp.full(19, -2.), jnp.full(19, 2.)
        env.command_center, env.command_scale = env.standing_ctrl, env.action_scale
        env.control_slew_rad_s = 8.
        state = SimpleNamespace(pipeline_state=SimpleNamespace(xmat=jnp.asarray([rotation(.02)]), ctrl=jnp.full(19, .12)),
            info={"gait_phase": jnp.float32(120.), "step_count": jnp.int32(0), "phase_offset": jnp.int32(120),
                  "inject_ctrl": jnp.full(19, .5), "injected": jnp.float32(0.),
                  "stab_prev": jnp.float32(.02), "stab_filt": jnp.float32(0.)})
        target, ctrl = jax.jit(lambda action: namespace["controller_prefix"](env, state, action))(jnp.zeros(19))
        expected = np.full(19, .12)
        expected[[0, 5, 9]] -= .02
        expected[4] += .02
        np.testing.assert_allclose(np.asarray(target), expected, atol=1e-6)
        np.testing.assert_allclose(np.asarray(ctrl), expected, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
