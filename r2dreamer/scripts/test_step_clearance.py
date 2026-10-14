"""Domain regressions for configurable strict-step clearance without full MJX imports.

Run: python scripts/test_step_clearance.py
Optional CPU JAX jit/vmap check: STEP_CLEARANCE_TEST_JAX=1 python scripts/test_step_clearance.py
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "envs/mjx_walking/env.py"
HELPERS = {"foot_swing_qualification", "foot_progress_records", "foot_landing_events"}


def load_helpers(array_module):
    nodes = [node for node in ast.parse(SOURCE.read_text(encoding="utf-8")).body
             if isinstance(node, ast.FunctionDef) and node.name in HELPERS]
    if {node.name for node in nodes} != HELPERS:
        raise RuntimeError("Strict-step helpers are missing")
    namespace = {"jnp": array_module}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


FUNCTIONS = load_helpers(np)


def qualify(clearance, threshold=.015, *, load=(.8, 0.), stable=True, stance_speed=0.):
    return FUNCTIONS["foot_swing_qualification"](
        clearance=np.asarray([0., clearance]), load=np.asarray(load),
        stable_support=np.bool_(stable), foot_speed_sq=np.asarray([stance_speed ** 2, 0.]),
        step_clearance_m=threshold)


def count_landings(rows, threshold=.015):
    contact = np.ones(2, dtype=bool)
    qualified = np.zeros(2, dtype=bool)
    takeoff = np.zeros(2)
    record = np.zeros(2)
    counts = np.zeros(2, dtype=int)
    for clearance, forward, load, stable, stance_speed in rows:
        load = np.asarray(load)
        current = np.where(load > .10, True, np.where(load <= .08, False, contact))
        qnow = qualify(clearance, threshold, load=load, stable=stable, stance_speed=stance_speed)
        progress, takeoff, record = FUNCTIONS["foot_progress_records"](
            forward_position=np.asarray([0., forward]), contact_before=contact,
            contact_now=current, takeoff_forward=takeoff, landing_record=record)
        _, touched, contact, qualified = FUNCTIONS["foot_landing_events"](
            load=load, contact_before=contact, qualified_before=qualified,
            qualified_now=qnow, stable_support=stable, step_forward=progress)
        counts += touched
    return counts


def step_rows(clearance, distance=.03, *, supported=True, stable=True, stance_speed=0.):
    stance_load = .8 if supported else .4
    return [(clearance, 0., (stance_load, 0.), stable, stance_speed),
            (clearance, distance, (stance_load, 0.), stable, stance_speed),
            (0., distance, (.7, .3), True, 0.)]


class StepClearanceTests(unittest.TestCase):
    def test_constructor_bounds_before_any_physics(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        environment = next(node for node in tree.body
                           if isinstance(node, ast.ClassDef) and node.name == "MJXWalkingEnv")
        constructor = next(node for node in environment.body
                           if isinstance(node, ast.FunctionDef) and node.name == "__init__")

        class PhysicsSentinel(Exception):
            pass

        def stop_before_physics(*args, **kwargs):
            raise PhysicsSentinel

        namespace = {"np": np, "Path": Path, "build_mjx_scene": stop_before_physics}
        exec(compile(ast.Module(body=[constructor], type_ignores=[]), str(SOURCE), "exec"), namespace)
        initialize = namespace["__init__"]
        for threshold in (.0145, .015, .020):
            with self.subTest(accepted=threshold), self.assertRaises(PhysicsSentinel):
                initialize(SimpleNamespace(), "unused.xml", step_clearance_m=threshold)
        for threshold in (.0144, .0201, float("nan"), float("inf"), -float("inf")):
            with self.subTest(rejected=threshold), self.assertRaisesRegex(ValueError, "step_clearance_m"):
                initialize(SimpleNamespace(), "unused.xml", step_clearance_m=threshold)

    def test_selected_15mm_minimum_is_inclusive(self):
        self.assertFalse(qualify(.0149)[1])
        self.assertTrue(qualify(.015)[1])
        self.assertTrue(qualify(.020)[1])
        self.assertTrue(qualify(.030)[1])

    def test_configurable_14point5_and_20mm_boundaries(self):
        self.assertFalse(qualify(.01449, .0145)[1])
        self.assertTrue(qualify(.0145, .0145)[1])
        self.assertFalse(qualify(.01999, .020)[1])
        self.assertTrue(qualify(.020, .020)[1])
        self.assertTrue(qualify(.030, .020)[1])

    def test_lower_admission_counts_18mm_lift(self):
        np.testing.assert_array_equal(count_landings(step_rows(.018), .015), [0, 1])
        np.testing.assert_array_equal(count_landings(step_rows(.018), .020), [0, 0])

    def test_higher_lift_still_counts(self):
        for threshold in (.0145, .015, .020):
            with self.subTest(threshold=threshold):
                np.testing.assert_array_equal(count_landings(step_rows(.025), threshold), [0, 1])

    def test_support_and_motion_gates_are_unchanged(self):
        self.assertFalse(qualify(.025, load=(.55, 0.))[1])
        self.assertFalse(qualify(.025, load=(.8, .10))[1])
        self.assertFalse(qualify(.025, stable=False)[1])
        self.assertFalse(qualify(.025, stance_speed=.03)[1])
        self.assertTrue(qualify(.025, stance_speed=.029)[1])

    def test_standing_and_skating_are_not_steps(self):
        rows = [(0., forward, (.5, .5), True, 0.) for forward in (0., .02, .04)]
        np.testing.assert_array_equal(count_landings(rows), [0, 0])
        np.testing.assert_array_equal(count_landings(step_rows(0.)), [0, 0])

    def test_contact_bounce_does_not_replace_a_lift(self):
        rows = [(0., .01, (.7, .09), True, 0.),
                (0., .03, (.7, .11), True, 0.),
                (0., .04, (.7, .07), True, 0.),
                (0., .06, (.7, .11), True, 0.)]
        np.testing.assert_array_equal(count_landings(rows), [0, 0])

    def test_no_prior_supported_lift_does_not_count(self):
        np.testing.assert_array_equal(count_landings(step_rows(.025, supported=False)), [0, 0])
        np.testing.assert_array_equal(count_landings(step_rows(.025, stance_speed=.04)), [0, 0])

    def test_lost_support_consumes_lift_qualification(self):
        rows = step_rows(.025)
        rows.insert(2, (.025, .03, (.8, 0.), False, 0.))
        np.testing.assert_array_equal(count_landings(rows), [0, 0])

    def test_forward_progress_stays_at_20mm(self):
        np.testing.assert_array_equal(count_landings(step_rows(.025, distance=.0199)), [0, 0])
        np.testing.assert_array_equal(count_landings(step_rows(.025, distance=.020)), [0, 1])

    def test_landing_consumes_credit_before_contact_bounce(self):
        rows = step_rows(.025)
        rows += [(0., .03, (.7, .07), True, 0.), (0., .06, (.7, .3), True, 0.)]
        np.testing.assert_array_equal(count_landings(rows), [0, 1])

    @unittest.skipUnless(os.environ.get("STEP_CLEARANCE_TEST_JAX") == "1", "Optional CPU JAX check")
    def test_cpu_jax_jit_and_vmap(self):
        os.environ["JAX_PLATFORMS"] = "cpu"
        import jax
        import jax.numpy as jnp

        fn = load_helpers(jnp)["foot_swing_qualification"]
        batched = jax.jit(jax.vmap(lambda clearance, load, stable, speed_sq:
            fn(clearance=clearance, load=load, stable_support=stable,
               foot_speed_sq=speed_sq, step_clearance_m=.015)))
        result = batched(jnp.asarray([[0., .015], [0., .025]]),
                         jnp.asarray([[.8, 0.], [.8, 0.]]),
                         jnp.asarray([True, True]), jnp.zeros((2, 2)))
        np.testing.assert_array_equal(np.asarray(result), [[False, True], [False, True]])


if __name__ == "__main__":
    unittest.main()
