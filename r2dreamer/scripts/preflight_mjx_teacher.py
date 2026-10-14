"""Bounded reward checks and fail-closed live CUDA smoke for the teacher."""

import argparse
import json
import os
from pathlib import Path

import hydra
import numpy as np
import torch


def reward_only_report():
    """Check reward preference and event history without constructing physics."""
    import jax
    import jax.numpy as jnp
    from envs.mjx_walking.env import (
        foot_landing_events, foot_progress_records, walking_reward_terms)

    checks = []

    def check(name, condition):
        if not bool(condition):
            raise AssertionError(f"Reward preflight failed: {name}")
        checks.append(name)

    def inputs(**overrides):
        values = dict(tilt=0., height_error=0., load=[.5, .5], clearance=[0., 0.],
                      foot_speed=[0., 0.], root_forward=0., forward_record=0.,
                      lift_credit=[0., 0.], last_step_side=-1,
                      touched=[False, False], fallen=False, dt=.02)
        values.update(overrides)
        return {key: jnp.asarray(value, dtype=(jnp.bool_ if key in ("touched", "fallen")
                    else jnp.int32 if key == "last_step_side" else jnp.float32))
                for key, value in values.items()}

    def score(terms):
        return sum(value for key, value in terms.items() if key.startswith("reward_")) - sum(
            value for key, value in terms.items() if key.startswith("penalty_"))

    reward_jit = jax.jit(lambda state: walking_reward_terms(**state))
    standing = inputs()
    shift = inputs(load=[1., 0.])
    partial = inputs(load=[1., 0.], clearance=[0., .010])
    lifted = inputs(load=[1., 0.], clearance=[0., .025])
    skating = inputs(foot_speed=[.08, .08], root_forward=.0016)
    states = [standing, shift, partial, lifted, skating]
    batch = {key: jnp.stack([state[key] for state in states]) for key in standing}
    batch_result = jax.jit(jax.vmap(lambda state: walking_reward_terms(**state)))(batch)
    check("jit_vmap_finite", all(np.isfinite(np.asarray(jax.device_get(leaf))).all()
          for leaf in jax.tree_util.tree_leaves(batch_result)))
    terms = [reward_jit(state)[0] for state in states]
    check("jit_vmap_parity", all(np.allclose(np.asarray(batch_result[0][key]),
          np.asarray([term[key] for term in terms]), atol=1e-7) for key in terms[0]))
    required = {"reward_balance", "reward_lift_progress", "reward_forward",
                "reward_touchdown", "reward_alternation", "penalty_scrub", "penalty_fall"}
    check("term_contract", set(terms[0]) == required)
    check("stationary_forward_zero", all(float(term["reward_forward"]) == 0.
          for term in terms[:4]))
    check("dense_transfer_and_partial_lift", 0. < float(terms[1]["reward_lift_progress"])
          < float(terms[2]["reward_lift_progress"]) < float(terms[3]["reward_lift_progress"]))
    check("skating_no_progress", float(terms[4]["reward_forward"]) == 0.)
    check("skating_below_standing", float(score(terms[4])) < float(score(terms[0])))

    # Exercise state memory through a compiled scan, rather than resetting credit
    # on each sample. Each touchdown frame has contact and zero clearance.
    observed_keys = tuple(key for key in standing
                          if key not in ("lift_credit", "forward_record", "last_step_side"))

    def scan_step(memory, observed):
        term, credit, record, side = walking_reward_terms(
            **observed, lift_credit=memory[0], forward_record=memory[1], last_step_side=memory[2])
        return (credit, record, side), term

    scan_jit = jax.jit(lambda memory, observed: jax.lax.scan(scan_step, memory, observed))

    def trajectory(rows, memory=None):
        observed = {key: jnp.stack([row[key] for row in rows]) for key in observed_keys}
        if memory is None:
            memory = (jnp.zeros(2, dtype=jnp.float32), jnp.asarray(0., dtype=jnp.float32),
                      jnp.asarray(-1, dtype=jnp.int32))
        end, series = scan_jit(memory, observed)
        summed = {key: float(np.asarray(jax.device_get(value)).sum())
                  for key, value in series.items()}
        return float(score(summed)), summed, end, series

    duration = 8.
    count = int(round(duration / .02))
    stand_total = trajectory([standing] * count)[0]
    held_total, held_terms, _, held_series = trajectory([lifted] * count)
    skate_rows = [inputs(foot_speed=[.08, .08], root_forward=(index + 1) * .08 * .02)
                  for index in range(count)]
    skate_total = trajectory(skate_rows)[0]
    check("long_held_lift_bounded_credit", np.isclose(held_terms["reward_lift_progress"], .5))
    check("long_held_lift_plateau", np.allclose(
          np.asarray(held_series["reward_lift_progress"])[1:], 0., atol=1e-7))

    inplace_rows = []
    for _ in range(8):
        inplace_rows.extend([lifted] * 20)
        inplace_rows.extend([inputs(load=[.8, .2])] * 5)
    _, inplace_terms, _, _ = trajectory(inplace_rows)
    check("inplace_cycles_no_renewed_credit", np.isclose(
          inplace_terms["reward_lift_progress"], .5) and inplace_terms["reward_touchdown"] == 0.)

    forward_rows = []
    for index in range(count):
        swing = (index // 25) % 2
        stance = 1 - swing
        loads, heights, speeds = [0., 0.], [0., 0.], [0., 0.]
        loads[stance], heights[swing], speeds[swing] = 1., .025, .08
        touched = [False, False]
        if index % 25 == 24:
            loads[stance], loads[swing] = .8, .2
            heights[swing], speeds[swing], touched[swing] = 0., .03, True
        forward_rows.append(inputs(load=loads, clearance=heights, foot_speed=speeds,
                                  root_forward=(index + 1) * .08 * .02, touched=touched))
    forward_total, forward_terms, _, _ = trajectory(forward_rows)
    check("equal_duration_trajectory_order", forward_total > held_total > stand_total > skate_total)
    check("alternating_trajectory_event_count", np.isclose(forward_terms["reward_touchdown"], 16.)
          and np.isclose(forward_terms["reward_alternation"], 7.5))

    first_left = reward_jit(inputs(load=[.2, .8], touched=[True, False]))
    first_right = reward_jit(inputs(load=[.8, .2], touched=[False, True]))
    check("first_either_leg", float(first_left[0]["reward_touchdown"]) == 1.
          and float(first_right[0]["reward_touchdown"]) == 1.
          and int(first_left[3]) == 0 and int(first_right[3]) == 1)
    repeat = reward_jit(inputs(load=[.2, .8], clearance=[.025, 0.],
                              last_step_side=0, touched=[True, False]))
    check("repeated_same_leg_no_event_or_lift", float(repeat[0]["reward_touchdown"]) == 0.
          and float(repeat[0]["reward_alternation"]) == 0.
          and float(repeat[0]["reward_lift_progress"]) == 0. and int(repeat[3]) == 0)
    alternate = reward_jit(inputs(load=[.8, .2], last_step_side=0, touched=[False, True],
                                 lift_credit=[.2, .7]))
    check("opposite_leg_event_and_credit_reset", float(alternate[0]["reward_touchdown"]) == 1.
          and float(alternate[0]["reward_alternation"]) == .5
          and int(alternate[3]) == 1 and float(alternate[1][1]) == 0.)
    both = reward_jit(inputs(touched=[True, True]))
    check("simultaneous_touchdowns_not_a_step", float(both[0]["reward_touchdown"]) == 0.
          and float(both[0]["reward_alternation"]) == 0. and int(both[3]) == -1)

    rocked = [inputs(load=[1., 0.], clearance=[0., .025], root_forward=x)
              for x in (.02, 0., .02, 0., .02)]
    _, _, _, rocking_series = trajectory(rocked)
    check("root_rocking_no_repeated_progress", float(rocking_series["reward_forward"][0]) > 0.
          and np.allclose(np.asarray(rocking_series["reward_forward"])[1:], 0.))
    bank = reward_jit(inputs(foot_speed=[.08, .08], root_forward=.04))
    unbank = reward_jit(inputs(load=[1., 0.], clearance=[0., .025], root_forward=.04,
                             forward_record=bank[2], lift_credit=bank[1], last_step_side=bank[3]))
    check("skating_cannot_bank_distance", np.isclose(float(bank[2]), .04)
          and float(bank[0]["reward_forward"]) == 0. and float(unbank[0]["reward_forward"]) == 0.)

    def one_second(dt):
        return trajectory([inputs(load=[1., 0.], clearance=[0., .025], dt=dt,
                           root_forward=(index + 1) * .08 * dt)
                           for index in range(int(round(1. / dt)))])[0]

    check("10ms_20ms_duration_scaling", np.isclose(one_second(.01), one_second(.02), atol=1e-5))
    fallen = reward_jit(inputs(load=[1., 0.], clearance=[0., .025], root_forward=.01,
                              touched=[True, False], fallen=True))
    check("fallen_positive_terms_zero", all(float(value) == 0. for key, value in fallen[0].items()
          if key.startswith("reward_")) and float(fallen[0]["penalty_fall"]) == 3.)
    falling_rows = [inputs(load=[1., 0.], clearance=[0., .025],
                          root_forward=(index + 1) * .08 * .02) for index in range(25)]
    falling_rows.append(inputs(fallen=True, root_forward=.04))
    check("short_fall_return_negative", trajectory(falling_rows)[0] < 0.)

    mirror_a = inputs(load=[.7, .1], clearance=[.001, .012], foot_speed=[.005, .08],
                      root_forward=.003, lift_credit=[.1, .3], last_step_side=0,
                      touched=[False, True])
    mirror_b = dict(mirror_a)
    for key in ("load", "clearance", "foot_speed", "lift_credit", "touched"):
        mirror_b[key] = mirror_a[key][::-1]
    mirror_b["last_step_side"] = jnp.asarray(1, dtype=jnp.int32)
    mirrored_a, mirrored_b = reward_jit(mirror_a), reward_jit(mirror_b)
    check("reward_mirror_parity", all(np.allclose(np.asarray(mirrored_a[0][key]),
          np.asarray(mirrored_b[0][key]), atol=1e-7) for key in required)
          and np.allclose(np.asarray(mirrored_a[1]), np.asarray(mirrored_b[1])[::-1])
          and np.isclose(float(mirrored_a[2]), float(mirrored_b[2]))
          and int(mirrored_a[3]) == 1 - int(mirrored_b[3]))
    for key, bad_value in (("tilt", np.nan), ("height_error", np.inf),
                           ("load", [np.nan, 1.]), ("clearance", [0., np.nan]),
                           ("foot_speed", [np.inf, 0.]), ("root_forward", np.nan),
                           ("forward_record", np.inf), ("lift_credit", [0., np.nan])):
        bad = reward_jit(inputs(**{key: bad_value}))
        check(f"nonfinite_{key}_safe", all(np.isfinite(np.asarray(jax.device_get(leaf))).all()
              for leaf in jax.tree_util.tree_leaves(bad))
              and all(float(value) == 0. for name, value in bad[0].items()
                      if name.startswith("reward_")) and float(bad[0]["penalty_fall"]) == 3.)

    def landing_inputs(**overrides):
        values = dict(load=[.6, .11], contact_before=[True, False],
                      qualified_before=[False, True], qualified_now=[False, False],
                      stable_support=True, step_forward=[0., .02])
        values.update(overrides)
        return {key: jnp.asarray(value, dtype=(jnp.float32 if key in ("load", "step_forward")
                    else jnp.bool_)) for key, value in values.items()}

    landing_jit = jax.jit(lambda state: foot_landing_events(**state))
    contact, qualified = jnp.asarray([True, False]), jnp.asarray([False, True])
    landed_count, touched_count = 0, 0
    for swing_load in (.05, .09, .095, .105, .12):
        landed, touched, contact, qualified = landing_jit(landing_inputs(
            load=[.6, swing_load], contact_before=contact, qualified_before=qualified))
        landed_count += int(landed[1])
        touched_count += int(touched[1])
    check("gradual_contact_crossing_once", landed_count == 1 and touched_count == 1)
    hold_contact = landing_jit(landing_inputs(load=[.6, .09], contact_before=[True, True]))
    check("contact_hysteresis_holds_deadband", bool(hold_contact[2][1])
          and not bool(hold_contact[0][1]))
    clear_contact = landing_jit(landing_inputs(load=[.6, .07], contact_before=[True, True]))
    check("contact_hysteresis_rearms_after_unload", not bool(clear_contact[2][1]))
    inplace = landing_jit(landing_inputs(step_forward=[0., 0.]))
    check("inplace_landing_clears_qualification", bool(inplace[0][1])
          and not bool(inplace[1][1]) and not bool(inplace[3][1]))
    stale = landing_jit(landing_inputs(qualified_before=inplace[3], step_forward=[0., .03]))
    check("stale_lift_cannot_pay_later", not bool(stale[1][1]))
    instant = landing_jit(landing_inputs(qualified_before=[False, False],
                                      qualified_now=[False, True]))
    check("instant_qualification_cannot_land", bool(instant[0][1])
          and not bool(instant[1][1]) and not bool(instant[3][1]))
    below = landing_jit(landing_inputs(step_forward=[0., .019]))
    above = landing_jit(landing_inputs(step_forward=[0., .021]))
    check("forward_step_20mm_threshold", not bool(below[1][1]) and bool(above[1][1]))
    unsupported = landing_jit(landing_inputs(load=[.54, .11]))
    unstable = landing_jit(landing_inputs(stable_support=False))
    check("landing_requires_opposite_support_and_stability", not bool(unsupported[1][1])
          and not bool(unstable[1][1]) and not bool(unstable[3].any()))
    landing_a = landing_inputs()
    landing_b = {key: value if value.ndim == 0 else value[::-1] for key, value in landing_a.items()}
    result_a, result_b = landing_jit(landing_a), landing_jit(landing_b)
    check("landing_mirror_parity", all(np.array_equal(np.asarray(a), np.asarray(b)[::-1])
          for a, b in zip(result_a, result_b)))

    def progress_inputs(**overrides):
        values = dict(forward_position=[0., 0.], contact_before=[True, True],
                      contact_now=[True, False], takeoff_forward=[0., 0.],
                      landing_record=[0., 0.])
        values.update(overrides)
        return {key: jnp.asarray(value, dtype=(jnp.bool_ if key.startswith("contact_")
                    else jnp.float32)) for key, value in values.items()}

    progress_jit = jax.jit(lambda state: foot_progress_records(**state))

    def landing_record_trajectory(samples):
        contact = jnp.asarray([True, True])
        takeoff, record = jnp.zeros(2), jnp.zeros(2)
        event_count = 0
        for position, foot_down in samples:
            next_contact = jnp.asarray([True, foot_down])
            progress, takeoff, record = progress_jit(progress_inputs(
                forward_position=[0., position], contact_before=contact, contact_now=next_contact,
                takeoff_forward=takeoff, landing_record=record))
            if foot_down and not bool(contact[1]):
                # Grant fresh genuine lift history, so rejection here must come
                # from forward progress rather than missing qualification.
                _, touched, _, _ = landing_jit(landing_inputs(step_forward=progress))
                event_count += int(touched[1])
            contact = next_contact
        return event_count, takeoff, record

    backward_loop = [(0., False), (-.04, False), (-.04, True),
                     (-.04, False), (0., False), (0., True)] * 8
    loop_events, _, loop_record = landing_record_trajectory(backward_loop)
    check("backward_landing_forward_loop_cannot_earn", loop_events == 0
          and np.allclose(np.asarray(loop_record), 0.))
    repeated_record = [(0., False), (.03, False), (.03, True)] + [
        (.03, False), (0., False), (0., True), (0., False), (.03, False), (.03, True)] * 8
    repeated_events, _, repeated_final = landing_record_trajectory(repeated_record)
    check("forward_record_cannot_be_earned_twice", repeated_events == 1
          and np.isclose(float(repeated_final[1]), .03))

    slide = progress_jit(progress_inputs(forward_position=[0., .035],
                         contact_before=[True, True], contact_now=[True, True]))
    slide_takeoff = progress_jit(progress_inputs(forward_position=[0., .035],
        takeoff_forward=slide[1], landing_record=slide[2]))
    slide_landing = progress_jit(progress_inputs(forward_position=[0., .035],
        contact_before=[True, False], contact_now=[True, True],
        takeoff_forward=slide_takeoff[1], landing_record=slide_takeoff[2]))
    slide_event = landing_jit(landing_inputs(step_forward=slide_landing[0]))
    check("stance_slide_before_takeoff_no_inplace_step", np.isclose(float(slide_takeoff[1][1]), .035)
          and np.isclose(float(slide_landing[0][1]), 0.) and not bool(slide_event[1][1]))
    normal_events, _, normal_record = landing_record_trajectory([
        (0., False), (.021, False), (.021, True)])
    check("normal_forward_landing_earns", normal_events == 1
          and np.isclose(float(normal_record[1]), .021))
    in_flight = progress_jit(progress_inputs(forward_position=[0., .08],
        contact_before=[True, False], contact_now=[True, False]))
    check("foot_record_advances_only_at_landing", np.allclose(np.asarray(in_flight[2]), 0.))
    progress_a = progress_inputs(forward_position=[.035, .08],
        contact_before=[False, True], contact_now=[True, False],
        takeoff_forward=[.01, .03], landing_record=[.02, .04])
    progress_b = {key: value[::-1] for key, value in progress_a.items()}
    progress_result_a, progress_result_b = progress_jit(progress_a), progress_jit(progress_b)
    check("foot_progress_mirror_parity", all(np.allclose(np.asarray(a), np.asarray(b)[::-1])
          for a, b in zip(progress_result_a, progress_result_b)))
    return dict(status="REWARD_PREFLIGHT_OK", profile="step_progress_v1",
                backend=jax.default_backend(), checks=len(checks),
                scenarios_8s=dict(standing=stand_total, held_lift=held_total,
                                  skating=skate_total, forward_alternating=forward_total),
                scope="Bounded reward math and event history; no physics, training, or walking proof.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--worlds", type=int, default=8)
    parser.add_argument("--config-name", default="walking_mjx_privileged_teacher_200k",
                        help="Compose the same recipe that will be used by the trainer")
    parser.add_argument("--override", action="append", default=[],
                        help="Hydra override, repeatable; match the intended launch settings")
    parser.add_argument("--evaluation", action="store_true",
                        help="Exercise the evaluation reset path, including balanced pose IDs")
    parser.add_argument("--reward-only", action="store_true",
                        help="Run bounded JAX reward/event checks without constructing physics")
    parser.add_argument("--pose-rsi-only", action="store_true",
                        help="Check saved pose structure; run 2 s CPU holds unless config permits falling resets")
    parser.add_argument("--pose-json", type=Path,
                        help="Override the bundled four-pose RSI JSON")
    parser.add_argument("--rsi-off", action="store_true", help="Explicitly disable RSI for CUDA smoke")
    parser.add_argument("--report", type=Path, help="Save reward-only results as a JSON runtime artifact")
    args = parser.parse_args()
    if args.reward_only:
        report = reward_only_report()
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, sort_keys=True))
        return
    if args.scene is None:
        parser.error("--scene is required unless --reward-only is used")
    if args.pose_rsi_only:
        import mujoco
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from envs.mjx_walking.scene import build_mjx_scene, build_pose_rsi_pool
        with hydra.initialize_config_dir(
                version_base=None,
                config_dir=str(Path(__file__).resolve().parents[1] / "configs")):
            config = hydra.compose(args.config_name, overrides=args.override)
        bundle = Path(__file__).resolve().parents[1]
        pose_json = args.pose_json or bundle / str(config.env.pose_rsi_json)
        source = build_mjx_scene(args.scene,
                                 motor_torque_kgcm=float(config.env.mjx_motor_torque_kgcm))
        model = mujoco.MjModel.from_xml_path(str(source))
        scale = np.asarray([float(config.env.mjx_leg_action_scale)] * 10
                           + [float(config.env.mjx_arm_action_scale)] * 6
                           + [float(config.env.mjx_other_action_scale)] * 3)
        exploratory = bool(getattr(config.env, "pose_rsi_allow_falls", False))
        pool = build_pose_rsi_pool(model, pose_json, scale,
                                   allow_unstable=exploratory)
        report = dict(scene=str(source), pose_json=str(pose_json),
                      motor_torque_nm=float(np.max(model.actuator_forcerange[:, 1])),
                      counts=pool["counts"].tolist(),
                      categories=pool["category_names"],
                      physical_hold_validated=not exploratory,
                      stats=pool["stats"])
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, sort_keys=True))
        if pool["counts"][1] == 0 or pool["counts"][2] == 0:
            qualifier = "exploratory" if exploratory else "stable"
            raise RuntimeError(f"Pose RSI preflight failed: {qualifier} left and right swing starts are required")
        return
    if args.report is not None:
        parser.error("--report is supported with --reward-only or --pose-rsi-only")
    if args.worlds < 2 or args.worlds > 512:
        parser.error("Preflight supports 2..512 worlds")
    scene = args.scene.resolve(strict=True)
    os.environ["TEACHER_SCENE"] = str(scene)
    os.environ.setdefault("TEACHER_RUN_DIR", str(scene.parent / "teacher_preflight_not_a_training_run"))
    if not torch.cuda.is_available():
        raise RuntimeError("Torch CUDA unavailable; refusing CPU fallback")
    import jax
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"JAX backend is {jax.default_backend()!r}, not GPU")
    from envs.mjx_teacher_env import MJXTeacherEnv
    from reset_validation import validate_reset_repeatability
    from training_budget import validate_learning_budget
    with hydra.initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[1] / "configs")):
        config = hydra.compose(args.config_name, overrides=args.override)
    config.env.env_num = args.worlds
    config.env.eval_episode_num = args.worlds if args.evaluation else 0
    if args.rsi_off:
        config.env.rsi_stratified = False
    validate_learning_budget(args.worlds, 64, 10, max(8192, args.worlds * 128), 200000)
    env = MJXTeacherEnv(config.env, args.worlds, evaluation=args.evaluation)
    try:
        if args.evaluation:
            env.begin_evaluation()
        expected_scale = np.asarray(
            [float(config.env.mjx_leg_action_scale)] * 10
            + [float(config.env.mjx_arm_action_scale)] * 6
            + [float(config.env.mjx_other_action_scale)] * 3,
            dtype=np.float32)
        actual_scale = np.asarray(jax.device_get(env.raw.action_scale))
        if not np.allclose(actual_scale, expected_scale, rtol=0., atol=1e-6):
            raise RuntimeError("MJX action scale does not match the selected 19-channel config")
        done = torch.ones(args.worlds, dtype=torch.bool, device="cuda:0")
        action = torch.zeros((args.worlds, 19), device="cuda:0")
        first, ended = env.step(action, done)
        if not args.evaluation and getattr(config.env, "training_pose_id", None) is not None:
            assert env.reset_pose_ids() == [int(config.env.training_pose_id)] * args.worlds
        first_pose_ids = env.reset_pose_ids() if args.evaluation else None
        balanced_evaluation = bool(getattr(env, "_balanced_evaluation", False))
        repeatability_enabled = args.evaluation and (
            balanced_evaluation or bool(getattr(env, "_repeatable_evaluation", False)))
        if repeatability_enabled:
            if balanced_evaluation:
                expected_ids = np.tile(np.arange(5), args.worlds // 5).tolist()
            elif not env.raw.rsi_stratified:
                expected_ids = [0] * args.worlds
            else:
                expected_ids = first_pose_ids
            assert first_pose_ids == expected_ids, (first_pose_ids, expected_ids)

            def reset_snapshot(transition):
                # Independent host copies are essential: a retained DLPack
                # view cannot itself serve as evidence against aliasing.
                result = {name: transition[name].detach().cpu().numpy().copy()
                          for name in ("proprio", "teacher_state")}
                data = env.state.pipeline_state
                for name in ("qpos", "qvel", "ctrl", "time"):
                    result[name] = np.asarray(jax.device_get(getattr(data, name))).copy()
                for key, name in (("step_count", "step_count"),
                                  ("pose_ids", "rsi_pose_id"), ("state_rng", "rng")):
                    result[key] = np.asarray(jax.device_get(env.state.info[name])).copy()
                result["bridge_rng"] = np.asarray(jax.device_get(env.rng)).copy()
                return result

            first_snapshot = reset_snapshot(first)
        transitions = [first]
        for index in range(5):
            action.fill_(0.02 if index % 2 == 0 else -0.02)
            transition, ended = env.step(action, ended)
            transitions.append(transition)
            metrics = {key: np.asarray(jax.device_get(value))
                       for key, value in env.state.metrics.items()}
            assert all(np.isfinite(value).all() for value in metrics.values())
            if env.raw.task == "walking":
                accounted_reward = sum(value for key, value in metrics.items()
                                       if key.startswith("reward_")) - sum(
                    value for key, value in metrics.items() if key.startswith("penalty_"))
                assert np.allclose(accounted_reward, np.asarray(jax.device_get(env.state.reward)),
                                   rtol=1e-5, atol=1e-5)
                assert torch.allclose(transition["proprio"][:, 34:37],
                    torch.tensor([.08, 0., 0.], device="cuda:0").expand(args.worlds, -1))
            elif env.raw.task == "standing":
                assert torch.allclose(transition["proprio"][:, 34:37],
                    torch.tensor([0., 0., 1.], device="cuda:0").expand(args.worlds, -1))
                # standing_reward_terms() accounts for everything except the
                # shared action-rate cost applied once at the tail of step();
                # unlike step_progress_v1, the standing task does NOT zero
                # that weight (.25, see env.py), so it must be included here.
                accounted_reward = (
                    sum(value for key, value in metrics.items()
                        if key.startswith("standing_reward_"))
                    + metrics["standing_streak_bonus"]
                    - sum(value for key, value in metrics.items()
                          if key.startswith("standing_penalty_"))
                    - metrics["penalty_action_rate"])
                assert np.allclose(accounted_reward, np.asarray(jax.device_get(env.state.reward)),
                                   rtol=1e-5, atol=1e-5)
            if index == 0:
                assert np.allclose(np.asarray(jax.device_get(env.state.metrics["penalty_action_rate"])), 0.)
        for transition in transitions:
            assert transition["proprio"].shape == (args.worlds, 45)
            assert transition["teacher_state"].shape == (args.worlds, 98)
            assert torch.isfinite(transition["teacher_state"]).all().item()
            assert torch.isfinite(transition["reward"]).all().item()
        assert first["is_first"].all().item()
        if repeatability_enabled:
            env.begin_evaluation()
            repeated, _ = env.step(torch.zeros_like(action), torch.ones_like(done))
            retained = {name: first[name].detach().cpu().numpy().copy()
                        for name in ("proprio", "teacher_state")}
            reset_report = validate_reset_repeatability(
                first_snapshot, reset_snapshot(repeated), retained, expected_ids)
            print("MJX_RESET_REPEATABILITY " + json.dumps(reset_report, sort_keys=True))
            if not reset_report["passed"]:
                raise AssertionError("MJX reset numerical repeatability failed: "
                                     + ", ".join(reset_report["errors"]))
            print("Bounded numerical reset repeatability passed with exact state/RNG and "
                  "unchanged retained observations; this does not guarantee bitwise GPU physics.")
        print(f"MJX_TEACHER_PREFLIGHT_OK worlds={args.worlds} device=cuda:0 "
              f"obs=45+98 actions=19 scale_deg={np.degrees(actual_scale[0]):.1f} "
              f"reward_profile={env.raw.reward_profile} rsi={env.raw.rsi_stratified}")
        print("Five nonzero-action CUDA/JIT/DLPack steps and reset action-rate masking passed; not proof of learning or walking.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
