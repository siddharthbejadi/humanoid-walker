"""CPU checks for balanced teacher evaluation; not a CUDA physics test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.mjx_teacher_env import MJXTeacherEnv
from envs.mjx_walking.scene import (
    balanced_pose_reset_indices, build_mjx_scene, build_pose_rsi_pool)


def check_schedule():
    checks = []

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    # Pool order follows support category, which is not numeric pose-ID order.
    ids = np.asarray([0, 4, 3, 1, 2], dtype=np.int32)
    schedule = balanced_pose_reset_indices(ids, 20)
    check("numeric_pose_order_across_pool_categories",
          np.array_equal(ids[schedule], np.tile(np.arange(5), 4)))
    check("exact_four_trials_per_pose", np.array_equal(
        np.bincount(ids[schedule]), np.full(5, 4)))
    check("repeatable_schedule", np.array_equal(schedule,
          balanced_pose_reset_indices(ids, 20)))
    for name, bad_ids, worlds in (
            ("unbalanced_world_count_rejected", ids, 8),
            ("zero_worlds_rejected", ids, 0),
            ("missing_standing_rejected", [1, 2, 3, 4], 20),
            ("ambiguous_pose_ids_rejected", [0, 1, 1, 3, 4], 20),
            ("fractional_pose_ids_rejected", [0., 1., 2., 3., 4.], 20)):
        try:
            balanced_pose_reset_indices(bad_ids, worlds)
        except ValueError:
            checks.append(name)
        else:
            raise AssertionError(name)

    # Test lifecycle independently of unavailable CUDA/JAX execution. A fresh
    # seed and state must be selected before each first evaluation reset.
    random = SimpleNamespace(PRNGKey=lambda seed: ("test_key", seed))
    env = SimpleNamespace(evaluation=True, _balanced_evaluation=True,
                          _evaluation_seed=10042, jax=SimpleNamespace(random=random),
                          rng="advanced", state="previous_evaluation")
    MJXTeacherEnv.begin_evaluation(env)
    check("new_evaluation_clears_old_state", env.state is None)
    check("new_evaluation_restores_seed", env.rng == ("test_key", 10042))
    env.state, env.rng = "another_episode", "advanced_again"
    MJXTeacherEnv.begin_evaluation(env)
    check("checkpoint_evaluations_repeat_seed", env.rng == ("test_key", 10042)
          and env.state is None)
    env._balanced_evaluation = False
    env.state, env.rng = "legacy_state", "legacy_rng"
    MJXTeacherEnv.begin_evaluation(env)
    check("legacy_random_mix_preserved", env.state == "legacy_state"
          and env.rng == "legacy_rng")
    env.evaluation = False
    try:
        MJXTeacherEnv.begin_evaluation(env)
    except RuntimeError:
        checks.append("training_environment_cannot_restart_as_evaluation")
    else:
        raise AssertionError("training_environment_cannot_restart_as_evaluation")

    # Actual manual-pose loader and MuJoCo model, without a training rollout.
    import mujoco

    assets = Path(__file__).resolve().parents[1] / "envs/mjx_walking/assets"
    scene = build_mjx_scene(assets / "humanoid_rubber_2mm_candidate.xml",
                            motor_torque_kgcm=18.)
    model = mujoco.MjModel.from_xml_path(str(scene))
    pool = build_pose_rsi_pool(model, assets / "manual_rsi_poses.json",
                               np.full(19, np.pi / 2), allow_unstable=True)
    live_schedule = balanced_pose_reset_indices(pool["pose_ids"], 20)
    check("actual_manual_pool_has_five_evaluation_starts", np.array_equal(
        np.sort(pool["pose_ids"]), np.arange(5)))
    check("actual_manual_pool_is_balanced", np.array_equal(
        pool["pose_ids"][live_schedule], np.tile(np.arange(5), 4)))
    category = np.searchsorted(pool["offsets"], np.arange(len(pool["pose_ids"])),
                               side="right") - 1
    train_probabilities = np.asarray([0., .25, .25, .50])
    per_pose = train_probabilities[category] / pool["counts"][category]
    check("training_probabilities_remain_uniform_manual_poses", np.allclose(
        per_pose, np.where(pool["pose_ids"] == 0, 0., .25)))
    return dict(status="passed", checks=checks, check_count=len(checks),
                start_pose_ids=pool["pose_ids"][live_schedule].tolist(),
                scope="CPU schedule, reset lifecycle and MuJoCo manual-pool loading only; no CUDA/JAX reset or physics validation.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = check_schedule()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
