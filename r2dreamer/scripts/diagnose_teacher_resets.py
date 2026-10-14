"""Bounded CUDA reset diagnostics; never train or relax the preflight gate.

Compare identical evaluation resets before and after five physics calls, and
check whether retained DLPack observation views changed after later calls.
MJWarp may have numerical nondeterminism in derived sensors/forces, so report
that separately from exact reset inputs. The strict observation tolerance stays
at rtol=0, atol=1e-6. A successful report is not a walking or capacity result.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ATOL = 1e-6
PROPRIO_GROUPS = {
    "last_command": (0, 19), "imu_accel": (19, 22),
    "imu_gyro": (22, 25), "imu_mag": (25, 28),
    "onboard_accel": (28, 31), "onboard_gyro": (31, 34),
    "task_command": (34, 37), "filtered_fsr": (37, 45),
}
TEACHER_GROUPS = {
    "joint_position": (0, 19), "joint_velocity": (19, 38),
    "tracking_error": (38, 57), "actuator_force": (57, 76),
    "base_linear_velocity": (76, 79), "base_angular_velocity": (79, 82),
    "com_offset": (82, 84), "foot_contact": (84, 86),
    "foot_load": (86, 88), "foot_clearance": (88, 90), "fsr": (90, 98),
}
EXACT_RESET_FIELDS = (
    "qpos", "qvel", "ctrl", "time", "step_count", "pose_ids",
    "state_rng", "bridge_rng",
)


def difference(reference, current):
    """JSON-safe error measurements, including nonfinite and shape failures."""
    reference, current = np.asarray(reference), np.asarray(current)
    if reference.shape != current.shape:
        return dict(shape_match=False, reference_shape=list(reference.shape),
                    current_shape=list(current.shape), exact=False,
                    finite=False, strict_1e6_pass=False)
    finite = np.isfinite(reference) & np.isfinite(current)
    delta = np.abs(reference.astype(np.float64) - current.astype(np.float64))
    finite_delta = np.where(finite, delta, -1.)
    max_abs = float(finite_delta.max()) if finite_delta.size and finite.any() else None
    worst = (list(map(int, np.unravel_index(finite_delta.argmax(), delta.shape)))
             if max_abs is not None else None)
    exceeded = finite & (delta > ATOL)
    return dict(shape_match=True, shape=list(reference.shape),
                exact=bool(np.array_equal(reference, current)),
                finite=bool(finite.all()), nonfinite_pairs=int((~finite).sum()),
                max_abs=max_abs, count_abs_gt_1e6=int(exceeded.sum()),
                worst_index=worst,
                strict_1e6_pass=bool(finite.all() and not exceeded.any()))


def compare(reference, current):
    fields = {name: difference(value, current[name]) for name, value in reference.items()}
    groups = {}
    for observation, definitions in (("proprio", PROPRIO_GROUPS),
                                     ("teacher_state", TEACHER_GROUPS)):
        groups[observation] = {
            name: difference(reference[observation][:, start:end],
                             current[observation][:, start:end])
            for name, (start, end) in definitions.items()
        }
    return dict(fields=fields, observation_groups=groups,
                exact_reset_inputs=all(fields[name]["exact"] for name in EXACT_RESET_FIELDS),
                strict_observations_pass=all(fields[name]["strict_1e6_pass"]
                                             for name in ("proprio", "teacher_state")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path,
                        help="Defaults to TEACHER_SCENE or the bundled 2 mm scene")
    parser.add_argument("--config-name", default="walking_mjx_teacher_next")
    parser.add_argument("--override", action="append", default=[],
                        help="Repeatable Hydra override matching the intended run")
    parser.add_argument("--worlds", type=int, default=20)
    parser.add_argument("--repeat-resets", type=int, default=3,
                        help="Immediate repeated resets before five physics calls (1..5)")
    parser.add_argument("--report", type=Path, help="Write full JSON diagnostics to this file")
    args = parser.parse_args()
    if args.worlds < 5 or args.worlds > 60 or args.worlds % 5:
        parser.error("--worlds must be 5..60 and divisible by five for balanced evaluation")
    if not 1 <= args.repeat_resets <= 5:
        parser.error("--repeat-resets must be 1..5")

    report = dict(status="incomplete", strict_1e6_pass=False,
                  config_name=args.config_name, overrides=args.override,
                  worlds=args.worlds, repeat_resets=args.repeat_resets,
                  physics_calls=5, tolerance=dict(rtol=0., atol=ATOL),
                  scope="Reset and DLPack observation diagnostics only; no training, capacity or walking validation.")
    env = None
    exit_code = 1
    try:
        import hydra
        import torch
        import jax

        sys.path.insert(0, str(ROOT))
        from envs.mjx_teacher_env import MJXTeacherEnv

        scene = (args.scene or Path(os.environ.get("TEACHER_SCENE", str(
            ROOT / "envs/mjx_walking/assets/humanoid_rubber_2mm_candidate.xml")))).resolve(strict=True)
        os.environ["TEACHER_SCENE"] = str(scene)
        os.environ.setdefault("TEACHER_RUN_DIR", str(ROOT / "reset_diagnostics_not_training"))
        if not torch.cuda.is_available():
            raise RuntimeError("Torch CUDA unavailable; refusing CPU fallback")
        if jax.default_backend() != "gpu":
            raise RuntimeError("JAX GPU unavailable; refusing CPU fallback")
        with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
            config = hydra.compose(args.config_name, overrides=args.override)
        config.env.env_num = args.worlds
        config.env.eval_episode_num = args.worlds
        if not bool(getattr(config.env, "eval_balanced_pose_ids", False)):
            raise ValueError("Diagnostic requires eval_balanced_pose_ids=true in the selected recipe")
        env = MJXTeacherEnv(config.env, args.worlds, evaluation=True)
        report.update(scene=str(scene), configured_seed=int(config.env.seed),
                      effective_evaluation_seed=env._evaluation_seed,
                      torch_version=str(torch.__version__), jax_version=str(jax.__version__),
                      cuda_device=torch.cuda.get_device_name(0))

        def host(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().numpy().copy()
            return np.asarray(jax.device_get(value)).copy()

        def snapshot(transition):
            data = env.state.pipeline_state
            result = {name: host(transition[name]) for name in ("proprio", "teacher_state")}
            for name in ("qpos", "qvel", "ctrl", "time", "qacc", "sensordata"):
                result[name] = host(getattr(data, name))
            result.update(step_count=host(env.state.info["step_count"]),
                          pose_ids=host(env.state.info["rsi_pose_id"]),
                          state_rng=host(env.state.info["rng"]),
                          bridge_rng=host(env.rng))
            if hasattr(data._impl, "overflow"):
                result["overflow"] = host(data._impl.overflow)
            return result

        def reset():
            env.begin_evaluation()
            transition, ended = env.step(
                torch.zeros((args.worlds, 19), device="cuda:0"),
                torch.ones(args.worlds, dtype=torch.bool, device="cuda:0"))
            if not transition["is_first"].all().item() or ended.any().item():
                raise AssertionError("Fresh reset must mark all worlds first and none ended")
            return transition, ended

        first, ended = reset()
        baseline = snapshot(first)
        report["baseline"] = dict(
            pose_ids=baseline["pose_ids"].tolist(), state_rng=baseline["state_rng"].tolist(),
            bridge_rng=baseline["bridge_rng"].tolist(),
            field_shapes={name: list(value.shape) for name, value in baseline.items()},
            all_finite=all(bool(np.isfinite(value).all()) for value in baseline.values()),
            overflow=(baseline["overflow"].tolist() if "overflow" in baseline else None))
        expected_pose_ids = np.tile(np.arange(5), args.worlds // 5)
        report["baseline"]["balanced_pose_ids_exact"] = bool(
            np.array_equal(baseline["pose_ids"], expected_pose_ids))
        report["comparisons"] = []
        for index in range(args.repeat_resets):
            repeated, ended = reset()
            report["comparisons"].append(dict(
                stage=f"immediate_reset_{index + 1}", **compare(baseline, snapshot(repeated))))

        def alias_measurements():
            return {name: difference(baseline[name], host(first[name]))
                    for name in ("proprio", "teacher_state")}

        report["retained_views_after_resets"] = alias_measurements()
        for index in range(5):
            action = torch.full((args.worlds, 19), .02 if index % 2 == 0 else -.02,
                                device="cuda:0")
            _, ended = env.step(action, ended)
        # Readback synchronizes before examining retained original DLPack views.
        jax.block_until_ready(env.state.obs)
        torch.cuda.synchronize()
        report["retained_views_after_steps"] = alias_measurements()
        after_steps_overflow = (host(env.state.pipeline_state._impl.overflow)
                                if hasattr(env.state.pipeline_state._impl, "overflow") else None)
        report["overflow_after_physics_calls"] = (
            after_steps_overflow.tolist() if after_steps_overflow is not None else None)
        repeated, _ = reset()
        report["comparisons"].append(dict(
            stage="reset_after_five_physics_calls", **compare(baseline, snapshot(repeated))))
        alias_unchanged = all(
            item["exact"] and item["finite"]
            for stage in ("retained_views_after_resets", "retained_views_after_steps")
            for item in report[stage].values())
        reset_inputs_exact = all(item["exact_reset_inputs"] for item in report["comparisons"])
        observations_pass = all(item["strict_observations_pass"] for item in report["comparisons"])
        overflow_clear = ("overflow" not in baseline or bool(np.all(baseline["overflow"] == 0)))
        if after_steps_overflow is not None:
            overflow_clear = overflow_clear and bool(np.all(after_steps_overflow == 0))
        for item in report["comparisons"]:
            if "overflow" in item["fields"]:
                overflow_clear = overflow_clear and item["fields"]["overflow"]["exact"]
        strict_pass = (report["baseline"]["all_finite"]
                       and report["baseline"]["balanced_pose_ids_exact"]
                       and alias_unchanged and reset_inputs_exact and observations_pass and overflow_clear)
        report.update(status="passed" if strict_pass else "strict_check_failed",
                      strict_1e6_pass=bool(strict_pass), reset_inputs_exact=reset_inputs_exact,
                      retained_observation_views_unchanged=alias_unchanged,
                      overflow_clear=bool(overflow_clear))
        exit_code = 0 if strict_pass else 2
    except Exception as exc:
        report.update(status="error", error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc())
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                report["cleanup_error"] = f"{type(exc).__name__}: {exc}"
                report["strict_1e6_pass"] = False
                report["status"] = "error"
                exit_code = 1
        encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(encoded, encoding="utf-8")
        print(encoded, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
