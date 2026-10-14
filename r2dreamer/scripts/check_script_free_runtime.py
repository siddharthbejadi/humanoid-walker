"""Isolated pinned-GPU student reset/command/render check; never claims walking."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def forbidden_reference(*args, **kwargs):
    raise AssertionError("Script-free runtime attempted to read a trajectory or phase pool")


def check(seed):
    from scripts.preflight_reference_training import gpu_runtime_report
    runtime = gpu_runtime_report()
    import hydra
    from script_free_contract import validate_student_config, validate_student_environment
    from scripts.collect_script_free_demonstrations import GPUStudentProvider
    from envs.mjx_walking import scene

    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        config = hydra.compose("walking_mjx_script_free", overrides=[f"seed={seed}"])
    validate_student_config(config)
    config.env.env_num = 8
    # Poison before construction as well as the per-step lookup. This child
    # process exits before any collector/learner, releasing its GPU memory.
    for name in ("scripted_gait_reference", "vendor_gait_reference", "build_scripted_rsi_pool"):
        setattr(scene, name, forbidden_reference)
    provider = GPUStudentProvider(config, 8)
    try:
        provider.env.raw._reference_at = forbidden_reference
        contract = validate_student_environment(provider.contract)
        reset = provider.check_reset_repeatability(seed)
        first = provider.reset(seed)
        hold = provider.normalize(first["audit"]["ctrl"])
        transition = None
        for _ in range(16):
            transition = provider.step(hold)
            for value in (transition["proprio"], transition["reward"],
                          *transition["audit"].values(), *transition["logs"].values()):
                if not np.isfinite(np.asarray(value)).all():
                    raise RuntimeError("Nonfinite student transition/audit/reward during runtime admission")
        frame = np.asarray(provider.video_frame())
        if frame.shape != (240, 320, 3) or not np.isfinite(frame).all() or frame.std() <= 1.:
            raise RuntimeError("GPU student renderer is missing, malformed or blank")
        return dict(status="passed_pinned_gpu_student_runtime_only", runtime=runtime,
                    training_contract=contract, reset_validation=reset, worlds=8,
                    controller="autonomous_full_command_v1", encoder_inputs=["proprio"],
                    reference_loaders_poisoned=True, reference_lookup_poisoned=True,
                    rendered_frame_shape=list(frame.shape), rendered_frame_std=float(frame.std()),
                    transition_finite=bool(np.isfinite(transition["proprio"]).all()),
                    learned_policy_evaluated=False, walking_qualified=False,
                    scope="Pinned GPU imports, real full-command MJX steps, reset aliases and rendering; not gait quality")
    finally:
        provider.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.seed < 0 or not args.output.is_dir():
        parser.error("Use a nonnegative seed and supervisor-created output directory")
    report = check(args.seed)
    path = args.output / "gpu_runtime_report.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
