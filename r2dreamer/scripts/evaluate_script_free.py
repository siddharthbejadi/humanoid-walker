"""Evaluate a student checkpoint with no demonstration/reference file loaded."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--seconds", type=float, default=20.)
    parser.add_argument("--worlds", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists() or args.seed < 0 or not 1 <= args.worlds <= 32 or not 2. <= args.seconds <= 60.:
        parser.error("Use a fresh output, nonnegative seed, 1..32 worlds and 2..60 seconds")
    import hydra
    import torch
    from omegaconf import OmegaConf
    from envs.mjx_teacher_env import MJXTeacherEnv
    from script_free_contract import validate_student_config
    from script_free_policy import ScriptFreePolicy
    from training_contract import build_training_contract, check_checkpoint_contract
    from trainer import OnlineTrainer
    import tools

    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        config = hydra.compose("walking_mjx_script_free")
    validate_student_config(config)
    policy = ScriptFreePolicy(args.checkpoint.resolve(strict=True), device="cuda:0")
    env_config = OmegaConf.create(OmegaConf.to_container(config.env, resolve=True))
    env_config.scene_path = str((ROOT / str(env_config.scene_path)).resolve(strict=True))
    env_config.eval_seed = args.seed
    import envs.mjx_walking.scene as scene
    def forbidden(*unused_args, **unused_kwargs):
        raise RuntimeError("Student evaluation attempted to load a scripted trajectory")
    for name in ("scripted_gait_reference", "vendor_gait_reference", "build_scripted_rsi_pool"):
        setattr(scene, name, forbidden)
    env = MJXTeacherEnv(env_config, args.worlds, evaluation=True)
    try:
        contract = build_training_contract(env, env_config)
        checkpoint = {"training_contract": policy.contract}
        check_checkpoint_contract(checkpoint, contract, mode="resume")
        env.raw._reference_at = forbidden
        trained_horizon = env.horizon
        requested_horizon = int(round(args.seconds / float(env.raw.control_dt)))
        if requested_horizon > env.horizon:
            # Longer qualification changes only evaluation truncation, declared below.
            env.horizon = requested_horizon
        evaluation_contract = build_training_contract(env, env_config)
        args.output.mkdir(parents=True)
        logger = tools.Logger(args.output)
        settings = OmegaConf.create(OmegaConf.to_container(config.trainer, resolve=True))
        settings.eval_max_steps = requested_horizon
        settings.eval_episode_num = args.worlds
        settings.eval_video = True
        settings.eval_video_save_mp4 = True
        settings.eval_video_final_only = False
        settings.eval_video_every = 1
        settings.eval_video_one_per_pose = False
        evaluator = OnlineTrainer(settings, None, logger, args.output, env, env)
        evaluator.training_contract = evaluation_contract
        evaluator.imitation_metadata = policy.imitation
        weights = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        step = int(weights.get("step", 0))
        summary = evaluator.eval(policy.agent, step)
        report = {"status": "completed_script_free_evaluation", "checkpoint": str(args.checkpoint),
                  "training_step": step, "summary": summary, "protocol": env.evaluation_protocol(),
                  "training_horizon_control_steps": trained_horizon,
                  "requested_evaluation_control_steps": requested_horizon,
                  "evaluation_horizon_explicitly_extended": requested_horizon > trained_horizon,
                  "trajectory_loaded": False, "explicit_stabilizer": False,
                  "dataset_loaded": False, "privileged_actor_inputs": False,
                  "walking_qualified": False,
                  "scope": "Closed-loop trial metrics/videos need physical review; completing evaluation is not walking proof"}
        (args.output / "script_free_evaluation.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        logger.write(step)
        print(json.dumps(report, indent=2, allow_nan=False))
    finally:
        env.close()


if __name__ == "__main__":
    main()
