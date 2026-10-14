"""Bounded fresh full-command imitation pretraining; held-out losses are not walking proof."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--config-name", default="walking_mjx_script_free")
    args = parser.parse_args()
    import hydra
    import numpy as np
    from omegaconf import OmegaConf
    import torch
    from dreamer import Dreamer
    from envs.mjx_teacher_env import MJXTeacherEnv
    from imitation_data import ImitationDataset, file_sha256
    from imitation_training import save_imitation_checkpoint, full_episode_imitation_evidence
    from training_contract import build_training_contract
    from script_free_contract import validate_student_config, check_imitation_source
    import tools

    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        config = hydra.compose(args.config_name)
    if args.seed is not None:
        if args.seed < 0:
            raise ValueError("Seed must be nonnegative")
        config.seed = args.seed
        config.imitation.seed = args.seed
    if any(config.get(key) for key in ("resume_from", "init_from", "transfer_from")):
        raise ValueError("Offline pretraining must initialize fresh weights")
    validate_student_config(config)
    if (int(config.env.env_num), int(config.env.steps), int(config.env.action_repeat),
            int(config.env.train_ratio)) != (128, 1000000, 10, 64):
        raise ValueError("Use the bounded 128-world/1M-tick full-command recipe")
    dist = config.model.actor.dist.cont
    if (float(dist.min_std), float(dist.max_std)) != (.0003, .002):
        raise ValueError("Full-command exploration must preserve the declared low physical noise")
    updates = int(config.imitation.pretrain_updates if args.updates is None else args.updates)
    if updates <= 0 or int(config.imitation.validation_every) <= 0:
        raise ValueError("Update/validation budgets must be positive")
    output = args.output.resolve()
    if output.exists():
        raise ValueError("Output exists; use a fresh pretraining directory")
    dataset = ImitationDataset(args.dataset.resolve(strict=True))
    tools.set_seed_everywhere(int(config.imitation.seed))
    # Only build the physical contract here; no simulator transitions are collected.
    env_config = OmegaConf.create(OmegaConf.to_container(config.env, resolve=True))
    env_config.env_num = 1
    if not Path(str(env_config.scene_path)).is_absolute():
        env_config.scene_path = str((ROOT / str(env_config.scene_path)).resolve(strict=True))
    env = MJXTeacherEnv(env_config)
    try:
        contract = build_training_contract(env, env_config)
        observations, actions = env.observation_space, env.action_space
    finally:
        env.close()
    del env
    check_imitation_source(dataset, contract)
    if set(observations.spaces) != {"proprio"} or actions.shape != (19,):
        raise ValueError("Unexpected student observation/action interface")
    model_export = OmegaConf.to_container(config.model, resolve=True)
    agent = Dreamer(config.model, observations, actions).to(config.device)
    agent.export_model_config = model_export
    output.mkdir(parents=True)
    (output / "config.yaml").write_text(OmegaConf.to_yaml(config, resolve=True), encoding="utf-8")
    rng = np.random.default_rng(int(config.imitation.seed))
    def batch(split, sample_rng):
        return dataset.batch(int(config.imitation.batch_size), int(config.imitation.batch_length),
            burnin=int(config.imitation.burn_in), split=split, device=str(config.device), rng=sample_rng)
    validation = batch("validation", np.random.default_rng(12345))
    initial = {key: float(value) for key, value in agent.validate_imitation(validation).items()}
    best = initial["imitation/bc"]
    changed = 0
    metadata = {"version": 1, "dataset_fingerprint": dataset.fingerprint,
                "pretraining_updates": 0, "mode": "full_command_bc",
                "seed": int(config.imitation.seed),
                "validation_protocol": "Held-out teacher-forced previous commands; deterministic posterior modes",
                "initial_validation": initial, "qualification": "Imitation loss only; no walking rollout"}
    with (output / "metrics.jsonl").open("x", encoding="utf-8") as log:
        log.write(json.dumps({"update": 0, "validation": initial}, allow_nan=False) + "\n")
        for update in range(1, updates + 1):
            before = {name: value.detach().clone() for name, value in agent.named_parameters()
                      if value.requires_grad} if update == 1 else None
            train_metrics = agent.pretrain_imitation(batch("train", rng))
            if before is not None:
                changed = sum(not torch.equal(before[name], value.detach())
                              for name, value in agent.named_parameters() if name in before)
                if not changed or float(train_metrics["imitation/actor_grad_norm"]) <= 0:
                    raise RuntimeError("First imitation update did not change finite weights with actor gradients")
                print(f"IMITATION_LEARNING_VERIFIED changed_tensors={changed} update={update}", flush=True)
            metadata["pretraining_updates"] = update
            if update % int(config.imitation.validation_every) == 0 or update == updates:
                validation_metrics = {key: float(value) for key, value in agent.validate_imitation(validation).items()}
                row = {"update": update, "train": {key: float(value) for key, value in train_metrics.items()},
                       "validation": validation_metrics}
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                metadata["last_validation"] = validation_metrics
                if validation_metrics["imitation/bc"] < best:
                    best = validation_metrics["imitation/bc"]
                    save_imitation_checkpoint(agent, output / "best_bc.pt", contract, dict(metadata), update)
                save_imitation_checkpoint(agent, output / "latest.pt", contract, dict(metadata), update)
                print(json.dumps(row), flush=True)
    final_validation_bc = validation_metrics["imitation/bc"]
    final_validation_improved = final_validation_bc < initial["imitation/bc"]
    full_validation = full_episode_imitation_evidence(agent, dataset, "validation")
    full_test = full_episode_imitation_evidence(agent, dataset, "test")
    for evidence in (full_validation, full_test):
        evidence["optimizer_updates"] = updates
    metadata.update(initial_validation_bc=initial["imitation/bc"], final_validation_bc=final_validation_bc,
                    best_validation_bc=best, final_validation_improved=final_validation_improved,
                    validation_improved=final_validation_improved,
                    full_episode_validation=full_validation, full_episode_test=full_test)
    # Bind the final metrics to the same latest weights used by online warmstart.
    save_imitation_checkpoint(agent, output / "latest.pt", contract, dict(metadata), updates)
    report = {"status": "completed_bounded_imitation", "updates": updates,
              "changed_tensors_first_update": changed, "dataset_fingerprint": dataset.fingerprint,
              "best_validation_bc": best, "initial_validation_bc": initial["imitation/bc"],
              "final_validation_bc": final_validation_bc,
              "final_validation_improved": final_validation_improved,
              "validation_improved": final_validation_improved,
              "latest_checkpoint_sha256": file_sha256(output / "latest.pt"),
              "full_episode_validation": full_validation, "full_episode_test": full_test,
              "validation_protocol": metadata["validation_protocol"],
              "walking_qualified": False, "next_step": "Script-free standing-start closed-loop evaluation"}
    (output / "pretraining_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
