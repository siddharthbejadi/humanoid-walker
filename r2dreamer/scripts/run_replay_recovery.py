"""Bounded replay-repair recovery from a preserved student checkpoint."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.script_free_run_ops import RunJournal, atomic_json, run_stage, sha256, stop_signals


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def recovery_command(pilot, output, updates, seed=100, initial=None):
    if updates not in (32, 128) or seed < 0:
        raise ValueError("Recovery accepts one explicit 32- or 128-update stage")
    worlds = 256
    budget = 68 * worlds * 10 + (updates + 2) * 1024 * 10
    initial = Path(pilot) / "imitation/latest.pt" if initial is None else Path(initial)
    return [sys.executable, "-u", "train.py", "--config-name", "walking_mjx_script_free",
            f"seed={seed}", f"imitation.dataset={(Path(pilot) / 'demonstrations.npz').as_posix()}",
            f"init_from={initial.as_posix()}",
            f"logdir={(Path(output) / 'online').as_posix()}", "env.env_num=256", "env.train_ratio=64",
            f"env.steps={budget}", "env.eval_seed=1100", "trainer.learning_start_deadline_steps=176640",
            f"trainer.stop_after_optimizer_updates={updates}", "trainer.gait_guard_enabled=true",
            "trainer.best_metric_key=walking_success_rate",
            "trainer.gait_guard_eval_updates=8", "trainer.eval_video_one_per_pose=true",
            "trainer.eval_video=true", "trainer.eval_video_save_mp4=true",
            "trainer.eval_video_final_only=false"]


def validate_command(command):
    import hydra
    from script_free_contract import validate_student_config
    from training_budget import validate_learning_budget
    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        config = hydra.compose("walking_mjx_script_free", overrides=command[5:])
    validate_student_config(config)
    if (config.resume_from is not None or config.transfer_from is not None or
            int(config.env.env_num) != 256 or int(config.env.train_ratio) != 64 or
            int(config.env.eval_seed) != 1100 or int(config.env.eval_episode_num) != 32 or
            float(config.model.imitation_weight) != .1 or float(config.model.imitation_pretrain_weight) != 1. or
            not config.trainer.gait_guard_enabled or int(config.trainer.gait_guard_eval_updates) != 8 or
            str(config.trainer.best_metric_key) != "walking_success_rate" or
            int(config.trainer.eval_max_steps) != 1000 or not config.trainer.eval_video_one_per_pose or
            int(config.trainer.stop_after_optimizer_updates) not in (32, 128)):
        raise ValueError("Recovery changed the replay-repair isolation recipe")
    warmup = validate_learning_budget(256, int(config.batch_length), 10,
                                     int(config.buffer.max_size), int(config.env.steps))
    return dict(worlds=256, train_ratio=64, optimizer_update_cap=int(config.trainer.stop_after_optimizer_updates),
                collection_tick_budget=int(config.env.steps), conservative_replay_warmup_ticks=warmup,
                evaluation_seed_argument=1100, effective_evaluation_seed=11100,
                evaluation_worlds=32, evaluation_control_steps=1000,
                checkpoint_metric="walking_success_rate",
                gait_eval_interval_updates=8, trajectory_playback=False,
                initialization="all selected student weights; fresh optimizer, replay and training counters",
                reward_bc_and_exploration_unchanged=True)


def input_snapshot(pilot, initial=None, source_report=None):
    pilot = Path(pilot)
    required = ("demonstrations.npz", "demonstrations.manifest.json", "imitation/latest.pt",
                "imitation/pretraining_report.json", "imitation_eval/script_free_evaluation.json")
    optional = ("online/best.pt", "online/latest.pt", "independent_eval/script_free_evaluation.json")
    paths = [pilot / name for name in required]
    if any(not path.is_file() for path in paths):
        raise ValueError("Recovery requires the completed pilot's original imitation and demonstration evidence")
    paths.extend(pilot / name for name in optional if (pilot / name).is_file())
    if initial is not None:
        paths.append(Path(initial))
    if source_report is not None:
        paths.append(Path(source_report))
    return {str(path.resolve()): sha256(path) for path in paths}


def validate_selected_checkpoint(info, path, *, explicit=False, source_report=None):
    if (not info.get("finite_agent") or type(info.get("step")) is not int or info["step"] < 0 or
            type(info.get("optimizer_updates")) is not int or info["optimizer_updates"] < 0):
        raise ValueError("Recovery requires a finite checkpoint with verified training counters")
    if not explicit:
        if info["step"] != 0:
            raise ValueError("Default recovery must start from the original zero-step imitation checkpoint")
        return dict(kind="original_imitation", checkpoint=str(path), checkpoint_sha256=sha256(path),
                    previous_training_step=info["step"], previous_optimizer_updates=info["optimizer_updates"])
    if source_report is None:
        raise ValueError("Explicit recovery initialization requires its preserved stage report")
    report = read_json(source_report)
    if (report.get("checkpoint") != info or report.get("latest_checkpoint_sha256") != sha256(path) or
            report.get("preserved_inputs_unchanged") is not True or report.get("source_unchanged") is not True):
        raise ValueError("Selected checkpoint does not match its preserved recovery-stage evidence")
    return dict(kind="selected_recovery_stage", checkpoint=str(path), checkpoint_sha256=sha256(path),
                source_report=str(source_report), source_report_sha256=sha256(source_report),
                previous_training_step=info["step"], previous_optimizer_updates=info["optimizer_updates"],
                previous_stage_status=report.get("status"), fresh_optimizer_and_replay=True)


def source_snapshot():
    excluded = {".venv", "__pycache__", ".git", "logdir"}
    return {path.relative_to(ROOT).as_posix(): sha256(path) for path in sorted(ROOT.rglob("*"))
            if path.is_file() and path.suffix in (".py", ".yaml", ".xml", ".npz")
            and not excluded.intersection(path.relative_to(ROOT).parts)}


def validate_guard_history(rows, updates):
    if (not rows or rows[0].get("phase") != "baseline" or rows[0].get("optimizer_updates") != 0 or
            any(row.get("protocol") != rows[0].get("protocol") for row in rows)):
        raise ValueError("Missing initial same-seed physical baseline")
    protocol = rows[0]["protocol"]
    if (protocol.get("worlds") != 32 or protocol.get("seed") != 11100 or
            protocol.get("horizon_control_steps") != 1000 or protocol.get("repeatable") is not True):
        raise ValueError("Recovery gait checks do not match the declared 32-world, 20-second seed")
    failed = [row for row in rows if row.get("passed") is not True]
    if failed:
        return dict(passed=False, first_failure=failed[0], completed_updates=rows[-1]["optimizer_updates"])
    expected = list(range(0, updates + 1, 8))
    if [row["optimizer_updates"] for row in rows] != expected:
        raise ValueError("Recovery lacks every declared update-boundary gait check")
    return dict(passed=True, completed_updates=updates, baseline=rows[0], final=rows[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", "--old-pilot", dest="old_pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init-from", type=Path, help="Preserved recovery-stage latest.pt; starts fresh optimizer and replay")
    parser.add_argument("--updates", type=int, choices=(32, 128), default=32)
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()
    pilot, output = args.old_pilot.resolve(strict=True), args.output.resolve()
    if args.seed < 0 or output.exists() or output.is_relative_to(pilot) or output.is_relative_to(ROOT.parent):
        parser.error("Use a fresh recovery output outside the original pilot and recovery source")
    initial_path = (pilot / "imitation/latest.pt" if args.init_from is None else args.init_from.resolve(strict=True))
    source_report = None if args.init_from is None else initial_path.parent.parent / "recovery_report.json"
    if args.init_from is not None and (not source_report.is_file() or output.is_relative_to(source_report.parent)):
        parser.error("Use a checkpoint with a preserved recovery_report.json and a fresh sibling output")
    command = recovery_command(pilot, output, args.updates, args.seed, initial_path)
    recipe = validate_command(command)
    old_hashes, source_hashes = input_snapshot(pilot, initial_path, source_report), source_snapshot()
    from imitation_data import ImitationDataset
    from script_free_contract import check_imitation_source
    from scripts.run_script_free_pipeline import checkpoint_report
    dataset = ImitationDataset(pilot / "demonstrations.npz")
    initial = checkpoint_report(initial_path)
    check_imitation_source(dataset, initial["training_contract"], initial, minimum_updates=1000)
    initialization = validate_selected_checkpoint(initial, initial_path, explicit=args.init_from is not None,
                                                   source_report=source_report)
    output.mkdir(parents=True)
    journal = RunJournal(output, "replay_repair_recovery", args.seed)
    atomic_json(output / "input_admission.json", dict(recipe=recipe, command=command,
                source_sha256=source_hashes, preserved_input_sha256=old_hashes,
                dataset_fingerprint=dataset.fingerprint, initial_checkpoint=initial, initialization=initialization,
                walking_qualified=False, automatic_extension=False))
    del dataset
    environment = dict(os.environ, MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
    try:
        with stop_signals():
            log = run_stage(command, output / "online.log", 7200, cwd=ROOT, env=environment, journal=journal)
        if input_snapshot(pilot, initial_path, source_report) != old_hashes or source_snapshot() != source_hashes:
            raise ValueError("Preserved input or recovery source changed during the bounded stage")
        online = output / "online"
        rows = [json.loads(line) for line in (online / "gait_guard.jsonl").read_text(encoding="utf-8").splitlines()]
        guard = validate_guard_history(rows, args.updates)
        final = checkpoint_report(online / "latest.pt")
        if guard["passed"] and ("LEARNING VERIFIED:" not in log or not final.get("finite_agent") or
                                final.get("optimizer_updates") != args.updates or
                                final.get("stop_reason") != "optimizer_update_limit"):
            raise ValueError("Recovery lacks finite, capped optimizer evidence")
        status = "passed_bounded_gait_preservation_only" if guard["passed"] else "halted_gait_regression"
        report = dict(status=status, recipe=recipe, initialization=initialization, guard=guard, checkpoint=final,
                      latest_checkpoint_sha256=sha256(online / "latest.pt"),
                      baseline_checkpoint_sha256=sha256(online / "gait_baseline.pt"),
                      preserved_inputs_unchanged=True, source_unchanged=True,
                      walking_qualified=False, automatic_extension=False, automatic_resume=False,
                      qualification="Same-seed early-gait preservation only; independent seeds and video review remain required")
        atomic_json(output / "recovery_report.json", report)
        journal.update(status="completed" if guard["passed"] else "halted", child_pid=None,
                       stage="recovery_report.json", error=None if guard["passed"] else guard["first_failure"]["failures"])
        print(json.dumps(report, indent=2, allow_nan=False))
        if not guard["passed"]:
            raise SystemExit(2)
    except Exception as exc:
        journal.update(status="failed", child_pid=None, error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
