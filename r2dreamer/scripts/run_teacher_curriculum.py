"""Six sequential, bounded teacher adaptations. Run only on the prepared GPU host.

All agent weights carry forward via init_from. Optimizer/replay and local step
counters restart in each phase. This does not merge independently trained models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def phases():
    return [
        dict(name="01_standing", ticks=1000000, pose=None, task="standing"),
        *[dict(name=f"0{pose + 1}_pose_{pose}", ticks=1000000, pose=pose,
               task="walking") for pose in range(1, 5)],
        dict(name="06_mixed", ticks=3000000, pose=None, task="walking"),
    ]


def phase_overrides(phase, parent):
    options = [f"env.steps={phase['ticks']}", f"env.mjx_task={phase['task']}",
               f"init_from={json.dumps(str(parent))}"]
    if phase["task"] == "standing":
        options += ["env.rsi_stratified=false", "env.eval_pose_rsi=false",
                    "env.eval_balanced_pose_ids=false",
                    "trainer.best_metric_key=standing_stable_seconds"]
    elif phase["pose"] is not None:
        options += [f"env.training_pose_id={phase['pose']}"]
    else:
        # Category 'other' contains two poses: each of the five starts gets20%.
        options += ["env.rsi_probabilities=[0.2,0.2,0.2,0.4]"]
    return options


def checkpoint_metadata(path):
    # Keep the supervisor on CPU; only the child trainer creates CUDA state.
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint["agent_state_dict"]
    if not state or not all(bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError(f"Nonfinite or empty agent checkpoint: {path}")
    if not checkpoint.get("training_contract"):
        raise ValueError(f"Curriculum requires a recorded physical contract: {path}")
    with Path(path).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(path=str(path), sha256=digest, step=int(checkpoint["step"]),
                optimizer_updates=int(checkpoint["optimizer_updates"]),
                stop_reason=checkpoint.get("stop_reason"))


def validate_phase_result(phase, metadata, evaluation):
    if metadata["stop_reason"] is not None:
        raise ValueError("Interrupted/update-capped phases must not advance automatically")
    if not phase["ticks"] <= metadata["step"] < phase["ticks"] + 1280:
        raise ValueError("Checkpoint did not reach the declared phase budget")
    if metadata["optimizer_updates"] <= 0:
        raise ValueError("Phase completed without optimizer updates")
    if evaluation.get("step") != metadata["step"]:
        raise ValueError("Final evaluation does not match saved phase checkpoint")
    trials = evaluation.get("trials", [])
    ids = [trial.get("start_pose_id") for trial in trials]
    if phase["task"] == "standing":
        valid = len(ids) == 20 and all(pose == 0 for pose in ids)
    else:
        valid = len(ids) == 20 and all(ids.count(pose) == 4 for pose in range(5))
    if not valid:
        raise ValueError("Phase evaluation did not cover its declared starts")


def write_report(path, report):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-from", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    parent = args.init_from.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    schedule = phases()
    if args.dry_run:
        for phase in schedule:
            print(json.dumps(dict(**phase, command=[sys.executable, "-u", "train.py",
                "--config-name", "walking_mjx_teacher_curriculum",
                *phase_overrides(phase, parent)])))
            parent = output / phase["name"] / "latest.pt"
        return
    initial = checkpoint_metadata(parent)
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(XLA_PYTHON_CLIENT_PREALLOCATE="false", JAX_PLATFORM_NAME="gpu",
               JAX_PLATFORMS="cuda", MUJOCO_GL="egl",
               TEACHER_SCENE=str(ROOT / "envs/mjx_walking/assets/humanoid_rubber_2mm_candidate.xml"))
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT.parent), str(ROOT), env.get("PYTHONPATH", "")])
    report = dict(status="running", initial_checkpoint=initial,
                  planned_counted_ticks=sum(phase["ticks"] for phase in schedule),
                  semantics="Full agent init_from; fresh optimizer/replay and local counters per phase",
                  qualification="Experimental; completing budgets does not establish standing or walking",
                  phases=[])
    report_path = output / "curriculum.json"
    write_report(report_path, report)
    active = None
    stop = None

    def request_stop(signum, frame):
        nonlocal stop
        stop = signal.Signals(signum).name
        if active is not None and active.poll() is None:
            active.send_signal(signum)
        # A repeated foreground Ctrl+C only repeats the child's stop request.

    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for phase in schedule:
            if stop:
                raise RuntimeError(f"Curriculum stopped by {stop}; no next phase started")
            run = output / phase["name"]
            env["TEACHER_RUN_DIR"] = str(run)
            entry = dict(**phase, parent=checkpoint_metadata(parent), status="running")
            report["phases"].append(entry)
            write_report(report_path, report)
            command = [sys.executable, "-u", "train.py", "--config-name",
                       "walking_mjx_teacher_curriculum", *phase_overrides(phase, parent)]
            print(f"CURRICULUM_PHASE {phase['name']} budget={phase['ticks']} parent={parent}", flush=True)
            if stop:
                raise RuntimeError(f"Curriculum stopped by {stop}; no next phase started")
            active = subprocess.Popen(command, cwd=ROOT, env=env)
            if stop and active.poll() is None:
                active.send_signal(getattr(signal, stop))
            returncode = active.wait()
            active = None
            if stop or returncode:
                raise RuntimeError(f"Phase stopped: signal={stop}, returncode={returncode}")
            metadata = checkpoint_metadata(run / "latest.pt")
            evaluations = [json.loads(line) for line in
                           (run / "walking_evaluations.jsonl").read_text().splitlines() if line.strip()]
            validate_phase_result(phase, metadata, evaluations[-1])
            videos = sorted((run / "eval_videos").glob(f"step_{metadata['step']:07d}*.mp4"))
            expected_videos = 1 if phase["task"] == "standing" else 5
            if len(videos) != expected_videos or any(path.stat().st_size == 0 for path in videos):
                raise RuntimeError(f"Expected {expected_videos} phase-end videos; inspect renderer/codec logs")
            entry.update(status="completed_budget", checkpoint=metadata,
                         final_evaluation=evaluations[-1], videos=[str(path) for path in videos])
            write_report(report_path, report)
            parent = run / "latest.pt"
        report["status"] = "completed_budgets_not_gait_qualified"
    except BaseException as exc:
        report.update(status="stopped", error=f"{type(exc).__name__}: {exc}")
        if report["phases"] and report["phases"][-1]["status"] == "running":
            report["phases"][-1]["status"] = "stopped"
        raise
    finally:
        write_report(report_path, report)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    print(f"CURRICULUM_COMPLETE report={report_path} final_checkpoint={parent}", flush=True)


if __name__ == "__main__":
    main()
