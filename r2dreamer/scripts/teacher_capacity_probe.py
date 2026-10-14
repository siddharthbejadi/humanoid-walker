"""Bounded complete-learner GPU checks in isolated processes; never auto-launch a long run.

Unlike the historical collection-only scale smoke, each case constructs both
environments, the learner and GPU replay, updates weights, evaluates, and reads
back its checkpoint. A pass means short-run capacity only, not gait quality or
proof that a long run cannot OOM.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def case_overrides(worlds, updates):
    # 67 control frames covers the existing conservative fresh-replay gate.
    warmup = 67 * worlds * 10
    # Permit all requested updates even at small world count/ratio16. The
    # update cap normally stops first; a time cap bounds failed startup.
    budget = warmup + (updates + 2) * 1024 * 10 + worlds * 10
    return [f"env.env_num={worlds}", f"buffer.max_size={worlds * 128}",
            f"env.steps={budget}", f"trainer.learning_start_deadline_steps={warmup + worlds * 20}",
            f"trainer.stop_after_optimizer_updates={updates}",
            "trainer.profile_synchronize=true", "trainer.eval_video=false",
            "trainer.train_video=false", "trainer.video_pred_log=false"]


def validate_case(checkpoint, evaluations, requested_updates):
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint metadata must be a mapping")
    if not isinstance(checkpoint.get("step"), int) or checkpoint["step"] <= 0:
        raise ValueError("Checkpoint has no completed transitions")
    if checkpoint.get("optimizer_updates") != requested_updates:
        raise ValueError("Checkpoint did not reach the exact bounded optimizer-update count")
    if checkpoint.get("stop_reason") != "optimizer_update_limit":
        raise ValueError("Trainer did not stop at its requested optimizer-update limit")
    if not checkpoint.get("finite_agent"):
        raise ValueError("Checkpoint has empty or nonfinite agent weights")
    if not isinstance(evaluations, list) or len(evaluations) < 2:
        raise ValueError("Initial and final complete-learner evaluations are required")
    final = evaluations[-1]
    if not isinstance(final, dict) or final.get("step") != checkpoint["step"]:
        raise ValueError("Final evaluation and checkpoint do not match")
    if not isinstance(evaluations[0], dict) or evaluations[0].get("step") != 0:
        raise ValueError("A fresh step-zero initial evaluation is required")
    for evaluation in (evaluations[0], final):
        trials = evaluation.get("trials")
        if not isinstance(trials, list) or not all(isinstance(trial, dict) for trial in trials):
            raise ValueError("Evaluation trials are missing or malformed")
        ids = [trial.get("start_pose_id") for trial in trials]
        if len(ids) != 20 or any(ids.count(pose) != 4 for pose in range(5)):
            raise ValueError("Evaluation did not cover each of five starts four times")


def read_checkpoint(path):
    # CPU-only readback in another process keeps this supervisor free of CUDA
    # allocations before the next case. Load only checkpoints created by us.
    source = (
        "import json,sys,torch; "
        "c=torch.load(sys.argv[1],map_location='cpu',weights_only=False); "
        "s=c['agent_state_dict']; "
        "print(json.dumps(dict(step=int(c['step']),optimizer_updates=int(c['optimizer_updates']),"
        "stop_reason=c.get('stop_reason'),"
        "finite_agent=bool(s) and all(bool(torch.isfinite(v).all()) for v in s.values()))))"
    )
    result = subprocess.run([sys.executable, "-c", source, str(path)],
                            capture_output=True, text=True, check=True, timeout=120)
    return json.loads(result.stdout.strip().splitlines()[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--updates", type=int, default=32)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if any(w < 8 or w > 256 for w in args.worlds) or len(set(args.worlds)) != len(args.worlds):
        parser.error("Use distinct world counts between 8 and 256")
    if not 1 <= args.updates <= 256 or args.timeout_seconds <= 0:
        parser.error("updates must be 1..256 and timeout must be positive")
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("Output directory must be empty; previous evidence is never overwritten")
    commands = [[sys.executable, "-u", "train.py", "--config-name", "walking_mjx_teacher_next",
                 *case_overrides(w, args.updates)] for w in args.worlds]
    if args.dry_run:
        print(json.dumps({"cwd": str(ROOT), "commands": commands,
                          "scope": "No GPU checks or training executed"}, indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false", JAX_PLATFORMS="cuda",
               JAX_PLATFORM_NAME="gpu", PYTHONUNBUFFERED="1")
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT.parent), str(ROOT), env.get("PYTHONPATH", "")])
    env["TEACHER_SCENE"] = str(ROOT / "envs/mjx_walking/assets/humanoid_rubber_2mm_candidate.xml")
    results = []
    for worlds, command in zip(args.worlds, commands):
        run = output / f"worlds_{worlds}"
        run.mkdir()
        env["TEACHER_RUN_DIR"] = str(run)
        result = {"worlds": worlds, "requested_updates": args.updates,
                  "command": command, "status": "failed"}
        started = time.monotonic()
        print(f"FULL_TEACHER_PROBE worlds={worlds} updates={args.updates}", flush=True)
        try:
            with (run / "launcher.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    result["returncode"] = process.wait(timeout=args.timeout_seconds)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise RuntimeError("Probe timed out; no capacity pass")
            if result["returncode"]:
                raise RuntimeError(f"Trainer exited {result['returncode']}; inspect launcher.log")
            log_text = (run / "launcher.log").read_text(encoding="utf-8", errors="replace")
            if "LEARNING VERIFIED:" not in log_text:
                raise ValueError("No verified changed/finite first optimizer update")
            # An overflow can silently truncate contacts even when the trainer
            # exits successfully. Preserve the log and reject this capacity.
            if "overflow" in log_text.lower():
                raise ValueError("Overflow mentioned in trainer log; inspect before accepting capacity")
            checkpoint = read_checkpoint(run / "latest.pt")
            evaluations = [json.loads(line) for line in
                           (run / "walking_evaluations.jsonl").read_text().splitlines() if line.strip()]
            validate_case(checkpoint, evaluations, args.updates)
            metrics_path = run / "metrics.jsonl"
            if metrics_path.exists():
                for line in metrics_path.read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    if "perf/environment_transitions_per_second" in row:
                        result["performance"] = {key: value for key, value in row.items()
                                                 if key.startswith("perf/")}
            with (run / "latest.pt").open("rb") as checkpoint_file:
                digest = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()
            result.update(checkpoint, status="passed_short_capacity_only", checkpoint_sha256=digest)
        except (OSError, ValueError, RuntimeError, TypeError, KeyError,
                subprocess.SubprocessError) as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        result["wall_seconds"] = time.monotonic() - started
        results.append(result)
        (output / "capacity_report.json").write_text(json.dumps({"cases": results,
            "scope": "Short capacity checks only; no long-run VRAM or walking qualification"}, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        if result["status"] == "failed":
            print("Stopping the sweep at the first failure. Smaller successful cases remain recorded.")
            break
    if any(case["status"] == "failed" for case in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
