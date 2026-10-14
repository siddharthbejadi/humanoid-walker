"""Bounded GPU/runtime/video checks for the six-phase supervisor, no long run."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from run_teacher_curriculum import ROOT, checkpoint_metadata, phase_overrides, phases
from teacher_capacity_probe import case_overrides


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-from", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--completed-pose-checks", type=Path,
                        help="Reuse five unchanged reset/step checks from this prior attempt")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    parent = args.init_from.resolve(strict=True)
    checkpoint_metadata(parent)
    env = os.environ.copy()
    env.update(MUJOCO_GL="egl", XLA_PYTHON_CLIENT_PREALLOCATE="false",
               JAX_PLATFORM_NAME="gpu", JAX_PLATFORMS="cuda",
               TEACHER_SCENE=str(ROOT / "envs/mjx_walking/assets/humanoid_rubber_2mm_candidate.xml"),
               TEACHER_RUN_DIR=str(output / "preflight_not_training"))
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT.parent), str(ROOT), env.get("PYTHONPATH", "")])
    checks = []

    def run(name, command):
        print(f"CURRICULUM_PREFLIGHT {name}", flush=True)
        with (output / f"{name}.log").open("w") as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=180)
        if result.returncode:
            raise RuntimeError(f"{name} failed ({result.returncode}); inspect {output / (name + '.log')}")
        checks.append(name)

    common = [sys.executable, "scripts/preflight_mjx_teacher.py", "--config-name",
              "walking_mjx_teacher_curriculum", "--scene", env["TEACHER_SCENE"]]
    for phase in phases()[:5]:
        if args.completed_pose_checks:
            prior = args.completed_pose_checks / (phase["name"] + "_reset_and_step.log")
            if "MJX_TEACHER_PREFLIGHT_OK worlds=8" not in prior.read_text():
                raise RuntimeError(f"No previous successful reset/step check: {prior}")
            checks.append(phase["name"] + "_reset_and_step (previous unchanged run)")
            continue
        command = common + ["--worlds", "8"]
        for override in phase_overrides(phase, parent):
            command += ["--override", override]
        run(phase["name"] + "_reset_and_step", command)
    run("walking_balanced_evaluation", common + ["--worlds", "20", "--evaluation"])
    command = common + ["--worlds", "20", "--evaluation"]
    for override in phase_overrides(phases()[0], parent):
        command += ["--override", override]
    run("standing_evaluation", command)

    for phase in (phases()[0], phases()[1]):
        run_dir = output / (phase["name"] + "_8updates")
        env["TEACHER_RUN_DIR"] = str(run_dir)
        options = phase_overrides(phase, parent) + case_overrides(128, 8)
        # Exercise the actual renderer/MP4 path at step0 and after8updates.
        options += ["trainer.eval_video=true", "trainer.eval_video_final_only=false",
                    "trainer.eval_every=100000000"]
        run(phase["name"] + "_learner_video", [sys.executable, "-u", "train.py",
            "--config-name", "walking_mjx_teacher_curriculum", *options])
        metadata = checkpoint_metadata(run_dir / "latest.pt")
        if metadata["optimizer_updates"] != 8 or metadata["stop_reason"] != "optimizer_update_limit":
            raise RuntimeError("Smoke did not complete exactly8 optimizer updates")
        videos = list((run_dir / "eval_videos").glob(f"step_{metadata['step']:07d}*.mp4"))
        if len(videos) != (1 if phase["task"] == "standing" else 5):
            raise RuntimeError(f"Missing phase-end diagnostic MP4s in {run_dir}")
    report = dict(status="passed", checks=checks, scope="GPU resets,8update128-world learners and MP4 output; no gait qualification")
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
