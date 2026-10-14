"""Exact reference recipe checks; GPU baseline and 32-update capacity are opt-in.

Physical reference outcomes are recorded independently of runtime admission.
An unsuccessful reference is an experiment, not a qualified walking teacher.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import hydra
import numpy as np
from omegaconf import OmegaConf
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG = "walking_mjx_reference_pilot"
CORE_RUNTIME = {"torch": "2.8.0", "torchrl": "0.9.2", "tensordict": "0.9.1",
                "mujoco": "3.14.0", "mujoco-mjx": "3.14.0", "jax": "0.10.2",
                "jaxlib": "0.10.2", "warp-lang": "1.17.0", "numpy": "2.4.6"}


def validate_runtime_versions(provider=metadata.version, python_version=None):
    """Validate the pinned core; a version provider allows CPU-only admission tests."""
    python_version = sys.version_info[:2] if python_version is None else tuple(python_version[:2])
    if tuple(python_version) != (3, 11):
        raise RuntimeError(f"Pinned runtime requires Python 3.11, found {python_version}")
    versions = {name: str(provider(name)) for name in CORE_RUNTIME}
    mismatches = {name: dict(expected=expected, actual=versions[name])
                  for name, expected in CORE_RUNTIME.items() if Version(versions[name]).public != expected}
    if mismatches:
        raise RuntimeError("Pinned core runtime mismatch: " + json.dumps(mismatches, sort_keys=True))
    return dict(python="3.11", versions=versions,
                scope="Pinned core package identity; GPU execution and full-learner capacity still required")


def validate_cuda_build(version):
    if not version or Version(str(version)).major != 12:
        raise RuntimeError(f"Pinned runtime requires a Torch CUDA 12 build, found {version!r}")
    return str(version)


def gpu_runtime_report():
    report = validate_runtime_versions()
    import torch
    import jax

    validate_cuda_build(torch.version.cuda)
    if not torch.cuda.is_available():
        raise RuntimeError("Pinned runtime requires working Torch CUDA; refusing CPU fallback")
    if jax.default_backend() != "gpu" or not jax.devices("gpu"):
        raise RuntimeError("Pinned runtime requires working JAX GPU; refusing CPU fallback")
    properties = torch.cuda.get_device_properties(0)
    report.update(torch_cuda_build=torch.version.cuda, torch_gpu_name=properties.name,
                  torch_gpu_vram_bytes=int(properties.total_memory),
                  jax_devices=[str(device) for device in jax.devices("gpu")])
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                 "--format=csv,noheader"], capture_output=True, text=True, timeout=15, check=True)
        report["nvidia_smi"] = result.stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        report["nvidia_smi_unavailable"] = f"{type(exc).__name__}: {exc}"
    return report


def compose(overrides=()):
    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return hydra.compose(CONFIG, overrides=list(overrides))


def asset(path):
    value = Path(str(path))
    return (value if value.is_absolute() else ROOT / value).resolve(strict=True)


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def structural_report(config):
    from training_budget import validate_learning_budget

    for name in ("resume_from", "init_from", "transfer_from"):
        if config[name] is not None:
            raise ValueError(f"The reference pilot must start fresh: {name}")
    if bool(config.allow_unverified_resume):
        raise ValueError("Unverified checkpoint resume is forbidden")
    env = config.env
    if (int(env.env_num), int(env.steps), int(env.action_repeat), int(env.train_ratio)) != (128, 1000000, 10, 64):
        raise ValueError("The pilot budget/world count/replay ratio contract changed")
    if int(env.eval_episode_num) != 32 or not bool(env.eval_repeatable):
        raise ValueError("The pilot requires 32 repeatable held-out standing trials")
    if list(env.eval_rsi_probabilities) != [1., 0., 0., 0.] or bool(env.eval_balanced_pose_ids):
        raise ValueError("The held-out evaluation must start standing")
    if list(env.rsi_probabilities) != [.15, 0., 0., .85] or env.pose_rsi_json is not None:
        raise ValueError("Use the scripted phase pool, not manual hold validation")
    if env.mjx_task != "walking" or env.mjx_reward_profile != "goal_corridor_v3":
        raise ValueError("Unexpected task/reward contract")
    if not bool(env.mjx_gait_residual) or not bool(env.mjx_phase_obs):
        raise ValueError("Reference residual and phase observation must both be enabled")
    for name in ("mjx_leg_action_scale", "mjx_arm_action_scale", "mjx_other_action_scale"):
        if float(env[name]) != .1:
            raise ValueError(f"Unexpected residual action authority: {name}")
    if (float(env.mjx_motor_torque_kgcm), float(env.mjx_control_slew_rad_s),
            float(env.mjx_step_clearance_m)) != (18., 8., .015):
        raise ValueError("Unexpected torque/slew/15 mm detector settings")
    if list(env.mjx_stabilizer) != [1., .04, -1., .25]:
        raise ValueError("The tilt stabilizer is part of the declared controller")
    dist = config.model.actor.dist.cont
    if (float(dist.min_std), float(dist.max_std), int(config.model.warmup)) != (.005, .03, 100):
        raise ValueError("Unexpected actor exploration or optimizer warmup")
    trainer = config.trainer
    if bool(trainer.eval_video_one_per_pose):
        raise ValueError("Capacity requires independent final videos for all 32 evaluation worlds")
    if (trainer.best_metric_key, trainer.best_safety_metric_key,
            trainer.best_tiebreak_metric_key, trainer.checkpoint_order) != (
                "total_valid_steps", "safety_failure_count", "alternating_steps", "safety_first"):
        raise ValueError("Physical checkpoint selection contract changed")
    if int(config.buffer.max_size) != 131072:
        raise ValueError("Replay capacity must remain 131072 transitions")
    warmup = validate_learning_budget(128, int(config.batch_length), 10,
                                      int(config.buffer.max_size), int(env.steps))
    if int(trainer.learning_start_deadline_steps) <= warmup:
        raise ValueError("Learning deadline does not cover fresh-replay warmup")
    scene = asset(env.scene_path)
    if scene.name != "humanoid_rubber_2mm_servo_v2_kp40.xml":
            raise ValueError("Use the selected candidate servo_v2_kp40 scene")
    tree = ET.parse(scene)
    actuators = tree.getroot().findall("actuator/position")
    if len(actuators) != 19 or any(float(a.get("kp", "nan")) != 40. or
                                  float(a.get("kv", "nan")) != .6 for a in actuators):
        raise ValueError("The scene must contain 19 kp40/kv0.6 position actuators")
    compiler = tree.getroot().find("compiler")
    meshdir = scene.parent / (compiler.get("meshdir", "") if compiler is not None else "")
    meshes = [(meshdir / mesh.get("file")).resolve(strict=True)
              for mesh in tree.getroot().findall("asset/mesh") if mesh.get("file")]
    table_path, pool_path = asset(env.mjx_scripted_gait), asset(env.mjx_scripted_phase_pool)
    with np.load(table_path, allow_pickle=False) as table:
        ctrl = table["ctrl"]
        start, length, dt = int(table["cycle_start"]), int(table["cycle_len"]), float(table["dt"])
        if ctrl.ndim != 2 or ctrl.shape[1] != 19 or not np.isfinite(ctrl).all():
            raise ValueError("Reference control table is malformed or nonfinite")
        if not (0 <= start < len(ctrl) and 0 < length <= len(ctrl) - start and dt == .02):
            raise ValueError("Reference cycle or 50 Hz clock is invalid")
        table_shape = list(ctrl.shape)
    with np.load(pool_path, allow_pickle=False) as pool:
        n = int(pool["n"])
        for key, width in (("qpos", 26), ("qvel", 25), ("ctrl", 19)):
            if pool[key].shape != (n + 1, width) or not np.isfinite(pool[key]).all():
                raise ValueError(f"Malformed/nonfinite scripted pool field: {key}")
        offsets = pool["phase_offset"]
        if n < 32 or offsets.shape != (n + 1,) or not np.isfinite(offsets).all() or np.any(offsets < 0):
            raise ValueError("Scripted phase schedule is invalid")
    paths = [scene, table_path, pool_path, *meshes,
             ROOT / "envs/mjx_walking/env.py", ROOT / "envs/mjx_walking/scene.py",
             ROOT / "envs/mjx_teacher_env.py", ROOT / "training_contract.py",
             ROOT / "trainer.py", ROOT / "configs" / f"{CONFIG}.yaml"]
    return dict(status="passed_structural_only", gpu_tested=False,
                physical_reference_qualified=False, config=OmegaConf.to_container(config, resolve=True),
                source_sha256={path.relative_to(ROOT).as_posix(): digest(path) for path in paths},
                replay_warmup_counted_ticks=warmup, reference_ctrl_shape=table_shape,
                cycle_start=start, cycle_len=length, phase_pool_states=n,
                scope="Asset/config closure; no CUDA execution, learned walking or hardware qualification")


def reset_snapshot(env, transition):
    result = {name: transition[name].detach().cpu().numpy().copy()
              for name in ("proprio", "teacher_state")}
    data = env.state.pipeline_state
    for name in ("qpos", "qvel", "ctrl", "time"):
        result[name] = np.asarray(env.jax.device_get(getattr(data, name))).copy()
    for target, source in (("step_count", "step_count"), ("pose_ids", "rsi_pose_id"), ("state_rng", "rng")):
        result[target] = np.asarray(env.jax.device_get(env.state.info[source])).copy()
    result["bridge_rng"] = np.asarray(env.jax.device_get(env.rng)).copy()
    for name in ("phase_offset", "gait_phase", "stab_prev", "stab_filt", "injected"):
        if name in env.state.info:
            result["history_" + name] = np.asarray(env.jax.device_get(env.state.info[name])).copy()
    return result


def validate_histories(first, repeated):
    names = [name for name in first if name.startswith("history_")]
    if not names or any(name not in repeated or not np.array_equal(first[name], repeated[name]) for name in names):
        raise AssertionError("Reference/stabilizer reset history is not exactly repeatable")
    return names


def gpu_baseline(config, output):
    runtime = gpu_runtime_report()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Torch CUDA unavailable; refusing CPU fallback")
    import jax
    if jax.default_backend() != "gpu":
        raise RuntimeError("JAX GPU unavailable; refusing CPU fallback")
    from envs.mjx_teacher_env import MJXTeacherEnv
    from reset_validation import validate_reset_repeatability
    from checkpoint_selection import walking_trial_success
    import imageio.v2 as imageio

    summaries = []
    for kind in ("standing", "fixed_phase"):
        cfg = OmegaConf.create(OmegaConf.to_container(config.env, resolve=True))
        cfg.eval_rsi_probabilities = [1., 0., 0., 0.] if kind == "standing" else [0., 0., 0., 1.]
        env = MJXTeacherEnv(cfg, int(cfg.eval_episode_num), evaluation=True)
        writer = None
        try:
            worlds = env.env_num
            indices = np.zeros(worlds, dtype=np.int32)
            if kind == "fixed_phase":
                if env.raw.rsi_source_kind != "scripted_phase" or env.raw.gait_reference_sample_offset != -1.:
                    raise AssertionError("Scripted pool/controller identity is incorrect")
                indices = np.linspace(1, len(env.raw.rsi_pose_ids) - 1, worlds, dtype=np.int32)
                if np.any(np.asarray(jax.device_get(env.raw.rsi_injected))[indices]):
                    raise AssertionError("Scripted phase states were classified as live injected poses")
                jindices = env.jnp.asarray(indices)
                env._reset_batch = jax.jit(lambda keys: jax.vmap(env.raw.reset)(keys, jindices))
            expected_ids = np.asarray(jax.device_get(env.raw.rsi_pose_ids))[indices].tolist()
            action = torch.zeros((worlds, 19), device="cuda:0")
            done = torch.ones(worlds, dtype=torch.bool, device="cuda:0")
            env.begin_evaluation()
            first, _ = env.step(action, done)
            snapshot = reset_snapshot(env, first)
            for _ in range(3):
                env.step(action, torch.zeros_like(done))
            env.begin_evaluation()
            repeated, ended = env.step(action, done)
            repeat_snapshot = reset_snapshot(env, repeated)
            retained = {name: first[name].detach().cpu().numpy().copy()
                        for name in ("proprio", "teacher_state")}
            reset_report = validate_reset_repeatability(snapshot, repeat_snapshot, retained, expected_ids)
            reset_report["exact_reference_history"] = validate_histories(snapshot, repeat_snapshot)
            if not reset_report["passed"]:
                raise AssertionError("Reset/retained-view numerical validation failed: " + ", ".join(reset_report["errors"]))
            if np.any(np.asarray(jax.device_get(env.state.info["injected"]))):
                raise AssertionError("Baseline reset entered the live pose injection controller")
            # Saved phase p is the state after ctrl[p-1]; its next target is ctrl[p].
            next_phase = env.state.info["gait_phase"] + 1.
            sampled = np.asarray(jax.device_get(jax.vmap(env.raw._reference_at)(next_phase)))
            with np.load(asset(cfg.mjx_scripted_gait), allow_pickle=False) as table:
                phases = np.asarray(jax.device_get(next_phase)).astype(np.int64) - 1
                phases = np.maximum(phases, 0)
                start, length = int(table["cycle_start"]), int(table["cycle_len"])
                source_indices = np.where(phases < start, phases, start + (phases - start) % length)
                if not np.allclose(sampled, table["ctrl"][source_indices], atol=1e-6, rtol=0):
                    raise AssertionError("Scripted reference/reset clock is misaligned")
            base = np.asarray(jax.device_get(jax.vmap(env.raw._residual_base)(
                env.state.info, env.state.info["step_count"] + 1, next_phase)))
            if not np.allclose(base, sampled, rtol=0., atol=1e-6):
                raise AssertionError("Scripted phase reset entered an injected-pose hold/blend")
            sum_names = ("alive_seconds", "valid_left_step", "valid_right_step", "total_valid_steps",
                         "alternating_steps", "physical_fall", "fall_count", "safety_failure_count", "wall_hit", "stall_out",
                         "goal_reached", "terminated", "time_limit", "forward_distance", "scrub_distance", "single_support_seconds",
                         "left_support_seconds", "right_support_seconds", "dots_reached")
            mean_names = ("body_tilt_rad_mean", "tracking_error_rad_mean", "torque_saturation_fraction_mean",
                          "control_slew_limited_fraction_mean", "left_load_bw_mean", "right_load_bw_mean")
            peak_names = ("body_tilt_rad_peak", "root_drop_m_peak", "left_clearance_m_peak", "right_clearance_m_peak")
            sums = {name: np.zeros(worlds, dtype=np.float64) for name in (*sum_names, *mean_names)}
            peaks = {name: np.zeros(worlds, dtype=np.float64) for name in peak_names}
            samples = np.zeros(worlds, dtype=np.int32)
            finished = np.zeros(worlds, dtype=bool)
            writer = imageio.get_writer(output / f"zero_residual_{kind}_world0.mp4", fps=25)
            with (output / f"zero_residual_{kind}_trace.jsonl").open("x", encoding="utf-8") as trace:
                for step in range(int(config.trainer.eval_max_steps)):
                    transition, ended = env.step(action, ended)
                    active = ~finished
                    samples += active
                    rows = {}
                    for name in (*sum_names, *mean_names, *peak_names):
                        key = "log_" + name
                        if key not in transition:
                            raise RuntimeError(f"Missing diagnostic metric: {key}")
                        value = transition[key].detach().cpu().numpy().reshape(worlds)
                        if not np.isfinite(value).all():
                            raise RuntimeError(f"Nonfinite diagnostic metric: {key}")
                        rows[name] = value.tolist()
                        if name in peaks:
                            peaks[name] = np.maximum(peaks[name], np.where(active, value, 0.))
                        else:
                            sums[name] += np.where(active, value, 0.)
                    trace.write(json.dumps(dict(control_step=step + 1, active_worlds=np.flatnonzero(active).tolist(),
                                                metrics=rows), allow_nan=False) + "\n")
                    world_zero_ended = bool(ended[0].item())
                    if (step % 2 == 0 or world_zero_ended) and active[0]:
                        record = env.render_selected([0], width=320, height=240)[0]
                        if record is None or record["error"] or record["frame"] is None:
                            raise RuntimeError(f"Baseline rendering failed: {record}")
                        writer.append_data(record["frame"])
                    finished |= ended.detach().cpu().numpy()
                    if finished.all():
                        break
            trials = []
            for world in range(worlds):
                trial = {name: float(value[world]) for name, value in sums.items()}
                for name in mean_names:
                    trial[name] /= max(1, int(samples[world]))
                trial.update({name: float(value[world]) for name, value in peaks.items()})
                trial.update(world=world, start_pose_id=int(expected_ids[world]),
                             pool_index=int(indices[world]), initial_phase=float(snapshot["history_gait_phase"][world]),
                             control_samples=int(samples[world]), horizon_censored=bool(not finished[world]))
                trials.append(trial)
            qualified = all(walking_trial_success(trial) for trial in trials)
            summaries.append(dict(start_group=kind, reset_validation=reset_report, trials=trials,
                                  physical_reference_qualified=qualified,
                                  video=f"zero_residual_{kind}_world0.mp4"))
        finally:
            if writer is not None:
                writer.close()
            env.close()
    return dict(status="passed_gpu_runtime_only", gpu_tested=True, learned_policy_evaluated=False,
                runtime=runtime,
                physical_reference_qualified=all(group["physical_reference_qualified"] for group in summaries),
                controller="humanlike reference + tilt stabilizer, ZERO learned residual",
                groups=summaries,
                scope="Failed bilateral reference motion is recorded; it does not masquerade as a runtime failure or teacher qualification")


def validate_capacity(checkpoint, evaluations, requested=32):
    if checkpoint.get("optimizer_updates") != requested or checkpoint.get("stop_reason") != "optimizer_update_limit":
        raise ValueError("The full learner did not stop at exactly 32 verified updates")
    if not checkpoint.get("finite_agent") or int(checkpoint.get("step", 0)) <= 0:
        raise ValueError("The full learner checkpoint is empty/nonfinite or has no transitions")
    if len(evaluations) < 2 or evaluations[0].get("step") != 0 or evaluations[-1].get("step") != checkpoint["step"]:
        raise ValueError("Fresh initial and matching final capacity evaluation are required")
    for row in (evaluations[0], evaluations[-1]):
        protocol = row.get("protocol", {}).get("environment", {})
        if (row.get("schema_version") != 2 or protocol.get("worlds") != 32 or
                protocol.get("repeatable") is not True or protocol.get("seed") != 11000 or
                protocol.get("rsi_probabilities") != [1., 0., 0., 0.] or
                protocol.get("reset_source_kind") != "scripted_phase" or
                protocol.get("controller_sample_offset") != -1. or
                protocol.get("valid_step_clearance_m") != .015):
            raise ValueError("Capacity evaluation protocol does not match the held-out reference recipe")
        contract = row.get("training_contract", {})
        controller = contract.get("controller", {})
        if (contract.get("task", {}).get("valid_step_clearance_m") != .015 or
                controller.get("reference_kind") != "scripted" or
                controller.get("gait_residual") is not True or
                controller.get("reference_sample_offset_control_steps") != -1. or
                controller.get("stabilizer") != [1., .04, -1., .25]):
            raise ValueError("Capacity evidence lacks the exact controller/detector contract")
        trials = row.get("trials")
        if not isinstance(trials, list) or len(trials) != 32 or any(t.get("start_pose_id") != 0 for t in trials):
            raise ValueError("Capacity must include all 32 standing evaluation worlds")
        for trial in trials:
            required = {"alive_seconds", "total_valid_steps", "valid_left_step", "valid_right_step",
                        "alternating_steps", "fall_count", "safety_failure_count", "wall_hit", "stall_out", "goal_reached"}
            if not required <= trial.keys():
                raise ValueError("Physical/cause metrics are missing from capacity evidence")
            for value in trial.values():
                if isinstance(value, (float, int)) and not np.isfinite(value):
                    raise ValueError("Nonfinite capacity evaluation")


def validate_capacity_videos(run, step, worlds=32):
    """Require every final body-view clip and its matching per-world sidecar."""
    if type(step) is not int or step <= 0 or worlds != 32:
        raise ValueError("Final capacity video admission requires a positive step and 32 worlds")
    video_dir = Path(run) / "eval_videos"
    expected = {f"step_{step:07d}" + ("" if world == 0 else f"_env_{world:02d}") + ".mp4"
                for world in range(worlds)}
    actual = {path.name for path in video_dir.glob(f"step_{step:07d}*.mp4") if path.is_file()}
    if actual != expected:
        raise ValueError("Incomplete final capacity video set: " + json.dumps(
            dict(missing=sorted(expected - actual), unexpected=sorted(actual - expected))))
    records = []
    for world in range(worlds):
        name = f"step_{step:07d}" + ("" if world == 0 else f"_env_{world:02d}") + ".mp4"
        path = video_dir / name
        metadata_path = path.with_suffix(".json")
        if path.stat().st_size <= 0 or not metadata_path.is_file():
            raise ValueError(f"Empty final video or missing metadata: {name}")
        record = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise ValueError(f"Malformed final video metadata: {name}")
        for key, wanted in (("training_step", step), ("evaluation_world", world), ("start_pose_id", 0)):
            if type(record.get(key)) is not int or record[key] != wanted:
                raise ValueError(f"Final video metadata mismatch {key}: {name}")
        if (type(record.get("frames")) is not int or record["frames"] <= 0 or
                isinstance(record.get("fps"), bool) or record.get("fps") != 25):
            raise ValueError(f"Invalid final video frames/fps: {name}")
        records.append(dict(path=str(path), metadata=str(metadata_path), **record))
    return dict(training_step=step, worlds=worlds, records=records,
                scope="Exact final filenames, nonempty files and matching sidecars; codec contents are not decoded by this gate")


def capacity_probe(config, output, overrides):
    from scripts.teacher_capacity_probe import case_overrides, read_checkpoint

    # Metadata validation stays CPU-only in the supervisor, so the learner
    # starts in a fresh process with the complete device memory available.
    runtime = validate_runtime_versions()
    build = subprocess.run([sys.executable, "-c", "import json,torch; print(json.dumps(torch.version.cuda))"],
                           capture_output=True, text=True, timeout=120, check=True)
    runtime["torch_cuda_build"] = validate_cuda_build(json.loads(build.stdout.strip().splitlines()[-1]))
    run = output / "worlds_128"
    run.mkdir()
    # Reuse the existing budget/update-cap mechanism; preserve the pilot's
    # exact replay size and its full 32-world evaluation/rendering allocation.
    command = [sys.executable, "-u", "train.py", "--config-name", CONFIG,
               *case_overrides(128, 32), "buffer.max_size=131072", "trainer.eval_video=true",
               "trainer.eval_video_every=1", f"logdir={run.as_posix()}", *overrides]
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false", JAX_PLATFORMS="cuda", JAX_PLATFORM_NAME="gpu")
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")])
    env["TEACHER_RUN_DIR"] = str(run)
    with (run / "launcher.log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=1200)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise RuntimeError("Full-learner capacity timed out; pilot not admitted")
    if code:
        raise RuntimeError(f"Full-learner capacity exited {code}; inspect launcher.log")
    log_text = (run / "launcher.log").read_text(encoding="utf-8", errors="replace")
    if "LEARNING VERIFIED:" not in log_text or "overflow" in log_text.lower():
        raise RuntimeError("Unverified changed weights or overflow in capacity log")
    checkpoint = read_checkpoint(run / "latest.pt")
    evaluations = [json.loads(line) for line in (run / "walking_evaluations.jsonl").read_text().splitlines() if line.strip()]
    validate_capacity(checkpoint, evaluations)
    videos = validate_capacity_videos(run, checkpoint["step"])
    return dict(status="passed_short_capacity_only", command=command, checkpoint=checkpoint,
                runtime=runtime,
                checkpoint_sha256=digest(run / "latest.pt"), training_worlds=128, evaluation_worlds=32,
                final_video_evidence=videos,
                physical_reference_qualified=False,
                scope="32 complete-learner updates plus full evaluation/video allocation; no long-run or walking guarantee")


def gpu_worker(config, output, overrides):
    # A child process captures native CUDA/Warp stdout as well as Python logs,
    # and releases every GPU allocation before the full-learner capacity case.
    command = [sys.executable, str(Path(__file__).resolve()), "--mode", "gpu",
               "--gpu-worker", "--output", str(output), *overrides]
    log_path = output / "gpu_runtime.log"
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=1200)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise RuntimeError("GPU reference baseline timed out; pilot not admitted")
    if code:
        raise RuntimeError(f"GPU reference baseline exited {code}; inspect gpu_runtime.log")
    if "overflow" in log_path.read_text(encoding="utf-8", errors="replace").lower():
        raise RuntimeError("Overflow mentioned in GPU runtime log; pilot not admitted")
    report = json.loads((output / "gpu_baseline_report.json").read_text(encoding="utf-8"))
    if report.get("status") != "passed_gpu_runtime_only":
        raise RuntimeError("GPU worker did not produce its required runtime report")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("structural", "gpu", "capacity"), default="structural")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    if any(not override.startswith("seed=") or not override[5:].isdigit() for override in args.overrides):
        parser.error("Only seed=N is accepted; physics, checkpoint and budget overrides require a separately reviewed recipe")
    config = compose(args.overrides)
    report = structural_report(config)
    if args.gpu_worker:
        if args.mode != "gpu" or args.output is None or not args.output.is_dir():
            parser.error("GPU worker requires its supervisor-created output directory")
        report = gpu_baseline(config, args.output.resolve())
        (args.output / "gpu_baseline_report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return
    if args.mode == "structural":
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    if args.output is None:
        parser.error("GPU/capacity checks require a fresh --output directory")
    output = args.output.resolve()
    if output.exists():
        parser.error("Output exists; preserve earlier evidence and select a fresh path")
    output.mkdir(parents=True)
    (output / "structural_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    try:
        if args.mode == "gpu":
            report = gpu_worker(config, output, args.overrides)
        else:
            report = capacity_probe(config, output, args.overrides)
    except Exception as exc:
        report = dict(status="failed", mode=args.mode, error=f"{type(exc).__name__}: {exc}",
                      physical_reference_qualified=False)
        (output / "readiness_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        raise
    (output / "readiness_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
