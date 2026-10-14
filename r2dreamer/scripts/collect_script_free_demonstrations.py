"""Collect dense external-expert trajectories through the script-free student env.

The external expert is never an environment controller or actor observation.
Failed rollouts remain in the archive, with unsafe behavior-cloning labels off.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from imitation_data import file_sha256, save_dataset, validate_arrays
from scripted_expert import ScriptedExpert


def collect_batch(provider, expert, seed, max_steps, split, rollout_start=0, video_sink=None, sigma=0., rho=.9, fail_margin=25):
    """Provider protocol keeps temporal alignment testable without CUDA physics."""
    if max_steps < 1 or split not in (0, 1, 2):
        raise ValueError("A demonstration batch needs a positive horizon and an episode split")
    first = provider.reset(seed)
    observation = np.asarray(first["proprio"], dtype=np.float32)
    worlds = provider.worlds
    if observation.shape != (worlds, 45):
        raise ValueError("Collector must observe only the student's 45D proprioception")
    audit = {name: np.asarray(value) for name, value in first["audit"].items()}
    expert.reset(audit["rotation"])
    if video_sink is not None:
        video_sink(provider.video_frame())
    arrays = dict(proprio=np.zeros((worlds, max_steps + 1, 45), np.float32),
                  action=np.zeros((worlds, max_steps, 19), np.float32),
                  executed_action=np.zeros((worlds, max_steps, 19), np.float32),
                  reward=np.zeros((worlds, max_steps), np.float32),
                  is_terminal=np.zeros((worlds, max_steps), bool),
                  valid=np.zeros((worlds, max_steps), bool), present=np.zeros((worlds, max_steps), bool),
                  desired_ctrl=np.zeros((worlds, max_steps, 19), np.float32),
                  applied_ctrl=np.zeros((worlds, max_steps, 19), np.float32),
                  targets_within_range=np.zeros((worlds, max_steps), bool),
                  split=np.full(worlds, split, np.int8), episode_seed=np.full(worlds, seed, np.int64),
                  rollout_id=np.arange(rollout_start, rollout_start + worlds, dtype=np.int64))
    arrays["proprio"][:, 0] = observation
    for name, value in audit.items():
        if value.shape[0] != worlds or not np.isfinite(value).all():
            raise ValueError(f"Invalid initial audit field: {name}")
        arrays["state_" + name] = np.zeros((worlds, max_steps + 1, *value.shape[1:]), value.dtype)
        arrays["state_" + name][:, 0] = value
    finished = np.zeros(worlds, bool)
    unsafe = np.zeros(worlds, bool)
    noise_rng = np.random.default_rng(int(seed) * 1009 + int(rollout_start))
    noise = np.zeros((worlds, 19), np.float32)
    for step in range(max_steps):
        active = ~finished
        desired = np.asarray(expert.targets(step, audit["rotation"]), np.float32)
        if desired.shape != (worlds, 19) or not np.isfinite(desired).all():
            raise RuntimeError("External expert produced malformed/nonfinite full motor targets")
        low, high = provider.command_center - provider.command_scale, provider.command_center + provider.command_scale
        reachable = np.all((desired >= low - 1e-6) & (desired <= high + 1e-6), axis=-1)
        action = provider.normalize(desired)
        if action.shape != (worlds, 19) or not np.isfinite(action).all() or np.any(np.abs(action) > 1.000001):
            raise RuntimeError("Normalized full motor commands are malformed, nonfinite or out of range")
        executed = action
        if sigma > 0.:
            noise = rho * noise + np.sqrt(1. - rho ** 2) * sigma * noise_rng.standard_normal((worlds, 19)).astype(np.float32)
            executed = np.clip(action + noise, -1., 1.).astype(np.float32)
        following = provider.step(executed)
        next_observation = np.asarray(following["proprio"], np.float32)
        next_audit = {name: np.asarray(value) for name, value in following["audit"].items()}
        reward = np.asarray(following["reward"], np.float32).reshape(worlds)
        terminal = np.asarray(following["terminal"], bool).reshape(worlds)
        ended = np.asarray(following["ended"], bool).reshape(worlds)
        current_causes = {name: np.asarray(value, bool).reshape(worlds) for name, value in following["causes"].items()}
        if next_observation.shape != observation.shape or not np.isfinite(next_observation).all() or not np.isfinite(reward).all():
            raise RuntimeError("Nonfinite/malformed student transition; dataset collection stopped")
        arrays["present"][active, step] = True
        arrays["targets_within_range"][active, step] = reachable[active]
        arrays["valid"][active, step] = reachable[active]
        arrays["proprio"][active, step + 1] = next_observation[active]
        arrays["action"][active, step] = action[active]
        arrays["executed_action"][active, step] = executed[active]
        arrays["desired_ctrl"][active, step] = desired[active]
        arrays["applied_ctrl"][active, step] = next_audit["ctrl"][active]
        arrays["reward"][active, step] = reward[active]
        arrays["is_terminal"][active, step] = terminal[active]
        for name, value in next_audit.items():
            if value.shape != audit[name].shape or not np.isfinite(value).all():
                raise RuntimeError(f"Invalid post-action audit field: {name}")
            arrays["state_" + name][active, step + 1] = value[active]
        for name, value in following.get("logs", {}).items():
            value = np.asarray(value, np.float32).reshape(worlds)
            if not np.isfinite(value).all():
                raise RuntimeError(f"Nonfinite physical episode metric: {name}")
            key = "metric_" + name
            if key not in arrays:
                arrays[key] = np.zeros((worlds, max_steps), np.float32)
            arrays[key][active, step] = value[active]
        for name, value in current_causes.items():
            key = "cause_" + name
            if key not in arrays:
                arrays[key] = np.zeros((worlds, max_steps), bool)
            arrays[key][active, step] = value[active]
        unsafe |= active & (current_causes.get("physical_fall", False) | current_causes.get("wall_hit", False))
        if video_sink is not None and active[0] and ((step + 1) % 2 == 0 or ended[0]):
            video_sink(provider.video_frame())
        finished |= ended
        observation, audit = next_observation, next_audit
        if finished.all():
            break
    # A rollout ending in a physical fall/wall failure is not a safe expert:
    # retain its finite observed dynamics but exclude its entire BC trajectory.
    # Noisy (DART) rollouts keep their labels up to `margin` steps before the first failure;
    # clean rollouts that fail are excluded whole, as before.
    for world in np.flatnonzero(unsafe):
        if sigma > 0.:
            last = int(arrays["present"][world].sum()) - 1
            arrays["valid"][world, max(0, last - fail_margin):] = False
        else:
            arrays["valid"][world] = False
    arrays["episode_safe"] = ~unsafe
    arrays["episode_length"] = arrays["present"].sum(-1).astype(np.int32)
    outcomes = []
    for world in range(worlds):
        left = float(arrays.get("metric_valid_left_step", np.zeros((worlds, max_steps)))[world].sum())
        right = float(arrays.get("metric_valid_right_step", np.zeros((worlds, max_steps)))[world].sum())
        alternations = float(arrays.get("metric_alternating_steps", np.zeros((worlds, max_steps)))[world].sum())
        reasons = [name for name in following["causes"] if arrays.get("cause_" + name, np.zeros((worlds, max_steps)))[world].any()]
        if not finished[world]:
            reasons.append("collector_horizon")
        outcomes.append(dict(rollout_id=int(rollout_start + world), seed=int(seed), split=int(split), world=world,
                             control_transitions=int(arrays["episode_length"][world]), termination_reasons=reasons,
                             safe_expert_labels=int(arrays["valid"][world].sum()), physical_safe=bool(not unsafe[world]),
                             valid_left_steps=left, valid_right_steps=right, alternating_steps=alternations,
                             bilateral_motion_observed=bool(left > 0 and right > 0 and alternations > 0 and not unsafe[world]),
                             scope="Measured finite rollout; safety or bilateral events do not establish an independently learned policy"))
    validate_arrays(arrays)
    return arrays, outcomes


class GPUStudentProvider:
    def __init__(self, config, worlds):
        import torch
        from envs.mjx_teacher_env import MJXTeacherEnv
        from training_contract import build_training_contract

        self.torch = torch
        scene = Path(str(config.env.scene_path))
        config.env.scene_path = str((scene if scene.is_absolute() else ROOT / scene).resolve(strict=True))
        self.env = MJXTeacherEnv(config.env, worlds, evaluation=False)
        self.worlds = worlds
        self.command_center = np.asarray(self.env.jax.device_get(self.env.raw.command_center), np.float32)
        self.command_scale = np.asarray(self.env.jax.device_get(self.env.raw.command_scale), np.float32)
        self.contract = build_training_contract(self.env, config.env)
        if not self.env.raw.full_command or not np.all(self.command_scale > 0):
            raise ValueError("Demonstrations require the ordinary full-command student environment")
        self.done = None

    def normalize(self, desired):
        from full_command import normalize_motor_targets

        return np.asarray(normalize_motor_targets(desired, self.command_center, self.command_scale), np.float32)

    def audit(self):
        env, data = self.env, self.env.state.pipeline_state
        host = lambda value: np.asarray(env.jax.device_get(value)).copy()
        sensors = host(data.sensordata)
        contacts = []
        for name in ("mjx_left_sole_contacts", "mjx_right_sole_contacts"):
            descriptor = env.raw.m.sensor(name)
            address, dimension = int(descriptor.adr[0]), int(descriptor.dim[0])
            if dimension != 32 * 7:
                raise RuntimeError(f"Unexpected raw foot-contact sensor layout: {name}")
            contacts.append(sensors[:, address:address + dimension].reshape(self.worlds, -1, 7))
        features = env.jax.vmap(env.raw._features)(data)
        return dict(qpos=host(data.qpos), qvel=host(data.qvel), ctrl=host(data.ctrl), time=host(data.time),
                    state_rng=host(env.state.info["rng"]), step_count=host(env.state.info["step_count"]),
                    bridge_rng=np.broadcast_to(host(env.rng), (self.worlds, 2)).copy(),
                    rotation=host(data.xmat[:, env.raw.base_id]),
                    actuator_force=host(data.qfrc_actuator)[:, env.raw.act_dof],
                    contact_sensor=np.stack(contacts, axis=1),
                    fsr=host(features[0]), foot_load=host(features[1]), clearance=host(features[2]), foot_xy=host(features[3]))

    def reset(self, seed):
        env, torch = self.env, self.torch
        env.rng = env.jax.random.PRNGKey(int(seed))
        env.state = None
        env._pending_injection_first = False
        action = torch.zeros((self.worlds, 19), device="cuda:0")
        transition, self.done = env.step(action, torch.ones(self.worlds, dtype=torch.bool, device="cuda:0"))
        self.last_transition = transition
        if "teacher_state" in transition or tuple(transition["proprio"].shape) != (self.worlds, 45):
            raise RuntimeError("Privileged/phase controller information leaked into student transitions")
        return dict(proprio=transition["proprio"].detach().cpu().numpy().copy(), audit=self.audit())

    def step(self, action):
        transition, self.done = self.env.step(self.torch.as_tensor(action, device="cuda:0"), self.done)
        if "teacher_state" in transition:
            raise RuntimeError("Collector received a privileged teacher observation")
        host = lambda value: value.detach().cpu().numpy().copy()
        names = ("physical_fall", "wall_hit", "stall_out", "goal_reached", "time_limit")
        return dict(proprio=host(transition["proprio"]), audit=self.audit(),
                    reward=host(transition["reward"]), terminal=host(transition["is_terminal"]), ended=host(self.done),
                    causes={name: host(transition["log_" + name]) for name in names},
                    logs={name[4:]: host(value) for name, value in transition.items() if name.startswith("log_")})

    def video_frame(self):
        record = self.env.render_selected([0], width=320, height=240)[0]
        if record is None or record.get("error") or record.get("frame") is None:
            raise RuntimeError(f"Expert demonstration rendering failed: {record}")
        return record["frame"]

    def check_reset_repeatability(self, seed):
        first = self.reset(seed)
        retained = self.last_transition["proprio"]
        hold = self.normalize(first["audit"]["ctrl"])
        self.step(hold)
        self.step(hold)
        repeated = self.reset(seed)
        report = validate_student_reset(first, repeated, retained.detach().cpu().numpy().copy())
        if not report["passed"]:
            raise RuntimeError("Student reset/retained-observation validation failed: " + ", ".join(report["errors"]))
        return report

    def close(self):
        self.env.close()


def validate_student_reset(first, repeated, retained):
    from reset_validation import OBSERVATION_GROUPS

    errors, measurements = [], {}
    for name in ("qpos", "qvel", "ctrl", "time", "state_rng", "bridge_rng", "step_count"):
        a, b = np.asarray(first["audit"][name]), np.asarray(repeated["audit"][name])
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.array_equal(a, b):
            errors.append(name + ":reset_state_changed")
    reference, current = np.asarray(first["proprio"]), np.asarray(repeated["proprio"])
    if reference.shape != current.shape or reference.ndim != 2 or reference.shape[1] != 45:
        errors.append("proprio:shape")
    else:
        for name, (start, end, tolerance) in OBSERVATION_GROUPS["proprio"].items():
            difference = np.abs(reference[:, start:end].astype(np.float64) - current[:, start:end])
            measurements[name] = dict(atol=tolerance, max_abs=float(difference.max(initial=0.)))
            if not np.isfinite(difference).all() or np.any(difference > tolerance):
                errors.append(name + ":numerical_repeatability")
    if not np.array_equal(reference, retained):
        errors.append("proprio:retained_view_changed")
    return dict(passed=not errors, errors=errors, observations=measurements,
                scope="Exact reset inputs/RNG and retained DLPack views; bounded derived-channel noise, not bitwise physics")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expert-reference", type=Path, default=ROOT / "envs/mjx_walking/assets/humanlike_gait.npz")
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--worlds", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=20.)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-scale", type=float, default=1.)
    parser.add_argument("--amplitude", type=float, default=1.)
    parser.add_argument("--stabilizer", type=float, nargs=3, metavar=("KP", "KD", "LIMIT"), default=(1., .04, .25))
    parser.add_argument("--dart-sigmas", type=float, nargs="+", default=[0.], help="Train-batch noise levels (normalized action units), cycled; validation/test stay clean")
    parser.add_argument("--video", action="store_true", help="Require one first-episode MP4 per seed group, with metadata")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (not 1 <= args.worlds <= 32 or not 3 * args.worlds <= args.episodes <= 256
            or args.episodes % args.worlds or not 2. <= args.seconds <= 30. or args.seed < 0):
        parser.error("Use 1..32 worlds, 3..32 whole seed batches up to 256 episodes, and a 2..30 second horizon")
    if args.output.exists() or args.output.with_suffix(".manifest.json").exists():
        parser.error("Output/manifest exists; select a fresh dataset path")
    video_dir = args.output.resolve().parent / (args.output.stem + "_videos")
    if args.video and video_dir.exists():
        parser.error("Video evidence directory exists; select a fresh dataset output")
    steps = int(round(args.seconds / .02))
    batches = args.episodes // args.worlds
    schedule = [dict(seed=args.seed + batch, split=0 if batch < batches - 2 else batch - (batches - 2) + 1,
                     rollout_start=batch * args.worlds,
                     sigma=float(args.dart_sigmas[batch % len(args.dart_sigmas)]) if batch < batches - 2 else 0.)
                for batch in range(batches)]
    kp, kd, limit = args.stabilizer
    expert = ScriptedExpert.from_npz(args.expert_reference, stabilizer=(kp, kd, -1., limit),
                                     time_scale=args.time_scale, amplitude=args.amplitude)
    if args.dry_run:
        print(json.dumps(dict(config="walking_mjx_script_free", episodes=args.episodes, worlds=args.worlds,
                              control_hz=50, transitions_per_episode=steps, schedule=schedule,
                              external_expert_sha256=file_sha256(args.expert_reference),
                              video_required=args.video, video_directory=str(video_dir),
                              scope="No GPU physics, demonstrations or training executed"), indent=2))
        return
    import hydra
    from scripts.preflight_reference_training import gpu_runtime_report

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_PLATFORMS", "cuda")
    runtime = gpu_runtime_report()
    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        config = hydra.compose("walking_mjx_script_free", overrides=[f"seed={args.seed}"])
    config.env.env_num = args.worlds
    config.env.eval_episode_num = 0
    # The collection cutoff is independent of the student's physical horizon:
    # an 8 s capacity dataset retains the same 20 s deployment contract.
    forbidden = ("mjx_scripted_gait", "mjx_scripted_phase_pool", "mjx_stabilizer", "pose_rsi_json", "gait_reference_csv")
    if (not bool(config.env.mjx_full_command) or bool(config.env.mjx_gait_residual)
            or bool(config.env.mjx_phase_obs) or any(config.env.get(name) is not None for name in forbidden)):
        raise ValueError("Collector must act through the exact script-free student controller contract")
    provider = GPUStudentProvider(config, args.worlds)
    try:
        reset_report = provider.check_reset_repeatability(args.seed)
        arrays, outcomes = [], []
        videos = []
        if args.video:
            import imageio.v2 as imageio
            video_dir.mkdir(parents=True)
        for row in schedule:
            frames = [0]
            writer = None
            try:
                if args.video:
                    video_path = video_dir / f"expert_seed_{row['seed']:07d}_split_{row['split']}.mp4"
                    writer = imageio.get_writer(video_path, fps=25)

                def sink(frame):
                    writer.append_data(frame)
                    frames[0] += 1

                batch, reports = collect_batch(provider, expert, max_steps=steps,
                                               video_sink=sink if writer is not None else None, **row)
            finally:
                if writer is not None:
                    writer.close()
            if args.video:
                if frames[0] < 1 or not video_path.is_file() or video_path.stat().st_size == 0:
                    raise RuntimeError("Required external-expert video evidence is missing or empty")
                video_record = dict(path=video_path.relative_to(args.output.resolve().parent).as_posix(), seed=row["seed"], split=row["split"],
                                    rollout_id=row["rollout_start"], evaluation_world=0, frames=frames[0], fps=25,
                                    sha256=file_sha256(video_path), control_transitions=reports[0]["control_transitions"],
                                    termination_reasons=reports[0]["termination_reasons"],
                                    scope="First expert episode for this seed group, including terminal frame")
                video_path.with_suffix(".json").write_text(json.dumps(video_record, indent=2) + "\n", encoding="utf-8")
                videos.append(video_record)
            arrays.append(batch)
            outcomes.extend(reports)
        if any(set(batch) != set(arrays[0]) for batch in arrays):
            raise RuntimeError("Demonstration audit/metric fields changed between rollout batches")
        combined = {name: np.concatenate([batch[name] for batch in arrays], axis=0) for name in arrays[0]}
        manifest = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), collector="external_scripted_expert_v1",
                        training_contract=provider.contract, student_training_contract=provider.contract,
                        action_contract=dict(kind="full_motor_command", normalization="(desired_ctrl-command_center)/command_scale; clipped[-1,1]",
                                             command_center=provider.command_center.tolist(), command_scale=provider.command_scale.tolist(),
                                             command_slew_rad_s=float(config.env.mjx_control_slew_rad_s), peak_torque_kgcm=18., control_dt_seconds=.02),
                        external_expert=dict(reference_path=str(args.expert_reference.resolve()),
                                             reference_sha256=file_sha256(args.expert_reference), stabilizer=list(expert.stabilizer),
                                             time_scale=expert.time_scale, amplitude=expert.amplitude,
                                             startup="dense standing-start prefix", cycle_start=expert.cycle_start, cycle_len=expert.cycle_len),
                        provenance={name: file_sha256(ROOT / name) for name in (
                            "imitation_data.py", "scripted_expert.py", "scripts/collect_script_free_demonstrations.py",
                            "full_command.py", "envs/mjx_teacher_env.py", "envs/mjx_walking/env.py", "training_contract.py")},
                        runtime=runtime, reset_validation=reset_report, schedule=schedule, episode_outcomes=outcomes,
                        video_required=args.video, video_evidence=videos,
                        acceptance=dict(safe_episodes=int(combined["episode_safe"].sum()),
                                        unsafe_episodes=int((~combined["episode_safe"]).sum()),
                                        safe_bc_labels=int(combined["valid"].sum()),
                                        walking_teacher_qualified=False,
                                        criterion="Exclude every physical-fall/wall rollout from BC; preserve finite dynamics and measured bilateral diagnostics"),
                        learned_policy_evaluated=False, dagger_implemented=False)
        report = save_dataset(args.output, combined, manifest)
        print(json.dumps(dict(path=str(args.output), data_sha256=report["data_sha256"], summary=report["summary"],
                              acceptance=report["acceptance"]), indent=2))
    finally:
        provider.close()


if __name__ == "__main__":
    main()
