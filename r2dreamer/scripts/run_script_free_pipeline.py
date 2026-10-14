"""Guarded imitation + Dreamer pipeline; default admission never runs the pilot.

Stages run in separate processes so JAX/Torch allocations are released. Runtime
admission, imitation error and closed-loop physical results stay separate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import hydra
import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG = "walking_mjx_script_free"
from scripts.script_free_run_ops import PipelineStopped, RunJournal, atomic_json, run_stage, stop_signals

RUN_JOURNAL = None


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    atomic_json(path, value)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def finite_json(value):
    if isinstance(value, dict):
        return all(finite_json(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_json(item) for item in value)
    return bool(np.isfinite(value)) if isinstance(value, (float, int)) else True


def compose(seed):
    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return hydra.compose(CONFIG, overrides=[f"seed={seed}"])


def structural_report(config):
    from script_free_contract import validate_student_config
    from scripted_expert import ScriptedExpert
    from training_budget import validate_learning_budget

    guard = validate_student_config(config)
    if any(config.get(name) is not None for name in ("resume_from", "init_from", "transfer_from")):
        raise ValueError("The packaged pipeline starts fresh; old checkpoints are not admitted")
    env, trainer = config.env, config.trainer
    if (int(env.env_num), int(env.steps), int(env.action_repeat), int(env.train_ratio),
            int(config.buffer.max_size)) != (128, 1000000, 10, 64, 131072):
        raise ValueError("The reviewed pilot world/replay/learning budget changed")
    if (int(env.eval_episode_num) != 32 or not env.eval_repeatable or env.eval_balanced_pose_ids or
            list(env.eval_rsi_probabilities) != [1., 0., 0., 0.] or list(env.rsi_probabilities) != [1., 0., 0., 0.] or
            int(env.time_limit) != 10000 or env.eval_pose_rsi):
        raise ValueError("Require 32 repeatable standing-start 20-second evaluation worlds")
    if (float(env.mjx_motor_torque_kgcm), float(env.mjx_control_slew_rad_s),
            float(env.mjx_step_clearance_m), float(env.mjx_reset_joint_noise_rad),
            float(env.mjx_reset_velocity_noise)) != (18., 8., .015, .005, .005):
        raise ValueError("The reviewed torque/slew/detector/reset recipe changed")
    if (trainer.best_metric_key, trainer.best_safety_metric_key, trainer.best_tiebreak_metric_key,
            trainer.checkpoint_order) != ("total_valid_steps", "safety_failure_count", "alternating_steps", "safety_first"):
        raise ValueError("Physical safety-first checkpoint ranking changed")
    if (not trainer.eval_video or not trainer.eval_video_save_mp4 or trainer.eval_video_one_per_pose or
            int(trainer.eval_max_steps) != 1000 or int(trainer.eval_video_fps) != 25 or
            int(config.imitation.pretrain_updates) != 1000 or float(config.model.imitation_weight) != .1 or
            float(config.model.imitation_pretrain_weight) != 1.):
        raise ValueError("Imitation/evaluation evidence budget changed")
    warmup = validate_learning_budget(128, int(config.batch_length), 10, 131072, 1000000)
    if int(trainer.learning_start_deadline_steps) <= warmup:
        raise ValueError("Fresh replay warmup is outside the learning deadline")
    scene = ROOT / str(env.scene_path)
    tree = ET.parse(scene)
    actuators = tree.getroot().findall("actuator/position")
    if len(actuators) != 19 or any(float(item.get("kp", "nan")) != 40. or
                                  float(item.get("kv", "nan")) != .6 for item in actuators):
        raise ValueError("Expected 19 kp40/kv0.6 position actuators")
    compiler = tree.getroot().find("compiler")
    meshdir = scene.parent / (compiler.get("meshdir", "") if compiler is not None else "")
    meshes = [(meshdir / item.get("file")).resolve(strict=True)
              for item in tree.getroot().findall("asset/mesh") if item.get("file")]
    expert_path = ROOT / "envs/mjx_walking/assets/humanlike_gait.npz"
    expert = ScriptedExpert.from_npz(expert_path)
    paths = [scene, expert_path, *meshes, ROOT / "full_command.py", ROOT / "script_free_contract.py",
             ROOT / "script_free_policy.py", ROOT / "dreamer.py", ROOT / "imitation_training.py",
             ROOT / "imitation_data.py", ROOT / "scripted_expert.py", ROOT / "train.py", ROOT / "trainer.py",
             ROOT / "training_contract.py", ROOT / "envs/mjx_walking/env.py", ROOT / "envs/mjx_teacher_env.py",
             ROOT / "configs" / (CONFIG + ".yaml")]
    return dict(status="passed_source_structure_only", gpu_tested=False, walking_qualified=False,
                student=guard, training_worlds=128, evaluation_worlds=32,
                replay_warmup_counted_ticks=warmup, pilot_counted_ticks=1000000,
                source_sha256={path.relative_to(ROOT).as_posix(): digest(path) for path in paths},
                external_expert=dict(sha256=digest(expert_path), cycle_start=expert.cycle_start, cycle_len=expert.cycle_len),
                config=OmegaConf.to_container(config, resolve=True),
                scope="Source/asset closure only; pinned GPU execution, dense demonstrations and learned walking remain untested")


def stage(command, log, timeout):
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false", JAX_PLATFORMS="cuda",
               JAX_PLATFORM_NAME="gpu", PYTHONUNBUFFERED="1")
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")])
    print("RUN_STAGE " + " ".join(command), flush=True)
    return run_stage(command, log, timeout, cwd=ROOT, env=env, journal=RUN_JOURNAL)


def checkpoint_report(path):
    # Only load trusted locally generated checkpoints. CPU readback does not
    # leave Torch allocations in this supervisor ahead of the next GPU stage.
    code = """
import hashlib,json,sys,torch
c=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
s=c['agent_state_dict']
groups={}
for prefix in ('actor.','rssm.','encoder.','value.'):
    h=hashlib.sha256(); count=0
    for name,v in sorted(s.items()):
        if name.startswith(prefix):
            h.update(name.encode()); h.update(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()); count+=1
    groups[prefix]=h.hexdigest() if count else None
print(json.dumps(dict(step=c.get('step'),optimizer_updates=c.get('optimizer_updates'),stop_reason=c.get('stop_reason'),
    finite_agent=bool(s) and all(bool(torch.isfinite(v).all()) for v in s.values()),weight_hashes=groups,
    training_contract=c.get('training_contract'),imitation=c.get('imitation'),model_config=c.get('model_config')),allow_nan=False))
"""
    result = subprocess.run([sys.executable, "-c", code, str(path)], cwd=ROOT,
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
                            capture_output=True, text=True, timeout=180, check=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


def validate_demonstrations(path, episodes, seed):
    from demonstration_audit import validate_trajectory_contract
    from imitation_data import ImitationDataset
    from script_free_contract import validate_student_environment

    dataset = ImitationDataset(path)
    manifest = dataset.manifest
    contract = validate_student_environment(manifest.get("training_contract"))
    if contract != manifest.get("student_training_contract") or contract["task"]["episode_horizon_control_steps"] != 1000:
        raise ValueError("Collection altered the student's 20-second physical task")
    if (dataset.arrays["proprio"].shape[0] != episodes or manifest.get("video_required") is not True or
            manifest.get("reset_validation", {}).get("passed") is not True or
            manifest.get("collector") != "external_scripted_expert_v1" or
            manifest.get("external_expert", {}).get("reference_sha256") != digest(ROOT / "envs/mjx_walking/assets/humanlike_gait.npz")):
        raise ValueError("Dataset runtime/reset/external-expert/video provenance is incomplete")
    provenance = manifest.get("provenance", {})
    expected_provenance = {"imitation_data.py", "scripted_expert.py", "scripts/collect_script_free_demonstrations.py",
                           "full_command.py", "envs/mjx_teacher_env.py", "envs/mjx_walking/env.py", "training_contract.py"}
    if set(provenance) != expected_provenance:
        raise ValueError("Demonstration provenance is missing required collector/environment code")
    for name, sha in provenance.items():
        if digest(ROOT / name) != sha:
            raise ValueError(f"Demonstration collector source changed: {name}")
    action = manifest.get("action_contract", {})
    control = contract["control"]
    if (action.get("kind") != "full_motor_command" or action.get("command_slew_rad_s") != 8. or
            action.get("peak_torque_kgcm") != 18. or action.get("control_dt_seconds") != .02 or
            action.get("command_center") != control["command_center_rad"] or
            action.get("command_scale") != control["action_scale_rad"]):
        raise ValueError("Demonstration action normalization/physical limits are not the student contract")
    arrays = dataset.arrays
    trajectory_audit = validate_trajectory_contract(arrays, contract)
    audit_required = {"present", "episode_seed", "rollout_id", "episode_safe", "episode_length", "desired_ctrl",
                      "applied_ctrl", "targets_within_range", "cause_physical_fall", "cause_wall_hit",
                      "state_qpos", "state_qvel", "state_ctrl", "state_time", "state_contact_sensor", "state_actuator_force"}
    if not audit_required <= arrays.keys():
        raise ValueError("Demonstrations lack dense state/action/contact/termination audit arrays")
    observed, labels = arrays["present"].astype(bool), arrays["valid"].astype(bool)
    steps = arrays["action"].shape[1]
    shapes = {"state_qpos": (episodes, steps + 1, 26), "state_qvel": (episodes, steps + 1, 25),
              "state_ctrl": (episodes, steps + 1, 19), "state_time": (episodes, steps + 1),
              "state_contact_sensor": (episodes, steps + 1, 2, 32, 7),
              "state_actuator_force": (episodes, steps + 1, 19),
              "desired_ctrl": (episodes, steps, 19), "applied_ctrl": (episodes, steps, 19),
              "cause_physical_fall": (episodes, steps), "cause_wall_hit": (episodes, steps),
              "targets_within_range": (episodes, steps), "episode_safe": (episodes,), "episode_length": (episodes,)}
    if any(arrays[name].shape != shape for name, shape in shapes.items()):
        raise ValueError("Malformed dense demonstration audit shapes")
    unsafe = np.any((arrays["cause_physical_fall"].astype(bool) | arrays["cause_wall_hit"].astype(bool)) & observed, axis=1)
    limits = np.asarray(control["actuator_ctrlrange_rad"])
    reachable = np.all((arrays["desired_ctrl"] >= limits[:, 0] - 1e-6) &
                       (arrays["desired_ctrl"] <= limits[:, 1] + 1e-6), axis=-1)
    expected_labels = observed & arrays["targets_within_range"].astype(bool) & ~unsafe[:, None]
    normalized = np.clip((arrays["desired_ctrl"] - np.asarray(control["command_center_rad"])) /
                         np.asarray(control["action_scale_rad"]), -1., 1.)
    if (not np.array_equal(arrays["episode_safe"].astype(bool), ~unsafe) or
            not np.array_equal(arrays["targets_within_range"].astype(bool) & observed, reachable & observed) or
            not np.array_equal(arrays["episode_length"], observed.sum(axis=1)) or
            not np.array_equal(labels, expected_labels) or
            not np.allclose(arrays["action"][observed], normalized[observed], rtol=0., atol=1e-6) or
            not np.allclose(arrays["applied_ctrl"][observed], arrays["state_ctrl"][:, 1:][observed], rtol=0., atol=1e-6)):
        raise ValueError("Unsafe labels or causally mismatched commanded/applied targets in demonstrations")
    split_stats = dataset.split_statistics
    for code, split in enumerate(("train", "validation", "test")):
        row = split_stats[split]
        if row["safe_expert_labels"] < 256 or row["safe_labels_after_64_frames"] < 256:
            raise ValueError(f"Insufficient safe startup/context supervision in {split}; inspect expert outcomes")
        selected = dataset.arrays["split"] == code
        if not dataset.arrays["valid"][selected, 0].any():
            raise ValueError(f"No safe standing-start labels in {split}")
    batches = episodes // 8
    schedule = [dict(seed=seed + group, split=0 if group < batches - 2 else group - batches + 3,
                     rollout_start=group * 8) for group in range(batches)]
    if manifest.get("schedule") != schedule:
        raise ValueError("Whole-seed train/validation/test schedule changed")
    if (not np.array_equal(arrays["episode_seed"], np.repeat([row["seed"] for row in schedule], 8)) or
            not np.array_equal(arrays["split"], np.repeat([row["split"] for row in schedule], 8)) or
            not np.array_equal(arrays["rollout_id"], np.arange(episodes))):
        raise ValueError("Dense rollout IDs/seed splits differ from the collection schedule")
    videos = manifest.get("video_evidence", [])
    if len(videos) != batches:
        raise ValueError("Missing per-seed expert video evidence")
    for record, expected in zip(videos, schedule):
        video = (Path(path).parent / record.get("path", "")).resolve(strict=True)
        if not video.is_relative_to(Path(path).parent.resolve()) or video.suffix != ".mp4":
            raise ValueError("Expert video path escapes the dataset directory")
        if (video.stat().st_size <= 0 or digest(video) != record.get("sha256") or
                read_json(video.with_suffix(".json")) != record or record.get("frames", 0) < 1 or
                record.get("fps") != 25 or record.get("seed") != expected["seed"] or
                record.get("split") != expected["split"] or record.get("rollout_id") != expected["rollout_start"]):
            raise ValueError("Expert video identity/sidecar/split evidence changed")
    return dict(fingerprint=dataset.fingerprint, split_statistics=split_stats,
                acceptance=manifest["acceptance"], training_contract=contract,
                video_records=videos, trajectory_contract_audit=trajectory_audit, walking_teacher_qualified=False)


def validate_imitation(report, checkpoint, fingerprint, updates, contract, checkpoint_sha256):
    from training_contract import check_checkpoint_contract
    if (report.get("status") != "completed_bounded_imitation" or report.get("updates") != updates or
            report.get("changed_tensors_first_update", 0) <= 0 or not report.get("final_validation_improved") or
            report.get("validation_improved") is not True or
            not isinstance(report.get("initial_validation_bc"), (int, float)) or
            not isinstance(report.get("final_validation_bc"), (int, float)) or
            report["final_validation_bc"] >= report["initial_validation_bc"] or
            report.get("latest_checkpoint_sha256") != checkpoint_sha256 or
            report.get("dataset_fingerprint") != fingerprint or not finite_json(report)):
        raise ValueError("Imitation did not complete finite, changed-weight, held-out-improving updates")
    check_checkpoint_contract(checkpoint, contract, mode="resume")
    if (not checkpoint.get("finite_agent") or checkpoint.get("step") != 0 or
            checkpoint.get("optimizer_updates") != updates or
            checkpoint.get("imitation", {}).get("pretraining_updates") != updates or
            checkpoint.get("imitation", {}).get("dataset_fingerprint") != fingerprint or
            checkpoint.get("imitation", {}).get("last_validation", {}).get("imitation/bc") != report["final_validation_bc"] or
            checkpoint.get("imitation", {}).get("final_validation_bc") != report["final_validation_bc"] or
            not checkpoint.get("model_config")):
        raise ValueError("Imitation checkpoint/readback does not match its dataset and update evidence")
    for split in ("validation", "test"):
        evidence = report.get("full_episode_" + split, {})
        if (evidence.get("protocol") != "offline_teacher_forced_full_episode_v1" or
                evidence.get("split") != split or evidence.get("labels", 0) <= 0 or
                evidence.get("optimizer_updates") != updates or
                evidence.get("deterministic_posterior") is not True or
                evidence.get("walking_qualified") is not False or
                checkpoint.get("imitation", {}).get("full_episode_" + split) != evidence):
            raise ValueError("Imitation lacks checkpoint-bound full-episode held-out evidence")


def evaluation_rows(directory):
    return [json.loads(line) for line in (Path(directory) / "walking_evaluations.jsonl").read_text().splitlines() if line.strip()]


def validate_evaluations(rows, step, contract, fresh=True):
    from script_free_contract import validate_student_environment
    from training_contract import check_checkpoint_contract

    if not rows or rows[-1].get("step") != step or (fresh and (len(rows) < 2 or rows[0].get("step") != 0)):
        raise ValueError("Missing initial/final evaluation matching the checkpoint step")
    for row in (rows[0], rows[-1]):
        if row.get("schema_version") != 2 or not finite_json(row):
            raise ValueError("Malformed or nonfinite closed-loop evaluation")
        validate_student_environment(row.get("training_contract"))
        check_checkpoint_contract({"training_contract": row["training_contract"]}, contract, mode="resume")
        protocol = row.get("protocol", {}).get("environment", {})
        if (protocol.get("worlds") != 32 or protocol.get("repeatable") is not True or
                protocol.get("full_command") is not True or protocol.get("reset_source_kind") != "standing" or
                protocol.get("rsi_stratified") is not False or protocol.get("rsi_probabilities") is not None or
                protocol.get("valid_step_clearance_m") != .015 or protocol.get("horizon_control_steps") != 1000 or
                protocol.get("control_dt_seconds") != .02 or protocol.get("reset_joint_noise_rad") != .005 or
                protocol.get("reset_velocity_noise") != .005 or protocol.get("observation_keys") != ["proprio"] or
                protocol.get("controller_identity") != "autonomous_full_command_v1" or
                protocol.get("reference_kind") != "none" or protocol.get("stabilizer_enabled") is not False or
                protocol.get("phase_observations") is not False or
                row.get("protocol", {}).get("eval_max_control_steps") != 1000):
            raise ValueError("Evaluation is not the 32-world script-free standing-start protocol")
        trials = row.get("trials", [])
        required = {"alive_seconds", "total_valid_steps", "valid_left_step", "valid_right_step", "alternating_steps",
                    "fall_count", "safety_failure_count", "wall_hit", "stall_out", "goal_reached", "time_limit",
                    "body_tilt_rad_peak", "root_drop_m_peak", "left_clearance_m_peak", "right_clearance_m_peak",
                    "torque_saturation_fraction_mean", "control_slew_limited_fraction_mean", "termination_reasons"}
        if (len(trials) != 32 or [trial.get("evaluation_world") for trial in trials] != list(range(32)) or
                any(trial.get("start_pose_id") != 0 or not required <= trial.keys() for trial in trials)):
            raise ValueError("Closed-loop evaluation lacks all 32 starts and physical/cause metrics")


def online_command(run, dataset, initial, seed, capacity):
    command = [sys.executable, "-u", "train.py", "--config-name", CONFIG, f"seed={seed}",
               f"imitation.dataset={Path(dataset).as_posix()}", f"init_from={Path(initial).as_posix()}",
               f"logdir={Path(run).as_posix()}"]
    if capacity:
        from scripts.teacher_capacity_probe import case_overrides
        command.extend([*case_overrides(128, 32), "buffer.max_size=131072", "imitation.pretrain_updates=32",
                        "trainer.eval_video=true", "trainer.eval_video_save_mp4=true",
                        "trainer.eval_video_final_only=false", "trainer.eval_video_every=1"])
    return command


def evaluate(checkpoint, output, log, seed):
    from scripts.preflight_reference_training import validate_capacity_videos
    stage([sys.executable, "-u", "scripts/evaluate_script_free.py", "--checkpoint", str(checkpoint),
           "--output", str(output), "--seed", str(seed + 1000), "--seconds", "20", "--worlds", "32"], log, 1800)
    report = read_json(output / "script_free_evaluation.json")
    if (report.get("status") != "completed_script_free_evaluation" or not finite_json(report) or
            any(report.get(name) is not False for name in
                ("trajectory_loaded", "explicit_stabilizer", "dataset_loaded", "privileged_actor_inputs"))):
        raise ValueError("Independent evaluation has a controller/dataset dependency or incomplete report")
    info = checkpoint_report(checkpoint)
    validate_evaluations(evaluation_rows(output), info["step"], info["training_contract"], fresh=False)
    # The existing final-video gate excludes step0; the offline checkpoint has
    # its own exact zero-step filenames and matching sidecar admission.
    if info["step"] > 0:
        videos = validate_capacity_videos(output, info["step"])
    else:
        videos = validate_zero_step_videos(output)
    return dict(report=report, final_video_evidence=videos)


def validate_zero_step_videos(output):
    expected = {"step_0000000" + ("" if world == 0 else f"_env_{world:02d}") + ".mp4" for world in range(32)}
    directory = Path(output) / "eval_videos"
    if {path.name for path in directory.glob("step_0000000*.mp4")} != expected:
        raise ValueError("Pretrained-policy evaluation lacks all 32 independent videos")
    for world in range(32):
        path = directory / ("step_0000000" + ("" if world == 0 else f"_env_{world:02d}") + ".mp4")
        record = read_json(path.with_suffix(".json"))
        if (path.stat().st_size <= 0 or type(record.get("training_step")) is not int or record["training_step"] != 0 or
                type(record.get("evaluation_world")) is not int or record["evaluation_world"] != world or
                type(record.get("start_pose_id")) is not int or record["start_pose_id"] != 0 or
                type(record.get("frames")) is not int or record["frames"] <= 0 or
                isinstance(record.get("fps"), bool) or record.get("fps") != 25):
            raise ValueError("Pretrained-policy video evidence is incomplete")
    return dict(training_step=0, worlds=32, scope="All nonempty files and matching sidecars; human gait review remains required")


def learning_run(directory, seed, capacity):
    from scripts.preflight_reference_training import validate_capacity_videos
    from training_contract import check_checkpoint_contract

    directory.mkdir()
    episodes, seconds, updates = (24, 8, 32) if capacity else (64, 20, 1000)
    dataset = directory / "demonstrations.npz"
    stage([sys.executable, "-u", "scripts/collect_script_free_demonstrations.py", "--output", str(dataset),
           "--episodes", str(episodes), "--worlds", "8", "--seconds", str(seconds), "--seed", str(seed), "--video"],
          directory / "collection.log", 1800)
    demonstrations = validate_demonstrations(dataset, episodes, seed)
    write_json(directory / "demonstration_admission.json", demonstrations)
    imitation = directory / "imitation"
    log = stage([sys.executable, "-u", "scripts/pretrain_script_free.py", "--dataset", str(dataset),
                 "--output", str(imitation), "--updates", str(updates), "--seed", str(seed)],
                directory / "imitation.log", 1800 if capacity else 7200)
    if "IMITATION_LEARNING_VERIFIED" not in log:
        raise ValueError("Imitation has no verified first gradient/update")
    before = checkpoint_report(imitation / "latest.pt")
    pretrain_report = read_json(imitation / "pretraining_report.json")
    validate_imitation(pretrain_report, before, demonstrations["fingerprint"], updates,
                       demonstrations["training_contract"], digest(imitation / "latest.pt"))
    imitation_eval = None
    if not capacity:
        imitation_eval = evaluate(imitation / "latest.pt", directory / "imitation_eval", directory / "imitation_eval.log", seed)
    online = directory / "online"
    command = online_command(online, dataset, imitation / "latest.pt", seed, capacity)
    log = stage(command, directory / "online.log", 1800 if capacity else 43200)
    if "LEARNING VERIFIED:" not in log:
        raise ValueError("Online Dreamer has no verified finite changed-weight update")
    after = checkpoint_report(online / "latest.pt")
    check_checkpoint_contract(after, demonstrations["training_contract"], mode="resume")
    if (not after.get("finite_agent") or after.get("step", 0) <= 0 or after.get("optimizer_updates", 0) <= 0 or
            after.get("imitation", {}).get("dataset_fingerprint") != demonstrations["fingerprint"] or
            after.get("imitation", {}).get("pretraining_updates") != updates):
        raise ValueError("Online checkpoint lacks finite learner/imitation identity")
    if capacity:
        if after["optimizer_updates"] != 32 or after.get("stop_reason") != "optimizer_update_limit":
            raise ValueError("Capacity did not stop at exactly 32 full-learner updates")
    elif after["step"] < 1000000 or after["step"] >= 1000000 + 128 * 10:
        raise ValueError("Pilot did not complete the declared bounded 1M counted-tick budget")
    for prefix in ("actor.", "rssm.", "value."):
        if not after["weight_hashes"].get(prefix) or before["weight_hashes"].get(prefix) == after["weight_hashes"][prefix]:
            raise ValueError(f"Online learner did not change finite {prefix} weights")
    rows = evaluation_rows(online)
    validate_evaluations(rows, after["step"], after["training_contract"])
    videos = validate_capacity_videos(online, after["step"])
    independent = evaluate(online / "latest.pt", directory / "independent_eval", directory / "independent_eval.log", seed)
    report = dict(status="passed_short_complete_learner_capacity_only" if capacity else "completed_bounded_script_free_pilot",
                  dataset=demonstrations, pretraining=pretrain_report, checkpoint=after,
                  checkpoint_sha256=digest(online / "latest.pt"), command=command,
                  final_video_evidence=videos, imitation_evaluation=imitation_eval, independent_evaluation=independent,
                  walking_qualified=False, automatic_extension=False,
                  scope="Measured learning/runtime plus script-free trial evidence; no guaranteed walking or hardware transfer")
    write_json(directory / "report.json", report)
    return report


def main():
    global RUN_JOURNAL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("structural", "preflight", "pilot"), default="structural")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("Seed must be nonnegative")
    structure = structural_report(compose(args.seed))
    if args.mode == "structural":
        print(json.dumps(structure, indent=2, allow_nan=False))
        return
    if args.output is None or args.output.exists():
        parser.error("Use a fresh --output directory; prior evidence is never overwritten")
    from scripts.verify_script_free_bundle import verify
    verify(ROOT.parent / "SOURCE_MANIFEST.json")
    output = args.output.resolve()
    output.mkdir(parents=True)
    RUN_JOURNAL = RunJournal(output, args.mode, args.seed)
    write_json(output / "structural_report.json", structure)
    try:
        with stop_signals():
            report = execute_pipeline(output, args)
    except BaseException as exc:
        interrupted = isinstance(exc, (PipelineStopped, KeyboardInterrupt))
        report = dict(status="interrupted" if interrupted else "failed", mode=args.mode,
                      error=f"{type(exc).__name__}: {exc}", walking_qualified=False,
                      next_step="Preserve logs and inspect status; no automatic retry, resume or training extension")
        write_json(output / "readiness_report.json", report)
        RUN_JOURNAL.update(status=report["status"], child_pid=None, error=report["error"])
        raise
    else:
        write_json(output / "readiness_report.json", report)
        RUN_JOURNAL.update(status="completed", stage_status="completed", child_pid=None, error=None)
        print(json.dumps(report, indent=2, allow_nan=False))
    finally:
        RUN_JOURNAL = None


def execute_pipeline(output, args):
    runtime_dir = output / "runtime"
    runtime_dir.mkdir()
    stage([sys.executable, "-u", "scripts/check_script_free_runtime.py", "--output", str(runtime_dir),
           "--seed", str(args.seed)], output / "runtime.log", 1800)
    runtime = read_json(runtime_dir / "gpu_runtime_report.json")
    if runtime.get("status") != "passed_pinned_gpu_student_runtime_only" or not runtime["reset_validation"]["passed"]:
        raise ValueError("Student GPU runtime/reset report is incomplete")
    capacity = learning_run(output / "capacity", args.seed, capacity=True)
    pilot = learning_run(output / "pilot", args.seed + 100, capacity=False) if args.mode == "pilot" else None
    return dict(status="passed_gpu_preflight_only" if pilot is None else "completed_bounded_script_free_pilot",
                mode=args.mode, runtime=runtime, capacity_report="capacity/report.json",
                pilot_report=None if pilot is None else "pilot/report.json", gpu_tested=True,
                walking_qualified=False, automatic_extension=False,
                next_step="Review admission and expert videos; explicitly enable the pilot in a fresh run directory" if pilot is None
                          else "Review all independent trial metrics/videos before considering any longer training",
                scope="This report cannot certify fully trained walking or hardware readiness")


if __name__ == "__main__":
    main()
