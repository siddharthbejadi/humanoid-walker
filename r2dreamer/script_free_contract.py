"""Admission checks for the full-command, demonstration-trained student."""
from __future__ import annotations

import json

import numpy as np

from training_contract import check_checkpoint_contract


def validate_student_config(config):
    env = config.env
    if not bool(getattr(env, "mjx_full_command", False)):
        raise ValueError("Script-free training requires full joint-command control")
    for name in ("mjx_scripted_gait", "mjx_scripted_phase_pool", "gait_reference_csv",
                 "gait_reference_signs", "pose_rsi_json", "mjx_stabilizer"):
        if getattr(env, name, None) is not None:
            raise ValueError(f"Student must not depend on {name}")
    for name in ("mjx_gait_residual", "mjx_phase_obs", "rsi_stratified", "mjx_footsteps",
                 "encoder_training_aid", "auxiliary_state_targets"):
        if bool(getattr(env, name, False)):
            raise ValueError(f"Student dependency is forbidden: {name}")
    if float(getattr(env, "mjx_phase_control", 0.)) != 0.:
        raise ValueError("Student must not use an externally controlled gait clock")
    if env.mjx_reward_profile != "autonomous_walk_v1":
        raise ValueError("Student requires the standalone locomotion reward")
    for encoder in (env.encoder, config.model.encoder):
        if encoder.mlp_keys != "^proprio$" or encoder.cnn_keys != "$^":
            raise ValueError("The actor encoder must accept only sensor-style proprio")
    if bool(getattr(config.model, "auxiliary_state", False)):
        raise ValueError("Privileged auxiliary targets are not part of this student recipe")
    if bool(config.allow_unverified_resume) or config.transfer_from is not None:
        raise ValueError("Unverified/partial legacy weight loading is forbidden")
    settings = config.imitation
    if (int(settings.batch_size), int(settings.batch_length), int(settings.burn_in)) != (16, 64, 64):
        raise ValueError("Imitation batch/context contract must remain 16 x (64 + 64)")
    weights = np.asarray([config.model.imitation_weight, config.model.imitation_pretrain_weight], dtype=float)
    if not np.isfinite(weights).all() or np.any(weights <= 0.):
        raise ValueError("Both offline imitation and the online imitation anchor are required")
    distribution = config.model.actor.dist.cont
    if (float(distribution.min_std), float(distribution.max_std)) != (.0003, .002):
        raise ValueError("Full-command exploration must preserve the declared small physical noise")
    return {"full_command": True, "encoder_inputs": ["proprio"],
            "trajectory_playback": False, "explicit_stabilizer": False,
            "phase_navigation_remapping": False, "privileged_actor_inputs": False,
            "scope": "Script-independent simulator student; sensor calibration and hardware validation remain pending"}


def validate_student_environment(contract):
    if not isinstance(contract, dict):
        raise ValueError("Student requires an effective environment contract")
    if contract.get("version") != 1 or contract.get("backend") != "mjx-warp":
        raise ValueError("Unsupported student contract version or backend")
    if contract.get("interface", {}).get("observations") != {"proprio": [45]}:
        raise ValueError("Student observation interface must be exactly 45 proprio values")
    if contract.get("interface", {}).get("action_shape") != [19]:
        raise ValueError("Student must command all 19 joints")
    if contract.get("control", {}).get("action_mode") != "full_command":
        raise ValueError("Residual command semantics cannot be loaded into the student")
    controller = contract.get("controller", {})
    if (controller.get("full_command") is not True or controller.get("gait_residual") is not False or
            controller.get("reference_kind") != "none" or controller.get("stabilizer") is not None or
            controller.get("phase_observations") is not False or controller.get("phase_rate_range") != 0.):
        raise ValueError("Student environment still contains a scripted controller dependency")
    if (controller.get("identity") != "autonomous_full_command_v1" or
            controller.get("observation_semantics") != "proprio45_sensor_slots_no_reference_v1"):
        raise ValueError("Unrecognized full-command controller or observation semantics")
    control = contract.get("control", {})
    ranges = np.asarray(control.get("actuator_ctrlrange_rad", []), dtype=float)
    center = np.asarray(control.get("command_center_rad", []), dtype=float)
    scale = np.asarray(control.get("action_scale_rad", []), dtype=float)
    if (ranges.shape != (19, 2) or center.shape != (19,) or scale.shape != (19,) or
            not all(np.isfinite(value).all() for value in (ranges, center, scale)) or
            np.any(scale <= 0.) or np.any(ranges[:, 1] <= ranges[:, 0]) or
            not np.allclose(center, ranges.mean(axis=1), rtol=0., atol=1e-6) or
            not np.allclose(scale, (ranges[:, 1] - ranges[:, 0]) / 2., rtol=0., atol=1e-6)):
        raise ValueError("Full-command normalization must cover all 19 finite actuator ranges")
    if (control.get("control_dt_seconds") != .02 or control.get("action_repeat") != 10 or
            not 5. <= control.get("slew_rad_per_second", 0.) <= 8.):
        raise ValueError("Full-command control clock/slew contract changed")
    task = contract.get("task", {})
    if (task.get("name") != "walking" or task.get("reward_profile") != "autonomous_walk_v1" or
            task.get("gait_curriculum") is not False or task.get("footsteps") is not None or
            task.get("valid_step_clearance_m") != .015 or
            task.get("valid_step_clearance_comparison") != "greater_or_equal" or
            type(task.get("episode_horizon_control_steps")) is not int or
            task["episode_horizon_control_steps"] <= 0):
        raise ValueError("Student requires the declared standalone walking task and 15 mm detector")
    if any(controller.get(name) is not None for name in
           ("mjx_scripted_gait_sha256", "gait_reference_csv_sha256", "mjx_scripted_phase_pool_sha256",
            "effective_reference_sha256")):
        raise ValueError("Student contract still fingerprints a runtime reference")
    reset = contract.get("reset", {})
    if reset.get("rsi_stratified") is not False or reset.get("source_kind") != "standing":
        raise ValueError("Student reset path must not read a scripted phase pool")
    if (reset.get("joint_noise_rad") != .005 or reset.get("velocity_noise_rad_s") != .005):
        raise ValueError("Student standing-reset noise contract changed")
    json.dumps(contract, allow_nan=False)
    return contract


def check_imitation_source(dataset, current_contract, checkpoint=None, minimum_updates=1):
    validate_student_environment(current_contract)
    saved_contract = dataset.manifest.get("training_contract")
    validate_student_environment(saved_contract)
    check_checkpoint_contract({"training_contract": saved_contract}, current_contract, mode="resume")
    fingerprint = dataset.fingerprint
    json.dumps(fingerprint, allow_nan=False)
    if checkpoint is None:
        return {"version": 1, "mode": "full_command_bc", "dataset_fingerprint": fingerprint}
    check_checkpoint_contract(checkpoint, current_contract, mode="resume")
    metadata = checkpoint.get("imitation")
    if not isinstance(metadata, dict) or metadata.get("version") != 1 or metadata.get("mode") != "full_command_bc":
        raise ValueError("Full-command fine-tuning requires a verified imitation checkpoint")
    if metadata.get("dataset_fingerprint") != fingerprint:
        raise ValueError("Checkpoint and online imitation dataset fingerprints disagree")
    updates = metadata.get("pretraining_updates")
    if type(updates) is not int or updates < int(minimum_updates):
        raise ValueError("Imitation checkpoint has insufficient verified pretraining updates")
    return metadata
