"""Setup-only checkpoint identity for the MJX teacher's physical task.

Resume preserves optimizer/model state only for a matching environment contract.
``init_from`` and ``transfer_from`` explicitly adapt to a new physical task;
they still require matching observation/action shapes. Legacy unverified resume
needs an explicit opt-in. This is not an exact RNG/replay-state continuation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


CONTRACT_VERSION = 1


def _hash_file(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _array(value):
    # These small device-to-host reads happen once before training, never while
    # collecting transitions or saving later checkpoints.
    return np.asarray(value).tolist()


def _hash_arrays(values):
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(name.encode("utf-8"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _scene_files(scene):
    """Fingerprint XML plus referenced external assets, independent of root path.

    The bundled scene has no includes. Included MJCF is tracked recursively;
    unresolved files fail closed rather than silently weaken the fingerprint.
    Compiled mesh arrays below additionally record geometry used by MuJoCo.
    """
    scene = Path(scene).resolve(strict=True)
    records = {"xml": _hash_file(scene)}
    visited = set()

    def visit(path, key, inherited):
        if path in visited:
            return
        visited.add(path)
        root = ET.parse(path).getroot()
        compiler = dict(inherited)
        for element in root.iter("compiler"):
            compiler.update(element.attrib)
        for index, element in enumerate(root.iter()):
            filename = element.get("file")
            if not filename:
                continue
            if element.tag == "include":
                resolved = (path.parent / filename).resolve(strict=True)
                child_key = f"{key}/include:{index}"
                records[child_key] = _hash_file(resolved)
                visit(resolved, child_key, compiler)
                continue
            assetdir = compiler.get("assetdir", "")
            directory = (compiler.get("meshdir", assetdir) if element.tag == "mesh"
                         else compiler.get("texturedir", assetdir)
                         if element.tag == "texture" else assetdir)
            resolved = (path.parent / directory / filename).resolve(strict=True)
            records[f"{key}/{element.tag}:{index}"] = _hash_file(resolved)

    visit(scene, "xml", {})
    return records


def _configured_file(config, name):
    value = getattr(config, name, None) if config is not None else None
    if not value:
        return None
    path = Path(str(value)).expanduser()
    # Matches MJXTeacherEnv's bundled reference resolution.
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path.resolve(strict=True)


def build_training_contract(env, env_config=None):
    """Return JSON-safe effective MJX settings, or None for another backend.

    Parallel world count, eval mix, logging, device and run budgets are omitted:
    they may change without changing a policy's physical command semantics.
    Episode horizon and per-world contact capacity are part of the contract.
    """
    raw = getattr(env, "raw", None)
    if raw is None or getattr(raw, "backend", None) != "mjx-warp":
        return None
    model = raw.m
    generated_scene = Path(raw.scene).resolve(strict=True)
    source_value = getattr(env_config, "scene_path", None) if env_config is not None else None
    source_scene = (Path(str(source_value)).expanduser().resolve(strict=True)
                    if source_value else generated_scene)
    observation_space = env.observation_space
    spaces = getattr(observation_space, "spaces", observation_space)
    contract = {
        "version": CONTRACT_VERSION,
        "backend": "mjx-warp",
        "interface": {
            "observations": {str(name): list(space.shape) for name, space in sorted(spaces.items())},
            "action_shape": list(env.action_space.shape),
        },
        "control": {
            "actuator_names": [model.actuator(i).name for i in range(model.nu)],
            "action_mode": "full_command" if getattr(raw, "full_command", False) else "residual_offset",
            "command_center_rad": _array(getattr(raw, "command_center", raw.standing_ctrl)),
            "action_scale_rad": _array(raw.action_scale),
            "standing_ctrl_rad": _array(raw.standing_ctrl),
            "actuator_ctrlrange_rad": _array(model.actuator_ctrlrange),
            "actuator_forcerange_nm": _array(model.actuator_forcerange),
            "actuator_gainprm": _array(model.actuator_gainprm),
            "actuator_biasprm": _array(model.actuator_biasprm),
            "control_dt_seconds": float(raw.control_dt),
            "action_repeat": int(raw.action_repeat),
            "slew_rad_per_second": float(raw.control_slew_rad_s),
        },
        "observation_mask": {"fsr": bool(getattr(raw, "mask_fsr_obs", False)),
                             "mag": bool(getattr(raw, "mask_mag_obs", False))},
        "realism": {"command_delay_steps": int(getattr(raw, "command_delay_steps", 0)),
                    "imu_accel_noise": float(getattr(raw, "imu_accel_noise", 0.)),
                    "imu_gyro_noise": float(getattr(raw, "imu_gyro_noise", 0.))},
        "physics": {
            "source_files": _scene_files(source_scene),
            "generated_files": _scene_files(generated_scene),
            "timestep_seconds": float(model.opt.timestep),
            "contacts_per_world": int(raw.naconmax) // int(env.env_num),
            "compiled_geometry_sha256": _hash_arrays({
                name: getattr(model, name) for name in (
                    "body_mass", "body_inertia", "body_pos", "body_quat",
                    "jnt_range", "geom_pos", "geom_quat", "geom_size",
                    "geom_friction", "mesh_vert", "mesh_face")}),
        },
        "task": {
            "name": str(raw.task),
            "target_forward_speed_m_s": float(getattr(raw, "target_forward_speed", .08)),
            "reward_profile": str(raw.reward_profile),
            "valid_step_clearance_m": float(raw.step_clearance_m),
            "valid_step_clearance_comparison": "greater_or_equal",
            "simplified_rewards": bool(raw.simplified_rewards),
            "gait_curriculum": bool(raw.gait_curriculum),
            "backward_drift_weight": float(raw.backward_drift_weight),
            "stance_knee_sag_weight": float(raw.stance_knee_sag_weight),
            "episode_horizon_control_steps": int(env.horizon),
            "reward_sizes": dict(getattr(raw, "reward_sizes", {})),
            "alive_bonus": float(getattr(raw, "alive_bonus", 0.)),
            "standing_posture_weight": float(getattr(raw, "standing_posture_weight", 0.)),
            "footsteps": getattr(raw, "footsteps", None),
        },
        "controller": {
            "full_command": bool(getattr(raw, "full_command", False)),
            "identity": str(getattr(raw, "controller_identity", "legacy_reference_controller")),
            "observation_semantics": ("proprio45_sensor_slots_no_reference_v1"
                                      if getattr(raw, "full_command", False)
                                      else "legacy_teacher_phase_navigation_v1"),
            "gait_residual": bool(getattr(raw, "gait_residual", False)),
            "reference_kind": str(getattr(raw, "gait_reference_kind", "none")),
            "reference_sample_offset_control_steps": float(
                getattr(raw, "gait_reference_sample_offset", 0.)),
            "reference_speed": float(getattr(raw, "gait_reference_speed", 1.)),
            "phase_rate_range": float(getattr(raw, "phase_rate_range", 0.)),
            "phase_observations": bool(getattr(raw, "phase_obs", False)),
            "stabilizer": (list(raw.stabilizer) if getattr(raw, "stabilizer", None) is not None else None),
            "stabilizer_reset_history": "reset_roll_zero_rate_v1",
            "manual_injection_hold_seconds": float(getattr(raw, "INJECT_HOLD_S", 2.)),
            "manual_injection_blend_seconds": float(getattr(raw, "INJECT_BLEND_S", 1.)),
        },
        "reset": {"rsi_stratified": bool(raw.rsi_stratified),
                  "source_kind": str(getattr(raw, "rsi_source_kind", "standing")),
                  "joint_noise_rad": float(getattr(raw, "reset_joint_noise_rad", .005)),
                  "velocity_noise_rad_s": float(getattr(raw, "reset_velocity_noise", 0.)),
                  "injected_pose_min_id": int(getattr(raw, "injected_pose_min_id", 1 << 30))},
    }
    for field in ("mjx_scripted_gait", "gait_reference_csv", "mjx_scripted_phase_pool"):
        source = _configured_file(env_config, field)
        contract["controller"][field + "_sha256"] = _hash_file(source) if source else None
    if getattr(raw, "gait_reference", None) is not None:
        contract["controller"]["effective_reference_sha256"] = _hash_arrays(
            {"joint_targets": raw.gait_reference})
    if getattr(env, "training_pose_id", None) is not None:
        contract["reset"]["training_pose_id"] = int(env.training_pose_id)
    if raw.rsi_stratified:
        reference = (_configured_file(env_config, "pose_rsi_json")
                     or _configured_file(env_config, "gait_reference_csv"))
        contract["reset"].update({
            "probabilities": _array(raw.rsi_probabilities),
            "category_counts": _array(raw.rsi_counts),
            "category_offsets": _array(raw.rsi_offsets),
            "pose_ids": _array(raw.rsi_pose_ids),
            "reference_velocity": bool(raw.rsi_reference_velocity),
            "reference_file_sha256": _hash_file(reference) if reference else None,
            "pool_sha256": _hash_arrays({
                name: getattr(raw, name) for name in (
                    "rsi_qpos", "rsi_qvel", "rsi_ctrl", "rsi_phase_offset")}),
        })
        if getattr(raw, "rsi_injected", None) is not None:
            contract["reset"]["injection_flags_sha256"] = _hash_arrays(
                {"injected": raw.rsi_injected})
    # Track reward/detector/reset and observation implementation as well as its
    # profile name. Merely reusing 'step_progress_v1' after edits is insufficient.
    package = Path(__file__).resolve().parent
    contract["environment_code"] = {
        name: _hash_file(package / name) for name in (
            "envs/mjx_walking/env.py", "envs/mjx_walking/phase1_task.py",
            "envs/mjx_walking/scene.py", "envs/mjx_teacher_env.py",
            "envs/teacher_state.py")}
    if getattr(raw, "full_command", False):
        contract["environment_code"]["full_command.py"] = _hash_file(package / "full_command.py")
    # Reject NaN/Infinity and unsupported objects now, before a long run starts.
    json.dumps(contract, allow_nan=False)
    return contract


def _differences(saved, current, prefix=""):
    if isinstance(saved, dict) and isinstance(current, dict):
        changes = []
        for key in sorted(set(saved) | set(current)):
            name = f"{prefix}.{key}" if prefix else key
            if key not in saved or key not in current:
                changes.append(name)
            else:
                changes.extend(_differences(saved[key], current[key], name))
        return changes
    return [] if saved == current else [prefix]


def check_checkpoint_contract(checkpoint, current_contract, *, mode="resume",
                              allow_unverified_resume=False):
    """Validate before loading checkpoint tensors; return messages to log.

    Known mismatches always reject resume, even with the legacy opt-in flag.
    Explicit init/transfer adapts physical/task semantics with reported diffs;
    shape mismatches still fail. Existing strict tensor loaders remain required.
    """
    if mode not in ("resume", "init", "transfer"):
        raise ValueError(f"Unknown checkpoint load mode: {mode}")
    if current_contract is None:
        return []
    saved = checkpoint.get("training_contract")
    if saved is None:
        if mode == "resume" and not allow_unverified_resume:
            raise ValueError(
                "Checkpoint has no training_contract metadata. Exact physical-task "
                "compatibility is unverified. Use init_from/transfer_from for explicit "
                "adaptation, or set +allow_unverified_resume=true only after verifying "
                "the legacy checkpoint's source/config/action mapping.")
        return [f"UNVERIFIED {mode}: checkpoint has no training_contract; "
                "physical/task compatibility cannot be checked. Strict tensor-shape "
                "loading remains required; replay and RNG are not restored."]
    if not isinstance(saved, dict) or saved.get("version") != CONTRACT_VERSION:
        raise ValueError("Unsupported or malformed checkpoint training_contract version")
    if not isinstance(saved.get("interface"), dict):
        raise ValueError("Checkpoint training_contract is missing its interface")
    shape_changes = _differences(saved["interface"], current_contract["interface"], "interface")
    if shape_changes:
        raise ValueError("Checkpoint observation/action contract mismatch: " + ", ".join(shape_changes))
    changes = _differences(saved, current_contract)
    if changes and mode == "resume":
        raise ValueError(
            "Checkpoint training contract mismatch: " + ", ".join(changes)
            + ". resume_from cannot silently change physical/task semantics; "
            "use init_from or transfer_from for explicit adaptation with a fresh optimizer.")
    if changes:
        return [f"EXPLICIT {mode} ADAPTATION: changed training contract fields: "
                + ", ".join(changes) + ". Physical behavior is not preserved by matching tensor shapes."]
    return [f"Verified {mode} training contract v{CONTRACT_VERSION}; replay and RNG are not restored."]
