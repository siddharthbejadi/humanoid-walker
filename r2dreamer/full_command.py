"""Complete motor-target mapping shared by the student and external expert collector."""

import numpy as np


def normalize_motor_targets(targets, command_center, command_scale, *, array_module=np):
    """Convert finite physical targets to normalized actions; callers validate labels."""
    return array_module.clip((targets - command_center) / command_scale, -1., 1.)


def motor_targets_from_actions(actions, command_center, command_scale, *, array_module=np):
    return command_center + command_scale * array_module.nan_to_num(
        array_module.clip(actions, -1., 1.))


def validate_script_free_settings(*, full_command, task, reward_profile,
                                 gait_residual, rsi_stratified, phase_obs,
                                 phase_control, gait_curriculum, footsteps,
                                 references):
    if not full_command:
        if reward_profile == "autonomous_walk_v1":
            raise ValueError("autonomous_walk_v1 requires full_command")
        return
    if task != "walking" or reward_profile != "autonomous_walk_v1":
        raise ValueError("Full-command walking requires autonomous_walk_v1")
    if any((gait_residual, rsi_stratified, phase_obs, phase_control, gait_curriculum)) or footsteps is not None:
        raise ValueError("Full-command student forbids reference, phase, RSI and footstep controllers")
    if any(value is not None for value in references):
        raise ValueError("Full-command student requires null gait, pose, CSV and stabilizer references")
