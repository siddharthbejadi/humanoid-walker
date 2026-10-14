"""Versioned simulator-only observation for a privileged walking teacher.

This is intentionally separate from the 45D hardware-realizable ``proprio``.
It must never be used as a deployable robot observation without distillation.
All joint blocks follow ACTUATOR_NAMES order; feet are left, then right.
"""

import numpy as np


TEACHER_STATE_VERSION = 1
TEACHER_STATE_DIM = 98
TEACHER_STATE_SEGMENTS = {
    "joint_position": slice(0, 19),
    "joint_velocity": slice(19, 38),
    "target_error": slice(38, 57),
    "actuator_force": slice(57, 76),
    "base_linear_velocity": slice(76, 79),
    "base_angular_velocity": slice(79, 82),
    "com_relative_to_feet": slice(82, 84),
    "sole_contact": slice(84, 86),
    "sole_load": slice(86, 88),
    "sole_clearance": slice(88, 90),
    "ideal_fsr": slice(90, 98),
}


def validate_teacher_state(value):
    """Reject missing, nonfinite, or reordered/unexpected teacher vectors."""
    if value.shape[-1] != TEACHER_STATE_DIM:
        raise ValueError(f"teacher_state v{TEACHER_STATE_VERSION} requires {TEACHER_STATE_DIM} values")
    if not np.all(np.isfinite(np.asarray(value))):
        raise ValueError("teacher_state contains nonfinite values")
    return value
