"""Pure NumPy checks for repeated MJX teacher resets.

GPU measurements on 2026-10-04 found identical pose/control/RNG state and no
retained-view mutation, but small differences in derived acceleration and
contact observations. Only those channels get the documented bounds below.
This is bounded numerical repeatability, never a bitwise physics guarantee.
"""

from __future__ import annotations

import numpy as np


EXACT_STATE_FIELDS = ("qpos", "qvel", "ctrl", "time", "step_count", "pose_ids",
                      "state_rng", "bridge_rng")
OBSERVATION_GROUPS = {
    "proprio": {
        # Across fresh GPU processes measured acceleration noise reached1.26e-4
        # m/s^2. Bound at0.001m/s^2 (about0.0001g), not a bitwise comparison.
        "last_command": (0, 19, 1e-6), "imu_accel": (19, 22, 1e-3),
        "imu_gyro": (22, 25, 1e-6), "imu_mag": (25, 28, 1e-6),
        "onboard_accel": (28, 31, 1e-3), "onboard_gyro": (31, 34, 1e-6),
        # teacher FSR = raw/5; filtered FSR = tanh(raw/.5), whose
        # slope relative to teacher FSR is at most10. Match its1e-5 bound.
        "task_command": (34, 37, 1e-6), "filtered_fsr": (37, 45, 1e-4),
    },
    "teacher_state": {
        "joint_position": (0, 19, 1e-6), "joint_velocity": (19, 38, 1e-6),
        "tracking_error": (38, 57, 1e-6), "actuator_force": (57, 76, 1e-6),
        "base_linear_velocity": (76, 79, 1e-6),
        "base_angular_velocity": (79, 82, 1e-6), "com_offset": (82, 84, 1e-6),
        "foot_contact": (84, 86, 1e-6), "foot_load": (86, 88, 1e-5),
        "foot_clearance": (88, 90, 1e-6), "fsr": (90, 98, 1e-5),
    },
}


def _measure(reference, current, atol):
    reference, current = np.asarray(reference), np.asarray(current)
    result = dict(atol=float(atol), rtol=0., exact=False, passed=False,
                  reference_shape=list(reference.shape), current_shape=list(current.shape))
    if reference.shape != current.shape:
        return dict(result, error="shape_mismatch")
    if reference.dtype.kind not in "biuf" or current.dtype.kind not in "biuf":
        return dict(result, error="non_numeric_dtype")
    finite = np.isfinite(reference) & np.isfinite(current)
    if not finite.all():
        return dict(result, error="nonfinite", nonfinite_pairs=int((~finite).sum()))
    # float64 avoids unsigned subtraction and retains all float32 differences.
    difference = np.abs(reference.astype(np.float64) - current.astype(np.float64))
    exceeded = difference > atol
    return dict(result, exact=bool(np.array_equal(reference, current)),
                passed=bool(not exceeded.any()), max_abs=float(difference.max(initial=0.)),
                count_over_tolerance=int(exceeded.sum()))


def validate_reset_repeatability(reference, repeated, retained, expected_pose_ids):
    """Validate independent host snapshots, plus original live observation views.

    Required snapshots contain 45D/98D observations and EXACT_STATE_FIELDS.
    ``retained`` is a fresh readback of the original observation views after
    later environment calls; it must match the original snapshots exactly.
    Results are JSON-safe, including malformed input diagnostics.
    """
    report = dict(passed=False, scope="bounded_numerical_reset_repeatability",
                  exact_state={}, observations={}, retained_views={}, errors=[])
    required = ("proprio", "teacher_state", *EXACT_STATE_FIELDS)
    for label, values, fields in (("reference", reference, required),
                                   ("repeated", repeated, required),
                                   ("retained", retained, ("proprio", "teacher_state"))):
        for name in fields:
            if name not in values:
                report["errors"].append(f"{label}.{name}:missing")
    if report["errors"]:
        return report

    reference_shape = np.asarray(reference["proprio"]).shape
    if len(reference_shape) != 2 or reference_shape[0] < 1 or reference_shape[1] != 45:
        report["errors"].append("reference.proprio:expected_nonempty_worlds_by_45")
        return report
    worlds = reference_shape[0]
    shapes = {"proprio": (worlds, 45), "teacher_state": (worlds, 98),
              "ctrl": (worlds, 19), "time": (worlds,), "step_count": (worlds,),
              "pose_ids": (worlds,), "state_rng": (worlds, 2), "bridge_rng": (2,)}
    for label, values in (("reference", reference), ("repeated", repeated)):
        for name, shape in shapes.items():
            if np.asarray(values[name]).shape != shape:
                report["errors"].append(f"{label}.{name}:unexpected_shape")
        for name in ("qpos", "qvel"):
            shape = np.asarray(values[name]).shape
            if len(shape) != 2 or shape[0] != worlds or shape[1] < 1:
                report["errors"].append(f"{label}.{name}:unexpected_shape")
    if report["errors"]:
        return report

    expected_pose_ids = np.asarray(expected_pose_ids)
    if expected_pose_ids.shape != (worlds,):
        report["errors"].append("expected_pose_ids:unexpected_shape")
    else:
        for label, values in (("reference", reference), ("repeated", repeated)):
            if not np.array_equal(values["pose_ids"], expected_pose_ids):
                report["errors"].append(f"{label}.pose_ids:unexpected_schedule")
            if not np.all(np.asarray(values["step_count"]) == 0):
                report["errors"].append(f"{label}.step_count:nonzero_reset")
            if not np.all(np.asarray(values["time"]) == 0):
                report["errors"].append(f"{label}.time:nonzero_reset")

    for name in EXACT_STATE_FIELDS:
        result = _measure(reference[name], repeated[name], 0.)
        report["exact_state"][name] = result
        if not result["passed"] or not result["exact"]:
            report["errors"].append(f"{name}:reset_state_changed")
    for observation, groups in OBSERVATION_GROUPS.items():
        report["observations"][observation] = {}
        for name, (start, end, atol) in groups.items():
            result = _measure(np.asarray(reference[observation])[:, start:end],
                              np.asarray(repeated[observation])[:, start:end], atol)
            report["observations"][observation][name] = result
            if not result["passed"]:
                report["errors"].append(f"{observation}.{name}:repeatability_failed")
        result = _measure(reference[observation], retained[observation], 0.)
        report["retained_views"][observation] = result
        if not result["passed"] or not result["exact"]:
            report["errors"].append(f"{observation}:retained_view_changed")
    report["passed"] = not report["errors"]
    return report
