"""Pure recorded-transition checks; coherent commands do not establish walking."""
from __future__ import annotations

import numpy as np

from imitation_data import validate_arrays


CAUSES = ("physical_fall", "wall_hit", "stall_out", "goal_reached", "time_limit")
CONTROL_ATOL_RAD = 2e-6
OBSERVATION_ATOL = 2e-6


def validate_trajectory_contract(arrays, contract):
    """Verify the ordinary full-command clock, reset identity and actuation path.

    The floating-time tolerance covers accumulated float32 substep rounding.
    Generalized states/contact forces still require independent physical review.
    """
    summary = validate_arrays(arrays)
    control = contract.get("control", {}) if isinstance(contract, dict) else {}
    if control.get("action_mode") != "full_command":
        raise ValueError("Trajectory audit requires absolute full-command semantics")
    ranges = np.asarray(control.get("actuator_ctrlrange_rad", []), dtype=np.float64)
    center = np.asarray(control.get("command_center_rad", []), dtype=np.float64)
    scale = np.asarray(control.get("action_scale_rad", []), dtype=np.float64)
    standing = np.asarray(control.get("standing_ctrl_rad", []), dtype=np.float64)
    if (ranges.shape != (19, 2) or center.shape != (19,) or scale.shape != (19,) or standing.shape != (19,) or
            not all(np.isfinite(value).all() for value in (ranges, center, scale, standing)) or
            np.any(ranges[:, 1] <= ranges[:, 0]) or np.any(scale <= 0.) or
            not np.allclose(center, ranges.mean(axis=1), rtol=0., atol=1e-6) or
            not np.allclose(scale, np.diff(ranges, axis=1)[:, 0] / 2., rtol=0., atol=1e-6)):
        raise ValueError("Malformed full-command range/midpoint/halfspan contract")
    dt, slew = control.get("control_dt_seconds"), control.get("slew_rad_per_second")
    horizon = contract.get("task", {}).get("episode_horizon_control_steps")
    if (dt != .02 or control.get("action_repeat") != 10 or slew != 8. or
            type(horizon) is not int or horizon <= 0):
        raise ValueError("Trajectory control period, slew or horizon contract changed")
    episodes, steps = np.asarray(arrays["action"]).shape[:2]
    shapes = {"present": (episodes, steps), "desired_ctrl": (episodes, steps, 19), "applied_ctrl": (episodes, steps, 19),
              "state_ctrl": (episodes, steps + 1, 19), "state_time": (episodes, steps + 1),
              "state_step_count": (episodes, steps + 1),
              "state_state_rng": (episodes, steps + 1, 2), "state_bridge_rng": (episodes, steps + 1, 2),
              **{"cause_" + name: (episodes, steps) for name in CAUSES}}
    missing = set(shapes) - arrays.keys()
    if missing:
        raise ValueError("Missing trajectory identity/control audit arrays: " + ", ".join(sorted(missing)))
    if any(np.asarray(arrays[name]).shape != shape for name, shape in shapes.items()):
        raise ValueError("Malformed trajectory identity/control audit shapes")
    if any(np.asarray(arrays[name]).dtype.kind not in "iu" for name in
           ("state_step_count", "state_state_rng", "state_bridge_rng")):
        raise ValueError("Trajectory step counts and RNG keys must remain integer identities")
    observed = np.asarray(arrays["present"], dtype=bool)
    lengths = observed.sum(axis=1)
    if np.any(lengths == 0) or np.any(lengths > horizon):
        raise ValueError("Each trajectory needs a nonempty prefix within its physical horizon")
    states = np.concatenate((np.ones((episodes, 1), dtype=bool), observed), axis=1)
    counters = np.asarray(arrays["state_step_count"])
    if (np.any(counters[:, 0] != 0) or
            not np.array_equal(counters[:, 1:][observed], np.broadcast_to(np.arange(1, steps + 1), observed.shape)[observed])):
        raise ValueError("Trajectory contains a skipped step or an internal episode reset")
    # The collector prefixes its audit field named state_rng with state_.
    rng = np.asarray(arrays["state_state_rng"])
    if not np.array_equal(rng[states], np.broadcast_to(rng[:, :1], rng.shape)[states]):
        raise ValueError("Per-episode raw RNG identity changed within an observed trajectory")
    causes = []
    for name in CAUSES:
        value = np.asarray(arrays["cause_" + name])
        if not np.isin(value, [False, True]).all() or np.any(value.astype(bool) & ~observed):
            raise ValueError("Termination causes must be binary and absent from padding")
        causes.append(value.astype(bool))
    terminal = np.asarray(arrays["is_terminal"], dtype=bool)
    physical_terminal = np.logical_or.reduce(causes[:-1])
    expected_limit = observed & (counters[:, 1:] >= horizon) & ~terminal
    final_transition = np.arange(steps)[None] == lengths[:, None] - 1
    if (not np.array_equal(terminal, physical_terminal) or
            not np.array_equal(causes[-1], expected_limit) or
            np.any((physical_terminal | causes[-1]) & ~final_transition)):
        raise ValueError("Terminal/time-limit causes do not match the first-episode boundary")
    ctrl = np.asarray(arrays["state_ctrl"], dtype=np.float64)
    low, high = ranges[:, 0], ranges[:, 1]
    if (np.any(ctrl[states] < low - CONTROL_ATOL_RAD) or np.any(ctrl[states] > high + CONTROL_ATOL_RAD) or
            not np.allclose(ctrl[:, 0], standing, rtol=0., atol=CONTROL_ATOL_RAD)):
        raise ValueError("Recorded controls violate the standing reset or actuator bounds")
    action = np.asarray(arrays["action"], dtype=np.float64)
    target = np.clip(center + scale * action, low, high)
    desired = np.clip(np.asarray(arrays["desired_ctrl"], dtype=np.float64), low, high)
    mapping_error = np.abs(target - desired)[observed]
    expected_ctrl = np.clip(target, ctrl[:, :-1] - slew * dt, ctrl[:, :-1] + slew * dt)
    slew_error = np.abs(ctrl[:, 1:] - expected_ctrl)[observed]
    applied_error = np.abs(np.asarray(arrays["applied_ctrl"], dtype=np.float64) - ctrl[:, 1:])[observed]
    if any(np.any(value > CONTROL_ATOL_RAD) for value in (mapping_error, slew_error, applied_error)):
        raise ValueError("Recorded desired/action/applied controls disagree with absolute mapping and slew")
    commands = np.clip(2. * (ctrl - low) / (high - low) - 1., -1., 1.)
    observation_error = np.abs(np.asarray(arrays["proprio"], dtype=np.float64)[..., :19] - commands)[states]
    if np.any(observation_error > OBSERVATION_ATOL):
        raise ValueError("Student command observation does not match the current applied control")
    recorded_time = np.asarray(arrays["state_time"])
    time = recorded_time.astype(np.float64)
    epsilon = np.finfo(recorded_time.dtype).eps if recorded_time.dtype.kind == "f" else np.finfo(np.float32).eps
    tolerance = 5e-6 + 16. * epsilon * np.maximum(1., np.maximum(np.abs(time[:, :-1]), np.abs(time[:, 1:])))
    clock_error = np.abs(np.diff(time, axis=1) - dt)[observed]
    if np.any(np.abs(time[:, 0]) > 5e-6) or np.any(clock_error > tolerance[observed]):
        raise ValueError("Trajectory clock did not reset to zero or advance one 20 ms control interval")
    return dict(episodes=episodes, observed_transitions=summary["present_transitions"],
                max_action_mapping_error_rad=float(mapping_error.max(initial=0.)),
                max_slew_error_rad=float(slew_error.max(initial=0.)),
                max_applied_state_error_rad=float(applied_error.max(initial=0.)),
                max_observation_command_error=float(observation_error.max(initial=0.)),
                max_clock_increment_error_s=float(clock_error.max(initial=0.)),
                time_tolerance_max_s=float(tolerance[observed].max(initial=0.)),
                scope="Recorded first-episode identity, clock and command consistency only; not physics or walking qualification")
