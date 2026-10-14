"""Physical preservation checks for a bounded recovery experiment."""
from __future__ import annotations

import math


GAIT_KEYS = ("walking_success_rate", "valid_left_step", "valid_right_step", "alternating_steps")


def gait_metrics(summary):
    values = {key: float(summary[key]) for key in GAIT_KEYS}
    values["safety_failure_count"] = float(summary["safety_failure_count"])
    values["bilateral_min_steps"] = min(values["valid_left_step"], values["valid_right_step"])
    if any(not math.isfinite(value) or value < 0. for value in values.values()):
        raise ValueError("Gait guard requires finite nonnegative physical metrics")
    if values["walking_success_rate"] > 1. or values["safety_failure_count"] > 1.:
        raise ValueError("Gait guard requires per-trial success and failure rates")
    return values


def preservation_result(baseline, summary, *, minimum_initial_success=.5,
                        success_fraction=.8, step_fraction=.7,
                        extra_failure_rate=2. / 32.):
    current = gait_metrics(summary)
    settings = (minimum_initial_success, success_fraction, step_fraction, extra_failure_rate)
    if any(not math.isfinite(value) or not 0. <= value <= 1. for value in settings):
        raise ValueError("Invalid gait-preservation thresholds")
    if baseline is None:
        failures = []
        if current["walking_success_rate"] < minimum_initial_success:
            failures.append("initial_early_gait_success_below_minimum")
        if any(current[key] <= 0. for key in GAIT_KEYS[1:]):
            failures.append("initial_bilateral_or_alternating_gait_missing")
        return {"criterion": "bilateral_minimum_and_alternation_v2",
                "phase": "baseline", "passed": not failures,
                "baseline": current, "current": current, "failures": failures}
    baseline = gait_metrics(baseline)
    floors = {"walking_success_rate": baseline["walking_success_rate"] * success_fraction,
              "bilateral_min_steps": baseline["bilateral_min_steps"] * step_fraction,
              "alternating_steps": baseline["alternating_steps"] * step_fraction}
    ceiling = min(1., baseline["safety_failure_count"] + extra_failure_rate)
    failures = [key + "_below_floor" for key, floor in floors.items()
                if current[key] + 1e-12 < floor]
    if current["safety_failure_count"] > ceiling + 1e-12:
        failures.append("safety_failure_count_above_ceiling")
    return {"criterion": "bilateral_minimum_and_alternation_v2",
            "phase": "preservation", "passed": not failures,
            "baseline": baseline, "current": current, "floors": floors,
            "failure_rate_ceiling": ceiling, "failures": failures}
