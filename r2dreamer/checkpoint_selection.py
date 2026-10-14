"""Pure ranking logic: exploration checkpoints are not deployment certificates."""
import math


def checkpoint_rank(metric, safety, tie, order="safety_first"):
    if order not in ("safety_first", "task_first"):
        raise ValueError("checkpoint_order must be safety_first or task_first")
    if any(v is None or not math.isfinite(float(v)) for v in (metric, safety, tie)):
        raise ValueError("Missing/non-finite checkpoint selection metrics")
    metric, safety, tie = float(metric), float(safety), float(tie)
    return ((-safety, metric, tie) if order == "safety_first"
            else (metric, -safety, tie))


def walking_trial_success(record):
    """Early gait-progress gate, NOT completion of the configured finish goal."""
    return (record.get("safety_failure_count", record.get("fall_count", float("inf"))) == 0
            and record.get("wall_hit", 0) == 0 and record.get("stall_out", 0) == 0
            and record["alive_seconds"] >= 8.
            and record["total_valid_steps"] >= 4 and record["alternating_steps"] >= 2
            and record["valid_left_step"] >= 1 and record["valid_right_step"] >= 1
            and record["forward_distance"] >= .04)


def finish_trial_success(record):
    """Only score a finish after the entire trial, never from crossing alone."""
    return (record.get("goal_reached", 0) > 0
            and record.get("safety_failure_count", record.get("fall_count", float("inf"))) == 0
            and record.get("wall_hit", 0) == 0 and record.get("stall_out", 0) == 0)
