"""Pure JAX schedule and outcome logic for the first single-support pilot.

This is shaping for a bounded *experimental* left-stance/right-swing task. Its
success flags are not the independent 64-seed MuJoCo physical qualification.
"""

import jax.numpy as jnp


PHASE1_STEPS = 400  # 8 seconds at the existing 20 ms control interval.


def phase1_schedule(step):
    """Return phase, commanded swing velocity, time remaining, swing target.

    0 settle, 1 shift, 2 lift, 3 forward, 4 back, 5 lower, 6 hold.
    Positive longitudinal values mean the robot's established forward axis.
    """
    step = jnp.clip(jnp.asarray(step, dtype=jnp.int32), 0, PHASE1_STEPS - 1)
    boundaries = jnp.asarray((50, 100, 150, 200, 250, 300), dtype=jnp.int32)
    phase = jnp.sum(step >= boundaries)
    starts = jnp.asarray((0, 50, 100, 150, 200, 250, 300), dtype=jnp.int32)
    durations = jnp.asarray((50, 50, 50, 50, 50, 50, 100), dtype=jnp.int32)
    remaining = (starts[phase] + durations[phase] - step) / durations[phase]
    # The command is observed by the 45D sensor policy, not privileged state.
    swing_velocity = jnp.asarray((0., 0., 0., .025, -.040, 0., 0.), dtype=jnp.float32)[phase]
    target_x = jnp.asarray((0., 0., 0., .025, -.015, 0., 0.), dtype=jnp.float32)[phase]
    return phase, swing_velocity, remaining, target_x


def phase1_outcomes(phase, *, left_load, right_load, right_clearance,
                    right_forward, upright, tilt, lifted_before,
                    forward_before, backward_before):
    """Measure a strict sequence; an unloaded but planted foot is not a lift."""
    lifted_before = jnp.asarray(lifted_before, dtype=jnp.bool_)
    forward_before = jnp.asarray(forward_before, dtype=jnp.bool_)
    backward_before = jnp.asarray(backward_before, dtype=jnp.bool_)
    support = (left_load > .70) & (right_load < .05) & (tilt < .22) & (upright > .8)
    airborne = support & (right_clearance > .020)
    new_lift = (phase >= 2) & (phase <= 4) & airborne & ~lifted_before
    new_forward = (phase == 3) & airborne & (right_forward >= .020) & ~forward_before
    new_backward = ((phase == 4) & airborne & (right_forward <= -.010)
                    & forward_before & ~backward_before)
    new_landing = ((phase >= 5) & lifted_before & forward_before & backward_before
                   & (right_load > .10) & (left_load > .50) & (tilt < .22))
    return support, airborne, new_lift, new_forward, new_backward, new_landing
