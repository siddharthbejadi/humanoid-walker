"""Experimental 45D walking environment backed by official MJX-Warp.

The source robot and production MJCF are immutable. A sensor-augmented copy
of the 2 mm rubber-sole candidate supplies public fixed-capacity foot contacts.
This physics task is used by the model-based R2-Dreamer teacher bridge.
"""

from __future__ import annotations

from pathlib import Path

from brax.envs.base import Env, State
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from full_command import motor_targets_from_actions, validate_script_free_settings

from .phase1_task import phase1_outcomes, phase1_schedule
from .scene import (CONTACTS_PER_FOOT, build_mjx_scene, build_pose_rsi_pool,
                    build_vendor_rsi_pool, build_scripted_rsi_pool,
                    rsi_injection_flags)


def foot_progress_records(*, forward_position, contact_before, contact_now,
                          takeoff_forward, landing_record):
    """Exclude stance sliding and backward-reset/forward-repeat step farming."""
    takeoff = contact_before & (~contact_now)
    landed = (~contact_before) & contact_now
    takeoff_forward = jnp.where(takeoff, forward_position, takeoff_forward)
    progress = jnp.minimum(forward_position - takeoff_forward,
                           forward_position - landing_record)
    landing_record = jnp.where(landed, jnp.maximum(landing_record, forward_position),
                               landing_record)
    return progress, takeoff_forward, landing_record


def foot_swing_qualification(*, clearance, load, stable_support, foot_speed_sq,
                             step_clearance_m=.020):
    """Require an unloaded lift with stable, sufficiently still opposite support."""
    return ((clearance >= step_clearance_m) & (load < .10) & (load[::-1] > .55)
            & stable_support & (foot_speed_sq[::-1] < .03 ** 2))


def reference_sample_phase(phase, sample_offset):
    """Script exports label the state after applying the preceding sample."""
    return jnp.maximum(phase + sample_offset, 0.)


def residual_gait_target(gait, inject_ctrl, injected, step_count, control_dt,
                         hold_seconds=2., blend_seconds=1.):
    weight = injected * jnp.clip(
        1. - (step_count * control_dt - hold_seconds) / blend_seconds, 0., 1.)
    return weight * inject_ctrl + (1. - weight) * gait


def stabilizer_reset_state(rotation):
    """Start fresh filter history without differentiating from an invented zero tilt."""
    roll = jnp.arcsin(jnp.clip(rotation[0, 2], -1., 1.))
    return roll, jnp.zeros_like(roll)


def alternating_step_events(touched, last_side):
    """Count ordered side changes, not the minimum of two independent foot counts."""
    side = jnp.argmax(touched).astype(jnp.int32)
    event = jnp.sum(touched.astype(jnp.int32)) == 1
    changed = event & ((last_side < 0) | (side != last_side))
    return jnp.where(changed, side, last_side), changed & (last_side >= 0)


def autonomous_walk_reward_terms(*, velocity, lateral_velocity, tilt, height_error,
                                 load, foot_speed_sq, action, action_delta,
                                 fallen, dt, target_vx=.08):
    """Smooth full-command locomotion shaping, independent of any reference table.

    Strict landing events are diagnostics, not income: their retained support
    threshold can reject a genuine rapid transfer at the sampled touchdown.
    """
    finite = (jnp.isfinite(velocity) & jnp.isfinite(lateral_velocity)
              & jnp.isfinite(tilt) & jnp.isfinite(height_error)
              & jnp.all(jnp.isfinite(load)) & jnp.all(jnp.isfinite(foot_speed_sq))
              & jnp.all(jnp.isfinite(action)) & jnp.all(jnp.isfinite(action_delta)))
    failed = jnp.asarray(fallen, dtype=jnp.bool_) | (height_error < -.10) | ~finite
    alive = (~failed).astype(jnp.float32)
    load = jnp.clip(jnp.nan_to_num(load), 0., 2.)
    foot_speed_sq = jnp.clip(jnp.nan_to_num(foot_speed_sq), 0., 4.)
    quality = (jnp.exp(-(jnp.nan_to_num(tilt) / .25) ** 2
                       - (jnp.nan_to_num(height_error) / .05) ** 2)
               * jnp.clip(jnp.sum(load) / .60, 0., 1.) * alive)
    velocity = jnp.nan_to_num(velocity)
    speed = (jnp.clip(velocity / target_vx, 0., 1.5)
             * jnp.exp(-(jnp.maximum(velocity - target_vx, 0.) / .10) ** 2))
    scrub = jnp.sum(load * foot_speed_sq / .10 ** 2)
    terms = dict(reward_balance=.25 * dt * quality,
                 reward_forward=dt * quality * speed,
                 penalty_scrub=.05 * dt * jnp.clip(scrub, 0., 10.),
                 penalty_lateral=.10 * dt * jnp.clip(
                     (jnp.nan_to_num(lateral_velocity) / target_vx) ** 2, 0., 10.),
                 penalty_action_magnitude=.01 * dt * jnp.mean(jnp.nan_to_num(action) ** 2),
                 penalty_action_rate=.005 * dt * jnp.mean(jnp.nan_to_num(action_delta) ** 2),
                 penalty_fall=3. * failed.astype(jnp.float32))
    reward = (sum(value for key, value in terms.items() if key.startswith("reward_"))
              - sum(value for key, value in terms.items() if key.startswith("penalty_")))
    return reward, terms, failed


# Fraction of body weight the opposite foot must still carry at touchdown. The scripted
# gait shifts weight early (opposite foot ~0.34-0.42 at contact), so the old 0.55 rejected
# every one of its steps; 0.30 still rejects a landing with the other foot unloaded.
LANDING_SUPPORT_LOAD = .30


def foot_landing_events(*, load, contact_before, qualified_before,
                        qualified_now, stable_support, step_forward):
    """Hysteretic landing; a completed step needs an earlier supported lift."""
    stable_support = jnp.asarray(stable_support, dtype=jnp.bool_)
    contact = jnp.where(load > .10, True,
                        jnp.where(load <= .08, False, contact_before))
    landed = (~contact_before) & contact
    touched = (landed & qualified_before & stable_support
               & (load[::-1] > LANDING_SUPPORT_LOAD) & (step_forward >= .020))
    # Every landing consumes the old lift, including a landing in place.
    qualified = jnp.where(landed | (~stable_support), False,
                          qualified_before | (qualified_now & (~contact)))
    return landed, touched, contact, qualified


def walking_reward_terms(*, tilt, height_error, load, clearance, foot_speed,
                         root_forward, forward_record, lift_credit,
                         last_step_side, touched, fallen, dt):
    """Step-progress v1: five rewards and two costs, in control-time units.

    r = .5*dt*Q + .5*sum(delta_credit) + 10*delta_x*G
        + 1*forward_landing + .5*alternating_landing - .05*dt*slip - 3*fall.

    Lift credit is a bounded record, replenished only by a rewarded landing.
    Root progress pays only new distance during supported swing, capped at the
    .08 m/s command. This is task shaping, not policy-invariant PBRS. Either
    foot may start; after that the opposite foot must complete the next step.
    """
    finite = (jnp.isfinite(tilt) & jnp.isfinite(height_error)
              & jnp.isfinite(root_forward) & jnp.isfinite(forward_record)
              & jnp.all(jnp.isfinite(load)) & jnp.all(jnp.isfinite(clearance))
              & jnp.all(jnp.isfinite(foot_speed)) & jnp.all(jnp.isfinite(lift_credit)))
    failed = jnp.asarray(fallen, dtype=jnp.bool_) | (~finite)
    load = jnp.clip(jnp.nan_to_num(load), 0., 2.)
    clearance = jnp.clip(jnp.nan_to_num(clearance), 0., .10)
    foot_speed = jnp.clip(jnp.nan_to_num(foot_speed), 0., 2.)
    tilt = jnp.nan_to_num(tilt, nan=jnp.pi, posinf=jnp.pi, neginf=-jnp.pi)
    height_error = jnp.nan_to_num(height_error, nan=1., posinf=1., neginf=-1.)
    forward_record = jnp.nan_to_num(forward_record)
    root_forward = jnp.where(jnp.isfinite(root_forward), root_forward, forward_record)
    lift_credit = jnp.clip(jnp.nan_to_num(lift_credit), 0., 1.)
    quality = (jnp.exp(-(tilt / .25) ** 2 - (height_error / .06) ** 2)
               * jnp.clip(jnp.sum(load) / .60, 0., 1.) * (~failed))
    support = jnp.clip(load[::-1] / .60, 0., 1.)
    still = jnp.exp(-(foot_speed[::-1] / .10) ** 2)
    # Load asymmetry gives a gradient as soon as weight moves off either foot.
    unweight = jnp.clip((load[::-1] - load) / jnp.maximum(jnp.sum(load), .10), 0., 1.)
    lift_quality = (quality * support * unweight * still
                    * (.25 + .75 * jnp.clip(clearance / .025, 0., 1.)))
    eligible_feet = (last_step_side < 0) | (jnp.arange(2) != last_step_side)
    improvement = jnp.maximum(lift_quality - lift_credit, 0.) * eligible_feet
    paid_credit = jnp.maximum(lift_credit, lift_quality)
    side = jnp.argmax(touched).astype(jnp.int32)
    event = ((jnp.sum(touched.astype(jnp.int32)) == 1) & (~failed)
             & ((last_step_side < 0) | (side != last_step_side)))
    alternating = event & (last_step_side >= 0)
    paid_credit = jnp.where(event & (jnp.arange(2) == side), 0., paid_credit)
    next_side = jnp.where(event, side, last_step_side)
    swing_gate = (quality * support * still
                  * jnp.clip(1. - load / .20, 0., 1.)
                  * jnp.clip((clearance - .002) / .018, 0., 1.))
    advance = jnp.clip(root_forward - forward_record, 0., .08 * dt)
    # Advance the record even when gated off: skating cannot bank later credit.
    next_record = jnp.maximum(forward_record, root_forward)
    slip = jnp.sum(load * (foot_speed / .10) ** 2)
    terms = dict(reward_balance=.5 * dt * quality,
                 reward_lift_progress=.5 * jnp.sum(improvement),
                 reward_forward=10. * advance * jnp.max(swing_gate),
                 reward_touchdown=event.astype(jnp.float32),
                 reward_alternation=.5 * alternating.astype(jnp.float32),
                 penalty_scrub=.05 * dt * slip,
                 penalty_fall=3. * failed.astype(jnp.float32))
    return terms, paid_credit, next_record, next_side


def standing_reward_terms(*, tilt, height_error, load, com_offset, horizontal_speed,
                          angular_speed, foot_speed_sq, action, target_error, fallen,
                          stable, streak_seconds, dt, streak_weight=.5,
                          streak_ramp_seconds=2.0, pose_error=None, posture_weight=0.):
    """Standing task: continuous balance shaping, zeroed on a fall, plus a
    consecutive-stability streak bonus.

    r = alive*(.25*Qu+.15*Qh+.20*Qc+.15*Qb+.15*Qcom+.10*Qv)
        - slip - .002*mean(a^2) - .01*mean(target_error^2) - 3*fall + streak_bonus

    `stable` is the strict standing_stable gate (tight tilt/height/velocity
    thresholds). `streak_seconds` resets to zero the instant `stable` is
    False, so the bonus rewards reaching and holding a stand quickly rather
    than accumulating idle credit across repeated wobble/recover cycles.
    Unlike the raw quality terms, the bonus cannot be earned while falling:
    `stable` requires tilt/height margins far tighter than the fall gate, so
    it is already False whenever `fallen` is True.
    """
    alive = (~fallen).astype(jnp.float32)
    load = jnp.clip(jnp.nan_to_num(load), 0., 2.)
    tilt = jnp.nan_to_num(tilt, nan=jnp.pi, posinf=jnp.pi, neginf=-jnp.pi)
    height_error = jnp.nan_to_num(height_error, nan=1., posinf=1., neginf=-1.)
    com_offset = jnp.nan_to_num(com_offset, nan=1., posinf=1., neginf=-1.)
    foot_speed_sq = jnp.clip(jnp.nan_to_num(foot_speed_sq), 0., 4.)

    upright_quality = jnp.exp(-(tilt / .15) ** 2)
    height_quality = jnp.exp(-(height_error / .03) ** 2)
    # Stance-aware, not fixed-to-double-support: contact_quality only needs
    # SOME foot adequately loaded (single-leg stance has one foot at ~0 load
    # by design), and load_balance_quality's target asymmetry follows
    # whichever regime the robot is actually in -- 0 (even split) in double
    # support, full load-on-one-foot in single support -- rather than always
    # demanding a 50/50 split. Fixes poses 3/4 (single-leg RSI starts),
    # where the old min()/50-50 formulation made genuine single-leg balance
    # score *worse* than letting the lifted foot drop. No behavior change
    # for poses 1/2 or the standing pilot (naturally near-even double
    # support throughout, so single_support stays False there).
    contact_quality = jnp.clip(jnp.max(load) / .35, 0., 1.)
    total_load = load[0] + load[1]
    asymmetry = jnp.abs(load[0] - load[1])
    single_support = jnp.minimum(load[0], load[1]) < .05
    ideal_asymmetry = jnp.where(single_support, total_load, 0.)
    load_balance_quality = jnp.exp(-((asymmetry - ideal_asymmetry) / .30) ** 2)
    com_center_quality = jnp.exp(-jnp.sum((com_offset / .04) ** 2))
    velocity_quality = jnp.exp(-(horizontal_speed / .08) ** 2 - (angular_speed / .20) ** 2)
    quality = (.25 * upright_quality + .15 * height_quality + .20 * contact_quality
               + .15 * load_balance_quality + .15 * com_center_quality
               + .10 * velocity_quality)

    supported_slip = (jnp.sum(foot_speed_sq * jnp.clip(load, 0., 3.))
                      / jnp.maximum(jnp.sum(load), .1))
    slip_penalty = .08 * jnp.clip(supported_slip / (.05 ** 2), 0., 5.)
    action_penalty = .002 * jnp.mean(jnp.nan_to_num(action) ** 2)
    track_penalty = .01 * jnp.mean(jnp.nan_to_num(target_error) ** 2)
    fall_penalty = 3. * fallen.astype(jnp.float32)

    next_streak = jnp.where(jnp.asarray(stable, dtype=jnp.bool_), streak_seconds + dt, 0.)
    streak_bonus = alive * streak_weight * jnp.clip(next_streak / streak_ramp_seconds, 0., 1.)

    # Posture (2026-10-09): pays for every joint being back at the standing keyframe
    # (pose_error = mean squared joint error, rad^2; halves at ~12 deg RMS). Without it a
    # policy can survive by leaning or holding an awkward pose and still score most of
    # the quality terms; from a start pose this asks for recovery to standing straight.
    posture = (alive * posture_weight * jnp.exp(-.6931 * jnp.nan_to_num(pose_error, nan=1.) / .21 ** 2)
               if pose_error is not None else jnp.zeros(()))
    reward = (alive * quality - slip_penalty - action_penalty - track_penalty - fall_penalty + streak_bonus
              + posture)
    terms = dict(standing_reward_upright=alive * .25 * upright_quality,
                 standing_reward_height=alive * .15 * height_quality,
                 standing_reward_contact=alive * .20 * contact_quality,
                 standing_reward_balance=alive * .15 * load_balance_quality,
                 standing_reward_com_center=alive * .15 * com_center_quality,
                 standing_reward_velocity=alive * .10 * velocity_quality,
                 standing_penalty_slip=slip_penalty,
                 standing_penalty_action=action_penalty,
                 standing_penalty_track=track_penalty,
                 standing_penalty_fall=fall_penalty,
                 standing_streak_bonus=streak_bonus,
                 standing_reward_posture=posture)
    return reward, terms, next_streak


def walking_transfer_reward_terms(*, stability_quality, com_offset, forward_quality,
                                  shift_quality, swing_bonus, scrub_penalty, touched,
                                  last_step_side, fallen):
    """Walking-from-standing-transfer reward: v1.

    Built for init_from transfers from a standing-trained checkpoint, not a
    from-scratch gait. Reuses stability_quality/forward_quality/
    shift_quality/swing_bonus/scrub_penalty exactly as already computed in
    step() (the same quantities the "legacy" walking profile uses) instead
    of a new parallel quality formula, plus a com-centering term in the
    style of standing_reward_terms (the transferred policy already optimizes
    it well). Touchdown/alternation reuse the same honest landing detector
    as step_progress_v1 (foot_landing_events/foot_progress_records, called
    by the caller before this function), so "valid step" means the same
    physical thing in both walking profiles -- this function only changes
    the reward weighting, not what counts as a real step.

    Deliberately has NO reward for standing still or a stability streak:
    unlike standing_reward_terms, rewarding stillness here would directly
    fight the forward-walking goal.
    """
    alive = (~fallen).astype(jnp.float32)
    com_center_quality = jnp.exp(-jnp.sum((com_offset / .06) ** 2))
    quality = (.20 * stability_quality + .15 * com_center_quality
               + .30 * forward_quality + .15 * shift_quality + .20 * swing_bonus)
    side = jnp.argmax(touched).astype(jnp.int32)
    event = ((jnp.sum(touched.astype(jnp.int32)) == 1) & (~fallen)
             & ((last_step_side < 0) | (side != last_step_side)))
    alternating = event & (last_step_side >= 0)
    next_side = jnp.where(event, side, last_step_side)
    reward = (alive * quality + event.astype(jnp.float32)
              + .5 * alternating.astype(jnp.float32)
              - scrub_penalty - 3. * fallen.astype(jnp.float32))
    terms = dict(reward_balance=alive * .20 * stability_quality,
                 reward_com_center=alive * .15 * com_center_quality,
                 reward_forward=alive * .30 * forward_quality,
                 reward_shift=alive * .15 * shift_quality,
                 reward_swing=alive * .20 * swing_bonus,
                 reward_touchdown=event.astype(jnp.float32),
                 reward_alternation=.5 * alternating.astype(jnp.float32),
                 penalty_scrub=scrub_penalty,
                 penalty_fall=3. * fallen.astype(jnp.float32))
    return reward, terms, next_side, alternating


def walking_transfer_v2_reward_terms(*, velocity, com_offset, shift_quality, scrub_penalty,
                                     touched, last_step_side, height_error,
                                     torso_lateral_offset, base_fallen,
                                     corridor_half_width=.135, walking_first=False):
    """Walking-from-standing-transfer reward: v2, simplified and strict.

    User-directed redesign (2026-10-06): drops v1's blended
    upright/height/swing quality shaping entirely. Reward is dominated by a
    heavily-weighted forward-velocity term, gated by two hard physical
    constraints treated as fall-equivalent terminal violations (massive
    penalties, not just a shaping deduction):
    1. Torso must not sag more than 10cm below standing height (-100).
    2. The torso must stay within a 27cm-wide virtual corridor
       (+/-13.5cm) (-75). This is NOT a physical wall -- no scene geometry,
       no collision -- only the torso's own lateral position is checked,
       so hands/arms swinging wider than the corridor are explicitly not
       penalized, exactly as requested.

    REMOVED 2026-10-06: a third constraint ("nothing other than the feet
    may touch the ground") was attempted via a tilt > 0.6 rad proxy, since
    exact per-geom contact detection was unverified at the time. Verified
    on a live GPU instance immediately after: this MJX-Warp build's `Data`
    object has NO `contact` attribute at all (confirmed via direct
    introspection -- `hasattr(d, 'contact')` is False, and no `ncon` field
    either), and no touch/collision field of any kind besides
    `actuator_force`. The tilt proxy was then shown empirically to
    over-trigger badly in the first real run using it: normal dynamic
    walking lean under +/-90 degree action authority routinely exceeds 34
    degrees without any other body part touching anything, and the run
    collapsed to 100% falls by step 100k with alive_seconds crashing to
    ~0.5s, far worse than the run before this constraint existed. Removed
    rather than re-guessing a second threshold with no better grounding
    than the first. Correct implementation requires adding dedicated touch
    sensors to the derived scene (scene.py:build_mjx_scene, same mechanism
    already used for the two foot-sole contact sensors) for whichever
    non-foot geoms should count, which is unbuilt -- this is a known gap,
    not a silent omission.

    base_fallen is the environment's existing collapse detector
    (root_height<.17 or upright<.55 or non-finite state); this function
    ORs its own two violations into it and returns the combined value,
    which the caller must reassign to the outer `fallen` so the episode
    actually terminates (the done flag the caller returns), not just the
    reward going to zero while the episode keeps running.

    FIXED 2026-10-07: added height_quality, a smooth shaping term absent
    from the original "simplified and strict" design. Without it, height
    only mattered via the binary -.10 cliff -- zero gradient on approach,
    so nothing discouraged riding right up to the edge of it. A 1M-step
    run confirmed this empirically: total_valid_steps stayed 0 the entire
    run, and by step ~340k the policy had converged on sagging the torso
    to right at the violation boundary (root_drop_peak ~10.3cm, just past
    the 10cm cutoff) on ~100% of eval episodes, dying in ~1s every time --
    classic reward-hacking against an unshaped hard constraint, chasing
    the uncapped forward-velocity term with nothing pushing back before
    the cliff. height_quality decays smoothly (5cm scale) as height_error
    goes negative, giving the optimizer gradient to back off well before
    -.10, not just a cliff it falls off of.

    FIXED 2026-10-07 (second pass): added corridor_quality, the same kind
    of smooth shaping for the lateral-corridor wall that height_quality
    added for the sag wall -- it had the identical bug (a bare
    `jnp.abs(torso_lateral_offset) > corridor_half_width` cliff, no
    gradient on approach). Confirmed empirically: once height_quality
    fixed the sag problem (root_drop_peak dropped 10.3cm -> 2.8cm in the
    very next run), the policy immediately found the next unshaped cliff
    instead -- corridor violations jumped to ~87.5% of eval episodes by
    step 150k, same "ride the wall, nothing pushes back" pattern, just
    relocated. corridor_quality decays smoothly (7cm scale, chosen so
    it's already near-zero at the actual 13.5cm half-width, mirroring
    height_quality's shape) as |torso_lateral_offset| grows.
    """
    height_violation = height_error < -.10
    corridor_violation = jnp.abs(torso_lateral_offset) > corridor_half_width
    fallen = base_fallen | height_violation | corridor_violation
    alive = (~fallen).astype(jnp.float32)

    com_center_quality = jnp.exp(-jnp.sum((com_offset / .06) ** 2))
    height_quality = jnp.exp(-(jnp.clip(-height_error, 0.) / .05) ** 2)
    corridor_quality = jnp.exp(-(torso_lateral_offset / .07) ** 2)
    forward_speed = jnp.clip(velocity, 0., .30)

    side = jnp.argmax(touched).astype(jnp.int32)
    event = ((jnp.sum(touched.astype(jnp.int32)) == 1) & (~fallen)
             & ((last_step_side < 0) | (side != last_step_side)))
    alternating = event & (last_step_side >= 0)
    next_side = jnp.where(event, side, last_step_side)

    base_violation = base_fallen | height_violation
    if walking_first:
        # walking_transfer_v3 (2026-10-08): v2 paid ~16/step for standing still
        # (8 height + 8 corridor), more than walking earns, so the policy froze.
        # Height/corridor quality now only scale forward speed (keeping their
        # anti-sag/anti-drift gradient), and valid steps pay 10x more.
        forward = 75. * forward_speed * height_quality * corridor_quality
        height_term = corridor_term = jnp.zeros(())
        touchdown_weight, alternation_weight = 10., 5.
    else:
        forward = 75. * forward_speed
        height_term, corridor_term = 8. * height_quality, 8. * corridor_quality
        touchdown_weight, alternation_weight = 1., .5
    reward = (alive * (forward + .15 * com_center_quality + .15 * shift_quality
                        + height_term + corridor_term)
              + touchdown_weight * event.astype(jnp.float32)
              + alternation_weight * alternating.astype(jnp.float32)
              - scrub_penalty - 100. * base_violation.astype(jnp.float32)
              - 75. * corridor_violation.astype(jnp.float32))
    terms = dict(reward_forward=alive * forward,
                 reward_com_center=alive * .15 * com_center_quality,
                 reward_shift=alive * .15 * shift_quality,
                 reward_height=alive * height_term,
                 reward_corridor_center=alive * corridor_term,
                 reward_touchdown=touchdown_weight * event.astype(jnp.float32),
                 reward_alternation=alternation_weight * alternating.astype(jnp.float32),
                 penalty_scrub=scrub_penalty,
                 penalty_fall=100. * base_violation.astype(jnp.float32),
                 penalty_height_violation=height_violation.astype(jnp.float32),
                 penalty_corridor=75. * corridor_violation.astype(jnp.float32))
    return reward, terms, next_side, alternating, fallen


GOAL_METRIC_KEYS = ('reward_dot', 'reward_dot_approach', 'dots_reached', 'dot_event_delta', 'physical_fall', 'wall_hit', 'stall_out', 'goal_reached', 'reward_alive', 'reward_track', 'reward_landing', 'reward_progress', 'reward_first_step', 'reward_step', 'reward_goal', 'penalty_stall', 'penalty_stall_out', 'penalty_lateral', 'penalty_heading', 'penalty_wall')


def goal_corridor_reward_terms(*, forward, lateral, yaw_error, height_error, base_fallen,
                               touched, best_forward, stall_anchor, stall_seconds,
                               first_step_timer, first_step_paid, step_anchor, dt,
                               goal_distance=100., wall_half_width=.25, back_wall=-.10,
                               exploration_landing=False, landing_length=None,
                               fall_penalty=100., track_error=None, track_weight=0., alive_bonus=0.,
                               landing_weight=50., first_step_bonus=1000., goal_bonus=10000.,
                               stall_out_penalty=40., stall_out_seconds=5.,
                               action_cost=None, action_weight=0., stall_grace_seconds=1.,
                               still_pays=True, lateral_scale=.05):
    """goal_corridor_v1 (2026-10-08, user-specified): walk 100 m straight down a
    walled corridor (walls left, right and behind; open ahead).

    forward/lateral are metres from the episode start along/across the corridor;
    yaw_error is radians off the straight heading. State arguments come from
    env.info and are returned updated. Ordering this must keep (checked by
    scripts/check_goal_corridor_reward.py): reach goal > walk far > one step then
    stand > stand still > fall; lunge-and-fall < stand; drift/crouch < straight.
      - stall: < 1 cm new ground per second -> -5/s after 1 s; ends at 5 s with -40,
        so stalling (~-60 total) always beats falling (-100)
      - fall / >10 cm sag / wall: -100 and the episode ends
      - first valid step: +1000, paid only after 1 s more without falling
        (lunge-and-fall gets nothing)
      - each later valid step that gained >= 1 cm new ground: +20
      - progress: +100 per metre of new ground, scaled by torso height quality
      - lateral/heading: -10/s at 5 cm or 11 deg, growing quadratically
      - reach goal_distance: +10000 and the episode ends

    goal_corridor_v2 (exploration_landing=True, user-specified 2026-10-08): a step is
    any foot landing >= 2 cm ahead of its standing-pose home (or its best earlier
    landing), with no balance checks, and pays up to +50 (scaled by step length, full
    pay at 8 cm, so tiny shuffles don't out-earn strides) the moment it lands even if the
    robot then falls -- so poses with a leg already moved discover stepping. This
    replaces the +20 later-step reward. Ordering: one step + stay up >> land then
    fall > stand still > fall without landing.

    goal_corridor_v3 (2026-10-08): v2 + fall/wall -200 (so land-then-fall < stand
    still: v2 taught the robot to dive) + track_weight * exp(-track_error / 0.26^2)
    per step for following the vendor gait (track_error = mean squared joint error
    to the reference, rad^2; 0.26 rad = 15 deg).

    still_pays=False (user spec 2026-10-09): standing in one place earns 0 -- the alive
    and gait-tracking income stop once the torso has gained < 1 cm for 1 s -- and costs
    -5/s after stall_grace_seconds (2.5 s); the episode ends at stall_out_seconds.
    """
    height_violation = height_error < -.10
    wall = (jnp.abs(lateral) > wall_half_width) | (forward < back_wall)
    goal = forward >= goal_distance
    fell = base_fallen | height_violation
    alive = ~(fell | wall)

    height_quality = jnp.exp(-(jnp.maximum(-height_error, 0.) / .05) ** 2)
    gained = jnp.maximum(forward - best_forward, 0.)
    best_forward = jnp.maximum(best_forward, forward)
    progress = alive * 100. * gained * height_quality

    moved = best_forward - stall_anchor >= .01
    stall_anchor = jnp.where(moved, best_forward, stall_anchor)
    stall_seconds = jnp.where(moved, 0., stall_seconds + dt)
    stall = alive * 5. * dt * (stall_seconds > stall_grace_seconds)
    # Income (alive bonus, gait tracking) is paid only while the robot is getting somewhere.
    # (stall_anchor > 0: it has gained >= 1 cm at least once, so the first second is not free.)
    earning = alive if still_pays else alive & (stall_seconds < 1.) & (stall_anchor > 0.)
    stalled_out = alive & (stall_seconds >= stall_out_seconds) & ~goal

    landing = (jnp.sum(touched * jnp.minimum(jnp.maximum(landing_length, 0.) / .08, 1.))
               if exploration_landing else jnp.zeros(()))
    stepped = jnp.any(touched) & alive
    start_timer = stepped & (first_step_timer < 0.) & (first_step_paid < .5)
    first_step_timer = jnp.where(start_timer, 0.,
                                 jnp.where(first_step_timer >= 0., first_step_timer + dt, first_step_timer))
    pay_first = alive & (first_step_timer >= 1.) & (first_step_paid < .5)
    first_step_paid = jnp.where(pay_first, 1., first_step_paid)
    first_step_timer = jnp.where(pay_first, -1., first_step_timer)
    later_step = (stepped & (first_step_paid > .5) & ~pay_first & (best_forward - step_anchor >= .01)
                  & (not exploration_landing))
    step_anchor = jnp.where(later_step | start_timer, best_forward, step_anchor)

    track = (earning * track_weight * jnp.exp(-track_error / .26 ** 2)
             if track_error is not None else jnp.zeros(()))
    # alive_bonus (2026-10-09): without income for staying up, an episode doomed to the -200
    # fall scores best by ending at once (per-step penalties only accumulate), so walking
    # policies learned to fall backwards in ~0.6 s. Standing still stays capped by the 5 s
    # stall-out, so the bonus pays for walking, not for standing.
    terms = dict(
        reward_alive=earning * alive_bonus,
        reward_track=track,
        reward_landing=landing_weight * landing,
        reward_progress=progress,
        reward_first_step=first_step_bonus * pay_first.astype(jnp.float32),
        reward_step=20. * later_step.astype(jnp.float32),
        reward_goal=goal_bonus * (goal & alive).astype(jnp.float32),
        penalty_stall=stall,
        penalty_stall_out=stall_out_penalty * stalled_out.astype(jnp.float32),
        penalty_lateral=alive * 10. * dt * (lateral / lateral_scale) ** 2,
        penalty_heading=alive * 10. * dt * (yaw_error / .2) ** 2,
        penalty_wall=fall_penalty * (wall & ~fell).astype(jnp.float32),
        penalty_fall=fall_penalty * fell.astype(jnp.float32),
        penalty_height_violation=height_violation.astype(jnp.float32),
        # Residual walking: cost on correction size (mean squared normalized action) so the
        # policy only departs from the gait when it pays (walk runs drifted to ~7 deg corrections).
        penalty_action_magnitude=(alive * action_weight * action_cost if action_cost is not None
                                  else jnp.zeros(())))
    reward = (sum(v for k, v in terms.items() if k.startswith("reward_"))
              - sum(v for k, v in terms.items() if k.startswith("penalty_") and k != "penalty_height_violation"))
    terms.update(physical_fall=fell.astype(jnp.float32),
                 wall_hit=wall.astype(jnp.float32),
                 stall_out=stalled_out.astype(jnp.float32),
                 goal_reached=(goal & alive).astype(jnp.float32))
    done = fell | wall | goal | stalled_out
    state = dict(goal_best_forward=best_forward, goal_stall_anchor=stall_anchor,
                 goal_stall_seconds=stall_seconds, goal_first_step_timer=first_step_timer,
                 goal_first_step_paid=first_step_paid, goal_step_anchor=step_anchor)
    return reward, terms, state, done


def footstep_target_terms(*, heel_forward, foot_lateral, landed, alive, dot_index, prev_dist,
                          anchor_forward, anchor_lateral, start_side,
                          step=.03, radius=.025, hit_reward=10., approach_weight=50.,
                          toe_forward=None, first_behind_toe=None, force_side=None, max_dots=None):
    """Footstep "dots" (2026-10-09, user spec): each step, the BACK EDGE (heel) of the
    stepping foot must land `step` ahead of where that same foot's heel last landed
    (or stood at reset), on that foot's own line. Feet alternate (zig-zag); dot k
    belongs to start_side when k is even, the other foot when odd (start_side < 0: either
    foot may take step 1 on its own dot). Rewards:
      - hit_reward when the dot's foot LANDS (bears weight) with its heel within `radius`
        of the dot; that foot's anchor moves to where it actually landed, and the next
        dot (other foot) becomes the target.
      - approach_weight * (previous distance - current distance) every step: credit for
        moving the heel toward the dot, taken back for moving away (no farming).
    Arrays are per foot (index 0 left, 1 right). Returns (hit, approach, new_index,
    new_prev_dist, new_start_side, new_anchor_forward, new_anchor_lateral,
    (dx, dy, side_code)); dx/dy = dot minus heel (metres), side_code +1 right, -1 left,
    0 either foot (before step 1 in either-foot mode).

    first_behind_toe (user spec, 2026-10-09 evening): the FIRST step instead counts when the
    stepping heel lands no more than first_behind_toe (5 cm) behind the OTHER foot's toe
    (toe_forward, per foot), within `radius` of its own line sideways. Its approach target is
    1 cm past that line and overshooting is not penalised. Later dots are unchanged.

    force_side (user spec, 2026-10-09): >= 0 credits step 1 to that foot right now, whatever the
    pose (a hand-injected pose IS a step); the zig-zag then continues with the other foot.

    max_dots: stop after this many dots (no more hits or approach credit). With the single-dot
    toe drill the caller passes TOE positions as `heel_forward` (user spec 2026-10-09 night).
    """
    first_rule = first_behind_toe is not None

    def target(index, anchor):
        plain = anchor + step                           # each foot's own next dot
        return jnp.where(index == 0, toe_forward[::-1] - first_behind_toe + .01, plain) if first_rule else plain

    def distance(index, anchor_f, anchor_l):
        fwd = heel_forward - target(index, anchor_f)
        if first_rule:
            fwd = jnp.where(index == 0, jnp.minimum(fwd, 0.), fwd)
        return jnp.sqrt(fwd ** 2 + (foot_lateral - anchor_l) ** 2)

    dot_f = target(dot_index, anchor_forward)
    dot_l = anchor_lateral
    dist = distance(dot_index, anchor_forward, anchor_lateral)
    undecided = start_side < 0
    turn = jnp.where(jnp.mod(dot_index, 2) == 0, jnp.maximum(start_side, 0), 1 - jnp.maximum(start_side, 0))
    near = jnp.argmin(dist)
    side = jnp.where(undecided, near, turn)
    hit_each = alive & landed & (dist < radius)
    if first_rule:
        crossed = (heel_forward >= toe_forward[::-1] - first_behind_toe) & (jnp.abs(foot_lateral - dot_l) < radius)
        hit_each = jnp.where(dot_index == 0, alive & landed & crossed, hit_each)
    forcing = jnp.zeros((), bool)
    if force_side is not None:
        forcing = (force_side >= 0) & (dot_index == 0)
        hit_each = jnp.where(forcing, alive & (jnp.arange(2) == force_side), hit_each)
        undecided = undecided | forcing
    finished = jnp.zeros((), bool) if max_dots is None else dot_index >= max_dots
    hit_each = hit_each & ~finished
    hit = jnp.where(undecided, jnp.any(hit_each), hit_each[turn])
    hit_side = jnp.where(undecided, jnp.where(hit_each[1], 1, 0), turn)
    approach = jnp.where(forcing | finished, 0., alive * approach_weight * (prev_dist - dist[side]))
    new_start = jnp.where(undecided & hit, hit_side, start_side).astype(start_side.dtype)
    new_index = dot_index + hit.astype(dot_index.dtype)
    moved = hit & (jnp.arange(2) == hit_side)
    new_af = jnp.where(moved, heel_forward, anchor_forward)
    new_al = jnp.where(moved, foot_lateral, anchor_lateral)
    nd_f, nd_l = target(new_index, new_af), new_al
    nd = distance(new_index, new_af, new_al)
    still_undecided = new_start < 0
    turn_n = jnp.where(jnp.mod(new_index, 2) == 0, jnp.maximum(new_start, 0), 1 - jnp.maximum(new_start, 0))
    side_n = jnp.where(still_undecided, jnp.argmin(nd), turn_n)
    new_prev = jnp.where(hit, nd[side_n], dist[side])
    obs = (nd_f[side_n] - heel_forward[side_n], nd_l[side_n] - foot_lateral[side_n],
           jnp.where(still_undecided, 0., jnp.where(side_n == 1, 1., -1.)))
    return hit_reward * hit.astype(jnp.float32), approach, new_index, new_prev, new_start, new_af, new_al, obs


class MJXWalkingEnv(Env):
    """One humanoid; Brax's VmapWrapper supplies parallel worlds."""

    def __init__(self, source_xml: str | Path, *, action_repeat: int = 10,
                 worlds: int = 1, contacts_per_world: int = 256,
                 arm_action_scale: float = .25, leg_action_scale: float = .65,
                 other_action_scale: float = .25,
                 control_slew_rad_s: float = 2., step_clearance_m: float = .020,
                 mask_fsr_obs: bool = False, mask_mag_obs: bool = False,
                 command_delay_steps: int = 0, imu_accel_noise: float = 0., imu_gyro_noise: float = 0.,
                 gait_curriculum: bool = False,
                 task: str = "walking", backward_drift_weight: float = 0.,
                 stance_knee_sag_weight: float = 0., rsi_stratified: bool = True,
                 injected_pose_min_id: int | None = None,
                 rsi_probabilities=(.10, .42, .42, .06),
                 gait_reference_csv: str | Path | None = None,
                 pose_rsi_json: str | Path | None = None,
                 pose_rsi_allow_falls: bool = False,
                 gait_reference_signs=None, gait_reference_speed: float = 1.,
                 rsi_reference_velocity: bool = True, simplified_rewards: bool = True,
                 reward_profile: str = "legacy",
                 motor_torque_kgcm: float | None = None,
                 imitation_csv: str | Path | None = None, gait_residual: bool = False,
                 scripted_gait: str | Path | None = None, scripted_phase_pool: str | Path | None = None,
                 phase_control: float | None = None, stabilizer=None, phase_obs: bool = False,
                 alive_bonus: float = 0., reward_sizes: dict | None = None,
                 standing_posture_weight: float = 0., footsteps: dict | None = None,
                 full_command: bool = False, reset_joint_noise_rad: float = .005,
                 reset_velocity_noise: float = 0., target_forward_speed: float = .08):
        self.full_command = bool(full_command)
        if self.full_command or reward_profile == "autonomous_walk_v1":
            validate_script_free_settings(
                full_command=self.full_command, task=task, reward_profile=reward_profile,
                gait_residual=gait_residual, rsi_stratified=rsi_stratified,
                phase_obs=phase_obs, phase_control=phase_control, gait_curriculum=gait_curriculum,
                footsteps=footsteps, references=(scripted_gait, scripted_phase_pool,
                    gait_reference_csv, pose_rsi_json, imitation_csv, gait_reference_signs, stabilizer))
        if not (np.isfinite(reset_joint_noise_rad) and 0. <= reset_joint_noise_rad <= .03):
            raise ValueError("reset_joint_noise_rad must be in [0, 0.03]")
        if not (np.isfinite(reset_velocity_noise) and 0. <= reset_velocity_noise <= .05):
            raise ValueError("reset_velocity_noise must be in [0, 0.05]")
        if not (np.isfinite(target_forward_speed) and .01 <= target_forward_speed <= .30):
            raise ValueError("target_forward_speed must be in [0.01, 0.30]")
        self.reset_joint_noise_rad = float(reset_joint_noise_rad)
        self.reset_velocity_noise = float(reset_velocity_noise)
        self.target_forward_speed = float(target_forward_speed)
        self.step_clearance_m = float(step_clearance_m)
        if not (np.isfinite(self.step_clearance_m)
                and .0145 <= self.step_clearance_m <= .020):
            raise ValueError("step_clearance_m must be within [0.0145, 0.020] metres")
        if scripted_phase_pool is not None and (
                float(gait_reference_speed) != 1. or not rsi_reference_velocity):
            raise ValueError("Scripted phase pools require speed=1 and recorded reference velocities")
        self.scene = build_mjx_scene(Path(source_xml),
                                     motor_torque_kgcm=motor_torque_kgcm)
        m = mujoco.MjModel.from_xml_path(str(self.scene))
        if (m.nq, m.nv, m.nu) != (26, 25, 19):
            raise ValueError(f"Unexpected robot contract: {(m.nq, m.nv, m.nu)}")
        if action_repeat < 1:
            raise ValueError("action_repeat must be positive")
        self.m = m
        if worlds < 1 or contacts_per_world < 32:
            raise ValueError("Invalid contact capacity")
        self.naconmax = worlds * contacts_per_world
        self.action_repeat = action_repeat
        self.control_dt = float(m.opt.timestep * action_repeat)
        if abs(self.control_dt - .02) > 1e-5:
            raise ValueError(f"Expected 20 ms control period, got {self.control_dt}")
        self.key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "standing_start")
        self.base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        if min(self.key_id, self.base_id) < 0:
            raise ValueError("Missing standing_start or base_link")
        self.initial_qpos = jnp.asarray(m.key_qpos[self.key_id], dtype=jnp.float32)
        self.standing_height = float(m.key_qpos[self.key_id, 2])
        w, x, y, z = m.key_qpos[self.key_id, 3:7]
        self.start_yaw = float(np.arctan2(2. * (w * z + x * y), 1. - 2. * (y * y + z * z)))
        self.act_qpos = np.asarray(m.jnt_qposadr[m.actuator_trnid[:, 0]], dtype=np.int32)
        self.act_dof = np.asarray(m.jnt_dofadr[m.actuator_trnid[:, 0]], dtype=np.int32)
        knee_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,
                                     side + "_knee_revolute") for side in ("left", "right")]
        if min(knee_ids) < 0:
            raise ValueError("Both stance-knee joints are required")
        self.knee_qpos = jnp.asarray(m.jnt_qposadr[knee_ids], dtype=jnp.int32)
        self.low = jnp.asarray(m.actuator_ctrlrange[:, 0], dtype=jnp.float32)
        self.high = jnp.asarray(m.actuator_ctrlrange[:, 1], dtype=jnp.float32)
        self.span = self.high - self.low
        self.force_limit = jnp.asarray(
            np.maximum(np.max(np.abs(m.actuator_forcerange), axis=1), 1e-6),
            dtype=jnp.float32)
        self.standing_ctrl = jnp.clip(self.initial_qpos[self.act_qpos], self.low, self.high)
        self.standing_knees = self.initial_qpos[self.knee_qpos]
        max_action_scale = np.pi / 2
        if not .05 <= arm_action_scale <= max_action_scale:
            raise ValueError("arm_action_scale must be in [0.05, pi/2] rad")
        if not .05 <= leg_action_scale <= max_action_scale:
            raise ValueError("leg_action_scale must be in [0.05, pi/2] rad")
        if not .05 <= other_action_scale <= max_action_scale:
            raise ValueError("other_action_scale must be in [0.05, pi/2] rad")
        if not .1 <= control_slew_rad_s <= 20.:
            raise ValueError("control_slew_rad_s must be in [0.1, 20] rad/s")
        arm_names = [m.actuator(i).name for i in range(10, 16)]
        if not all("arm" in name or "shoulder" in name or "elbow" in name for name in arm_names):
            raise ValueError(f"Unexpected arm actuator order: {arm_names}")
        self.action_scale = jnp.asarray([leg_action_scale] * 10 + [arm_action_scale] * 6
                                        + [other_action_scale] * 3,
                                        dtype=jnp.float32)
        self.command_center = self.standing_ctrl
        self.command_scale = self.action_scale
        if self.full_command:
            if not np.all(np.isfinite(m.actuator_ctrlrange)) or np.any(np.asarray(self.span) <= 0.):
                raise ValueError("Full-command actuators require finite positive control ranges")
            self.command_center = (self.low + self.high) / 2.
            self.command_scale = self.span / 2.
            self.action_scale = self.command_scale
        self.control_slew_rad_s = control_slew_rad_s
        # The real robot has no foot-pressure sensing and no calibrated magnetometer: zero those inputs.
        self.mask_fsr_obs, self.mask_mag_obs = bool(mask_fsr_obs), bool(mask_mag_obs)
        # Real-robot realism: servo command latency (control steps of 20 ms) and IMU white noise.
        self.command_delay_steps = int(command_delay_steps)
        self.imu_accel_noise, self.imu_gyro_noise = float(imu_accel_noise), float(imu_gyro_noise)
        if not 0 <= self.command_delay_steps <= 5 or self.imu_accel_noise < 0. or self.imu_gyro_noise < 0.:
            raise ValueError('command_delay_steps must be 0..5 and IMU noise nonnegative')
        if not (0. <= backward_drift_weight <= 1. and
                0. <= stance_knee_sag_weight <= 1.):
            raise ValueError("MJX walking penalty weights must be in [0, 1]")
        self.backward_drift_weight = float(backward_drift_weight)
        self.stance_knee_sag_weight = float(stance_knee_sag_weight)
        self.gait_curriculum = gait_curriculum
        if task not in ("walking", "phase1_right_swing", "standing"):
            raise ValueError(f"Unknown walking task: {task}")
        self.task = task
        self.simplified_rewards = bool(simplified_rewards) and task == "walking"
        if reward_profile not in ("legacy", "step_progress_v1", "walking_transfer_v1",
                                  "walking_transfer_v2", "walking_transfer_v3", "goal_corridor_v1", "goal_corridor_v2", "goal_corridor_v3", "autonomous_walk_v1"):
            raise ValueError(f"Unknown reward profile: {reward_profile}")
        self.reward_profile = reward_profile
        self.mass_bw = float(m.body_subtreemass[0] * 9.81)
        self.forward = jnp.asarray([0., -1.], dtype=jnp.float32)
        self.lateral = jnp.asarray([1., 0.], dtype=jnp.float32)
        self.pad_ids = []
        self.pad_vertices = []
        home = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, home, self.key_id)
        mujoco.mj_forward(m, home)
        self.contact_adrs = []
        for side in ("left", "right"):
            names = [m.geom(i).name for i in range(m.ngeom)]
            candidates = [i for i, name in enumerate(names) if name.startswith(side + "_foot_rubber")]
            if len(candidates) != 1:
                raise ValueError(f"Expected one {side} rubber pad geom, got {candidates}")
            geom = candidates[0]
            self.standing_foot_forward = getattr(self, "standing_foot_forward", []) + [
                float((home.geom_xpos[geom, :2] - home.qpos[:2]) @ np.asarray([0., -1.]))]
            self.standing_foot_lateral = getattr(self, "standing_foot_lateral", []) + [
                float((home.geom_xpos[geom, :2] - home.qpos[:2]) @ np.asarray([1., 0.]))]
            mesh_v = m.mesh_vert[m.mesh_vertadr[m.geom_dataid[geom]]:
                                 m.mesh_vertadr[m.geom_dataid[geom]] + m.mesh_vertnum[m.geom_dataid[geom]]]
            pts = mesh_v @ home.geom_xmat[geom].reshape(3, 3).T + home.geom_xpos[geom]
            self.standing_heel_forward = getattr(self, "standing_heel_forward", []) + [
                float(np.min((pts[:, :2] - home.qpos[:2]) @ np.asarray([0., -1.])))]
            self.standing_toe_forward = getattr(self, "standing_toe_forward", []) + [
                float(np.max((pts[:, :2] - home.qpos[:2]) @ np.asarray([0., -1.])))]
            mesh = int(m.geom_dataid[geom])
            if mesh < 0:
                raise ValueError("Rubber pad must have a mesh for geometric FSR bounds")
            begin, count = int(m.mesh_vertadr[mesh]), int(m.mesh_vertnum[mesh])
            self.pad_ids.append(geom)
            self.pad_vertices.append(jnp.asarray(m.mesh_vert[begin:begin + count], dtype=jnp.float32))
            sensor = m.sensor("mjx_" + side + "_sole_contacts")
            if int(sensor.dim[0]) != CONTACTS_PER_FOOT * 7:
                raise ValueError("Foot contact sensor has wrong shape")
            self.contact_adrs.append(int(sensor.adr[0]))
        self.sensor_slices = {}
        for name in ("imu_accel", "imu_gyro", "imu_mag", "onboard_accel", "onboard_gyro"):
            sensor = m.sensor(name)
            self.sensor_slices[name] = (int(sensor.adr[0]), int(sensor.dim[0]))
        self.rsi_stratified = bool(rsi_stratified)
        self.rsi_source_kind = "standing"
        # Pool poses with id >= this (the Interactive-RSI injected ones) start like a live injection:
        # motors hold the authored pose 2 s, then blend into the gait, and the pose counts as step 1.
        self.injected_pose_min_id = int(injected_pose_min_id) if injected_pose_min_id is not None else 1 << 30
        # Learned clock (phase_control = range r): the head actuator's action sets the reference playback rate
        # 1 + r*a in control steps per step (never above 1 during the start-up part); the head itself holds standing.
        # Built-in tilt stabiliser (the user's zip idea, tuned in scripts/humanlike_walk.py): torso sideways tilt (+ rate)
        # -> ankle roll and hip roll targets, added to the reference before the policy's own correction.
        # stabilizer = (k_roll, d_roll, hip_roll_sign, limit_rad). The ankle signs are mirrored (axes are mirrored).
        self.stabilizer = None if stabilizer is None else tuple(float(v) for v in stabilizer)
        self.phase_obs = bool(phase_obs)
        self.phase_rate_range = float(phase_control or 0.)
        self.cycle_start_idx, self.cycle_len_idx = 1, 1
        if self.phase_rate_range and not gait_residual:
            raise ValueError("phase_control needs gait_residual")
        # Residual walking (2026-10-09): motor target = vendor gait reference(t) + action_scale*action
        # instead of standing pose + action_scale*action. The gait supplies the large joint motion,
        # so the policy only needs small (+-22.5 deg) balance corrections -- the range it learned to
        # use safely in the balance stage (+-1 rad walking collapsed twice).
        self.gait_residual = bool(gait_residual)
        self.gait_reference_kind = "none"
        self.gait_reference_sample_offset = 0.
        self.gait_reference_speed = float(gait_reference_speed)
        self.controller_identity = "autonomous_full_command_v1" if self.full_command else "reference_or_standing_relative"
        self.observation_semantics = ("sensor_command_imu_fsr_45_v1" if self.full_command
                                      else "proprio_with_task_fields_v1")
        self.alive_bonus = float(alive_bonus)
        self.standing_posture_weight = float(standing_posture_weight)
        # Footstep dots (goal_corridor_v3 only): {} or None = off.
        # Any dict (even empty: all defaults) turns dots on; None turns them off.
        self.footsteps = None if footsteps is None else {"on": 1., **{k: float(v) for k, v in footsteps.items()}}
        if self.footsteps and reward_profile != "goal_corridor_v3":
            raise ValueError("footstep dots require reward_profile goal_corridor_v3")
        # Optional sizes for goal_corridor_v3's sparse terms (fall/landing/first step/goal/stall-out).
        # Dreamer's two-hot reward head spreads probability onto rare huge bins, so +1000/+10000
        # spikes inflated every predicted reward (+6.3/step predicted vs -1.3 real) and the policy
        # chased phantom bonuses by diving. Smaller spikes keep the same common-sense ordering.
        self.reward_sizes = {k: float(v) for k, v in (reward_sizes or {}).items()}
        if self.gait_residual and reward_profile != "goal_corridor_v3":
            raise ValueError("gait_residual requires reward_profile goal_corridor_v3")
        if reward_profile == "goal_corridor_v3":
            from .scene import scripted_gait_reference, vendor_gait_reference
            if scripted_gait is None and (imitation_csv is None or gait_reference_signs is None):
                raise ValueError("goal_corridor_v3 needs gait_reference_csv and gait_reference_signs (or scripted_gait)")
            if scripted_gait is not None:
                self.gait_reference_kind = "scripted"
                self.gait_reference_sample_offset = -1.
                table, self.gait_cycle_start, self.gait_period = scripted_gait_reference(
                    scripted_gait, self.control_dt, 400., speed=gait_reference_speed)
            else:
                self.gait_reference_kind = "vendor"
                table, self.gait_cycle_start, self.gait_period = vendor_gait_reference(
                    m, Path(imitation_csv), gait_reference_signs, self.control_dt, 400.,
                    speed=gait_reference_speed)
            self.gait_reference = jnp.asarray(table)
            self.cycle_start_idx = max(int(round(self.gait_cycle_start / self.control_dt)), 1)
            self.cycle_len_idx = max(int(round(self.gait_period / self.control_dt)), 1)
            self.phase_idx = [m.actuator(i).name for i in range(m.nu)].index("act_head_revolute")
        if self.rsi_stratified:
            # Extended 2026-10-06: standing now a legitimate second user,
            # not just walking. Use case: train balance recovery starting
            # from each manually-saved RSI pose directly with the standing
            # reward, before ever reintroducing the walking objective --
            # standing_reward_terms doesn't care how the start pose was
            # chosen, it just scores whatever state results after reset.
            if self.task not in ("walking", "standing"):
                raise ValueError("Stratified RSI requires the walking or standing task")
            if (scripted_phase_pool is None and pose_rsi_json is None
                    and (gait_reference_csv is None or gait_reference_signs is None)):
                raise ValueError("Stratified RSI needs poses or a gait CSV and 19 signs")
            probabilities = np.asarray(rsi_probabilities, dtype=np.float64)
            if (probabilities.shape != (4,) or not np.all(np.isfinite(probabilities))
                    or np.any(probabilities < 0.) or not np.isclose(probabilities.sum(), 1.)):
                raise ValueError("RSI probabilities must be four nonnegative values summing to one")
            if scripted_phase_pool is not None:
                if scripted_gait is None or not self.gait_residual:
                    raise ValueError("Scripted phase resets require a scripted residual gait")
                pool = build_scripted_rsi_pool(m, Path(scripted_phase_pool))
                self.rsi_source_kind = "scripted_phase"
            else:
                self.rsi_source_kind = "manual_pose" if pose_rsi_json is not None else "vendor_reference"
                pool = (build_pose_rsi_pool(m, Path(pose_rsi_json),
                                        np.asarray(self.action_scale),
                                        allow_unstable=pose_rsi_allow_falls)
                    if pose_rsi_json is not None else
                    build_vendor_rsi_pool(m, Path(gait_reference_csv), gait_reference_signs,
                                          np.asarray(self.action_scale), speed=gait_reference_speed))
            unavailable = [name for name, probability, count in zip(
                pool.get("category_names", ("standing", "left_swing", "right_swing", "second_step")),
                    probabilities, pool["counts"]) if probability > 0. and count == 0]
            if unavailable:
                raise ValueError(
                    f"RSI categories {unavailable} have no eligible reset states "
                    f"with the selected pose mode and action scale. "
                    f"Reference scan: {pool['stats']}")
            self.rsi_probabilities = jnp.asarray(probabilities, dtype=jnp.float32)
            self.rsi_counts = jnp.asarray(pool["counts"], dtype=jnp.int32)
            self.rsi_offsets = jnp.asarray(pool["offsets"], dtype=jnp.int32)
            self.rsi_qpos = jnp.asarray(pool["qpos"])
            self.rsi_qvel = jnp.asarray(pool["qvel"])
            self.rsi_ctrl = jnp.asarray(pool["ctrl"])
            self.rsi_phase_offset = jnp.asarray(pool["phase_offset"])
            self.rsi_pose_ids = jnp.asarray(
                pool.get("pose_ids", np.zeros(len(pool["qpos"]), dtype=np.int32)))
            self.rsi_injected = jnp.asarray(rsi_injection_flags(
                np.asarray(self.rsi_pose_ids), manual_pool=self.rsi_source_kind == "manual_pose",
                injected_pose_min_id=self.injected_pose_min_id))
            self.rsi_reference_velocity = bool(rsi_reference_velocity)
            self.rsi_pool_counts = tuple(int(value) for value in pool["counts"])
            self.rsi_reference_stats = pool["stats"]
        # Above 64 worlds MJX-Warp's default CUDA-graph capture fails to allocate collision
        # scratch ("Failed to allocate ... bytes") even with free memory; staged capture
        # runs 512 worlds in ~5 GB. <=64 keeps the default mode every earlier run used.
        if worlds > 64:
            from mujoco.mjx import warp as mjx_warp
            self.mx = mjx.put_model(m, impl="warp", graph_mode=mjx_warp.types.GraphMode.WARP_STAGED)
        else:
            self.mx = mjx.put_model(m, impl="warp")
        self.template = mjx.make_data(m, impl="warp", naconmax=self.naconmax, njmax=1024)

    @property
    def observation_size(self) -> int:
        return 45

    @property
    def action_size(self) -> int:
        return 19

    @property
    def backend(self) -> str:
        return "mjx-warp"

    def _sensor(self, data, name):
        adr, dim = self.sensor_slices[name]
        return data.sensordata[adr:adr + dim]

    INJECT_HOLD_S, INJECT_BLEND_S = 2., 1.

    def _reference_at(self, phase):
        # Reference joint targets at a (fractional) gait position in control steps.
        phase = reference_sample_phase(phase, self.gait_reference_sample_offset)
        i0 = jnp.floor(phase).astype(jnp.int32)
        frac = phase - i0
        a = self.gait_reference[self._gait_index(i0)]
        b = self.gait_reference[self._gait_index(i0 + 1)]
        return a + frac * (b - a)

    def _residual_base(self, info, i, phase=None):
        # Gait reference at control step i (or at gait position `phase`); an injected episode first holds the
        # authored pose (else the gait would swing a hand-placed leg back to standing within ~0.25 s).
        gait = self._reference_at((i + info["phase_offset"]).astype(jnp.float32) if phase is None else phase)
        return residual_gait_target(gait, info["inject_ctrl"], info["injected"], i,
                                    self.control_dt, self.INJECT_HOLD_S, self.INJECT_BLEND_S)

    def _gait_index(self, i):
        # Past the end of the table, keep looping the stride cycle (episodes have no time limit).
        n = self.gait_reference.shape[0]
        start = int(round(self.gait_cycle_start / self.control_dt))
        period = max(int(round(self.gait_period / self.control_dt)), 1)
        return jnp.where(i < n, i, start + jnp.mod(i - start, period))

    def _heel_toe(self, data):
        """Per foot: heel (back edge), pad-centre sideways, toe (front edge), along the walk."""
        heel, lat, toe = [], [], []
        for side in (0, 1):
            gid = self.pad_ids[side]
            points = jnp.einsum("ij,vj->vi", data.geom_xmat[gid], self.pad_vertices[side]) + data.geom_xpos[gid]
            heel.append(jnp.min(points[:, :2] @ self.forward))
            toe.append(jnp.max(points[:, :2] @ self.forward))
            lat.append(data.geom_xpos[gid, :2] @ self.lateral)
        return jnp.stack(heel), jnp.stack(lat), jnp.stack(toe)

    def _heels(self, data):
        """Per foot: back edge (heel) of the rubber pad along the walking direction, and the
        pad centre sideways. Used by the footstep dots."""
        heel, lat = [], []
        for side in (0, 1):
            gid = self.pad_ids[side]
            points = jnp.einsum("ij,vj->vi", data.geom_xmat[gid], self.pad_vertices[side]) + data.geom_xpos[gid]
            heel.append(jnp.min(points[:, :2] @ self.forward))
            lat.append(data.geom_xpos[gid, :2] @ self.lateral)
        return jnp.stack(heel), jnp.stack(lat)

    def _foot(self, data, side):
        gid = self.pad_ids[side]
        vertices = self.pad_vertices[side]
        points = jnp.einsum("ij,vj->vi", data.geom_xmat[gid], vertices) + data.geom_xpos[gid]
        along = points[:, :2] @ self.forward
        across = points[:, :2] @ self.lateral
        lo_x, hi_x = jnp.min(along), jnp.max(along)
        lo_y, hi_y = jnp.min(across), jnp.max(across)
        samples = data.sensordata[self.contact_adrs[side]:self.contact_adrs[side] + CONTACTS_PER_FOOT * 7]
        samples = samples.reshape(CONTACTS_PER_FOOT, 7)
        force = jnp.where(samples[:, 0] > 0., jnp.maximum(samples[:, 1], 0.), 0.)
        normal_bw = force / self.mass_bw
        xy = samples[:, 4:6]
        toe = jnp.clip((xy @ self.forward - lo_x) / jnp.maximum(hi_x - lo_x, 1e-8), 0., 1.)
        high = jnp.clip((xy @ self.lateral - lo_y) / jnp.maximum(hi_y - lo_y, 1e-8), 0., 1.)
        outer = high if side == 0 else 1. - high
        weights = jnp.stack(((1. - toe) * outer, (1. - toe) * (1. - outer),
                             toe * outer, toe * (1. - outer)), -1)
        fsr = jnp.sum(weights * normal_bw[:, None], axis=0)
        return fsr, jnp.sum(normal_bw), jnp.min(points[:, 2]), data.geom_xpos[gid, :2]

    def _features(self, data):
        left = self._foot(data, 0)
        right = self._foot(data, 1)
        fsr = jnp.concatenate((left[0], right[0]))
        load = jnp.stack((left[1], right[1]))
        clearance = jnp.maximum(jnp.stack((left[2], right[2])), 0.)
        foot_xy = jnp.stack((left[3], right[3]))
        return fsr, load, clearance, foot_xy

    def _nav(self, data, goal_start_lateral):
        """Heading error from the corridor direction and sideways offset from the start line."""
        quat = data.qpos[3:7]
        yaw = jnp.arctan2(2. * (quat[0] * quat[3] + quat[1] * quat[2]), 1. - 2. * (quat[2] ** 2 + quat[3] ** 2))
        return jnp.stack((jnp.arctan2(jnp.sin(yaw - self.start_yaw), jnp.cos(yaw - self.start_yaw)),
                          data.xpos[self.base_id, :2] @ self.lateral - goal_start_lateral))

    def _observation(self, data, last_command, filtered_fsr, step_count, dot=None, phase=None, nav=None, imu_key=None):
        # The second IMU (onboard_*) duplicates the first, so its slots carry extra task information:
        #   onboard_accel <- (sin, cos) of the position in the repeating gait cycle and the start-up progress
        #   onboard_gyro  <- (heading error / 0.2 rad, sideways offset / 15 cm, 0)   [teacher observation]
        acc2, gyro2 = self._sensor(data, "onboard_accel"), self._sensor(data, "onboard_gyro")
        if (self.phase_rate_range or self.phase_obs) and phase is not None:
            ang = 2. * jnp.pi * (phase - self.cycle_start_idx) / self.cycle_len_idx
            acc2 = jnp.stack((jnp.sin(ang), jnp.cos(ang), jnp.clip(phase / self.cycle_start_idx, 0., 1.)))
        if self.phase_obs and nav is not None:
            gyro2 = jnp.stack((jnp.clip(nav[0] / .2, -1.5, 1.5), jnp.clip(nav[1] / .15, -1.5, 1.5), jnp.zeros(())))
        if self.task == "phase1_right_swing":
            _, swing_velocity, remaining, _ = phase1_schedule(step_count)
            command = jnp.stack((swing_velocity, jnp.ones_like(remaining), remaining))
        elif self.task == "standing":
            command = jnp.asarray([0., 0., 1.], dtype=jnp.float32)
        elif dot is not None:
            # Footstep dots: where the next dot is relative to the foot that must step on it
            # (dx forward, dy sideways, scaled by 20 cm) and which foot (+1 right, -1 left).
            command = jnp.stack((jnp.clip(dot[0] / .20, -1.5, 1.5), jnp.clip(dot[1] / .20, -1.5, 1.5),
                                 jnp.asarray(dot[2], jnp.float32)))
        elif self.reward_profile == "goal_corridor_v3":
            # Gait clock for imitation: phase of the vendor stride (start-up runs into it).
            phase = 2. * jnp.pi * (step_count * self.control_dt - self.gait_cycle_start) / self.gait_period
            command = jnp.stack((jnp.asarray(.08, jnp.float32), jnp.sin(phase), jnp.cos(phase)))
        else:
            command = jnp.asarray([self.target_forward_speed if self.full_command else .08, 0., 0.], dtype=jnp.float32)
        accel, gyro = self._sensor(data, "imu_accel"), self._sensor(data, "imu_gyro")
        if imu_key is not None and (self.imu_accel_noise or self.imu_gyro_noise):
            k1, k2, k3, k4 = jax.random.split(imu_key, 4)
            accel = accel + self.imu_accel_noise * jax.random.normal(k1, accel.shape)
            gyro = gyro + self.imu_gyro_noise * jax.random.normal(k2, gyro.shape)
            acc2 = acc2 + self.imu_accel_noise * jax.random.normal(k3, acc2.shape)
            gyro2 = gyro2 + self.imu_gyro_noise * jax.random.normal(k4, gyro2.shape)
        mag = self._sensor(data, "imu_mag")
        if self.mask_mag_obs:
            mag = jnp.zeros_like(mag)
        if self.mask_fsr_obs:
            filtered_fsr = jnp.zeros_like(filtered_fsr)
        obs = jnp.concatenate((last_command,
                               accel, gyro,
                               mag, acc2, gyro2, command, filtered_fsr))
        return jnp.nan_to_num(obs, nan=0., posinf=0., neginf=0.)

    def reset(self, rng: jax.Array, forced_rsi_index=-1, use_forced_pose=False,
              forced_qpos=None, forced_qvel=None, forced_ctrl=None) -> State:
        """``use_forced_pose`` is a plain Python bool, never a traced value: it is
        only ever True inside a dedicated jit+vmap wrapper built once in
        MJXTeacherEnv.inject_pose's ``_inject_reset_batch``, never inside the
        normal training ``_reset_batch``/evaluation wrappers, which never pass
        these kwargs at all. That keeps this a static, trace-time branch with
        zero effect on the existing RSI-sampling path or its compiled behavior.
        """
        if self.full_command and (use_forced_pose or forced_rsi_index >= 0):
            raise ValueError("Full-command student resets use standing and bounded joint noise")
        rng, noise_key, category_key, index_key = jax.random.split(rng, 4)
        imu_key = jax.random.fold_in(noise_key, 7)
        qpos = self.initial_qpos.at[self.act_qpos].add(
            jax.random.uniform(noise_key, (19,), minval=-self.reset_joint_noise_rad,
                               maxval=self.reset_joint_noise_rad))
        qvel = jnp.zeros((self.m.nv,), dtype=jnp.float32)
        if self.reset_velocity_noise:
            qvel = qvel.at[self.act_dof].set(jax.random.uniform(
                jax.random.fold_in(noise_key, 1), (19,), minval=-self.reset_velocity_noise,
                maxval=self.reset_velocity_noise))
        ctrl = self.standing_ctrl
        phase_offset = jnp.zeros((), dtype=jnp.int32)
        rsi_category = jnp.zeros((), dtype=jnp.int32)
        rsi_pose_id = jnp.zeros((), dtype=jnp.int32)
        pool_injected = jnp.zeros((), bool)
        if use_forced_pose:
            qpos = forced_qpos
            qvel = forced_qvel
            ctrl = forced_ctrl
            # Sentinel values distinct from the 0-3 real RSI categories/pose ids,
            # so a live-injected episode is identifiable in eval/log breakdowns.
            rsi_category = jnp.asarray(5, dtype=jnp.int32)
            rsi_pose_id = jnp.asarray(999, dtype=jnp.int32)
        elif self.rsi_stratified:
            rsi_category = jax.random.choice(category_key, 4, p=self.rsi_probabilities)
            index = (self.rsi_offsets[rsi_category]
                     + jax.random.randint(index_key, (), 0,
                                          jnp.maximum(self.rsi_counts[rsi_category], 1)))
            # An evaluation schedule may select an exact pool entry. The
            # default -1 preserves the configured training distribution.
            index = jnp.where(forced_rsi_index >= 0, forced_rsi_index, index)
            rsi_category = jnp.sum(index >= self.rsi_offsets).astype(jnp.int32) - 1
            use_reference = rsi_category != 0
            qpos = jnp.where(use_reference, self.rsi_qpos[index], qpos)
            if self.rsi_reference_velocity:
                qvel = jnp.where(use_reference, self.rsi_qvel[index], qvel)
            ctrl = jnp.where(use_reference, self.rsi_ctrl[index], ctrl)
            phase_offset = jnp.where(use_reference, self.rsi_phase_offset[index], phase_offset)
            rsi_pose_id = jnp.where(use_reference, self.rsi_pose_ids[index], rsi_pose_id)
            pool_injected = use_reference & self.rsi_injected[index]
        data = self.template.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)
        data = mjx.forward(self.mx, data)
        fsr, load, clearance, foot_xy = self._features(data)
        filtered = jnp.clip(jnp.tanh(fsr / .5), 0., 1.)
        command = jnp.clip(2. * (ctrl - self.low) / self.span - 1., -1., 1.)
        stab_prev, stab_filt = stabilizer_reset_state(data.xmat[self.base_id])
        info = dict(step_count=jnp.zeros((), dtype=jnp.int32), last_command=command,
                    last_action=jnp.zeros((19,), dtype=jnp.float32),
                    imu_key=imu_key,
                    cmd_queue=jnp.tile(jnp.clip((ctrl - self.command_center) / self.command_scale, -1., 1.)[None],
                                       (max(self.command_delay_steps, 1), 1)).astype(jnp.float32),
                    phase_offset=phase_offset, rsi_category=rsi_category,
                    gait_phase=phase_offset.astype(jnp.float32), stab_prev=stab_prev, stab_filt=stab_filt,
                    rsi_pose_id=rsi_pose_id,
                    # Live-injected pose (Interactive RSI): residual walking holds these motor
                    # targets for INJECT_HOLD_S, then eases into the gait over INJECT_BLEND_S.
                    inject_ctrl=jnp.where(pool_injected, jnp.clip(qpos[self.act_qpos], self.low, self.high), ctrl),
                    injected=jnp.where(pool_injected, 1., 1. if use_forced_pose else 0.).astype(jnp.float32),
                    filtered_fsr=filtered, last_foot_xy=foot_xy,
                    last_touchdown_xy=foot_xy,
                    scrub=jnp.zeros(()), start_forward=qpos[:2] @ self.forward,
                    max_clearance=jnp.where(rsi_category != 0, clearance, jnp.zeros((2,))),
                    valid_steps=jnp.zeros((2,)),
                    qualified_swing=jnp.zeros((2,), dtype=jnp.bool_),
                    foot_in_contact=load > .10,
                    takeoff_forward=foot_xy @ self.forward,
                    landing_forward_record=foot_xy @ self.forward,
                    lift_credit=jnp.zeros((2,), dtype=jnp.float32),
                    forward_record=qpos[:2] @ self.forward,
                    last_step_side=jnp.asarray(-1, dtype=jnp.int32),
                    alternating_steps=jnp.zeros((), dtype=jnp.float32),
                    prev_load=load, standing_stable_seconds=jnp.zeros(()), rng=rng,
                    goal_best_forward=jnp.zeros(()), goal_stall_anchor=jnp.zeros(()),
                    goal_stall_seconds=jnp.zeros(()), goal_first_step_timer=-jnp.ones(()),
                    goal_first_step_paid=jnp.zeros(()), goal_step_anchor=jnp.zeros(()),
                    goal_start_lateral=qpos[:2] @ self.lateral,
                    dot_index=jnp.zeros((), dtype=jnp.int32), dot_prev_dist=jnp.zeros(()),
                    dot_start_side=-jnp.ones((), dtype=jnp.int32),
                    dot_anchor_forward=jnp.zeros((2,)), dot_anchor_lateral=jnp.zeros((2,)))
        dot0 = None
        if self.footsteps:
            fs = self.footsteps
            # first_side 1 = zig-zag starting with the right foot (default), 0 = left,
            # -1 = either foot may take step 1. Each foot's first dot: its heel + step.
            first_side = int(fs.get("first_side", 1.))
            info["dot_start_side"] = jnp.asarray(first_side, dtype=jnp.int32)
            # First dot (user spec): `first_from_toe` (5 cm) in front of each foot's STANDING toe;
            # the HEEL must land on it (right foot: 12.2 cm pad + 5 cm = ~17 cm first step). Later
            # dots: heel `step` (3 cm) ahead of where that heel last landed. The anchor is set so
            # anchor + step = first dot, and standing (root-relative) spots are used so a start
            # pose with a hand-lifted leg still has its first dot in front of the standing toe.
            info["dot_anchor_forward"] = (info["start_forward"] + jnp.asarray(self.standing_toe_forward, jnp.float32)
                                          + fs.get("first_from_toe", .05) - fs.get("step", .03))
            info["dot_anchor_lateral"] = info["goal_start_lateral"] + jnp.asarray(self.standing_foot_lateral, jnp.float32)
            heel_f, heel_l, toe_f = self._heel_toe(data)
            if bool(fs.get("toe_target", 0.)) and fs.get("first_behind_toe") is None:
                # Toe drill: each foot's first dot is `first_from_toe` ahead of its ACTUAL toe (not the standing
                # toe), on its actual line, so a start in the middle of a stride still has sensible dots.
                info["dot_anchor_forward"] = toe_f + fs.get("first_from_toe", .05) - fs.get("step", .03)
                info["dot_anchor_lateral"] = heel_l
            behind = fs.get("first_behind_toe")
            if behind is not None:
                # First step = heel lands <= `behind` behind the other foot's toe; each foot's
                # later dots start from where its heel is now.
                info["dot_anchor_forward"] = heel_f
                first_f = toe_f[::-1] - behind + .01
                d0 = jnp.sqrt(jnp.minimum(heel_f - first_f, 0.) ** 2 + (heel_l - info["dot_anchor_lateral"]) ** 2)
            else:
                if bool(fs.get("toe_target", 0.)):
                    heel_f = toe_f          # toe drill: the TOE must reach the dot
                first_f = info["dot_anchor_forward"] + fs.get("step", .03)
                d0 = jnp.sqrt((heel_f - first_f) ** 2 + (heel_l - info["dot_anchor_lateral"]) ** 2)
            k = jnp.argmin(d0) if first_side < 0 else first_side
            info["dot_prev_dist"] = d0[k]
            dot0 = (first_f[k] - heel_f[k],
                    info["dot_anchor_lateral"][k] - heel_l[k],
                    0. if first_side < 0 else (1. if first_side == 1 else -1.))
        if self.reward_profile in ("goal_corridor_v2", "goal_corridor_v3"):
            # A step is judged from each foot's standing-pose home, so a start pose with a
            # leg already moved forward earns a step when that foot lands.
            home = qpos[:2] @ self.forward + jnp.asarray(self.standing_foot_forward, dtype=jnp.float32)
            info.update(takeoff_forward=home, landing_forward_record=home)
        if self.reward_profile == "step_progress_v1":
            _, info["lift_credit"], _, _ = walking_reward_terms(
                tilt=jnp.arccos(jnp.clip(data.xmat[self.base_id, 2, 2], -1., 1.)),
                height_error=data.xpos[self.base_id, 2] - self.standing_height,
                load=load, clearance=clearance, foot_speed=jnp.zeros((2,)),
                root_forward=info["forward_record"], forward_record=info["forward_record"],
                lift_credit=info["lift_credit"], last_step_side=info["last_step_side"],
                touched=jnp.zeros((2,), dtype=jnp.bool_), fallen=jnp.asarray(False),
                dt=self.control_dt)
        if self.task == "phase1_right_swing":
            info.update(start_foot_xy=foot_xy, phase1_lift_seconds=jnp.zeros(()),
                        phase1_lifted=jnp.asarray(False), phase1_forward_hit=jnp.asarray(False),
                        phase1_backward_hit=jnp.asarray(False), phase1_landing_hit=jnp.asarray(False))
        if self.task == "standing":
            info.update(standing_streak_seconds=jnp.zeros(()))
        metrics = dict(scrub=info["scrub"], left_clearance=clearance[0],
                       right_clearance=clearance[1], left_steps=info["valid_steps"][0],
                       right_steps=info["valid_steps"][1], forward_distance=jnp.zeros(()),
                       shift_quality=jnp.zeros(()), supported_forward=jnp.zeros(()),
                       backward_drift_cost=jnp.zeros(()),
                       stance_knee_sag_cost=jnp.zeros(()),
                       standing_stable_seconds=info["standing_stable_seconds"])
        metrics.update({key: jnp.zeros((), dtype=jnp.float32) for key in (
            "single_support_seconds", "left_support_seconds", "right_support_seconds",
            "left_knee_rad_mean", "right_knee_rad_mean", "root_height_m_mean",
            "body_tilt_rad_mean", "left_load_bw_mean", "right_load_bw_mean",
            "left_clearance_m_peak", "right_clearance_m_peak", "tracking_error_rad_mean",
            "target_clipped_fraction_mean", "reward_upright", "reward_load",
            "body_tilt_rad_peak", "root_drop_m_peak", "stance_knee_bend_rad_peak",
            "actuator_force_fraction_peak", "torque_saturation_fraction_mean",
            "control_slew_limited_fraction_mean",
            "reward_shift", "reward_swing", "reward_forward", "reward_touchdown", "reward_height",
            "raw_scrub_cost", "action_rate_cost", "action_magnitude_cost",
            "penalty_scrub", "penalty_action_rate", "penalty_action_magnitude",
            "penalty_backward_drift", "penalty_knee_sag", "penalty_fall",
            "reward_balance", "reward_lift_progress", "reward_alternation",
            "reward_com_center", "reward_corridor_center", "penalty_height_violation", "penalty_corridor",
            "standing_reward_upright", "standing_reward_height", "standing_reward_contact",
            "standing_reward_balance", "standing_reward_com_center", "standing_reward_velocity",
            "standing_penalty_slip", "standing_penalty_action", "standing_penalty_track",
            "standing_penalty_fall", "standing_streak_bonus", "standing_reward_posture") + GOAL_METRIC_KEYS})
        if self.task == "phase1_right_swing":
            metrics.update(phase1_supported_lift_seconds=jnp.zeros(()),
                           phase1_forward_hit=jnp.zeros(()), phase1_backward_hit=jnp.zeros(()),
                           phase1_landing_hit=jnp.zeros(()), phase1_swing_forward_m=jnp.zeros(()))
        return State(data, self._observation(data, command, filtered,
                                             info["step_count"] + phase_offset, dot=dot0,
                                             imu_key=jax.random.fold_in(imu_key, 0),
                                              phase=None if self.full_command else info["gait_phase"],
                                              nav=None if self.full_command else self._nav(data, info["goal_start_lateral"])),
                     jnp.zeros(()), jnp.zeros(()), metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        data = state.pipeline_state
        action = jnp.nan_to_num(jnp.clip(action, -1., 1.))
        applied_action, cmd_queue = action, state.info["cmd_queue"]
        if self.command_delay_steps:
            applied_action = cmd_queue[0]
            cmd_queue = jnp.concatenate((cmd_queue[1:], action[None]), axis=0)
        base = self.command_center
        phase_new = state.info["gait_phase"] + 1.
        if self.phase_rate_range:
            rate = 1. + self.phase_rate_range * action[self.phase_idx]
            rate = jnp.where(state.info["gait_phase"] < self.cycle_start_idx, jnp.minimum(rate, 1.), rate)
            phase_new = state.info["gait_phase"] + rate
            action = action.at[self.phase_idx].set(0.)         # that channel is the clock, not a joint
        if self.gait_residual:
            base = self._residual_base(state.info, state.info["step_count"] + 1, phase_new)
        stab_prev, stab_filt = state.info["stab_prev"], state.info["stab_filt"]
        if self.stabilizer is not None:
            k_r, d_r, hip_sign, lim = self.stabilizer
            roll = jnp.arcsin(jnp.clip(data.xmat[self.base_id][0, 2], -1., 1.))
            stab_filt = .8 * stab_filt + .2 * (roll - stab_prev) / self.control_dt
            c = jnp.clip(k_r * roll + d_r * stab_filt, -lim, lim)
            base = base + jnp.zeros(19).at[0].set(hip_sign * c).at[5].set(hip_sign * c).at[4].set(c).at[9].set(-c)
            stab_prev = roll
        target = jnp.clip(motor_targets_from_actions(
            applied_action, base, self.command_scale, array_module=jnp), self.low, self.high)
        delta = self.control_slew_rad_s * self.control_dt
        ctrl = jnp.clip(target, data.ctrl - delta, data.ctrl + delta)
        data = data.replace(ctrl=ctrl)
        data = jax.lax.fori_loop(0, self.action_repeat, lambda _, d: mjx.step(self.mx, d), data)
        fsr, load, clearance, foot_xy = self._features(data)
        alpha = 1. - jnp.exp(-self.control_dt / .02)
        filtered = jnp.clip(state.info["filtered_fsr"] + alpha *
                            (jnp.tanh(fsr / .5) - state.info["filtered_fsr"]), 0., 1.)
        step_count = state.info["step_count"] + 1
        command = jnp.clip(2. * (ctrl - self.low) / self.span - 1., -1., 1.)
        obs = self._observation(data, command, filtered,
                                step_count + state.info["phase_offset"],
                                imu_key=jax.random.fold_in(state.info["imu_key"], step_count),
                                phase=None if self.full_command else phase_new,
                                nav=None if self.full_command else self._nav(data, state.info["goal_start_lateral"]))
        rot = data.xmat[self.base_id]
        upright = jnp.clip((rot[2, 2] + 1.) * .5, 0., 1.)
        tilt = jnp.arccos(jnp.clip(rot[2, 2], -1., 1.))
        root_height = data.xpos[self.base_id, 2]
        finite = jnp.all(jnp.isfinite(data.qpos)) & jnp.all(jnp.isfinite(data.qvel))
        fallen = (root_height < .17) | (upright < .55) | (~finite)
        if self.task == "phase1_right_swing":
            fallen = fallen | (tilt >= .22)
        velocity = data.qvel[:2] @ self.forward
        backward_drift_cost = (jnp.clip((-velocity - .02) / .18, 0., 1.) ** 2
                               * (upright > .8) * jnp.clip(jnp.sum(load) / .6, 0., 1.))
        loaded_side = jnp.argmax(load)
        free_side = 1 - loaded_side
        stable_support = ((tilt < .35) & (root_height > self.standing_height - .03)
                          & (~fallen))
        single_support = ((load[loaded_side] > .55) & (load[free_side] < .10)
                          & (clearance[free_side] > .01) & stable_support)
        stance_knee_bend = jnp.maximum(
            self.standing_knees[loaded_side] - data.qpos[self.knee_qpos[loaded_side]], 0.)
        stance_knee_sag_cost = (jnp.clip((load[loaded_side] - .20) / .35, 0., 1.)
            * jnp.clip((stance_knee_bend - .15) / .20, 0., 1.)
            * jnp.clip((self.standing_height - root_height - .01) / .03, 0., 1.))
        
        foot_delta = foot_xy - state.info["last_foot_xy"]
        slip = jnp.linalg.norm(foot_delta, axis=-1) * (load > .05)
        scrub = state.info["scrub"] + jnp.sum(slip)
        foot_speed_sq = jnp.sum((foot_delta / self.control_dt) ** 2, axis=-1)
        raw_scrub_cost = 2.5 * jnp.sum(foot_speed_sq * jnp.clip(load, 0., 3.))
        foot_speed = jnp.sqrt(jnp.maximum(foot_speed_sq, 1e-8))

        # Normalized foot scrub penalty
        supported_speed_sq = (
            jnp.sum(foot_speed_sq * jnp.clip(load, 0., 2.))
            / jnp.maximum(jnp.sum(load), .10)
        )
        scrub_penalty = .04 * jnp.clip(jnp.sqrt(supported_speed_sq) / .15, 0., 3.)

        stance = loaded_side
        swing = 1 - stance

        # Continuous postural and support shaping
        posture_quality = jnp.exp(-((tilt / .20) ** 2))
        height_quality = jnp.exp(-(((root_height - self.standing_height) / .04) ** 2))
        stability_quality = posture_quality * height_quality * (~fallen).astype(jnp.float32)

        support_strength = jnp.clip((load[stance] - .30) / .40, 0., 1.)
        swing_unloaded = 1. - jnp.clip(load[swing] / .15, 0., 1.)
        stance_still = jnp.exp(-((foot_speed[stance] / .10) ** 2))
        clearance_quality = jnp.clip(clearance[swing] / .025, 0., 1.)

        shift_quality = support_strength * swing_unloaded * stability_quality
        swing_bonus = (clearance_quality * support_strength * swing_unloaded
                       * stance_still * stability_quality)

        # Monotonic directional velocity tracking (prevents rewarding 0.0 m/s standing)
        target_v = .08
        vel_progress = jnp.clip(velocity / target_v, 0., 1.5)
        vel_excess = jnp.maximum(velocity - target_v, 0.)
        speed_score = vel_progress * jnp.exp(-((vel_excess / .10) ** 2))
        forward_quality = speed_score * jnp.clip(jnp.sum(load) / .40, 0., 1.) * stability_quality
        supported_forward = forward_quality

        # Step qualification and touchdown detection
        new_max = jnp.maximum(state.info["max_clearance"], clearance)
        opposite_load = load[::-1]
        qualified = ((clearance > .018) & (load < .12) & (opposite_load > .50)
                     & stable_support & (foot_speed_sq[::-1] < .05 ** 2))
        qualified_swing = state.info["qualified_swing"] | qualified

        landed = (load > .10) & (state.info["prev_load"] <= .08)
        step_forward = (foot_xy - state.info["last_touchdown_xy"]) @ self.forward
        touched = landed & qualified_swing & (opposite_load > .50) & stable_support & (step_forward >= .015)
        foot_in_contact = load > .10
        takeoff_forward = state.info["takeoff_forward"]
        landing_forward_record = state.info["landing_forward_record"]
        if (self.reward_profile in ("step_progress_v1", "walking_transfer_v1",
                                    "walking_transfer_v2", "walking_transfer_v3", "goal_corridor_v1", "goal_corridor_v2", "goal_corridor_v3", "autonomous_walk_v1")
                and self.task == "walking"):
            # The configured lift threshold is independent of forward step length.
            # Support, stance speed and landing history retain their strict gates.
            qualified = foot_swing_qualification(
                clearance=clearance, load=load, stable_support=stable_support,
                foot_speed_sq=foot_speed_sq, step_clearance_m=self.step_clearance_m)
            contact_now = jnp.where(load > .10, True,
                jnp.where(load <= .08, False, state.info["foot_in_contact"]))
            step_progress, takeoff_forward, landing_forward_record = foot_progress_records(
                forward_position=foot_xy @ self.forward,
                contact_before=state.info["foot_in_contact"], contact_now=contact_now,
                takeoff_forward=takeoff_forward, landing_record=landing_forward_record)
            landed, touched, foot_in_contact, qualified_swing = foot_landing_events(
                load=load, contact_before=state.info["foot_in_contact"],
                qualified_before=state.info["qualified_swing"], qualified_now=qualified,
                stable_support=stable_support, step_forward=step_progress)
        
        last_touchdown_xy = jnp.where(landed[:, None], foot_xy, state.info["last_touchdown_xy"])
        valid_steps = state.info["valid_steps"] + touched.astype(jnp.float32)
        new_max = jnp.where(landed, 0., new_max)
        qualified_swing = jnp.where(landed | (~stable_support), False, qualified_swing)

        standing_stable_seconds = state.info["standing_stable_seconds"]
        upright_reward = height_reward = load_reward = shift_reward = jnp.zeros(())
        swing_reward = forward_reward = touchdown_reward = jnp.zeros(())
        new_terms = None
        lift_credit = state.info["lift_credit"]
        forward_record = state.info["forward_record"]
        last_step_side = state.info["last_step_side"]
        alternating_steps = state.info["alternating_steps"]

        standing_terms = None
        standing_streak_seconds = jnp.zeros(())
        if self.task == "standing":
            horizontal_speed = jnp.linalg.norm(data.qvel[:2])
            angular_speed = jnp.linalg.norm(data.qvel[3:6])
            both_supported = jnp.all(load > .20)
            standing_stable = (both_supported & (upright > .99)
                & (root_height > self.standing_height - .03)
                & (horizontal_speed < .05) & (angular_speed < .12)
                & jnp.all(foot_speed < .025))
            standing_stable_seconds = standing_stable_seconds + (
                self.control_dt * standing_stable.astype(jnp.float32))
            com_xy = data.subtree_com[self.base_id, :2]
            com_offset = com_xy - jnp.mean(foot_xy, axis=0)
            target_error = (data.ctrl - data.qpos[self.act_qpos]) / self.span
            reward, standing_terms, standing_streak_seconds = standing_reward_terms(
                tilt=tilt, height_error=root_height - self.standing_height, load=load,
                com_offset=com_offset, horizontal_speed=horizontal_speed,
                angular_speed=angular_speed, foot_speed_sq=foot_speed_sq, action=action,
                target_error=target_error, fallen=fallen, stable=standing_stable,
                streak_seconds=state.info["standing_streak_seconds"], dt=self.control_dt,
                pose_error=jnp.mean((data.qpos[self.act_qpos] - self.initial_qpos[self.act_qpos]) ** 2),
                posture_weight=self.standing_posture_weight)
        elif self.task == "phase1_right_swing":
            task_phase, _, _, swing_target = phase1_schedule(step_count)
            swing_forward = (foot_xy[1] - state.info["start_foot_xy"][1]) @ self.forward
            support, airborne, new_lift, new_forward, new_backward, new_landing = phase1_outcomes(
                task_phase, left_load=load[0], right_load=load[1],
                right_clearance=clearance[1], right_forward=swing_forward,
                upright=upright, tilt=tilt, lifted_before=state.info["phase1_lifted"],
                forward_before=state.info["phase1_forward_hit"],
                backward_before=state.info["phase1_backward_hit"])
            new_landing = new_landing & ~state.info["phase1_landing_hit"]
            lift_seconds = state.info["phase1_lift_seconds"] + self.control_dt * airborne
            lift_quality = (jnp.clip(clearance[1] / .02, 0., 1.)
                            * jnp.clip(load[0] / .7, 0., 1.) * (load[1] < .05))
            swing_tracking = (jnp.exp(-((swing_forward - swing_target) / .02) ** 2)
                              * airborne * ((task_phase == 3) | (task_phase == 4)))
            safe = ~fallen
            positive = ((.20 * upright + .15 * jnp.clip(jnp.sum(load), 0., 1.)
                         + .20 * jnp.clip((load[0] - .5) / .25, 0., 1.)
                         + .45 * lift_quality + .30 * swing_tracking) * safe)
            events = (.30 * new_lift + .50 * new_forward + .50 * new_backward
                      + .50 * new_landing)
            reward = (positive + events - scrub_penalty - .002 * jnp.sum(action ** 2)
                      - 3. * fallen.astype(jnp.float32))
        elif self.reward_profile == "autonomous_walk_v1":
            action_delta = jnp.where(state.info["step_count"] > 0,
                                     action - state.info["last_action"], jnp.zeros_like(action))
            reward, new_terms, fallen = autonomous_walk_reward_terms(
                velocity=velocity, lateral_velocity=data.qvel[0], tilt=tilt,
                height_error=root_height - self.standing_height, load=load,
                foot_speed_sq=foot_speed_sq, action=action, action_delta=action_delta,
                fallen=fallen, dt=self.control_dt, target_vx=self.target_forward_speed)
            last_step_side, alternating = alternating_step_events(touched, last_step_side)
            alternating_steps = alternating_steps + alternating.astype(jnp.float32)
            forward_reward = new_terms["reward_forward"]
            scrub_penalty = new_terms["penalty_scrub"]
        elif self.reward_profile == "step_progress_v1":
            new_terms, lift_credit, forward_record, last_step_side = walking_reward_terms(
                tilt=tilt, height_error=root_height - self.standing_height,
                load=load, clearance=clearance, foot_speed=jnp.sqrt(foot_speed_sq),
                root_forward=data.qpos[:2] @ self.forward,
                forward_record=forward_record, lift_credit=lift_credit,
                last_step_side=last_step_side, touched=touched, fallen=fallen,
                dt=self.control_dt)
            reward = (sum(value for key, value in new_terms.items() if key.startswith("reward_"))
                      - sum(value for key, value in new_terms.items() if key.startswith("penalty_")))
            alternating_steps = alternating_steps + 2. * new_terms["reward_alternation"]
            forward_reward = new_terms["reward_forward"]
            touchdown_reward = new_terms["reward_touchdown"]
            scrub_penalty = new_terms["penalty_scrub"]
        elif self.reward_profile == "walking_transfer_v1":
            com_xy = data.subtree_com[self.base_id, :2]
            com_offset = com_xy - jnp.mean(foot_xy, axis=0)
            reward, new_terms, last_step_side, alternating = walking_transfer_reward_terms(
                stability_quality=stability_quality, com_offset=com_offset,
                forward_quality=forward_quality, shift_quality=shift_quality,
                swing_bonus=swing_bonus, scrub_penalty=scrub_penalty, touched=touched,
                last_step_side=last_step_side, fallen=fallen)
            alternating_steps = alternating_steps + 2. * alternating.astype(jnp.float32)
            forward_reward = new_terms["reward_forward"]
            touchdown_reward = new_terms["reward_touchdown"]
            scrub_penalty = new_terms["penalty_scrub"]
        elif self.reward_profile in ("walking_transfer_v2", "walking_transfer_v3"):
            com_xy = data.subtree_com[self.base_id, :2]
            com_offset = com_xy - jnp.mean(foot_xy, axis=0)
            torso_lateral_offset = data.xpos[self.base_id, :2] @ self.lateral
            reward, new_terms, last_step_side, alternating, fallen = walking_transfer_v2_reward_terms(
                velocity=velocity, com_offset=com_offset, shift_quality=shift_quality,
                scrub_penalty=scrub_penalty, touched=touched, last_step_side=last_step_side,
                height_error=root_height - self.standing_height,
                torso_lateral_offset=torso_lateral_offset, base_fallen=fallen,
                walking_first=self.reward_profile == "walking_transfer_v3")
            alternating_steps = alternating_steps + 2. * alternating.astype(jnp.float32)
            forward_reward = new_terms["reward_forward"]
            touchdown_reward = new_terms["reward_touchdown"]
            scrub_penalty = new_terms["penalty_scrub"]
        elif self.reward_profile in ("goal_corridor_v1", "goal_corridor_v2", "goal_corridor_v3"):
            exploration = self.reward_profile in ("goal_corridor_v2", "goal_corridor_v3")
            imitation = self.reward_profile == "goal_corridor_v3"
            if imitation:
                ref = self._residual_base(state.info, step_count, phase_new)
                track_error = jnp.mean((data.qpos[self.act_qpos] - ref) ** 2)
            quat = data.qpos[3:7]
            yaw = jnp.arctan2(2. * (quat[0] * quat[3] + quat[1] * quat[2]),
                              1. - 2. * (quat[2] ** 2 + quat[3] ** 2))
            yaw_error = jnp.arctan2(jnp.sin(yaw - self.start_yaw), jnp.cos(yaw - self.start_yaw))
            dot_obs = None
            dot_touched = None
            if self.footsteps:
                fs = self.footsteps
                heel_f, heel_l, toe_f = self._heel_toe(data)
                toe_mode = bool(fs.get("toe_target", 0.))
                dot_hit, dot_approach, dot_index, dot_prev, dot_side, anchor_f, anchor_l, dot_obs = footstep_target_terms(
                    heel_forward=toe_f if toe_mode else heel_f, foot_lateral=heel_l, landed=landed, alive=~fallen & (root_height - self.standing_height > -.10),
                    dot_index=state.info["dot_index"], prev_dist=state.info["dot_prev_dist"],
                    anchor_forward=state.info["dot_anchor_forward"], anchor_lateral=state.info["dot_anchor_lateral"],
                    start_side=state.info["dot_start_side"], step=fs.get("step", .03),
                    radius=fs.get("radius", .025), hit_reward=fs.get("hit_reward", 10.),
                    approach_weight=fs.get("approach_weight", 50.),
                    toe_forward=toe_f, first_behind_toe=fs.get("first_behind_toe"),
                    max_dots=None if fs.get("max_dots") is None else int(fs["max_dots"]),
                    # A hand-injected pose counts as step 1 on its first control step (user spec):
                    # credited to the foot that is furthest ahead of its standing spot.
                    force_side=jnp.where((state.info["injected"] > 0) & (state.info["step_count"] == 0),
                                         jnp.argmax(heel_f - state.info["start_forward"]
                                                    - jnp.asarray(self.standing_heel_forward, jnp.float32)), -1))
                # A dot hit IS the step: it alone drives the first-step bonus (paid after 1 s upright);
                # stray landings don't pay. Only "any foot touched" matters downstream.
                dot_touched = jnp.stack([dot_hit > 0, jnp.zeros((), bool)])
            reward, new_terms, goal_state, fallen = goal_corridor_reward_terms(
                forward=data.xpos[self.base_id, :2] @ self.forward - state.info["start_forward"],
                lateral=data.xpos[self.base_id, :2] @ self.lateral - state.info["goal_start_lateral"],
                yaw_error=yaw_error, height_error=root_height - self.standing_height,
                base_fallen=fallen, dt=self.control_dt, exploration_landing=exploration,
                touched=(dot_touched if dot_touched is not None else
                         ((landed & (step_progress >= .02)) if exploration else touched)),
                landing_length=(jnp.zeros(2) if dot_touched is not None else step_progress),
                fall_penalty=self.reward_sizes.get("fall_penalty", 200.) if imitation else 100.,
                **({k: v for k, v in self.reward_sizes.items() if k != "fall_penalty"} if imitation else {}),
                # v3: back wall 20 cm (10 cm ended almost every exploration episode as
                # the robot swayed back under action noise).
                back_wall=-.20 if imitation else -.10,
                track_error=track_error if imitation else None, track_weight=.5 if imitation else 0.,
                alive_bonus=self.alive_bonus if imitation else 0.,
                action_cost=jnp.mean(action ** 2) if imitation else None,
                **{k[5:]: state.info[k] for k in ("goal_best_forward", "goal_stall_anchor",
                   "goal_stall_seconds", "goal_first_step_timer", "goal_first_step_paid",
                   "goal_step_anchor")})
            if self.footsteps:
                reward = reward + dot_hit + dot_approach
                new_terms.update(reward_dot=dot_hit, reward_dot_approach=dot_approach,
                                 dots_reached=dot_index.astype(jnp.float32),
                                 dot_event_delta=(dot_index - state.info["dot_index"]).astype(jnp.float32))
                goal_state.update(dot_index=dot_index, dot_prev_dist=dot_prev, dot_start_side=dot_side,
                                  dot_anchor_forward=anchor_f, dot_anchor_lateral=anchor_l)
                if bool(fs.get("end_on_success", 0.)):
                    # Drill done: the step was made and held for 1 s (first-step bonus paid).
                    fallen = fallen | (new_terms["reward_first_step"] > 0)
                    new_terms["goal_reached"] = jnp.maximum(
                        new_terms["goal_reached"], (new_terms["reward_first_step"] > 0).astype(jnp.float32))
            forward_reward = new_terms["reward_progress"]
            touchdown_reward = (new_terms["reward_first_step"] + new_terms["reward_step"]
                                + new_terms["reward_landing"])
            last_step_side, alternating = alternating_step_events(touched, last_step_side)
            alternating_steps = alternating_steps + alternating.astype(jnp.float32)
        else:
            # Active walking branch (Dense, smooth, exploration-friendly)
            upright_reward = .15 * posture_quality * (~fallen)
            height_reward = .10 * height_quality * (~fallen)
            load_reward = .10 * jnp.clip(jnp.sum(load) / .60, 0., 1.) * (~fallen)
            shift_reward = .20 * shift_quality
            swing_reward = .25 * swing_bonus
            forward_reward = .40 * forward_quality
            touchdown_reward = .50 * jnp.sum(touched.astype(jnp.float32))

            reward = (upright_reward + height_reward + load_reward + shift_reward
                      + swing_reward + forward_reward + touchdown_reward
                      - scrub_penalty - 3.0 * fallen.astype(jnp.float32))

        action_rate_cost = jnp.where(
            state.info["step_count"] > 0,
            jnp.mean((action - state.info["last_action"]) ** 2),
            0.0,
        )
        action_rate_weight = .02 if self.task == "walking" else .25
        if new_terms is not None:
            # The existing 2 rad/s command slew remains; no extra action cost.
            action_rate_weight = 0.
        reward = jnp.nan_to_num(reward - action_rate_weight * action_rate_cost,
                                nan=-3., posinf=-3., neginf=-3.)

        # Brax info dict update
        info = dict(state.info)
        info.update(step_count=step_count, gait_phase=phase_new, stab_prev=stab_prev, stab_filt=stab_filt,
                    last_command=command, last_action=action, cmd_queue=cmd_queue,
                    filtered_fsr=filtered,
                    last_foot_xy=foot_xy, scrub=scrub, max_clearance=new_max,
                    valid_steps=valid_steps, prev_load=load,
                    qualified_swing=qualified_swing,
                    last_touchdown_xy=last_touchdown_xy,
                    foot_in_contact=foot_in_contact, lift_credit=lift_credit,
                    takeoff_forward=takeoff_forward,
                    landing_forward_record=landing_forward_record,
                    forward_record=forward_record, last_step_side=last_step_side,
                    alternating_steps=alternating_steps,
                    standing_stable_seconds=standing_stable_seconds)
        if self.reward_profile in ("goal_corridor_v1", "goal_corridor_v2", "goal_corridor_v3"):
            info.update(goal_state)
        if self.task == "phase1_right_swing":
            info.update(phase1_lift_seconds=lift_seconds,
                        phase1_lifted=state.info["phase1_lifted"] | new_lift,
                        phase1_forward_hit=state.info["phase1_forward_hit"] | new_forward,
                        phase1_backward_hit=state.info["phase1_backward_hit"] | new_backward,
                        phase1_landing_hit=state.info["phase1_landing_hit"] | new_landing)
        if self.task == "standing":
            info.update(standing_streak_seconds=standing_streak_seconds)

        metrics = dict(state.metrics)
        metrics.update(physical_fall=fallen.astype(jnp.float32),
                       wall_hit=jnp.zeros(()), stall_out=jnp.zeros(()),
                       goal_reached=jnp.zeros(()), dot_event_delta=jnp.zeros(()))
        if self.reward_profile in ("walking_transfer_v2", "walking_transfer_v3"):
            metrics.update(physical_fall=(new_terms["penalty_fall"] > 0).astype(jnp.float32),
                           wall_hit=(new_terms["penalty_corridor"] > 0).astype(jnp.float32))
        metrics.update(scrub=scrub, left_clearance=clearance[0], right_clearance=clearance[1],
                       left_steps=valid_steps[0], right_steps=valid_steps[1],
                       forward_distance=data.qpos[:2] @ self.forward - info["start_forward"],
                       shift_quality=shift_quality, supported_forward=supported_forward,
                       backward_drift_cost=backward_drift_cost,
                       stance_knee_sag_cost=stance_knee_sag_cost,
                       standing_stable_seconds=standing_stable_seconds)
        diagnostic = dict(
            single_support_seconds=single_support * self.control_dt,
            left_support_seconds=single_support * (loaded_side == 0) * self.control_dt,
            right_support_seconds=single_support * (loaded_side == 1) * self.control_dt,
            left_knee_rad_mean=data.qpos[self.knee_qpos[0]],
            right_knee_rad_mean=data.qpos[self.knee_qpos[1]],
            root_height_m_mean=root_height, body_tilt_rad_mean=tilt,
            left_load_bw_mean=load[0], right_load_bw_mean=load[1],
            left_clearance_m_peak=clearance[0], right_clearance_m_peak=clearance[1],
            tracking_error_rad_mean=jnp.mean(jnp.abs(ctrl - data.qpos[self.act_qpos])),
            target_clipped_fraction_mean=jnp.mean(
                (jnp.abs(base + self.action_scale * action - target) > 1e-6)
                .astype(jnp.float32)),
            body_tilt_rad_peak=tilt,
            root_drop_m_peak=jnp.maximum(self.standing_height - root_height, 0.),
            stance_knee_bend_rad_peak=stance_knee_bend,
            actuator_force_fraction_peak=jnp.max(jnp.abs(data.actuator_force) / self.force_limit),
            torque_saturation_fraction_mean=jnp.mean(
                (jnp.abs(data.actuator_force) / self.force_limit > .95).astype(jnp.float32)),
            control_slew_limited_fraction_mean=jnp.mean(
                (jnp.abs(target - ctrl) > 1e-6).astype(jnp.float32)),
            penalty_scrub=scrub_penalty, penalty_action_rate=action_rate_weight * action_rate_cost,
            penalty_action_magnitude=jnp.zeros(()),
            penalty_backward_drift=jnp.zeros(()),
            penalty_knee_sag=jnp.zeros(()),
            penalty_fall=3. * fallen.astype(jnp.float32),
            reward_upright=upright_reward, reward_height=height_reward,
            reward_load=load_reward, reward_shift=shift_reward,
            reward_swing=swing_reward, reward_forward=forward_reward,
            reward_touchdown=touchdown_reward,
            raw_scrub_cost=raw_scrub_cost, action_rate_cost=action_rate_cost,
            action_magnitude_cost=jnp.mean(action ** 2))
        metrics.update(diagnostic)
        if new_terms is not None:
            metrics.update(new_terms)
        if standing_terms is not None:
            metrics.update(standing_terms)
        if self.task == "phase1_right_swing":
            metrics.update(phase1_supported_lift_seconds=lift_seconds,
                           phase1_forward_hit=info["phase1_forward_hit"].astype(jnp.float32),
                           phase1_backward_hit=info["phase1_backward_hit"].astype(jnp.float32),
                           phase1_landing_hit=info["phase1_landing_hit"].astype(jnp.float32),
                           phase1_swing_forward_m=swing_forward)
        if self.footsteps and dot_obs is not None:
            obs = self._observation(data, command, filtered, step_count + state.info["phase_offset"], dot=dot_obs,
                                    phase=phase_new, nav=self._nav(data, state.info["goal_start_lateral"]))
        return State(data, obs, reward, fallen.astype(jnp.float32), metrics, info)
