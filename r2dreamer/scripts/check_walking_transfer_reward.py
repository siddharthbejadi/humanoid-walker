"""Standalone numeric check of walking_transfer_reward_terms (no MuJoCo/GPU).

Exercises the real function from envs/mjx_walking/env.py against synthetic
quality-term scenarios to verify: a real walking step scores higher than
standing still (no reward for stillness, unlike standing_reward_terms);
alternating steps score higher than repeated same-side steps; falling
zeroes the positive terms and applies the fixed penalty regardless of how
good the other quality terms look.

Run: python scripts/check_walking_transfer_reward.py
"""
from pathlib import Path

import jax.numpy as jnp

# env.py pulls in brax/mujoco at module scope, not needed just to exercise
# the pure reward function and not installed in this check environment.
_ENV_SRC = (Path(__file__).resolve().parents[1] / "envs/mjx_walking/env.py").read_text()
_START = _ENV_SRC.index("def walking_transfer_reward_terms")
_END = _ENV_SRC.index("\n\nclass MJXWalkingEnv")
_ns = {"jnp": jnp}
exec(compile(_ENV_SRC[_START:_END], "env.py:walking_transfer_reward_terms", "exec"), _ns)
walking_transfer_reward_terms = _ns["walking_transfer_reward_terms"]


def scenario(stability=.9, com_offset=(0., 0.), forward=0., shift=0., swing=0.,
             scrub=0., touched=(False, False), last_side=-1, fallen=False):
    reward, terms, next_side, alternating = walking_transfer_reward_terms(
        stability_quality=jnp.asarray(stability), com_offset=jnp.asarray(com_offset),
        forward_quality=jnp.asarray(forward), shift_quality=jnp.asarray(shift),
        swing_bonus=jnp.asarray(swing), scrub_penalty=jnp.asarray(scrub),
        touched=jnp.asarray(touched), last_step_side=jnp.asarray(last_side),
        fallen=jnp.asarray(fallen))
    return float(reward), {k: float(v) for k, v in terms.items()}, int(next_side), bool(alternating)


def check(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    if not condition:
        raise SystemExit(1)


standing_still, _, _, _ = scenario(stability=.9, forward=0., shift=0., swing=0., touched=(False, False))
walking_step, step_terms, next_side_a, alt_a = scenario(
    stability=.9, forward=.9, shift=.8, swing=.8, touched=(True, False), last_side=-1)
alternating_step, _, next_side_b, alt_b = scenario(
    stability=.9, forward=.9, shift=.8, swing=.8, touched=(False, True), last_side=next_side_a)
same_side_repeat, _, _, alt_c = scenario(
    stability=.9, forward=.9, shift=.8, swing=.8, touched=(True, False), last_side=next_side_a)
fallen_reward, fallen_terms, _, _ = scenario(
    stability=.9, forward=.9, shift=.8, swing=.8, touched=(True, False), last_side=-1, fallen=True)
slipping, _, _, _ = scenario(stability=.9, forward=.5, scrub=.3, touched=(False, False))

print(f"standing still (no steps) reward = {standing_still:.4f}")
print(f"walking step (first touchdown)   = {walking_step:.4f}")
print(f"alternating second step          = {alternating_step:.4f}")
print(f"same-side repeated touchdown     = {same_side_repeat:.4f}")
print(f"fallen (otherwise-good quality)  = {fallen_reward:.4f}")
print(f"slipping (high scrub penalty)    = {slipping:.4f}")

check("a real walking step scores higher than standing still (no stillness reward)",
      walking_step > standing_still)
check("alternating to the other foot scores higher than a repeated same-side touchdown",
      alternating_step > same_side_repeat)
check("alternation bonus actually fires on the second, opposite-side step",
      alt_b and not alt_c)
check("falling zeroes every positive reward term (including touchdown -- event requires ~fallen)",
      all(v == 0. for k, v in fallen_terms.items() if k.startswith("reward_")))
check("falling still pays the -3 penalty", fallen_terms["penalty_fall"] == 3.0)
check("falling's net reward is strongly negative despite good inputs", fallen_reward < -2.5)
check("slip penalty measurably reduces reward vs a clean equivalent",
      slipping < scenario(stability=.9, forward=.5, scrub=0., touched=(False, False))[0])

print("\nAll walking_transfer_reward_terms checks passed.")
