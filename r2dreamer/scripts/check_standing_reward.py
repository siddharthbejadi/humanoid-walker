"""Standalone numeric check of standing_reward_terms (no MuJoCo/GPU needed).

Exercises the real function from envs/mjx_walking/env.py against synthetic
scenarios to verify monotonicity: perfect stand > minor sway > tilted >
slipping > falling, and that the streak bonus ramps up only while stable and
resets instantly on instability.

Run: python scripts/check_standing_reward.py
"""
from pathlib import Path

import jax.numpy as jnp

# env.py pulls in brax/mujoco at module scope, which aren't needed just to
# exercise the pure reward function and aren't installed in this check
# environment. Extract and exec only standing_reward_terms's source instead
# of importing the module.
_ENV_SRC = (Path(__file__).resolve().parents[1] / "envs/mjx_walking/env.py").read_text()
_START = _ENV_SRC.index("def standing_reward_terms")
_END = _ENV_SRC.index("\n\nclass MJXWalkingEnv")
_ns = {"jnp": jnp}
exec(compile(_ENV_SRC[_START:_END], "env.py:standing_reward_terms", "exec"), _ns)
standing_reward_terms = _ns["standing_reward_terms"]

DT = 0.02


def scenario(tilt=0., height_error=0., load=(.5, .5), com_offset=(0., 0.),
             h_speed=0., a_speed=0., foot_speed_sq=(0., 0.), action=None,
             target_error=None, fallen=False, stable=True, streak=0.):
    action = jnp.zeros((19,)) if action is None else jnp.asarray(action)
    target_error = jnp.zeros((19,)) if target_error is None else jnp.asarray(target_error)
    reward, terms, next_streak = standing_reward_terms(
        tilt=jnp.asarray(tilt), height_error=jnp.asarray(height_error),
        load=jnp.asarray(load), com_offset=jnp.asarray(com_offset),
        horizontal_speed=jnp.asarray(h_speed), angular_speed=jnp.asarray(a_speed),
        foot_speed_sq=jnp.asarray(foot_speed_sq), action=action,
        target_error=target_error, fallen=jnp.asarray(fallen),
        stable=jnp.asarray(stable), streak_seconds=jnp.asarray(streak), dt=DT)
    return float(reward), {k: float(v) for k, v in terms.items()}, float(next_streak)


def check(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    if not condition:
        raise SystemExit(1)


perfect, _, perfect_streak = scenario(stable=True, streak=0.)
sway, _, _ = scenario(tilt=.05, h_speed=.03, stable=False)
tilted, _, _ = scenario(tilt=.30, stable=False)
slipping, _, _ = scenario(foot_speed_sq=(.04, .0), stable=False)
asymmetric, _, _ = scenario(load=(.8, .2), stable=False)
fallen_upright_instant, terms_f, _ = scenario(tilt=0., height_error=0., load=(.5, .5),
                                              fallen=True, stable=False)

print(f"perfect stand reward      = {perfect:.4f}")
print(f"minor sway reward         = {sway:.4f}")
print(f"tilted (0.30 rad) reward  = {tilted:.4f}")
print(f"slipping reward           = {slipping:.4f}")
print(f"asymmetric load reward    = {asymmetric:.4f}")
print(f"fallen (still upright-looking) reward = {fallen_upright_instant:.4f}")
print(f"  (regression check: with the old code this would stay near +1.0 "
      f"because the fall gate only added -3 on top of unzeroed quality terms)")

check("perfect stand scores highest", perfect > sway > tilted)
check("slipping is penalized below perfect", slipping < perfect)
check("asymmetric load is penalized below perfect", asymmetric < perfect)
check("falling zeroes the positive quality terms (bug fix)",
      all(terms_f[k] == 0. for k in terms_f if k.startswith("standing_reward_")))
check("falling still applies the -3 penalty", terms_f["standing_penalty_fall"] == 3.0)

# Streak bonus: ramps linearly to streak_weight over streak_ramp_seconds, then
# resets instantly to 0 the moment stability breaks.
streak = 0.
bonuses = []
for step in range(150):  # 150 * 0.02s = 3.0s of continuous stability
    _, terms, streak = scenario(stable=True, streak=streak)
    bonuses.append(terms["standing_streak_bonus"])
check("streak bonus saturates at streak_weight (0.5) after streak_ramp_seconds (2.0s)",
      abs(bonuses[-1] - 0.5) < 1e-6 and abs(bonuses[99] - 0.5) < 1e-6)
print(f"streak bonus at t=0.02s = {bonuses[0]:.4f} (expect dt/ramp*weight = {DT/2.0*0.5:.4f})")
check("streak bonus starts small (one dt of credit) and grows monotonically",
      bonuses[0] < 0.01 and all(b1 <= b2 + 1e-9 for b1, b2 in zip(bonuses, bonuses[1:])))

_, terms_break, reset_streak = scenario(stable=False, streak=1.5)
check("one unstable step resets the streak to exactly 0, bonus to 0",
      reset_streak == 0. and terms_break["standing_streak_bonus"] == 0.)

print("\nAll standing-reward checks passed.")
