"""Standalone numeric check of walking_transfer_v2_reward_terms (no MuJoCo/GPU).

Verifies: forward-velocity reward scales/caps correctly (75x weight, 0.3 m/s
cap -> 22.5/step max); each of the two hard violations (height-sag,
corridor) independently terminates the episode (fallen=True) and fires its
own penalty; falling zeroes every positive term; hands/arms are
structurally impossible to penalize since the function only ever receives
the torso's own lateral position, never any hand/arm position.

The third constraint this profile originally shipped with (a tilt-based
proxy for "nothing but the feet touches the ground") was removed after
being confirmed to over-trigger on real GPU hardware -- see env.py's
docstring and the README entry dated the same day. This check no longer
exercises it.

Run: python scripts/check_walking_transfer_v2_reward.py
"""
from pathlib import Path

import jax.numpy as jnp

_ENV_SRC = (Path(__file__).resolve().parents[1] / "envs/mjx_walking/env.py").read_text()
_START = _ENV_SRC.index("def walking_transfer_v2_reward_terms")
_END = _ENV_SRC.index("\n\nclass MJXWalkingEnv")
_ns = {"jnp": jnp}
exec(compile(_ENV_SRC[_START:_END], "env.py:walking_transfer_v2_reward_terms", "exec"), _ns)
walking_transfer_v2_reward_terms = _ns["walking_transfer_v2_reward_terms"]


def scenario(velocity=0., com_offset=(0., 0.), shift=0., scrub=0., touched=(False, False),
             last_side=-1, height_error=0., lateral=0., base_fallen=False, walking_first=False):
    reward, terms, next_side, alternating, fallen = walking_transfer_v2_reward_terms(
        velocity=jnp.asarray(velocity), com_offset=jnp.asarray(com_offset),
        shift_quality=jnp.asarray(shift), scrub_penalty=jnp.asarray(scrub),
        touched=jnp.asarray(touched), last_step_side=jnp.asarray(last_side),
        height_error=jnp.asarray(height_error),
        torso_lateral_offset=jnp.asarray(lateral), base_fallen=jnp.asarray(base_fallen),
        walking_first=walking_first)
    return (float(reward), {k: float(v) for k, v in terms.items()}, int(next_side),
            bool(alternating), bool(fallen))


def check(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    if not condition:
        raise SystemExit(1)


# --- forward reward scaling (reward_forward specifically, not the total
# reward, which also includes the small com_center/shift baseline terms) ---
_, terms_still, _, _, _ = scenario(velocity=0.)
_, terms_target, _, _, _ = scenario(velocity=.08)
_, terms_cap, _, _, _ = scenario(velocity=.30)
_, terms_above_cap, _, _, _ = scenario(velocity=1.0)
f_still, f_target, f_cap, f_above = (terms_still["reward_forward"], terms_target["reward_forward"],
                                     terms_cap["reward_forward"], terms_above_cap["reward_forward"])
print(f"velocity=0.00 reward_forward={f_still:.3f}")
print(f"velocity=0.08 reward_forward={f_target:.3f} (expect 75*.08=6.0)")
print(f"velocity=0.30 reward_forward={f_cap:.3f} (expect 75*.30=22.5, the cap)")
print(f"velocity=1.00 reward_forward={f_above:.3f} (expect same as cap -- no reward above .3 m/s)")
check("standing still gives zero forward reward", f_still == 0.0)
check("0.08 m/s gives exactly 6.0", abs(f_target - 6.0) < 1e-4)
check("0.30 m/s gives exactly 22.5 (the cap)", abs(f_cap - 22.5) < 1e-4)
check("1.0 m/s is clipped to the same reward as the cap", abs(f_above - f_cap) < 1e-4)

# --- height violation ---
_, terms_h, _, _, fallen_h = scenario(velocity=.2, height_error=-.11)
_, terms_h_ok, _, _, fallen_h_ok = scenario(velocity=.2, height_error=-.09)
print(f"height_error=-0.11 (violates) -> fallen={fallen_h}, penalty_fall={terms_h['penalty_fall']}")
print(f"height_error=-0.09 (ok)       -> fallen={fallen_h_ok}, penalty_fall={terms_h_ok['penalty_fall']}")
check("sagging 11cm (>10cm) triggers fallen + -100 penalty_fall",
      fallen_h and terms_h["penalty_fall"] == 100.0)
check("sagging only 9cm (<10cm) does not trigger fallen", not fallen_h_ok)

# --- corridor violation (separate -75 penalty, not the -100 base) ---
_, terms_corridor, _, _, fallen_corridor = scenario(velocity=.2, lateral=.14)
_, terms_corridor_ok, _, _, fallen_corridor_ok = scenario(velocity=.2, lateral=.13)
print(f"lateral=0.14m (past 0.135 half-width) -> fallen={fallen_corridor}, "
      f"penalty_corridor={terms_corridor['penalty_corridor']}, penalty_fall={terms_corridor['penalty_fall']}")
print(f"lateral=0.13m (within corridor)        -> fallen={fallen_corridor_ok}")
check("leaving the 27cm corridor triggers fallen + exactly -75 penalty_corridor",
      fallen_corridor and terms_corridor["penalty_corridor"] == 75.0)
check("corridor violation does NOT also charge the -100 base fall penalty",
      terms_corridor["penalty_fall"] == 0.0)
check("staying within the corridor does not trigger fallen", not fallen_corridor_ok)

# --- falling zeroes every positive term ---
_, terms_fallen, _, _, _ = scenario(velocity=.3, shift=.9, base_fallen=True)
check("base_fallen zeroes every positive reward term",
      all(v == 0. for k, v in terms_fallen.items() if k.startswith("reward_")))

# --- hands/arms structurally cannot be penalized ---
import inspect
sig = inspect.signature(walking_transfer_v2_reward_terms)
check("function signature has no hand/arm position input at all -- only torso_lateral_offset",
      "torso_lateral_offset" in sig.parameters
      and not any("hand" in p or "arm" in p for p in sig.parameters))

# walking_transfer_v3 (walking_first=True): walking must beat standing still.
stand3 = scenario(walking_first=True)[0]
walk3 = scenario(velocity=.10, walking_first=True)[0]
sag3 = scenario(velocity=.10, height_error=-.06, walking_first=True)[0]
drift3 = scenario(velocity=.10, lateral=.08, walking_first=True)[0]
step3 = scenario(touched=(True, False), last_side=1, walking_first=True)[0]
print(f"v3 per-step reward: stand={stand3:.2f} walk0.1={walk3:.2f} sag6cm={sag3:.2f} "
      f"drift8cm={drift3:.2f} alternating_step={step3:.2f}")
check("v3: standing still earns < 1/step (v2 paid ~16)", stand3 < 1.)
check("v3: walking at 0.1 m/s earns > 5x standing", walk3 > 5 * max(stand3, .1))
check("v3: sagging 6cm cuts the forward reward (smooth, before the -10cm cliff)", sag3 < .5 * walk3)
check("v3: drifting 8cm sideways cuts the forward reward", drift3 < .5 * walk3)
check("v3: an alternating valid step pays 15 (10 touchdown + 5 alternation)", abs(step3 - stand3 - 15.) < 1e-4)
check("v3: a fall still costs 100", scenario(base_fallen=True, walking_first=True)[0] <= -100.)

print("\nAll walking_transfer_v2_reward_terms checks passed.")
