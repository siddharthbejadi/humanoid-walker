"""Common-sense justification test for goal_corridor_v1 (no MuJoCo/GPU).

Replays scripted 400 s episodes through env.py's goal_corridor_reward_terms
(the exact training code, run with numpy) and checks the episode totals rank
the way common sense says they must. If any ordering fails, the reward pays
the robot for the wrong thing.

Run: python scripts/check_goal_corridor_reward.py
"""
import math
from pathlib import Path

import numpy as np

src = (Path(__file__).resolve().parent.parent / "envs/mjx_walking/env.py").read_text()
start, end = src.index("GOAL_METRIC_KEYS = "), src.index("class MJXWalkingEnv")
ns = {"jnp": np}
exec(compile(src[start:end], "env.py:goal_corridor_reward_terms", "exec"), ns)
reward_fn = ns["goal_corridor_reward_terms"]

DT, HORIZON_S = .02, 400.
STEPS = int(HORIZON_S / DT)


def episode(path, *, stride=.08, exploration=False, fall_penalty=100., track_err=None, track_weight=0., back_wall=-.10, alive_bonus=0., sizes=None):
    """path(t) -> (forward m, lateral m, yaw rad, height_error m, fallen, stepping)."""
    state = dict(goal_best_forward=np.float64(0.), goal_stall_anchor=np.float64(0.),
                 goal_stall_seconds=np.float64(0.), goal_first_step_timer=np.float64(-1.),
                 goal_first_step_paid=np.float64(0.), goal_step_anchor=np.float64(0.))
    total, totals, last_touch_forward, side = 0., {}, 0., 0
    for i in range(STEPS):
        t = (i + 1) * DT
        forward, lateral, yaw, height_error, fallen, stepping = path(t)
        touched = np.zeros(2, dtype=bool)
        # A valid step = the foot lands >= 1.5 cm ahead of its last landing (env.py rule).
        if stepping and forward - last_touch_forward >= max(stride, .015):
            touched[side] = True
            side, last_touch_forward = 1 - side, forward
        reward, terms, state, done = reward_fn(
            forward=np.float64(forward), lateral=np.float64(lateral), yaw_error=np.float64(yaw),
            height_error=np.float64(height_error), base_fallen=np.bool_(fallen), touched=touched,
            dt=DT, exploration_landing=exploration, landing_length=np.full(2, stride),
            fall_penalty=fall_penalty, track_weight=track_weight, back_wall=back_wall, alive_bonus=alive_bonus, **(sizes or {}),
            track_error=None if track_err is None else np.float64(track_err),
            **{k[5:]: v for k, v in state.items()})
        total += float(reward)
        for k, v in terms.items():
            totals[k] = totals.get(k, 0.) + float(v)
        if done:
            return total, t, totals
    return total, HORIZON_S, totals


def walk(speed, *, lateral_drift=0., height=0., yaw=0., fall_at=None, stop_at=None, stride=.08):
    def path(t):
        moving_t = t if stop_at is None else min(t, stop_at)
        return (speed * moving_t, lateral_drift * t, yaw, height,
                fall_at is not None and t >= fall_at, stop_at is None or t < stop_at)
    return path


S = {
    "reach 100 m at 0.25 m/s":          walk(.25),
    "walk 0.10 m/s all episode (40 m)": walk(.10),
    "walk 10 m then fall":              walk(.10, fall_at=100.),
    "crouch-walk 0.10 m/s (6 cm sag)":  walk(.10, height=-.06),
    "walk 0.10 m/s turned 15 deg":      walk(.10, yaw=math.radians(15)),
    "walk 0.10 m/s drifting sideways":  walk(.10, lateral_drift=.002),
    "one step then stand":              walk(.10, stop_at=.9, stride=.05),
    "stand still":                      lambda t: (0., 0., 0., 0., False, False),
    "rock back and forth 2 cm":         lambda t: (.02 * math.sin(t), 0., 0., 0., False, False),
    "step in place (no new ground)":    lambda t: (.016 * (int(t / .5) % 2), 0., 0., 0., False, True),
    "lunge one step then fall":         walk(.30, fall_at=.8, stride=.05),
    "fall immediately":                 lambda t: (0., 0., 0., 0., True, False),
    "walk backwards into back wall":    walk(-.05),
}
R = {}
print(f"{'scenario':36s} {'return':>10s} {'ends at':>8s}  biggest terms")
for name, path in S.items():
    total, t_end, terms = episode(path, stride=.05 if "one step" in name or "lunge" in name else .08)
    R[name] = total
    big = sorted(((k, v) for k, v in terms.items() if abs(v) > .5 and k != "penalty_height_violation"),
                 key=lambda kv: -abs(kv[1]))[:3]
    print(f"{name:36s} {total:10.1f} {t_end:7.1f}s  " + ", ".join(f"{k}={v:.0f}" for k, v in big))

checks = [
    ("reaching the goal beats everything", all(R["reach 100 m at 0.25 m/s"] > v for k, v in R.items() if k != "reach 100 m at 0.25 m/s")),
    ("walking all episode > walking then falling", R["walk 0.10 m/s all episode (40 m)"] > R["walk 10 m then fall"]),
    ("walking 10 m then falling > one step then stand", R["walk 10 m then fall"] > R["one step then stand"]),
    ("straight walking > crouch-walking", R["walk 0.10 m/s all episode (40 m)"] > R["crouch-walk 0.10 m/s (6 cm sag)"]),
    ("straight walking > walking turned 15 deg", R["walk 0.10 m/s all episode (40 m)"] > R["walk 0.10 m/s turned 15 deg"]),
    ("straight walking > drifting sideways into the wall", R["walk 0.10 m/s all episode (40 m)"] > R["walk 0.10 m/s drifting sideways"]),
    ("one step then stand > stand still (first step +1000)", R["one step then stand"] > R["stand still"] + 900),
    ("stand still is penalised (< 0)", R["stand still"] < 0),
    ("standing still > falling immediately (falling is never the escape)", R["stand still"] > R["fall immediately"]),
    ("lunge-and-fall < stand still (no +1000 for throwing itself forward)", R["lunge one step then fall"] < R["stand still"]),
    ("rocking earns no more than standing + 5", R["rock back and forth 2 cm"] <= R["stand still"] + 5),
    ("stepping in place earns no more than one real step", R["step in place (no new ground)"] <= R["one step then stand"]),
    ("walking backwards < standing still", R["walk backwards into back wall"] < R["stand still"]),
]
failed = 0
for name, ok in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    failed += not ok

# goal_corridor_v2: any forward landing pays +50 at once, even if the robot then falls.
V2 = {
    "reach 100 m at 0.25 m/s":          walk(.25),
    "walk 0.10 m/s all episode (40 m)": walk(.10),
    "shuffle 3 cm landings at 0.10 m/s": walk(.10, stride=.03),
    "one step then stand":              walk(.10, stop_at=.9, stride=.05),
    "land one step then fall":          walk(.30, fall_at=.8, stride=.2),
    "stand still":                      S["stand still"],
    "step in place (no new ground)":    S["step in place (no new ground)"],
    "fall immediately (no landing)":    S["fall immediately"],
}
R2 = {}
print(f"\n--- goal_corridor_v2 (exploration landing) ---\n{'scenario':36s} {'return':>10s} {'ends at':>8s}  biggest terms")
for name, path in V2.items():
    stride = .03 if "shuffle" in name else .2 if "then fall" in name else .05 if "one step" in name else .08
    total, t_end, terms = episode(path, stride=stride, exploration=True)
    R2[name] = total
    big = sorted(((k, v) for k, v in terms.items() if abs(v) > .5 and k != "penalty_height_violation"),
                 key=lambda kv: -abs(kv[1]))[:3]
    print(f"{name:36s} {total:10.1f} {t_end:7.1f}s  " + ", ".join(f"{k}={v:.0f}" for k, v in big))
checks2 = [
    ("v2: reaching the goal beats everything", all(R2["reach 100 m at 0.25 m/s"] > v for k, v in R2.items() if k != "reach 100 m at 0.25 m/s")),
    ("v2: walking all episode > one step then stand", R2["walk 0.10 m/s all episode (40 m)"] > R2["one step then stand"]),
    ("v2: one step + stay up >> land a step then fall", R2["one step then stand"] > R2["land one step then fall"] + 900),
    ("v2: landing a step then falling > standing still (stepping is discoverable)", R2["land one step then fall"] > R2["stand still"]),
    ("v2: standing still > falling without landing", R2["stand still"] > R2["fall immediately (no landing)"]),
    ("v2: stepping in place earns nothing extra over standing", R2["step in place (no new ground)"] <= R2["stand still"] + 5),
    ("v2: 3 cm shuffling earns no more than full strides at the same speed",
     R2["shuffle 3 cm landings at 0.10 m/s"] <= R2["walk 0.10 m/s all episode (40 m)"] + 1),
]
for name, ok in checks2:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    failed += not ok
print(f"shuffle vs stride at 0.10 m/s: {R2['shuffle 3 cm landings at 0.10 m/s']:.0f} vs {R2['walk 0.10 m/s all episode (40 m)']:.0f}"
      " (landing pay is per centimetre, not per tap)")

# goal_corridor_v3: v2 + fall/wall -200 + following the vendor gait (+0.5/step at zero error).
FOLLOW, IGNORE = 0., .30 ** 2   # mean squared joint error (rad^2): on the gait vs ~17 deg off it
V3 = [  # name, path, stride, track error
    ("reach 100 m following the gait",   walk(.25), .08, FOLLOW),
    ("walk 0.10 m/s following the gait", walk(.10), .08, FOLLOW),
    ("walk 0.10 m/s ignoring the gait",  walk(.10), .08, IGNORE),
    ("one step then stand",              walk(.10, stop_at=.9, stride=.05), .05, .10 ** 2),
    ("stand still",                      S["stand still"], .08, .10 ** 2),
    ("land one step then fall (dive)",   walk(.30, fall_at=.8, stride=.2), .2, IGNORE),
    ("fall immediately",                 S["fall immediately"], .08, IGNORE),
    ("walk into the side wall",          walk(.10, lateral_drift=.002), .08, FOLLOW),
    ("walk backwards into the back wall", walk(-.05), .08, IGNORE),
]
R3 = {}
print(f"\n--- goal_corridor_v3 (fall -200 + follow vendor gait) ---\n{'scenario':36s} {'return':>10s} {'ends at':>8s}  biggest terms")
for name, path, stride, err in V3:
    total, t_end, terms = episode(path, stride=stride, exploration=True, fall_penalty=200., track_err=err, track_weight=.5,
                                  back_wall=-.20)
    R3[name] = total
    big = sorted(((k, v) for k, v in terms.items() if abs(v) > .5 and k != "penalty_height_violation"),
                 key=lambda kv: -abs(kv[1]))[:3]
    print(f"{name:36s} {total:10.1f} {t_end:7.1f}s  " + ", ".join(f"{k}={v:.0f}" for k, v in big))
checks3 = [
    ("v3: reaching the goal beats everything", all(R3["reach 100 m following the gait"] > v for k, v in R3.items() if k != "reach 100 m following the gait")),
    ("v3: following the gait beats walking the same speed without it",
     R3["walk 0.10 m/s following the gait"] > R3["walk 0.10 m/s ignoring the gait"]),
    ("v3: one step + stay up >> diving", R3["one step then stand"] > R3["land one step then fall (dive)"] + 900),
    ("v3: standing still > diving forward (the v2 dive is no longer worth it)", R3["stand still"] > R3["land one step then fall (dive)"]),
    ("v3: standing still > falling", R3["stand still"] > R3["fall immediately"]),
    ("v3: walking straight > drifting into the wall", R3["walk 0.10 m/s following the gait"] > R3["walk into the side wall"]),
    ("v3: walking backwards (20 cm back wall) < standing still", R3["walk backwards into the back wall"] < R3["stand still"]),
]
for name, ok in checks3:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    failed += not ok

# goal_corridor_v3 + alive bonus 0.5/step (what the residual walk stage trains with).
def sway_then_fall(fall_t):  # wobbles in place (5 cm sideways, 0.15 rad heading) until it falls
    return lambda t: (0., .05 * math.sin(3 * t), .15 * math.sin(2 * t), 0., t >= fall_t, False)
R4 = {}
print(f"\n--- goal_corridor_v3 + alive bonus 0.5 ---\n{'scenario':36s} {'return':>10s} {'ends at':>8s}  biggest terms")
for name, path, stride, err in V3 + [("sway then fall at 0.5 s", sway_then_fall(.5), .08, IGNORE),
                                      ("sway then fall at 3.0 s", sway_then_fall(3.), .08, IGNORE)]:
    total, t_end, terms = episode(path, stride=stride, exploration=True, fall_penalty=200., track_err=err,
                                  track_weight=.5, back_wall=-.20, alive_bonus=.5)
    R4[name] = total
    big = sorted(((k, v) for k, v in terms.items() if abs(v) > .5 and k != "penalty_height_violation"),
                 key=lambda kv: -abs(kv[1]))[:3]
    print(f"{name:36s} {total:10.1f} {t_end:7.1f}s  " + ", ".join(f"{k}={v:.0f}" for k, v in big))
checks4 = [
    ("v3+alive: surviving longer beats falling sooner (no 'die early' optimum)",
     R4["sway then fall at 3.0 s"] > R4["sway then fall at 0.5 s"]),
    ("v3+alive: reaching the goal beats everything", all(R4["reach 100 m following the gait"] > v for k, v in R4.items() if k != "reach 100 m following the gait")),
    ("v3+alive: walking all episode >> standing still (stall-out caps the bonus)",
     R4["walk 0.10 m/s following the gait"] > 10 * R4["stand still"]),
    ("v3+alive: standing still > diving", R4["stand still"] > R4["land one step then fall (dive)"]),
    ("v3+alive: standing still > falling", R4["stand still"] > R4["fall immediately"]),
    ("v3+alive: walking backwards < standing still", R4["walk backwards into the back wall"] < R4["stand still"]),
    ("v3+alive: following the gait > ignoring it", R4["walk 0.10 m/s following the gait"] > R4["walk 0.10 m/s ignoring the gait"]),
]
for name, ok in checks4:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    failed += not ok
print(f"(without the bonus, v3: sway-and-fall at 0.5 s vs 3.0 s = {episode(sway_then_fall(.5), exploration=True, fall_penalty=200., track_err=IGNORE, track_weight=.5, back_wall=-.20)[0]:.1f} vs "
      f"{episode(sway_then_fall(3.), exploration=True, fall_penalty=200., track_err=IGNORE, track_weight=.5, back_wall=-.20)[0]:.1f})")

# What the residual walk stage trains with: alive 0.5 + smaller spikes (Dreamer reward head).
SIZES = dict(landing_weight=10., first_step_bonus=20., goal_bonus=500., stall_out_penalty=10.)
R5 = {}
print(f"\n--- goal_corridor_v3 + alive 0.5 + small spikes (fall -50, landing 10, first step 20, goal 500, stall-out 10) ---")
for name, path, stride, err in V3 + [("sway then fall at 0.5 s", sway_then_fall(.5), .08, IGNORE),
                                      ("sway then fall at 3.0 s", sway_then_fall(3.), .08, IGNORE)]:
    total, t_end, terms = episode(path, stride=stride, exploration=True, fall_penalty=50., track_err=err,
                                  track_weight=.5, back_wall=-.20, alive_bonus=.5, sizes=SIZES)
    R5[name] = total
    big = sorted(((k, v) for k, v in terms.items() if abs(v) > .5 and k != "penalty_height_violation"),
                 key=lambda kv: -abs(kv[1]))[:3]
    print(f"{name:36s} {total:10.1f} {t_end:7.1f}s  " + ", ".join(f"{k}={v:.0f}" for k, v in big))
checks5 = [
    ("small spikes: surviving longer beats falling sooner", R5["sway then fall at 3.0 s"] > R5["sway then fall at 0.5 s"]),
    ("small spikes: reaching the goal beats everything", all(R5["reach 100 m following the gait"] > v for k, v in R5.items() if k != "reach 100 m following the gait")),
    ("small spikes: walking all episode >> standing still", R5["walk 0.10 m/s following the gait"] > 10 * R5["stand still"]),
    ("small spikes: one step then stand > stand still", R5["one step then stand"] > R5["stand still"]),
    ("small spikes: standing still > diving", R5["stand still"] > R5["land one step then fall (dive)"]),
    ("small spikes: diving (lands a step) > falling without landing", R5["land one step then fall (dive)"] > R5["fall immediately"]),
    ("small spikes: walking backwards < standing still", R5["walk backwards into the back wall"] < R5["stand still"]),
    ("small spikes: walking straight > drifting into the wall", R5["walk 0.10 m/s following the gait"] > R5["walk into the side wall"]),
    ("small spikes: following the gait > ignoring it", R5["walk 0.10 m/s following the gait"] > R5["walk 0.10 m/s ignoring the gait"]),
    ("small spikes: largest single reward <= 25x the per-step alive income... except goal (terminal, rare)",
     max(SIZES["landing_weight"], SIZES["first_step_bonus"], 50.) <= 25 * .5 * 4),
]
for name, ok in checks5:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    failed += not ok

# Half-speed gait: stall-out after 12 s (start-up is slow); the -5/s after 1 s rule is unchanged.
SIZES12 = dict(SIZES, stall_out_seconds=12.)
R6 = {}
print("\n--- same + stall-out at 12 s (half-speed gait) ---")
for name, path, stride, err in V3 + [("sway then fall at 0.5 s", sway_then_fall(.5), .08, IGNORE),
                                      ("sway then fall at 3.0 s", sway_then_fall(3.), .08, IGNORE)]:
    total, t_end, terms = episode(path, stride=stride, exploration=True, fall_penalty=50., track_err=err,
                                  track_weight=.5, back_wall=-.20, alive_bonus=.5, sizes=SIZES12)
    R6[name] = total
    print(f"{name:36s} {total:10.1f} {t_end:7.1f}s")
checks6 = [
    ("12 s stall-out: surviving longer beats falling sooner", R6["sway then fall at 3.0 s"] > R6["sway then fall at 0.5 s"]),
    ("12 s stall-out: walking all episode >> standing still", R6["walk 0.10 m/s following the gait"] > 10 * R6["stand still"]),
    ("12 s stall-out: standing still > diving > falling", R6["stand still"] > R6["land one step then fall (dive)"] > R6["fall immediately"]),
    ("12 s stall-out: walking backwards < standing still", R6["walk backwards into the back wall"] < R6["stand still"]),
    ("12 s stall-out: reaching the goal beats everything", all(R6["reach 100 m following the gait"] > v for k, v in R6.items() if k != "reach 100 m following the gait")),
]
for name, ok in checks6:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    failed += not ok

# User spec (2026-10-09): standing in one place pays 0; after 2.5 s still it costs -5/s; ends at 8 s.
SIZES_STILL = dict(SIZES, stall_out_seconds=8., stall_grace_seconds=2.5, still_pays=0.)
R8 = {}
print("\n--- same + standing pays 0, -5/s after 2.5 s still, stall-out at 8 s (current walk stage) ---")
for name, path, stride, err in V3 + [("sway then fall at 0.5 s", sway_then_fall(.5), .08, IGNORE),
                                      ("sway then fall at 3.0 s", sway_then_fall(3.), .08, IGNORE)]:
    total, t_end, terms = episode(path, stride=stride, exploration=True, fall_penalty=50., track_err=err,
                                  track_weight=.5, back_wall=-.20, alive_bonus=.5, sizes=SIZES_STILL)
    R8[name] = total
    print(f"{name:36s} {total:10.1f} {t_end:7.1f}s")
first_still = episode(V3[[n for n, *_ in V3].index("stand still")][1], exploration=True, fall_penalty=50.,
                      track_err=0., track_weight=.5, back_wall=-.20, alive_bonus=.5, sizes=dict(SIZES_STILL, stall_out_seconds=2.4, stall_out_penalty=0.))[0]
checks8 = [
    ("still: standing still on a perfect gait pose earns ~0 for the first 2.4 s", abs(first_still) < 1.),
    ("still: standing still ends negative (-5/s after 2.5 s)", R8["stand still"] < -20),
    ("still: standing still > falling", R8["stand still"] > R8["fall immediately"]),
    ("still: walking all episode >> standing still", R8["walk 0.10 m/s following the gait"] > 10 * abs(R8["stand still"])),
    ("still: one step then stand > stand still", R8["one step then stand"] > R8["stand still"]),
    ("still: walking backwards < standing still", R8["walk backwards into the back wall"] < R8["stand still"]),
    ("still: reaching the goal beats everything", all(R8["reach 100 m following the gait"] > v for k, v in R8.items() if k != "reach 100 m following the gait")),
]
for name, ok in checks8:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    failed += not ok

# Footstep dots: replay heel movements through env.py's footstep_target_terms.
# Rule: the stepping foot's HEEL must land 3 cm ahead of where that same heel last landed.
fs_start, fs_end = src.index("def footstep_target_terms"), src.index("class MJXWalkingEnv")
exec(compile(src[fs_start:fs_end], "env.py:footstep_target_terms", "exec"), ns)
dots = ns["footstep_target_terms"]
HEEL0, LAT = np.array([-.02, -.02]), np.array([.05, -.05])   # left, right heels at reset

def dot_episode(track, first_side=1):
    """track: per control step (heel_left, heel_right, landed_left, landed_right, alive)."""
    idx, side0 = np.int32(0), np.int32(first_side)
    af, al, prev, total, hits = HEEL0.copy(), LAT.copy(), np.float64(.03), 0., 0
    for hl, hr, la, ra, alive in track:
        hit, appr, idx, prev, side0, af, al, _ = dots(
            heel_forward=np.array([hl, hr]), foot_lateral=LAT, landed=np.array([la, ra]), alive=np.bool_(alive),
            dot_index=idx, prev_dist=np.float64(prev), anchor_forward=af, anchor_lateral=al, start_side=side0)
        total += float(hit) + float(appr); hits += int(hit > 0)
    return total, hits

def steps(seq, fall_last=False):
    """seq: list of (foot, distance moved forward) per step; the foot swings over 10 control steps then lands."""
    heel, track = HEEL0.copy(), []
    for n, (foot, move) in enumerate(seq):
        start = heel[foot]
        for j in range(1, 11):
            heel[foot] = start + move * j / 10
            landed = j == 10
            track.append((heel[0], heel[1], landed and foot == 0, landed and foot == 1,
                          not (fall_last and landed and n == len(seq) - 1)))
    return track

zig = lambda n, first=1, move=.03: [((first if k % 2 == 0 else 1 - first), move) for k in range(n)]
def wiggle(n):
    track = []
    for _ in range(n):
        for j in list(range(1, 11)) + list(range(9, -1, -1)):
            track.append((HEEL0[0], HEEL0[1] + .025 * j / 10, False, False, True))
    return track

D = {"zig-zag 10 steps, right first, 3 cm each": dot_episode(steps(zig(10))),
     "march in place (heels land where they were)": dot_episode(steps(zig(10, move=0.))),
     "lunge 6 cm each step (3 cm past the dot)": dot_episode(steps(zig(10, move=.06))),
     "short 1.5 cm steps (within tolerance)": dot_episode(steps(zig(10, move=.015))),
     "right foot twice in a row": dot_episode(steps([(1, .03), (1, .03), (1, .03)])),
     "only a left step (right-first mode)": dot_episode(steps([(0, .03)])),
     "left foot first (either-foot mode)": dot_episode(steps(zig(10, first=0)), first_side=-1),
     "fall on the first landing": dot_episode(steps([(1, .03)], fall_last=True)),
     "swing heel toward the dot and back, never land": dot_episode(wiggle(5))}
print("\n--- footstep dots: heel lands 3 cm ahead of its own last spot (radius 2.5 cm, +10 per hit) ---")
for k, (tot, hits) in D.items():
    print(f"{k:50s} reward {tot:7.2f}  dots hit {hits}")
checks7 = [
    ("dots: zig-zag of 3 cm steps hits every dot", D["zig-zag 10 steps, right first, 3 cm each"][1] == 10),
    ("dots: zig-zag walking >> marching in place", D["zig-zag 10 steps, right first, 3 cm each"][0] > 50 + D["march in place (heels land where they were)"][0]),
    ("dots: marching in place hits nothing", D["march in place (heels land where they were)"][1] == 0),
    ("dots: 6 cm lunges (3 cm past the dot) hit nothing", D["lunge 6 cm each step (3 cm past the dot)"][1] == 0),
    ("dots: same foot twice only scores the first", D["right foot twice in a row"][1] == 1),
    ("dots: right-first mode, a left step first scores nothing", D["only a left step (right-first mode)"][1] == 0),
    ("dots: either-foot mode, starting left hits every dot", D["left foot first (either-foot mode)"][1] == 10),
    ("dots: falling on the landing earns no hit", D["fall on the first landing"][1] == 0),
    ("dots: swinging toward the dot and back nets ~0", abs(D["swing heel toward the dot and back, never land"][0]) < .01),
]
for name, ok in checks7:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    failed += not ok

# First step (user spec 2026-10-09 evening): the right heel must land no more than 5 cm behind the LEFT
# toe (standing: 12.3 cm behind), then the usual +3 cm per-foot dots. Foot pads are 12.3 cm long.
FOOT = .123
def first_episode(track, lat_right=-.05):
    lat = np.array([.05, lat_right])
    idx, side0, af, al = np.int32(0), np.int32(1), HEEL0.copy(), LAT.copy()
    first_f = (HEEL0 + FOOT)[::-1] - .05 + .01
    prev, total, hits = np.float64(max(first_f[1] - HEEL0[1], 0.)), 0., 0
    for hl, hr, la, ra, alive in track:
        heel = np.array([hl, hr])
        hit, appr, idx, prev, side0, af, al, _ = dots(
            heel_forward=heel, foot_lateral=lat, landed=np.array([la, ra]), alive=np.bool_(alive),
            dot_index=idx, prev_dist=np.float64(prev), anchor_forward=af, anchor_lateral=al, start_side=side0,
            toe_forward=heel + FOOT, first_behind_toe=.05)
        total += float(hit) + float(appr); hits += int(hit > 0)
    return total, hits
F1 = {"right 8 cm, then zig-zag 3 cm (10 steps)": first_episode(steps([(1, .08)] + zig(9, first=0))),
      "right 9.2 cm (the user's demonstrated pose)": first_episode(steps([(1, .092)])),
      "right 15 cm (overshoot)": first_episode(steps([(1, .15)])),
      "right 5 cm (heel 7.3 cm behind the left toe)": first_episode(steps([(1, .05)])),
      "right 3 cm": first_episode(steps([(1, .03)])),
      "march in place": first_episode(steps(zig(6, move=0.))),
      "left 8 cm first (right-first mode)": first_episode(steps([(0, .08)])),
      "right 8 cm but 4 cm off its line sideways": first_episode(steps([(1, .08)]), lat_right=-.09),
      "right 8 cm and fall on landing": first_episode(steps([(1, .08)], fall_last=True))}
print("\n--- first step: right heel lands <= 5 cm behind the left toe, then +3 cm dots ---")
for k, (tot, hits) in F1.items():
    print(f"{k:50s} reward {tot:7.2f}  steps counted {hits}")
checks9 = [
    ("first: 8 cm right step then 3 cm zig-zag counts every step", F1["right 8 cm, then zig-zag 3 cm (10 steps)"][1] == 10),
    ("first: the demonstrated 9.2 cm step counts", F1["right 9.2 cm (the user's demonstrated pose)"][1] == 1),
    ("first: overshooting still counts", F1["right 15 cm (overshoot)"][1] == 1),
    ("first: a 5 cm step does not count", F1["right 5 cm (heel 7.3 cm behind the left toe)"][1] == 0),
    ("first: marching in place counts nothing", F1["march in place"][1] == 0),
    ("first: left foot first counts nothing (right-first)", F1["left 8 cm first (right-first mode)"][1] == 0),
    ("first: landing off its line sideways does not count", F1["right 8 cm but 4 cm off its line sideways"][1] == 0),
    ("first: falling on the landing does not count", F1["right 8 cm and fall on landing"][1] == 0),
    ("first: a counted step pays more than a short one", F1["right 9.2 cm (the user's demonstrated pose)"][0] > F1["right 5 cm (heel 7.3 cm behind the left toe)"][0] + 5),
]
for name, ok in checks9:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    failed += not ok

# Injected poses (user spec 2026-10-09): whatever pose is injected counts as step 1 on its first control
# step, credited to the foot furthest forward; the zig-zag then continues with the other foot.
def injected_episode(track, lead):
    lat = LAT.copy(); idx, side0, af, al = np.int32(0), np.int32(1), HEEL0.copy(), LAT.copy()
    prev, total, hits = np.float64(.05), 0., 0
    for n, (hl, hr, la, ra, alive) in enumerate(track):
        heel = np.array([hl, hr])
        hit, appr, idx, prev, side0, af, al, _ = dots(
            heel_forward=heel, foot_lateral=lat, landed=np.array([la, ra]), alive=np.bool_(alive),
            dot_index=idx, prev_dist=np.float64(prev), anchor_forward=af, anchor_lateral=al, start_side=side0,
            toe_forward=heel + FOOT, first_behind_toe=.05, force_side=np.int32(lead if n == 0 else -1))
        total += float(hit) + float(appr); hits += int(hit > 0)
    return total, hits, int(side0)
still = [(HEEL0[0], HEEL0[1], False, False, True)] * 20
I1 = {"inject, then stand still (right credited)": injected_episode(still, 1),
      "inject, then left 3 cm, right 3 cm": injected_episode(still[:1] + steps([(0, .03), (1, .03)]), 1),
      "inject with LEFT ahead, then right 3 cm, left 3 cm": injected_episode(still[:1] + steps([(1, .03), (0, .03)]), 0),
      "inject, then the same foot again (no credit)": injected_episode(still[:1] + steps([(1, .03)]), 1),
      "not injected, stand still": injected_episode(still, -1)}
print("\n--- injected pose counts as step 1 ---")
for k, (tot, hits, side0) in I1.items():
    print(f"{k:50s} reward {tot:7.2f}  steps counted {hits}  first foot {'right' if side0 == 1 else 'left'}")
checks10 = [
    ("inject: the injected pose itself counts as one step", I1["inject, then stand still (right credited)"][1] == 1),
    ("inject: zig-zag continues with the other foot", I1["inject, then left 3 cm, right 3 cm"][1] == 3),
    ("inject: a left-ahead pose credits the left foot, then right/left continue",
     I1["inject with LEFT ahead, then right 3 cm, left 3 cm"][1] == 3 and I1["inject with LEFT ahead, then right 3 cm, left 3 cm"][2] == 0),
    ("inject: the same foot again earns nothing more", I1["inject, then the same foot again (no credit)"][1] == 1),
    ("inject: without injection nothing is credited", I1["not injected, stand still"][1] == 0),
]
for name, ok in checks10:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    failed += not ok

# Final goal (user, 2026-10-09 night): unlimited episode to 100 m; resets only on a fall (heavy) or standing
# too long (heavy: -5/s after 2.5 s, stall-out -30 at 8 s). Fall -150 must stay worse than any stall-out and than a dive.
FALL_HEAVY = 150.
SIZES_HEAVY = dict(landing_weight=10., first_step_bonus=20., goal_bonus=500., stall_out_penalty=30.,
                   stall_out_seconds=8., stall_grace_seconds=2.5, still_pays=0.)
R11 = {}
print("\n--- heavy penalties: fall -150, stall-out -30 (+ -5/s after 2.5 s) ---")
for name, path, stride, err in V3 + [("sway then fall at 0.5 s", sway_then_fall(.5), .08, IGNORE),
                                      ("sway then fall at 3.0 s", sway_then_fall(3.), .08, IGNORE)]:
    total, t_end, terms = episode(path, stride=stride, exploration=True, fall_penalty=FALL_HEAVY, track_err=err,
                                  track_weight=.5, back_wall=-.20, alive_bonus=.5, sizes=SIZES_HEAVY)
    R11[name] = total
    print(f"{name:36s} {total:10.1f} {t_end:7.1f}s")
checks11 = [
    ("heavy: standing still is heavily punished (< -50)", R11["stand still"] < -50),
    ("heavy: falling is worse than standing still (no escape by falling)", R11["fall immediately"] < R11["stand still"]),
    ("heavy: diving then falling is worse than standing still", R11["land one step then fall (dive)"] < R11["stand still"]),
    ("heavy: walking all episode >> standing still", R11["walk 0.10 m/s following the gait"] > 10 * abs(R11["stand still"])),
    ("heavy: reaching 100 m beats everything", all(R11["reach 100 m following the gait"] > v for k, v in R11.items() if k != "reach 100 m following the gait")),
    ("heavy: walking backwards is worse than standing still", R11["walk backwards into the back wall"] < R11["stand still"]),
]
for name, ok in checks11:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    failed += not ok
total_checks = len(checks) + len(checks2) + len(checks3) + len(checks4) + len(checks5) + len(checks6) + len(checks7) + len(checks8) + len(checks9) + len(checks10) + len(checks11)
print(f"\n{total_checks - failed}/{total_checks} common-sense checks passed.")
raise SystemExit(1 if failed else 0)
