# Why it never walked, and what changed (2026-10-08)

## Root cause: the simulated servos could not hold the robot up

`humanoid_rubber_2mm_candidate.xml` gives each position actuator a different
`kp` that shrinks down the leg, with `kv` fixed at 4% of `kp` (an
inertia-scaled pattern, not a servo model):

| joint | kp (N·m/rad) | error needed for full 1.77 N·m |
|---|---|---|
| hip (torso) | 45 | 2° |
| hip (upper) | 17 | 6° |
| thigh bracket | 6.8 | 15° |
| knee | 2.3 | 43° |
| ankle | 0.43 | 236° (limp) |

Measured with no policy, every servo holding `standing_start`:

| scene | stand still | vendor gait open-loop |
|---|---|---|
| candidate (CPU) | falls at 0.7 s (knees fold 10°, sits back) | falls at 0.7 s |
| candidate (MJX/GPU) | all 8 worlds fall at 0.46–0.48 s | same |
| servo kp 20 (CPU) | stands 5 s | +30 cm, falls at 3.1 s |
| servo kp 20 (MJX/GPU) | stands until the 5 s stall rule ends the episode | ~3 cm, falls at ~3.5 s |

The centre of mass is inside the feet (4 cm from the back edge) and both soles
have 8 floor contacts, so geometry is not the problem. Every earlier run
(1M, 14M, all reward versions) trained a policy fighting limp ankles; "stand
still" or "dive forward" were its best options.

## Changes

| file | change |
|---|---|
| `envs/mjx_walking/assets/humanoid_rubber_2mm_servo.xml` | new: copy of the candidate scene, every servo kp 20, kv 0.5 (forcerange still set by `mjx_motor_torque_kgcm`). The candidate scene is unchanged. |
| `envs/mjx_walking/scene.py` | `vendor_gait_reference()`: joint-target table from `validation/vendor_forward_3_steps.csv` — start-up frames 1–6 (5.5 s), then the stride cycle frames 7–10 (3.0 s, left + right step) looped; frames 7–10 repeat exactly as 11–14 in the CSV. |
| `envs/mjx_walking/env.py` | reward profile `goal_corridor_v3` = `goal_corridor_v2` (step = foot landing ≥ 2 cm ahead of its standing home, length-scaled) + fall/wall −200 (v2's −100 made diving forward beat standing) + gait tracking `0.5·exp(−mse/0.26²)` per step; the policy's 3 command slots carry (0.08, sin φ, cos φ) of the stride phase. |
| `envs/mjx_teacher_env.py` | passes the gait CSV to the env; logs `reward_track`. |
| `scripts/check_goal_corridor_reward.py` | 26 common-sense orderings across v1/v2/v3 (all pass). |
| `scripts/deploy/start_interactive_walking.sh` | defaults: servo scene (`SCENE=`), `goal_corridor_v3` (`REWARD_PROFILE=`), actor std 0.03/0.15 (`MIN_STD`/`MAX_STD`; the 0.0075/0.0375 inherited from the transfer config explored only ~3°). |
| `scripts/interactive_rsi.py` | editor uses the servo scene by default; `--scene` to choose. |

## Still to confirm

- kp 20 is a reasonable hobby-servo stiffness, not a measurement. Measure the
  real knee sag under the robot's weight and set kp from it.
- The vendor gait alone does not walk on MJX; the policy must learn the
  balance corrections on top of it.
- Checkpoints trained on the soft scene learned limp-ankle compensation; they
  are only a starting point here (`init_from`), not a transfer.
