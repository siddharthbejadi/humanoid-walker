# Project review: why walking stalled, and what is fixed (2026-10-09)

Written overnight while the two-stage 1024-robot run trains on the RTX 5090.
Every claim below was measured in this project; numbers are from the logs.

## The mistakes, in order of damage

### 1. The simulated servos could not hold the robot up (root cause)
- `humanoid_rubber_2mm_candidate.xml` gave each servo an inertia-scaled `kp`
  (hip 45 → knee 2.3 → ankle 0.43 N·m/rad; `kv` always 4% of `kp`).
- With no policy and every servo holding `standing_start`, the robot fell in
  0.7 s (CPU) / 0.47 s (MJX). Centre of mass was inside the feet; the knees
  folded 10° and it sat back.
- Every run before 2026-10-08 (1M, 14M with 88/88 falls, all reward
  versions) trained a policy fighting limp ankles. Standing needed constant
  effort; lifting a foot was suicide; "stand still" or "dive" were rational.
- **Fix:** `humanoid_rubber_2mm_servo.xml`, every servo kp 20, kv 0.5.
  The robot stands with zero action (5 s until the stall rule, on CPU and MJX).
- **Open:** kp 20 is a typical hobby-servo value, not measured. Measure the
  real knee sag under the robot's weight and set kp from it.

### 2. Rewards paid for the wrong thing (four times)
| version | what it paid for | consequence | fix |
|---|---|---|---|
| walking_transfer_v2 | ~16/step for standing still | froze, then fell | v3: quality scales forward speed |
| v2 height/corridor | hard cliffs, no gradient | rode the −10 cm / ±13.5 cm edge | smooth shaping |
| goal_corridor_v1 | sound (13 checks) | no learning yet | — |
| goal_corridor_v2 | landing pays even if it falls | land-then-fall (−27) > stand (−60): dove forward | v3: fall −200 |
- **Lesson:** every reward change now goes through
  `scripts/check_goal_corridor_reward.py` (27 scripted episodes, ordering
  checks) before any GPU time. It caught 4 real bugs (lunge exploit,
  shuffle-farming, dive, backward-wall loophole).

### 3. Runs barely learned
- `train_ratio 64` + `warmup 1000` with 8 worlds gave ~260 updates per 50k
  steps and a learning rate stuck at 6% of target. Several "it didn't walk"
  runs were really "it didn't train".
- **Fix:** `warmup 100`; train_ratio chosen per world count
  (8–64 worlds: 512; 512–1024 worlds: 64). Always check
  `train/opt/updates` and `train/opt/lr`, not just step count.

### 4. Exploration settings fought the robot
- Inherited actor std 0.0075–0.0375 (~3°): too small to discover a gait.
- Raised to 0.03–0.15: with stiff servos, ~8°/step noise sent robots into the
  10 cm back wall in 0.8 s; every training episode ended in <1 s and the
  policy learned nothing useful (64-robot run eval at 49k: 0.26 s).
- **Fix:** std 0.01–0.06 (~3° at start), v3 back wall 20 cm, action range
  ±1.0 rad (covers the vendor gait's 0.91 rad max) in both stages.

### 4b. With stiff servos, the action range is the stability knob (found overnight)
- Balance at ±1.0 rad (1024 robots): evals got worse, 5.4 s → 2.1 s → 1.6 s
  (8/8 falling backwards by 2M). The policy's action spread grew to 0.14
  (≈8°); at kp 20 a 0.14 rad error already asks for more than the 1.77 N·m
  limit, so small policy errors became full-torque shoves.
- Balance at ±22.5° (0.39 rad, the range that learned standing before):
  1M: 0 falls, full 10 s; 2M: score 599, upright 115/125, stable-stance bonus
  191/250. `best.pt` = 2M.
- Walking at ±1.0 rad (standing-relative actions) collapsed the same way:
  training episodes 1.8 → 0.7 s, action spread 0.18–0.20, critic predicting
  +50..+80 while real returns were negative.
- **Fix: residual walking** (`env.mjx_gait_residual=true`): motor target =
  vendor gait reference(t) + 0.39 rad × action. The gait supplies the large
  motion (it needs ~0.91 rad); the policy only learns ±22.5° balance
  corrections, the range it proved it can use. This is what
  `walking_vendor_residual_20m.yaml` intended but the MJX env never had.
  With zero action the robots follow the gait (track reward ~80–89) and walk
  ~13–16 cm before falling at ~3.3 s; the policy's job is to stop those falls.
- One-leg poses 3/4/7 need 70–75° of joint offset; at ±22.5° the pose loader
  clips them into "other" without their lift, so balance trains standing +
  reachable variants, and one-leg support is learned inside the gait.

### 4c. Two walking-reward traps that only show up inside Dreamer (found overnight)
- **"Die early" optimum.** v3 had no income for staying up, only per-step
  penalties (drift, heading, stall). Once every early episode ends in the
  fall penalty anyway, ending sooner scores better: sway 3 s then fall =
  −214.7 vs fall at 0.5 s = −199.8. Residual walking learned to fall
  backwards in 0.57 s. **Fix:** `mjx_alive_bonus=0.5` per step (standing
  still stays capped by the 5 s stall-out).
- **Giant sparse bonuses poison the reward predictor.** Dreamer predicts
  reward as a two-hot distribution over symlog bins; with +1000 / +10000 /
  +50 spikes in the data, small probability on those bins inflated every
  prediction: +6.3/step predicted vs −1.3 real (balance stage, no spikes:
  0.25 predicted vs ~0.38 real). The policy chased phantom bonuses by diving
  forward (0.61 s, +26 cm) even though the real reward ranks diving below
  following the gait. **Fix:** sizes fall/wall −50, landing +10, first step
  +20, goal +500, stall-out −10 (`mjx_fall_penalty`, `mjx_landing_weight`,
  `mjx_first_step_bonus`, `mjx_goal_bonus`, `mjx_stall_out_penalty`); all 44
  common-sense orderings still pass.
- A wrong lead worth recording: the continuation predictor read ≈1.0 with a
  near-zero loss. Replay was checked directly (terminals stored and sampled
  at the expected 0.6%/step); falls are simply easy to predict from state.

### 5. Old checkpoints carried the old physics
- Policies trained on soft servos over-steer on stiff ones (7/8 wall hits at
  step 0, dive by 10k). **Fix:** from scratch on the servo scene.

### 6. Scale problems hidden below 64 worlds
- MJX-Warp default CUDA-graph capture fails past ~64 worlds ("Failed to
  allocate", with free memory). **Fix:** `GraphMode.WARP_STAGED` above 64
  (512 worlds ≈ 5 GB, 1024 passed preflight).
- The "no updates yet" guard (100k steps) killed 512-world runs before the
  replay filled (~343k). **Fix:** deadline scales with world count.
- Replay 8192 transitions = 2.6 s per robot at 64 worlds. **Fix:** 131k–524k.
- `best.pt` was chosen by fall_count first, but under v3 every episode ends
  early. **Fix:** `best_metric_key=score`, `task_first`.

### 7. Process mistakes
- Two working copies (`/mnt/c/.../r2dreamer` stale, `~/r2dreamer` real).
- The old zip ran on the instance after fixes were made locally (deadline
  crash). **Fix:** `vast_setup.sh`/preflight check the code version by
  running it; bundle rebuilt and re-uploaded after every fix.
- TensorBoard video logging crashes (numpy `reshape(newshape=)`): keep
  `eval_video_tensorboard=false`; MP4s work.

## What is running now
- RTX 5090, 1024 robots, tmux `train`:
  1. **balance** — done (3M steps, ±22.5°, 50% standing / 50% other poses);
     `runs/servo_balance/best.pt` is the 2M policy (score 599, 0 falls).
  2. **walk** — residual on the vendor gait, ±22.5° corrections,
     goal_corridor_v3, 20M steps, from the balance policy → `runs/servo_gait_v3`.
- Failed/collapsed runs are kept in `runs/old/` for comparison.

## What would most likely still stop walking
1. **The vendor gait does not walk open-loop on MJX** (~3 cm, falls at
   ~3.5 s). The policy must learn balance corrections on top of it.
2. **Per-robot experience is short at 1024 worlds** (20M steps ≈ 39 s per
   robot); episodes rarely reach late-gait states. Mitigated by many robots
   and RSI starts; watch alive_seconds.
3. **kp 20 unmeasured**; a real servo may be stiffer or softer.
4. **Sim-to-real** is untouched: no domain randomisation of mass, friction,
   servo gains or latency yet.

## Next steps, in order
1. Judge stage 1 at its end: one-leg pose starts (3, 4) surviving several
   seconds, falls well below 50%.
2. Judge stage 2 every 1M steps: `eval_reward_first_step` > 0, landing and
   track rewards rising, alive time staying high; MP4s for diving.
3. If walking appears: add domain randomisation, then a standalone eval
   (full 400 s horizon, 100 m goal) before any hardware test.
4. Measure real servo stiffness; rebuild the scene with measured kp.
