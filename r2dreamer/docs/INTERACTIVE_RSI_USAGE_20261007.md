# Interactive RSI v1 — Usage Guide

Companion to `INTERACTIVE_RSI_DESIGN_20261007.md` (the brainstorm/design
doc). This is the usage guide for the **v1 tool that actually exists**:
`r2dreamer/scripts/interactive_rsi.py`.

## What v1 actually is (and isn't)

This started as the **offline-only** version from the design doc (§4c's
"stop training, author a pose, launch/resume pointed at it" path). As of
2026-10-08 it also supports **live injection into an already-running
training process** — see "Live injection into a live run" below — but
still not the IK-driven foot dragging (§4a), which remains future work.

What it does right now:
- Opens the robot in a real, interactive MuJoCo viewer window, running
  **locally on your machine** (this needs a real display — it will not
  run on the headless remote GPU instance, which is why it's a local tool).
- Lets you pose the robot by hand using MuJoCo's native perturbation drag.
- Shows a live, continuously-updating readout in the terminal of the pure
  forward-kinematics quantities that drive the reward (`height_error`,
  `tilt`, root height, center-of-mass xy) — these are exact, copied
  directly from the same one-line formulas in `env.py`'s reward
  functions, not reimplemented guesses.
- Lets you commit the current pose as a new, disposable **scratch** RSI
  pose, then tells you the exact command-line flags to feed it into the
  next training launch.

**Not included yet:** the foot-pressure-sensor reward terms
(`contact_quality`, `load_balance_quality`) — those depend on a bespoke
per-foot sensor array tied to the training scene's contact solver, which
isn't safe to hand-reimplement without real risk of silently-wrong
numbers. What's shown instead is still directly useful: `height_error` is
exactly the quantity behind both `standing_reward_terms`' height shaping
and `walking_transfer_v2_reward_terms`' fall-equivalent sag violation —
the same quantity whose missing-gradient bug we found and fixed today.

## Prerequisites

Run this on your **local machine**, not the remote instance:

```
pip install mujoco mujoco-mjx
```

(`mujoco-mjx` isn't strictly required by this script, but installing it
is a quick way to confirm your local `mujoco` install is recent enough —
version 3.15.0 was confirmed working.)

## Running it

From the `r2dreamer` directory:

```
python scripts/interactive_rsi.py
```

Starts from the `standing_start` keyframe (upright, neutral pose) by
default. To start from one of the four existing saved poses instead:

```
python scripts/interactive_rsi.py --start-pose 3
```

(Pose IDs are `1`–`4`, matching `manual_rsi_poses.json`.)

## Controls

| Action | How |
|---|---|
| Pose the robot | **ctrl + right-click-drag** on any body part (standard MuJoCo perturbation — drag toward where you want it to go, the physics settles under the applied force) |
| Rotate the camera view | left-click-drag |
| Zoom | scroll wheel |
| Commit the current pose | press **S** *while the viewer window is focused* |
| Quit | close the viewer window |

The terminal window (separate from the viewer) continuously prints:

```
height_error=+0.0123m  tilt=  4.2deg  root_height=0.3018m  com_xy=(+0.012,-0.034)
```

- `height_error`: how far the torso is above (+) or below (−) the
  standing-height reference (0.289478m). This is the exact quantity that
  the -10cm fall-equivalent cliff in `walking_transfer_v2` checks, and
  that `standing_reward_terms`' height shaping rewards.
- `tilt`: how far the torso has tipped from fully upright, in degrees.
- `root_height`: the raw torso height in meters (not relative).
- `com_xy`: the robot's center-of-mass horizontal position.

## Committing a pose and using it in training

Press **S** in the viewer. The terminal prints something like:

```
*** Committed scratch pose '1' to manual_rsi_poses_scratch.json ***
To use it in training, launch/resume with:
  pose_rsi_json=envs/mjx_walking/assets/manual_rsi_poses_scratch.json training_pose_id=1 rsi_probabilities=[0.0,0.0,0.0,1.0]
```

Copy that into your next training launch command (on the **remote**
instance, via `init_from=<latest checkpoint>` plus those three flags) —
same pattern already used throughout this project's pose curriculum,
just pointed at the scratch file instead of the permanent
`manual_rsi_poses.json`.

Each commit gets the next integer ID automatically
(`manual_rsi_poses_scratch.json` keeps growing — `1`, `2`, `3`, ...), so
you can author and try several poses across sessions without overwriting
earlier ones.

**Promoting a scratch pose to permanent**: copy its entry by hand from
`manual_rsi_poses_scratch.json` into `manual_rsi_poses.json` (as pose
`5`, `6`, etc.) once you've confirmed via training it's actually useful.
Nothing is promoted automatically — that's a deliberate design decision
(see design doc §3, point 1).

## Live injection into a live run

This pauses **every world** in a running training process at once (not
one world while the rest keep training — that would require freezing a
single lane inside a vectorized JAX step, which is a much harder change;
pausing the whole batch together was the design decision made for this
feature), lets you author one pose by hand, and on commit resets **all**
worlds to that pose: world 0 gets your exact pose, the rest get the same
pose with small random noise added to the actuated joint angles (and the
matching ctrl target), so one authored pose becomes a small cluster of
nearby starting states rather than `env_num` identical duplicates.
Training then resumes automatically from there — no relaunch needed.

**This only works if `interactive_rsi.py` can write into the exact same
directory the training process reads from.** Run it directly on the
training machine (e.g. the friend's laptop), not from a different machine
unless that run's logdir is mounted/synced there.

1. Find the live run's logdir (the `logdir=...` path passed to `train.py`,
   e.g. `~/runs/walking_from_mixed_20261008_120000`).
2. Launch the tool pointed at it:
   ```
   python scripts/interactive_rsi.py --inject-rundir ~/runs/walking_from_mixed_20261008_120000
   ```
   Optional: `--noise-std 0.03` (radians, default) controls how much the
   non-world-0 copies vary from your authored pose; `--seed N` makes that
   noise reproducible.
3. Press **P** in the viewer (2026-10-08: does the `touch` below for you, then
   loads the paused run's live world 0 into the viewer so you edit where the
   robot actually is). Or, manually: request a pause (training keeps running until this file appears —
   create it any time, from any terminal, including a second one):
   ```
   touch ~/runs/walking_from_mixed_20261008_120000/pose_injection_request.flag
   ```
   Training's log will print `*** Pose injection requested ... pausing
   training ***` within one loop iteration (typically a few seconds) and
   block there. There is no episode timer running during this wait —
   it stays paused however long you take.
4. Pose the robot in the viewer as usual (ctrl + right-click-drag), using
   the live geometry readout to judge it.
5. Press **I** to commit. This writes `injection_edited_state.npz` into
   the run's logdir. Training picks it up within ~1s, logs `*** Pose
   injection applied ... resuming training ***`, and continues.
6. Repeat from step 3 as many times as you like, at any point in the run.

**Automatic backups**: every injection snapshots, before applying, into
`<logdir>/injections/<timestamp>/`: `pre_injection_latest.pt` (checkpoint
at the moment of injection), `injection_edited_state.npz` (the pose you
authored), and `injection_current_state.npz` (what world 0 looked like
right before, if available). A one-line append-only record of every
injection (step, timestamp, noise_std, backup path) is kept in
`<logdir>/injections.jsonl`.

**What's NOT included**: no live reward-score readback from the actual
training process during authoring — the geometry readout is computed
locally against your own CPU MuJoCo copy, same pure-FK quantities as the
offline mode, not round-tripped through the GPU/MJX physics. The one-time
sanity check from the design doc (§5.1: confirm the CPU preview matches
what training actually computes for that world) has not been done —
worth doing once before trusting the numbers blindly for anything
high-stakes.

## Known limitations of this v1

- No live reward-score HUD beyond the geometry readout above (no
  `contact_quality`/`load_balance_quality`, no total reward number) —
  see "What v1 actually is" above for why.
- Dragging is plain MuJoCo perturbation, not hierarchy-aware IK — grabbing
  a foot does not yet solve the whole leg while anchoring the other foot.
  It behaves like grabbing any other body: a local force, not a
  constrained whole-chain solve.
- Live injection pauses the entire world batch, not a single world while
  the rest keep training — see "Live injection into a live run" above.
- The live-injection code path (env.py's `use_forced_pose`, mjx_teacher_env.py's
  `inject_pose`/`_pending_injection_first`, trainer.py's
  `_check_pose_injection`) has been written and syntax-checked but **not
  yet run against a real JAX/MJX GPU training process** — there was no
  local GPU available to validate it end-to-end before handing it off.
  Treat the first real use as a test: watch the training log closely for
  the injection print statements, and confirm (via TensorBoard or
  `episode/eval_*`) that the run behaves sensibly afterward.
