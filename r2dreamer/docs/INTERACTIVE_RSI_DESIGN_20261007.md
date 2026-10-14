# Interactive RSI ("Director Mode") — Design Brainstorm

## 1. The core idea

A tool that lets a human directly pose/manipulate the robot inside the
simulator — live, by hand — while watching the *exact* reward and penalty
terms the policy is trained on update in real time. Once a state (or short
motion) looks right, it's handed back to training as a new starting point
(a reset state), using the *same* network weights and *same* reward
function, unchanged. The only thing that changes is *how an episode
starts* — a human-authored state instead of one the policy had to
discover through its own random exploration.

This project already has a formal name for "where an episode starts":
**RSI** (Reference State Initialization — `manual_rsi_poses.json`,
currently 4 hand-authored poses, no live feedback during authoring). This
feature is RSI authoring turned into an interactive, live-scored process,
so the natural name is:

> **Interactive RSI** (nickname: **Director Mode** — you block the scene,
> then hand it to the actor).

## 2. Why this matters (motivation)

- **Exploration is slow and can fail entirely.** The policy has to
  randomly stumble into a useful configuration before it can learn
  anything about it. We watched this happen repeatedly this project —
  pose-3's single-leg balance took ~375k steps of near-total silence
  before any signal appeared, purely waiting on exploration luck.
- **Reward bugs are currently invisible until a full training run fails.**
  The height-sag reward bug just found (zero gradient before the -10cm
  cliff, policy converges on riding the edge of it) would have been
  immediately visible by hand-dragging the torso down in an interactive
  viewer and watching `penalty_fall` spike at the exact boundary — no
  1M-step run required to discover it.
- **Curriculum design becomes direct instead of indirect.** Right now,
  getting the robot to experience "foot lifted 8cm, torso leaning
  forward 15°, weight shifting onto the other leg" requires either (a)
  hoping RSI + exploration gets there, or (b) guessing joint angles
  blind and checking a saved JSON file's `qpos` numbers with no feedback
  on whether the pose is even physically sensible. A live tool collapses
  that to: drag it there, look at the score, done.

## 3. Decided design (resolved 2026-10-07)

Four design questions came up while scoping this; here's where each one
landed:

1. **Scratch poses, not auto-permanent.** Every authored pose starts as a
   disposable scratch entry. Nothing is added to the permanent
   `manual_rsi_poses.json` curriculum library automatically — promoting a
   scratch pose to permanent is a separate, explicit action.
2. **Live injection into an already-running episode — not an offline
   stop/restart workflow.** Entering manual mode happens *inside* a live
   training run, mid-episode. Training is not stopped and relaunched;
   the running process is paused/diverted at the point of entry and
   resumed from the edited state afterward.
3. **No episode timer while in manual mode.** The normal
   `time_limit` auto-reset must be suspended for the duration of an
   editing session — an episode being hand-posed cannot be allowed to
   time out mid-edit. It stays open until the session is explicitly
   finished (committed or cancelled).
4. **Repeated, ongoing use — not a one-shot tool.** A single injected
   pose shifts the training distribution; the ability to do this
   *many times* over a run's lifetime, iteratively steering the
   curriculum, is the actual point of the feature, not an edge case.
   The tool needs to support being invoked over and over against the
   same live run.

A fifth requirement, not originally a question but added alongside these
answers: **automatic backups around every injection** — the current
checkpoint, the exact pose/state being committed, and run metadata, all
snapshotted before the edit is applied. Since this directly mutates a
live run's trajectory, every injection needs to be cheaply reversible.

## 4. What it requires, broken into parts

### 4a. Hierarchy-aware manipulation (FK for terminal links, IK for feet)

The manipulation model isn't just "grab any body and apply a spring
force" — different body parts should respond differently depending on
their position in the kinematic chain, matching how a rigged animation
character behaves:

- **Grabbing a terminal link (head) only moves that link.** This is
  forward kinematics, and it's *already* how MuJoCo works — the model is
  a parent-child kinematic tree by construction. Applying a modest
  perturbation force near a leaf body (head/neck) moves that local joint
  without dragging the rest of the body along, as long as the applied
  force stays small relative to the rest of the body's inertia. Nothing
  to build here beyond using the existing mechanism correctly.
- **Grabbing a foot drives a real inverse-kinematics solve, anchoring
  the other foot.** This is genuine IK and *is* something to build:
  MuJoCo doesn't ship a ready-made solver for "move this end-effector,
  solve the whole chain above it, while another named point stays
  pinned." The standard technique is a damped-least-squares Jacobian
  solve — get the Jacobian of the dragged foot's site via
  `mj_jacSite`/`mj_jacBody`, solve for a joint-angle update that moves it
  toward the cursor, and add the opposite foot's Jacobian as a secondary
  constraint (drive its projected velocity toward zero) so it stays
  planted. This is well-understood, standard robotics code (the same
  category of thing `dm_control`'s `inverse_kinematics` module does) —
  real work, not exotic, and not a MuJoCo setting to flip.
- **Grabbing the torso/chest allows soft, organic deformation across the
  connected chain**, with a springy release rather than a sharp snap.
  This maps to MuJoCo's existing joint stiffness/damping parameters —
  tuning those gives a natural "settle back" feel when the mouse is
  released, which is most of what reads as "jellyfish-like" in practice.
  **Honest caveat:** this is springy *joints*, not a deformable *body*.
  MuJoCo simulates rigid links connected by joints; the metal/plastic
  links themselves won't visually bulge or squish like silicone. Worth
  being explicit about this distinction so the tool doesn't promise
  soft-body material deformation the physics engine doesn't do.

### 4b. Live reward HUD

The reward functions (`standing_reward_terms`, `walking_transfer_v2_reward_terms`,
etc.) are pure math over `qpos`/`qvel`/sensor-derived quantities — nothing
GPU-specific about the computation itself. Given any state snapshot (even
from a plain CPU MuJoCo mirror), the exact same term-by-term breakdown
already logged in training (`reward_forward`, `reward_height`,
`penalty_corridor`, `penalty_fall`, ...) can be recomputed and displayed
every frame, cheaply — it's small-tensor math, no batching needed.

Display options, roughly in order of effort:
1. Terminal printout, refreshed per frame (cheapest, works immediately).
2. MuJoCo viewer's built-in text overlay (`mujoco.viewer` supports custom
   on-screen text) — keeps pose + score in one window.
3. A live-updating bar chart of every term — most useful for directly
   seeing *which* term is punishing a pose as you move it (e.g. drag the
   torso down and watch `reward_height` decay smoothly now, then watch
   `penalty_fall` snap on at the cliff — exactly the signal that was
   missing when that bug was designed).

### 4c. Live injection into a running training process

Given the decision in §3, this is the actual mechanism, not a stretch
option: the live training process (64 parallel MJX worlds on GPU) needs
a side channel it polls at reset time — e.g. a small flag/file/queue
checked each episode boundary: *"is there a pending manual-override state
for world N? If so, use it instead of sampling the normal RSI
distribution, and suspend that world's time-limit reset until the editor
signals completion."*

Practical shape: the interactive editor runs as a **separate process**
from the training loop (it needs its own responsive render loop and
can't share the GPU training step), and the two coordinate through a
lightweight shared signal:
1. Editor requests control of world N → training loop stops stepping
   that world (the other 63 keep training normally) and copies its
   current `qpos`/`qvel` out to the editor.
2. Editor runs its own CPU-side MuJoCo copy of that state, lets the user
   drag/pose it with the hierarchy-aware (FK/IK) model from §4a, showing
   the live reward HUD from §4b. No timer — stays open until "commit" or
   "cancel."
3. On commit: snapshot the backup bundle from §4d, then write the edited
   `qpos`/`qvel` back into world N's slot and let training resume
   stepping it normally from that state.
4. The editor can be invoked again at any later point, against the same
   or a different world — this is the "repeated use" requirement, not a
   single-shot tool.

### 4d. Backup strategy

Every injection snapshots, before applying the edit:
- The current training checkpoint (or confirmation one was already just
  written — don't inject against a checkpoint that doesn't exist yet).
- The exact pre-edit and post-edit state (`qpos`/`qvel`/`ctrl`) for the
  affected world, so an injection can be undone exactly.
- A small metadata record (timestamp, step count, which world, who/what
  triggered it) appended to a log — cheap, append-only, never
  overwritten, so the full history of manual interventions on a run is
  reconstructable later.

## 5. Remaining open questions

1. **Single-robot preview vs. matching the live 64-world batch exactly.**
   The editor's CPU-side MuJoCo copy should numerically match the
   GPU/MJX physics closely but isn't guaranteed bit-identical. Worth a
   one-time sanity check (author a pose, inject it, confirm the
   live-previewed reward score matches what training actually computes
   for that world next step) before trusting the tool's numbers blindly.
2. **Trajectory authoring** (keyframe sequences rather than a single
   static pose) — in scope for v1, or a v2 idea once the static-pose,
   single-world-injection version is proven useful?

## 6. Suggested v1 scope (concrete, buildable)

Given the decisions in §3, v1 is larger than a simple offline authoring
script, but still cleanly separable from `train.py`'s core loop:

1. A side-channel mechanism in the training loop: at each world's reset
   check, look for a pending manual-override request; if present, hand
   that world's control to the editor process and suspend its timer.
2. A standalone editor process that, once handed a world's state:
   - Mirrors it into a single CPU-side MuJoCo instance for responsive
     interactivity (no GPU/JAX involved in the editor itself).
   - Supports hierarchy-aware dragging — FK for terminal links, a
     damped-least-squares IK solve (with the opposite foot pinned) for
     feet, spring/damping-tuned release for the torso.
   - Shows a live, per-term reward/penalty breakdown every frame, using
     the real reward-term functions.
   - On commit: writes the backup bundle from §4d, then hands the edited
     state back to the training loop for that world and releases control.
3. Designed to be invoked repeatedly against the same live run, not
   torn down after one use.

This still reuses almost everything that already exists — the reward
functions, the RSI pose schema, the MJX environment itself — the new
work is specifically the side-channel handoff protocol, the IK solver,
and the editor's interactive loop.
