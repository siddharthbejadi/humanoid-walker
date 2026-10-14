# Script-Free Policy Training: Start Here

Prepared 13 October 2026, V2. This archive contains source preparation, not a
trained walking policy or a passed GPU qualification. The user has reported
renting a GPU; this package has not been executed on that instance. Local CPU checks exercise the
software contracts; CUDA, demonstration quality and learned walking still
need the target machine.

V2 improves launch integrity, setup diagnostics, signal cleanup, stage status
and stopped-run backup. It does not change the reviewed physical recipe or
automatically extend training. The verified 12 October package remains
preserved; use one declared source version per run.

The student observes only 45 proprioceptive channels and outputs 19 complete
motor commands. Its environment uses actuator midpoints and half-ranges to
map normalized commands to physical joint targets. It has no gait table,
explicit stabilizer, phase clock, phase RSI pool, privileged actor state or
reference target inside the policy/environment at inference.

The separate demonstration collector uses the humanlike reference and roll
stabilizer as an external expert, acting through that same full-command
student environment. Recurrent behavior cloning teaches complete commands;
R2Dreamer then continues with online world-model reinforcement learning and
retains supervised command loss on observed demonstration sequences. This is
a BC-first route. DAgger or student-policy aggregation is not implemented.

## Package And Hardware

The archive contains all source, the selected scene and meshes, pinned runtime
requirements and an entry-by-entry SHA-256 manifest. `humanlike_gait.npz` is an
external collector input only. The old 57-pose snapshot pool is not a dense
demonstration dataset and is not used by the student. Generated demonstrations,
checkpoints, logs and virtual environments are excluded from the archive.

Plan for one Linux NVIDIA GPU with at least 32 GB VRAM, around 64 GB host RAM
and 100 GB disk headroom. These are planning estimates, not measured minimums
or a guarantee that the full learner fits. Prefer a recent driver such as 580
or newer as a conservative CUDA 12/PTX starting point. Actual runtime and
learner checks decide admission. NVIDIA's
[compatibility guidance](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
explains the limits of minor-version compatibility and PTX support.

The target runtime is Python 3.11 with the pinned core in `requirements-lock.txt`:
Torch 2.8.0 (CUDA 12 build), TorchRL 0.9.2, TensorDict 0.9.1, MuJoCo/MJX 3.14.0,
JAX/JAXlib 0.10.2, Warp 1.17.0 and NumPy 2.4.6. Driver visibility alone does not
establish working Torch CUDA, JAX GPU, rendering or full-learner capacity.

## Upload And Setup

Use your rented instance's actual SSH host, port and user in PowerShell:

```powershell
$GpuHost = 'YOUR_INSTANCE_HOST'
$GpuPort = 22
$GpuUser = 'YOUR_INSTANCE_USER'
scp -P $GpuPort 'D:\Final Project\humanoid-world-model-control-mjx-teacher\walking\script_free_training_20261013_v2\r2dreamer_script_free_next_20261013_v2.zip' "${GpuUser}@${GpuHost}:/workspace/"
ssh -p $GpuPort "${GpuUser}@${GpuHost}"
```

On the Linux GPU host, unpack into a fresh directory and verify before installing:

```bash
set -euo pipefail
cd /workspace
test ! -e script_free_source_20261013_v2
unzip r2dreamer_script_free_next_20261013_v2.zip -d script_free_source_20261013_v2
cd script_free_source_20261013_v2
python3 r2dreamer/scripts/verify_script_free_bundle.py
bash r2dreamer/scripts/deploy/setup_script_free.sh 2>&1 | tee /workspace/script_free_setup_v2.log
cd r2dreamer
```

The source verifier uses only the standard library and supports system Python
3.8 or newer. It checks the script-free bundle identity, required closure,
file hashes, path containment and unlisted source/configuration shadow files.
Runtime caches, run-output folders and scenes generated from listed originals
are allowed without changing the listed source hashes.
Setup installs the pinned Python 3.11 runtime
and probes EGL/OSMesa rendering. A failed renderer blocks the guarded workflow.
Keep the source unchanged after verification; the launcher checks it again.

The Linux image must provide `python3` (3.8+), `curl`, `unzip`, `nvidia-smi`,
`tar` and `sha256sum`, network access for pinned packages, and a writable
workspace. On Debian/Ubuntu, graphics/tmux installation needs package-manager
permission; otherwise install those system dependencies yourself. Setup records
GPU inventory, RAM and free disk in `.setup_host_report.json`. Resource estimates
are advisory, not a measured capacity pass. Setup preserves an existing `.venv`
by refusing to recreate it; use a fresh extraction after investigating a partial
install. Do not rerun setup over a running learner.

Run preflight and pilot inside an interactive persistent session:

```bash
tmux new -s script_free_v2
```

To return after an SSH disconnect, use `tmux attach -t script_free_v2`.

## Preflight First

Use a new output directory. The default switch is preflight-only:

```bash
SCRIPT_FREE_SEED=0 \
SCRIPT_FREE_RUN_DIR=/workspace/script_free_preflight_v2_seed0 \
SCRIPT_FREE_PREFLIGHT_ONLY=1 \
bash scripts/deploy/start_script_free.sh
```

Only the exact value `0` permits the pilot. `1` is the default; other values
are rejected. `SCRIPT_FREE_SEED` defaults to 0, and an optional `seed=N` launch
argument can select another declared seed. Output directories must be fresh.
Preflight performs bounded work on the rented GPU; stopping the program does
not terminate or stop billing for the rented instance.

The guarded workflow validates the pinned runtime, student controller/observation
contract, reset/RNG repeatability, retained observation views, 15 mm step detector,
script independence, finite physics and rendering. Its capacity route collects
24 episodes, eight worlds per seed group, with an external eight-second cutoff.
Whole seed groups split into eight training, eight validation and eight test
episodes. Collection does not alter the student's 20-second physical contract.

Each split must have at least 256 safe expert labels, at least 256 safe labels
after the 64-frame history prefix, and safe standing-start labels. Failed expert
rollouts cannot supply these labels. BC must produce finite changed weights and
improve held-out validation loss to pass the next gate.

Capacity performs 32 real BC/world-model optimizer updates, then exactly 32
online RL optimizer updates with the full 128 training worlds, 32 evaluation
worlds and replay capacity 131072. It requires all 32 nonempty final-checkpoint
evaluation MP4s with matching world, standing-start, training-step, frame-count
and FPS sidecars. Initial-only or partial videos do not pass. Independent
checkpoint-only evaluation follows with another complete 32-world video set.
Runtime, source/contract/hash mismatches, overflow, nonfinite values, missing
safe labels, incomplete updates or missing final diagnostic evidence block
admission. A capacity pass is a short software/runtime result, not a walking claim.

Inspect these outputs before extending:

| Artifact | Path Under The Run Directory |
| --- | --- |
| Runtime/device evidence | `runtime/gpu_runtime_report.json` |
| Demonstration arrays and provenance | `capacity/demonstrations.npz` and `.manifest.json` |
| External-expert MP4s/sidecars | `capacity/demonstrations_videos/` |
| BC checkpoint | `capacity/imitation/latest.pt` |
| Online capacity checkpoint | `capacity/online/latest.pt` |
| Independent student evaluation | `capacity/independent_eval/script_free_evaluation.json` |
| Capacity and overall decisions | `capacity/report.json`, `readiness_report.json` |
| Durable current stage and history | `run_status.json`, `stage_events.jsonl` |

## Bounded Pilot

After reviewing preflight evidence, choose another fresh output directory and
explicitly enable the bounded pilot:

```bash
SCRIPT_FREE_SEED=0 \
SCRIPT_FREE_RUN_DIR=/workspace/script_free_pilot_v2_seed0 \
SCRIPT_FREE_PREFLIGHT_ONLY=0 \
bash scripts/deploy/start_script_free.sh
```

The pilot repeats capacity first. Its main collection/training seed starts at
`SCRIPT_FREE_SEED + 100`, keeping it separate from capacity; exact per-rollout
seeds and whole-group splits are recorded in the dataset manifest. It collects
64 dense 50 Hz episodes with
a 20-second maximum, using eight parallel worlds. Entire independent seed groups
split 48 training / eight validation / eight test episodes. Each group's world-0
first episode has a required MP4 and matching metadata; short or failed episodes
remain visibly short or failed. Collection records pre-action observations,
current complete expert commands, desired/applied targets, next observations,
body/joint state, raw contact sensors, loads, rewards and termination causes.

Behavior cloning uses 1000 bounded updates before a 128-world, 1M-counted-tick
online pilot with 32 repeatable held-out standing evaluations. `pilot/imitation/`,
`pilot/imitation_eval/`, `pilot/online/` and `pilot/independent_eval/` retain the
checkpoints, metrics and video evidence. One million counted ticks is about
100,000 active control transitions across all training worlds, not a million
decisions per robot. Budgets and update counts establish optimization only.

The loader aligns `obs_t` with previous command/reward/terminal and the current
expert command as a label. Startup windows supervise standing initiation from
the true initial recurrent state; later windows retain burn-in history. Actor
encoder inputs contain only `proprio`, never the expert label or audit state.
The checkpoint binds both NPZ bytes and semantic manifest bytes through
`data_sha256` and `manifest_sha256`; keep both files unchanged when transferring.

Rollouts ending in a physical fall or wall failure retain raw audit data but
their BC labels are disabled. Fully unsafe episodes are currently audit evidence,
not offline world-model samples, because the default sampler selects windows
with safe BC labels. Online RL later supplies failure dynamics. No-safe-label
datasets stop behavior cloning rather than being relabelled as expert success.

## Checkpoint-Only Evaluation

Evaluate the student from sensors and its learned recurrent state, without
loading demonstrations or references. Restore the renderer and GPU settings
explicitly, including in a fresh SSH session:

```bash
set -euo pipefail
cd /workspace/script_free_source_20261013_v2/r2dreamer
export MUJOCO_GL="$(< .mujoco_gl)"
if [[ "$MUJOCO_GL" == "osmesa" ]]; then
  export PYOPENGL_PLATFORM=osmesa
else
  unset PYOPENGL_PLATFORM
fi
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=gpu
.venv/bin/python scripts/evaluate_script_free.py \
  --checkpoint /workspace/script_free_pilot_v2_seed0/pilot/online/latest.pt \
  --output /workspace/script_free_eval_v2_seed1000 \
  --seed 1000 --seconds 20 --worlds 32
```

The evaluator rejects controller/physical contract mismatches and poisons
trajectory loaders to expose accidental scripted inference. It saves raw trials,
MP4s and `script_free_evaluation.json`. A declared longer follow-up can use
`--seconds 60` with a new output path; the report records the evaluation horizon
extension. Completion is not automatic physical qualification.

## Stop, Inspect And Back Up

For an intentional stop, press Ctrl-C in the attached tmux session and wait for
the command to return. The supervisor requests graceful child-group termination,
waits up to 120 seconds, then force-stops the owned process group if necessary.
Inspect from the extracted `r2dreamer` directory:

```bash
.venv/bin/python scripts/show_script_free_status.py --run /workspace/script_free_pilot_v2_seed0
```

An intentional stop should show `status: interrupted` and `child_pid: null`.
Completed/failed/interrupted runs can be backed up; a running or stale crash
status is refused. SIGKILL, lost power and host failure can leave stale status
and cannot guarantee a new checkpoint. Inspect actual checkpoint contents and
saved steps before reuse, not only filenames or the last console step.

```bash
.venv/bin/python scripts/backup_script_free_run.py \
  --run /workspace/script_free_pilot_v2_seed0 \
  --output /workspace/script_free_pilot_v2_seed0_backup.tar.gz
cd /workspace
sha256sum -c script_free_pilot_v2_seed0_backup.tar.gz.sha256
```

The fresh archive contains per-file hashes and a verified archive SHA-256
sidecar. Download both from a local PowerShell window using the same SSH values:

```powershell
$LocalBackupDir = 'D:\Final Project\script_free_v2_downloaded_evidence'
New-Item -ItemType Directory -Force -Path $LocalBackupDir | Out-Null
scp -P $GpuPort "${GpuUser}@${GpuHost}:/workspace/script_free_pilot_v2_seed0_backup.tar.gz" $LocalBackupDir
scp -P $GpuPort "${GpuUser}@${GpuHost}:/workspace/script_free_pilot_v2_seed0_backup.tar.gz.sha256" $LocalBackupDir
$Archive = Join-Path $LocalBackupDir 'script_free_pilot_v2_seed0_backup.tar.gz'
$Expected = ((Get-Content -LiteralPath ($Archive + '.sha256') -Raw).Trim() -split '\s+')[0]
$Actual = (Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLowerInvariant()
if ($Actual -ne $Expected) { throw 'Downloaded archive hash mismatch; preserve the instance evidence' }
```

Keep the source ZIP, setup log and `.setup_host_report.json` with the run backup.
Backup is artifact recovery, not an implemented automatic resume: replay,
simulator state and random-number state are not restored for exact continuation.
No next stage is automatically retried or extended. Do not delete the rented
instance until downloaded hashes and checkpoint readback are verified; stopping
the pipeline does not stop instance billing.

Judge standing-start initiation, supported left and right lift/landing, ordered
alternation, falls/walls/stalls, body tilt, foot clearance, contact loads, scrub,
displacement, slew/torque saturation and videos. The strict detector requires
15 mm lift. Physical motion that appears to step can disagree with that detector;
report both contact/video evidence and strict counts. Do not lower the threshold
to manufacture a walking result. Reward, dot hits, forward displacement or
optimizer updates cannot substitute for supported bilateral walking.

Stop or revise an unchanged extension when matched trials still fall before
one second with zero supported landings, or gait outcomes deteriorate despite
increasing reward. A promising student needs repeatable standing-start bilateral
walking without the external expert, held-out seed/duration checks and multiple
training seeds before robust or hardware claims. Servo stiffness, dynamics and
sim-to-real remain unmeasured assumptions. This route supplies a concrete,
reviewable experiment; it does not guarantee that the policy will learn walking.
