# Next teacher session: bounded, measurable, recoverable

Prepared 4 October 2026. **Local preparation, not GPU qualification or a walking result.**
Use `walking_mjx_teacher_next`; archived recipes retain their historical meaning.
No cloud machine was rented or training launched during this preparation.

## Why this differs from the failed runs

The [primary-artifact audit](TRAINING_EVIDENCE_20261004.md) found 88/88 falls,
zero valid steps, and survival falling from 0.900 to 0.483 seconds in the 14M
run. The preceding fresh pilot also failed. The 48-degree checkpoint was
adapted to 90 degrees, different torque, reward and resets simultaneously.
More optimizer updates cannot identify which change caused the failure.

This candidate preserves 25% each of your four training poses, nominal
90-degree authority, 2 rad/s command slew, 18 kg-cm torque, the 2 mm scene,
45+98 observations, 19 actions, and `step_progress_v1`. Falling poses remain
an explicit experiment. No collision/contact thresholds were weakened.

The single learning ablation is replay ratio **16 versus 64**. Hypothesis:
fewer updates per collected transition reduce repeated optimization on short
failed trajectories and improve collection per wall-clock hour. This is not
a demonstrated remedy. The actor, world model and optimizer start fresh in
both arms, avoiding the old changed-action continuation. A matched ratio64
control uses the same new runtime/evaluation code. Compare at the same valid
transition budget and report elapsed time separately.

Other changes improve reliability and measurement:

- Fixed evaluation schedule: standing plus each saved pose, four trials each,
  at the same seed and eight-second limit every time. Manual poses repeat
  exactly; these are not 20 independent randomized initial conditions.
- Cached replay trajectory boundaries between writes, optional omission of
  unused replay logging fields, fewer per-world host scalar reads, and no
  unused evaluation transition clones when prediction video is disabled.
- Cooperative SIGINT/SIGTERM checkpointing at a completed collection/update
  boundary, atomic `latest.pt`, and physical-contract metadata/validation.
  SIGKILL/power loss cannot save in-memory state. Replay still refills on resume.
- Explicit control transitions/s, optimizer updates/s and phase timings.
  Old ~281 FPS meant ~28 control transitions/s because action_repeat=10.
- A complete-learner capacity probe with an exact optimizer-update cap,
  initial/final balanced evaluation, finite checkpoint readback and SHA-256.

## Instance and setup

Start with **one Linux NVIDIA GPU with at least 24 GiB VRAM**, Python3.11,
adequate host RAM and disk for the checkpoints. This is a starting envelope
based on the previous 64-world 4090 run, not a purchase recommendation or a
guarantee. Prefer more VRAM if cost is acceptable; world count must be measured.
Do not rent on a promised FPS figure. Confirm usable CUDA before package work;
NVML/nvidia-smi visibility alone did not establish CUDA health on a prior host.

After unpacking the new source ZIP into a fresh directory:

```bash
conda create -n teacher-next python=3.11 -y
conda activate teacher-next
python -m pip install -r r2dreamer/requirements-mjx-teacher-linux.txt
python -m pip check
python -c "import torch,jax; assert torch.cuda.is_available(); assert jax.default_backend()=='gpu'; print(torch.cuda.get_device_name(), jax.devices())"
python -m pip freeze > runtime-freeze.txt
```

Install the requirements file directly. Do not install the legacy pyproject
walking extra. JAX remains host-resolved in the existing requirement file;
the resolved versions must be retained with the run. This machine has CPU
PyTorch and no JAX, so a fresh Linux dependency resolution is untested here.

## First run: capacity, then the controlled pilot

From the unpacked bundle root, first run the short CUDA bridge checks:

```bash
TEACHER_PREFLIGHT_ONLY=1 bash r2dreamer/scripts/deploy/start_teacher_next.sh
cd r2dreamer
python scripts/teacher_capacity_probe.py --worlds 64 128 256 --updates 32 --output /workspace/teacher_capacity_20261004
```

Each case runs in a new process with training+evaluation environments, the
actual model, GPU replay, and 32 verified optimizer updates. The sweep stops
at its first failure. Logs and `capacity_report.json` stay in the output
directory. Each case has a 20-minute wall timeout. A pass is **short capacity
only**, not proof against a later OOM, contact overflow, or failed walking.
The initial JIT and replay warmup dominate such a short test; use phase timings
and a longer bounded pilot for steady throughput comparisons. Do not select
the biggest world count merely because it fits.

Keep the first learning comparison at 64 worlds, so robot count is not a second
learning intervention. Run these sequentially, each in its own fresh directory:

```bash
TEACHER_RUN_DIR=/workspace/teacher_ratio16_seed0 bash scripts/deploy/start_teacher_next.sh seed=0
TEACHER_RUN_DIR=/workspace/teacher_ratio64_seed0 bash scripts/deploy/start_teacher_next.sh seed=0 env.train_ratio=64
```

Each arm stops at 200,000 counted ticks (about20,000 active control transitions,
with bounded vector-step overshoot). Initial, 50k, 100k, 150k and final physical
evaluations preserve raw per-pose trials. Capacity checks use separate fresh
policies and are not learning comparison results. For any later world-count
change, set replay capacity to at least128*worlds and budget/deadline beyond
67*worlds*10 counted ticks; the probe computes these automatically.

This is an early screening budget. After replay warmup the ratio16 arm gets
only about245 updates, still inside the inherited1000-update learning-rate
warmup; the ratio64 arm also gets a different optimizer schedule position.
That is part of comparing compute spent per collected transition. It cannot
establish final algorithm quality. A later learning comparison needs a declared
budget beyond warmup and multiple seeds; do not infer that ratio16 is inferior
merely because it has not walked at200k. The immediate stop rule below rejects
an unchanged expensive extension while its physical prerequisite is absent.

Legacy12M/14M checkpoints lack contract metadata. Direct `train.py` resume
requires explicit `allow_unverified_resume=true` with this recipe (use the
Hydra `+` prefix only for recipes where the key is absent), after verifying
the archived physical settings. That exception cannot bypass a known contract
mismatch. `init_from`/`transfer_from` are explicit adaptations with fresh
optimizer state, not equivalent continuation. The fixed pilot launcher accepts
only seed and ratio overrides so its preflight matches the launched recipe.

To stop a running pilot, send SIGTERM to its verified trainer PID or press
Ctrl+C once in its foreground terminal. Wait for the checkpoint save message,
then read/hash `latest.pt` before transferring or destroying an instance.
Stopping the trainer does not stop cloud billing.

## Declare success and failure before looking at rewards

- At50k/100k, inspect each pose separately: survival, falls, supported lift,
  left/right steps, alternation, scrub, body tilt, slew and torque saturation.
  If all starts still fall before one second and no supported landing appears,
  stop/revise the pilot; do not extend the unchanged recipe to millions.
- Runtime pass: finite changed weights, requested updates, matching final
  evaluation/checkpoint, no overflow/OOM. This is independent of gait progress.
- A promising pilot must improve matched physical outcomes and exhibit genuine
  supported lift/landing. Ratio16 is not preferred merely for higher FPS.
- Walking promotion needs bilateral valid steps and ordered alternation,
  no falls over the full eight-second evaluation, plus a separate original
  standing-start CPU validation and video/contact inspection. Repeated manual
  pose starts cannot establish walking initiation. Repeat promising arms with
  additional training seeds before claiming algorithmic improvement.

If both arms still fail, next work is a physically validated balance/transfer/
lift/lower reference, not a bigger network or more parallel falling robots.
The existing standing task is not a drop-in recovery mode: it rejects the
manual-pose RSI contract and has a different objective. Build and qualify such
a curriculum separately before using it as a new teacher.

## GPU scope and deferred optimizations

Physics/task features/reward are JAX+MJX-Warp; policy/world-model optimization
and replay storage are CUDA Torch, with DLPack at the boundary. Host work still
exists for orchestration, episode metadata, sampling decisions, files and
optional video. This is not literally an all-GPU program. No speedup was
measured on this CPU-only preparation host.

The bridge still computes batched resets each collection call. A conditional
around a whole batched reset may save work but is deferred until a GPU test
checks none/all/mixed reset masks against the current path. The prior
per-world conditional failed MJX-Warp custom batching. Compilation, mixed
precision, smaller contact buffers and solver changes remain separate measured
experiments, not untested defaults.

[MuJoCo's performance guidance](https://mujoco.readthedocs.io/en/latest/mjwarp/#performance-tuning)
identifies contact/constraint buffers as performance and memory factors and
warns against reducing solver iterations enough to harm physics.
[JAX's GPU memory guidance](https://docs.jax.dev/en/latest/gpu_memory_allocation.html)
documents preallocation competition; launchers disable JAX preallocation so
Torch and JAX can share the device. These documents explain mechanisms, not
this robot's achievable FPS. The pinned TorchRL0.9 sampler exposes cache
behavior in its [official API](https://docs.pytorch.org/rl/0.9/reference/generated/torchrl.data.replay_buffers.SliceSampler.html).

## Local verification and remaining gate

See `../../dist/next_training_20261004/` for local validation records and the
source ZIP manifest. CPU tests cover schedule construction/lifecycle, replay
cache behavior, stop/checkpoint boundaries, capacity admission and contract
checks. Hydra composition, Bash syntax, Python compilation and ZIP entry
hashes are checked locally. Actual CUDA/JAX execution, device memory headroom,
full-run speed and improved physical behavior require the rented instance.
