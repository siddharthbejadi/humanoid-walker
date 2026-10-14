# Reference-assisted policy pilot, 12 October 2026

This package prepares the next experiment. It does not contain a trained
walking policy, a rented instance, or a passed CUDA readiness report. The
latest nearby archived evaluations still show zero valid alternating steps.
The robot already has a humanlike joint reference and a tilt stabilizer;
successful motion with those controllers must not be credited to learning.

## Exact candidate

Use `walking_mjx_reference_pilot`, not the older vendor-residual YAML or
automatic two-stage launcher. Its contract is:

| Setting | Value |
| --- | --- |
| Scene | `humanoid_rubber_2mm_servo_v2_kp40.xml`, kp 40, kv 0.6 |
| Peak torque / command slew | 18 kg-cm = 1.765197 N m / 8 rad/s |
| Reference / feedback | `humanlike_gait.npz` + `[1, 0.04, -1, 0.25]` tilt stabilizer |
| Learned residual | 19 channels, +/- 0.1 rad around reference + stabilizer |
| Actor std / learning-rate warmup | 0.005-0.03 / 100 optimizer updates |
| Step detector | 15 mm lift; supported lift/landing and ordered alternation |
| Training starts | 15% standing, 85% sampled scripted phases |
| Held-out evaluation | 32 repeatable standing starts; configured seed 1000, effective bridge seed 11000 |
| Pilot budget | 128 worlds, 1M counted ticks = about 100k active control transitions |
| Replay / train ratio | 131072 transitions / 64 |
| Evaluation / video | Every 100k counted ticks; 20 s horizon; MP4 + raw trials |
| Checkpoint preference | Safety failures first, then valid steps, then ordered alternation |

The kp40 scene is a simulation candidate. It is not measured hardware servo
stiffness. The recipe initializes policy, model and optimizer fresh: all
resume/init/transfer sources are null. Do not import the old soft-servo
checkpoints. Evaluation starts are held out by seed and start distribution;
they are diagnostics of one training seed, not independent training replicates.

Scripted phase states are normal reference starts, not live pose injections.
Their exported clock is the state after the previous control; the next target
advances consistently. Stabilizer history resets to current tilt and zero
filtered rate. That is a declared fresh-controller start, not an exact replay
of unknown CPU feedback history. No manual-pose hold test is required or used.

## After renting a GPU

Unpack the new ZIP into a fresh directory. Use the bundled setup script and
existing pinned Linux runtime. Its servo preflight is a setup smoke; the
reference-specific checks below are still required. Record resolved runtime
versions and verify actual Torch CUDA and JAX GPU before interpreting results.

Plan for one Linux NVIDIA GPU with at least 32 GB VRAM, roughly 64 GB host RAM
and 100 GB disk headroom. These are planning estimates for this candidate, not
measured minimums or a capacity guarantee. Prefer a recent driver such as
580 or newer as a conservative starting point for CUDA 12 wheels and possible
PTX compilation; the actual pinned runtime must pass the checks. NVIDIA's
[CUDA compatibility guidance](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
describes why runtime/toolkit support and PTX behavior require more than a
driver version number alone.

Readiness validates Python 3.11 and the pinned core Torch 2.8.0, TorchRL 0.9.2,
TensorDict 0.9.1, MuJoCo/MJX 3.14.0, JAX/JAXlib 0.10.2, Warp 1.17.0 and NumPy
2.4.6. The Torch CUDA wheel's local tag, such as `+cu128`, is allowed; its
working CUDA build must be major version 12. GPU baseline evidence records
the resolved versions, Torch GPU name/VRAM, JAX devices and `nvidia-smi` driver
information when available. A missing optional `nvidia-smi` command is recorded
explicitly, while missing Torch CUDA or JAX GPU fails admission.

```bash
python3 r2dreamer/scripts/verify_reference_bundle.py
bash r2dreamer/scripts/deploy/vast_setup.sh
cd r2dreamer
.venv/bin/python -m pip freeze > runtime-freeze.txt
REFERENCE_PREFLIGHT_ONLY=1 \
TEACHER_RUN_DIR=/workspace/reference_readiness_seed0 \
bash scripts/deploy/start_reference_pilot.sh seed=0
```

Run source verification before installing the runtime. The verifier uses only
the Python standard library and supports a system Python 3.8 or newer; the
learner remains pinned to Python 3.11. It checks every source/reference entry
against `SOURCE_MANIFEST.json` and rejects changed bytes or paths outside the
extracted bundle. `START_HERE.md` at the archive root is a copy of this guide.
The pilot launcher requires the manifest and repeats verification before any
package, CUDA, baseline or capacity checks. Launching directly from an unpackaged
working copy is rejected: package the reviewed source first.

The launcher refuses existing output paths. Readiness uses two fresh
subdirectories and never loads a checkpoint:

1. Structural checks compose the exact recipe, verify scene/mesh/reference
   closure, finite NPZ fields, controller settings and replay admission.
2. GPU diagnostics test zero learned residual with the reference/stabilizer
   from standing and 32 fixed phase indices. They retain numerical reset and
   DLPack view checks, exact reference history, raw metrics, phase IDs, causes
   of termination, and one MP4 per start group. Check the exported reference
   clock and confirm scripted starts never enter the injected-pose path.
3. A fresh `train.py` subprocess constructs the actual 128-world learner,
   GPU replay and all 32 evaluation worlds, verifies exactly 32 optimizer
   updates, changed finite weights, matching final evaluation/checkpoint,
   MP4 rendering and SHA-256 read-back. It has a 20-minute wall timeout.

Capacity video admission requires all 32 body-view MP4s from the exact final
checkpoint step, including world 0 and worlds 1-31. Each clip must be nonempty
and have a JSON sidecar with the same final training step, evaluation-world
index, standing pose ID 0, a positive integer frame count, and 25 fps. Initial
clips, a world-0-only clip, camera-only clips, missing sidecars or mismatched
metadata cannot admit the pilot. This gate verifies file identity and metadata;
the clips still need visual/contact review before a walking claim.

Runtime, aliasing/reset errors, overflow, nonfinite values, missing evidence,
rendering failure, timeout or incomplete update counts block the pilot.
Poor physical reference outcomes are retained as experimental evidence; zero
alternation is not reclassified as a CUDA failure, and no readiness status
claims teacher qualification. In particular, inspect left-stance support and
whether the robot falls before the first alternating landing.

The preflight reports have separate `physical_reference_qualified` fields.
They must remain false when supported bilateral walking is absent. A 32-update
capacity pass does not prove that a long run will fit memory or learn walking.

## Bounded learning

After reviewing readiness, choose fresh paths and explicitly enable the pilot:

```bash
REFERENCE_PREFLIGHT_ONLY=0 \
TEACHER_RUN_DIR=/workspace/reference_pilot_seed0 \
bash scripts/deploy/start_reference_pilot.sh seed=0
```

This repeats the readiness checks for the exact fresh run contract, then
launches only the 1M-tick pilot. It does not automatically extend to 20M,
change physics, import a standing policy, or weaken the 15 mm threshold.
Before any extension inspect initial/100k/200k and subsequent matched trials:
alive seconds, physical falls, wall/stall/goal causes, valid left/right steps,
ordered alternation, clearance, opposite-foot load, scrub, tilt, target
tracking, slew/torque saturation, real updates/lr and control transitions/s.

Compare the learned controller with zero residual under the same reference
and stabilizer. Increasing return, landing reward, dot count, displacement or
optimizer updates alone does not establish walking. Safety-first `best.pt`
can be a safe stationary policy: inspect all task metrics and `latest.pt`
before deciding what improved. Stop/revise an unchanged extension if the
policy still falls before one second with zero supported landings, or if
matched gait outcomes deteriorate despite increasing reward.

Promotion requires standing-start initiation, supported left and right
landings in order, no physical falls/wall failures over the declared horizon,
and video/contact confirmation that progress is not sliding or diving.
Reproduce a promising policy with additional training seeds and a longer
held-out evaluation before sim-to-real work. Reference or stabilizer removal
is a separate controller ablation; report assisted walking explicitly until
those dependencies are measured.

## Package verification

`scripts/package_reference_training.py` creates a new ZIP, an internal and
external SHA-256 manifest, and an archive digest sidecar. It includes the
teacher code, configs, scene meshes, gait NPZs and runbook, and excludes virtual
environments, logs, checkpoints, generated sensor scenes and stale package
metadata. Previous ZIPs are preserved. A hash-verified package establishes
source identity and closure only; GPU admission is performed after rental.

Local preparation checks cover Hydra composition/asset closure, fresh-init
rejection, controller/detector admission, pinned-version mismatch rejection,
and malformed/nonfinite capacity evidence. They do not run CUDA or validate
the physical behavior of the next policy.
