# Preparation verification and debugging report

Status: **DONE_WITH_CONCERNS**. Local preparation complete; Linux CUDA/JAX
execution and physical improvement remain unverified.

| Verification | Result | Scope |
|---|---|---|
| `scripts/test_trainer_runtime.py` | 10 tests passed | CPU Torch2.8/TorchRL0.9.2; actual replay and checkpoint files, mocked robot |
| `scripts/test_training_contract.py` | 11 tests passed | Resume rejection/adaptation, asset/pool identity, relocation/world scaling |
| `scripts/check_mjx_evaluation_schedule.py` | 16 checks passed | Actual MuJoCo3.14 pose pool, schedule and mocked evaluation lifecycle |
| `scripts/check_teacher_capacity_probe.py` | 17 checks passed | Admission, shutdown reason, malformed evidence, timeout/overflow/mock processes |
| `scripts/test_teacher_capacity_probe.py` | 2 tests passed | Budget/replay scaling and rejected partial run artifacts |
| Pose structural preflight | Passed | Four preserved exploratory poses accepted; `physical_hold_validated:false` |
| Hydra recipes/probe overrides | Passed | Next recipe,64/128/256 capacity budgets,20 evaluation worlds |
| Python compilation and Bash syntax | Passed | Active source and next launcher |
| Independent implementation review | No remaining launch blockers found | Contract, replay caching, shutdown, launcher/preflight agreement |

The final review fixed an external signal arriving during the update that
reaches the optimizer cap: SIGTERM now retains priority and skips a final
evaluation rather than being relabelled as an ordinary capacity completion.
The capacity gate separately rejects signal stops even if their exit status
is zero. Legacy random evaluation remains non-repeatable unless the balanced
protocol is explicitly enabled.

## Debug report

Symptom: 14M teacher optimized weights but all88 trials fell with zero valid
steps; stopping earlier trainers lost unsaved work; scaling probes overstated
training readiness.

Root cause: No isolated causal explanation for the robot's falls is established.
Verified software/process problems were missing physical-contract validation,
changing evaluation mixtures, missing cooperative final save, repeated replay
boundary scanning, per-world scalar synchronization and misleading FPS units.

Fix: `training_contract.py` plus `train.py`/`trainer.py` load/save hooks;
balanced evaluation in `envs/mjx_teacher_env.py` and reset pool helpers;
bounded update stop and phase timing in `trainer.py`; replay caching in
`buffer.py`; complete learner checks in `scripts/teacher_capacity_probe.py`.

Evidence: Passing tests above exercise saved optimizer/weights/counters,
failed-write preservation, signal priority, cached/uncached sample parity
across ring wrap/refill, and fail-closed contract/capacity checks. Earlier
14M outcomes are independently documented in the evidence audit.

Related: Project lessonsL01/L02/L05/L19-L21/L28/L34/L40/L41 and newL42-L44.
Root workspace owns active teacher files; maintained repository owns memory.
Existing unrelated edits and historical checkpoint archives were preserved.

Remaining gate: No NVIDIA CUDA device is available through local Torch
(`2.8.0+cpu`, `cuda.is_available()==False`) and local test environments lack
JAX. GPU preflight explicitly refuses CPU fallback. The source contains
GPU checks; their existence is not evidence they passed.
