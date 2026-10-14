# Sequential teacher curriculum, 4 October 2026

User-approved schedule: one teacher, six sequential phases, continuing from
the completed 128-world 1M run. These are additional budgets, not six independent
models to merge. All phases use128 training worlds and the current19-action,
45+98-observation,18kg-cm torque,90-degree nominal command contract.

| Phase | Objective and training reset | Additional counted ticks |
| --- | --- | ---: |
| 1 | Actual standing objective, normal standing resets | 1,000,000 |
| 2 | Walking objective, saved pose1 only | 1,000,000 |
| 3 | Walking objective, saved pose2 only | 1,000,000 |
| 4 | Walking objective, saved pose3 only | 1,000,000 |
| 5 | Walking objective, saved pose4 only | 1,000,000 |
| 6 | Walking objective, standing and all four poses,20% each | 3,000,000 |

Total8M counted ticks, about800,000 active control transitions across all
robots. Each phase may overshoot by fewer than1280 counted ticks because
collection is batched. The preceding1M experiment is not included in8M.

`init_from` carries the complete agent state, including policy and world model,
between phases. `transfer_from` would discard the policy and is unsuitable.
Optimizer, replay and local step counters restart at each phase. The inherited
1000-update learning-rate warmup also restarts;1M phases only modestly exceed it
at ratio16. This is an experimental curriculum, not an exact uninterrupted
optimizer trajectory or a promise of faster gait acquisition. Isolated poses
can cause forgetting; the mixed phase and repeated evaluation expose that risk.

Standing uses `mjx_task=standing`, with no pose RSI and20 standing evaluation
trials. Walking phases evaluate standing and each saved pose four times every
100k counted ticks. Exact `training_pose_id` selection resolves the ambiguity
that poses1/2 share the original fourth reset category. Evaluation keeps the
complete pose pool. Standing stable time is logged as increments, not repeatedly
summed cumulative time.

Video is off during collection and recorded only during each phase's final
evaluation. One standing clip is saved for phase1; subsequent phases save one
clip for each of the five starts, stopping each at its first terminal frame.
320x240,25fps clips use every second50Hz control frame. Rendering selected
worlds adds phase-end wall time; policy observations still contain no images.
MP4 sidecars record pose ID, world index, training step and frame count.

The supervisor writes `curriculum.json` with checkpoint parents/hashes, phase
budgets, physical outcomes and video paths. It stops on a child crash, signal,
incomplete budget, missing/mismatched evaluation, nonfinite checkpoint or missing
video. Completing a phase budget is not physical qualification. Each phase
inherits the previous final `latest.pt`; `best.pt` is also retained for review.
Previously failed outcomes are not relabelled as success.

Run on the prepared GPU host from the isolated source copy:

```bash
python -u scripts/run_teacher_curriculum.py \
  --init-from /workspace/teacher_128w_1m_2onsmx/latest.pt \
  --output /workspace/teacher_curriculum_8m_20261004
```

The output directory must not already exist. To inspect the six commands
without launching, append `--dry-run`. Press Ctrl+C once to request a cooperative
trainer save; no subsequent phase starts after interruption. Replay is not
checkpointed. Resuming an interrupted curriculum requires selecting its actual
saved phase and remaining budget, not rerunning this launcher over the same path.

## Measured prerequisites

The original128-world run finished at1,000,280 counted ticks and1434optimizer
updates. Browser-observed final evaluation:0valid steps, meanfall count1,
mean survival0.504s. This is the requested adaptation source, not a walking teacher.

A bounded20-world GPU reset diagnostic on the original runtime found exact
qpos/qvel/ctrl/time/pose IDs/RNG and unchanged retained DLPack observations. The
strict1e-6 observation assertion failed on derived sensor variation: maxIMU
acceleration2.1457672e-5, onboard acceleration1.7523766e-5, filteredFSR3.8743019e-6,
normalized footload1.0505319e-6; no reported overflow. All other groups were
exact except normalizedFSR4.3958426e-7. These are browser-observed diagnostic
values; the full report remains `/workspace/teacher_reset_diagnostic_20261004.json`.
The revised preflight must retain exact reset-input/alias checks and bound
derived channels separately before admitting the new source. It does not change
simulator physics or the walking detector.

CPU tests and live GPU qualification results are recorded in project memory;
source preparation alone does not establish GPU or video success.

Fresh-process follow-up: filteredFSR reached1.104338e-5 and IMU/onboard
acceleration1.20163e-4/1.25885e-4m/s², while exact reset-input and retained-view
checks still passed. The final bounded checks use acceleration1e-3m/s²
(about0.0001g), filteredFSR1e-4, teacher normalizedload/FSR1e-5, and other
observation channels1e-6. FilteredFSR=tanh(raw/.5) has at most10times the
sensitivity of teacherFSR=raw/5. Exact qpos/qvel/ctrl/time/pose/RNG checks remain
zero-tolerance. These bounds are declared numerical acceptance limits, not
bitwise determinism or changed physics. Tests reject excesses, aliasing,
nonfinite observations and any control-state difference.
