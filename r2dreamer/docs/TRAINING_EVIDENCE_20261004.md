# Teacher training evidence audit, 4 October 2026

## Conclusion

The latest saved teacher performed numerical optimization but did not learn
walking. The 14M adaptation run recorded **88 falls in 88 evaluation trials,
zero valid steps, and zero touchdown/alternation reward**. Every reset-pose
category failed. Its preceding fresh 200k pilot also recorded 40/40 falls and
zero valid steps. Repeating either setup for longer is unsupported by these
results.

This audit reads the SHA-verified local backup and actual deployed source. No
cloud machine was contacted, rented, or changed; no checkpoint was modified.
Project `graph.py recall` returned `neo4j_unavailable_using_local_fallback`.
Relevant lessons are L01/L02 (reward and more samples do not establish walking),
L05 (contracts), L07 (resume state), L20 (full-trainer memory), L37/L38 (reward
and landing events), and L40/L41 (unqualified poses and per-pose evaluation).

## Primary evidence and identities

Paths below are relative to the teacher bundle root, the parent of `r2dreamer`.

| Evidence | Location |
|---|---|
| Backup and destruction record | `dist/instance54061209_14m_backup_20261004/BACKUP_MANIFEST.md` |
| Latest run | `dist/instance54061209_14m_backup_20261004/mjx_teacher_resume_to_14m_QmOwfE/` |
| Fresh 200k pilot | `dist/instance54061209_14m_backup_20261004/mjx_teacher_90deg_mixed_64w_200k_ikhyHy/` |
| Latest deployed source | `dist/instance54061209_14m_backup_20261004/teacher90_resume_bundle_vtSwIV/r2dreamer/` |
| Older standing run | `dist/paused_12001780_20261003/` |

For each run the primary outcome source is `walking_evaluations.jsonl`, with
optimizer/throughput data in `metrics.jsonl` and settings in `.hydra/config.yaml`
(the older selected backup uses `config.yaml`). The latest `launcher.log`
identifies an RTX 4090 with 24 GiB, CUDA toolkit 12.9, Warp 1.17.0 and 9,639,717
optimizer parameters. This is archived runtime evidence, not live GPU status.

Local read-back with `torch.load(..., map_location="cpu", weights_only=True)`
verified the following checkpoints and hashes again during this audit:

| Checkpoint | Counted step | Optimizer updates | SHA-256 |
|---|---:|---:|---|
| Starting `step_12001780.pt` | 12,001,780 | 73,188 | `51991d80b7a78b7c0a9466acf252d397c6459db3400ceccbd359265566740681` |
| Final `latest.pt` | 14,000,220 | 85,419 | `44bb0a233bbc3e757fb955f92fd8ba3d3a98b4b11a9226d36f0a9c89f695349a` |
| Run `best.pt` | 12,001,780 | 73,188 | `a5770569b66f9b1ded93edb653a6ee466e50e6731a4d367bf597a0c7d55af56b` |

All 160 final agent tensors changed from the starting checkpoint; none contains
nonfinite values. `best.pt` has exactly the starting agent tensors. It was
selected under the adaptation run's initial evaluation and does not certify
the older standing result under the new settings. Replay is absent from the
checkpoint: deployed `trainer.py:424` and `train.py:125` preserve model,
optimizer and counters, then refill replay.

## Latest physical outcomes

The 11 evaluations span counted steps 12,001,780 through 14,000,220, with eight
trials per evaluation and a 400-control-step / eight-second limit.

| Metric, mean per trial | Initial evaluation | Final evaluation |
|---|---:|---:|
| Survival, seconds | 0.9000 | 0.4825 |
| Falls / trials | 8 / 8 | 8 / 8 |
| Valid left/right/alternating steps | 0 / 0 / 0 | 0 / 0 / 0 |
| Body tilt, radians | 0.3840 | 0.5531 |
| Scrub distance, metres | 0.04520 | 0.01476 |
| Single-support time, seconds | 0.1500 | 0.1000 |
| Command slew-limited fraction | 0.7230 | 0.8632 |
| Actuator torque-saturation fraction | 0.03427 | 0.01319 |
| Tracking error, radians | 0.10470 | 0.07181 |
| Touchdown / alternation reward | 0 / 0 | 0 / 0 |
| Lift-progress reward | 0.32845 | 0.38273 |

The lower scrub and tracking error do not demonstrate improved gait: episodes
are shorter, starts differ, and every trial still falls. The final eight starts
omit poses 1 and 2. Lift-progress credit can improve without completing a step.

All-evaluation results grouped by recorded `start_pose_id`:

| Start | Trials | Mean survival, s | Mean tilt, rad | Mean scrub, m | Mean single support, s | Mean slew-limited fraction |
|---|---:|---:|---:|---:|---:|---:|
| 0, standing | 28 | 0.5936 | 0.4632 | 0.04567 | 0.0086 | 0.7818 |
| 1, manual pose 1 | 17 | 0.7012 | 0.4375 | 0.04219 | 0.0094 | 0.7071 |
| 2, manual pose 2 | 9 | 0.6867 | 0.3748 | 0.03965 | 0.0778 | 0.7362 |
| 3, manual pose 3 | 13 | 0.5954 | 0.3700 | 0.02633 | 0.2323 | 0.7896 |
| 4, manual pose 4 | 21 | 0.6629 | 0.5045 | 0.01464 | 0.2324 | 0.7413 |

Every row has a 100% fall rate and zero valid steps. Even within the standing
category, the initial two trials averaged 1.26 seconds and the final two
averaged 0.41 seconds. These are small samples with uncontrolled reset noise,
not a matched statistical causal estimate. The 88 trials are evaluations of
one training run, not 88 independent training replications.

## What changed from the older standing run

The saved config comparison finds simultaneous changes to:

- Nominal action amplitude: 0.837758 rad (48 degrees) to 1.570796 rad (90
  degrees), a 1.875-fold change before physical range clipping and slew limits.
- Motor torque: historical 1.08 N m scene to the generated 18 kg-cm / 1.765197
  N m scene. This remains a peak-motor model assumption.
- Reward: the older simplified legacy profile to `step_progress_v1`.
- Training resets: standing-only (`rsi_stratified: false`) to 25% each of four
  manually saved poses, with zero standing probability and
  `pose_rsi_allow_falls: true`.
- Evaluation resets: standing-only to 20% each of standing and four poses.
- Evaluation cadence: every 100k to every 200k counted steps.

The same actor, world model, reward/value heads and optimizer were resumed
with all these changes. This is an adaptation experiment, not an unchanged
continuation. Deployed `envs/mjx_walking/env.py:405` maps actions to
`standing_ctrl + action_scale * action` before clipping, so action shape parity
does not preserve physical commands. At the unchanged 2 rad/s slew limit and
0.02-second control period, controls move at most 0.04 rad per decision.

The old last evaluation at step 12,001,780 had 5/8 eight-second survivors,
mean survival 6.080 seconds and mean tilt 0.1762 rad, but still zero valid steps
and no right-foot peak clearance. Earlier in that old log, seven trials at
9,801,820 counted one right-side event each; all seven fell after 1.22-2.08
seconds, with no alternation. These isolated events under the older detector
do not establish sustained walking. Historical missing scrub fields must be
reported as unavailable, never zero.

The fresh 200k pilot has five evaluations / 40 trials, all falling with zero
steps. Its mean survival was 0.6625 seconds initially and 0.5425 seconds
finally. Therefore resetting the optimizer or starting fresh alone has not
already solved the changed task.

## Optimizer and throughput interpretation

All 1,957 logged `train/opt/loss` values in the 14M run are finite. The loss
went from 14.77366 to 0.74134; its range includes negative values because it
combines actor and model objectives. This establishes optimization, not policy
competence. Updates increased by 12,231.

The run uses 64 worlds, CUDA policy and replay, batch size 16, batch length 64,
replay capacity 8192, training ratio 64, with compilation and mixed precision
off. Deployed `trainer.py:518` counts active transitions times
`action_repeat=10`; `tools.py:212` derives FPS from that counter and wall time.
Its 1,957 nonzero FPS samples have median **280.704 counted simulation ticks/s**
and mean 285.934; median over the final 500 is 278.738. Dividing by ten gives
approximately **28.070 counted control transitions/s**. This includes learning
work and logging, not only GPU physics. Interval means/medians are not a
substitute for total-run elapsed-time throughput.

The additional 1,998,440 counted ticks represent 199,844 active transitions,
not two million policy decisions. Raising action repeat or parallel world
count can make counters look larger without improving useful learning speed.
The deployed bridge already runs batched MJX-Warp and uses Torch/JAX DLPack;
"convert everything to GPU" should target remaining host synchronization,
logging and reset overhead rather than replace an already-GPU simulator.
The hot loop's per-world Python `done` handling at `trainer.py:507` and
`int((~done).sum())` are concrete host synchronization points, but their cost
has not been separately profiled here.

## Prioritized next experiment

These are recommendations and hypotheses, not verified future improvements.

1. **Preserve command meaning and recovery state.** Add an explicit physical
   contract to checkpoints and reject accidental action/reward/scene/reset
   mismatches. Compare the old checkpoint under its original contract with a
   separate reset-optimizer adaptation only after an identical no-update
   rollout. Do not silently rescale or relabel the 14M checkpoint as walking.
2. **Establish achievable balance and bilateral motion.** The archived CPU
   pose audit in the bundle README says the four saved controls fell in
   0.96/0.80/0.65/0.60 seconds at 18 kg-cm. First validate contact, action
   reachability, eight-second free-base support, then either-side transfer,
   lift and stable landing. Keep unqualified poses experimental; a pose's
   presence in JSON is not a successful demonstration. Increasing torque or
   increasing reward weights does not itself repair support geometry.
3. **Use fixed evaluation groups and observable reset distributions.** Run
   standing separately; use the same seeds and equal counts for each manual
   pose in diagnostics. Log actual training reset IDs and training valid-step
   events, which the archived episode metrics did not retain. Keep raw trials,
   pose summaries, detector outcomes and video identities together.
4. **Bound the learning test and stop on physical failure.** Admit a short
   pilot only after runtime and physics checks. Evaluate at 0/50k/100k/200k
   counted ticks, and stop or revise if all starts still fall before one
   second with zero landings. Require measured bilateral supported
   lift/landing before promotion; use a longer held-out standing-start test
   before describing walking. The precise promotion criteria should be
   declared in the next run config/runbook, not chosen after seeing reward.
5. **Profile full training before scaling robots.** Measure physics-only and
   complete collection/replay/optimizer throughput separately, including
   compilation, evaluation, peak VRAM, real updates and contact overflow.
   Sweep 64/128/256 worlds only after a full-trainer bounded smoke; historical
   512/1600-world failures show why a physics-only benchmark is insufficient.
   Compare compile/mixed-precision settings as isolated changes with finite
   losses and physically matched rollouts. Record control transitions/s,
   optimizer updates/s and wall-clock time to physical milestones.

The strongest immediate causal hypotheses are changed command semantics,
unqualified short-lived reset states and severe command slew saturation. The
logs do not isolate their contributions or demonstrate that model capacity,
torque shortage, GPU utilization or insufficient training duration is the
primary cause. A paired, single-change test is needed before assigning cause.
