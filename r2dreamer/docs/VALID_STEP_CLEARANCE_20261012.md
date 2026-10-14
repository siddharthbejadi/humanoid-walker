# Valid-step clearance change, 12 October 2026

User requested a qualifying clearance within 14.5-20 mm. The active V4 recipe
now selects **at least 15 mm** of measured sole clearance during an unloaded,
supported swing. This replaces the earlier strictly-greater-than-20-mm lift
criterion for this recipe. A lift above 20 mm remains valid: the allowed range
applies to the configured minimum, not to a maximum physical foot height.

The measured 17.51/18.52-mm peaks from one CPU reference replay are not an
established robot maximum. This is a deliberate evaluation-protocol change,
with old results retained under their original definition.

## Configuration

- Active config: `env.mjx_step_clearance_m: 0.015` in
  `configs/walking_mjx_privileged_teacher_200k.yaml`.
- V4 launcher override: `STEP_CLEARANCE_M=0.015` (default). Values from
  `0.0145` through `0.020` metres are accepted; nonfinite/out-of-range values
  are rejected before physics construction.
- Direct launcher override: `env.mjx_step_clearance_m=0.015` when calling
  `scripts/deploy/start_interactive_walking.sh`.
- Raw environment and bridge fallback remain `0.020` for configurations that
  do not select a minimum. The shared comparison is now inclusive (`>=`).

## Criteria Retained

The changed swing predicate still requires swing load below 0.10 body weight,
opposite-foot load above 0.55, stable torso height/tilt and opposite-foot centre
speed below 0.03 m/s. Landing still requires an earlier qualified swing,
hysteretic contact, adequate opposite support and at least **20 mm forward
progress**. Reducing lift height does not reduce the forward-step requirement.

No physics, motor torque, reference trajectory, residual authority, reset
distribution or dot reward was changed. Dots retain their separate exploratory
semantics and must not be equated with valid supported steps.

## Checkpoints

The effective minimum is recorded as
`training_contract.task.valid_step_clearance_m`. A changed minimum or missing
field rejects a supposedly matching `resume_from`, including when the legacy
override is enabled. Use an explicitly declared `init_from`/`transfer_from`
adaptation for older checkpoints; replay and optimizer semantics differ by
load mode. No existing checkpoint bytes were modified.

## Validation Evidence

All 13 focused helper tests passed, including CPU JAX jit/vmap, threshold
boundaries, higher lifts, support and stance-motion failures, contact bounce,
missing earlier lift and unchanged forward progress. All 14 contract tests
passed, including threshold changes and missing metadata. YAML parsing and
launcher Bash syntax checks passed.
The original [20-mm CPU audit](../../../../review_20261012/v4_cpu_probe.json)
and its script are preserved.

The new comparison uses one physical trajectory and evaluates both minimums
against the same features; it does not change the robot motion to produce a
different count. Local MuJoCo 3.15 / Python 3.13 differs from the pinned
MuJoCo 3.14 / Python 3.11 MJX-Warp runtime.

| Same 10-second reference trajectory | 15 mm minimum | 20 mm minimum |
| --- | --- | --- |
| Qualified swing frames, left / right | 26 / 37 | 0 / 0 |
| Valid landing events, left / right | 0 / 7 | 0 / 0 |
| Alternating transitions | 0 | 0 |

The 8-second standing control counts zero steps under both thresholds. All
nine left landings in the reference had an earlier qualified lift, stable
height/tilt and 51.7-54.5 mm forward progress, but opposite-foot load was
0.377-0.538 body weight, below the retained >0.55 requirement. This change
admits right-foot events; it does not establish bilateral walking.

Evidence: [comparison script](../../../../review_20261012/v4_clearance_comparison.py)
and [comparison result](../../../../review_20261012/v4_clearance_comparison.json).
GPU MJX-Warp qualification and full training are not established by these
CPU checks.

No GPU training, cloud operation or robot actuation was launched for this
change. The previously identified phase-reset and metric-accounting issues
remain separate work.
