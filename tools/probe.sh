#!/bin/bash
# usage: probe.sh NAME OVERRIDES...
set -u
NAME=$1; shift
PY=/workspace/script_free_recovery_source_20261013_v3b/r2dreamer/.venv/bin/python
PILOT=/workspace/script_free_pilot_v2_w256_r64_seed0
OUT=/workspace/script_free_v4_probe_$NAME
cd /workspace/script_free_v4/r2dreamer
mkdir -p $OUT
$PY -u train.py --config-name walking_mjx_script_free seed=100 \
 imitation.dataset=$PILOT/demonstrations.npz init_from=$PILOT/imitation/latest.pt \
 logdir=$OUT/online env.env_num=256 env.train_ratio=64 env.steps=2000000 env.eval_seed=1100 \
 trainer.learning_start_deadline_steps=176640 trainer.stop_after_optimizer_updates=128 \
 trainer.gait_guard_enabled=true trainer.best_metric_key=walking_success_rate \
 trainer.gait_guard_eval_updates=32 trainer.eval_video=false "$@" > $OUT/online.log 2>&1
echo "$NAME exit $?" >> /workspace/script_free_v4_probes.status
