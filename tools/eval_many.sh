#!/bin/bash
# usage: eval_many.sh CHECKPOINT TAG SEED...   -> /workspace/sf_evals/TAG_seed/
CK=$1; TAG=$2; shift 2
PY=/workspace/script_free_recovery_source_20261013_v3b/r2dreamer/.venv/bin/python
mkdir -p /workspace/sf_evals; cd /workspace/script_free_v4/r2dreamer
for S in "$@"; do
  rm -rf /workspace/sf_evals/${TAG}_$S
  $PY -u scripts/evaluate_script_free.py --checkpoint $CK --output /workspace/sf_evals/${TAG}_$S --seed $S --seconds 20 --worlds 32 > /workspace/sf_evals/${TAG}_$S.log 2>&1
  echo "$TAG $S exit $?" >> /workspace/sf_evals/status
done
