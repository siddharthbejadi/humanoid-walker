#!/bin/bash
# realistic student (masked + 40 ms delay + IMU noise): wait for collection, BC 12k at default start fraction, 3-seed eval
PY=/workspace/script_free_recovery_source_20261013_v3b/r2dreamer/.venv/bin/python
R=/workspace/sf_v6_run1; S=/workspace/script_free_v6/r2dreamer; cd $S
while pgrep -f "[c]ollect_script_free" > /dev/null; do sleep 10; done
[ -f $R/demonstrations.npz ] || { echo "no demonstrations" > $R/status9; exit 1; }
export STARTUP_FRACTION=0.25
rm -rf $R/bc12k; echo collected > $R/status9
$PY -u scripts/pretrain_script_free.py --dataset $R/demonstrations.npz --output $R/bc12k --updates 12000 > $R/bc12k.log 2>&1 || { echo "pretrain failed" > $R/status9; exit 1; }
echo pretrained > $R/status9
for SEED in 1100 1200 1300; do
  rm -rf /workspace/sf_evals/real_$SEED
  $PY -u scripts/evaluate_script_free.py --checkpoint $R/bc12k/latest.pt --output /workspace/sf_evals/real_$SEED --seed $SEED --seconds 20 --worlds 32 > /workspace/sf_evals/real_$SEED.log 2>&1
done
echo evaluated > $R/status9
