#!/usr/bin/env bash
# 8-world local walking run with live pose injection (scripts/interactive_rsi.py).
# Usage: start_interactive_walking.sh <logdir> [extra hydra overrides...]
#   fresh stage:  ... <new logdir>  (initializes from FROM_CKPT)
#   resume:       ... <same logdir> resume_from=<logdir>/latest.pt init_from=null
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
logdir="$1"; shift
FROM_CKPT="${FROM_CKPT:-/mnt/c/robot/mixed_training/mjx_walking_from_mixed_20261007_172519/best_safe.pt}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TEACHER_RUN_DIR="$logdir"
# Servo-like actuators by default (2026-10-08); SCENE=humanoid_rubber_2mm_servo.xml for kp 20 / no joint friction, humanoid_rubber_2mm_candidate.xml for the old soft gains.
export TEACHER_SCENE="$PWD/envs/mjx_walking/assets/${SCENE:-humanoid_rubber_2mm_servo_v2_kp40.xml}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.5}"
export JAX_PLATFORM_NAME=gpu
export MUJOCO_GL="${MUJOCO_GL:-egl}"  # headless rendering for eval videos
mkdir -p "$logdir"
exec .venv/bin/python -u train.py --config-name walking_mjx_privileged_teacher_200k \
  env.env_num="${ENV_NUM:-8}" env.steps="${STEPS:-100000}" trainer.steps="${STEPS:-100000}" \
  trainer.checkpoint_interval="${CKPT_EVERY:-10000}" \
  env.mjx_reward_profile="${REWARD_PROFILE:-goal_corridor_v3}" \
  model.actor.dist.cont.min_std="${MIN_STD:-0.01}" model.actor.dist.cont.max_std="${MAX_STD:-0.06}" \
  init_from="$FROM_CKPT" "$@"
