#!/usr/bin/env bash
# Default: GPU admission only. An explicit 0 admits the bounded pilot afterward.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
seed="${SCRIPT_FREE_SEED:-0}"
if (( $# > 1 )); then echo "Only an optional seed=N argument is accepted" >&2; exit 2; fi
if (( $# == 1 )); then
  if [[ ! "$1" =~ ^seed=[0-9]+$ ]]; then echo "Only seed=N is accepted" >&2; exit 2; fi
  seed="${1#seed=}"
fi
if [[ ! "$seed" =~ ^[0-9]+$ ]]; then echo "SCRIPT_FREE_SEED must be nonnegative" >&2; exit 2; fi
mode="${SCRIPT_FREE_PREFLIGHT_ONLY:-1}"
case "$mode" in
  1) pipeline_mode=preflight ;;
  0) pipeline_mode=pilot ;;
  *) echo "SCRIPT_FREE_PREFLIGHT_ONLY must be exactly 1 or 0" >&2; exit 2 ;;
esac
python3 scripts/verify_script_free_bundle.py
if [[ ! -x .venv/bin/python || ! -f .mujoco_gl ]]; then
  echo "Run bash scripts/deploy/setup_script_free.sh first" >&2; exit 2
fi
export MUJOCO_GL="$(< .mujoco_gl)"
case "$MUJOCO_GL" in
  egl) unset PYOPENGL_PLATFORM ;;
  osmesa) export PYOPENGL_PLATFORM=osmesa ;;
  *) echo "Invalid saved renderer" >&2; exit 2 ;;
esac
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=gpu
export PYTHONUNBUFFERED=1
run="${SCRIPT_FREE_RUN_DIR:-/workspace/script_free_${pipeline_mode}_$(date -u +%Y%m%d_%H%M%S)}"
if [[ -e "$run" ]]; then echo "Run directory already exists; use a fresh SCRIPT_FREE_RUN_DIR" >&2; exit 2; fi
exec .venv/bin/python -u scripts/run_script_free_pipeline.py --mode "$pipeline_mode" --output "$run" --seed "$seed"
