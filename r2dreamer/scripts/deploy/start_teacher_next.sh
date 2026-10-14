#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$script_dir/../.." && pwd)"
export PYTHONPATH="$root/..:$root${PYTHONPATH:+:$PYTHONPATH}"
export TEACHER_SCENE="$root/envs/mjx_walking/assets/humanoid_rubber_2mm_candidate.xml"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORM_NAME=gpu
export JAX_PLATFORMS=cuda
python_bin="${PYTHON_BIN:-python}"
export TEACHER_RUN_DIR="${TEACHER_RUN_DIR:-/workspace/teacher_next_200k}"
cd "$root"

# This launcher is deliberately the matched64-world ablation. Other world
# counts and physics settings must go through the capacity probe first.
preflight_overrides=()
for arg in "$@"; do
  if [[ "$arg" =~ ^seed=[0-9]+$ || "$arg" =~ ^env.train_ratio=(16|64)$ ]]; then
    preflight_overrides+=(--override "$arg")
  else
    echo "Unsupported pilot override: $arg (allowed: seed=N, env.train_ratio=16 or64)" >&2
    exit 2
  fi
done
if [[ -e "$TEACHER_RUN_DIR" ]] && [[ ! -d "$TEACHER_RUN_DIR" || -n "$(ls -A "$TEACHER_RUN_DIR")" ]]; then
  echo "Refusing to overwrite existing run: $TEACHER_RUN_DIR" >&2
  exit 2
fi
"$python_bin" -m pip check
# Exact recipe, physical contract and GPU-only bridge, before a paid pilot.
"$python_bin" scripts/preflight_mjx_teacher.py --config-name walking_mjx_teacher_next --scene "$TEACHER_SCENE" --worlds 8 "${preflight_overrides[@]}"
"$python_bin" scripts/preflight_mjx_teacher.py --config-name walking_mjx_teacher_next --scene "$TEACHER_SCENE" --worlds 64 "${preflight_overrides[@]}"
"$python_bin" scripts/preflight_mjx_teacher.py --config-name walking_mjx_teacher_next --scene "$TEACHER_SCENE" --worlds 20 --evaluation "${preflight_overrides[@]}"
if [[ "${TEACHER_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  echo "Preflight-only complete; no learner started."
  exit 0
fi
# Fixed 64-world bounded fresh pilot. Additional explicit Hydra overrides can
# select the matched ratio=64 control or another declared training seed.
"$python_bin" -u train.py --config-name walking_mjx_teacher_next "$@"
