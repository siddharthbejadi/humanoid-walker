#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bundle_root="$(cd "$script_dir/../../.." && pwd)"
export PYTHONPATH="$bundle_root:$bundle_root/r2dreamer${PYTHONPATH:+:$PYTHONPATH}"
export TEACHER_SCENE="$bundle_root/r2dreamer/envs/mjx_walking/assets/humanoid_rubber_2mm_candidate.xml"
export TEACHER_RUN_DIR="${TEACHER_RUN_DIR:-/workspace/mjx_teacher_standing_pilot}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORM_NAME=gpu
python_bin="${PYTHON_BIN:-python}"

if [[ ! -f "$TEACHER_SCENE" ]]; then
  echo "Teacher candidate scene missing: $TEACHER_SCENE" >&2
  exit 2
fi
"$python_bin" -m pip check
cd "$bundle_root/r2dreamer"
"$python_bin" scripts/preflight_mjx_teacher.py --scene "$TEACHER_SCENE" --worlds 8 \
  --config-name walking_mjx_standing_teacher
"$python_bin" scripts/preflight_mjx_teacher.py --scene "$TEACHER_SCENE" --worlds 64 \
  --config-name walking_mjx_standing_teacher
if [[ "${TEACHER_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  echo "Preflight-only mode: no training started."
  exit 0
fi
if [[ -e "$TEACHER_RUN_DIR" ]] && [[ -n "$(ls -A "$TEACHER_RUN_DIR")" ]]; then
  echo "Refusing to overwrite a nonempty run directory: $TEACHER_RUN_DIR" >&2
  exit 2
fi
"$python_bin" -u train.py --config-name walking_mjx_standing_teacher
