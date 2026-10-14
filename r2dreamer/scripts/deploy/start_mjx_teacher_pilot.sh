#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bundle_root="$(cd "$script_dir/../../.." && pwd)"
export PYTHONPATH="$bundle_root:$bundle_root/r2dreamer${PYTHONPATH:+:$PYTHONPATH}"
export TEACHER_SCENE="$bundle_root/r2dreamer/envs/mjx_walking/assets/humanoid_rubber_2mm_candidate.xml"
export TEACHER_RUN_DIR="${TEACHER_RUN_DIR:-/workspace/mjx_teacher_pilot_200k}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Confirmed necessary on real hardware 2026-10-05/06 (standing-stage runs):
# without an explicit cap, JAX grabs most of VRAM on first use before Torch
# gets a chance, and the first optimizer update (first point Torch actually
# allocates) OOMs on the Warp CUDA graph launch. 0.6 is the only value
# that survived a full run to completion at env_num=64 on a 16GB card.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.6}"
export JAX_PLATFORM_NAME=gpu
python_bin="${PYTHON_BIN:-python}"

if [[ ! -f "$TEACHER_SCENE" ]]; then
  echo "Teacher candidate scene missing: $TEACHER_SCENE" >&2
  exit 2
fi
"$python_bin" -m pip check
cd "$bundle_root/r2dreamer"
"$python_bin" scripts/preflight_mjx_teacher.py --scene "$TEACHER_SCENE" --worlds 8
"$python_bin" scripts/preflight_mjx_teacher.py --scene "$TEACHER_SCENE" --worlds 64
if [[ "${TEACHER_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  echo "Preflight-only mode: no training started."
  exit 0
fi
if [[ -e "$TEACHER_RUN_DIR" ]] && [[ -n "$(ls -A "$TEACHER_RUN_DIR")" ]]; then
  echo "Refusing to overwrite a nonempty run directory: $TEACHER_RUN_DIR" >&2
  exit 2
fi
"$python_bin" -u train.py --config-name walking_mjx_privileged_teacher_200k
