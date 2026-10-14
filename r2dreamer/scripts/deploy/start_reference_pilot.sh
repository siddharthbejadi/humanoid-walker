#!/usr/bin/env bash
# Fresh 128-world pilot after exact-reference diagnostics and full-learner capacity.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$script_dir/../.." && pwd)"
cd "$root"
python_bin="${PYTHON_BIN:-$root/.venv/bin/python}"
preflight_only="${REFERENCE_PREFLIGHT_ONLY:-1}"
case "$preflight_only" in
  0|1) ;;
  *) echo "REFERENCE_PREFLIGHT_ONLY must be exactly 0 or 1; only explicit 0 launches the pilot." >&2; exit 2 ;;
esac
manifest="$root/../SOURCE_MANIFEST.json"
if [[ ! -f "$manifest" ]]; then
  echo "Missing SOURCE_MANIFEST.json. Launch from the verified extracted reference bundle." >&2
  exit 2
fi
"${SOURCE_VERIFY_PYTHON:-python3}" scripts/verify_reference_bundle.py --manifest "$manifest"
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=gpu
export TEACHER_RUN_DIR="${TEACHER_RUN_DIR:-/workspace/reference_pilot_20261012}"
readiness_dir="${REFERENCE_READINESS_DIR:-${TEACHER_RUN_DIR}_readiness}"
gl="${MUJOCO_GL:-$(cat .mujoco_gl 2>/dev/null || echo egl)}"
export MUJOCO_GL="$gl"
if [[ "$gl" == osmesa ]]; then export PYOPENGL_PLATFORM=osmesa; fi
if [[ "$gl" == none ]]; then
  echo "Reference pilot requires MP4 rendering; configure EGL or OSMesa first." >&2
  exit 2
fi
for arg in "$@"; do
  if [[ ! "$arg" =~ ^seed=[0-9]+$ ]]; then
    echo "Unsupported pilot override: $arg (allowed: seed=N)." >&2
    exit 2
  fi
done
for path in "$TEACHER_RUN_DIR" "$readiness_dir"; do
  if [[ -e "$path" ]]; then
    echo "Refusing an existing output path: $path. Select fresh run/readiness paths." >&2
    exit 2
  fi
done
"$python_bin" -m pip check
"$python_bin" scripts/preflight_reference_training.py --mode structural "$@"
timeout --signal=TERM 1200 "$python_bin" scripts/preflight_reference_training.py --mode gpu --output "$readiness_dir/baseline" "$@"
"$python_bin" scripts/preflight_reference_training.py --mode capacity --output "$readiness_dir/capacity" "$@"
if [[ "$preflight_only" == 1 ]]; then
  echo "Readiness checks complete. No pilot launched. Physical results are in $readiness_dir."
  echo "To train, use a fresh output pair and explicitly set REFERENCE_PREFLIGHT_ONLY=0."
  exit 0
fi
"$python_bin" -u train.py --config-name walking_mjx_reference_pilot "$@"
