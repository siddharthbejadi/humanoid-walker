#!/usr/bin/env bash
# One-time setup on a fresh vast.ai (or any Linux CUDA) instance.
# Usage (from the unzipped bundle):  bash r2dreamer/scripts/deploy/vast_setup.sh
# Optional: ENV_NUM=64 (worlds to preflight with; must match training).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
echo "== GPU =="; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
# Headless video rendering: EGL first, OSMesa as fallback. Needs root; harmless if absent.
if command -v apt-get >/dev/null 2>&1; then
  (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libegl1 libgl1 libosmesa6 tmux >/dev/null) \
    || echo "apt-get failed (no root?): training still works, videos may need VIDEO=0"
fi
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "== Python env (exact versions from requirements-lock.txt) =="
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt
echo "== Reward common-sense test =="
.venv/bin/python scripts/check_goal_corridor_reward.py | tail -1
export PYTHONPATH="$PWD" XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.5}"
echo "== GPU preflight (${ENV_NUM:-64} worlds) =="
for gl in egl osmesa; do
  export MUJOCO_GL=$gl
  [ "$gl" = osmesa ] && export PYOPENGL_PLATFORM=osmesa
  set +e; .venv/bin/python scripts/preflight_servo_v3.py --worlds "${ENV_NUM:-64}"; code=$?; set -e
  if [ $code -eq 0 ]; then echo "$gl" > .mujoco_gl; echo "Setup complete. Next: REFERENCE_PREFLIGHT_ONLY=1 bash scripts/deploy/start_reference_pilot.sh seed=0"; exit 0; fi
  [ $code -ne 3 ] && { echo "Preflight failed (not a rendering problem) - do not start training."; exit $code; }
  echo "Rendering failed with $gl, trying the next backend..."
done
echo none > .mujoco_gl
echo "Setup complete WITHOUT video rendering. Reference pilot is blocked until EGL or OSMesa rendering works."
