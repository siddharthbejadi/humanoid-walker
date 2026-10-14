#!/usr/bin/env bash
# Install the pinned runtime; actual student/learner checks run in the next stage.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
python3 scripts/verify_script_free_bundle.py
if [[ -e .venv ]]; then
  echo "Existing .venv preserved. Use a fresh source extraction for setup; do not recreate an active runtime." >&2
  exit 2
fi
python3 scripts/check_script_free_host.py --workspace "$PWD" --output .setup_host_report.json
if command -v apt-get >/dev/null 2>&1; then
  (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libegl1 libgl1 libosmesa6 tmux >/dev/null) \
    || echo "System graphics installation failed; renderer readiness must still pass."
fi
if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required for the documented persistent session. Install it with your host's package manager first." >&2
  exit 2
fi
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt
uv pip check --python .venv/bin/python
for gl in egl osmesa; do
  export MUJOCO_GL="$gl"
  if [[ "$gl" == osmesa ]]; then export PYOPENGL_PLATFORM=osmesa; else unset PYOPENGL_PLATFORM; fi
  set +e
  .venv/bin/python scripts/check_script_free_render.py
  code=$?
  set -e
  if [[ "$code" == 0 ]]; then
    printf '%s\n' "$gl" > .mujoco_gl
    echo "Setup complete. Next: SCRIPT_FREE_PREFLIGHT_ONLY=1 bash scripts/deploy/start_script_free.sh seed=0"
    exit 0
  fi
  if [[ "$code" != 3 ]]; then exit "$code"; fi
done
echo "No renderer passed. Do not start the script-free pilot." >&2
exit 3
