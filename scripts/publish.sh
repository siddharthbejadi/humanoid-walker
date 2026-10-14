#!/usr/bin/env bash
# Create the GitHub repo and push main. Usage: bash scripts/publish.sh [private|public] [repo-name] [--with-checkpoint /path/to/student_realistic_12k.pt]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
vis="${1:-private}"; name="${2:-humanoid-walker}"; ckpt=""
[[ "${3:-}" == "--with-checkpoint" ]] && ckpt="${4:?path to checkpoint required}"
[[ "$vis" == private || "$vis" == public ]] || { echo "visibility must be private or public" >&2; exit 2; }
command -v gh >/dev/null || { echo "gh not installed: sudo apt-get install -y gh" >&2; exit 2; }
gh auth status >/dev/null 2>&1 || { echo "not logged in: run gh auth login" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "working tree not clean; commit first" >&2; exit 2; }
python3 r2dreamer/scripts/verify_script_free_bundle.py | grep -q verified_script_free_source_bundle || { echo "source manifest mismatch: python3 scripts/make_source_manifest.py" >&2; exit 2; }
if [[ "$vis" == public && ! -f LICENSE ]]; then echo "add a LICENSE file before publishing publicly" >&2; exit 2; fi
git remote get-url origin >/dev/null 2>&1 || gh repo create "$name" "--$vis" --source=. --remote=origin \
  --description "Script-free walking policy for a 19-servo humanoid (MuJoCo/MJX, R2Dreamer behaviour cloning)"
git push -u origin main
if [[ -n "$ckpt" ]]; then
  want=1d1af540214b7be6f201eed97bd47c0adca665b309b230968688cf1a5db806ce
  [[ "$(sha256sum "$ckpt" | cut -d' ' -f1)" == "$want" ]] || { echo "checkpoint hash mismatch" >&2; exit 2; }
  gh release create v0.1.0 "$ckpt" --title "student_realistic_12k" --notes "Behaviour-cloned walking student (simulation only). sha256 $want"
fi
gh repo view --web >/dev/null 2>&1 || true
echo "published: $(git remote get-url origin)"
