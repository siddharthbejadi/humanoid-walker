"""Rewrite SOURCE_MANIFEST.json (SHA-256 of every source file + START_HERE.md), reusing the project's own packager rules.
Run after ANY edit under r2dreamer/, otherwise r2dreamer/scripts/verify_script_free_bundle.py (called by setup/start scripts) refuses to run.
    python3 scripts/make_source_manifest.py
"""
import hashlib, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "r2dreamer"))
from scripts.package_script_free import source_files, EXPERT_NPZ, BUNDLE_ID, ROOT as SRC  # noqa: E402

assert SRC.resolve() == (REPO / "r2dreamer").resolve()
content = {"r2dreamer/" + p.relative_to(SRC).as_posix(): p.read_bytes() for p in source_files()}
content["START_HERE.md"] = (REPO / "START_HERE.md").read_bytes()
manifest = dict(schema_version=1, bundle=BUNDLE_ID,
                files_sha256={k: hashlib.sha256(v).hexdigest() for k, v in content.items()},
                prepared_date="2026-10-13", recipe_changed=False,
                roles={"r2dreamer/" + EXPERT_NPZ: "External demonstration collection only; never loaded by student inference"},
                generated_demonstrations_included=False, policy_checkpoints_included=False,
                snapshot_pose_pool_is_dataset=False, gpu_qualified=False, walking_qualified=False,
                scope="Source/reference identity only; GPU runtime, demonstration quality and learned walking are qualified separately")
(REPO / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(len(content), "files hashed")
