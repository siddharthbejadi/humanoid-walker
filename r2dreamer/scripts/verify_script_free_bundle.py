"""Verify a script-free source bundle using only a Python 3.8+ standard library."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

BUNDLE_ID = "r2dreamer_script_free_next_20261013_v2"
RUNTIME_DIRS = {"__pycache__", ".venv", "venv", ".git", ".secrets", ".pytest_cache",
                "logdir", "runs", "dist", "datasets", "demonstrations", "artifacts",
                "tmp", "temp", "reports", "capacity", "pilot", "runtime", "backup", "backups",
                "eval_videos", "imitation", "online", "independent_eval", "imitation_eval", "PREP"}
GENERATED_REPORTS = {"run_status.json", "readiness_report.json", "structural_report.json",
                     "gpu_runtime_report.json", "demonstration_admission.json", "pretraining_report.json",
                     "script_free_evaluation.json", "backup_manifest.json", "host_prerequisites.json"}
SOURCE_SUFFIXES = {".py", ".pyw", ".pyc", ".sh", ".bash", ".yaml", ".yml", ".toml", ".json",
                   ".xml", ".npz", ".csv", ".stl", ".obj", ".dae", ".so", ".pyd", ".dll", ".pth"}
REQUIRED_ENTRIES = {
    "START_HERE.md", "r2dreamer/train.py", "r2dreamer/trainer.py", "r2dreamer/dreamer.py",
    "r2dreamer/full_command.py", "r2dreamer/script_free_contract.py", "r2dreamer/script_free_policy.py",
    "r2dreamer/imitation_data.py", "r2dreamer/imitation_training.py", "r2dreamer/scripted_expert.py",
    "r2dreamer/configs/walking_mjx_script_free.yaml", "r2dreamer/requirements-lock.txt",
    "r2dreamer/scripts/collect_script_free_demonstrations.py", "r2dreamer/scripts/pretrain_script_free.py",
    "r2dreamer/scripts/evaluate_script_free.py", "r2dreamer/scripts/run_script_free_pipeline.py",
    "r2dreamer/scripts/check_script_free_runtime.py", "r2dreamer/scripts/deploy/setup_script_free.sh",
    "r2dreamer/scripts/deploy/start_script_free.sh", "r2dreamer/scripts/verify_script_free_bundle.py",
    "r2dreamer/envs/mjx_teacher_env.py", "r2dreamer/envs/mjx_walking/env.py",
    "r2dreamer/envs/mjx_walking/assets/humanoid_rubber_2mm_servo_v2_kp40.xml",
    "r2dreamer/envs/mjx_walking/assets/humanlike_gait.npz",
    "r2dreamer/scripts/test_imitation_data.py", "r2dreamer/scripts/test_imitation_learner.py",
    "r2dreamer/scripts/test_imitation_reporting_v2.py",
    "r2dreamer/scripts/test_full_command_env.py", "r2dreamer/scripts/test_script_free_contract.py",
    "r2dreamer/scripts/test_script_free_online_hook.py", "r2dreamer/scripts/test_script_free_pipeline.py",
    "r2dreamer/scripts/test_script_free_package.py",
    "r2dreamer/scripts/check_script_free_host.py", "r2dreamer/scripts/test_script_free_host.py",
    "r2dreamer/scripts/show_script_free_status.py", "r2dreamer/scripts/backup_script_free_run.py",
    "r2dreamer/scripts/script_free_run_ops.py", "r2dreamer/scripts/test_script_free_run_ops.py",
    "r2dreamer/demonstration_audit.py", "r2dreamer/scripts/test_demonstration_audit.py",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_path(relative):
    return any(part in RUNTIME_DIRS or part.endswith((".egg-info", ".dist-info"))
               for part in relative.parts) or relative.name in GENERATED_REPORTS or relative.name == ".setup_host_report.json"


def generated_contact_scene(relative, entries):
    if relative.parent.as_posix() != "r2dreamer/envs/mjx_walking/assets":
        return False
    match = re.fullmatch(r"(.+)_mjx_contacts(?:_[0-9]+(?:\.[0-9]+)?kgcm)?\.xml", relative.name)
    return bool(match and (relative.parent / (match.group(1) + ".xml")).as_posix() in entries)


def reject_unlisted_source(root, entries):
    source = root / "r2dreamer"
    for path in source.rglob("*"):
        relative = path.relative_to(root)
        if runtime_path(relative):
            continue
        name = relative.as_posix()
        if name in entries or generated_contact_scene(relative, entries):
            continue
        if path.is_symlink() or (path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES):
            raise ValueError(f"Unlisted script-free source/configuration: {name}")


def verify(manifest_path):
    manifest_path = Path(manifest_path).resolve(strict=True)
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("bundle") != BUNDLE_ID or manifest.get("schema_version") != 1:
        raise ValueError("Expected the reviewed script-free source bundle identity/schema")
    entries = manifest.get("files_sha256")
    if not isinstance(entries, dict) or not REQUIRED_ENTRIES <= entries.keys():
        raise ValueError("Script-free source manifest is missing required closure entries")
    for name, expected in entries.items():
        if not isinstance(name, str) or not name or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Malformed source name or SHA-256 fingerprint")
        relative = PurePosixPath(name)
        if (relative.is_absolute() or ".." in relative.parts or "\\" in name or relative.as_posix() != name or
                (name != "START_HERE.md" and relative.parts[0] != "r2dreamer")):
            raise ValueError(f"Invalid bundle source path: {name}")
        path = (root / name).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Invalid bundle source path: {name}") from exc
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Script-free source/reference mismatch: {name}")
    reject_unlisted_source(root, entries)
    return dict(status="verified_script_free_source_bundle", files_verified=len(entries),
                manifest=str(manifest_path), manifest_sha256=sha256_file(manifest_path), gpu_tested=False,
                unlisted_source_checked=True,
                scope="Source/reference identity only; runtime, demonstration quality and learned walking remain unqualified")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).resolve().parents[2] / "SOURCE_MANIFEST.json")
    args = parser.parse_args()
    print(json.dumps(verify(args.manifest), indent=2))


if __name__ == "__main__":
    main()
