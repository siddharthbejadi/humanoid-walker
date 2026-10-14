"""Build a new self-contained teacher source bundle; preserve earlier ZIPs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {".py", ".yaml", ".yml", ".json", ".xml", ".stl", ".obj", ".dae",
           ".md", ".txt", ".csv", ".sh", ".toml", ".npz"}
EXCLUDED = {"__pycache__", ".venv", "venv", "logdir", "runs", "dist", ".git", ".secrets", ".pytest_cache"}
REQUIRED = {
    "train.py", "trainer.py", "training_contract.py", "training_budget.py", "reset_validation.py",
    "envs/__init__.py", "envs/mjx_teacher_env.py", "envs/mjx_walking/env.py", "envs/mjx_walking/scene.py",
    "configs/walking_mjx_reference_pilot.yaml", "scripts/preflight_reference_training.py",
    "scripts/teacher_capacity_probe.py", "scripts/deploy/start_reference_pilot.sh", "scripts/deploy/vast_setup.sh",
    "requirements-lock.txt", "envs/mjx_walking/assets/humanoid_rubber_2mm_servo_v2_kp40.xml",
    "envs/mjx_walking/assets/humanlike_gait.npz", "envs/mjx_walking/assets/humanlike_phase_pool.npz",
    "validation/vendor_forward_3_steps.csv", "docs/REFERENCE_TRAINING_NEXT_20261012.md",
    "scripts/verify_reference_bundle.py",
}


def source_files():
    files = []
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if not path.is_file() or path.suffix.lower() not in ALLOWED:
            continue
        if any(part in EXCLUDED or part.endswith((".egg-info", ".dist-info")) for part in relative.parts):
            continue
        if "(1)" in path.name or "_mjx_contacts" in path.stem or path.name.startswith("."):
            continue
        files.append(path)
    missing = REQUIRED - {path.relative_to(ROOT).as_posix() for path in files}
    if missing:
        raise RuntimeError(f"Missing required teacher closure files: {sorted(missing)}")
    # Every mesh referenced by the actual scene must travel with the source.
    scene = ROOT / "envs/mjx_walking/assets/humanoid_rubber_2mm_servo_v2_kp40.xml"
    tree = ET.parse(scene)
    compiler = tree.getroot().find("compiler")
    meshdir = scene.parent / (compiler.get("meshdir", "") if compiler is not None else "")
    for mesh in tree.getroot().findall("asset/mesh"):
        if not mesh.get("file"):
            continue
        path = (meshdir / mesh.get("file")).resolve(strict=True)
        if path not in files:
            raise RuntimeError(f"Referenced mesh missing from bundle closure: {path}")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    destination = args.output.resolve()
    manifest_path = destination.with_suffix(".manifest.json")
    digest_path = destination.with_suffix(".sha256")
    if any(path.exists() for path in (destination, manifest_path, digest_path)):
        parser.error("ZIP or verification sidecar exists; select a new name")
    files = source_files()
    content = {"r2dreamer/" + path.relative_to(ROOT).as_posix(): path.read_bytes() for path in files}
    content["START_HERE.md"] = (ROOT / "docs/REFERENCE_TRAINING_NEXT_20261012.md").read_bytes()
    entries = {name: hashlib.sha256(data).hexdigest() for name, data in content.items()}
    manifest = dict(bundle="reference_training_next_20261012", files_sha256=entries,
                    policy_checkpoints_included=False, gpu_qualified=False,
                    scope="Source/config/reference preparation only; CUDA readiness and physical results must be measured after rental")
    if args.dry_run:
        print(json.dumps(dict(path=str(destination), source_files=len(files),
                              uncompressed_bytes=sum(map(len, content.values())),
                              required_files=sorted(REQUIRED), scope=manifest["scope"]), indent=2))
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in content.items():
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 12, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o755 if name.endswith(".sh") else 0o644) << 16
            archive.writestr(info, data)
        archive.writestr("SOURCE_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP CRC validation failed")
        for name, expected in entries.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise RuntimeError(f"ZIP source mismatch: {name}")
        archived_manifest = json.loads(archive.read("SOURCE_MANIFEST.json"))
        if archived_manifest != manifest:
            raise RuntimeError("Archived manifest did not round-trip")
    with destination.open("rb") as handle:
        archive_digest = hashlib.file_digest(handle, "sha256").hexdigest()
    manifest_path.write_text(json.dumps(dict(manifest, archive_sha256=archive_digest), indent=2) + "\n", encoding="utf-8")
    digest_path.write_text(archive_digest + "  " + destination.name + "\n", encoding="ascii")
    print(json.dumps(dict(path=str(destination), source_files=len(files), bytes=destination.stat().st_size,
                          sha256=archive_digest, manifest=str(manifest_path),
                          validation="CRC, manifest round-trip and every archived entry hash matched",
                          gpu_tested=False), indent=2))


if __name__ == "__main__":
    main()
