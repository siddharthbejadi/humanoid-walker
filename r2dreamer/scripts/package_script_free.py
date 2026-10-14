"""Build a new, self-contained script-free training source bundle and hashes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.verify_script_free_bundle import BUNDLE_ID, GENERATED_REPORTS, REQUIRED_ENTRIES, RUNTIME_DIRS, sha256_file
ALLOWED = {".py", ".yaml", ".yml", ".json", ".xml", ".stl", ".obj", ".dae",
           ".md", ".txt", ".csv", ".sh", ".toml", ".npz", ".png", ".jpg", ".jpeg"}
EXCLUDED = RUNTIME_DIRS
EXPERT_NPZ = "envs/mjx_walking/assets/humanlike_gait.npz"
GUIDE = "docs/SCRIPT_FREE_TRAINING_20261013_V2.md"


def source_files():
    files = []
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        name = relative.as_posix()
        if not path.is_file() or path.suffix.lower() not in ALLOWED:
            continue
        if any(part in EXCLUDED or part.endswith((".egg-info", ".dist-info")) for part in relative.parts):
            continue
        if ("(1)" in path.name or "_mjx_contacts" in path.stem or path.name.startswith(".")
                or path.name.endswith(".manifest.json") or path.name == "SOURCE_MANIFEST.json"
                or path.name in GENERATED_REPORTS):
            continue
        if path.is_symlink():
            raise RuntimeError(f"Source packaging refuses symbolic links: {path}")
        if path.suffix.lower() == ".npz" and name != EXPERT_NPZ:
            continue
        files.append(path)
    names = {"r2dreamer/" + path.relative_to(ROOT).as_posix() for path in files}
    missing = REQUIRED_ENTRIES - names - {"START_HERE.md"}
    if missing or ROOT / GUIDE not in files:
        raise RuntimeError(f"Missing required script-free source closure: {sorted(missing)}; guide required={GUIDE}")
    scene = ROOT / "envs/mjx_walking/assets/humanoid_rubber_2mm_servo_v2_kp40.xml"
    tree = ET.parse(scene)
    compiler = tree.getroot().find("compiler")
    meshdir = scene.parent / (compiler.get("meshdir", "") if compiler is not None else "")
    for mesh in tree.getroot().findall("asset/mesh"):
        if mesh.get("file"):
            path = (meshdir / mesh.get("file")).resolve(strict=True)
            if path not in files:
                raise RuntimeError(f"Referenced student mesh absent from bundle: {path}")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT.parent / (BUNDLE_ID + ".zip"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    destination = args.output.resolve()
    manifest_path, hash_path = destination.with_suffix(".manifest.json"), destination.with_suffix(".sha256")
    if any(path.exists() for path in (destination, manifest_path, hash_path)):
        parser.error("Archive or sidecar exists; preserve earlier evidence and choose a new output name")
    files = source_files()
    content = {"r2dreamer/" + path.relative_to(ROOT).as_posix(): path.read_bytes() for path in files}
    content["START_HERE.md"] = (ROOT / GUIDE).read_bytes()
    import hashlib
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in content.items()}
    manifest = dict(schema_version=1, bundle=BUNDLE_ID, files_sha256=hashes,
                    prepared_date="2026-10-13", recipe_changed=False,
                    roles={"r2dreamer/" + EXPERT_NPZ: "External demonstration collection only; never loaded by student inference"},
                    generated_demonstrations_included=False, policy_checkpoints_included=False,
                    snapshot_pose_pool_is_dataset=False, gpu_qualified=False, walking_qualified=False,
                    scope="No GPU execution performed during package preparation; actual demonstrations and trained walking remain unverified")
    if args.dry_run:
        print(json.dumps(dict(path=str(destination), source_files=len(files), payload_files=len(content),
                              uncompressed_bytes=sum(map(len, content.values())), scope=manifest["scope"]), indent=2))
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in content.items():
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 13, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o755 if name.endswith(".sh") else 0o644) << 16
            archive.writestr(info, data)
        archive.writestr("SOURCE_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Script-free ZIP CRC validation failed")
        if set(archive.namelist()) != set(content) | {"SOURCE_MANIFEST.json"}:
            raise RuntimeError("ZIP payload entry set does not match the reviewed closure")
        for name, expected in hashes.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise RuntimeError(f"ZIP source hash mismatch: {name}")
        if json.loads(archive.read("SOURCE_MANIFEST.json")) != manifest:
            raise RuntimeError("Source manifest failed its archive round-trip")
    archive_hash = sha256_file(destination)
    manifest_path.write_text(json.dumps(dict(manifest, archive_sha256=archive_hash), indent=2) + "\n", encoding="utf-8")
    hash_path.write_text(archive_hash + "  " + destination.name + "\n", encoding="ascii")
    print(json.dumps(dict(path=str(destination), sha256=archive_hash, bytes=destination.stat().st_size,
                          source_files=len(files), payload_files=len(content), manifest=str(manifest_path),
                          validation="CRC, entry set, every payload SHA-256 and manifest round-trip passed",
                          gpu_tested=False), indent=2))


if __name__ == "__main__":
    main()
