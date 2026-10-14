"""Build a portable source-only teacher ZIP with content-verified entries."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    destination = args.output.resolve()
    if destination.exists():
        parser.error("Output exists; choose a new filename to preserve the previous bundle")
    allowed = {".py", ".yaml", ".yml", ".json", ".xml", ".stl", ".obj", ".dae",
               ".md", ".txt", ".csv", ".sh", ".toml"}
    files = [root / "README.md"]
    for path in sorted((root / "r2dreamer").rglob("*")):
        if not path.is_file() or path.suffix.lower() not in allowed:
            continue
        relative = path.relative_to(root)
        if any(part in {"__pycache__", ".venv", "logdir", "dist", ".git", ".secrets"}
               or part.endswith((".egg-info", ".dist-info")) for part in relative.parts):
            continue
        files.append(path)
    content = {path.relative_to(root).as_posix(): path.read_bytes() for path in files}
    required = {"r2dreamer/train.py", "r2dreamer/training_contract.py",
                "r2dreamer/configs/walking_mjx_teacher_next.yaml",
                "r2dreamer/scripts/deploy/start_teacher_next.sh",
                "r2dreamer/envs/mjx_walking/assets/manual_rsi_poses.json"}
    if not required <= content.keys():
        raise RuntimeError(f"Missing required files: {sorted(required - content.keys())}")
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in content.items()}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in content.items():
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 4, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o755 if name.endswith(".sh") else 0o644) << 16
            archive.writestr(info, data)
        archive.writestr("SOURCE_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP CRC validation failed")
        for name, expected in manifest.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise RuntimeError(f"ZIP source mismatch: {name}")
    with destination.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    print(json.dumps({"path": str(destination), "source_files": len(content),
                      "bytes": destination.stat().st_size, "sha256": digest,
                      "validation": "CRC and every source entry hash matched",
                      "gpu_tested": False}, indent=2))


if __name__ == "__main__":
    main()
