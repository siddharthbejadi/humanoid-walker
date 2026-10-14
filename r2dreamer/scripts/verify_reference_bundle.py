"""Verify every packaged source/reference byte after upload, using only stdlib."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def verify(manifest_path):
    manifest_path = Path(manifest_path).resolve(strict=True)
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("files_sha256")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("Bundle manifest has no source fingerprints")
    for name, expected in entries.items():
        path = (root / name).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Invalid bundle source path: {name}") from exc
        if not path.is_file():
            raise ValueError(f"Invalid bundle source path: {name}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != expected:
            raise ValueError(f"Bundle source/reference mismatch: {name}")
    return {"status": "verified_source_bundle", "files_verified": len(entries),
            "gpu_tested": False, "manifest": str(manifest_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).resolve().parents[2] / "SOURCE_MANIFEST.json")
    args = parser.parse_args()
    print(json.dumps(verify(args.manifest), indent=2))


if __name__ == "__main__":
    main()
