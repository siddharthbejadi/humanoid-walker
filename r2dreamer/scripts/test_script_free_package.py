"""Stdlib-only source-integrity and package-closure tests, without GPU execution."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import package_script_free
from scripts.verify_script_free_bundle import BUNDLE_ID, REQUIRED_ENTRIES, sha256_file, verify


def source_fixture(root):
    fingerprints = {}
    for name in REQUIRED_ENTRIES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("fixture " + name).encode("utf-8"))
        fingerprints[name] = sha256_file(path)
    manifest = root / "SOURCE_MANIFEST.json"
    manifest.write_text(json.dumps(dict(schema_version=1, bundle=BUNDLE_ID, files_sha256=fingerprints)), encoding="utf-8")
    return manifest


class ScriptFreePackageTests(unittest.TestCase):
    def test_complete_source_closure_verifies_without_runtime_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            report = verify(source_fixture(Path(directory)))
            self.assertEqual(report["status"], "verified_script_free_source_bundle")
            self.assertEqual(report["files_verified"], len(REQUIRED_ENTRIES))
            self.assertFalse(report["gpu_tested"])

    def test_changed_bytes_or_removed_required_entry_are_rejected(self):
        for variant in ("tamper", "missing"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = source_fixture(root)
                name = "r2dreamer/script_free_policy.py"
                if variant == "tamper":
                    (root / name).write_bytes(b"modified actor source")
                else:
                    record = json.loads(manifest.read_text())
                    del record["files_sha256"][name]
                    manifest.write_text(json.dumps(record), encoding="utf-8")
                with self.assertRaises(ValueError):
                    verify(manifest)

    def test_parent_path_escape_is_rejected_before_reading_outside_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "bundle"
            manifest = source_fixture(root)
            outside = base / "outside.txt"
            outside.write_bytes(b"outside fixture")
            record = json.loads(manifest.read_text())
            record["files_sha256"]["../outside.txt"] = sha256_file(outside)
            manifest.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid bundle source path"):
                verify(manifest)

    def test_wrong_bundle_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = source_fixture(Path(directory))
            record = json.loads(manifest.read_text())
            for identity in ("old_residual_teacher_bundle", "r2dreamer_script_free_next_20261012"):
                record["bundle"] = identity
                manifest.write_text(json.dumps(record), encoding="utf-8")
                with self.subTest(identity=identity), self.assertRaisesRegex(ValueError, "identity"):
                    verify(manifest)

    def test_unlisted_import_shadow_config_script_and_native_source_are_rejected(self):
        for name in ("torch.py", "configs/override.yaml", "scripts/injected.sh", ".hidden.yaml",
                     "sitecustomize.pyc", "injected.pth", "injected.so"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = source_fixture(root)
                source = root / "r2dreamer" / name
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_bytes(b"unlisted source fixture")
                with self.assertRaisesRegex(ValueError, "Unlisted"):
                    verify(manifest)

    def test_legitimate_runtime_outputs_and_known_generated_contact_scene_are_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = source_fixture(root)
            for name in (".venv/lib/runtime.py", "__pycache__/runtime.pyc", "capacity/report.json",
                         "pilot/report.json", "backups/backup_manifest.json", ".setup_host_report.json",
                         ".mujoco_gl", "envs/mjx_walking/assets/humanoid_rubber_2mm_servo_v2_kp40_mjx_contacts_18kgcm.xml"):
                path = root / "r2dreamer" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"runtime output fixture")
            self.assertTrue(verify(manifest)["unlisted_source_checked"])
            unknown = root / "r2dreamer/envs/mjx_walking/assets/unknown_mjx_contacts_18kgcm.xml"
            unknown.write_bytes(b"unknown scene fixture")
            with self.assertRaisesRegex(ValueError, "Unlisted"):
                verify(manifest)

    def test_legacy_asset_skips_apply_only_to_the_two_known_student_identities(self):
        for script in ("test_reference_preparation.py", "test_reference_resets.py"):
            tree = ast.parse((ROOT / "scripts" / script).read_text(encoding="utf-8"))
            node = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "script_free_archive_only")
            namespace = dict(ROOT=ROOT, json=json)
            exec(compile(ast.Module(body=[node], type_ignores=[]), script, "exec"), namespace)
            with tempfile.TemporaryDirectory() as directory:
                parent = Path(directory)
                root = parent / "r2dreamer"
                root.mkdir()
                helper = namespace["script_free_archive_only"]
                self.assertFalse(helper(root))
                for identity, expected in (("r2dreamer_script_free_next_20261012", True),
                        (BUNDLE_ID, True), ("r2dreamer_script_free_next_20261013_unknown", False),
                        ("old_residual_teacher_bundle", False)):
                    (parent / "SOURCE_MANIFEST.json").write_text(json.dumps(dict(bundle=identity)), encoding="utf-8")
                    self.assertEqual(helper(root), expected)

    def test_builder_excludes_generated_data_runtime_metadata_and_snapshot_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source_fixture(base)
            root = base / "r2dreamer"
            guide = root / package_script_free.GUIDE
            guide.parent.mkdir(exist_ok=True)
            guide.write_bytes(b"source guide fixture")
            scene = root / "envs/mjx_walking/assets/humanoid_rubber_2mm_servo_v2_kp40.xml"
            scene.write_text('<mujoco><compiler meshdir="meshes"/><asset><mesh file="foot.stl"/></asset></mujoco>', encoding="utf-8")
            mesh = scene.parent / "meshes/foot.stl"
            mesh.parent.mkdir()
            mesh.write_bytes(b"mesh fixture")
            excluded = ("datasets/demonstrations.npz", "datasets/demonstrations.manifest.json",
                        "envs/mjx_walking/assets/humanlike_phase_pool.npz", ".venv/runtime.py",
                        "package.egg-info/metadata.json", "runs/latest.pt", "tmp/debug.json",
                        "capacity/report.json", "pilot/report.json", "backup/backup_manifest.json",
                        "backups/run_status.json", "run_status.json", "readiness_report.json")
            for name in excluded:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"excluded fixture")
            with patch.object(package_script_free, "ROOT", root):
                names = {path.relative_to(root).as_posix() for path in package_script_free.source_files()}
            self.assertIn(package_script_free.EXPERT_NPZ, names)
            self.assertIn("envs/mjx_walking/assets/meshes/foot.stl", names)
            self.assertTrue(names.isdisjoint(excluded))


if __name__ == "__main__":
    unittest.main()
