"""CPU admission checks; these tests do not certify CUDA or walking behavior."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.package_reference_training import REQUIRED, source_files
from scripts.preflight_reference_training import (
    CORE_RUNTIME, compose, structural_report, validate_capacity, validate_capacity_videos,
    validate_cuda_build, validate_runtime_versions)
from scripts.verify_reference_bundle import verify


def script_free_archive_only(root=ROOT):
    manifest = root.parent / "SOURCE_MANIFEST.json"
    if not manifest.is_file():
        return False
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    return isinstance(metadata, dict) and metadata.get("bundle") in {
        "r2dreamer_script_free_next_20261012", "r2dreamer_script_free_next_20261013_v2"}


class ReferencePreparationTests(unittest.TestCase):
    @staticmethod
    def manifest_fixture(root, content=b"reference source fixture"):
        source = root / "r2dreamer" / "fixture.py"
        source.parent.mkdir(parents=True)
        source.write_bytes(content)
        manifest = root / "SOURCE_MANIFEST.json"
        manifest.write_text(json.dumps(dict(files_sha256={"r2dreamer/fixture.py": hashlib.sha256(content).hexdigest()})),
                            encoding="utf-8")
        return manifest, source

    def test_bundle_manifest_verifies_source_without_runtime_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, _ = self.manifest_fixture(Path(directory))
            report = verify(manifest)
            self.assertEqual(report["status"], "verified_source_bundle")
            self.assertEqual(report["files_verified"], 1)
            self.assertFalse(report["gpu_tested"])

    def test_bundle_manifest_rejects_changed_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, source = self.manifest_fixture(Path(directory))
            source.write_bytes(b"tampered fixture")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                verify(manifest)

    def test_bundle_manifest_rejects_relative_or_absolute_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "bundle"
            root.mkdir()
            outside = base / "outside.txt"
            outside.write_bytes(b"outside fixture")
            manifest = root / "SOURCE_MANIFEST.json"
            for name in ("../outside.txt", str(outside.resolve())):
                with self.subTest(name=name):
                    manifest.write_text(json.dumps(dict(files_sha256={name: hashlib.sha256(outside.read_bytes()).hexdigest()})),
                                        encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "Invalid bundle source path"):
                        verify(manifest)

    def test_pinned_runtime_accepts_cuda_local_tag_without_importing_gpu(self):
        versions = dict(CORE_RUNTIME, torch="2.8.0+cu128")
        report = validate_runtime_versions(versions.__getitem__, python_version=(3, 11))
        self.assertEqual(report["versions"]["torch"], "2.8.0+cu128")

    def test_wrong_python_or_core_version_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Python 3.11"):
            validate_runtime_versions(CORE_RUNTIME.__getitem__, python_version=(3, 13))
        versions = dict(CORE_RUNTIME, mujoco="3.15.0")
        with self.assertRaisesRegex(RuntimeError, "runtime mismatch"):
            validate_runtime_versions(versions.__getitem__, python_version=(3, 11))

    def test_cuda_build_major_is_checked_without_importing_gpu(self):
        self.assertEqual(validate_cuda_build("12.8"), "12.8")
        for value in (None, "11.8", "13.0"):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                validate_cuda_build(value)

    @unittest.skipIf(script_free_archive_only(), "Script-free archive excludes the unused residual RSI phase pool")
    def test_exact_candidate_has_mesh_and_reference_closure(self):
        report = structural_report(compose(["seed=7"]))
        self.assertEqual(report["status"], "passed_structural_only")
        self.assertFalse(report["gpu_tested"])
        self.assertFalse(report["physical_reference_qualified"])
        self.assertEqual(report["replay_warmup_counted_ticks"], 85760)
        self.assertEqual(report["config"]["env"]["eval_seed"], 1000)

    def test_old_checkpoint_is_rejected(self):
        config = compose()
        config.init_from = "old_soft_servo.pt"
        with self.assertRaisesRegex(ValueError, "start fresh"):
            structural_report(config)

    def test_changed_physics_and_reset_distribution_are_rejected(self):
        for key, value in (("mjx_step_clearance_m", .020),
                           ("rsi_probabilities", [.5, 0., 0., .5]),
                           ("pose_rsi_json", "manual_poses.json")):
            with self.subTest(key=key):
                config = compose()
                config.env[key] = value
                with self.assertRaises(ValueError):
                    structural_report(config)

    @unittest.skipIf(script_free_archive_only(), "Script-free archive excludes the unused residual RSI phase pool")
    def test_archive_carries_binary_references_and_no_runtime_artifacts(self):
        files = {path.relative_to(ROOT).as_posix() for path in source_files()}
        self.assertTrue(REQUIRED <= files)
        self.assertIn("envs/mjx_walking/assets/humanlike_gait.npz", files)
        self.assertFalse(any(".venv/" in path or ".egg-info/" in path or
                             "_mjx_contacts" in path or path.endswith(".pt") for path in files))

    @staticmethod
    def capacity_evidence():
        checkpoint = dict(step=90000, optimizer_updates=32, stop_reason="optimizer_update_limit", finite_agent=True)
        trial = dict(start_pose_id=0, alive_seconds=.5, total_valid_steps=0, valid_left_step=0,
                     valid_right_step=0, alternating_steps=0, fall_count=1, safety_failure_count=1,
                     wall_hit=0, stall_out=0, goal_reached=0)
        protocol = dict(worlds=32, repeatable=True, seed=11000, rsi_probabilities=[1., 0., 0., 0.],
                        reset_source_kind="scripted_phase", controller_sample_offset=-1., valid_step_clearance_m=.015)
        contract = dict(task=dict(valid_step_clearance_m=.015), controller=dict(reference_kind="scripted",
                        gait_residual=True, reference_sample_offset_control_steps=-1., stabilizer=[1., .04, -1., .25]))
        rows = [dict(step=step, schema_version=2, protocol=dict(environment=copy.deepcopy(protocol)),
                     training_contract=copy.deepcopy(contract), trials=[dict(trial) for _ in range(32)])
                for step in (0, 90000)]
        return checkpoint, rows

    def test_runtime_capacity_does_not_require_a_walking_claim(self):
        checkpoint, rows = self.capacity_evidence()
        validate_capacity(checkpoint, rows)

    def test_incomplete_malformed_or_nonfinite_capacity_is_rejected(self):
        checkpoint, rows = self.capacity_evidence()
        variants = []
        incomplete = dict(checkpoint, optimizer_updates=31)
        variants.append((incomplete, rows))
        signaled = dict(checkpoint, stop_reason="signal")
        variants.append((signaled, rows))
        changed_trial = copy.deepcopy(rows)
        changed_trial[-1]["trials"][0]["start_pose_id"] = 5
        variants.append((checkpoint, changed_trial))
        missing = copy.deepcopy(rows)
        del missing[-1]["trials"][0]["safety_failure_count"]
        variants.append((checkpoint, missing))
        nonfinite = copy.deepcopy(rows)
        nonfinite[-1]["trials"][0]["alive_seconds"] = float("nan")
        variants.append((checkpoint, nonfinite))
        wrong_controller = copy.deepcopy(rows)
        wrong_controller[-1]["training_contract"]["controller"]["gait_residual"] = False
        variants.append((checkpoint, wrong_controller))
        for value, evaluations in variants:
            with self.subTest(value=value, evaluations=evaluations[-1]["trials"][0]):
                with self.assertRaises(ValueError):
                    validate_capacity(value, evaluations)

    @staticmethod
    def video_fixture(run, step=90000):
        video_dir = run / "eval_videos"
        video_dir.mkdir(exist_ok=True)
        for world in range(32):
            name = f"step_{step:07d}" + ("" if world == 0 else f"_env_{world:02d}") + ".mp4"
            video = video_dir / name
            video.write_bytes(b"nonempty video fixture, not a codec test")
            video.with_suffix(".json").write_text(json.dumps(dict(training_step=step,
                evaluation_world=world, start_pose_id=0, frames=2, fps=25)), encoding="utf-8")
        return video_dir

    def test_capacity_requires_all_32_final_clips(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            self.video_fixture(run, step=1)
            self.video_fixture(run)
            report = validate_capacity_videos(run, 90000)
            self.assertEqual(report["worlds"], 32)
            self.assertEqual(len(report["records"]), 32)

    def test_initial_videos_cannot_satisfy_final_capacity_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            self.video_fixture(run, step=1)
            with self.assertRaisesRegex(ValueError, "Incomplete final capacity video set"):
                validate_capacity_videos(run, 90000)

    def test_missing_last_world_video_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            videos = self.video_fixture(run)
            (videos / "step_0090000_env_31.mp4").unlink()
            with self.assertRaisesRegex(ValueError, "Incomplete final capacity video set"):
                validate_capacity_videos(run, 90000)

    def test_wrong_sidecars_missing_metadata_or_empty_clips_are_rejected(self):
        cases = (("training_step", 89999), ("evaluation_world", 30), ("start_pose_id", 1),
                 ("frames", 0), ("frames", True), ("fps", 10), ("missing_metadata", None), ("empty_video", None))
        for key, value in cases:
            with self.subTest(key=key, value=value), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                videos = self.video_fixture(run)
                video = videos / "step_0090000_env_31.mp4"
                metadata = video.with_suffix(".json")
                if key == "missing_metadata":
                    metadata.unlink()
                elif key == "empty_video":
                    video.write_bytes(b"")
                else:
                    record = json.loads(metadata.read_text(encoding="utf-8"))
                    record[key] = value
                    metadata.write_text(json.dumps(record), encoding="utf-8")
                with self.assertRaises(ValueError):
                    validate_capacity_videos(run, 90000)

    def test_duplicate_final_step_clip_names_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            videos = self.video_fixture(run)
            (videos / "step_0090000_env_31_duplicate.mp4").write_bytes(b"unexpected clip")
            with self.assertRaisesRegex(ValueError, "Incomplete final capacity video set"):
                validate_capacity_videos(run, 90000)


if __name__ == "__main__":
    unittest.main()
