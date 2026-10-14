"""CPU admission/orchestration regressions; fixture files are not walking evidence."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from imitation_data import ImitationDataset, save_dataset
from full_command import motor_targets_from_actions, normalize_motor_targets
from scripts import run_script_free_pipeline as pipeline
from scripts.collect_script_free_demonstrations import collect_batch
from scripts.test_script_free_contract import full_command_contract


PROVENANCE = ("imitation_data.py", "scripted_expert.py", "scripts/collect_script_free_demonstrations.py",
              "full_command.py", "envs/mjx_teacher_env.py", "envs/mjx_walking/env.py", "training_contract.py")
DIAGNOSTICS = ("alive_seconds", "total_valid_steps", "valid_left_step", "valid_right_step", "alternating_steps",
    "fall_count", "safety_failure_count", "wall_hit", "stall_out", "goal_reached", "time_limit",
    "body_tilt_rad_peak", "root_drop_m_peak", "left_clearance_m_peak", "right_clearance_m_peak",
    "torque_saturation_fraction_mean", "control_slew_limited_fraction_mean")


class AuditProvider:
    worlds = 8

    def __init__(self, contract):
        self.command_center = np.asarray(contract["control"]["command_center_rad"], np.float32)
        self.command_scale = np.asarray(contract["control"]["action_scale_rad"], np.float32)
        self.standing_ctrl = np.asarray(contract["control"]["standing_ctrl_rad"], np.float32)

    def normalize(self, targets):
        return normalize_motor_targets(targets, self.command_center, self.command_scale)

    def audit(self):
        return dict(rotation=np.tile(np.eye(3, dtype=np.float32), (8, 1, 1)), ctrl=self.ctrl.copy(),
            qpos=np.zeros((8, 26), np.float32), qvel=np.zeros((8, 25), np.float32),
            time=np.full(8, self.steps * .02, np.float32), contact_sensor=np.zeros((8, 2, 32, 7), np.float32),
            actuator_force=np.zeros((8, 19), np.float32), step_count=np.full(8, self.steps, np.int32),
            state_rng=np.tile(np.asarray([1, 2], np.uint32), (8, 1)),
            bridge_rng=np.tile(np.asarray([self.steps + 3, 4], np.uint32), (8, 1)))

    def proprio(self):
        result = np.zeros((8, 45), np.float32)
        result[:, :19] = self.normalize(self.ctrl)
        return result

    def reset(self, seed):
        self.steps = 0
        self.ctrl = np.broadcast_to(self.standing_ctrl, (8, 19)).copy()
        return dict(proprio=self.proprio(), audit=self.audit())

    def step(self, action):
        target = motor_targets_from_actions(action, self.command_center, self.command_scale)
        self.ctrl = np.clip(target, self.ctrl - .16, self.ctrl + .16)
        self.steps += 1
        causes = {name: np.zeros(8, bool) for name in ("physical_fall", "wall_hit", "stall_out", "goal_reached", "time_limit")}
        return dict(proprio=self.proprio(), audit=self.audit(), reward=np.zeros(8, np.float32),
            terminal=np.zeros(8, bool), ended=np.zeros(8, bool), causes=causes,
            logs={name: np.zeros(8, np.float32) for name in ("valid_left_step", "valid_right_step", "alternating_steps")})


class AuditExpert:
    def __init__(self, provider):
        self.targets_value = np.broadcast_to(provider.command_center + .01, (8, 19)).copy()

    def reset(self, rotation):
        pass

    def targets(self, step, rotation):
        return self.targets_value


def evaluation_row(contract, step):
    protocol = dict(worlds=32, repeatable=True, full_command=True, reset_source_kind="standing",
        rsi_stratified=False, rsi_probabilities=None, valid_step_clearance_m=.015,
        horizon_control_steps=1000, control_dt_seconds=.02, reset_joint_noise_rad=.005,
        reset_velocity_noise=.005, observation_keys=["proprio"], controller_identity="autonomous_full_command_v1",
        reference_kind="none", stabilizer_enabled=False, phase_observations=False)
    trials = [dict.fromkeys(DIAGNOSTICS, 0.) for _ in range(32)]
    for world, trial in enumerate(trials):
        trial.update(evaluation_world=world, start_pose_id=0, termination_reasons=["time_limit"])
    return dict(step=step, schema_version=2, training_contract=contract,
                protocol=dict(eval_max_control_steps=1000, environment=protocol), trials=trials)


def unsafe_rollouts(arrays):
    arrays["cause_physical_fall"][:, -1] = True
    arrays["is_terminal"][:, -1] = True
    arrays["episode_safe"][:] = False
    arrays["valid"][:] = False


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.contract = full_command_contract()

    def tearDown(self):
        self.temporary.cleanup()

    def dataset(self, name="dataset", change_arrays=None, change_manifest=None):
        directory = self.directory / name
        directory.mkdir()
        schedule = [dict(seed=group, split=group, rollout_start=group * 8) for group in range(3)]
        provider = AuditProvider(self.contract)
        batches = [collect_batch(provider, AuditExpert(provider), max_steps=400, **row)[0] for row in schedule]
        arrays = {name: np.concatenate([batch[name] for batch in batches]) for name in batches[0]}
        if change_arrays:
            change_arrays(arrays)
        videos = []
        for row in schedule:
            video = directory / f"expert_seed_{row['seed']:07d}_split_{row['split']}.mp4"
            video.write_bytes(b"CPU fixture: nonempty video identity only, not encoded pixels")
            record = dict(path=video.name, seed=row["seed"], split=row["split"], rollout_id=row["rollout_start"],
                evaluation_world=0, frames=10, fps=25, sha256=pipeline.digest(video))
            video.with_suffix(".json").write_text(json.dumps(record), encoding="utf-8")
            videos.append(record)
        manifest = dict(training_contract=self.contract, student_training_contract=self.contract,
            video_required=True, reset_validation=dict(passed=True), collector="external_scripted_expert_v1",
            external_expert=dict(reference_sha256=pipeline.digest(ROOT / "envs/mjx_walking/assets/humanlike_gait.npz")),
            provenance={name: pipeline.digest(ROOT / name) for name in PROVENANCE},
            action_contract=dict(kind="full_motor_command", command_slew_rad_s=8., peak_torque_kgcm=18.,
                control_dt_seconds=.02, command_center=self.contract["control"]["command_center_rad"],
                command_scale=self.contract["control"]["action_scale_rad"]),
            schedule=schedule, video_evidence=videos, acceptance=dict(walking_teacher_qualified=False))
        if change_manifest:
            change_manifest(manifest)
        path = directory / "expert.npz"
        save_dataset(path, arrays, manifest)
        return path

    def test_current_structural_recipe_has_closure_and_never_claims_gpu_or_walking(self):
        report = pipeline.structural_report(pipeline.compose(0))
        self.assertEqual((report["training_worlds"], report["evaluation_worlds"], report["pilot_counted_ticks"]), (128, 32, 1000000))
        self.assertFalse(report["gpu_tested"])
        self.assertFalse(report["walking_qualified"])
        for name, digest in report["source_sha256"].items():
            self.assertEqual(pipeline.digest(ROOT / name), digest)
        self.assertEqual(report["student"]["encoder_inputs"], ["proprio"])
        self.assertFalse(report["student"]["privileged_actor_inputs"])

    def test_structural_guard_rejects_changed_budget_controller_or_evidence(self):
        original = pipeline.compose(0)
        for name, value in (("env.env_num", 64), ("env.steps", 2000000), ("buffer.max_size", 65536),
                ("env.time_limit", 4000), ("env.mjx_full_command", False), ("env.mjx_phase_obs", True),
                ("env.mjx_stabilizer", [1., 0., 0., 0.]), ("env.eval_episode_num", 16),
                ("trainer.eval_video_one_per_pose", True), ("model.imitation_weight", 0.),
                ("imitation.pretrain_updates", 32), ("init_from", "old_residual.pt")):
            with self.subTest(setting=name):
                changed = deepcopy(original)
                OmegaConf.update(changed, name, value)
                with self.assertRaises(ValueError):
                    pipeline.structural_report(changed)

    def test_capacity_and_pilot_commands_keep_fresh_identity_and_explicit_budgets(self):
        capacity = pipeline.online_command(Path("capacity/online"), Path("capacity/data.npz"), Path("capacity/imitation/latest.pt"), 7, True)
        self.assertIn("trainer.stop_after_optimizer_updates=32", capacity)
        self.assertIn("env.env_num=128", capacity)
        self.assertIn("buffer.max_size=131072", capacity)
        self.assertIn("imitation.pretrain_updates=32", capacity)
        self.assertIn("seed=7", capacity)
        self.assertFalse(any(item.startswith("resume_from=") for item in capacity))
        self.assertTrue(any(item.startswith("init_from=capacity/imitation/") for item in capacity))
        pilot = pipeline.online_command(Path("pilot/online"), Path("pilot/data.npz"), Path("pilot/imitation/latest.pt"), 107, False)
        self.assertFalse(any("stop_after_optimizer_updates=" in item for item in pilot))
        settings = pipeline.compose(107)
        self.assertEqual((settings.env.steps, settings.imitation.pretrain_updates), (1000000, 1000))
        self.assertEqual(settings.trainer.stop_after_optimizer_updates, 0)
        self.assertFalse(any(item.startswith("resume_from=") for item in pilot))

    def test_entire_capacity_and_pilot_commands_compose_under_hydra_struct_schema(self):
        paths = self.directory / "launch paths with spaces"
        for capacity in (True, False):
            with self.subTest(capacity=capacity):
                command = pipeline.online_command(paths / "online", paths / "data.npz",
                    paths / "imitation" / "latest.pt", 107, capacity)
                config_index = command.index("--config-name")
                self.assertEqual(command[:config_index], [sys.executable, "-u", "train.py"])
                with pipeline.hydra.initialize_config_dir(version_base=None,
                        config_dir=str(ROOT / "configs")):
                    config = pipeline.hydra.compose(command[config_index + 1],
                        overrides=command[config_index + 2:])
                self.assertEqual(config.seed, 107)
                self.assertEqual(config.logdir, (paths / "online").as_posix())
                self.assertEqual(config.imitation.dataset, (paths / "data.npz").as_posix())
                self.assertEqual(config.init_from, (paths / "imitation" / "latest.pt").as_posix())
                self.assertEqual((config.env.env_num, config.buffer.max_size), (128, 131072))
                self.assertTrue(config.trainer.eval_video)
                self.assertTrue(config.trainer.eval_video_save_mp4)
                self.assertFalse(config.trainer.eval_video_final_only)
                self.assertEqual(config.trainer.stop_after_optimizer_updates, 32 if capacity else 0)
                self.assertEqual(config.imitation.pretrain_updates, 32 if capacity else 1000)
                self.assertTrue(config.env.mjx_full_command)
                self.assertEqual(config.env.train_ratio, 64)

    def test_dataset_admission_binds_video_seeds_masks_physics_and_source(self):
        path = self.dataset()
        report = pipeline.validate_demonstrations(path, 24, 0)
        self.assertFalse(report["walking_teacher_qualified"])
        self.assertEqual(report["fingerprint"], ImitationDataset(path).fingerprint)
        self.assertEqual(len(report["video_records"]), 3)
        self.assertEqual(report["training_contract"]["task"]["episode_horizon_control_steps"], 1000)

    def test_dataset_admission_rejects_missing_masks_provenance_limits_seeds_or_video(self):
        changes = {
            "all_unsafe": (unsafe_rollouts, None),
            "no_safe_start": (lambda arrays: arrays["valid"][:, 0].fill(False), None),
            "unsafe_bc": (lambda arrays: (arrays["cause_physical_fall"][0, 399:400].fill(True),
                                           arrays["episode_safe"][0:1].fill(False)), None),
            "action_mapping": (lambda arrays: arrays["action"][0, 0, 0:1].fill(.5), None),
            "applied_control": (lambda arrays: arrays["applied_ctrl"][0, 0, 0:1].fill(.5), None),
            "wrong_length": (lambda arrays: arrays["episode_length"][0:1].fill(399), None),
            "missing_audit": (lambda arrays: arrays.pop("state_actuator_force"), None),
            "bad_provenance": (None, lambda manifest: manifest["provenance"].pop("full_command.py")),
            "bad_torque": (None, lambda manifest: manifest["action_contract"].update(peak_torque_kgcm=99.)),
            "bad_clock": (None, lambda manifest: manifest["action_contract"].update(control_dt_seconds=.04)),
            "bad_schedule": (None, lambda manifest: manifest["schedule"][1].update(seed=0)),
            "missing_video": (None, lambda manifest: manifest["video_evidence"].pop()),
            "bad_reset": (None, lambda manifest: manifest["reset_validation"].update(passed=False)),
        }
        for name, (arrays, manifest) in changes.items():
            with self.subTest(change=name):
                path = self.dataset(name, arrays, manifest)
                with self.assertRaises(ValueError):
                    pipeline.validate_demonstrations(path, 24, 0)
                if name == "all_unsafe":
                    with self.assertRaisesRegex(ValueError, "No safe expert labels"):
                        ImitationDataset(path).numpy_batch(2, 64, burnin=64)

    def test_dataset_admission_rejects_video_hash_or_sidecar_mismatch(self):
        for name in ("hash", "sidecar"):
            with self.subTest(change=name):
                path = self.dataset(name)
                video = path.parent / "expert_seed_0000000_split_0.mp4"
                if name == "hash":
                    video.write_bytes(b"replaced video")
                else:
                    video.with_suffix(".json").write_text("{}", encoding="utf-8")
                with self.assertRaises(ValueError):
                    pipeline.validate_demonstrations(path, 24, 0)

    def test_evaluation_admission_accepts_non_rsi_initial_and_final_full_protocol(self):
        pipeline.validate_evaluations([evaluation_row(self.contract, 0), evaluation_row(self.contract, 1000)], 1000, self.contract)
        pipeline.validate_evaluations([evaluation_row(self.contract, 0)], 0, self.contract, fresh=False)

    def test_evaluation_admission_rejects_wrong_protocol_trials_diagnostics_or_checkpoint_identity(self):
        changes = {
            "wrong_rsi": lambda row: row["protocol"]["environment"].update(rsi_stratified=True),
            "old_probabilities": lambda row: row["protocol"]["environment"].update(rsi_probabilities=[1., 0., 0., 0.]),
            "phase": lambda row: row["protocol"]["environment"].update(phase_observations=True),
            "world_count": lambda row: row["protocol"]["environment"].update(worlds=8),
            "duplicate_world": lambda row: row["trials"][31].update(evaluation_world=30),
            "missing_trial": lambda row: row["trials"].pop(),
            "missing_diagnostic": lambda row: row["trials"][31].pop("torque_saturation_fraction_mean"),
            "missing_cause": lambda row: row["trials"][31].pop("termination_reasons"),
            "nonfinite": lambda row: row["trials"][31].update(alive_seconds=float("nan")),
            "step": lambda row: row.update(step=999),
            "contract": lambda row: row["training_contract"]["task"].update(episode_horizon_control_steps=999),
        }
        for name, change in changes.items():
            with self.subTest(change=name):
                rows = [evaluation_row(deepcopy(self.contract), 0), evaluation_row(deepcopy(self.contract), 1000)]
                change(rows[-1])
                with self.assertRaises(ValueError):
                    pipeline.validate_evaluations(rows, 1000, self.contract)
        with self.assertRaises(ValueError):
            pipeline.validate_evaluations([evaluation_row(self.contract, 1000)], 1000, self.contract)

    def test_imitation_admission_requires_actual_finite_updates_and_heldout_improvement(self):
        fingerprint = {"data_sha256": "data", "manifest_sha256": "sidecar"}
        report = dict(status="completed_bounded_imitation", updates=32, changed_tensors_first_update=4,
            validation_improved=True, final_validation_improved=True, initial_validation_bc=.2,
            final_validation_bc=.1, latest_checkpoint_sha256="latest", dataset_fingerprint=fingerprint,
            best_validation_bc=.1, walking_qualified=False)
        for split in ("validation", "test"):
            report["full_episode_" + split] = dict(protocol="offline_teacher_forced_full_episode_v1",
                split=split, labels=100, optimizer_updates=32, deterministic_posterior=True, walking_qualified=False)
        checkpoint = dict(training_contract=self.contract, finite_agent=True, step=0, optimizer_updates=32,
            imitation=dict(version=1, mode="full_command_bc", pretraining_updates=32, dataset_fingerprint=fingerprint,
                last_validation={"imitation/bc": .1}, final_validation_bc=.1, full_episode_validation=report["full_episode_validation"],
                full_episode_test=report["full_episode_test"]),
            model_config={"encoder": {"mlp_keys": "^proprio$"}})
        pipeline.validate_imitation(report, checkpoint, fingerprint, 32, self.contract, "latest")
        for changed, value in (("updates", 31), ("changed_tensors_first_update", 0), ("validation_improved", False),
                               ("best_validation_bc", float("nan")), ("final_validation_bc", .3),
                               ("latest_checkpoint_sha256", "other"), ("final_validation_improved", False),
                               ("full_episode_test", {})):
            with self.subTest(report=changed):
                bad = dict(report, **{changed: value})
                with self.assertRaises(ValueError):
                    pipeline.validate_imitation(bad, checkpoint, fingerprint, 32, self.contract, "latest")
        for changed, value in (("finite_agent", False), ("optimizer_updates", 31), ("step", 1), ("model_config", None)):
            with self.subTest(checkpoint=changed):
                bad = dict(checkpoint, **{changed: value})
                with self.assertRaises(ValueError):
                    pipeline.validate_imitation(report, bad, fingerprint, 32, self.contract, "latest")

    def test_historical_best_cannot_admit_degraded_latest_checkpoint(self):
        report = dict(status="completed_bounded_imitation", updates=32, changed_tensors_first_update=4,
            validation_improved=True, final_validation_improved=True, initial_validation_bc=.2,
            best_validation_bc=.05, final_validation_bc=.3, latest_checkpoint_sha256="latest",
            dataset_fingerprint={"data_sha256": "data", "manifest_sha256": "sidecar"})
        with self.assertRaises(ValueError):
            pipeline.validate_imitation(report, {}, report["dataset_fingerprint"], 32, self.contract, "latest")

    def test_interrupted_runtime_records_terminal_status_and_never_starts_learning(self):
        output = self.directory / "interrupted"
        with patch.object(sys, "argv", ["pipeline", "--mode", "preflight", "--output", str(output)]), \
             patch("scripts.verify_script_free_bundle.verify"), \
             patch.object(pipeline, "stage", side_effect=pipeline.PipelineStopped("SIGINT")), \
             patch.object(pipeline, "learning_run") as learner, redirect_stdout(io.StringIO()):
            with self.assertRaises(pipeline.PipelineStopped):
                pipeline.main()
        learner.assert_not_called()
        self.assertEqual(pipeline.read_json(output / "run_status.json")["status"], "interrupted")
        self.assertEqual(pipeline.read_json(output / "readiness_report.json")["status"], "interrupted")
        self.assertIsNone(pipeline.RUN_JOURNAL)

    def zero_step_videos(self, name):
        output = self.directory / name
        directory = output / "eval_videos"
        directory.mkdir(parents=True)
        for world in range(32):
            path = directory / ("step_0000000" + ("" if world == 0 else f"_env_{world:02d}") + ".mp4")
            path.write_bytes(b"CPU fixture: not encoded video pixels")
            path.with_suffix(".json").write_text(json.dumps(dict(training_step=0, evaluation_world=world,
                start_pose_id=0, frames=10, fps=25)), encoding="utf-8")
        return output

    def test_zero_step_video_admission_requires_all_32_typed_nonempty_identities(self):
        output = self.zero_step_videos("video_good")
        self.assertEqual(pipeline.validate_zero_step_videos(output)["worlds"], 32)
        mutations = {"wrong_step": ("training_step", 1), "bool_step": ("training_step", False),
            "wrong_world": ("evaluation_world", 1), "float_world": ("evaluation_world", 0.),
            "bool_pose": ("start_pose_id", False), "zero_frames": ("frames", 0),
            "bool_frames": ("frames", True), "float_frames": ("frames", 10.), "wrong_fps": ("fps", 24)}
        for name, (key, value) in mutations.items():
            with self.subTest(change=name):
                output = self.zero_step_videos(name)
                sidecar = output / "eval_videos" / "step_0000000.json"
                record = pipeline.read_json(sidecar)
                record[key] = value
                sidecar.write_text(json.dumps(record), encoding="utf-8")
                with self.assertRaises(ValueError):
                    pipeline.validate_zero_step_videos(output)
        output = self.zero_step_videos("missing_world")
        (output / "eval_videos" / "step_0000000_env_31.mp4").unlink()
        with self.assertRaises(ValueError):
            pipeline.validate_zero_step_videos(output)
        output = self.zero_step_videos("empty_video")
        (output / "eval_videos" / "step_0000000.mp4").write_bytes(b"")
        with self.assertRaises(ValueError):
            pipeline.validate_zero_step_videos(output)

    def test_default_mode_is_structure_only_and_existing_run_is_never_reused(self):
        stream = io.StringIO()
        with patch.object(sys, "argv", ["pipeline"]), patch.object(pipeline, "stage", side_effect=AssertionError("Unexpected GPU stage")), redirect_stdout(stream):
            pipeline.main()
        report = json.loads(stream.getvalue())
        self.assertFalse(report["gpu_tested"])
        self.assertFalse(report["walking_qualified"])
        with patch.object(sys, "argv", ["pipeline", "--mode", "preflight", "--output", str(self.directory)]), redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                pipeline.main()

    def test_preflight_runs_runtime_then_capacity_without_automatic_pilot(self):
        output = self.directory / "preflight"
        def stage(command, log, timeout):
            if "scripts/check_script_free_runtime.py" not in command:
                raise AssertionError("Unexpected stage")
            pipeline.write_json(output / "runtime" / "gpu_runtime_report.json",
                dict(status="passed_pinned_gpu_student_runtime_only", reset_validation=dict(passed=True),
                     walking_qualified=False, transition_finite=True, reference_loaders_poisoned=True,
                     reference_lookup_poisoned=True, learned_policy_evaluated=False))
            return "runtime complete"
        with patch.object(sys, "argv", ["pipeline", "--mode", "preflight", "--output", str(output)]), \
             patch("scripts.verify_script_free_bundle.verify") as verifier, \
             patch.object(pipeline, "stage", side_effect=stage), \
             patch.object(pipeline, "learning_run", return_value={"status": "capacity_only"}) as learner, \
             redirect_stdout(io.StringIO()):
            pipeline.main()
        verifier.assert_called_once_with(ROOT.parent / "SOURCE_MANIFEST.json")
        learner.assert_called_once_with(output / "capacity", 0, capacity=True)
        report = pipeline.read_json(output / "readiness_report.json")
        self.assertEqual(report["status"], "passed_gpu_preflight_only")
        self.assertIsNone(report["pilot_report"])
        self.assertFalse(report["walking_qualified"])
        self.assertFalse(report["automatic_extension"])

    def test_failed_runtime_never_admits_capacity_or_retries(self):
        output = self.directory / "failed"
        with patch.object(sys, "argv", ["pipeline", "--mode", "preflight", "--output", str(output)]), \
             patch("scripts.verify_script_free_bundle.verify"), \
             patch.object(pipeline, "stage", side_effect=RuntimeError("bounded runtime failed")) as stage, \
             patch.object(pipeline, "learning_run") as learner, redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "bounded runtime failed"):
                pipeline.main()
        stage.assert_called_once()
        learner.assert_not_called()
        report = pipeline.read_json(output / "readiness_report.json")
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["walking_qualified"])

    def test_no_safe_demonstrations_stop_before_pretraining_or_online_stages(self):
        unsafe = self.dataset("unsafe", unsafe_rollouts)
        validator = pipeline.validate_demonstrations
        with patch.object(pipeline, "stage", return_value="collection complete") as stage, \
             patch.object(pipeline, "validate_demonstrations", side_effect=lambda *args: validator(unsafe, 24, 0)):
            with self.assertRaisesRegex(ValueError, "Insufficient safe"):
                pipeline.learning_run(self.directory / "blocked_learning", 0, capacity=True)
        stage.assert_called_once()
        self.assertIn("scripts/collect_script_free_demonstrations.py", stage.call_args.args[0])

    def test_failed_capacity_never_extends_even_explicit_pilot_mode(self):
        output = self.directory / "blocked_pilot"
        def stage(command, log, timeout):
            pipeline.write_json(output / "runtime" / "gpu_runtime_report.json",
                dict(status="passed_pinned_gpu_student_runtime_only", reset_validation=dict(passed=True),
                     walking_qualified=False, transition_finite=True, reference_loaders_poisoned=True,
                     reference_lookup_poisoned=True, learned_policy_evaluated=False))
            return "runtime complete"
        with patch.object(sys, "argv", ["pipeline", "--mode", "pilot", "--output", str(output)]), \
             patch("scripts.verify_script_free_bundle.verify"), patch.object(pipeline, "stage", side_effect=stage), \
             patch.object(pipeline, "learning_run", side_effect=ValueError("No safe demonstrations")) as learner, \
             redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "No safe demonstrations"):
                pipeline.main()
        learner.assert_called_once_with(output / "capacity", 0, capacity=True)
        self.assertEqual(pipeline.read_json(output / "readiness_report.json")["status"], "failed")

    def test_runtime_worker_rejects_nonfinite_proprio_reward_state_or_diagnostics(self):
        from scripts import check_script_free_runtime as runtime
        from envs.mjx_walking import scene
        contract = self.contract
        for faulty in ("proprio", "reward", "audit", "logs"):
            with self.subTest(field=faulty):
                providers = []
                class Provider:
                    def __init__(self, config, worlds):
                        self.contract = contract
                        self.env = SimpleNamespace(raw=SimpleNamespace())
                        self.steps, self.closed = 0, False
                        providers.append(self)
                    def check_reset_repeatability(self, seed):
                        return dict(passed=True)
                    def reset(self, seed):
                        return dict(audit=dict(ctrl=np.zeros((8, 19), np.float32)))
                    def normalize(self, targets):
                        return targets
                    def step(self, action):
                        self.steps += 1
                        value = dict(proprio=np.zeros((8, 45), np.float32), reward=np.zeros(8),
                                     audit=dict(ctrl=np.zeros((8, 19))), logs=dict(physical_fall=np.zeros(8)))
                        if self.steps == 4:
                            if faulty in ("audit", "logs"):
                                next(iter(value[faulty].values())).flat[0] = np.nan
                            else:
                                value[faulty].flat[0] = np.nan
                        return value
                    def video_frame(self):
                        return np.resize(np.asarray([0, 255], np.uint8), (240, 320, 3))
                    def close(self):
                        self.closed = True
                with patch("scripts.preflight_reference_training.gpu_runtime_report", return_value={"scope": "CPU fixture"}), \
                     patch("scripts.collect_script_free_demonstrations.GPUStudentProvider", Provider), \
                     patch.object(scene, "scripted_gait_reference"), patch.object(scene, "vendor_gait_reference"), \
                     patch.object(scene, "build_scripted_rsi_pool"):
                    with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
                        runtime.check(0)
                self.assertTrue(providers[0].closed)
                self.assertEqual(providers[0].steps, 4)


if __name__ == "__main__":
    unittest.main()
