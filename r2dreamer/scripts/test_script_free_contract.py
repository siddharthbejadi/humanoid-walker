"""CPU contract/inference checks; mocked physics is not walking evidence."""
from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
import ast
import importlib.util
import io
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from full_command import motor_targets_from_actions, normalize_motor_targets
from script_free_contract import (check_imitation_source, validate_student_config,
                                  validate_student_environment)
from script_free_policy import ScriptFreePolicy
from scripts.test_training_contract import make_environment
from training_contract import build_training_contract


def _full_environment(root):
    env, config = make_environment(root)
    raw = env.raw
    raw.m.actuator_ctrlrange[:] = [-1.2, 1.6]
    raw.m.actuator_forcerange[:] = [-1.765197, 1.765197]
    raw.standing_ctrl = np.full(19, .03, np.float32)
    raw.command_center = raw.m.actuator_ctrlrange.mean(axis=1)
    raw.command_scale = np.diff(raw.m.actuator_ctrlrange, axis=1)[:, 0] / 2.
    raw.action_scale = raw.command_scale.copy()
    raw.full_command = True
    raw.controller_identity = "autonomous_full_command_v1"
    raw.gait_residual = raw.phase_obs = raw.gait_curriculum = raw.rsi_stratified = False
    raw.phase_rate_range = 0.
    raw.gait_reference_kind = "none"
    raw.stabilizer = raw.footsteps = None
    raw.rsi_source_kind = "standing"
    raw.reset_joint_noise_rad = raw.reset_velocity_noise = .005
    raw.target_forward_speed = .08
    raw.reward_profile = "autonomous_walk_v1"
    raw.step_clearance_m = .015
    raw.control_slew_rad_s = 8.
    env.horizon = 1000
    env.observation_space = NS(spaces={"proprio": NS(shape=(45,))})
    config.pose_rsi_json = None
    return env, config


def full_command_contract():
    """Canonical effective fixture built by the production contract serializer."""
    with TemporaryDirectory() as directory:
        env, config = _full_environment(Path(directory))
        return build_training_contract(env, config)


def student_config():
    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return hydra.compose("walking_mjx_script_free")


def change(value, path, replacement):
    result = deepcopy(value)
    names = path.split(".")
    current = result
    for name in names[:-1]:
        current = current[name]
    current[names[-1]] = replacement
    return result


def imitation_metadata(updates=1000):
    return dict(version=1, mode="full_command_bc", pretraining_updates=updates,
                dataset_fingerprint={"data_sha256": "a" * 64, "manifest_sha256": "b" * 64})


def checkpoint(contract=None):
    return dict(training_contract=full_command_contract() if contract is None else contract,
                imitation=imitation_metadata(), agent_state_dict={"weight": torch.ones(1)},
                model_config={"device": "cpu", "encoder": {"mlp_keys": "^proprio$", "cnn_keys": "$^"},
                              "auxiliary_state": False})


class TinyInferenceAgent(torch.nn.Module):
    """Exercise the real loader/wrapper without training a synthetic teacher."""
    def __init__(self, config, observations, actions):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.device = torch.device(config.device)
        assert set(observations.spaces) == {"proprio"}
        assert observations.spaces["proprio"].shape == (45,)
        assert actions.shape == (19,)
        self.seen = []

    def clone_and_freeze(self):
        return None

    def get_initial_state(self, worlds):
        return TensorDict({"prev_action": torch.zeros(worlds, 19, device=self.device)}, batch_size=[worlds])

    def act(self, observation, state, eval=False):
        self.seen.append((set(observation.keys()), observation["is_first"].clone(), eval))
        action = self.weight * torch.full((state.batch_size[0], 19), .2, device=self.device)
        return action, TensorDict({"prev_action": action}, batch_size=state.batch_size)


class ScriptFreeContractTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.contract = full_command_contract()
        self.dataset = NS(manifest={"training_contract": deepcopy(self.contract)},
                          fingerprint=imitation_metadata()["dataset_fingerprint"])

    def test_shipped_config_is_sensor_only_and_script_independent(self):
        config = student_config()
        result = validate_student_config(config)
        self.assertEqual(result["encoder_inputs"], ["proprio"])
        self.assertFalse(result["trajectory_playback"])
        self.assertFalse(result["explicit_stabilizer"])
        self.assertEqual(config.env.mjx_reset_joint_noise_rad, .005)
        self.assertEqual(config.env.mjx_reset_velocity_noise, .005)

    def test_config_rejects_every_runtime_reference_and_actor_leak(self):
        config = student_config()
        poisons = {
            "env.mjx_full_command": False, "env.mjx_scripted_gait": "missing.npz",
            "env.mjx_scripted_phase_pool": "missing.npz", "env.gait_reference_csv": "missing.csv",
            "env.gait_reference_signs": [1.] * 19, "env.pose_rsi_json": "missing.json",
            "env.mjx_stabilizer": [1., .04, -1., .25], "env.mjx_gait_residual": True,
            "env.mjx_phase_obs": True, "env.mjx_phase_control": .2, "env.rsi_stratified": True,
            "env.mjx_footsteps": True, "env.encoder_training_aid": True,
            "env.auxiliary_state_targets": True, "model.auxiliary_state": True,
            "env.mjx_reward_profile": "goal_corridor_v3", "env.encoder.mlp_keys": ".*",
            "model.encoder.mlp_keys": ".*", "model.encoder.cnn_keys": "image",
            "allow_unverified_resume": True, "transfer_from": "legacy.pt",
        }
        for path, replacement in poisons.items():
            with self.subTest(field=path):
                poisoned = deepcopy(config)
                OmegaConf.update(poisoned, path, replacement)
                with self.assertRaises(ValueError):
                    validate_student_config(poisoned)

    def test_config_rejects_context_exploration_and_nonfinite_weights(self):
        config = student_config()
        for path, replacement in (
            ("imitation.batch_size", 8), ("imitation.batch_length", 32), ("imitation.burn_in", 0),
            ("model.actor.dist.cont.min_std", .01), ("model.actor.dist.cont.max_std", .2),
            ("env.mjx_phase_control", float("nan")),
            ("model.imitation_weight", 0.), ("model.imitation_pretrain_weight", -1.),
            ("model.imitation_weight", float("nan")), ("model.imitation_pretrain_weight", float("inf"))):
            with self.subTest(field=path, value=replacement):
                poisoned = deepcopy(config)
                OmegaConf.update(poisoned, path, replacement)
                with self.assertRaises(ValueError):
                    validate_student_config(poisoned)

    def test_contract_serializes_actual_midpoint_halfspan_mapping(self):
        self.assertIs(validate_student_environment(self.contract), self.contract)
        control = self.contract["control"]
        ranges = np.asarray(control["actuator_ctrlrange_rad"])
        center, scale = np.asarray(control["command_center_rad"]), np.asarray(control["action_scale_rad"])
        np.testing.assert_allclose(center, ranges.mean(axis=1))
        np.testing.assert_allclose(scale, np.diff(ranges, axis=1)[:, 0] / 2.)
        target = ranges[:, 0] + .7 * np.diff(ranges, axis=1)[:, 0]
        np.testing.assert_allclose(motor_targets_from_actions(normalize_motor_targets(target, center, scale), center, scale), target)
        self.assertNotEqual(control["command_center_rad"], control["standing_ctrl_rad"])
        self.assertIn("full_command.py", self.contract["environment_code"])
        self.assertEqual(self.contract["task"]["episode_horizon_control_steps"], 1000)

    def test_contract_rejects_mislabeled_absolute_normalization(self):
        for path, replacement in (
            ("control.action_mode", "residual_offset"), ("control.command_center_rad", [0.] * 19),
            ("control.action_scale_rad", [.1] * 19), ("control.action_scale_rad", [0.] * 19),
            ("control.action_scale_rad", [-1.] * 19), ("control.command_center_rad", [0.] * 18),
            ("control.actuator_ctrlrange_rad", [[0., 0.]] * 19),
            ("control.action_scale_rad", [float("nan")] * 19),
            ("control.command_center_rad", [float("inf")] * 19)):
            with self.subTest(field=path):
                with self.assertRaises(ValueError):
                    validate_student_environment(change(self.contract, path, replacement))

    def test_contract_rejects_controller_observation_and_reference_dependencies(self):
        for path, replacement in (
            ("interface.observations", {"proprio": [45], "teacher_state": [98]}),
            ("interface.action_shape", [18]), ("controller.full_command", False),
            ("controller.identity", "reference_or_standing_relative"),
            ("controller.observation_semantics", "legacy_teacher_phase_navigation_v1"),
            ("controller.gait_residual", True), ("controller.reference_kind", "scripted"),
            ("controller.stabilizer", [1., .04, -1., .25]), ("controller.phase_observations", True),
            ("controller.phase_rate_range", .2), ("controller.mjx_scripted_gait_sha256", "a" * 64),
            ("controller.gait_reference_csv_sha256", "a" * 64),
            ("controller.mjx_scripted_phase_pool_sha256", "a" * 64),
            ("controller.effective_reference_sha256", "a" * 64)):
            with self.subTest(field=path):
                with self.assertRaises(ValueError):
                    validate_student_environment(change(self.contract, path, replacement))

    def test_contract_rejects_changed_reset_task_clock_and_invalid_horizon(self):
        for path, replacement in (
            ("task.name", "standing"), ("task.reward_profile", "legacy"),
            ("task.gait_curriculum", True), ("task.footsteps", {}),
            ("task.valid_step_clearance_m", .020), ("task.episode_horizon_control_steps", 0),
            ("task.valid_step_clearance_comparison", "greater_than"),
            ("task.episode_horizon_control_steps", True), ("task.episode_horizon_control_steps", 1000.),
            ("control.control_dt_seconds", .04), ("control.action_repeat", 20),
            ("control.slew_rad_per_second", 2.), ("reset.rsi_stratified", True),
            ("reset.source_kind", "scripted_phase"), ("reset.joint_noise_rad", .01),
            ("reset.velocity_noise_rad_s", 0.)):
            with self.subTest(field=path):
                with self.assertRaises(ValueError):
                    validate_student_environment(change(self.contract, path, replacement))

    def test_contract_rejects_unknown_schema_backend_and_nonfinite_nested_values(self):
        for path, replacement in (("version", 99), ("backend", "legacy_cpu"),
                                  ("physics.timestep_seconds", float("nan"))):
            with self.subTest(field=path):
                with self.assertRaises(ValueError):
                    validate_student_environment(change(self.contract, path, replacement))

    def test_dataset_requires_exact_effective_contract(self):
        metadata = check_imitation_source(self.dataset, self.contract)
        self.assertEqual(metadata["dataset_fingerprint"], self.dataset.fingerprint)
        for path, replacement in (("physics.contacts_per_world", 512),
                                  ("task.episode_horizon_control_steps", 1500),
                                  ("task.target_forward_speed_m_s", .1),
                                  ("control.actuator_forcerange_nm", [[-1., 1.]] * 19)):
            with self.subTest(field=path):
                changed = NS(manifest={"training_contract": change(self.contract, path, replacement)},
                             fingerprint=self.dataset.fingerprint)
                with self.assertRaisesRegex(ValueError, "contract mismatch"):
                    check_imitation_source(changed, self.contract)

    def test_checkpoint_and_dataset_fingerprints_must_match(self):
        saved = checkpoint(self.contract)
        self.assertEqual(check_imitation_source(self.dataset, self.contract, saved, minimum_updates=1000), saved["imitation"])
        for field in ("data_sha256", "manifest_sha256"):
            altered = deepcopy(saved)
            altered["imitation"]["dataset_fingerprint"][field] = "c" * 64
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "fingerprints disagree"):
                check_imitation_source(self.dataset, self.contract, altered)
        altered = deepcopy(saved)
        altered["training_contract"]["task"]["episode_horizon_control_steps"] = 1500
        with self.assertRaisesRegex(ValueError, "episode_horizon_control_steps"):
            check_imitation_source(self.dataset, self.contract, altered)

    def test_checkpoint_rejects_missing_wrong_or_insufficient_imitation_metadata(self):
        for metadata in (None, {}, {"version": 2}, imitation_metadata(0), imitation_metadata(True),
                         imitation_metadata(999), {**imitation_metadata(), "mode": "residual_bc"}):
            with self.subTest(metadata=metadata):
                saved = checkpoint(self.contract)
                saved["imitation"] = metadata
                with self.assertRaises(ValueError):
                    check_imitation_source(self.dataset, self.contract, saved, minimum_updates=1000)

    def test_inference_checks_metadata_before_constructing_model(self):
        saved = checkpoint(self.contract)
        invalid = [change(saved, "training_contract.controller.full_command", False),
                   change(saved, "imitation.pretraining_updates", 0),
                   change(saved, "imitation.pretraining_updates", True),
                   change(saved, "model_config.encoder.mlp_keys", ".*"),
                   change(saved, "model_config.encoder.cnn_keys", "image")]
        for poisoned in invalid:
            with patch("script_free_policy.Dreamer", side_effect=AssertionError("model constructor reached")):
                with self.assertRaises(ValueError):
                    ScriptFreePolicy(poisoned)

    def test_checkpoint_only_inference_loads_no_reference_or_dataset(self):
        with patch("script_free_policy.Dreamer", TinyInferenceAgent), \
             patch("numpy.load", side_effect=AssertionError("reference/data file loader reached")):
            policy = ScriptFreePolicy(checkpoint(self.contract))
            action = policy.act(np.zeros((2, 45), np.float32))
            self.assertEqual(tuple(action.shape), (2, 19))
            self.assertTrue(policy.agent.seen[-1][1].all())
            self.assertEqual(policy.agent.seen[-1][0], {"proprio", "is_first"})
            self.assertTrue(policy.agent.seen[-1][2])
            policy.act(np.zeros((2, 45), np.float32))
            self.assertFalse(policy.agent.seen[-1][1].any())

    def test_inference_rejects_extra_inputs_nonfinite_weights_and_bad_actions(self):
        with patch("script_free_policy.Dreamer", TinyInferenceAgent):
            saved = checkpoint(self.contract)
            saved["agent_state_dict"]["weight"] = torch.tensor([float("nan")])
            with self.assertRaisesRegex(ValueError, "nonfinite weights"):
                ScriptFreePolicy(saved)
            policy = ScriptFreePolicy(checkpoint(self.contract))
            for observation in (np.zeros((1, 46)), np.full((1, 45), np.nan), np.zeros(45)):
                with self.assertRaises(ValueError):
                    policy.act(observation)
            policy.act(np.zeros((2, 45)))
            with self.assertRaisesRegex(ValueError, "world count"):
                policy.act(np.zeros((1, 45)))
            with self.assertRaisesRegex(ValueError, "worlds-by-1"):
                policy.act(np.zeros((2, 45)), is_first=np.zeros(2))
            with torch.no_grad():
                policy.agent.weight.fill_(10.)
            with self.assertRaisesRegex(RuntimeError, "invalid full motor actions"):
                policy.act(np.zeros((2, 45)))

    def test_evaluation_poison_order_and_explicit_horizon_extension(self):
        spec = importlib.util.spec_from_file_location("test_evaluate_script_free", ROOT / "scripts/evaluate_script_free.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        records = []
        scene = ModuleType("envs.mjx_walking.scene")
        loader_names = ("scripted_gait_reference", "vendor_gait_reference", "build_scripted_rsi_pool")
        for name in loader_names:
            setattr(scene, name, lambda *args, **kwargs: records.append("unpoisoned_loader"))
        package = ModuleType("envs")
        package.__path__ = []
        walking = ModuleType("envs.mjx_walking")
        walking.__path__ = []
        walking.scene = scene
        package.mjx_walking = walking

        class FakeEnv:
            def __init__(self, config, worlds, evaluation):
                for name in loader_names:
                    with self_test.assertRaisesRegex(RuntimeError, "scripted trajectory"):
                        getattr(scene, name)()
                self.horizon, self.env_num = 1000, worlds
                self.raw = NS(control_dt=.02)
                records.append("environment_constructed_after_loader_poison")

            def evaluation_protocol(self):
                return {"horizon_control_steps": self.horizon}

            def close(self):
                records.append("closed")

        class FakePolicy:
            def __init__(self, path, device):
                self.contract = deepcopy(self_test.contract)
                self.imitation = imitation_metadata()
                self.agent = object()

        class FakeTrainer:
            def __init__(self, settings, unused, logger, output, train_env, eval_env):
                records.append(("evaluation_limit", settings.eval_max_steps, eval_env.horizon))
                self.env = eval_env

            def eval(self, agent, step):
                with self_test.assertRaisesRegex(RuntimeError, "scripted trajectory"):
                    self.env.raw._reference_at()
                return {"test_scope": "mocked admission/order only"}

        def build(env, config):
            records.append(("contract_horizon", env.horizon))
            return change(self.contract, "task.episode_horizon_control_steps", env.horizon)

        self_test = self
        bridge = ModuleType("envs.mjx_teacher_env")
        bridge.MJXTeacherEnv = FakeEnv
        trainer = ModuleType("trainer")
        trainer.OnlineTrainer = FakeTrainer
        logger = ModuleType("tools")
        logger.Logger = lambda path: NS(write=lambda step: records.append(("logger_step", step)))
        policy_module = ModuleType("script_free_policy")
        policy_module.ScriptFreePolicy = FakePolicy
        modules = {"envs": package, "envs.mjx_teacher_env": bridge, "envs.mjx_walking": walking,
                   "envs.mjx_walking.scene": scene, "trainer": trainer, "tools": logger,
                   "script_free_policy": policy_module}
        with TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "student.pt"
            checkpoint_path.write_bytes(b"mocked checkpoint")
            output = Path(directory) / "evaluation"
            args = ["evaluate_script_free", "--checkpoint", str(checkpoint_path), "--output", str(output),
                    "--seconds", "30", "--worlds", "2"]
            with patch.dict(sys.modules, modules), patch.object(sys, "argv", args), \
                 patch("training_contract.build_training_contract", side_effect=build), \
                 patch("torch.load", return_value={"step": 7}), redirect_stdout(io.StringIO()):
                module.main()
            self.assertTrue((output / "script_free_evaluation.json").is_file())
        self.assertNotIn("unpoisoned_loader", records)
        self.assertEqual([row for row in records if isinstance(row, tuple) and row[0] == "contract_horizon"],
                         [("contract_horizon", 1000), ("contract_horizon", 1500)])
        self.assertIn(("evaluation_limit", 1500, 1500), records)
        self.assertEqual(records[-1], "closed")

    def _runtime_fixture(self, blank=False, transition_fault=None):
        spec = importlib.util.spec_from_file_location("test_check_script_free_runtime", ROOT / "scripts/check_script_free_runtime.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        records, self_test = [], self
        scene = ModuleType("envs.mjx_walking.scene")
        loader_names = ("scripted_gait_reference", "vendor_gait_reference", "build_scripted_rsi_pool")
        for name in loader_names:
            setattr(scene, name, lambda *args, **kwargs: records.append("unpoisoned_loader"))
        package = ModuleType("envs")
        package.__path__ = []
        walking = ModuleType("envs.mjx_walking")
        walking.__path__ = []
        walking.scene = scene
        package.mjx_walking = walking

        class FakeProvider:
            def __init__(self, config, worlds):
                for name in loader_names:
                    with self_test.assertRaisesRegex(AssertionError, "trajectory or phase pool"):
                        getattr(scene, name)()
                self.contract = deepcopy(self_test.contract)
                self.env = NS(raw=NS(_reference_at=None))
                self.worlds = worlds
                records.append("constructed_after_loader_poison")

            def check_reset_repeatability(self, seed):
                records.append("retained_view_and_reset_check")
                return {"passed": True, "scope": "mocked reset; no physics"}

            def reset(self, seed):
                return {"audit": {"ctrl": np.full((self.worlds, 19), .03, np.float32)}}

            def normalize(self, controls):
                return normalize_motor_targets(controls, np.asarray(self.contract["control"]["command_center_rad"]),
                                               np.asarray(self.contract["control"]["action_scale_rad"]))

            def step(self, actions):
                self_test.assertEqual(actions.shape, (8, 19))
                with self_test.assertRaisesRegex(AssertionError, "trajectory or phase pool"):
                    self.env.raw._reference_at()
                records.append("ordinary_step_after_lookup_poison")
                transition = {"proprio": np.zeros((8, 45), np.float32),
                              "reward": np.zeros((8, 1), np.float32),
                              "audit": {"ctrl": np.full((8, 19), .03, np.float32),
                                        "qpos": np.zeros((8, 26), np.float32)},
                              "logs": {"valid_left_step": np.zeros((8, 1), np.float32)}}
                if records.count("ordinary_step_after_lookup_poison") == 3:
                    if transition_fault == "missing_reward":
                        del transition["reward"]
                    elif transition_fault in ("proprio", "reward"):
                        transition[transition_fault][0, 0] = np.nan
                    elif transition_fault == "audit":
                        transition["audit"]["qpos"][0, 0] = np.nan
                    elif transition_fault == "logs":
                        transition["logs"]["valid_left_step"][0, 0] = np.nan
                return transition

            def video_frame(self):
                records.append("rendered")
                if blank:
                    return np.zeros((240, 320, 3), np.uint8)
                values = np.tile(np.arange(320, dtype=np.uint16) % 256, (240, 1)).astype(np.uint8)
                return np.stack([values] * 3, axis=-1)

            def close(self):
                records.append("closed")

        modules = {"envs": package, "envs.mjx_walking": walking, "envs.mjx_walking.scene": scene}
        return module, FakeProvider, modules, records

    def test_runtime_uses_exact_guard_and_poison_before_construction(self):
        module, provider, modules, records = self._runtime_fixture()
        with patch.dict(sys.modules, modules), \
             patch("scripts.preflight_reference_training.gpu_runtime_report", return_value={"scope": "mocked GPU report"}), \
             patch("scripts.collect_script_free_demonstrations.GPUStudentProvider", provider):
            report = module.check(3)
        self.assertTrue(report["reference_loaders_poisoned"])
        self.assertTrue(report["reference_lookup_poisoned"])
        self.assertFalse(report["learned_policy_evaluated"])
        self.assertFalse(report["walking_qualified"])
        self.assertTrue(report["reset_validation"]["passed"])
        self.assertEqual(records.count("ordinary_step_after_lookup_poison"), 16)
        self.assertNotIn("unpoisoned_loader", records)
        self.assertEqual(records[-1], "closed")

    def test_runtime_rejects_blank_frame_and_closes_provider(self):
        module, provider, modules, records = self._runtime_fixture(blank=True)
        with patch.dict(sys.modules, modules), \
             patch("scripts.preflight_reference_training.gpu_runtime_report", return_value={"scope": "mocked GPU report"}), \
             patch("scripts.collect_script_free_demonstrations.GPUStudentProvider", provider):
            with self.assertRaisesRegex(RuntimeError, "blank"):
                module.check(3)
        self.assertEqual(records[-1], "closed")

    def test_runtime_rejects_each_nonfinite_transition_before_later_clean_steps(self):
        for field in ("proprio", "reward", "audit", "logs"):
            with self.subTest(field=field):
                module, provider, modules, records = self._runtime_fixture(transition_fault=field)
                with patch.dict(sys.modules, modules), \
                     patch("scripts.preflight_reference_training.gpu_runtime_report", return_value={"scope": "mocked GPU report"}), \
                     patch("scripts.collect_script_free_demonstrations.GPUStudentProvider", provider):
                    with self.assertRaisesRegex(RuntimeError, "Nonfinite student transition"):
                        module.check(3)
                self.assertEqual(records.count("ordinary_step_after_lookup_poison"), 3)
                self.assertNotIn("rendered", records)
                self.assertEqual(records[-1], "closed")

    def test_runtime_missing_mandatory_transition_field_fails_admission(self):
        module, provider, modules, records = self._runtime_fixture(transition_fault="missing_reward")
        with patch.dict(sys.modules, modules), \
             patch("scripts.preflight_reference_training.gpu_runtime_report", return_value={"scope": "mocked GPU report"}), \
             patch("scripts.collect_script_free_demonstrations.GPUStudentProvider", provider):
            with self.assertRaisesRegex(KeyError, "reward"):
                module.check(3)
        self.assertEqual(records.count("ordinary_step_after_lookup_poison"), 3)
        self.assertNotIn("rendered", records)
        self.assertEqual(records[-1], "closed")

    def test_pipeline_admits_actual_full_mode_protocol_not_a_fictional_rsi_mix(self):
        from scripts.run_script_free_pipeline import validate_evaluations

        source = ast.parse((ROOT / "envs/mjx_teacher_env.py").read_text(encoding="utf-8"))
        bridge = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "MJXTeacherEnv")
        method = next(node for node in bridge.body if isinstance(node, ast.FunctionDef) and node.name == "evaluation_protocol")
        namespace = {"np": np}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "actual_evaluation_protocol", "exec"), namespace)
        with TemporaryDirectory() as directory:
            env, _ = _full_environment(Path(directory))
            env.raw.observation_semantics = "sensor_command_imu_fsr_45_v1"
            env.env_num, env.full_command = 32, True
            env._evaluation_seed = 11000
            env._repeatable_evaluation, env._balanced_evaluation = True, False
            protocol = namespace["evaluation_protocol"](env)
        self.assertIsNone(protocol["rsi_probabilities"])
        self.assertFalse(protocol["rsi_stratified"])
        self.assertEqual(protocol["observation_keys"], ["proprio"])
        required = ("alive_seconds", "total_valid_steps", "valid_left_step", "valid_right_step", "alternating_steps",
                    "fall_count", "safety_failure_count", "wall_hit", "stall_out", "goal_reached", "time_limit",
                    "body_tilt_rad_peak", "root_drop_m_peak", "left_clearance_m_peak", "right_clearance_m_peak",
                    "torque_saturation_fraction_mean", "control_slew_limited_fraction_mean")
        trials = [{**dict.fromkeys(required, 0.), "start_pose_id": 0, "evaluation_world": world,
                   "alive_seconds": 20., "control_steps": 1000, "termination_reasons": ["time_limit"]}
                  for world in range(32)]
        initial = dict(step=0, schema_version=2, training_contract=deepcopy(self.contract),
                       protocol={"environment": protocol, "eval_max_control_steps": 1000}, trials=trials)
        final = {**deepcopy(initial), "step": 128000}
        validate_evaluations([initial, final], 128000, self.contract)
        for invalid in (change(final, "protocol.environment.rsi_stratified", True),
                        change(final, "protocol.environment.rsi_probabilities", [1., 0., 0., 0.]),
                        change(final, "protocol.environment.horizon_control_steps", 1500),
                        change(final, "protocol.environment.reset_joint_noise_rad", .01),
                        change(final, "protocol.environment.reset_velocity_noise", 0.),
                        change(final, "protocol.environment.observation_keys", ["proprio", "teacher_state"]),
                        change(final, "protocol.environment.reference_kind", "scripted"),
                        change(final, "protocol.environment.stabilizer_enabled", True),
                        change(final, "protocol.environment.phase_observations", True),
                        change(final, "protocol.eval_max_control_steps", 1500),
                        {**deepcopy(final), "step": 127000},
                        {**deepcopy(final), "trials": trials[:-1]}):
            with self.assertRaises(ValueError):
                validate_evaluations([initial, invalid], 128000, self.contract)
        missing_cause = deepcopy(final)
        del missing_cause["trials"][0]["time_limit"]
        with self.assertRaises(ValueError):
            validate_evaluations([initial, missing_cause], 128000, self.contract)
        repeated_world = deepcopy(final)
        repeated_world["trials"][-1]["evaluation_world"] = 0
        with self.assertRaises(ValueError):
            validate_evaluations([initial, repeated_world], 128000, self.contract)
        nonfinite_metric = deepcopy(final)
        nonfinite_metric["trials"][0]["left_clearance_m_peak"] = float("nan")
        with self.assertRaises(ValueError):
            validate_evaluations([initial, nonfinite_metric], 128000, self.contract)


if __name__ == "__main__":
    unittest.main()
