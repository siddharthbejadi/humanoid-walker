"""Real tiny PyTorch learner checks; synthetic commands are not walking demonstrations."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import hydra
import numpy as np
from gymnasium import spaces
from omegaconf import OmegaConf
import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dreamer import Dreamer
from imitation_training import imitation_objective, save_imitation_checkpoint
from scripts.test_script_free_contract import full_command_contract


def tiny_config(weight=0.):
    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        config = hydra.compose("walking_mjx_reference_pilot", overrides=[
            "device=cpu", "model.units=16", "model.deter=32", "model.hidden=16", "model.discrete=4",
            "model.rssm.stoch=2", "model.rssm.blocks=4", "model.encoder.mlp.layers=1",
            "model.actor.layers=1", "model.imag_horizon=2", "model.lr=0.003", "model.warmup=0",
            "model.kl_free=0.0", "env.encoder.mlp_keys='^proprio$'", "model.diagnostic_every=0",
            f"++model.imitation_weight={weight}", "++model.imitation_pretrain_weight=1"])
    return config


def agent(weight=0.):
    config = tiny_config(weight)
    obs = spaces.Dict({"proprio": spaces.Box(-np.inf, np.inf, (45,), np.float32)})
    actions = spaces.Box(-1., 1., (19,), np.float32)
    return Dreamer(config.model, obs, actions).to("cpu")


def data():
    values = dict(proprio=torch.randn(2, 8, 45) * .02, action=torch.zeros(2, 8, 19),
        expert_action=torch.full((2, 8, 19), .3), expert_mask=torch.ones(2, 8, 1),
        loss_mask=torch.ones(2, 8, 1), reward=torch.zeros(2, 8, 1),
        is_first=torch.zeros(2, 8, 1, dtype=torch.bool),
        is_terminal=torch.zeros(2, 8, 1, dtype=torch.bool), is_last=torch.zeros(2, 8, 1, dtype=torch.bool))
    values["is_first"][:, 0] = True
    values["expert_mask"][:, :2] = 0
    values["loss_mask"][:, :2] = 0
    return TensorDict(values, batch_size=(2, 8))


class Replay:
    def __init__(self, learner, batch):
        self.learner, self.batch = learner, batch
        self.updated = False

    def sample(self):
        return self.batch, None, self.learner.rssm.initial(2)

    def update(self, index, stoch, deter):
        self.updated = True


class ImitationLearnerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(3)

    def test_imitation_changes_actor_world_and_keeps_critic_untouched(self):
        learner, batch = agent(), data()
        actor = {name: p.detach().clone() for name, p in learner.actor.named_parameters()}
        world = {name: p.detach().clone() for name, p in learner.rssm.named_parameters()}
        critic = {name: p.detach().clone() for name, p in learner.value.named_parameters()}
        before = float(learner.validate_imitation(batch)["imitation/bc"])
        for _ in range(25):
            metrics = learner.pretrain_imitation(batch)
        self.assertGreater(float(metrics["imitation/actor_grad_norm"]), 0)
        self.assertTrue(any(not torch.equal(actor[name], p) for name, p in learner.actor.named_parameters()))
        self.assertTrue(any(not torch.equal(world[name], p) for name, p in learner.rssm.named_parameters()))
        self.assertTrue(all(torch.equal(critic[name], p) for name, p in learner.value.named_parameters()))
        self.assertLess(float(learner.validate_imitation(batch)["imitation/bc"]), before)
        self.assertTrue(all(torch.isfinite(p).all() for p in learner.parameters()))

    def test_current_expert_label_never_enters_encoder_or_inference(self):
        learner, batch = agent(), data()
        changed = batch.clone()
        changed["expert_action"] = -changed["expert_action"]
        torch.testing.assert_close(learner.encoder(batch), learner.encoder(changed))
        obs = TensorDict({"proprio": batch["proprio"][:, 0], "is_first": batch["is_first"][:, 0]}, batch_size=(2,))
        command, _ = learner.act(obs, learner.get_initial_state(2), eval=True)
        self.assertEqual(command.shape, (2, 19))
        self.assertTrue(torch.isfinite(command).all())

    def test_burnin_and_padding_labels_are_not_supervised(self):
        learner, batch = agent(), data()
        changed = batch.clone()
        changed["expert_action"][:, :2] = -.9
        first = learner.validate_imitation(batch)["imitation/bc"]
        second = learner.validate_imitation(changed)["imitation/bc"]
        torch.testing.assert_close(first, second)
        batch["expert_mask"].zero_()
        with self.assertRaisesRegex(ValueError, "no valid post-burn-in labels"):
            learner.pretrain_imitation(batch)

    def test_previous_action_is_rssm_input_and_current_action_is_bc_target(self):
        learner, batch = agent(), data()
        captured = []
        observe = learner.rssm.observe
        def record(embed, action, initial, reset):
            captured.append(action.detach().clone())
            return observe(embed, action, initial, reset)
        learner.rssm.observe = record
        batch["action"][:, 1:] = .1
        imitation_objective(learner, batch, learner.rssm.initial(2), include_world=False)
        torch.testing.assert_close(captured[0], batch["action"])
        self.assertFalse(torch.equal(captured[0], batch["expert_action"]))

    def test_validation_uses_deployment_modes_without_sampling(self):
        learner, batch = agent(), data()
        step, inputs = learner.rssm.obs_step, []
        def record(stoch, deter, action, embed, reset, sample=True):
            inputs.append((action.detach().clone(), sample))
            return step(stoch, deter, action, embed, reset, sample=sample)
        learner.rssm.obs_step = record
        first = learner.validate_imitation(batch)["imitation/bc"]
        torch.manual_seed(999)
        second = learner.validate_imitation(batch)["imitation/bc"]
        torch.testing.assert_close(first, second)
        self.assertTrue(all(sample is False for _, sample in inputs))
        torch.testing.assert_close(torch.stack([value for value, _ in inputs[:8]], dim=1), batch["action"])

    def test_real_rl_update_also_anchors_actor_and_updates_replay(self):
        learner, batch = agent(.1), data()
        critic = {name: p.detach().clone() for name, p in learner.value.named_parameters()}
        replay = Replay(learner, batch)
        metrics = learner.update(replay, expert_batch=batch)
        self.assertTrue(replay.updated)
        self.assertIn("loss/policy", metrics)
        self.assertIn("imitation/bc", metrics)
        self.assertTrue(torch.isfinite(metrics["opt/loss"]))
        self.assertTrue(any(not torch.equal(critic[name], p) for name, p in learner.value.named_parameters()))
        with self.assertRaisesRegex(ValueError, "requires an expert batch"):
            learner.update(replay)

    def test_standard_checkpoint_preserves_full_command_imitation_metadata(self):
        learner, batch = agent(), data()
        learner.pretrain_imitation(batch)
        contract = {"version": 1, "controller": {"full_command": True}}
        metadata = {"version": 1, "dataset_fingerprint": "test-digest", "pretraining_updates": 1,
                    "mode": "full_command_bc"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            save_imitation_checkpoint(learner, path, contract, metadata, 1)
            saved = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(saved["imitation"], metadata)
            self.assertEqual(saved["training_contract"], contract)
            self.assertEqual(saved["step"], 0)
            clone = agent()
            clone.load_state_dict(saved["agent_state_dict"])
            torch.testing.assert_close(clone.validate_imitation(batch)["imitation/bc"],
                                       learner.validate_imitation(batch)["imitation/bc"])

    def test_checkpoint_only_policy_loads_without_reference_or_dataset_files(self):
        from script_free_policy import ScriptFreePolicy
        config = tiny_config()
        exported = OmegaConf.to_container(config.model, resolve=True)
        obs = spaces.Dict({"proprio": spaces.Box(-np.inf, np.inf, (45,), np.float32)})
        actions = spaces.Box(-1., 1., (19,), np.float32)
        learner = Dreamer(config.model, obs, actions).to("cpu")
        learner.export_model_config = exported
        learner.pretrain_imitation(data())
        contract = full_command_contract()
        metadata = {"version": 1, "mode": "full_command_bc", "pretraining_updates": 1,
                    "dataset_fingerprint": {"data_sha256": "unavailable", "manifest_sha256": "unavailable"}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "student.pt"
            save_imitation_checkpoint(learner, path, contract, metadata, 1)
            with patch("numpy.load", side_effect=RuntimeError("Reference/dataset file access is forbidden")):
                policy = ScriptFreePolicy(path)
                first = policy.act(np.zeros((2, 45), np.float32))
                second = policy.act(np.zeros((2, 45), np.float32))
            self.assertEqual((first.shape, second.shape), ((2, 19), (2, 19)))
            self.assertTrue(torch.isfinite(second).all())
            with self.assertRaisesRegex(ValueError, "world count"):
                policy.act(np.zeros((1, 45), np.float32))
            with self.assertRaisesRegex(ValueError, "finite"):
                policy.act(np.full((2, 45), np.nan, np.float32))

    def test_pretraining_cli_runs_bounded_real_updates_and_records_validation(self):
        import json
        from imitation_data import save_dataset
        from scripts import pretrain_script_free
        config = tiny_config(.1)
        OmegaConf.set_struct(config, False)
        config.env.mjx_full_command = True
        config.model.actor.dist.cont.min_std = .0003
        config.model.actor.dist.cont.max_std = .002
        config.imitation = dict(seed=0, pretrain_updates=2, validation_every=1,
                               batch_size=2, batch_length=4, burn_in=2)
        physical_contract = full_command_contract()
        obs = spaces.Dict({"proprio": spaces.Box(-np.inf, np.inf, (45,), np.float32)})
        actions = spaces.Box(-1., 1., (19,), np.float32)
        class FakeEnv:
            observation_space, action_space = obs, actions
            def __init__(self, config):
                pass
            def close(self):
                pass
        arrays = dict(proprio=np.zeros((3, 9, 45), np.float32), action=np.full((3, 8, 19), .3, np.float32),
            reward=np.zeros((3, 8), np.float32), is_terminal=np.zeros((3, 8), bool),
            valid=np.ones((3, 8), bool), split=np.asarray([0, 1, 2], np.int8))
        with tempfile.TemporaryDirectory() as directory:
            path, output = Path(directory) / "expert.npz", Path(directory) / "pretrain"
            save_dataset(path, arrays, {"training_contract": physical_contract})
            with patch.object(sys, "argv", ["pretrain", "--dataset", str(path), "--output", str(output), "--updates", "2", "--seed", "7"]), \
                 patch("hydra.compose", return_value=config), \
                 patch("script_free_contract.validate_student_config", return_value={"scope": "tiny CPU batch only"}), \
                 patch("envs.mjx_teacher_env.MJXTeacherEnv", FakeEnv), \
                 patch("training_contract.build_training_contract", return_value=physical_contract):
                pretrain_script_free.main()
            checkpoint = torch.load(output / "latest.pt", weights_only=True)
            report = json.loads((output / "pretraining_report.json").read_text())
            self.assertEqual(checkpoint["optimizer_updates"], 2)
            self.assertEqual(checkpoint["imitation"]["pretraining_updates"], 2)
            self.assertEqual(checkpoint["imitation"]["seed"], 7)
            resolved = OmegaConf.load(output / "config.yaml")
            self.assertEqual((resolved.seed, resolved.imitation.seed), (7, 7))
            self.assertIn("cont", checkpoint["model_config"]["actor"]["dist"])
            reconstructed = Dreamer(OmegaConf.create(checkpoint["model_config"]), obs, actions).to("cpu")
            reconstructed.load_state_dict(checkpoint["agent_state_dict"])
            inference = TensorDict({"proprio": torch.zeros(1, 45),
                                    "is_first": torch.ones(1, 1, dtype=torch.bool)}, batch_size=(1,))
            command, _ = reconstructed.act(inference, reconstructed.get_initial_state(1), eval=True)
            self.assertEqual(command.shape, (1, 19))
            self.assertTrue(torch.isfinite(command).all())
            self.assertGreater(report["changed_tensors_first_update"], 0)
            self.assertFalse(report["walking_qualified"])
            self.assertEqual(len((output / "metrics.jsonl").read_text().splitlines()), 3)


if __name__ == "__main__":
    unittest.main()
