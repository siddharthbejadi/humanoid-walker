"""CPU checks of real online admission and RL/BC hooks, not simulator qualification."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import hydra
import numpy as np
from gymnasium import spaces
from omegaconf import OmegaConf
import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from buffer import Buffer
from dreamer import Dreamer
from imitation_data import ImitationDataset, save_dataset
from imitation_training import save_imitation_checkpoint
from trainer import OnlineTrainer
import train
from scripts.test_script_free_contract import full_command_contract


def config():
    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        result = hydra.compose("walking_mjx_script_free", overrides=[
            "device=cpu", "buffer.device=cpu", "buffer.storage_device=cpu",
            "model.units=16", "model.deter=32", "model.hidden=16", "model.discrete=4",
            "model.rssm.stoch=2", "model.rssm.blocks=4", "model.encoder.mlp.layers=1",
            "model.actor.layers=1", "model.imag_horizon=2", "model.lr=0.003", "model.warmup=0",
            "model.kl_free=0.0", "model.diagnostic_every=0", "imitation.pretrain_updates=1"])
    return result


OBSERVATIONS = spaces.Dict({"proprio": spaces.Box(-np.inf, np.inf, (45,), np.float32)})
ACTIONS = spaces.Box(-1., 1., (19,), np.float32)


class Logger:
    def __init__(self, *unused):
        self.records, self.values = [], {}

    def log_hydra_config(self, unused):
        pass

    def scalar(self, name, value):
        self.values[name] = float(value)

    def write(self, step, **unused):
        self.records.append(dict(step=step, **self.values))
        self.values.clear()


class SensorEnvs:
    env_num = 2

    def __init__(self):
        self.ticks = 0
        self.last_updates = 0

    def set_training_updates(self, updates):
        self.last_updates = updates

    def step(self, action, done):
        if action.shape != (self.env_num, 19):
            raise AssertionError("Full motor commands must be forwarded to each world")
        self.ticks += 1
        ended = torch.zeros(self.env_num, dtype=torch.bool)
        observation = TensorDict({
            "proprio": torch.full((self.env_num, 45), self.ticks * .0001),
            "reward": (~done).float()[:, None] * .1,
            "is_first": done[:, None].clone(),
            "is_terminal": ended[:, None], "is_last": ended[:, None]}, batch_size=(self.env_num,))
        return observation, ended


class OnlineHookTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.contract = full_command_contract()
        arrays = dict(proprio=np.zeros((3, 145, 45), np.float32),
            action=np.full((3, 144, 19), .3, np.float32), reward=np.zeros((3, 144), np.float32),
            is_terminal=np.zeros((3, 144), bool), valid=np.ones((3, 144), bool),
            split=np.asarray([0, 1, 2], np.int8))
        self.dataset_path = self.directory / "expert.npz"
        save_dataset(self.dataset_path, arrays, {"training_contract": self.contract})
        self.dataset = ImitationDataset(self.dataset_path)
        model_config = config().model
        export = OmegaConf.to_container(model_config, resolve=True)
        with redirect_stdout(io.StringIO()):
            self.expert = Dreamer(model_config, OBSERVATIONS, ACTIONS).to("cpu")
        self.expert.export_model_config = export
        self.expert.pretrain_imitation(self.dataset.sample(2, 6, burnin=2, rng=3))
        self.metadata = dict(version=1, mode="full_command_bc", pretraining_updates=1,
                             dataset_fingerprint=self.dataset.fingerprint)
        self.checkpoint_path = self.directory / "pretrained.pt"
        save_imitation_checkpoint(self.expert, self.checkpoint_path, self.contract, self.metadata, 1)

    def tearDown(self):
        self.temporary.cleanup()

    def launch(self, settings, dataset=True):
        settings.logdir = str(self.directory / "online")
        if dataset:
            settings.imitation.dataset = str(self.dataset_path)
        captured = {}
        def factory(model, obs, actions):
            learner = Dreamer(model, obs, actions)
            learner.load_state_dict = Mock(wraps=learner.load_state_dict)
            captured["agent"] = learner
            return learner
        def begin(trainer, learner, step_offset=0, update_offset=0):
            captured.update(trainer=trainer, offsets=(step_offset, update_offset))
            trainer.last_train_step = step_offset
            trainer.update_count = update_offset
        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(patch("train.tools.setup_console_log", return_value=io.StringIO()))
            stack.enter_context(patch("train.tools.Logger", Logger))
            stack.enter_context(patch("train.make_envs", return_value=(SensorEnvs(), None, OBSERVATIONS, ACTIONS)))
            stack.enter_context(patch("train.build_training_contract", return_value=self.contract))
            stack.enter_context(patch("train.Dreamer", side_effect=factory))
            stack.enter_context(patch.object(OnlineTrainer, "begin", begin))
            train.main.__wrapped__(settings)
        captured["checkpoint"] = torch.load(self.directory / "online" / "latest.pt", weights_only=True)
        return captured

    def test_dataset_and_verified_initialization_are_required(self):
        for missing in ("dataset", "checkpoint"):
            with self.subTest(missing=missing):
                settings = config()
                if missing == "dataset":
                    with patch("imitation_data.ImitationDataset", side_effect=AssertionError("Missing dataset was loaded")):
                        with self.assertRaisesRegex(ValueError, "dataset"):
                            self.launch(settings, dataset=False)
                else:
                    with self.assertRaisesRegex(ValueError, "verified imitation checkpoint"):
                        self.launch(settings)

    def test_init_from_loads_weights_but_keeps_optimizer_and_counters_fresh(self):
        settings = config()
        settings.init_from = str(self.checkpoint_path)
        captured = self.launch(settings)
        self.assertEqual(captured["offsets"], (0, 0))
        self.assertEqual(len(captured["agent"]._optimizer.state), 0)
        self.assertEqual(captured["agent"]._scheduler.last_epoch, 0)
        captured["agent"].load_state_dict.assert_called_once()
        self.assertIsInstance(captured["trainer"].imitation_dataset, ImitationDataset)
        self.assertEqual(captured["trainer"].imitation_metadata, self.metadata)
        self.assertEqual(captured["checkpoint"]["imitation"], self.metadata)
        self.assertEqual(captured["checkpoint"]["training_contract"], self.contract)
        self.assertIn("cont", captured["checkpoint"]["model_config"]["actor"]["dist"])
        self.assertNotIn("cont", settings.model.actor.dist)
        reconstructed = Dreamer(OmegaConf.create(captured["checkpoint"]["model_config"]),
                                OBSERVATIONS, ACTIONS).to("cpu")
        reconstructed.load_state_dict(captured["checkpoint"]["agent_state_dict"])

    def test_resume_restores_offsets_optimizer_schedule_and_imitation_identity(self):
        checkpoint = torch.load(self.checkpoint_path, weights_only=True)
        checkpoint.update(step=250, optimizer_updates=7)
        torch.save(checkpoint, self.checkpoint_path)
        settings = config()
        settings.resume_from = str(self.checkpoint_path)
        captured = self.launch(settings)
        self.assertEqual(captured["offsets"], (250, 7))
        self.assertGreater(len(captured["agent"]._optimizer.state), 0)
        self.assertEqual(captured["agent"]._scheduler.last_epoch, 7)
        self.assertEqual(captured["checkpoint"]["optimizer_updates"], 7)
        self.assertEqual(captured["checkpoint"]["imitation"], self.metadata)

    def test_init_and_resume_reject_wrong_dataset_or_full_command_contract_before_loading(self):
        original = torch.load(self.checkpoint_path, weights_only=True)
        for mode in ("init_from", "resume_from"):
            for changed in ("fingerprint", "residual", "contract", "unverified", "update_count"):
                with self.subTest(mode=mode, changed=changed):
                    checkpoint = deepcopy(original)
                    if changed == "fingerprint":
                        checkpoint["imitation"]["dataset_fingerprint"]["manifest_sha256"] = "other-sidecar"
                    elif changed == "residual":
                        checkpoint["training_contract"]["control"]["action_mode"] = "residual"
                    elif changed == "contract":
                        checkpoint["training_contract"]["task"]["episode_horizon_control_steps"] += 1
                    elif changed == "unverified":
                        checkpoint.pop("imitation")
                    else:
                        checkpoint["imitation"]["pretraining_updates"] = 0
                    torch.save(checkpoint, self.checkpoint_path)
                    settings = config()
                    setattr(settings, mode, str(self.checkpoint_path))
                    with patch.object(Dreamer, "load_state_dict", side_effect=AssertionError("Unsafe weights loaded")):
                        with self.assertRaises(ValueError):
                            self.launch(settings)

    def test_real_trainer_samples_masks_and_forwards_expert_batch_into_real_rl_updates(self):
        settings = SimpleNamespace(steps=500, pretrain=1, eval_every=0, eval_episode_num=0,
            video_pred_log=False, params_hist_log=False, batch_length=8, batch_size=2,
            train_ratio=16, update_log_every=1, action_repeat=10, checkpoint_interval=0,
            stop_after_optimizer_updates=2, require_learning=True)
        replay = Buffer(SimpleNamespace(device="cpu", storage_device="cpu", batch_size=2,
                                        batch_length=8, max_size=64))
        directory = self.directory / "runtime"
        directory.mkdir()
        trainer = OnlineTrainer(settings, replay, Logger(), directory, SensorEnvs(), None)
        trainer.training_contract, trainer.imitation_metadata = self.contract, self.metadata
        trainer.imitation_dataset = self.dataset
        trainer.imitation_settings = SimpleNamespace(batch_size=2, batch_length=6, burn_in=2)
        trainer.imitation_rng = np.random.default_rng(9)
        before = {name: value.detach().clone() for name, value in self.expert.value.named_parameters()}
        updates = []
        update = self.expert.update
        def record(buffer, expert_batch=None):
            self.assertIsInstance(expert_batch, TensorDict)
            self.assertEqual(expert_batch.batch_size, torch.Size([2, 8]))
            self.assertEqual(expert_batch["proprio"].shape, (2, 8, 45))
            self.assertEqual(expert_batch["expert_action"].shape, (2, 8, 19))
            self.assertGreater(int(expert_batch["expert_mask"].sum()), 0)
            self.assertTrue(torch.equal(expert_batch["expert_mask"][1, :2], torch.zeros(2, 1)))
            metrics = update(buffer, expert_batch=expert_batch)
            updates.append(metrics)
            return metrics
        with patch.object(self.dataset, "sample", wraps=self.dataset.sample) as sampler, \
             patch.object(self.expert, "update", side_effect=record), redirect_stdout(io.StringIO()):
            trainer.begin(self.expert, update_offset=5)
        self.assertEqual(sampler.call_count, 2)
        for call in sampler.call_args_list:
            self.assertEqual(call.args, (2, 6))
            self.assertEqual(call.kwargs["burnin"], 2)
            self.assertEqual(call.kwargs["split"], "train")
            self.assertEqual(call.kwargs["device"], torch.device("cpu"))
            self.assertIs(call.kwargs["rng"], trainer.imitation_rng)
        self.assertTrue(all("loss/policy" in values and "imitation/bc" in values for values in updates))
        self.assertEqual(trainer.update_count, 7)
        self.assertTrue(any(not torch.equal(before[name], value) for name, value in self.expert.value.named_parameters()))
        replay_data, _, _ = replay.sample()
        self.assertEqual(replay_data["action"].shape, (2, 8, 19))
        self.assertEqual(replay_data["proprio"].shape, (2, 8, 45))
        checkpoint = torch.load(directory / "latest.pt", weights_only=True)
        self.assertEqual(checkpoint["imitation"], self.metadata)
        self.assertEqual(checkpoint["optimizer_updates"], 7)
        self.assertIn("cont", checkpoint["model_config"]["actor"]["dist"])
        self.assertFalse((directory / "latest.pt.tmp").exists())


if __name__ == "__main__":
    unittest.main()
