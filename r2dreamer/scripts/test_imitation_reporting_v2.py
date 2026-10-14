"""Bounded CPU checks of latest-checkpoint admission and held-out command evidence."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from gymnasium import spaces
from omegaconf import OmegaConf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from imitation_data import file_sha256, save_dataset
from imitation_training import full_episode_imitation_evidence
from scripts.test_imitation_learner import tiny_config
from scripts.test_script_free_contract import full_command_contract


class MemoryRSSM:
    def __init__(self):
        self.initial_batches, self.calls = [], []

    def initial(self, batch_size):
        self.initial_batches.append(batch_size)
        return torch.zeros(batch_size, 1), torch.zeros(batch_size, 1)

    def obs_step(self, stoch, deter, previous_action, embed, reset, sample=True):
        self.calls.append((previous_action.detach().clone(), reset.clone(), sample))
        deter = torch.where(reset, torch.zeros_like(deter), deter) + previous_action[:, :1] + embed
        return stoch, deter, torch.zeros_like(deter)

    def get_feat(self, stoch, deter):
        return deter


class MemoryAgent:
    device = torch.device("cpu")

    def __init__(self):
        self.rssm, self.training = MemoryRSSM(), True
        self.encoder_batches = []

    def encoder(self, data):
        self.encoder_batches.append(data.batch_size)
        return data["proprio"][..., :1]

    def actor(self, feature):
        return SimpleNamespace(mode=feature.repeat(1, 19))

    def preprocess(self, data):
        return data

    def train(self, mode=True):
        self.training = mode
        return self

    def eval(self):
        return self.train(False)


def episodes():
    arrays = dict(proprio=np.zeros((4, 5, 45), np.float32), action=np.full((4, 4, 19), .05, np.float32),
        reward=np.zeros((4, 4), np.float32), is_terminal=np.zeros((4, 4), bool),
        valid=np.ones((4, 4), bool), present=np.ones((4, 4), bool), split=np.asarray([1, 1, 1, 2], np.int8),
        episode_seed=np.arange(4), rollout_id=np.arange(10, 14))
    arrays["present"][0, 3:] = False
    arrays["valid"][0, 3:] = False
    arrays["proprio"][0, 4] = .99  # Padding must not enter recurrent history.
    arrays["valid"][2] = False
    return SimpleNamespace(arrays=arrays)


class ImitationReportingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(3)

    def test_full_episode_scores_true_history_weighted_labels_and_exclusions(self):
        learner = MemoryAgent()
        report = full_episode_imitation_evidence(learner, episodes(), "validation")
        self.assertEqual(learner.rssm.initial_batches, [1, 1])
        self.assertEqual(learner.encoder_batches, [torch.Size([1, 4]), torch.Size([1, 5])])
        self.assertEqual([bool(reset[0, 0]) for _, reset, _ in learner.rssm.calls],
                         [True, False, False, False, True, False, False, False, False])
        self.assertTrue(all(sample is False for _, _, sample in learner.rssm.calls))
        self.assertTrue(torch.equal(learner.rssm.calls[0][0], torch.zeros(1, 19)))
        torch.testing.assert_close(learner.rssm.calls[1][0], torch.full((1, 19), .05))
        self.assertEqual((report["labels"], report["episode_count"], report["scored_episode_count"]), (7, 3, 2))
        self.assertAlmostEqual(report["bc_mse"], .020 / 7, places=7)
        self.assertAlmostEqual(report["per_episode"][0]["bc_mse"], .005 / 3, places=7)
        self.assertAlmostEqual(report["per_episode"][1]["bc_mse"], .015 / 4, places=7)
        self.assertEqual(report["per_episode"][2]["exclusion_reason"], "no_safe_expert_labels")
        self.assertIsNone(report["per_episode"][2]["bc_mse"])
        self.assertEqual(report["per_episode"][1]["rollout_id"], 11)
        self.assertFalse(report["walking_qualified"])
        self.assertIn("not actual closed-loop", report["scope"])
        self.assertTrue(learner.training)

    def test_full_episode_is_deterministic_and_rejects_missing_heldout_labels(self):
        dataset = episodes()
        first = full_episode_imitation_evidence(MemoryAgent(), dataset, "test")
        torch.manual_seed(99)
        second = full_episode_imitation_evidence(MemoryAgent(), dataset, "test")
        self.assertEqual(first, second)
        dataset.arrays["valid"][3] = False
        learner = MemoryAgent()
        with self.assertRaisesRegex(ValueError, "No safe expert labels"):
            full_episode_imitation_evidence(learner, dataset, "test")
        self.assertTrue(learner.training)
        with self.assertRaisesRegex(ValueError, "held-out split"):
            full_episode_imitation_evidence(learner, dataset, "train")

    def test_early_best_then_final_regression_does_not_approve_latest_or_select_test(self):
        from dreamer import Dreamer
        from scripts import pretrain_script_free
        configuration = tiny_config(.1)
        OmegaConf.set_struct(configuration, False)
        configuration.env.mjx_full_command = True
        configuration.model.actor.dist.cont.min_std = .0003
        configuration.model.actor.dist.cont.max_std = .002
        configuration.imitation = dict(seed=0, pretrain_updates=2, validation_every=1,
                                       batch_size=2, batch_length=4, burn_in=2)
        contract = full_command_contract()
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
        sampled_scores = [{"imitation/bc": torch.tensor(score), "imitation/labels": torch.tensor(8.)}
                          for score in (.5, .1, .6)]
        evidence_calls = []
        def evidence(learner, dataset, split):
            evidence_calls.append((split, learner._scheduler.last_epoch))
            result = full_episode_imitation_evidence(learner, dataset, split)
            if split == "test":
                result["bc_mse"] = 3.9  # Reporting only: never select a checkpoint from test error.
            return result
        with tempfile.TemporaryDirectory() as directory:
            path, output = Path(directory) / "expert.npz", Path(directory) / "pretrain"
            save_dataset(path, arrays, {"training_contract": contract})
            with patch.object(sys, "argv", ["pretrain", "--dataset", str(path), "--output", str(output), "--updates", "2"]), \
                 patch("hydra.compose", return_value=configuration), \
                 patch("script_free_contract.validate_student_config", return_value={"scope": "tiny CPU learner fixture"}), \
                 patch("envs.mjx_teacher_env.MJXTeacherEnv", FakeEnv), \
                 patch("training_contract.build_training_contract", return_value=contract), \
                 patch.object(Dreamer, "validate_imitation", side_effect=sampled_scores), \
                 patch("imitation_training.full_episode_imitation_evidence", side_effect=evidence):
                pretrain_script_free.main()
            report = json.loads((output / "pretraining_report.json").read_text())
            latest = torch.load(output / "latest.pt", weights_only=True)
            best = torch.load(output / "best_bc.pt", weights_only=True)
            self.assertAlmostEqual(report["initial_validation_bc"], .5)
            self.assertAlmostEqual(report["best_validation_bc"], .1)
            self.assertAlmostEqual(report["final_validation_bc"], .6)
            self.assertFalse(report["final_validation_improved"])
            self.assertFalse(report["validation_improved"])
            self.assertEqual((latest["optimizer_updates"], best["optimizer_updates"]), (2, 1))
            self.assertEqual(report["latest_checkpoint_sha256"], file_sha256(output / "latest.pt"))
            self.assertEqual(evidence_calls, [("validation", 2), ("test", 2)])
            for name in ("full_episode_validation", "full_episode_test"):
                self.assertEqual(report[name]["optimizer_updates"], 2)
                self.assertEqual(report[name]["labels"], 8)
                self.assertEqual(latest["imitation"][name], report[name])
                self.assertNotIn(name, best["imitation"])
            self.assertEqual(report["full_episode_test"]["bc_mse"], 3.9)
            self.assertFalse(report["walking_qualified"])
            self.assertFalse(latest["imitation"]["validation_improved"])
            self.assertAlmostEqual(latest["imitation"]["last_validation"]["imitation/bc"], .6)


if __name__ == "__main__":
    unittest.main()
