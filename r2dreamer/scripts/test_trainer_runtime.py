"""CPU regressions for trainer boundaries and real TorchRL replay semantics.

These test control/replay contracts; they do not qualify CUDA or robot physics.
Run: python -m unittest discover -s scripts -p test_trainer_runtime.py -v
"""
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from buffer import Buffer
from trainer import OnlineTrainer
from train import _graceful_stop_signals


class Logger:
    def __init__(self):
        self.values = {}
        self.records = []

    def scalar(self, name, value):
        self.values[name] = float(value)

    def write(self, step, **kwargs):
        self.records.append({"step": step, **self.values})
        self.values = {}


class Envs:
    env_num = 2

    def __init__(self, fall_after=3):
        self.calls = 0
        self.fall_after = fall_after
        self.evaluations = 0
        self.on_step = None

    def set_training_updates(self, updates):
        self.updates = updates

    def begin_evaluation(self):
        self.calls = 0
        self.evaluations += 1

    def step(self, action, done):
        self.calls += 1
        if self.on_step:
            self.on_step()
        trans = TensorDict({"reward": (~done).float()[:, None],
                            "is_first": done[:, None].clone()}, batch_size=[2])
        ended = torch.full((2,), self.calls % self.fall_after == 0)
        return trans, ended


class Replay:
    max_size = 1000

    def __init__(self):
        self.size = 0

    def count(self):
        return self.size

    def add_transition(self, data):
        self.size += 2

    def clear(self):
        self.size = 0


class Agent(torch.nn.Module):
    device = torch.device("cpu")

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self._optimizer = torch.optim.Adam(self.parameters(), lr=.01)
        self.on_update = None
        self.completed_updates = 0

    def get_initial_state(self, count):
        return {key: torch.zeros(count, 1) for key in ("prev_action", "stoch", "deter")}

    def act(self, trans, state, eval=False):
        return torch.zeros(2, 1), state

    def update(self, replay):
        self.weight.sum().backward()
        if self.on_update:
            self.on_update()  # Request interruption before the optimizer finishes.
        self._optimizer.step()
        self._optimizer.zero_grad(set_to_none=True)
        self.completed_updates += 1
        return {"loss": self.weight.detach().sum()}


def config(**overrides):
    values = dict(steps=2000, pretrain=5, eval_every=0, eval_episode_num=0,
                  video_pred_log=False, params_hist_log=False, batch_length=2,
                  batch_size=1, train_ratio=12, update_log_every=10,
                  action_repeat=2, checkpoint_interval=0)
    values.update(overrides)
    return SimpleNamespace(**values)


class TrainerRuntimeTests(unittest.TestCase):
    def make_trainer(self, directory, **overrides):
        envs, logger = Envs(), Logger()
        trainer = OnlineTrainer(config(**overrides), Replay(), logger,
                                Path(directory), envs, None)
        return trainer, envs, logger, Agent()

    def test_signal_request_waits_for_completed_update_and_saves_it(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer, envs, logger, agent = self.make_trainer(directory, require_learning=True)
            trainer.training_contract = {"schema_version": 1, "identity": "unit-test-contract"}
            agent.on_update = lambda: trainer.request_stop("SIGTERM")
            trainer.begin(agent, step_offset=123, update_offset=41)
            saved = torch.load(Path(directory) / "latest.pt", weights_only=False)
            self.assertEqual(agent.completed_updates, 1)
            self.assertEqual(saved["optimizer_updates"], 42)
            self.assertEqual(saved["step"], trainer.last_train_step)
            self.assertEqual(saved["training_contract"], trainer.training_contract)
            self.assertEqual(saved["stop_reason"], "SIGTERM")
            self.assertGreater(saved["step"], 123)
            self.assertTrue(torch.equal(saved["agent_state_dict"]["weight"], agent.weight))
            self.assertEqual(int(saved["optims_state_dict"]["_optimizer"]["state"][0]["step"]), 1)
            self.assertFalse((Path(directory) / "latest.pt.tmp").exists())

    def test_update_cap_is_relative_to_resume_and_evaluates_final_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer, envs, logger, agent = self.make_trainer(directory, stop_after_optimizer_updates=3)
            trainer.eval_episode_num = 1
            trainer.eval_envs = Envs()
            evaluated = []
            trainer._evaluate_and_checkpoint = lambda a, step: evaluated.append((step, trainer.update_count))
            trainer.begin(agent, step_offset=1000, update_offset=77)
            self.assertEqual(agent.completed_updates, 3)
            self.assertEqual(trainer.update_count, 80)
            self.assertEqual(evaluated, [(trainer.last_train_step, 80)])
            self.assertEqual(trainer._stop_reason, "optimizer_update_limit")
            final = logger.records[-1]
            transitions = (trainer.last_train_step - 1000) / 2
            self.assertAlmostEqual(final["perf/environment_transitions_per_second"] * final["perf/wall_seconds"], transitions)
            self.assertEqual(final["perf/phase_timing_synchronized"], 0.)

    def test_signal_on_last_capped_update_preserves_reason_and_skips_final_eval(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer, envs, logger, agent = self.make_trainer(directory, stop_after_optimizer_updates=1)
            trainer.eval_episode_num = 1
            trainer.eval_envs = Envs()
            evaluated = []
            trainer._evaluate_and_checkpoint = lambda a, step: evaluated.append(step)
            agent.on_update = lambda: trainer.request_stop("SIGTERM")
            trainer.begin(agent, step_offset=1000, update_offset=77)
            saved = torch.load(Path(directory) / "latest.pt", weights_only=False)
            self.assertEqual(agent.completed_updates, 1)
            self.assertEqual(saved["optimizer_updates"], 78)
            self.assertEqual(saved["stop_reason"], "SIGTERM")
            self.assertEqual(trainer._stop_reason, "SIGTERM")
            self.assertEqual(evaluated, [])

    def test_stop_during_collection_preserves_resumed_offset_without_learning_error(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer, envs, logger, agent = self.make_trainer(directory, require_learning=True)
            envs.on_step = lambda: trainer.request_stop("SIGINT")
            trainer.begin(agent, step_offset=1000, update_offset=77)
            saved = torch.load(Path(directory) / "latest.pt", weights_only=False)
            self.assertEqual(saved["step"], 1000)  # Initial reset is not a physical transition.
            self.assertEqual(saved["optimizer_updates"], 77)
            self.assertEqual(agent.completed_updates, 0)

    def test_signal_handlers_only_request_and_restore_previous_handlers(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer, envs, logger, agent = self.make_trainer(directory)
            before = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
            with _graceful_stop_signals(trainer):
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
                self.assertTrue(trainer._stop_requested)
                self.assertEqual(trainer._stop_reason, "SIGTERM")
                self.assertFalse((Path(directory) / "latest.pt").exists())
            self.assertEqual(before, {sig: signal.getsignal(sig) for sig in before})

    def test_failed_save_does_not_replace_loadable_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer, envs, logger, agent = self.make_trainer(directory)
            trainer.save_checkpoint(agent, "latest.pt", step=10)
            with patch("trainer.torch.save", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    trainer.save_checkpoint(agent, "latest.pt", step=20)
            saved = torch.load(Path(directory) / "latest.pt", weights_only=False)
            self.assertEqual(saved["step"], 10)

    def test_episode_accounting_preserves_completed_scores_and_lengths(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer, envs, logger, agent = self.make_trainer(directory, steps=20)
            trainer.begin(agent)
            episodes = [r for r in logger.records if "episode/score" in r]
            self.assertEqual([(r["episode/score"], r["episode/length"]) for r in episodes], [(2., 2.)] * 4)
            self.assertEqual([r["step"] for r in episodes], [8, 9, 16, 17])

    def test_eval_lifecycle_hook_and_no_cache_when_prediction_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer, envs, logger, agent = self.make_trainer(directory)
            trainer.eval_envs = envs
            # Only cache uses TensorDict.clone in this eval path.
            with patch.object(TensorDict, "clone", side_effect=AssertionError("unused cache")):
                summary = trainer.eval(agent, 0)
            self.assertEqual(summary, {"score": 2., "length": 2.})
            self.assertEqual(envs.evaluations, 1)


class ReplayRuntimeTests(unittest.TestCase):
    def make_buffer(self, **overrides):
        values = dict(device="cpu", storage_device="cpu", batch_size=2,
                      batch_length=2, max_size=16)
        values.update(overrides)
        return Buffer(SimpleNamespace(**values))

    def transition(self, tick):
        return TensorDict({"episode": torch.arange(2),
            "observation": torch.full((2, 1), float(tick)),
            "action": torch.full((2, 1), float(tick + 10)),
            "stoch": torch.full((2, 1, 2), float(tick)),
            "deter": torch.full((2, 2), float(tick)),
            "log_reward_balance": torch.ones(2, 1)}, batch_size=[2], device="cpu")

    def assert_same_sample(self, cached, uncached):
        torch.manual_seed(345)
        actual = cached.sample()
        torch.manual_seed(345)
        expected = uncached.sample()
        for key in actual[0].keys():
            self.assertTrue(torch.equal(actual[0][key], expected[0][key]), key)
        for left, right in zip(actual[1] + list(actual[2]), expected[1] + list(expected[2])):
            self.assertTrue(torch.equal(left, right))

    def test_cached_sampling_matches_uncached_across_append_wrap_and_refill(self):
        cached = self.make_buffer(cache_trajectories=True)
        uncached = self.make_buffer(cache_trajectories=False)
        for tick in range(4):
            for buffer in (cached, uncached):
                buffer.add_transition(self.transition(tick))
        sampler = cached._buffer._sampler
        with patch.object(sampler, "_find_start_stop_traj", wraps=sampler._find_start_stop_traj) as scan:
            self.assert_same_sample(cached, uncached)
            self.assert_same_sample(cached, uncached)
            self.assertEqual(scan.call_count, 1)
            for buffer in (cached, uncached):
                data, index, _ = buffer.sample()
                buffer.update(index, torch.zeros_like(data["stoch"]), torch.zeros_like(data["deter"]))
            # Latent values differ by sampled index but boundaries are reusable.
            cached.sample()
            self.assertEqual(scan.call_count, 1)
            for tick in range(4, 14):
                for buffer in (cached, uncached):
                    buffer.add_transition(self.transition(tick))
            self.assert_same_sample(cached, uncached)
            self.assertEqual(scan.call_count, 2)
            for buffer in (cached, uncached):
                buffer.clear()
                for tick in range(101, 106):
                    buffer.add_transition(self.transition(tick))
            self.assert_same_sample(cached, uncached)
            self.assertEqual(scan.call_count, 3)

    def test_optional_metric_pruning_keeps_source_transition_unchanged(self):
        buffer = self.make_buffer(store_log_metrics=False)
        source = self.transition(0)
        for tick in range(4):
            buffer.add_transition(source)
        sampled, _, _ = buffer.sample()
        self.assertIn("log_reward_balance", source)
        self.assertNotIn("log_reward_balance", sampled)
        self.assertIn("observation", sampled)


if __name__ == "__main__":
    unittest.main()
