"""CPU regressions for transition causes, evaluation records and resume ranking.

Use normal imports when RL dependencies are installed, with an AST fallback for
minimal CPU environments. Simulator stepping is mocked; these do not qualify physics.
"""
import ast
from collections import defaultdict
import contextlib
import json
import pickle
from pathlib import Path
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace as NS
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from checkpoint_selection import checkpoint_rank, walking_trial_success, finish_trial_success
from training_budget import validate_learning_budget


class TensorDict(dict):
    def __init__(self, values, **kwargs):
        super().__init__(values)

    def to(self, *args, **kwargs):
        return self


def load_class(path, name):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == name)
    namespace = dict(torch=torch, np=np, Path=Path, TensorDict=TensorDict,
        contextlib=contextlib, json=json, pickle=pickle, shutil=shutil, time=time,
        TEACHER_STATE_DIM=98, validate_learning_budget=validate_learning_budget,
        checkpoint_rank=checkpoint_rank, walking_trial_success=walking_trial_success,
        finish_trial_success=finish_trial_success,
        tools=NS(Every=lambda interval: None, Once=lambda: None,
                 recursively_collect_optim_state_dict=lambda agent: {}))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / path), "exec"), namespace)
    return namespace[name]


try:
    from envs.mjx_teacher_env import MJXTeacherEnv as Bridge
    from trainer import OnlineTrainer as Trainer
except ModuleNotFoundError as error:
    if error.name not in ("gymnasium", "tensordict", "torchrl", "hydra"):
        raise
    Bridge = load_class("envs/mjx_teacher_env.py", "MJXTeacherEnv")
    Trainer = load_class("trainer.py", "OnlineTrainer")


def state(count, *, dots=0, terminal=None, elapsed=1, causes=None):
    metrics = defaultdict(lambda: torch.zeros(count))
    metrics["dots_reached"] = torch.full((count,), float(dots))
    metrics.update(causes or {})
    info = defaultdict(lambda: torch.zeros(count))
    info.update(valid_steps=torch.zeros(count, 2), scrub=torch.zeros(count),
                step_count=torch.full((count,), elapsed), standing_stable_seconds=torch.zeros(count))
    return NS(metrics=metrics, info=info, done=torch.zeros(count) if terminal is None else terminal,
              reward=torch.zeros(count), obs=torch.zeros(count, 143))


def bridge(old, current, count):
    env = Bridge.__new__(Bridge)
    env.env_num, env.device, env.horizon = count, torch.device("cpu"), 100
    env.full_command, env._packed_obs_dim = False, 45 + 98
    env.raw = NS(control_dt=.02, reward_profile="goal_corridor_v3")
    env.state, env.rng, env._pending_injection_first = old, 0, False
    env.jax = NS(random=NS(split=lambda key, count: np.arange(count)),
                 dlpack=NS(from_dlpack=lambda value: value))
    env._reset_batch = lambda keys: current
    env._step_batch = lambda old, action: current
    env._select_reset_batch = lambda reset, stepped, mask: current
    return env


class Logger:
    def __init__(self):
        self.values = {}

    def scalar(self, name, value):
        self.values[name] = float(value)

    def write(self, step, **kwargs):
        pass


class Agent(torch.nn.Module):
    device = torch.device("cpu")

    def __init__(self, count):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.count = count

    def get_initial_state(self, count):
        return {"prev_action": torch.zeros(count, 19)}

    def act(self, trans, state, eval=False):
        return torch.zeros(self.count, 19), state


def trainer(path, envs=None, **overrides):
    config = dict(steps=1000, pretrain=0, eval_every=0, eval_episode_num=0,
        video_pred_log=False, params_hist_log=False, batch_length=2, batch_size=1,
        train_ratio=1, update_log_every=10, action_repeat=10, checkpoint_interval=0,
        best_metric_key="total_valid_steps", best_safety_metric_key="safety_failure_count",
        best_tiebreak_metric_key="alternating_steps", checkpoint_order="safety_first",
        save_best_safe=True)
    config.update(overrides)
    return Trainer(NS(**config), None, Logger(), Path(path), None, envs)


class EvaluationEnvs:
    env_num = 4

    def begin_evaluation(self):
        self.calls = 0

    def evaluation_protocol(self):
        return {"version": 2, "seed": 10000, "repeatable": True, "worlds": 4}

    def reset_pose_ids(self):
        return [0] * 4

    def step(self, action, done):
        tick = self.calls
        self.calls += 1
        names = ("physical_fall", "wall_hit", "stall_out", "goal_reached")
        ended = torch.tensor([tick >= end for end in (1, 2, 3, 4)])
        logs = {name: torch.tensor([float(tick == i + 1) if i == j else 0.
                                   for j in range(4)]) for i, name in enumerate(names)}
        logs["safety_failure_count"] = logs["physical_fall"] + logs["wall_hit"]
        logs["fall_count"] = logs["physical_fall"]
        logs["terminated"] = ended.float()
        logs["time_limit"] = torch.zeros(4)
        for name in ("valid_left_step", "valid_right_step", "alternating_steps",
                     "dots_reached", "dot_event_delta"):
            logs[name] = torch.full((4,), float(tick > 0))
        logs["total_valid_steps"] = logs["valid_left_step"] + logs["valid_right_step"]
        logs["forward_distance"] = torch.full((4,), .01 * float(tick > 0))
        logs["alive_seconds"] = torch.full((4,), .02 * float(tick > 0))
        result = TensorDict({"reward": torch.zeros(4, 1)})
        result.update({"log_" + key: value[:, None] for key, value in logs.items()})
        return result, ended


class BridgeAccountingTests(unittest.TestCase):
    def test_all_terminal_causes_are_separate_and_time_limit_is_nonterminal(self):
        count = 5
        causes = {name: torch.nn.functional.one_hot(torch.tensor(i), count).float()
                  for i, name in enumerate(("physical_fall", "wall_hit", "stall_out", "goal_reached"))}
        current = state(count, terminal=torch.tensor([1., 1., 1., 1., 0.]), elapsed=100, causes=causes)
        env = bridge(state(count), current, count)
        trans, last = env.step(torch.zeros(count, 19), torch.zeros(count, dtype=torch.bool))
        self.assertTrue(last.all())
        self.assertEqual(trans["is_terminal"][:, 0].tolist(), [True, True, True, True, False])
        self.assertEqual(trans["log_fall_count"][:, 0].tolist(), [1., 0., 0., 0., 0.])
        self.assertEqual(trans["log_safety_failure_count"][:, 0].tolist(), [1., 1., 0., 0., 0.])
        self.assertEqual(trans["log_goal_reached"][:, 0].tolist(), [0., 0., 0., 1., 0.])
        self.assertEqual(trans["log_time_limit"][:, 0].tolist(), [0., 0., 0., 0., 1.])

    def test_legacy_raw_done_falls_back_and_reset_masks_flags(self):
        current = state(2, terminal=torch.ones(2))
        env = bridge(state(2), current, 2)
        trans, _ = env.step(torch.zeros(2, 19), torch.tensor([False, True]))
        self.assertEqual(trans["log_fall_count"][:, 0].tolist(), [1., 0.])
        self.assertEqual(trans["is_terminal"][:, 0].tolist(), [True, False])

    def test_cumulative_dots_emit_only_new_events_and_keep_final_count(self):
        env = bridge(state(2, dots=2), state(2, dots=3), 2)
        trans, _ = env.step(torch.zeros(2, 19), torch.zeros(2, dtype=torch.bool))
        self.assertEqual(trans["log_dots_reached"][:, 0].tolist(), [1., 1.])
        self.assertEqual(trans["log_dots_final_count_peak"][:, 0].tolist(), [3., 3.])
        env = bridge(state(2, dots=3), state(2, dots=0), 2)
        trans, _ = env.step(torch.zeros(2, 19), torch.ones(2, dtype=torch.bool))
        self.assertEqual(trans["log_dots_reached"][:, 0].tolist(), [0., 0.])

    def test_goal_alternation_uses_ordered_counter_not_bilateral_minimum(self):
        old, current = state(2), state(2)
        old.info["valid_steps"] = torch.tensor([[3., 0.], [3., 3.]])
        current.info["valid_steps"] = torch.tensor([[3., 3.], [0., 0.]])
        old.info["alternating_steps"] = torch.tensor([0., 1.])
        current.info["alternating_steps"] = torch.tensor([1., 0.])
        env = bridge(old, current, 2)
        trans, _ = env.step(torch.zeros(2, 19), torch.tensor([False, True]))
        self.assertEqual(trans["log_alternating_steps"][:, 0].tolist(), [1., 0.])


class EvaluationRecordTests(unittest.TestCase):
    def test_first_episode_masks_later_frames_and_persists_protocol_causes_and_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            run = trainer(directory, EvaluationEnvs())
            run.training_contract = {"version": 1, "task": {"valid_step_clearance_m": .015}}
            summary = run.eval(Agent(4), 123)
            record = json.loads((Path(directory) / "walking_evaluations.jsonl").read_text())
            trials = record["trials"]
            self.assertEqual([trial["termination_reasons"] for trial in trials],
                             [[name] for name in ("physical_fall", "wall_hit", "stall_out", "goal_reached")])
            self.assertEqual([trial["dots_reached"] for trial in trials], [1., 2., 3., 4.])
            self.assertEqual([trial["valid_left_step"] for trial in trials], [1., 2., 3., 4.])
            self.assertEqual(record["protocol"]["environment"]["seed"], 10000)
            self.assertEqual(record["training_contract"]["task"]["valid_step_clearance_m"], .015)
            self.assertEqual([trial["evaluation_world"] for trial in trials], [0, 1, 2, 3])
            self.assertEqual(summary["goal_success_rate"], .25)
            self.assertTrue(all(not trial["evaluation_truncated"] for trial in trials))

    def test_evaluation_cap_is_identified_as_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            run = trainer(directory, EvaluationEnvs(), eval_max_steps=1)
            run.eval(Agent(4), 1)
            trials = json.loads((Path(directory) / "walking_evaluations.jsonl").read_text())["trials"]
            self.assertEqual(trials[0]["termination_reasons"], ["physical_fall"])
            self.assertEqual(trials[3]["termination_reasons"], ["evaluation_limit"])
            self.assertTrue(trials[3]["evaluation_truncated"])
            self.assertEqual(trials[3]["goal_reached"], 0.)

    def test_wall_or_stall_is_not_a_successful_walking_trial(self):
        base = dict(fall_count=0, alive_seconds=10, total_valid_steps=10,
                    alternating_steps=5, valid_left_step=5, valid_right_step=5,
                    forward_distance=1., goal_reached=1)
        self.assertTrue(walking_trial_success(base))
        self.assertTrue(finish_trial_success(base))
        for extra in ({"safety_failure_count": 1}, {"wall_hit": 1}, {"stall_out": 1}):
            self.assertFalse(walking_trial_success({**base, **extra}))
            self.assertFalse(finish_trial_success({**base, **extra}))


class ResumeSelectionTests(unittest.TestCase):
    def evaluate(self, run, steps, valid):
        run.eval = lambda agent, step: dict(total_valid_steps=valid, safety_failure_count=0, alternating_steps=valid)
        run._evaluate_and_checkpoint(Agent(1), steps)

    def test_resume_restores_rank_and_does_not_replace_better_best(self):
        with tempfile.TemporaryDirectory() as directory:
            first = trainer(directory)
            self.evaluate(first, 10, 8)
            first.save_checkpoint(Agent(1), "latest.pt", step=10)
            before = (Path(directory) / "best.pt").read_bytes()
            checkpoint = torch.load(Path(directory) / "latest.pt", weights_only=True)
            json.dumps(checkpoint["checkpoint_selection"], allow_nan=False)
            second = trainer(directory)
            second.restore_checkpoint_selection(checkpoint)
            self.evaluate(second, 11, 1)
            self.assertEqual(before, (Path(directory) / "best.pt").read_bytes())
            self.evaluate(second, 12, 9)
            self.assertEqual(torch.load(Path(directory) / "best.pt", weights_only=True)["step"], 12)

    def test_newer_best_than_latest_is_merged_after_crash(self):
        with tempfile.TemporaryDirectory() as directory:
            first = trainer(directory)
            self.evaluate(first, 10, 5)
            first.save_checkpoint(Agent(1), "latest.pt", step=10)
            self.evaluate(first, 20, 10)
            before = (Path(directory) / "best.pt").read_bytes()
            second = trainer(directory)
            second.restore_checkpoint_selection(torch.load(Path(directory) / "latest.pt", weights_only=True))
            self.evaluate(second, 21, 7)
            self.assertEqual(before, (Path(directory) / "best.pt").read_bytes())

    def test_legacy_best_is_preserved_across_multiple_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            best = Path(directory) / "best.pt"
            best.write_bytes(b"unknown historical ranking")
            first = trainer(directory)
            first.restore_checkpoint_selection({})
            self.evaluate(first, 20, 10)
            first.save_checkpoint(Agent(1), "latest.pt", step=20)
            second = trainer(directory)
            second.restore_checkpoint_selection(torch.load(Path(directory) / "latest.pt", weights_only=True))
            self.evaluate(second, 21, 20)
            self.assertEqual(best.read_bytes(), b"unknown historical ranking")

    def test_protocol_change_rejects_overwriting_existing_best(self):
        with tempfile.TemporaryDirectory() as directory:
            first = trainer(directory)
            self.evaluate(first, 10, 5)
            checkpoint = {"checkpoint_selection": first._checkpoint_selection_state()}
            changed = trainer(directory, eval_max_steps=100)
            with self.assertRaisesRegex(ValueError, "protocol changed"):
                changed.restore_checkpoint_selection(checkpoint)

    def test_unreadable_best_is_preserved_when_latest_has_valid_ranks(self):
        with tempfile.TemporaryDirectory() as directory:
            first = trainer(directory)
            self.evaluate(first, 10, 5)
            checkpoint = {"checkpoint_selection": first._checkpoint_selection_state()}
            best = Path(directory) / "best.pt"
            best.write_bytes(b"unreadable saved best")
            second = trainer(directory)
            second.restore_checkpoint_selection(checkpoint)
            self.evaluate(second, 20, 10)
            self.assertEqual(best.read_bytes(), b"unreadable saved best")


if __name__ == "__main__":
    unittest.main()
