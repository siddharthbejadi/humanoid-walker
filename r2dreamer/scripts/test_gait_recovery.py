"""CPU checks for gait stop boundaries and the bounded recovery command."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gait_guard import preservation_result
from scripts.run_replay_recovery import (recovery_command, validate_command, validate_guard_history,
                                         validate_selected_checkpoint)
from scripts.script_free_run_ops import sha256
from scripts.test_trainer_runtime import Agent, Envs, Logger, Replay, config
from trainer import OnlineTrainer


BASELINE = dict(walking_success_rate=23. / 32., valid_left_step=11.9375,
                valid_right_step=1.6875, alternating_steps=2.78125,
                total_valid_steps=13.625, safety_failure_count=0., score=1.)
PROTOCOL = dict(worlds=32, repeatable=True, seed=11100, horizon_control_steps=1000)


class EvaluationEnvs(Envs):
    def evaluation_protocol(self):
        return dict(PROTOCOL)


class PreservationTests(unittest.TestCase):
    def test_admits_asymmetric_early_gait_without_calling_it_qualified(self):
        baseline = preservation_result(None, BASELINE)
        self.assertTrue(baseline["passed"])
        self.assertTrue(preservation_result(baseline["baseline"], BASELINE)["passed"])

    def test_rejects_standing_even_if_reward_rises(self):
        standing = dict(BASELINE, walking_success_rate=0., valid_left_step=0.,
                        valid_right_step=0., alternating_steps=0., score=100.)
        result = preservation_result(BASELINE, standing)
        self.assertFalse(result["passed"])
        self.assertEqual(len(result["failures"]), 3)

    def test_improved_balanced_gait_is_not_rejected_for_fewer_dominant_side_steps(self):
        initial = dict(BASELINE, walking_success_rate=.625, valid_left_step=12.375,
                       valid_right_step=1.5, alternating_steps=2.375, safety_failure_count=.03125)
        improved = dict(BASELINE, walking_success_rate=.78125, valid_left_step=8.15625,
                        valid_right_step=1.90625, alternating_steps=2.84375, safety_failure_count=0.)
        result = preservation_result(initial, improved)
        self.assertTrue(result["passed"])
        self.assertEqual(result["criterion"], "bilateral_minimum_and_alternation_v2")
        self.assertEqual(result["current"]["bilateral_min_steps"], 1.90625)

    def test_zero_steps_on_either_side_still_fails_preservation(self):
        for key in ("valid_left_step", "valid_right_step"):
            self.assertFalse(preservation_result(BASELINE, dict(BASELINE, **{key: 0.}))["passed"])

    def test_failure_rate_allowance_is_inclusive(self):
        self.assertTrue(preservation_result(BASELINE, dict(BASELINE, safety_failure_count=2. / 32.))["passed"])
        self.assertFalse(preservation_result(BASELINE, dict(BASELINE, safety_failure_count=3. / 32.))["passed"])

    def test_rejects_weak_or_unilateral_initial_policy(self):
        self.assertFalse(preservation_result(None, dict(BASELINE, walking_success_rate=.49))["passed"])
        self.assertFalse(preservation_result(None, dict(BASELINE, valid_right_step=0.))["passed"])

    def test_nonfinite_metrics_are_not_admitted(self):
        with self.assertRaises(ValueError):
            preservation_result(BASELINE, dict(BASELINE, valid_right_step=float("nan")))


class TrainerGuardTests(unittest.TestCase):
    def trainer(self, directory, **overrides):
        settings = config(steps=2000, stop_after_optimizer_updates=4, eval_episode_num=32,
                          eval_max_steps=1000, gait_guard_enabled=True, gait_guard_eval_updates=2,
                          best_metric_key="total_valid_steps", best_safety_metric_key="safety_failure_count",
                          best_tiebreak_metric_key="alternating_steps", **overrides)
        training, evaluation, agent = Envs(), EvaluationEnvs(), Agent()
        trainer = OnlineTrainer(settings, Replay(), Logger(), Path(directory), training, evaluation)
        return trainer, training, agent

    def test_checks_initial_and_each_update_boundary_without_duplicate_final_eval(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer, envs, agent = self.trainer(directory)
            evaluated = []
            def evaluate(agent, step):
                evaluated.append(trainer.update_count)
                return dict(BASELINE)
            trainer.eval = evaluate
            trainer.begin(agent)
            self.assertEqual(evaluated, [0, 2, 4])
            self.assertEqual(trainer._stop_reason, "optimizer_update_limit")
            self.assertTrue((Path(directory) / "gait_baseline.pt").is_file())
            rows = [json.loads(line) for line in (Path(directory) / "gait_guard.jsonl").read_text().splitlines()]
            self.assertEqual([row["optimizer_updates"] for row in rows], [0, 2, 4])

    def test_failed_gate_stops_before_more_collection_or_updates_and_preserves_best(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer, envs, agent = self.trainer(directory)
            collection_calls_at_failure = []
            def evaluate(agent, step):
                if trainer.update_count:
                    collection_calls_at_failure.append(envs.calls)
                    return dict(BASELINE, walking_success_rate=0., valid_right_step=0.)
                return dict(BASELINE)
            trainer.eval = evaluate
            trainer.begin(agent)
            self.assertEqual(agent.completed_updates, 2)
            self.assertEqual(envs.calls, collection_calls_at_failure[-1])
            self.assertEqual(trainer._stop_reason, "gait_regression")
            best = torch.load(Path(directory) / "best.pt", weights_only=True)
            latest = torch.load(Path(directory) / "latest.pt", weights_only=True)
            self.assertEqual(best["optimizer_updates"], 0)
            self.assertEqual(latest["optimizer_updates"], 2)
            self.assertEqual(latest["stop_reason"], "gait_regression")

    def test_rejected_initial_gait_stops_before_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer, envs, agent = self.trainer(directory)
            trainer.eval = lambda agent, step: dict(BASELINE, walking_success_rate=0.)
            trainer.begin(agent)
            self.assertEqual(envs.calls, 0)
            self.assertEqual(agent.completed_updates, 0)
            self.assertEqual(trainer._stop_reason, "gait_baseline_rejected")

    def test_protocol_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer, envs, agent = self.trainer(directory)
            trainer._record_gait_guard(BASELINE, 0)
            with patch.object(trainer.eval_envs, "evaluation_protocol", return_value=dict(PROTOCOL, seed=11101)):
                with self.assertRaises(ValueError):
                    trainer._record_gait_guard(BASELINE, 1)

    def test_guard_video_names_distinguish_updates_at_the_same_collection_step(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer, envs, agent = self.trainer(directory)
            trainer.update_count = 8
            first = "step_0171520" + trainer._evaluation_artifact_suffix(0) + ".mp4"
            trainer.update_count = 16
            second = "step_0171520" + trainer._evaluation_artifact_suffix(0) + ".mp4"
            self.assertNotEqual(first, second)
            self.assertEqual(second, "step_0171520_update_000016.mp4")
            trainer.gait_guard_enabled = False
            self.assertEqual(trainer._evaluation_artifact_suffix(0), "")
            self.assertEqual(trainer._evaluation_artifact_suffix(1), "_env_01")


class RecoveryCommandTests(unittest.TestCase):
    def test_composes_real_hydra_recipe_and_keeps_learning_objective(self):
        for cap in (32, 128):
            command = recovery_command(Path("D:/old pilot"), Path("D:/new run"), cap)
            report = validate_command(command)
            self.assertEqual(report["optimizer_update_cap"], cap)
            self.assertEqual(report["conservative_replay_warmup_ticks"], 171520)
            self.assertEqual(report["effective_evaluation_seed"], 11100)
            self.assertTrue(report["reward_bc_and_exploration_unchanged"])

    def test_completed_history_requires_all_declared_checks(self):
        rows = [dict(phase="baseline" if update == 0 else "preservation", optimizer_updates=update,
                     protocol=dict(PROTOCOL), passed=True) for update in range(0, 33, 8)]
        self.assertTrue(validate_guard_history(rows, 32)["passed"])
        with self.assertRaises(ValueError):
            validate_guard_history(rows[:2] + rows[3:], 32)

    def test_failed_history_cannot_be_promoted(self):
        rows = [dict(phase="baseline", optimizer_updates=0, protocol=dict(PROTOCOL), passed=True),
                dict(phase="preservation", optimizer_updates=8, protocol=dict(PROTOCOL), passed=False)]
        self.assertFalse(validate_guard_history(rows, 32)["passed"])

    def test_selected_initialization_keeps_counter_provenance_without_resuming_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "latest.pt"
            checkpoint.write_bytes(b"trusted fixture checkpoint")
            info = dict(finite_agent=True, step=171520, optimizer_updates=16)
            report = Path(directory) / "recovery_report.json"
            report.write_text(json.dumps(dict(checkpoint=info, latest_checkpoint_sha256=sha256(checkpoint),
                                             preserved_inputs_unchanged=True, source_unchanged=True,
                                             status="halted_gait_regression")))
            admitted = validate_selected_checkpoint(info, checkpoint, explicit=True, source_report=report)
            self.assertEqual(admitted["previous_optimizer_updates"], 16)
            self.assertTrue(admitted["fresh_optimizer_and_replay"])
            command = recovery_command(Path("D:/old pilot"), Path("D:/new run"), 128, initial=checkpoint)
            self.assertIn(f"init_from={checkpoint.as_posix()}", command)
            self.assertFalse(any(value.startswith("resume_from=") for value in command))
            report.write_text(json.dumps(dict(checkpoint=info, latest_checkpoint_sha256="changed")))
            with self.assertRaises(ValueError):
                validate_selected_checkpoint(info, checkpoint, explicit=True, source_report=report)

    def test_default_initialization_still_requires_zero_step_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "latest.pt"
            checkpoint.write_bytes(b"fixture")
            with self.assertRaises(ValueError):
                validate_selected_checkpoint(dict(finite_agent=True, step=171520, optimizer_updates=16), checkpoint)


if __name__ == "__main__":
    unittest.main()
