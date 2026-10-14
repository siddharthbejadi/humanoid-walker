"""CPU checks for curriculum selection, diagnostic rendering and clock logging."""

from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envs.mjx_teacher_env import MJXTeacherEnv
from envs.mjx_walking.scene import training_pose_reset_index


class CurriculumEnvironmentTests(unittest.TestCase):
    def test_exact_pose_id_maps_across_category_order(self):
        ids = np.array([0, 4, 3, 1, 2], dtype=np.int32)
        before = ids.copy()
        for pose_id in range(5):
            index = training_pose_reset_index(ids, pose_id)
            self.assertEqual(ids[index], pose_id)
        np.testing.assert_array_equal(ids, before)
        self.assertEqual(training_pose_reset_index(ids, np.int32(2)), 4)

    def test_invalid_or_ambiguous_selection_is_rejected(self):
        for pose_id in (True, False, 1.0, 1.5, "1", None, -1, 5):
            with self.subTest(pose_id=pose_id), self.assertRaises(ValueError):
                training_pose_reset_index([0, 4, 3, 1, 2], pose_id)
        for ids in ([0, 1, 1], [0, 2], [0., 1.], [[0, 1]]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                training_pose_reset_index(ids, 1)

    def test_standing_repeatable_eval_restores_seed_and_state(self):
        env = NS(evaluation=True, _balanced_evaluation=False,
                 _repeatable_evaluation=True, _evaluation_seed=10042,
                 jax=NS(random=NS(PRNGKey=lambda seed: ("key", seed))),
                 rng="advanced", state="previous")
        MJXTeacherEnv.begin_evaluation(env)
        self.assertEqual(env.rng, ("key", 10042))
        self.assertIsNone(env.state)

    def test_standing_clock_deltas_sum_true_seconds_across_resets(self):
        previous = torch.tensor([.02, .04, .10])
        current = torch.tensor([.04, .04, 0.])
        delta = MJXTeacherEnv._standing_seconds_delta(
            current, previous, torch.tensor([False, False, True]))
        torch.testing.assert_close(delta, torch.tensor([.02, 0., 0.]))
        total = 0.
        previous = None
        for index, value in enumerate((0., .02, .04, .04, .06)):
            current = torch.tensor([value])
            total += MJXTeacherEnv._standing_seconds_delta(
                current, previous, torch.tensor([index == 0])).item()
            previous = current
        self.assertAlmostEqual(total, .06, places=6)

    def make_render_env(self):
        accessed = []

        class SelectedArray:
            def __init__(self, name, width):
                self.name, self.width = name, width

            def __getitem__(self, index):
                accessed.append((self.name, index))
                return np.full(self.width, index, dtype=float)

            def __array__(self, *args, **kwargs):
                raise AssertionError("Must not transfer an entire world batch")

        env = MJXTeacherEnv.__new__(MJXTeacherEnv)
        env.env_num = 20
        env.raw = NS(m="model")
        env.state = NS(pipeline_state=NS(qpos=SelectedArray("qpos", 2),
                                         qvel=SelectedArray("qvel", 2),
                                         ctrl=SelectedArray("ctrl", 1)))
        env._diagnostic_renderer = env._diagnostic_render_size = env._diagnostic_data = None
        created = []

        class Renderer:
            def __init__(self, model, width, height):
                created.append(self)
                self.frame = np.zeros((height, width, 3), np.uint8)
                self.closed = False

            def update_scene(self, data, camera):
                self.frame.fill(int(data.qpos[0]))

            def render(self):
                return self.frame

            def close(self):
                self.closed = True

        fake = NS(Renderer=Renderer,
                  MjData=lambda model: NS(qpos=np.zeros(2), qvel=np.zeros(2), ctrl=np.zeros(1)),
                  mj_forward=lambda model, data: None)
        return env, fake, accessed, created

    def test_render_only_selected_worlds_and_reuse_renderer(self):
        env, fake, accessed, created = self.make_render_env()
        with patch.dict(sys.modules, {"mujoco": fake}):
            result = env.render_selected([1, 4], width=64, height=64)
            self.assertEqual(len(result), 20)
            self.assertTrue(all(value is None for i, value in enumerate(result) if i not in (1, 4)))
            self.assertTrue(np.all(result[1]["frame"] == 1))
            self.assertTrue(np.all(result[4]["frame"] == 4))
            self.assertEqual(accessed, [(field, index) for index in (1, 4)
                                        for field in ("qpos", "qvel", "ctrl")])
            first = env.render_first(width=64, height=64)
            self.assertIsNone(first["error"])
            self.assertTrue(np.all(first["frame"] == 0))
            self.assertEqual(len(created), 1)
            env.render_selected([], width=64, height=64)
            self.assertEqual(len(created), 1)

    def test_render_rejects_bad_world_indices(self):
        env, _, _, _ = self.make_render_env()
        for selected in ([True], [1.0], [-1], [20], [1, 1]):
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                env.render_selected(selected)

    def test_render_reports_failures_only_for_selected_worlds(self):
        env, fake, _, _ = self.make_render_env()
        env.state = None
        with patch.dict(sys.modules, {"mujoco": fake}):
            result = env.render_selected([3])
        self.assertIn("has not been reset", result[3]["error"])
        self.assertIsNone(result[3]["frame"])
        self.assertTrue(all(value is None for i, value in enumerate(result) if i != 3))


if __name__ == "__main__":
    unittest.main()
