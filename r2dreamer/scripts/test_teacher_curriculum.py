"""CPU checks for phase lineage, budget admission and video rendering scope."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from run_teacher_curriculum import phases, phase_overrides, validate_phase_result
from test_trainer_runtime import Agent, Envs, Logger, Replay, config
from trainer import OnlineTrainer


class VideoEnvs(Envs):
    def __init__(self, pose_ids):
        super().__init__()
        self.pose_ids = pose_ids
        self.render_calls = []

    def reset_pose_ids(self):
        return self.pose_ids

    def render_selected(self, indices, **kwargs):
        self.render_calls.append(list(indices))
        return [dict(frame=np.zeros((64, 64, 3), np.uint8), error=None)
                if index in indices else None for index in range(2)]


class CurriculumTests(unittest.TestCase):
    def test_curriculum_composes_actual_hydra_recipes(self):
        import os
        from hydra import compose, initialize_config_dir

        with patch.dict(os.environ, TEACHER_RUN_DIR="/tmp/phase", TEACHER_SCENE="/tmp/scene.xml"):
            with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
                configs = [compose(config_name="walking_mjx_teacher_curriculum",
                                   overrides=phase_overrides(phase, "/tmp/parent with spaces.pt"))
                           for phase in phases()]
        self.assertEqual(sum(c.env.steps for c in configs), 8000000)
        self.assertEqual(configs[0].env.mjx_task, "standing")
        self.assertFalse(configs[0].env.rsi_stratified)
        self.assertFalse(configs[0].env.eval_balanced_pose_ids)
        self.assertEqual(configs[0].trainer.best_metric_key, "standing_stable_seconds")
        self.assertEqual([c.env.training_pose_id for c in configs[1:5]], [1, 2, 3, 4])
        self.assertEqual(list(configs[-1].env.rsi_probabilities), [.2, .2, .2, .4])
        for c in configs:
            self.assertEqual(c.init_from, "/tmp/parent with spaces.pt")
            self.assertIsNone(c.resume_from)
            self.assertIsNone(c.transfer_from)
            self.assertEqual(c.env.env_num, 128)
            self.assertEqual(c.buffer.max_size, 16384)
            self.assertTrue(c.trainer.eval_video_final_only)
            self.assertFalse(c.trainer.train_video)

    def test_interrupted_or_short_phase_cannot_advance(self):
        phase = phases()[1]
        metadata = dict(step=1000280, optimizer_updates=1400, stop_reason=None)
        evaluation = dict(step=1000280, trials=[dict(start_pose_id=i) for i in range(5)] * 4)
        validate_phase_result(phase, metadata, evaluation)
        for update in (dict(step=900000), dict(stop_reason="SIGTERM"),
                       dict(stop_reason="optimizer_update_limit"), dict(optimizer_updates=0)):
            with self.assertRaises(ValueError):
                validate_phase_result(phase, {**metadata, **update}, evaluation)
        bad = deepcopy(evaluation)
        bad["trials"][0]["start_pose_id"] = 1
        with self.assertRaises(ValueError):
            validate_phase_result(phase, metadata, bad)

    def test_video_only_at_phase_end_and_one_complete_episode_per_pose(self):
        for ids, expected in (([1, 1], [0]), ([1, 2], [0, 1])):
            with self.subTest(ids=ids), tempfile.TemporaryDirectory() as directory:
                envs = VideoEnvs(ids)
                trainer = OnlineTrainer(config(steps=1000, eval_every=100, eval_video=True,
                    eval_video_final_only=True, eval_video_one_per_pose=True,
                    eval_video_tensorboard=False, eval_video_save_mp4=True),
                    Replay(), Logger(), Path(directory), envs, envs)
                agent = Agent()
                trainer.eval(agent, 500)
                self.assertFalse(envs.render_calls)
                def encode(path, frames, **kwargs):
                    Path(path).write_bytes(b"fake codec bytes")
                with patch("imageio.v2.mimsave", side_effect=encode):
                    trainer.eval(agent, 1001)
                self.assertEqual(envs.render_calls, [expected, expected])
                manifests = sorted((Path(directory) / "eval_videos").glob("*.json"))
                self.assertEqual(len(manifests), len(expected))
                self.assertEqual([json.loads(path.read_text())["start_pose_id"] for path in manifests],
                                 [ids[i] for i in expected])


if __name__ == "__main__":
    unittest.main()
