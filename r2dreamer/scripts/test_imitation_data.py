"""CPU fixtures for temporal labels, episode splits and demonstration provenance."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from full_command import motor_targets_from_actions, normalize_motor_targets
from imitation_data import ImitationDataset, align_episode, save_dataset, validate_arrays
from scripted_expert import ScriptedExpert
from scripts.collect_script_free_demonstrations import GPUStudentProvider, collect_batch, validate_student_reset


def fixture(steps=160):
    proprio = np.zeros((3, steps + 1, 45), np.float32)
    proprio[:, :, 0] = np.arange(steps + 1)
    action = np.zeros((3, steps, 19), np.float32)
    action[:, :, 0] = np.arange(steps) / (steps + 1)
    return dict(proprio=proprio, action=action, reward=np.tile(np.arange(steps, dtype=np.float32), (3, 1)),
                is_terminal=np.zeros((3, steps), bool), valid=np.ones((3, steps), bool),
                present=np.ones((3, steps), bool), split=np.arange(3, dtype=np.int8),
                episode_seed=np.arange(3, dtype=np.int64), rollout_id=np.arange(3, dtype=np.int64))


class ToyProvider:
    worlds = 1
    command_center = np.zeros(19, np.float32)
    command_scale = np.full(19, 10., np.float32)

    def __init__(self, fail=False):
        self.fail = fail

    def normalize(self, targets):
        return normalize_motor_targets(targets, self.command_center, self.command_scale)

    def audit(self):
        return dict(rotation=np.eye(3, dtype=np.float32)[None], ctrl=self.ctrl.copy(),
                    qpos=np.zeros((1, 26), np.float32), qvel=np.zeros((1, 25), np.float32),
                    time=np.asarray([self.steps * .02], np.float32), state_rng=np.asarray([[1, 2]], np.uint32),
                    bridge_rng=np.asarray([[3, 4]], np.uint32),
                    step_count=np.asarray([self.steps], np.int32),
                    contact_sensor=np.zeros((1, 2, 32, 7), np.float32))

    def reset(self, seed):
        self.steps = 0
        self.ctrl = np.zeros((1, 19), np.float32)
        self.observation = np.zeros((1, 45), np.float32)
        return dict(proprio=self.observation.copy(), audit=self.audit())

    def video_frame(self):
        return np.full((64, 64, 3), self.steps, np.uint8)

    def step(self, action):
        target = motor_targets_from_actions(action, self.command_center, self.command_scale)
        self.ctrl = np.clip(target, self.ctrl - .16, self.ctrl + .16)
        self.steps += 1
        self.observation[:, 0] = self.ctrl[:, 0]
        failed = self.fail and self.steps == 2
        ended = failed or self.steps == 4
        return dict(proprio=self.observation.copy(), audit=self.audit(), reward=np.asarray([target[0, 0]], np.float32),
                    terminal=np.asarray([failed]), ended=np.asarray([ended]),
                    causes=dict(physical_fall=np.asarray([failed]), wall_hit=np.asarray([False]),
                                time_limit=np.asarray([ended and not failed])),
                    logs=dict(valid_left_step=np.asarray([0.]), valid_right_step=np.asarray([0.]), alternating_steps=np.asarray([0.])))


class ToyExpert:
    def reset(self, rotation):
        pass

    def targets(self, step, rotation):
        result = np.zeros((1, 19), np.float32)
        result[:, 0] = step + 1
        return result


class ImitationDataTests(unittest.TestCase):
    def test_provider_audit_uses_model_contact_descriptors_not_imu_slices(self):
        worlds = 2
        sensors = np.arange(worlds * 463, dtype=np.float32).reshape(worlds, 463)
        descriptors = {"mjx_left_sole_contacts": SimpleNamespace(adr=np.asarray([15]), dim=np.asarray([224])),
                       "mjx_right_sole_contacts": SimpleNamespace(adr=np.asarray([239]), dim=np.asarray([224]))}
        data = SimpleNamespace(sensordata=sensors, qpos=np.zeros((worlds, 26), np.float32),
            qvel=np.zeros((worlds, 25), np.float32), ctrl=np.zeros((worlds, 19), np.float32), time=np.zeros(worlds),
            xmat=np.tile(np.eye(3, dtype=np.float32), (worlds, 3, 1, 1)),
            qfrc_actuator=np.arange(worlds * 25, dtype=np.float32).reshape(worlds, 25))
        # Real raw.sensor_slices contains only the IMUs. Contact descriptors
        # are public MuJoCo model accessors with one-element adr/dim arrays.
        raw = SimpleNamespace(sensor_slices={"imu_accel": (0, 3)},
            m=SimpleNamespace(sensor=descriptors.__getitem__), base_id=1, act_dof=np.arange(6, 25),
            _features=lambda _: (np.zeros((worlds, 8)), np.zeros((worlds, 2)),
                                  np.zeros((worlds, 2)), np.zeros((worlds, 2, 2))))
        env = SimpleNamespace(raw=raw, rng=np.asarray([7, 8], np.uint32), jax=SimpleNamespace(device_get=lambda value: value,
            vmap=lambda function: function), state=SimpleNamespace(pipeline_state=data,
            info=dict(rng=np.zeros((worlds, 2), np.uint32), step_count=np.zeros(worlds, np.int32))))
        provider = GPUStudentProvider.__new__(GPUStudentProvider)
        provider.env, provider.worlds = env, worlds
        audit = provider.audit()
        self.assertEqual(audit["contact_sensor"].shape, (2, 2, 32, 7))
        self.assertEqual(audit["contact_sensor"][0, 0, 0, 0], sensors[0, 15])
        self.assertEqual(audit["contact_sensor"][1, 1, 0, 0], sensors[1, 239])
        np.testing.assert_array_equal(audit["actuator_force"], data.qfrc_actuator[:, 6:25])
        self.assertEqual(audit["rotation"].shape, (worlds, 3, 3))
        np.testing.assert_array_equal(audit["bridge_rng"], np.asarray([[7, 8], [7, 8]], np.uint32))

    def test_current_label_and_previous_recurrent_inputs_are_causal(self):
        arrays = fixture(5)
        arrays["is_terminal"][:, -1] = True
        data = align_episode(arrays, 0)
        np.testing.assert_array_equal(data["action"][0], 0.)
        self.assertEqual(data["reward"][0, 0], 0.)
        self.assertTrue(data["is_first"][0, 0])
        np.testing.assert_array_equal(data["action"][2], arrays["action"][0, 1])
        np.testing.assert_array_equal(data["expert_action"][2], arrays["action"][0, 2])
        self.assertEqual(data["reward"][2, 0], arrays["reward"][0, 1])
        self.assertTrue(data["is_terminal"][-1, 0])
        self.assertEqual(data["expert_mask"][-1, 0], 0.)
        self.assertEqual(data["loss_mask"][-1, 0], 1.)

    def test_midsequence_burnin_does_not_invent_an_episode_reset(self):
        data = align_episode(fixture(), 0, start=32, length=64, burnin=16)
        self.assertFalse(data["is_first"].any())
        self.assertFalse(data["expert_mask"][:16].any())
        self.assertFalse(data["loss_mask"][:16].any())
        self.assertTrue(data["expert_mask"][16:].all())

    def test_sampler_supervises_true_startup_and_midsequence_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demonstrations.npz"
            save_dataset(path, fixture(), dict(training_contract={"controller": "full_command"}))
            dataset = ImitationDataset(path)
            data = dataset.numpy_batch(8, 32, burnin=16, rng=12)
            self.assertTrue(data["is_first"][:2, 0].all())
            self.assertTrue(data["expert_mask"][:2, 0].all())
            self.assertFalse(data["expert_mask"][2:, :16].any())
            self.assertFalse(data["loss_mask"][2:, :16].any())
            self.assertEqual(data["proprio"].shape, (8, 48, 45))

    def test_padding_and_unsafe_labels_stay_masked(self):
        arrays = fixture(6)
        arrays["present"][0, 3:] = False
        arrays["valid"][0] = False
        data = align_episode(arrays, 0, length=7)
        self.assertFalse(data["expert_mask"].any())
        self.assertTrue(data["loss_mask"][:4].all())
        self.assertFalse(data["loss_mask"][4:].any())

    def test_seed_leakage_and_nonfinite_data_are_rejected(self):
        arrays = fixture()
        arrays["episode_seed"][1] = arrays["episode_seed"][0]
        with self.assertRaisesRegex(ValueError, "leak"):
            validate_arrays(arrays)
        arrays = fixture()
        arrays["action"][0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            validate_arrays(arrays)

    def test_hash_tampering_and_pickle_data_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demonstrations.npz"
            save_dataset(path, fixture(), {})
            path.write_bytes(path.read_bytes() + b"tampered")
            with self.assertRaisesRegex(ValueError, "hash"):
                ImitationDataset(path)
        arrays = fixture()
        arrays["unsafe_object"] = np.asarray([{"value": 1}], dtype=object)
        with self.assertRaisesRegex(ValueError, "Object"):
            validate_arrays(arrays)

    def test_fingerprint_binds_semantic_manifest_as_well_as_npz(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demonstrations.npz"
            save_dataset(path, fixture(), dict(acceptance={"safe_bc_labels": 480}))
            before = ImitationDataset(path).fingerprint
            sidecar = path.with_suffix(".manifest.json")
            manifest = json.loads(sidecar.read_text(encoding="utf-8"))
            manifest["acceptance"]["safe_bc_labels"] = 1
            sidecar.write_text(json.dumps(manifest), encoding="utf-8")
            after = ImitationDataset(path).fingerprint
            self.assertEqual(before["data_sha256"], after["data_sha256"])
            self.assertNotEqual(before["manifest_sha256"], after["manifest_sha256"])

    def test_empty_safe_split_stops_behavior_cloning(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demonstrations.npz"
            arrays = fixture()
            arrays["valid"][:] = False
            save_dataset(path, arrays, {})
            with self.assertRaisesRegex(ValueError, "No safe expert labels"):
                ImitationDataset(path).numpy_batch(2, 64)

    def test_collector_records_pre_action_target_and_post_action_slew(self):
        arrays, reports = collect_batch(ToyProvider(), ToyExpert(), 0, 8, 0)
        self.assertEqual(arrays["proprio"][0, 0, 0], 0.)
        self.assertAlmostEqual(float(arrays["action"][0, 0, 0]), .1)
        self.assertEqual(arrays["desired_ctrl"][0, 0, 0], 1.)
        self.assertAlmostEqual(float(arrays["applied_ctrl"][0, 0, 0]), .16)
        self.assertAlmostEqual(float(arrays["proprio"][0, 1, 0]), .16)
        self.assertEqual(arrays["reward"][0, 0], 1.)
        self.assertEqual(reports[0]["control_transitions"], 4)
        self.assertFalse(reports[0]["bilateral_motion_observed"])

    def test_video_callback_captures_start_and_terminal_without_reset_frames(self):
        frames = []
        collect_batch(ToyProvider(fail=True), ToyExpert(), 0, 8, 0, video_sink=lambda frame: frames.append(frame.copy()))
        self.assertEqual(len(frames), 2)
        self.assertEqual(int(frames[0][0, 0, 0]), 0)
        self.assertEqual(int(frames[-1][0, 0, 0]), 2)

    def test_student_reset_alias_and_state_checks_use_bounded_sensor_noise(self):
        first = ToyProvider().reset(0)
        repeated = copy.deepcopy(first)
        repeated["proprio"][0, 19] += .0005
        self.assertTrue(validate_student_reset(first, repeated, first["proprio"].copy())["passed"])
        changed = copy.deepcopy(repeated)
        changed["audit"]["qpos"][0, 0] = .001
        self.assertFalse(validate_student_reset(first, changed, first["proprio"].copy())["passed"])
        changed = copy.deepcopy(repeated)
        changed["audit"]["bridge_rng"][0, 0] += 1
        self.assertFalse(validate_student_reset(first, changed, first["proprio"].copy())["passed"])
        retained = first["proprio"].copy()
        retained[0, 0] = .2
        self.assertFalse(validate_student_reset(first, repeated, retained)["passed"])

    def test_failed_collector_rollout_is_retained_without_safe_bc_labels(self):
        arrays, reports = collect_batch(ToyProvider(fail=True), ToyExpert(), 0, 8, 0)
        self.assertEqual(int(arrays["present"].sum()), 2)
        self.assertFalse(arrays["valid"].any())
        self.assertTrue(arrays["is_terminal"][0, 1])
        self.assertIn("physical_fall", reports[0]["termination_reasons"])
        self.assertFalse(reports[0]["physical_safe"])

    def test_external_expert_cycles_and_uses_pre_action_feedback(self):
        ctrl = np.zeros((5, 19), np.float32)
        ctrl[:, 1] = np.arange(5) * .1
        expert = ScriptedExpert(ctrl, cycle_start=2, cycle_len=2)
        rotation = np.eye(3, dtype=np.float32)[None]
        expert.reset(rotation)
        first = expert.targets(0, rotation)
        self.assertEqual(first[0, 1], 0.)
        repeat = expert.targets(4, rotation)
        self.assertAlmostEqual(float(repeat[0, 1]), .2)
        rotation = rotation.copy()
        rotation[0, 0, 2] = np.sin(.1)
        feedback = expert.targets(5, rotation)
        self.assertLess(feedback[0, 0], 0.)
        self.assertGreater(feedback[0, 4], 0.)


if __name__ == "__main__":
    unittest.main()
