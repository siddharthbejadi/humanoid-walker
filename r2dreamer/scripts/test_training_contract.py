"""CPU-only behavioral checks for checkpoint physical-task compatibility."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_contract import build_training_contract, check_checkpoint_contract


def make_environment(root):
    (root / "meshes").mkdir()
    (root / "meshes" / "foot.stl").write_bytes(b"geometry-one")
    (root / "scene.xml").write_text(
        '<mujoco><compiler meshdir="meshes"/><asset>'
        '<mesh name="foot" file="foot.stl"/></asset></mujoco>', encoding="utf-8")
    (root / "poses.json").write_text('{"poses": {}}', encoding="utf-8")
    arrays = {name: np.zeros((2, 3), np.float32) for name in (
        "body_mass", "body_inertia", "body_pos", "body_quat", "jnt_range",
        "geom_pos", "geom_quat", "geom_size", "geom_friction", "mesh_vert", "mesh_face")}
    model = NS(nu=19, actuator=lambda i: NS(name=f"joint_{i}"),
               actuator_ctrlrange=np.tile([-1.57, 1.57], (19, 1)),
               actuator_forcerange=np.tile([-1.08, 1.08], (19, 1)),
               actuator_gainprm=np.ones((19, 10)), actuator_biasprm=np.ones((19, 10)),
               opt=NS(timestep=.002), **arrays)
    raw = NS(backend="mjx-warp", m=model, scene=root / "scene.xml",
             action_scale=np.full(19, np.deg2rad(48)), standing_ctrl=np.zeros(19),
             control_dt=.02, action_repeat=10, control_slew_rad_s=2., naconmax=64 * 256,
             task="walking", reward_profile="step_progress_v1", step_clearance_m=.020,
             simplified_rewards=True,
             gait_curriculum=True, backward_drift_weight=.1, stance_knee_sag_weight=.08,
             rsi_stratified=True, rsi_probabilities=np.array([.2, .2, .2, .4]),
             rsi_counts=np.array([1, 1, 1, 2]), rsi_offsets=np.array([0, 1, 2, 3]),
             rsi_pose_ids=np.arange(5), rsi_reference_velocity=True,
             rsi_qpos=np.zeros((5, 26)), rsi_qvel=np.zeros((5, 25)),
             rsi_ctrl=np.zeros((5, 19)), rsi_phase_offset=np.zeros(5))
    env = NS(raw=raw, horizon=400, env_num=64,
             observation_space=NS(spaces={"proprio": NS(shape=(45,)), "teacher_state": NS(shape=(98,))}),
             action_space=NS(shape=(19,)))
    return env, NS(scene_path=str(root / "scene.xml"), pose_rsi_json=str(root / "poses.json"))


class TrainingContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.env, self.config = make_environment(self.root)
        self.contract = build_training_contract(self.env, self.config)
        self.checkpoint = {"training_contract": deepcopy(self.contract)}

    def tearDown(self):
        self.temp.cleanup()

    def test_json_and_verified_resume(self):
        self.assertEqual(json.loads(json.dumps(self.contract)), self.contract)
        self.assertIn("Verified resume", check_checkpoint_contract(self.checkpoint, self.contract)[0])

    def test_optional_pose_selector_is_recorded_and_requires_adaptation(self):
        self.assertNotIn("training_pose_id", self.contract["reset"])
        self.env.training_pose_id = 2
        current = build_training_contract(self.env, self.config)
        self.assertEqual(current["reset"]["training_pose_id"], 2)
        with self.assertRaisesRegex(ValueError, "training_pose_id"):
            check_checkpoint_contract(self.checkpoint, current)
        self.assertIn("training_pose_id", check_checkpoint_contract(
            self.checkpoint, current, mode="init")[0])
        self.env.training_pose_id = None
        self.assertEqual(build_training_contract(self.env, self.config), self.contract)

    def test_action_semantics_reject_resume_even_with_legacy_override(self):
        self.env.raw.action_scale[:] = np.deg2rad(90)
        current = build_training_contract(self.env, self.config)
        with self.assertRaisesRegex(ValueError, "action_scale_rad"):
            check_checkpoint_contract(self.checkpoint, current, allow_unverified_resume=True)

    def test_step_clearance_change_is_recorded_and_requires_adaptation(self):
        self.assertEqual(self.contract["task"]["valid_step_clearance_m"], .020)
        self.env.raw.step_clearance_m = .015
        current = build_training_contract(self.env, self.config)
        self.assertEqual(current["task"]["valid_step_clearance_m"], .015)
        with self.assertRaisesRegex(ValueError, "valid_step_clearance_m"):
            check_checkpoint_contract(self.checkpoint, current, allow_unverified_resume=True)
        self.assertIn("valid_step_clearance_m", check_checkpoint_contract(
            self.checkpoint, current, mode="init")[0])

    def test_missing_step_clearance_is_not_a_verified_resume(self):
        legacy = deepcopy(self.checkpoint)
        del legacy["training_contract"]["task"]["valid_step_clearance_m"]
        with self.assertRaisesRegex(ValueError, "valid_step_clearance_m"):
            check_checkpoint_contract(legacy, self.contract, allow_unverified_resume=True)

    def test_reference_controller_and_reward_changes_reject_resume(self):
        changes = {
            "gait_residual": True, "gait_reference_sample_offset": -1.,
            "gait_reference_speed": .5, "phase_rate_range": .5,
            "phase_obs": True, "stabilizer": (1., .04, -1., .25),
            "reward_sizes": {"fall_penalty": 150.}, "footsteps": {"step": .05},
        }
        for name, value in changes.items():
            with self.subTest(setting=name):
                setattr(self.env.raw, name, value)
                current = build_training_contract(self.env, self.config)
                with self.assertRaisesRegex(ValueError, "contract mismatch"):
                    check_checkpoint_contract(self.checkpoint, current)
                delattr(self.env.raw, name)

    def test_scripted_reference_assets_and_reset_flags_are_fingerprinted(self):
        gait = self.root / "gait.npz"
        gait.write_bytes(b"first-reference")
        self.config.mjx_scripted_gait = str(gait)
        self.env.raw.rsi_injected = np.zeros(5, dtype=bool)
        initial = build_training_contract(self.env, self.config)
        checkpoint = {"training_contract": initial}
        self.env.raw.rsi_injected[1] = True
        with self.assertRaisesRegex(ValueError, "injection_flags_sha256"):
            check_checkpoint_contract(checkpoint, build_training_contract(self.env, self.config))
        self.env.raw.rsi_injected[:] = False
        gait.write_bytes(b"different-reference")
        with self.assertRaisesRegex(ValueError, "mjx_scripted_gait_sha256"):
            check_checkpoint_contract(checkpoint, build_training_contract(self.env, self.config))

    def test_timing_torque_reward_and_reset_changes_each_reject_resume(self):
        for category, field, value in (
            ("control", "control_dt_seconds", .04),
            ("control", "actuator_forcerange_nm", [[-1.76, 1.76]] * 19),
            ("task", "reward_profile", "legacy"),
            ("reset", "probabilities", [0., .25, .25, .5])):
            with self.subTest(field=field):
                changed = deepcopy(self.contract)
                changed[category][field] = value
                with self.assertRaisesRegex(ValueError, field):
                    check_checkpoint_contract(self.checkpoint, changed)

    def test_legacy_resume_requires_explicit_opt_in(self):
        with self.assertRaisesRegex(ValueError, "allow_unverified_resume"):
            check_checkpoint_contract({}, self.contract)
        self.assertIn("UNVERIFIED resume", check_checkpoint_contract(
            {}, self.contract, allow_unverified_resume=True)[0])

    def test_explicit_adaptation_reports_physical_change(self):
        changed = deepcopy(self.contract)
        changed["control"]["action_scale_rad"] = [1.57] * 19
        for mode in ("init", "transfer"):
            messages = check_checkpoint_contract(self.checkpoint, changed, mode=mode)
            self.assertIn("action_scale_rad", messages[0])
            self.assertIn("EXPLICIT", messages[0])
            self.assertIn("UNVERIFIED", check_checkpoint_contract({}, changed, mode=mode)[0])

    def test_shape_mismatch_rejects_all_load_modes(self):
        changed = deepcopy(self.contract)
        changed["interface"]["observations"]["proprio"] = [34]
        for mode in ("resume", "init", "transfer"):
            with self.assertRaisesRegex(ValueError, "interface.observations.proprio"):
                check_checkpoint_contract(self.checkpoint, changed, mode=mode)

    def test_mesh_and_pose_file_changes_are_detected(self):
        for file in (self.root / "meshes" / "foot.stl", self.root / "poses.json"):
            with self.subTest(file=file.name):
                before = file.read_bytes()
                file.write_bytes(before + b" ")
                current = build_training_contract(self.env, self.config)
                with self.assertRaisesRegex(ValueError, "contract mismatch"):
                    check_checkpoint_contract(self.checkpoint, current)
                file.write_bytes(before)

    def test_missing_assets_fail_closed(self):
        (self.root / "meshes" / "foot.stl").unlink()
        with self.assertRaises(FileNotFoundError):
            build_training_contract(self.env, self.config)

    def test_physical_identity_survives_relocation_and_world_scaling(self):
        with TemporaryDirectory() as other:
            env, config = make_environment(Path(other))
            env.env_num = 128
            env.raw.naconmax = 128 * 256
            self.assertEqual(build_training_contract(env, config), self.contract)

    def test_non_mjx_paths_are_unchanged(self):
        self.assertIsNone(build_training_contract(NS()))
        self.assertEqual(check_checkpoint_contract({}, None), [])

    def test_unknown_schema_is_not_treated_as_legacy(self):
        with self.assertRaisesRegex(ValueError, "version"):
            check_checkpoint_contract({"training_contract": {"version": 99}}, self.contract,
                                      allow_unverified_resume=True)


if __name__ == "__main__":
    unittest.main()
