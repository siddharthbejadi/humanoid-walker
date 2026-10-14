"""CPU-only regression checks for capacity admission; no simulator required."""
import unittest

from teacher_capacity_probe import case_overrides, validate_case


class CapacityAdmissionTest(unittest.TestCase):
    def test_requires_updates_and_matching_balanced_final_evaluation(self):
        checkpoint = {"step": 100000, "optimizer_updates": 32, "finite_agent": True,
                      "stop_reason": "optimizer_update_limit"}
        trials = [{"start_pose_id": pose} for pose in range(5) for _ in range(4)]
        evaluations = [{"step": 0, "trials": trials}, {"step": 100000, "trials": trials}]
        validate_case(checkpoint, evaluations, 32)
        for patch in ({"optimizer_updates": 0}, {"finite_agent": False}, {"step": 99999},
                      {"stop_reason": "SIGTERM"}):
            with self.assertRaises(ValueError):
                validate_case({**checkpoint, **patch}, evaluations, 32)
        with self.assertRaises(ValueError):
            validate_case(checkpoint, evaluations[:1], 32)
        wrong = {"step": 100000, "trials": [{"start_pose_id": 4}] * 20}
        with self.assertRaises(ValueError):
            validate_case(checkpoint, [evaluations[0], wrong], 32)

    def test_replay_and_budget_scale_with_world_count(self):
        for worlds in (64, 128, 256):
            values = dict(item.split("=", 1) for item in case_overrides(worlds, 32))
            self.assertEqual(int(values["buffer.max_size"]) // worlds, 128)
            self.assertGreater(int(values["env.steps"]), 67 * worlds * 10)
            self.assertGreater(int(values["trainer.learning_start_deadline_steps"]), 67 * worlds * 10)
            self.assertEqual(values["trainer.stop_after_optimizer_updates"], "32")


if __name__ == "__main__":
    unittest.main()
