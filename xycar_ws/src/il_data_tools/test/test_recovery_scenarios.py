import math
import random
import unittest

from il_data_tools.recovery_scenario_manager import (
    sample_recovery_pose,
    stopped_recovery_requires_retry,
)


class RecoveryScenarioTests(unittest.TestCase):
    def test_stopped_recovery_retries_only_after_motion_and_hold(self):
        self.assertFalse(
            stopped_recovery_requires_retry(False, "recovery", 2.0, 1.0)
        )
        self.assertFalse(
            stopped_recovery_requires_retry(True, "bad_data", 2.0, 1.0)
        )
        self.assertFalse(
            stopped_recovery_requires_retry(True, "recovery", 0.9, 1.0)
        )
        self.assertTrue(
            stopped_recovery_requires_retry(True, "recovery", 1.0, 1.0)
        )

    def test_default_nominal_pose_matches_current_rule_path_bias(self):
        sample = sample_recovery_pose(random.Random(7), [0.0], [0.0])
        distance = math.hypot(
            sample["x"] - sample["yellow_x"],
            sample["y"] - sample["yellow_y"],
        )
        self.assertAlmostEqual(distance, 0.05, places=6)

    def test_pose_sampling_is_seed_reproducible(self):
        first = sample_recovery_pose(random.Random(42), [-0.1, 0.1], [-7.0, 7.0])
        second = sample_recovery_pose(random.Random(42), [-0.1, 0.1], [-7.0, 7.0])
        self.assertEqual(first, second)

    def test_samples_cover_straights_curves_and_both_error_signs(self):
        rng = random.Random(2026)
        samples = [
            sample_recovery_pose(rng, [-0.15, -0.06, 0.06, 0.15], [-10.0, -4.0, 4.0, 10.0])
            for _ in range(300)
        ]
        segment_indices = {int(sample["segment_index"]) for sample in samples}
        lateral = {sample["lateral_offset_m"] for sample in samples}
        yaw = {sample["yaw_offset_deg"] for sample in samples}
        self.assertGreater(len(segment_indices), 35)
        self.assertLess(min(lateral), 0.0)
        self.assertGreater(max(lateral), 0.0)
        self.assertLess(min(yaw), 0.0)
        self.assertGreater(max(yaw), 0.0)
        for sample in samples:
            self.assertTrue(-5.0 < sample["x"] < 3.8)
            self.assertTrue(-3.8 < sample["y"] < 3.0)
            self.assertTrue(math.isfinite(sample["yaw_rad"]))


if __name__ == "__main__":
    unittest.main()
