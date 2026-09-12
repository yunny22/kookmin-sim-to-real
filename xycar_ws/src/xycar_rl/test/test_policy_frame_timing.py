import unittest

from rclpy.qos import HistoryPolicy, ReliabilityPolicy

from xycar_rl.policy_runtime_node import (
    inference_frame_due,
    latest_sensor_qos,
    temporal_pair_requires_reset,
)


class PolicyFrameTimingTest(unittest.TestCase):
    def test_non_positive_rate_processes_every_source_frame(self):
        self.assertTrue(inference_frame_due(10.01, 10.0, 0.0, 0.0))

    def test_rate_slack_accepts_jitter_near_seven_hz(self):
        self.assertTrue(inference_frame_due(10.11, 10.0, 7.0, 0.04))
        self.assertFalse(inference_frame_due(10.08, 10.0, 7.0, 0.04))

    def test_temporal_pair_resets_after_a_long_or_reversed_gap(self):
        previous = 1_000_000_000
        self.assertFalse(
            temporal_pair_requires_reset(1_140_000_000, previous, 0.25)
        )
        self.assertTrue(
            temporal_pair_requires_reset(1_300_000_000, previous, 0.25)
        )
        self.assertTrue(
            temporal_pair_requires_reset(900_000_000, previous, 0.25)
        )

    def test_image_subscription_keeps_only_the_latest_frame(self):
        qos = latest_sensor_qos()
        self.assertEqual(qos.history, HistoryPolicy.KEEP_LAST)
        self.assertEqual(qos.depth, 1)
        self.assertEqual(qos.reliability, ReliabilityPolicy.BEST_EFFORT)


if __name__ == "__main__":
    unittest.main()
