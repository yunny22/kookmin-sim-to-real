import unittest

from xycar_rl.steering_stabilizer import (
    AdaptiveSteeringStabilizer,
    SteeringStabilizerConfig,
)


class AdaptiveSteeringStabilizerTest(unittest.TestCase):
    def test_small_alternating_commands_are_damped(self):
        stabilizer = AdaptiveSteeringStabilizer()
        outputs = [stabilizer.update(value) for value in (0.10, -0.10, 0.10, -0.10)]
        self.assertLess(max(abs(value) for value in outputs), 0.06)
        self.assertEqual(sum(a * b < 0.0 for a, b in zip(outputs, outputs[1:])), 0)

    def test_curve_command_responds_faster_than_straight_command(self):
        config = SteeringStabilizerConfig(
            straight_alpha=0.25,
            curve_alpha=0.80,
            straight_rate_limit=0.08,
            curve_rate_limit=0.30,
        )
        straight = AdaptiveSteeringStabilizer(config)
        curve = AdaptiveSteeringStabilizer(config)
        straight_output = straight.update(0.10)
        curve_output = curve.update(0.60)
        self.assertLess(straight_output, 0.05)
        self.assertGreater(curve_output, 0.25)

    def test_growing_same_direction_command_gets_turn_in_lead(self):
        config = SteeringStabilizerConfig(
            straight_alpha=1.0,
            curve_alpha=1.0,
            straight_rate_limit=1.0,
            curve_rate_limit=1.0,
            deadband=0.0,
            turn_in_anticipation_gain=0.5,
            turn_in_anticipation_threshold=0.04,
        )
        stabilizer = AdaptiveSteeringStabilizer(config)
        self.assertAlmostEqual(stabilizer.update(0.10), 0.15)
        self.assertAlmostEqual(stabilizer.update(0.20), 0.25)

    def test_turn_in_lead_does_not_amplify_countersteer(self):
        config = SteeringStabilizerConfig(
            straight_alpha=1.0,
            curve_alpha=1.0,
            straight_rate_limit=1.0,
            curve_rate_limit=1.0,
            deadband=0.0,
            zero_crossing_threshold=0.0,
            turn_in_anticipation_gain=0.5,
            turn_in_anticipation_threshold=0.04,
        )
        stabilizer = AdaptiveSteeringStabilizer(config)
        stabilizer.reset(0.10)
        self.assertAlmostEqual(stabilizer.update(-0.20), -0.20)

    def test_reset_removes_previous_episode_state(self):
        stabilizer = AdaptiveSteeringStabilizer()
        stabilizer.update(0.8, curve_hint=1.0)
        stabilizer.reset()
        self.assertEqual(stabilizer.update(0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
