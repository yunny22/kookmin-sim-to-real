import unittest

from xycar_rl.reward import (
    calculate_reward,
    large_steering_oscillation,
    straight_high_speed_objective_weights,
)
from xycar_rl.termination import EpisodeTermination, TerminationConfig
from xycar_rl.track_geometry import TrackProjection


def projection(cross_track=0.0, heading=0.0):
    return TrackProjection(
        x=0.0,
        y=0.0,
        tangent_yaw=0.0,
        progress_m=1.0,
        progress_fraction=0.1,
        cross_track_error_m=cross_track,
        heading_error_rad=heading,
        segment_index=0,
    )


class RewardTerminationTest(unittest.TestCase):
    def test_straight_high_speed_objective_targets_command_25_and_weave(self):
        weights = straight_high_speed_objective_weights()
        self.assertAlmostEqual(weights.straight_target_speed_mps, 2.0153)
        self.assertEqual(weights.curve_target_speed_mps, weights.straight_target_speed_mps)
        self.assertGreater(weights.large_oscillation, 1.0)
        self.assertGreater(weights.steering_rate, 0.1)

    def test_forward_centered_motion_is_positive(self):
        reward = calculate_reward(
            projection=projection(),
            progress_delta_m=0.08,
            steering_norm=0.0,
            previous_steering_norm=0.0,
        )
        self.assertGreater(reward.total, 0.0)

    def test_collision_dominates_progress(self):
        reward = calculate_reward(
            projection=projection(),
            progress_delta_m=0.2,
            steering_norm=0.0,
            previous_steering_norm=0.0,
            collision=True,
        )
        self.assertLess(reward.total, -40.0)

    def test_small_or_single_steering_reversal_is_not_oscillation(self):
        self.assertEqual(
            large_steering_oscillation([0.10, -0.10, 0.10]),
            0.0,
        )
        self.assertEqual(
            large_steering_oscillation([0.40, -0.40]),
            0.0,
        )

    def test_repeated_large_bilateral_steering_is_penalized_on_straight(self):
        reward = calculate_reward(
            projection=projection(),
            progress_delta_m=0.0,
            steering_norm=0.40,
            previous_steering_norm=-0.40,
            track_curvature=0.0,
            steering_history=(0.40, -0.40, 0.40),
        )
        self.assertLess(reward.large_oscillation, 0.0)

    def test_large_steering_sequence_is_allowed_on_curve(self):
        reward = calculate_reward(
            projection=projection(),
            progress_delta_m=0.0,
            steering_norm=0.40,
            previous_steering_norm=-0.40,
            track_curvature=1.0,
            steering_history=(0.40, -0.40, 0.40),
        )
        self.assertEqual(reward.large_oscillation, 0.0)

    def test_speed_reward_requires_centered_heading(self):
        centered = calculate_reward(
            projection=projection(),
            progress_delta_m=0.08,
            steering_norm=0.0,
            previous_steering_norm=0.0,
            linear_speed_mps=0.8,
        )
        risky = calculate_reward(
            projection=projection(cross_track=0.30, heading=0.4),
            progress_delta_m=0.08,
            steering_norm=0.0,
            previous_steering_norm=0.0,
            linear_speed_mps=0.8,
        )
        self.assertGreater(centered.safe_speed, 0.0)
        self.assertAlmostEqual(centered.unsafe_speed, 0.0)
        self.assertLess(risky.unsafe_speed, 0.0)
        self.assertGreater(centered.total, risky.total)

    def test_same_safe_progress_rewards_higher_speed(self):
        slow = calculate_reward(
            projection=projection(),
            progress_delta_m=0.08,
            steering_norm=0.0,
            previous_steering_norm=0.0,
            linear_speed_mps=0.8,
        )
        fast = calculate_reward(
            projection=projection(),
            progress_delta_m=0.08,
            steering_norm=0.0,
            previous_steering_norm=0.0,
            linear_speed_mps=1.7,
        )
        self.assertGreater(fast.safe_speed, slow.safe_speed)
        self.assertGreater(fast.total, slow.total)

    def test_preview_curve_penalizes_overspeed_before_tracking_error(self):
        straight = calculate_reward(
            projection=projection(),
            progress_delta_m=0.0,
            steering_norm=0.0,
            previous_steering_norm=0.0,
            linear_speed_mps=1.60,
            preview_curvature=0.0,
        )
        upcoming_curve = calculate_reward(
            projection=projection(),
            progress_delta_m=0.0,
            steering_norm=0.0,
            previous_steering_norm=0.0,
            linear_speed_mps=1.60,
            preview_curvature=1.0,
        )
        self.assertAlmostEqual(straight.unsafe_speed, 0.0)
        self.assertLess(upcoming_curve.unsafe_speed, 0.0)
        self.assertGreater(straight.total, upcoming_curve.total)

    def test_off_track_terminates(self):
        checker = EpisodeTermination()
        result = checker.update(
            dt_sec=0.1,
            elapsed_sec=1.0,
            cross_track_error_m=0.5,
            linear_speed_mps=0.3,
            speed_command=4.0,
            collision=False,
            cumulative_forward_progress_m=1.0,
            track_length_m=30.0,
        )
        self.assertTrue(result.terminated)
        self.assertEqual(result.reason, "off_track")

    def test_stuck_requires_continuous_timeout(self):
        checker = EpisodeTermination(
            TerminationConfig(stuck_timeout_sec=0.3)
        )
        for _ in range(2):
            result = checker.update(
                dt_sec=0.1,
                elapsed_sec=0.1,
                cross_track_error_m=0.0,
                linear_speed_mps=0.0,
                speed_command=4.0,
                collision=False,
                cumulative_forward_progress_m=0.0,
                track_length_m=30.0,
            )
            self.assertFalse(result.terminated)
        result = checker.update(
            dt_sec=0.1,
            elapsed_sec=0.3,
            cross_track_error_m=0.0,
            linear_speed_mps=0.0,
            speed_command=4.0,
            collision=False,
            cumulative_forward_progress_m=0.0,
            track_length_m=30.0,
        )
        self.assertTrue(result.terminated)
        self.assertEqual(result.reason, "stuck")

    def test_collision_terminates_immediately(self):
        checker = EpisodeTermination()
        result = checker.update(
            dt_sec=0.1,
            elapsed_sec=0.1,
            cross_track_error_m=0.0,
            linear_speed_mps=0.2,
            speed_command=4.0,
            collision=True,
            cumulative_forward_progress_m=0.0,
            track_length_m=30.0,
        )
        self.assertTrue(result.terminated)
        self.assertEqual(result.reason, "collision")

    def test_lap_completion_terminates_successfully(self):
        checker = EpisodeTermination()
        result = checker.update(
            dt_sec=0.1,
            elapsed_sec=50.0,
            cross_track_error_m=0.0,
            linear_speed_mps=0.3,
            speed_command=4.0,
            collision=False,
            cumulative_forward_progress_m=28.5,
            track_length_m=30.0,
        )
        self.assertTrue(result.terminated)
        self.assertTrue(result.lap_complete)
        self.assertEqual(result.reason, "lap_complete")


if __name__ == "__main__":
    unittest.main()
