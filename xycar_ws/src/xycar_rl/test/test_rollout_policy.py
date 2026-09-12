import unittest

from xycar_rl.rollout_policy import parse_args
from xycar_rl.rollout_dagger import parse_args as parse_dagger_args


class RolloutPolicyArgumentsTest(unittest.TestCase):
    def test_fixed_start_arguments_accept_competition_reference(self):
        args = parse_args(
            [
                "--policy-kind",
                "td3_bc",
                "--checkpoint",
                "model.pth",
                "--start-progress-fraction",
                "0.0",
                "--start-lateral-error-m",
                "0.0",
                "--start-yaw-error-deg",
                "0.0",
            ]
        )
        self.assertEqual(args.start_progress_fraction, 0.0)
        self.assertEqual(args.start_lateral_error_m, 0.0)
        self.assertEqual(args.start_yaw_error_deg, 0.0)

    def test_privileged_expert_trace_is_explicitly_opt_in(self):
        default_args = parse_args(
            ["--policy-kind", "td3_bc", "--checkpoint", "model.pth"]
        )
        traced_args = parse_args(
            [
                "--policy-kind",
                "td3_bc",
                "--checkpoint",
                "model.pth",
                "--trace-privileged-expert",
            ]
        )
        self.assertFalse(default_args.trace_privileged_expert)
        self.assertTrue(traced_args.trace_privileged_expert)

    def test_dagger_accepts_multiple_targeted_start_fractions(self):
        args = parse_dagger_args(
            [
                "--checkpoint",
                "model.pth",
                "--start-progress-fraction",
                "0.62",
                "--start-progress-fraction",
                "0.74",
            ]
        )
        self.assertEqual(args.start_progress_fraction, [0.62, 0.74])

    def test_dagger_can_disable_second_steering_stabilizer(self):
        args = parse_dagger_args(
            [
                "--checkpoint",
                "model.pth",
                "--disable-applied-steering-stabilizer",
            ]
        )
        self.assertTrue(args.disable_applied_steering_stabilizer)


if __name__ == "__main__":
    unittest.main()
