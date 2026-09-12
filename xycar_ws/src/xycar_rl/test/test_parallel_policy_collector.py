import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from xycar_rl.parallel_policy_collector import (
    apply_rollout_terminals,
    _episode_allocation,
    parse_args,
    _session_summary,
)


class ParallelPolicyCollectorTest(unittest.TestCase):
    def test_default_worker_count_matches_long_run_capacity(self):
        args = parse_args(["--checkpoint", "/tmp/model.pth"])
        self.assertEqual(args.workers, 10)
        self.assertEqual(args.target_right_offset_m, 0.0)
        self.assertEqual(args.speed_cap_command, 25.0)
        self.assertFalse(args.raw_steering_actions)
        self.assertFalse(args.straight_segment_only)
        self.assertFalse(args.trace_privileged_expert)
        self.assertEqual(args.recovery_min_lateral_m, 0.08)
        self.assertEqual(args.recovery_max_yaw_deg, 14.0)

    def test_episode_allocation_uses_every_requested_episode(self):
        self.assertEqual(_episode_allocation(10, 4), [3, 3, 2, 2])
        self.assertEqual(_episode_allocation(2, 4), [1, 1])

    def test_session_summary_keeps_complete_and_failed_episodes(self):
        fields = [
            "episode_id",
            "state_timestamp_ns",
            "termination_reason",
        ]
        rows = [
            {
                "episode_id": "0",
                "state_timestamp_ns": "1000000000",
                "termination_reason": "",
            },
            {
                "episode_id": "0",
                "state_timestamp_ns": "1200000000",
                "termination_reason": "lap_complete",
            },
            {
                "episode_id": "1",
                "state_timestamp_ns": "100000000",
                "termination_reason": "",
            },
            {
                "episode_id": "1",
                "state_timestamp_ns": "300000000",
                "termination_reason": "off_track",
            },
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "transitions.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            count, episodes, completed, failed, rate = _session_summary(path)
        self.assertEqual(count, 4)
        self.assertEqual(episodes, 2)
        self.assertEqual(completed, 1)
        self.assertEqual(failed, 1)
        self.assertAlmostEqual(rate, 10.0)

    def test_rollout_result_marks_each_recorded_episode_terminal(self):
        fields = [
            "episode_id",
            "state_timestamp_ns",
            "terminated",
            "truncated",
            "termination_reason",
        ]
        rows = [
            {
                "episode_id": "1",
                "state_timestamp_ns": "100",
                "terminated": "0",
                "truncated": "0",
                "termination_reason": "running",
            },
            {
                "episode_id": "2",
                "state_timestamp_ns": "200",
                "terminated": "0",
                "truncated": "0",
                "termination_reason": "running",
            },
        ]
        with TemporaryDirectory() as directory:
            csv_path = Path(directory) / "transitions.csv"
            log_path = Path(directory) / "rollout.log"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            log_path.write_text(
                "episode=0 steps=100 reason=lap_complete\n"
                "episode=1 steps=20 reason=off_track\n",
                encoding="utf-8",
            )
            apply_rollout_terminals(csv_path, log_path)
            with csv_path.open(newline="", encoding="utf-8") as handle:
                actual = list(csv.DictReader(handle))
        self.assertEqual(actual[0]["termination_reason"], "lap_complete")
        self.assertEqual(actual[0]["terminated"], "1")
        self.assertEqual(actual[1]["termination_reason"], "off_track")
        self.assertEqual(actual[1]["terminated"], "1")

    def test_straight_segment_terminal_is_a_truncation(self):
        fields = [
            "episode_id",
            "state_timestamp_ns",
            "terminated",
            "truncated",
            "termination_reason",
        ]
        with TemporaryDirectory() as directory:
            csv_path = Path(directory) / "transitions.csv"
            log_path = Path(directory) / "rollout.log"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "episode_id": "1",
                        "state_timestamp_ns": "100",
                        "terminated": "0",
                        "truncated": "0",
                        "termination_reason": "running",
                    }
                )
            log_path.write_text(
                "episode=0 steps=12 reason=straight_segment_complete\n",
                encoding="utf-8",
            )
            apply_rollout_terminals(csv_path, log_path)
            with csv_path.open(newline="", encoding="utf-8") as handle:
                actual = next(csv.DictReader(handle))
        self.assertEqual(actual["termination_reason"], "straight_segment_complete")
        self.assertEqual(actual["terminated"], "0")
        self.assertEqual(actual["truncated"], "1")


if __name__ == "__main__":
    unittest.main()
