import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from xycar_rl.camera_speed_models import (
    CameraSpeedActor,
    CompactCameraSpeedActor,
    denormalize_speed_command,
    initialize_temporal_actor,
    normalize_speed_command,
)
from xycar_rl.td3_bc import (
    CameraSpeedTD3BCAgent,
    TD3BCConfig,
    camera_speed_bc_loss,
)
from xycar_rl.train_camera_speed_bc import (
    speed_target_from_transition,
    steering_sample_weight,
    transition_rows_are_contiguous,
)
from xycar_rl.train_camera_speed_td3_bc import (
    parse_args as parse_td3_args,
    planned_simulation_cap,
    restore_agent_checkpoint,
    save_checkpoint,
    transition_root,
    transition_roots,
)
from xycar_rl.policy_runtime_node import apply_optional_speed_cap
from xycar_rl.rollout_policy import straight_segment_start_fractions
from xycar_rl.track_geometry import TrackReference
from xycar_rl.transition_dataset import (
    CameraSpeedTransitionDataset,
    camera_speed_action_targets,
    successful_bc_episode_keys,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
WORLD = PROJECT_ROOT / "worlds" / "kookmin_xycar_track_final.sdf"


class CameraSpeedContractTest(unittest.TestCase):
    def test_straight_training_arguments_are_explicit(self):
        args = parse_td3_args(
            [
                "--transitions",
                "/tmp/transitions",
                "--initial-checkpoint",
                "/tmp/model.pth",
                "--output-dir",
                "/tmp/output",
                "--straight-only",
                "--expand-speed-range",
                "--reward-objective",
                "straight_high_speed",
                "--max-speed-command",
                "25",
                "--straight-bc-target-speed-command",
                "25",
            ]
        )
        self.assertTrue(args.straight_only)
        self.assertTrue(args.expand_speed_range)
        self.assertEqual(args.reward_objective, "straight_high_speed")
        self.assertEqual(args.max_speed_command, 25.0)
        self.assertEqual(args.straight_bc_target_speed_command, 25.0)

    def test_milestones_do_not_add_a_speed_cap(self):
        self.assertEqual(planned_simulation_cap(5, 20), 0.0)
        self.assertEqual(planned_simulation_cap(20, 20), 0.0)

    def test_runtime_zero_speed_cap_preserves_learned_speed(self):
        self.assertEqual(apply_optional_speed_cap(19.5, 0.0), 19.5)
        self.assertEqual(apply_optional_speed_cap(19.5, -1.0), 19.5)
        self.assertEqual(apply_optional_speed_cap(19.5, 17.0), 17.0)

    def test_focus_transition_root_accepts_directory_or_csv(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "transitions.csv"
            csv_path.write_text("episode_id\n", encoding="utf-8")
            self.assertEqual(transition_root(root), root.resolve())
            self.assertEqual(transition_root(csv_path), root.resolve())

    def test_focus_transition_roots_expand_parallel_worker_tree(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            expected = set()
            for index in range(3):
                worker = root / f"worker_{index:02d}" / "transitions"
                worker.mkdir(parents=True)
                (worker / "transitions.csv").write_text(
                    "episode_id\n",
                    encoding="utf-8",
                )
                expected.add(worker.resolve())
            self.assertEqual(transition_roots([root]), expected)

    def test_steering_sample_weight_emphasizes_curves(self):
        self.assertEqual(steering_sample_weight(0.0, 4.0), 1.0)
        self.assertAlmostEqual(steering_sample_weight(0.5, 4.0), 2.0)
        self.assertAlmostEqual(steering_sample_weight(-1.0, 4.0), 5.0)

    def test_steering_sample_weight_can_emphasize_straights(self):
        self.assertEqual(steering_sample_weight(0.0, 4.0, 3.0, 0.12), 3.0)
        self.assertAlmostEqual(
            steering_sample_weight(0.5, 4.0, 3.0, 0.12),
            2.0,
        )

    def test_temporal_rows_require_exact_transition_continuity(self):
        previous = {
            "episode_id": "3",
            "step_id": "10",
            "state_timestamp_ns": "100",
            "next_timestamp_ns": "200",
        }
        current = {
            "episode_id": "3",
            "step_id": "11",
            "state_timestamp_ns": "200",
            "next_timestamp_ns": "300",
        }
        self.assertTrue(transition_rows_are_contiguous(previous, current))
        current["state_timestamp_ns"] = "201"
        self.assertFalse(transition_rows_are_contiguous(previous, current))

    def test_actor_is_camera_only_and_returns_two_actions(self):
        actor = CameraSpeedActor().eval()
        with torch.no_grad():
            output = actor(torch.rand(2, 3, 90, 160))
        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertLessEqual(float(output.abs().max()), 1.0)

    def test_temporal_actor_preserves_single_frame_initial_behavior(self):
        actor = CameraSpeedActor().eval()
        temporal = initialize_temporal_actor(actor).eval()
        current = torch.rand(2, 3, 90, 160)
        previous = torch.rand(2, 3, 90, 160)
        with torch.no_grad():
            expected = actor(current)
            actual = temporal(torch.cat([previous, current], dim=1))
        self.assertEqual(temporal.temporal_frames, 2)
        self.assertTrue(torch.allclose(actual, expected, atol=1.0e-6))

    def test_compact_temporal_actor_returns_two_actions(self):
        actor = CompactCameraSpeedActor(temporal_frames=2).eval()
        with torch.no_grad():
            output = actor(torch.rand(2, 6, 90, 160))
        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertLessEqual(float(output.abs().max()), 1.0)

    def test_speed_normalization_round_trip(self):
        for command in (4.0, 7.0, 9.0, 10.0):
            normalized = normalize_speed_command(command, 4.0, 10.0)
            self.assertAlmostEqual(
                denormalize_speed_command(normalized, 4.0, 10.0),
                command,
            )

    def test_dagger_expert_action_is_separate_from_applied_action(self):
        applied, bc_target = camera_speed_action_targets(
            {
                "action_norm": "0.2",
                "speed_command": "8.0",
                "expert_action_norm": "0.7",
                "expert_speed_command": "7.0",
            },
            4.0,
            12.0,
        )
        self.assertAlmostEqual(applied[0], 0.2)
        self.assertAlmostEqual(bc_target[0], 0.7)
        self.assertNotEqual(applied[1], bc_target[1])

    def test_legacy_transition_uses_applied_action_as_bc_target(self):
        applied, bc_target = camera_speed_action_targets(
            {"action_norm": "-0.3", "speed_command": "7.0"},
            4.0,
            12.0,
        )
        self.assertEqual(applied, bc_target)

    def test_straight_bc_can_target_full_speed_without_changing_applied_action(self):
        applied, bc_target = camera_speed_action_targets(
            {"action_norm": "0.1", "speed_command": "21.0"},
            4.0,
            25.0,
            bc_speed_target_command=25.0,
        )
        self.assertLess(applied[1], 1.0)
        self.assertEqual(bc_target, [0.1, 1.0])

    def test_straight_target_uses_maximum_speed(self):
        self.assertAlmostEqual(
            speed_target_from_transition(0.0, 0.0, 0.0),
            25.0,
        )

    def test_straight_filter_rejects_curve_and_curve_preview(self):
        track = TrackReference.from_sdf(WORLD, target_right_offset_m=0.0)
        samples = np.linspace(0.0, track.length_m, 1_000, endpoint=False)
        straight_progress = min(
            samples,
            key=lambda progress: (
                abs(track.curvature_at(progress, sample_distance_m=0.30))
                + track.max_abs_curvature_ahead(
                    progress,
                    preview_distance_m=0.80,
                )
            ),
        )
        curve_progress = max(
            samples,
            key=lambda progress: abs(
                track.curvature_at(progress, sample_distance_m=0.30)
            ),
        )
        dataset = CameraSpeedTransitionDataset.__new__(
            CameraSpeedTransitionDataset
        )
        dataset.straight_curvature_threshold = 0.10
        dataset.straight_guard_distance_m = 0.80
        self.assertTrue(
            dataset._is_straight_transition(
                {
                    "progress_m": str(straight_progress),
                    "progress_delta_m": "0.01",
                },
                track,
            )
        )
        self.assertFalse(
            dataset._is_straight_transition(
                {
                    "progress_m": str(curve_progress),
                    "progress_delta_m": "0.01",
                },
                track,
            )
        )

    def test_straight_rollout_starts_cover_guarded_intervals(self):
        track = TrackReference.from_sdf(WORLD, target_right_offset_m=0.0)
        starts = straight_segment_start_fractions(
            track,
            curvature_threshold=0.10,
            guard_distance_m=0.80,
        )
        self.assertEqual(len(starts), 3)
        for fraction in starts:
            progress_m = fraction * track.length_m
            self.assertLessEqual(
                abs(
                    track.curvature_at(
                        progress_m,
                        sample_distance_m=0.30,
                    )
                ),
                0.10,
            )
            self.assertLessEqual(
                track.max_abs_curvature_ahead(
                    progress_m,
                    preview_distance_m=0.80,
                ),
                0.10,
            )

    def test_straight_filter_keeps_pre_filter_lap_success_for_bc(self):
        track = TrackReference.from_sdf(WORLD, target_right_offset_m=0.0)
        samples = np.linspace(0.0, track.length_m, 1_000, endpoint=False)
        straight_progress = min(
            samples,
            key=lambda progress: (
                abs(track.curvature_at(progress, sample_distance_m=0.30))
                + track.max_abs_curvature_ahead(
                    progress,
                    preview_distance_m=0.80,
                )
            ),
        )
        curve_progress = max(
            samples,
            key=lambda progress: abs(
                track.curvature_at(progress, sample_distance_m=0.30)
            ),
        )
        fields = [
            "episode_id",
            "step_id",
            "state_timestamp_ns",
            "next_timestamp_ns",
            "state_image_path",
            "next_image_path",
            "action_norm",
            "speed_command",
            "reward",
            "terminated",
            "truncated",
            "termination_reason",
            "progress_m",
            "progress_delta_m",
        ]
        rows = [
            {
                "episode_id": "7",
                "step_id": "0",
                "state_timestamp_ns": "100",
                "next_timestamp_ns": "200",
                "state_image_path": "state.png",
                "next_image_path": "next.png",
                "action_norm": "0.0",
                "speed_command": "25.0",
                "reward": "0.0",
                "terminated": "0",
                "truncated": "0",
                "termination_reason": "running",
                "progress_m": str(straight_progress),
                "progress_delta_m": "0.01",
            },
            {
                "episode_id": "7",
                "step_id": "1",
                "state_timestamp_ns": "200",
                "next_timestamp_ns": "300",
                "state_image_path": "next.png",
                "next_image_path": "terminal.png",
                "action_norm": "0.0",
                "speed_command": "25.0",
                "reward": "1.0",
                "terminated": "1",
                "truncated": "0",
                "termination_reason": "lap_complete",
                "progress_m": str(curve_progress),
                "progress_delta_m": "0.01",
            },
        ]
        with TemporaryDirectory() as directory:
            csv_path = Path(directory) / "transitions.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            dataset = CameraSpeedTransitionDataset(
                [csv_path],
                world_sdf=WORLD,
                straight_only=True,
                bc_successful_episodes_only=True,
            )
        self.assertEqual(len(dataset), 1)
        self.assertIn((csv_path.parent, "7"), dataset.successful_episode_keys)

    def test_straight_segment_completion_is_successful_for_bc(self):
        rows = [
            (
                Path("/tmp/segment"),
                {
                    "episode_id": "3",
                    "termination_reason": "straight_segment_complete",
                },
            )
        ]
        successful = successful_bc_episode_keys(rows)
        self.assertEqual(successful, {(Path("/tmp/segment"), "3")})

    def test_sharp_or_recovery_target_slows_down(self):
        sharp = speed_target_from_transition(1.0, 0.0, 0.0)
        recovery = speed_target_from_transition(0.0, 0.30, 0.0)
        self.assertAlmostEqual(sharp, 4.0)
        self.assertAlmostEqual(recovery, 4.0)

    def test_camera_speed_td3_update_is_finite(self):
        agent = CameraSpeedTD3BCAgent(
            CameraSpeedActor(), config=TD3BCConfig(policy_delay=1)
        )
        batch = {
            "image": torch.rand(2, 3, 32, 32),
            "action": torch.zeros(2, 2),
            "reward": torch.ones(2, 1) * 0.1,
            "next_image": torch.rand(2, 3, 32, 32),
            "done": torch.zeros(2, 1),
        }
        metrics = agent.update(batch)
        for value in metrics.values():
            self.assertTrue(torch.isfinite(torch.tensor(value)))

    def test_camera_speed_bc_loss_can_emphasize_steering(self):
        prediction = torch.tensor([[1.0, 1.0]])
        target = torch.tensor([[0.0, 0.5]])
        balanced = camera_speed_bc_loss(prediction, target)
        steering_focused = camera_speed_bc_loss(
            prediction,
            target,
            steering_weight=3.0,
            speed_weight=1.0,
        )
        self.assertGreater(steering_focused, balanced)

    def test_failed_episode_can_be_excluded_from_bc_loss(self):
        prediction = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        target = torch.zeros_like(prediction)
        weighted = camera_speed_bc_loss(
            prediction,
            target,
            sample_weight=torch.tensor([[1.0], [0.0]]),
        )
        all_failed = camera_speed_bc_loss(
            prediction,
            target,
            sample_weight=torch.zeros(2, 1),
        )
        self.assertEqual(float(weighted), 0.0)
        self.assertEqual(float(all_failed), 0.0)

    def test_temporal_compact_camera_speed_td3_update_is_finite(self):
        agent = CameraSpeedTD3BCAgent(
            CompactCameraSpeedActor(temporal_frames=2),
            config=TD3BCConfig(policy_delay=1),
        )
        batch = {
            "image": torch.rand(2, 6, 32, 32),
            "action": torch.zeros(2, 2),
            "reward": torch.ones(2, 1) * 0.1,
            "next_image": torch.rand(2, 6, 32, 32),
            "done": torch.zeros(2, 1),
        }
        metrics = agent.update(batch)
        self.assertEqual(agent.temporal_frames, 2)
        for value in metrics.values():
            self.assertTrue(torch.isfinite(torch.tensor(value)))

    def test_full_td3_checkpoint_resume_restores_critics_and_learning_rates(self):
        source = CameraSpeedTD3BCAgent(
            CompactCameraSpeedActor(temporal_frames=2),
            config=TD3BCConfig(actor_lr=1.0e-5, critic_lr=3.0e-4),
        )
        source.update_count = 37
        with TemporaryDirectory() as directory:
            path = Path(directory) / "full.pth"
            save_checkpoint(
                path,
                source,
                epoch=15,
                train_config={
                    "min_speed_command": 4.0,
                    "max_speed_command": 12.0,
                },
                metrics={},
                model_type="camera_speed_temporal_compact",
            )
            resumed = CameraSpeedTD3BCAgent(
                CompactCameraSpeedActor(temporal_frames=2),
                config=TD3BCConfig(actor_lr=2.0e-6, critic_lr=1.0e-4),
            )
            payload = restore_agent_checkpoint(resumed, path)
        self.assertEqual(payload["epoch"], 15)
        self.assertEqual(resumed.update_count, 37)
        self.assertEqual(resumed.actor_optimizer.param_groups[0]["lr"], 2.0e-6)
        self.assertEqual(resumed.critic_optimizer.param_groups[0]["lr"], 1.0e-4)
        for expected, actual in zip(
            source.critic.parameters(),
            resumed.critic.parameters(),
        ):
            self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
