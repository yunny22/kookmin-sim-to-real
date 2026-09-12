import math
from pathlib import Path
import tempfile
import unittest

import torch

from xycar_rl.camera_speed_models import (
    CompactCameraSpeedActor,
    denormalize_speed_command,
    expand_actor_speed_range,
    load_camera_speed_actor,
    normalize_speed_command,
    retarget_actor_speed_range,
    speed_head_range_transform,
)
from xycar_rl.lap_time_selection import (
    select_fastest_safe_candidate,
    summarize_candidate,
)
from xycar_rl.reward import calculate_reward, lap_time_objective_weights
from xycar_rl.track_geometry import TrackProjection
from xycar_rl.train_lap_time_online import ReplayBuffer
from xycar_rl.train_lap_time_online import parse_args as parse_online_args


def projection(cross_track_error_m=0.0):
    return TrackProjection(
        x=0.0,
        y=0.0,
        tangent_yaw=0.0,
        progress_m=1.0,
        progress_fraction=0.1,
        cross_track_error_m=float(cross_track_error_m),
        heading_error_rad=0.0,
        segment_index=0,
    )


class OnlineTrainingArgumentTest(unittest.TestCase):
    def test_speed_extension_only_argument(self):
        args = parse_online_args(
            [
                "--initial-checkpoint",
                "/tmp/actor.pth",
                "--output-dir",
                "/tmp/output",
                "--speed-extension-only",
            ]
        )
        self.assertTrue(args.speed_extension_only)


class LapTimeRewardTest(unittest.TestCase):
    def test_same_progress_prefers_less_elapsed_time(self):
        weights = lap_time_objective_weights()
        fast = calculate_reward(
            projection=projection(),
            progress_delta_m=0.1,
            steering_norm=0.0,
            previous_steering_norm=0.0,
            dt_sec=0.1,
            weights=weights,
        )
        slow = calculate_reward(
            projection=projection(),
            progress_delta_m=0.1,
            steering_norm=0.0,
            previous_steering_norm=0.0,
            dt_sec=0.2,
            weights=weights,
        )
        self.assertGreater(fast.total, slow.total)

    def test_lane_margin_penalty_starts_near_departure(self):
        weights = lap_time_objective_weights(
            lane_margin_start_m=0.24,
            lane_departure_threshold_m=0.38,
        )
        safe = calculate_reward(
            projection=projection(0.20),
            progress_delta_m=0.0,
            steering_norm=0.0,
            previous_steering_norm=0.0,
            dt_sec=0.1,
            weights=weights,
        )
        near_edge = calculate_reward(
            projection=projection(0.35),
            progress_delta_m=0.0,
            steering_norm=0.0,
            previous_steering_norm=0.0,
            dt_sec=0.1,
            weights=weights,
        )
        self.assertEqual(safe.lane_margin, 0.0)
        self.assertLess(near_edge.lane_margin, 0.0)

    def test_departure_penalty_dominates_partial_progress(self):
        weights = lap_time_objective_weights()
        failed = calculate_reward(
            projection=projection(0.40),
            progress_delta_m=0.25,
            steering_norm=0.0,
            previous_steering_norm=0.0,
            off_track=True,
            dt_sec=0.1,
            weights=weights,
        )
        completed = calculate_reward(
            projection=projection(),
            progress_delta_m=0.01,
            steering_norm=0.0,
            previous_steering_norm=0.0,
            lap_complete=True,
            dt_sec=0.1,
            weights=weights,
        )
        self.assertLess(failed.total, 0.0)
        self.assertGreater(completed.total, 0.0)


class LexicographicSelectionTest(unittest.TestCase):
    def test_faster_unsafe_candidate_cannot_win(self):
        safe = summarize_candidate(
            "safe",
            [
                {"reason": "lap_complete", "lap_time_sec": 16.0},
                {"reason": "lap_complete", "lap_time_sec": 15.8},
            ],
            required_safe_laps=2,
        )
        fast_but_unsafe = summarize_candidate(
            "unsafe",
            [
                {"reason": "lap_complete", "lap_time_sec": 12.0},
                {"reason": "off_track", "lap_time_sec": 4.0},
            ],
            required_safe_laps=2,
        )
        self.assertEqual(
            select_fastest_safe_candidate([fast_but_unsafe, safe]),
            safe,
        )

    def test_no_candidate_is_selected_without_perfect_completion(self):
        failed = summarize_candidate(
            "failed",
            [{"reason": "off_track", "lap_time_sec": 3.0}],
            required_safe_laps=1,
        )
        self.assertIsNone(select_fastest_safe_candidate([failed]))


class SpeedRangeMigrationTest(unittest.TestCase):
    def test_logit_transform_preserves_reference_command(self):
        preserve = 18.5
        scale, offset = speed_head_range_transform(
            old_min_speed_command=4.0,
            old_max_speed_command=24.0,
            new_min_speed_command=4.0,
            new_max_speed_command=100.0,
            preserve_speed_command=preserve,
        )
        old_action = normalize_speed_command(preserve, 4.0, 24.0)
        old_logit = math.atanh(old_action)
        new_action = math.tanh(scale * old_logit + offset)
        self.assertAlmostEqual(
            denormalize_speed_command(new_action, 4.0, 100.0),
            preserve,
            places=5,
        )

    def test_actor_retarget_has_no_initial_speed_jump(self):
        preserve = 18.5
        actor = CompactCameraSpeedActor(temporal_frames=2)
        actor.disable_dropout()
        output_layer = actor.head[-2]
        with torch.no_grad():
            output_layer.weight[1].zero_()
            output_layer.bias[1] = math.atanh(
                normalize_speed_command(preserve, 4.0, 24.0)
            )
        retarget_actor_speed_range(
            actor,
            old_min_speed_command=4.0,
            old_max_speed_command=24.0,
            new_min_speed_command=4.0,
            new_max_speed_command=100.0,
            preserve_speed_command=preserve,
        )
        with torch.no_grad():
            action = actor(torch.zeros(1, 6, 90, 160))[0, 1].item()
        self.assertAlmostEqual(
            denormalize_speed_command(action, 4.0, 100.0),
            preserve,
            places=4,
        )

    def test_range_expansion_preserves_every_source_command(self):
        torch.manual_seed(20260723)
        base = CompactCameraSpeedActor(temporal_frames=2)
        base.disable_dropout()
        base.eval()
        images = torch.rand(16, 6, 90, 160)
        with torch.no_grad():
            source_action = base(images)
        expanded = expand_actor_speed_range(
            base,
            source_min_speed_command=4.0,
            source_max_speed_command=24.0,
            target_min_speed_command=4.0,
            target_max_speed_command=100.0,
        )
        expanded.eval()
        with torch.no_grad():
            target_action = expanded(images)
        source_speed = 4.0 + 0.5 * (source_action[:, 1] + 1.0) * 20.0
        target_speed = 4.0 + 0.5 * (target_action[:, 1] + 1.0) * 96.0
        torch.testing.assert_close(
            target_action[:, 0],
            source_action[:, 0],
            rtol=0.0,
            atol=1.0e-6,
        )
        torch.testing.assert_close(
            target_speed,
            source_speed,
            rtol=0.0,
            atol=1.0e-4,
        )

    def test_range_expanded_checkpoint_reloads_exactly(self):
        torch.manual_seed(7)
        base = CompactCameraSpeedActor(temporal_frames=2)
        base.disable_dropout()
        expanded = expand_actor_speed_range(
            base,
            source_min_speed_command=4.0,
            source_max_speed_command=24.0,
            target_min_speed_command=4.0,
            target_max_speed_command=100.0,
        ).eval()
        image = torch.rand(3, 6, 90, 160)
        with torch.no_grad():
            expected = expanded(image)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "expanded.pth"
            torch.save(
                {
                    "model_type": "camera_speed_temporal_compact_range_expanded",
                    "temporal_frames": 2,
                    "min_speed_command": 4.0,
                    "max_speed_command": 100.0,
                    "speed_range_expansion": expanded.speed_range_expansion,
                    "actor_state_dict": expanded.state_dict(),
                },
                checkpoint,
            )
            loaded, payload = load_camera_speed_actor(checkpoint)
            with torch.no_grad():
                actual = loaded(image)
        self.assertEqual(payload["max_speed_command"], 100.0)
        torch.testing.assert_close(actual, expected)

    def test_speed_only_training_freezes_perception_and_steering(self):
        base = CompactCameraSpeedActor(temporal_frames=2)
        expanded = expand_actor_speed_range(
            base,
            source_min_speed_command=4.0,
            source_max_speed_command=24.0,
            target_min_speed_command=4.0,
            target_max_speed_command=100.0,
        )
        expanded.freeze_base_policy()
        expanded.train()
        trainable = {
            name
            for name, parameter in expanded.named_parameters()
            if parameter.requires_grad
        }
        self.assertEqual(
            trainable,
            {"speed_extension.weight", "speed_extension.bias"},
        )
        self.assertFalse(expanded.image_encoder.training)
        self.assertFalse(expanded.head.training)
        self.assertTrue(expanded.speed_extension.training)

    def test_encoder_freeze_keeps_backbone_eval_and_heads_trainable(self):
        base = CompactCameraSpeedActor(temporal_frames=2)
        expanded = expand_actor_speed_range(
            base,
            source_min_speed_command=4.0,
            source_max_speed_command=24.0,
            target_min_speed_command=4.0,
            target_max_speed_command=100.0,
        )
        expanded.freeze_encoder()
        expanded.train()
        self.assertFalse(expanded.image_encoder.training)
        self.assertTrue(expanded.head.training)
        self.assertTrue(expanded.speed_extension.training)
        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in expanded.image_encoder.parameters()
            )
        )
        self.assertTrue(
            all(parameter.requires_grad for parameter in expanded.head.parameters())
        )


class CompletionOnlyBehaviorCloningTest(unittest.TestCase):
    def test_replay_marks_only_selected_episode_references(self):
        replay = ReplayBuffer(4, (6, 8, 8), seed=1)
        image = torch.zeros(6, 8, 8).numpy()
        successful = replay.append(
            image,
            torch.zeros(2).numpy(),
            1.0,
            image,
            False,
        )
        replay.append(
            image,
            torch.ones(2).numpy(),
            -1.0,
            image,
            True,
        )
        replay.mark_bc_eligible([successful])
        self.assertEqual(float(replay.bc_weights[successful[0], 0]), 1.0)
        self.assertEqual(float(replay.bc_weights.sum()), 1.0)


if __name__ == "__main__":
    unittest.main()
