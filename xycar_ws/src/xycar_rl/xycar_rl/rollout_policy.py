from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from xycar_rl.gazebo_env import GazeboXycarEnv
from xycar_rl.camera_speed_models import normalize_speed_command
from xycar_rl.canonical_preview import CanonicalPreviewSteering
from xycar_rl.policy_loader import load_camera_speed_policy, load_steering_policy
from xycar_rl.privileged_track_expert import PrivilegedTrackExpert
from xycar_rl.sampling import signed_uniform
from xycar_rl.steering_stabilizer import (
    AdaptiveSteeringStabilizer,
    SteeringStabilizerConfig,
)
from xycar_rl.train_td3_bc import resolve_device


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Roll out one steering policy for transition recording."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--policy-kind",
        choices=[
            "bc",
            "bc_scripted",
            "td3_bc",
            "scripted",
            "residual",
            "camera_speed_bc",
            "camera_speed_td3_bc",
        ],
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--residual-base-checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=1_200)
    parser.add_argument(
        "--control-rate-hz",
        type=float,
        default=7.0,
        help="Camera policy and physical command rate; real vehicle contract is 7 Hz.",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=0,
        help="Stop cleanly after this many environment steps; zero uses episodes only.",
    )
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--speed-command", type=float, default=4.0)
    parser.add_argument("--min-speed-command", type=float, default=4.0)
    parser.add_argument("--max-speed-command", type=float, default=25.0)
    parser.add_argument("--target-right-offset-m", type=float, default=0.0)
    parser.add_argument(
        "--speed-cap-command",
        type=float,
        default=0.0,
        help="Optional deployment cap inside the trained range; zero uses the model maximum.",
    )
    parser.add_argument(
        "--safe-speed-cap-command",
        type=float,
        default=0.0,
        help="When set, mix safe capped episodes with uncapped speed exploration.",
    )
    parser.add_argument(
        "--uncapped-speed-probability",
        type=float,
        default=0.35,
        help="Fraction of episodes that explore without the safe speed cap.",
    )
    parser.add_argument(
        "--start-progress-fraction",
        type=float,
        default=None,
        help="Fix every episode start on the track; 0.0 is the competition start reference.",
    )
    parser.add_argument("--start-lateral-error-m", type=float, default=0.0)
    parser.add_argument("--start-yaw-error-deg", type=float, default=0.0)
    parser.add_argument("--action-noise", type=float, default=0.02)
    parser.add_argument("--speed-action-noise", type=float, default=0.0)
    parser.add_argument(
        "--speed-action-bias",
        type=float,
        default=0.0,
        help="Positive normalized exploration bias; this is not a speed cap.",
    )
    parser.add_argument(
        "--steering-gain",
        type=float,
        default=1.0,
        help="Multiply normalized steering before clipping and smoothing.",
    )
    parser.add_argument(
        "--steering-temporal-alpha",
        type=float,
        default=0.90,
        help="Current-frame steering weight for clear curves.",
    )
    parser.add_argument("--straight-steering-temporal-alpha", type=float, default=0.20)
    parser.add_argument("--steering-straight-threshold", type=float, default=0.08)
    parser.add_argument("--steering-curve-threshold", type=float, default=0.35)
    parser.add_argument("--straight-steering-rate-limit", type=float, default=0.05)
    parser.add_argument("--curve-steering-rate-limit", type=float, default=0.38)
    parser.add_argument("--steering-deadband", type=float, default=0.02)
    parser.add_argument("--turn-in-anticipation-gain", type=float, default=0.35)
    parser.add_argument("--turn-in-anticipation-threshold", type=float, default=0.04)
    parser.add_argument("--preview-steering-blend", type=float, default=0.35)
    parser.add_argument("--enable-preview-steering", action="store_true")
    parser.add_argument(
        "--disable-adaptive-steering",
        action="store_true",
        help="Use the legacy fixed-alpha steering filter.",
    )
    parser.add_argument(
        "--speed-temporal-alpha",
        type=float,
        default=0.35,
        help="Current-frame speed weight; 1 disables temporal smoothing.",
    )
    parser.add_argument("--recovery-probability", type=float, default=0.40)
    parser.add_argument("--recovery-min-lateral-m", type=float, default=0.08)
    parser.add_argument("--recovery-max-lateral-m", type=float, default=0.22)
    parser.add_argument("--recovery-min-yaw-deg", type=float, default=4.0)
    parser.add_argument("--recovery-max-yaw-deg", type=float, default=14.0)
    parser.add_argument("--s-curve-focus-probability", type=float, default=0.50)
    parser.add_argument(
        "--straight-segment-only",
        action="store_true",
        help=(
            "Reset into a straight and end before the guarded curve preview; "
            "intended for straight-policy data collection."
        ),
    )
    parser.add_argument("--straight-curvature-threshold", type=float, default=0.10)
    parser.add_argument("--straight-guard-distance-m", type=float, default=0.80)
    parser.add_argument(
        "--trace-privileged-expert",
        action="store_true",
        help=(
            "Publish pose-based expert labels alongside learner actions. "
            "The expert is used only to label simulation training data."
        ),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args(argv)


def straight_segment_start_fractions(
    track,
    *,
    curvature_threshold: float,
    guard_distance_m: float,
    samples: int = 1_000,
) -> list[float]:
    """Return one early start point for every guarded straight interval."""
    threshold = max(0.0, float(curvature_threshold))
    guard = max(0.0, float(guard_distance_m))
    count = max(100, int(samples))
    flags = []
    for index in range(count):
        progress_m = track.length_m * index / count
        flags.append(
            abs(track.curvature_at(progress_m, sample_distance_m=0.30))
            <= threshold
            and track.max_abs_curvature_ahead(
                progress_m,
                preview_distance_m=guard,
            )
            <= threshold
        )
    starts: list[float] = []
    interval_start = None
    for index, is_straight in enumerate(flags + [False]):
        if is_straight and interval_start is None:
            interval_start = index
        elif not is_straight and interval_start is not None:
            interval_size = index - interval_start
            if interval_size >= 2:
                start_index = interval_start + max(1, interval_size // 20)
                starts.append((start_index / count) % 1.0)
            interval_start = None
    if not starts:
        raise ValueError("track has no guarded straight segment")
    return starts


def main(argv=None) -> None:
    args = parse_args(argv)
    device = resolve_device(args.device)
    root = args.project_root.expanduser().resolve()
    variable_speed = args.policy_kind in {
        "camera_speed_bc",
        "camera_speed_td3_bc",
    }
    if variable_speed:
        policy, policy_payload = load_camera_speed_policy(
            args.checkpoint, device=device
        )
        checkpoint_min = float(policy_payload.get("min_speed_command", 4.0))
        checkpoint_max = float(policy_payload.get("max_speed_command", 25.0))
        if (
            abs(args.min_speed_command - checkpoint_min) > 1.0e-6
            or abs(args.max_speed_command - checkpoint_max) > 1.0e-6
        ):
            raise ValueError(
                "rollout speed range must match the camera-speed checkpoint: "
                f"{checkpoint_min}..{checkpoint_max}"
            )
    else:
        policy = load_steering_policy(
            args.policy_kind,
            args.checkpoint,
            device=device,
            residual_base_checkpoint=args.residual_base_checkpoint,
        )
    env = GazeboXycarEnv(
        world_sdf=root / "worlds" / "kookmin_xycar_track_final.sdf",
        speed_command=args.speed_command,
        variable_speed=variable_speed,
        min_speed_command=args.min_speed_command,
        max_speed_command=args.max_speed_command,
        control_rate_hz=args.control_rate_hz,
        target_right_offset_m=args.target_right_offset_m,
    )
    rng = np.random.default_rng(args.seed)
    steering_stabilizer = AdaptiveSteeringStabilizer(
        SteeringStabilizerConfig(
            straight_alpha=args.straight_steering_temporal_alpha,
            curve_alpha=args.steering_temporal_alpha,
            straight_threshold=args.steering_straight_threshold,
            curve_threshold=args.steering_curve_threshold,
            straight_rate_limit=args.straight_steering_rate_limit,
            curve_rate_limit=args.curve_steering_rate_limit,
            deadband=args.steering_deadband,
            turn_in_anticipation_gain=args.turn_in_anticipation_gain,
            turn_in_anticipation_threshold=args.turn_in_anticipation_threshold,
        )
    )
    preview_steering = CanonicalPreviewSteering()
    expert = (
        PrivilegedTrackExpert(
            env.track,
            min_speed_command=args.min_speed_command,
            max_speed_command=args.max_speed_command,
            action_min_speed_command=args.min_speed_command,
            action_max_speed_command=args.max_speed_command,
        )
        if args.trace_privileged_expert
        else None
    )
    straight_starts = (
        straight_segment_start_fractions(
            env.track,
            curvature_threshold=args.straight_curvature_threshold,
            guard_distance_m=args.straight_guard_distance_m,
        )
        if args.straight_segment_only
        else []
    )
    global_step = 0
    try:
        for episode in range(args.episodes):
            episode_speed_cap = args.speed_cap_command
            if variable_speed and args.safe_speed_cap_command > 0.0:
                episode_speed_cap = (
                    0.0
                    if rng.random() < args.uncapped_speed_probability
                    else args.safe_speed_cap_command
                )
            options = {}
            if args.start_progress_fraction is not None:
                options["progress_fraction"] = float(
                    args.start_progress_fraction % 1.0
                )
                options["lateral_error_m"] = float(
                    args.start_lateral_error_m
                )
                options["yaw_error_rad"] = math.radians(
                    float(args.start_yaw_error_deg)
                )
            elif args.straight_segment_only:
                options["progress_fraction"] = float(
                    straight_starts[episode % len(straight_starts)]
                )
                if rng.random() < args.recovery_probability:
                    options["lateral_error_m"] = signed_uniform(
                        rng,
                        args.recovery_min_lateral_m,
                        args.recovery_max_lateral_m,
                    )
                    options["yaw_error_rad"] = signed_uniform(
                        rng,
                        math.radians(args.recovery_min_yaw_deg),
                        math.radians(args.recovery_max_yaw_deg),
                    )
                else:
                    options["lateral_error_m"] = 0.0
                    options["yaw_error_rad"] = 0.0
            else:
                if rng.random() < args.s_curve_focus_probability:
                    options["progress_fraction"] = float(
                        rng.uniform(0.17, 0.48)
                    )
                if rng.random() < args.recovery_probability:
                    options["lateral_error_m"] = signed_uniform(
                        rng,
                        args.recovery_min_lateral_m,
                        args.recovery_max_lateral_m,
                    )
                    options["yaw_error_rad"] = signed_uniform(
                        rng,
                        math.radians(args.recovery_min_yaw_deg),
                        math.radians(args.recovery_max_yaw_deg),
                    )
            observation, info = env.reset(
                seed=args.seed + episode,
                options=options or None,
            )
            if variable_speed:
                policy.reset()
                steering_stabilizer.reset()
                preview_steering.reset()
            if expert is not None:
                expert.reset()
            total_reward = 0.0
            speed_commands = []
            cross_track_errors = []
            steering_deltas = []
            straight_steering_flips = 0
            large_oscillation_events = 0
            large_oscillation_steps = 0
            large_oscillation_active = False
            previous_steering = 0.0
            metric_previous_steering = 0.0
            previous_speed = -1.0
            for step in range(1, args.max_steps + 1):
                action = policy(observation)
                if variable_speed:
                    action = np.asarray(action, dtype=np.float32).reshape(2)
                    if args.action_noise > 0.0:
                        action[0] += float(rng.normal(0.0, args.action_noise))
                    if args.speed_action_noise > 0.0:
                        action[1] += float(
                            rng.normal(0.0, args.speed_action_noise)
                        )
                    action[1] += float(args.speed_action_bias)
                    curve_hint = 0.0
                    if args.enable_preview_steering:
                        preview = preview_steering.update(observation["image"])
                        preview_blend = float(
                            np.clip(args.preview_steering_blend, 0.0, 1.0)
                        ) * preview.confidence
                        action[0] = (
                            (1.0 - preview_blend) * action[0]
                            + preview_blend * preview.steering_norm
                        )
                        curve_hint = preview.curve_hint * preview.confidence
                    action[0] *= float(args.steering_gain)
                    if episode_speed_cap > 0.0:
                        speed_cap_norm = normalize_speed_command(
                            episode_speed_cap,
                            args.min_speed_command,
                            args.max_speed_command,
                        )
                        action[1] = min(action[1], speed_cap_norm)
                    speed_alpha = float(
                        np.clip(args.speed_temporal_alpha, 0.0, 1.0)
                    )
                    if args.disable_adaptive_steering:
                        steering_alpha = float(
                            np.clip(args.steering_temporal_alpha, 0.0, 1.0)
                        )
                        action[0] = (
                            steering_alpha * action[0]
                            + (1.0 - steering_alpha) * previous_steering
                        )
                    else:
                        action[0] = steering_stabilizer.update(
                            float(action[0]), curve_hint=curve_hint
                        )
                    action[1] = (
                        speed_alpha * action[1]
                        + (1.0 - speed_alpha) * previous_speed
                    )
                    action = np.clip(action, -1.0, 1.0).astype(np.float32)
                    previous_steering = float(action[0])
                    previous_speed = float(action[1])
                else:
                    if args.action_noise > 0.0:
                        action += float(rng.normal(0.0, args.action_noise))
                    action = np.asarray(
                        [float(np.clip(action, -1.0, 1.0))],
                        dtype=np.float32,
                    )
                applied_steering = float(action[0])
                steering_deltas.append(
                    abs(applied_steering - metric_previous_steering)
                )
                if (
                    applied_steering * metric_previous_steering < 0.0
                    and max(
                        abs(applied_steering), abs(metric_previous_steering)
                    )
                    < 0.15
                ):
                    straight_steering_flips += 1
                metric_previous_steering = applied_steering
                expert_action = None if expert is None else expert.action(info)
                observation, reward, terminated, truncated, info = env.step(
                    action,
                    trace_action=expert_action,
                )
                if args.straight_segment_only and not (
                    abs(
                        env.track.curvature_at(
                            float(info.get("progress_m", 0.0)),
                            sample_distance_m=0.30,
                        )
                    )
                    <= args.straight_curvature_threshold
                    and env.track.max_abs_curvature_ahead(
                        float(info.get("progress_m", 0.0)),
                        preview_distance_m=args.straight_guard_distance_m,
                    )
                    <= args.straight_curvature_threshold
                ):
                    info = dict(info)
                    info["reason"] = "straight_segment_complete"
                    truncated = True
                oscillation_term = float(
                    info.get("reward_terms", {}).get("large_oscillation", 0.0)
                )
                oscillation_active = oscillation_term < -1.0e-9
                if oscillation_active:
                    large_oscillation_steps += 1
                    if not large_oscillation_active:
                        large_oscillation_events += 1
                large_oscillation_active = oscillation_active
                total_reward += reward
                speed_commands.append(float(info.get("speed_command", 0.0)))
                cross_track_errors.append(
                    abs(float(info.get("cross_track_error_m", 0.0)))
                )
                global_step += 1
                if (
                    terminated
                    or truncated
                    or (args.total_steps and global_step >= args.total_steps)
                ):
                    break
            print(
                f"episode={episode} seed={args.seed + episode} steps={step} "
                f"return={total_reward:.2f} "
                f"progress={float(info.get('cumulative_progress_m', 0.0)):.2f} "
                f"speed_mean={float(np.mean(speed_commands)) if speed_commands else 0.0:.2f} "
                f"speed_max={float(np.max(speed_commands)) if speed_commands else 0.0:.2f} "
                f"steer_delta_mean={float(np.mean(steering_deltas)) if steering_deltas else 0.0:.4f} "
                f"small_straight_flips={straight_steering_flips} "
                f"large_osc_events={large_oscillation_events} "
                f"large_osc_steps={large_oscillation_steps} "
                f"cte_max={float(np.max(cross_track_errors)) if cross_track_errors else 0.0:.3f} "
                f"cte_final={float(info.get('cross_track_error_m', 0.0)):.3f} "
                f"heading_final={math.degrees(float(info.get('heading_error_rad', 0.0))):.1f}deg "
                f"speed_cap={episode_speed_cap:.2f} "
                f"reason={info.get('reason', 'evaluation_limit')}",
                flush=True,
            )
            if args.total_steps and global_step >= args.total_steps:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
