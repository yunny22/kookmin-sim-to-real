from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from xycar_rl.gazebo_env import GazeboXycarEnv
from xycar_rl.privileged_track_expert import PrivilegedTrackExpert
from xycar_rl.sampling import signed_uniform


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Roll out the privileged track expert.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--total-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--min-speed-command", type=float, default=4.0)
    parser.add_argument("--max-speed-command", type=float, default=25.0)
    parser.add_argument("--control-rate-hz", type=float, default=7.0)
    parser.add_argument("--base-lookahead-m", type=float, default=0.28)
    parser.add_argument("--speed-lookahead-gain", type=float, default=0.45)
    parser.add_argument("--steering-gain", type=float, default=1.0)
    parser.add_argument("--curvature-preview-m", type=float, default=1.15)
    parser.add_argument("--full-speed-curvature", type=float, default=0.18)
    parser.add_argument("--minimum-speed-curvature", type=float, default=1.10)
    parser.add_argument("--curve-lookahead-reduction", type=float, default=0.0)
    parser.add_argument("--minimum-lookahead-m", type=float, default=0.24)
    parser.add_argument("--feedback-blend", type=float, default=0.35)
    parser.add_argument("--cross-track-gain", type=float, default=2.0)
    parser.add_argument("--heading-gain", type=float, default=1.4)
    parser.add_argument("--start-progress-fraction", type=float, default=0.0)
    parser.add_argument("--random-starts", action="store_true")
    parser.add_argument("--recovery-probability", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    root = args.project_root.expanduser().resolve()
    env = GazeboXycarEnv(
        world_sdf=root / "worlds" / "kookmin_xycar_track_final.sdf",
        variable_speed=True,
        require_lidar=False,
        min_speed_command=args.min_speed_command,
        max_speed_command=args.max_speed_command,
        control_rate_hz=args.control_rate_hz,
    )
    expert = PrivilegedTrackExpert(
        env.track,
        min_speed_command=args.min_speed_command,
        max_speed_command=args.max_speed_command,
        base_lookahead_m=args.base_lookahead_m,
        speed_lookahead_gain=args.speed_lookahead_gain,
        steering_gain=args.steering_gain,
        curvature_preview_m=args.curvature_preview_m,
        full_speed_curvature=args.full_speed_curvature,
        minimum_speed_curvature=args.minimum_speed_curvature,
        curve_lookahead_reduction=args.curve_lookahead_reduction,
        minimum_lookahead_m=args.minimum_lookahead_m,
        feedback_blend=args.feedback_blend,
        cross_track_gain=args.cross_track_gain,
        heading_gain=args.heading_gain,
    )
    rng = np.random.default_rng(args.seed)
    global_step = 0
    try:
        for episode in range(args.episodes):
            options = {
                "progress_fraction": (
                    float(rng.uniform(0.0, 1.0))
                    if args.random_starts
                    else float(args.start_progress_fraction % 1.0)
                ),
                "lateral_error_m": 0.0,
                "yaw_error_rad": 0.0,
            }
            if rng.random() < args.recovery_probability:
                options["lateral_error_m"] = signed_uniform(rng, 0.05, 0.16)
                options["yaw_error_rad"] = signed_uniform(
                    rng, math.radians(3), math.radians(10)
                )
            _, info = env.reset(seed=args.seed + episode, options=options)
            expert.reset()
            total_reward = 0.0
            speeds = []
            steering_deltas = []
            steering_sign_flips = 0
            previous_steering = 0.0
            max_abs_cross_track = 0.0
            max_abs_heading = 0.0
            for step in range(1, args.max_steps + 1):
                action = expert.action(info)
                steering = float(action[0])
                steering_deltas.append(abs(steering - previous_steering))
                if (
                    steering * previous_steering < 0.0
                    and max(abs(steering), abs(previous_steering)) < 0.15
                ):
                    steering_sign_flips += 1
                previous_steering = steering
                _, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                speeds.append(float(info["speed_command"]))
                max_abs_cross_track = max(
                    max_abs_cross_track,
                    abs(float(info["cross_track_error_m"])),
                )
                max_abs_heading = max(
                    max_abs_heading,
                    abs(float(info["heading_error_rad"])),
                )
                global_step += 1
                if terminated or truncated or (
                    args.total_steps and global_step >= args.total_steps
                ):
                    break
            print(
                f"episode={episode} steps={step} return={total_reward:.2f} "
                f"progress={float(info.get('cumulative_progress_m', 0.0)):.2f} "
                f"speed_mean={float(np.mean(speeds)):.2f} "
                f"speed_max={float(np.max(speeds)):.2f} "
                f"steer_delta_mean={float(np.mean(steering_deltas)):.4f} "
                f"straight_flips={steering_sign_flips} "
                f"cte_max={max_abs_cross_track:.3f} "
                f"heading_max={math.degrees(max_abs_heading):.1f}deg "
                f"reason={info.get('reason', 'limit')}",
                flush=True,
            )
            if args.total_steps and global_step >= args.total_steps:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
