from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from xycar_rl.gazebo_env import GazeboXycarEnv
from xycar_rl.policy_loader import load_camera_speed_policy
from xycar_rl.privileged_track_expert import PrivilegedTrackExpert
from xycar_rl.sampling import signed_uniform
from xycar_rl.steering_stabilizer import AdaptiveSteeringStabilizer
from xycar_rl.train_td3_bc import resolve_device


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Collect DAgger states using learner/expert blended control."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--total-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--learner-blend", type=float, default=0.35)
    parser.add_argument("--policy-min-speed-command", type=float, default=4.0)
    parser.add_argument("--expert-min-speed-command", type=float, default=7.0)
    parser.add_argument("--max-speed-command", type=float, default=12.0)
    parser.add_argument("--minimum-speed-curvature", type=float, default=0.9)
    parser.add_argument("--curvature-preview-m", type=float, default=2.0)
    parser.add_argument("--random-starts", action="store_true")
    parser.add_argument(
        "--start-progress-fraction",
        action="append",
        type=float,
        default=[],
        help="Repeatable targeted start fraction; values cycle across episodes.",
    )
    parser.add_argument("--start-progress-jitter", type=float, default=0.015)
    parser.add_argument("--recovery-probability", type=float, default=0.5)
    parser.add_argument(
        "--disable-applied-steering-stabilizer",
        action="store_true",
        help=(
            "Do not apply the runtime steering stabilizer to the learner "
            "before blending it with the already-stabilized expert action."
        ),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    device = resolve_device(args.device)
    root = args.project_root.expanduser().resolve()
    policy, payload = load_camera_speed_policy(args.checkpoint, device=device)
    checkpoint_range = (
        float(payload.get("min_speed_command", args.policy_min_speed_command)),
        float(payload.get("max_speed_command", args.max_speed_command)),
    )
    expected_range = (args.policy_min_speed_command, args.max_speed_command)
    if checkpoint_range != expected_range:
        raise ValueError(
            f"checkpoint speed range {checkpoint_range} != {expected_range}"
        )
    env = GazeboXycarEnv(
        world_sdf=root / "worlds" / "kookmin_xycar_track_final.sdf",
        variable_speed=True,
        require_lidar=False,
        min_speed_command=args.policy_min_speed_command,
        max_speed_command=args.max_speed_command,
    )
    expert = PrivilegedTrackExpert(
        env.track,
        min_speed_command=args.expert_min_speed_command,
        max_speed_command=args.max_speed_command,
        action_min_speed_command=args.policy_min_speed_command,
        action_max_speed_command=args.max_speed_command,
        minimum_speed_curvature=args.minimum_speed_curvature,
        curvature_preview_m=args.curvature_preview_m,
    )
    learner_blend = float(np.clip(args.learner_blend, 0.0, 1.0))
    rng = np.random.default_rng(args.seed)
    steering_stabilizer = AdaptiveSteeringStabilizer()
    global_step = 0
    try:
        for episode in range(args.episodes):
            if args.start_progress_fraction:
                base_progress = args.start_progress_fraction[
                    episode % len(args.start_progress_fraction)
                ]
                progress_fraction = float(
                    (
                        base_progress
                        + rng.uniform(
                            -abs(args.start_progress_jitter),
                            abs(args.start_progress_jitter),
                        )
                    )
                    % 1.0
                )
            elif args.random_starts:
                progress_fraction = float(rng.uniform(0.0, 1.0))
            else:
                progress_fraction = 0.0
            options = {
                "progress_fraction": progress_fraction,
                "lateral_error_m": 0.0,
                "yaw_error_rad": 0.0,
            }
            if rng.random() < args.recovery_probability:
                options["lateral_error_m"] = signed_uniform(rng, 0.04, 0.14)
                options["yaw_error_rad"] = signed_uniform(
                    rng, math.radians(3.0), math.radians(10.0)
                )
            observation, info = env.reset(
                seed=args.seed + episode, options=options
            )
            policy.reset()
            expert.reset()
            steering_stabilizer.reset()
            total_reward = 0.0
            speeds = []
            steering_disagreements = []
            for step in range(1, args.max_steps + 1):
                learner_action = np.asarray(
                    policy(observation), dtype=np.float32
                ).copy()
                if not args.disable_applied_steering_stabilizer:
                    learner_action[0] = steering_stabilizer.update(
                        float(learner_action[0])
                    )
                expert_action = expert.action(info)
                applied_action = np.clip(
                    learner_blend * learner_action
                    + (1.0 - learner_blend) * expert_action,
                    -1.0,
                    1.0,
                ).astype(np.float32)
                steering_disagreements.append(
                    abs(float(learner_action[0] - expert_action[0]))
                )
                observation, reward, terminated, truncated, info = env.step(
                    applied_action,
                    trace_action=expert_action,
                )
                total_reward += reward
                speeds.append(float(info["speed_command"]))
                global_step += 1
                if terminated or truncated or (
                    args.total_steps and global_step >= args.total_steps
                ):
                    break
            print(
                f"episode={episode} steps={step} return={total_reward:.2f} "
                f"progress={float(info.get('cumulative_progress_m', 0.0)):.2f} "
                f"applied_speed_mean={float(np.mean(speeds)):.2f} "
                f"steer_disagreement_mean="
                f"{float(np.mean(steering_disagreements)):.3f} "
                f"reason={info.get('reason', 'limit')}",
                flush=True,
            )
            if args.total_steps and global_step >= args.total_steps:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
