from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from xycar_rl.gazebo_env import GazeboXycarEnv
from xycar_rl.models import load_bc_actor, load_rl_actor
from xycar_rl.policy_loader import load_steering_policy
from xycar_rl.replay_buffer import ReplayBuffer
from xycar_rl.residual_td3 import ResidualTD3Agent, ResidualTD3Config
from xycar_rl.sampling import signed_uniform
from xycar_rl.train_td3_bc import resolve_device


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a bounded residual TD3 policy online in Gazebo."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--base-kind",
        choices=["bc", "bc_scripted", "td3_bc", "scripted"],
        default="td3_bc",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--replay-capacity", type=int, default=10_000)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--exploration-noise", type=float, default=0.08)
    parser.add_argument("--max-residual-norm", type=float, default=0.25)
    parser.add_argument("--speed-command", type=float, default=4.0)
    parser.add_argument("--focus-probability", type=float, default=0.70)
    parser.add_argument("--focus-start", type=float, default=0.17)
    parser.add_argument("--focus-end", type=float, default=0.48)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--save-every-steps", type=int, default=5_000)
    return parser.parse_args(argv)


def save_checkpoint(path: Path, agent: ResidualTD3Agent, args, step: int) -> None:
    torch.save(
        {
            "algorithm": "residual_td3",
            "step": int(step),
            "base_checkpoint": str(args.base_checkpoint.expanduser().resolve()),
            "base_kind": args.base_kind,
            "max_residual_norm": args.max_residual_norm,
            "action_max_steer_command": 42.0,
            "config": asdict(agent.config),
            "residual_state_dict": agent.residual_actor.state_dict(),
            "residual_target_state_dict": agent.residual_target.state_dict(),
            "critic_state_dict": agent.critic.state_dict(),
            "critic_target_state_dict": agent.critic_target.state_dict(),
            "actor_optimizer_state_dict": agent.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": agent.critic_optimizer.state_dict(),
        },
        path,
    )


def main(argv=None) -> None:
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    project_root = args.project_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.base_kind == "bc":
        base_actor, _ = load_bc_actor(args.base_checkpoint, device=device)
    elif args.base_kind == "td3_bc":
        base_actor, _ = load_rl_actor(args.base_checkpoint, device=device)
    else:
        base_actor = load_steering_policy(
            args.base_kind, args.base_checkpoint, device=device
        ).actor
    agent = ResidualTD3Agent(
        base_actor,
        max_residual_norm=args.max_residual_norm,
        device=device,
    )
    replay = ReplayBuffer(args.replay_capacity)
    env = GazeboXycarEnv(
        world_sdf=project_root / "worlds" / "kookmin_xycar_track_final.sdf",
        speed_command=args.speed_command,
    )
    metrics_path = output_dir / "episodes.csv"
    fields = [
        "episode",
        "global_step",
        "seed",
        "return",
        "steps",
        "progress_m",
        "mean_abs_cte_m",
        "max_abs_cte_m",
        "reason",
        "mean_abs_residual_norm",
    ]
    rng = np.random.default_rng(args.seed)
    global_step = 0
    episode = 0
    with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=fields)
        writer.writeheader()
        try:
            while global_step < args.total_steps:
                episode_seed = args.seed + episode
                options = None
                if rng.random() < args.focus_probability:
                    options = {
                        "progress_fraction": float(
                            rng.uniform(args.focus_start, args.focus_end)
                        ),
                        "lateral_error_m": signed_uniform(rng, 0.08, 0.22),
                        "yaw_error_rad": signed_uniform(
                            rng, math.radians(4), math.radians(14)
                        ),
                    }
                observation, _ = env.reset(seed=episode_seed, options=options)
                episode_return = 0.0
                cte_values = []
                residual_values = []
                reason = "running"
                episode_steps = 0
                while global_step < args.total_steps:
                    noise = args.exploration_noise
                    action, _, residual = agent.action(observation, noise)
                    next_observation, reward, terminated, truncated, info = env.step(
                        np.asarray([action], dtype=np.float32)
                    )
                    done = terminated or truncated
                    replay.add(observation, action, reward, next_observation, done)
                    observation = next_observation
                    global_step += 1
                    episode_steps += 1
                    episode_return += reward
                    cte_values.append(abs(float(info["cross_track_error_m"])))
                    residual_values.append(abs(residual))
                    reason = str(info["reason"])
                    if len(replay) >= max(args.batch_size, args.learning_starts):
                        for _ in range(args.updates_per_step):
                            agent.update(replay.sample(args.batch_size, device))
                    if global_step % args.save_every_steps == 0:
                        save_checkpoint(
                            output_dir / "residual_td3_latest.pth",
                            agent,
                            args,
                            global_step,
                        )
                    if done:
                        break
                row = {
                    "episode": episode,
                    "global_step": global_step,
                    "seed": episode_seed,
                    "return": episode_return,
                    "steps": episode_steps,
                    "progress_m": info["cumulative_progress_m"],
                    "mean_abs_cte_m": float(np.mean(cte_values)),
                    "max_abs_cte_m": float(np.max(cte_values)),
                    "reason": reason,
                    "mean_abs_residual_norm": float(np.mean(residual_values)),
                }
                writer.writerow(row)
                metrics_file.flush()
                print(
                    f"episode={episode} step={global_step}/{args.total_steps} "
                    f"return={episode_return:.2f} progress={row['progress_m']:.2f} "
                    f"cte={row['mean_abs_cte_m']:.3f} reason={reason}",
                    flush=True,
                )
                episode += 1
        finally:
            env.close()
    save_checkpoint(
        output_dir / "residual_td3_final.pth", agent, args, global_step
    )
    with (output_dir / "train_config.json").open("w", encoding="utf-8") as handle:
        serializable = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        json.dump(serializable, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
