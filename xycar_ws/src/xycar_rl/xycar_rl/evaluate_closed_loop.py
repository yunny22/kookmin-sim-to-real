from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from xycar_rl.gazebo_env import GazeboXycarEnv
from xycar_rl.policy_loader import load_steering_policy
from xycar_rl.train_td3_bc import resolve_device


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare BC and RL with identical Gazebo reset seeds."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--bc-checkpoint", type=Path, required=True)
    parser.add_argument("--rl-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--rl-kind", choices=["td3_bc", "residual"], default="td3_bc"
    )
    parser.add_argument("--residual-base-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--max-steps", type=int, default=1_200)
    parser.add_argument("--speed-command", type=float, default=4.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args(argv)


def run_episode(env, policy, seed: int, max_steps: int) -> dict:
    observation, reset_info = env.reset(seed=seed)
    total_reward = 0.0
    cte = []
    steering = []
    reason = "evaluation_limit"
    info = reset_info
    for step in range(1, max_steps + 1):
        action = policy(observation)
        observation, reward, terminated, truncated, info = env.step(
            np.asarray([action], dtype=np.float32)
        )
        total_reward += reward
        cte.append(abs(float(info["cross_track_error_m"])))
        steering.append(abs(action))
        reason = str(info["reason"])
        if terminated or truncated:
            break
    return {
        "seed": seed,
        "return": total_reward,
        "steps": step,
        "progress_m": float(info["cumulative_progress_m"]),
        "completion_fraction": min(
            1.0, float(info["cumulative_progress_m"]) / env.track.length_m
        ),
        "mean_abs_cte_m": float(np.mean(cte)) if cte else float("nan"),
        "max_abs_cte_m": float(np.max(cte)) if cte else float("nan"),
        "mean_abs_action": float(np.mean(steering)) if steering else float("nan"),
        "reason": reason,
        "lap_complete": reason == "lap_complete",
    }


def aggregate(rows: list[dict]) -> dict:
    numeric = [
        "return",
        "steps",
        "progress_m",
        "completion_fraction",
        "mean_abs_cte_m",
        "max_abs_cte_m",
        "mean_abs_action",
    ]
    result = {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in numeric
    }
    result["lap_success_rate"] = float(
        np.mean([float(row["lap_complete"]) for row in rows])
    )
    result["episode_count"] = len(rows)
    return result


def main(argv=None) -> None:
    args = parse_args(argv)
    device = resolve_device(args.device)
    root = args.project_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    policies = {
        "bc": load_steering_policy("bc", args.bc_checkpoint, device=device),
        "rl": load_steering_policy(
            args.rl_kind,
            args.rl_checkpoint,
            device=device,
            residual_base_checkpoint=args.residual_base_checkpoint,
        ),
    }
    env = GazeboXycarEnv(
        world_sdf=root / "worlds" / "kookmin_xycar_track_final.sdf",
        speed_command=args.speed_command,
    )
    rows = []
    try:
        for label, policy in policies.items():
            for offset in range(args.episodes):
                seed = args.seed + offset
                row = {"policy": label, **run_episode(env, policy, seed, args.max_steps)}
                rows.append(row)
                print(
                    f"{label} seed={seed} progress={row['progress_m']:.2f}m "
                    f"cte={row['mean_abs_cte_m']:.3f} reason={row['reason']}",
                    flush=True,
                )
    finally:
        env.close()
    fieldnames = list(rows[0])
    with (output_dir / "episodes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        label: aggregate([row for row in rows if row["policy"] == label])
        for label in policies
    }
    summary["same_seeds"] = [args.seed + i for i in range(args.episodes)]
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
