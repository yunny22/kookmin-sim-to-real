from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from xycar_rl.camera_speed_models import (
    expand_actor_speed_range,
    load_camera_speed_actor,
)
from xycar_rl.gazebo_env import GazeboXycarEnv
from xycar_rl.lap_time_selection import summarize_candidate
from xycar_rl.reward import lap_time_objective_weights
from xycar_rl.td3_bc import CameraSpeedTD3BCAgent, TD3BCConfig
from xycar_rl.termination import TerminationConfig
from xycar_rl.train_camera_speed_td3_bc import restore_agent_checkpoint
from xycar_rl.train_td3_bc import resolve_device


PHYSICAL_MAX_SPEED_COMMAND = 100.0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune a camera-speed policy with safe completion as a hard "
            "selection gate and successful lap time as the secondary objective."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=700)
    parser.add_argument("--control-rate-hz", type=float, default=7.0)
    parser.add_argument("--min-speed-command", type=float, default=4.0)
    parser.add_argument(
        "--max-speed-command",
        type=float,
        default=PHYSICAL_MAX_SPEED_COMMAND,
        help="Physical motor-interface maximum, not a curriculum cap.",
    )
    parser.add_argument("--preserve-speed-command", type=float, default=18.5)
    parser.add_argument("--off-track-threshold-m", type=float, default=0.38)
    parser.add_argument("--lane-margin-start-m", type=float, default=0.24)
    parser.add_argument("--target-right-offset-m", type=float, default=0.0)
    parser.add_argument("--recovery-probability", type=float, default=0.20)
    parser.add_argument("--recovery-max-lateral-m", type=float, default=0.10)
    parser.add_argument("--recovery-max-yaw-deg", type=float, default=6.0)
    parser.add_argument("--fixed-start-probability", type=float, default=0.50)
    parser.add_argument("--replay-capacity", type=int, default=5_000)
    parser.add_argument("--warmup-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--steering-noise", type=float, default=0.025)
    parser.add_argument("--speed-noise", type=float, default=0.025)
    parser.add_argument(
        "--maximum-speed-exploration-bias",
        type=float,
        default=0.20,
        help="Maximum positive normalized speed bias; this is exploration, not a cap.",
    )
    parser.add_argument("--actor-lr", type=float, default=3.0e-6)
    parser.add_argument("--critic-lr", type=float, default=1.0e-4)
    parser.add_argument(
        "--speed-extension-only",
        action="store_true",
        help=(
            "Freeze the approved visual encoder and steering head, and train "
            "only the range-expanded speed extension."
        ),
    )
    parser.add_argument("--bc-alpha", type=float, default=1.0)
    parser.add_argument("--steering-bc-weight", type=float, default=2.0)
    parser.add_argument("--speed-bc-weight", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy-noise", type=float, default=0.08)
    parser.add_argument("--noise-clip", type=float, default=0.20)
    parser.add_argument("--policy-delay", type=int, default=2)
    parser.add_argument("--evaluate-every-episodes", type=int, default=5)
    parser.add_argument("--evaluation-laps", type=int, default=5)
    parser.add_argument("--checkpoint-every-episodes", type=int, default=5)
    return parser.parse_args(argv)


class ReplayBuffer:
    def __init__(self, capacity: int, image_shape: tuple[int, int, int], seed: int):
        self.capacity = max(1, int(capacity))
        self.images = np.empty(
            (self.capacity, *image_shape),
            dtype=np.uint8,
        )
        self.next_images = np.empty_like(self.images)
        self.actions = np.empty((self.capacity, 2), dtype=np.float32)
        self.rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self.dones = np.empty((self.capacity, 1), dtype=np.float32)
        self.bc_weights = np.zeros((self.capacity, 1), dtype=np.float32)
        self.slot_tokens = np.zeros(self.capacity, dtype=np.int64)
        self.next_token = 1
        self.size = 0
        self.position = 0
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _encode(image: np.ndarray) -> np.ndarray:
        return np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)

    def append(
        self,
        image: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_image: np.ndarray,
        done: bool,
    ) -> tuple[int, int]:
        index = self.position
        token = self.next_token
        self.next_token += 1
        self.images[index] = self._encode(image)
        self.next_images[index] = self._encode(next_image)
        self.actions[index] = np.asarray(action, dtype=np.float32)
        self.rewards[index, 0] = float(reward)
        self.dones[index, 0] = float(done)
        self.bc_weights[index, 0] = 0.0
        self.slot_tokens[index] = token
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)
        return index, token

    def mark_bc_eligible(self, references: list[tuple[int, int]]) -> None:
        for index, token in references:
            if self.slot_tokens[index] == token:
                self.bc_weights[index, 0] = 1.0

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        if self.size < batch_size:
            raise ValueError("replay buffer has fewer rows than the batch size")
        indices = self.rng.integers(0, self.size, size=int(batch_size))
        image = torch.from_numpy(self.images[indices].astype(np.float32) / 255.0)
        next_image = torch.from_numpy(
            self.next_images[indices].astype(np.float32) / 255.0
        )
        action = torch.from_numpy(self.actions[indices])
        return {
            "image": image,
            "action": action,
            "bc_action": action,
            "bc_weight": torch.from_numpy(self.bc_weights[indices]),
            "reward": torch.from_numpy(self.rewards[indices]),
            "next_image": next_image,
            "done": torch.from_numpy(self.dones[indices]),
        }


def temporal_state(
    previous_image: np.ndarray | None,
    current_image: np.ndarray,
    temporal_frames: int,
) -> np.ndarray:
    current = np.asarray(current_image, dtype=np.float32)
    if temporal_frames == 1:
        return current
    previous = current if previous_image is None else previous_image
    return np.concatenate([previous, current], axis=0)


@torch.no_grad()
def actor_action(
    actor: torch.nn.Module,
    state: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
    return (
        actor(tensor)
        .squeeze(0)
        .clamp(-1.0, 1.0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )


def save_agent_checkpoint(
    path: Path,
    agent: CameraSpeedTD3BCAgent,
    *,
    episode: int,
    speed_range: tuple[float, float],
    metrics: dict,
    train_config: dict,
    model_type: str,
    speed_range_expansion: dict | None,
) -> None:
    torch.save(
        {
            "model_type": model_type,
            "temporal_frames": int(agent.temporal_frames),
            "algorithm": "camera_speed_online_td3_bc_lap_time",
            "epoch": int(episode),
            "min_speed_command": float(speed_range[0]),
            "max_speed_command": float(speed_range[1]),
            "speed_range_expansion": speed_range_expansion,
            "metrics": metrics,
            "train_config": train_config,
            "actor_state_dict": agent.actor.state_dict(),
            "actor_target_state_dict": agent.actor_target.state_dict(),
            "critic_state_dict": agent.critic.state_dict(),
            "critic_target_state_dict": agent.critic_target.state_dict(),
            "actor_optimizer_state_dict": agent.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": agent.critic_optimizer.state_dict(),
            "update_count": int(agent.update_count),
        },
        path,
    )


def save_actor_checkpoint(
    path: Path,
    agent: CameraSpeedTD3BCAgent,
    *,
    episode: int,
    speed_range: tuple[float, float],
    metrics: dict,
    model_type: str,
    speed_range_expansion: dict | None,
) -> None:
    torch.save(
        {
            "model_type": model_type,
            "temporal_frames": int(agent.temporal_frames),
            "algorithm": "camera_speed_online_td3_bc_lap_time",
            "epoch": int(episode),
            "min_speed_command": float(speed_range[0]),
            "max_speed_command": float(speed_range[1]),
            "speed_range_expansion": speed_range_expansion,
            "planned_simulation_speed_cap": 0.0,
            "suggested_shadow_speed_cap": 0.0,
            "selection_contract": (
                "all evaluation laps complete without departure, then minimum mean lap time"
            ),
            "metrics": metrics,
            "actor_state_dict": agent.actor.state_dict(),
            "update_count": int(agent.update_count),
        },
        path,
    )


def evaluation_options() -> dict[str, float]:
    return {
        "progress_fraction": 0.0,
        "lateral_error_m": 0.0,
        "yaw_error_rad": 0.0,
    }


def evaluate_actor(
    agent: CameraSpeedTD3BCAgent,
    env: GazeboXycarEnv,
    *,
    episode: int,
    laps: int,
    seed: int,
    max_steps: int,
) -> tuple[list[dict], dict]:
    actor = agent.actor
    was_training = actor.training
    actor.eval()
    rows: list[dict] = []
    try:
        for lap in range(max(1, int(laps))):
            observation, info = env.reset(
                seed=seed + 100_000 + lap,
                options=evaluation_options(),
            )
            previous_image = None
            reason = "evaluation_limit"
            max_cte = 0.0
            speed_commands: list[float] = []
            for step in range(1, max_steps + 1):
                state = temporal_state(
                    previous_image,
                    observation["image"],
                    agent.temporal_frames,
                )
                action = actor_action(actor, state, agent.device)
                previous_image = observation["image"].copy()
                observation, _, terminated, truncated, info = env.step(action)
                max_cte = max(
                    max_cte,
                    abs(float(info.get("cross_track_error_m", 0.0))),
                )
                speed_commands.append(float(info.get("speed_command", 0.0)))
                reason = str(info.get("reason", "running"))
                if terminated or truncated:
                    break
            rows.append(
                {
                    "candidate": f"episode_{episode:04d}",
                    "evaluation_lap": lap,
                    "reason": reason,
                    "lap_time_sec": float(info.get("elapsed_sec", float("nan"))),
                    "steps": step,
                    "progress_m": float(
                        info.get("cumulative_progress_m", 0.0)
                    ),
                    "max_abs_cte_m": max_cte,
                    "mean_speed_command": (
                        float(np.mean(speed_commands))
                        if speed_commands
                        else float("nan")
                    ),
                    "max_speed_command": (
                        float(np.max(speed_commands))
                        if speed_commands
                        else float("nan")
                    ),
                }
            )
    finally:
        actor.train(was_training)
    summary = asdict(
        summarize_candidate(
            f"episode_{episode:04d}",
            rows,
            required_safe_laps=max(1, int(laps)),
        )
    )
    summary["episode"] = int(episode)
    return rows, summary


def append_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> None:
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    root = args.project_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_path = args.initial_checkpoint.expanduser().resolve()
    actor_path = (
        initial_path
        if args.resume_checkpoint is None
        else args.resume_checkpoint.expanduser().resolve()
    )
    actor, source_payload = load_camera_speed_actor(actor_path, device=device)
    source_min = float(source_payload.get("min_speed_command", 4.0))
    source_max = float(source_payload.get("max_speed_command", 24.0))
    target_range = (
        float(args.min_speed_command),
        float(args.max_speed_command),
    )
    if target_range[1] > PHYSICAL_MAX_SPEED_COMMAND + 1.0e-6:
        raise ValueError(
            f"max speed exceeds the physical interface limit "
            f"{PHYSICAL_MAX_SPEED_COMMAND:g}"
        )
    migration = source_payload.get("speed_range_expansion")
    model_type = str(
        source_payload.get("model_type", "camera_speed_temporal_resnet18")
    )
    if (source_min, source_max) != target_range:
        actor = expand_actor_speed_range(
            actor,
            source_min_speed_command=source_min,
            source_max_speed_command=source_max,
            target_min_speed_command=target_range[0],
            target_max_speed_command=target_range[1],
        )
        migration = actor.speed_range_expansion
        model_type = f"{model_type}_range_expanded"
    if args.speed_extension_only:
        freeze_base_policy = getattr(actor, "freeze_base_policy", None)
        if freeze_base_policy is None:
            raise ValueError(
                "--speed-extension-only requires a range-expanded actor"
            )
        freeze_base_policy()

    config = TD3BCConfig(
        gamma=args.gamma,
        tau=args.tau,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_delay=args.policy_delay,
        bc_alpha=args.bc_alpha,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        steering_bc_weight=args.steering_bc_weight,
        speed_bc_weight=args.speed_bc_weight,
    )
    agent = CameraSpeedTD3BCAgent(actor, device=device, config=config)
    resume_episode = 0
    if args.resume_checkpoint is not None:
        resume_payload = restore_agent_checkpoint(
            agent,
            args.resume_checkpoint.expanduser().resolve(),
        )
        resumed_range = (
            float(resume_payload["min_speed_command"]),
            float(resume_payload["max_speed_command"]),
        )
        if resumed_range != target_range:
            raise ValueError(
                f"resume speed range {resumed_range} != {target_range}"
            )
        resume_episode = int(resume_payload.get("epoch", 0))

    reward_weights = lap_time_objective_weights(
        lane_margin_start_m=args.lane_margin_start_m,
        lane_departure_threshold_m=args.off_track_threshold_m,
    )
    termination = TerminationConfig(
        off_track_threshold_m=args.off_track_threshold_m,
        max_episode_sec=args.max_steps / max(1.0, args.control_rate_hz),
        minimum_lap_fraction=0.95,
    )
    env = GazeboXycarEnv(
        world_sdf=root / "worlds" / "kookmin_xycar_track_final.sdf",
        variable_speed=True,
        require_lidar=False,
        control_rate_hz=args.control_rate_hz,
        speed_command=args.preserve_speed_command,
        min_speed_command=target_range[0],
        max_speed_command=target_range[1],
        target_right_offset_m=args.target_right_offset_m,
        termination_config=termination,
        reward_weights=reward_weights,
    )
    replay = ReplayBuffer(
        args.replay_capacity,
        (3 * agent.temporal_frames, env.input_height, env.input_width),
        args.seed,
    )
    train_config = {
        **vars(args),
        "initial_checkpoint": str(initial_path),
        "resume_checkpoint": (
            None
            if args.resume_checkpoint is None
            else str(args.resume_checkpoint.expanduser().resolve())
        ),
        "output_dir": str(output_dir),
        "device": str(device),
        "source_model_type": source_payload.get("model_type"),
        "temporal_frames": agent.temporal_frames,
        "physical_speed_command_range": list(target_range),
        "additional_speed_cap": 0.0,
        "migration": migration,
        "reward_weights": asdict(reward_weights),
        "selection_contract": (
            "all evaluation laps complete without departure, then minimum mean lap time"
        ),
        "td3_bc": asdict(config),
    }
    for key, value in list(train_config.items()):
        if isinstance(value, Path):
            train_config[key] = str(value)
    (output_dir / "train_config.json").write_text(
        json.dumps(train_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rng = np.random.default_rng(args.seed)
    global_step = 0
    best_lap_time = float("inf")
    latest_metrics: dict = {}
    started = time.monotonic()
    evaluations_path = output_dir / "evaluation_laps.csv"
    summaries_path = output_dir / "evaluation_summaries.jsonl"
    training_path = output_dir / "training_episodes.csv"
    try:
        baseline_rows, baseline = evaluate_actor(
            agent,
            env,
            episode=resume_episode,
            laps=args.evaluation_laps,
            seed=args.seed,
            max_steps=args.max_steps,
        )
        append_csv(evaluations_path, baseline_rows)
        with summaries_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(baseline, sort_keys=True) + "\n")
        if baseline["eligible"]:
            best_lap_time = float(baseline["mean_success_lap_time_sec"])
            save_actor_checkpoint(
                output_dir / "camera_speed_lap_time_best.pth",
                agent,
                episode=resume_episode,
                speed_range=target_range,
                metrics=baseline,
                model_type=model_type,
                speed_range_expansion=migration,
            )
            (output_dir / "selected_summary.json").write_text(
                json.dumps(baseline, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print(
            f"baseline safe={int(baseline['eligible'])} "
            f"laps={baseline['completed_laps']}/{baseline['episode_count']} "
            f"mean_lap={baseline['mean_success_lap_time_sec']:.3f}s",
            flush=True,
        )

        for additional_episode in range(1, args.episodes + 1):
            episode = resume_episode + additional_episode
            options: dict[str, float] = {
                "progress_fraction": (
                    0.0
                    if rng.random() < args.fixed_start_probability
                    else float(rng.uniform(0.0, 1.0))
                ),
                "lateral_error_m": 0.0,
                "yaw_error_rad": 0.0,
            }
            if rng.random() < args.recovery_probability:
                options["lateral_error_m"] = float(
                    rng.uniform(
                        -args.recovery_max_lateral_m,
                        args.recovery_max_lateral_m,
                    )
                )
                options["yaw_error_rad"] = float(
                    rng.uniform(
                        -math.radians(args.recovery_max_yaw_deg),
                        math.radians(args.recovery_max_yaw_deg),
                    )
                )
            observation, info = env.reset(
                seed=args.seed + episode,
                options=options,
            )
            previous_image = None
            total_reward = 0.0
            speed_commands: list[float] = []
            max_cte = 0.0
            update_sums: dict[str, float] = {}
            update_counts: dict[str, int] = {}
            reason = "training_limit"
            episode_replay_references: list[tuple[int, int]] = []
            for step in range(1, args.max_steps + 1):
                state = temporal_state(
                    previous_image,
                    observation["image"],
                    agent.temporal_frames,
                )
                action = actor_action(agent.actor, state, agent.device)
                schedule_fraction = min(
                    1.0,
                    additional_episode / max(1, args.episodes // 2),
                )
                action[0] += float(rng.normal(0.0, args.steering_noise))
                action[1] += float(rng.normal(0.0, args.speed_noise))
                action[1] += (
                    float(args.maximum_speed_exploration_bias)
                    * schedule_fraction
                )
                action = np.clip(action, -1.0, 1.0).astype(np.float32)
                current_image = observation["image"].copy()
                next_observation, reward, terminated, truncated, info = env.step(
                    action
                )
                next_state = temporal_state(
                    current_image,
                    next_observation["image"],
                    agent.temporal_frames,
                )
                done = bool(terminated or truncated)
                episode_replay_references.append(
                    replay.append(
                        state,
                        action,
                        reward,
                        next_state,
                        done,
                    )
                )
                previous_image = current_image
                observation = next_observation
                total_reward += float(reward)
                speed_commands.append(float(info.get("speed_command", 0.0)))
                max_cte = max(
                    max_cte,
                    abs(float(info.get("cross_track_error_m", 0.0))),
                )
                global_step += 1

                if (
                    replay.size >= max(args.warmup_steps, args.batch_size)
                    and global_step >= args.warmup_steps
                ):
                    for _ in range(max(1, args.updates_per_step)):
                        update = agent.update(replay.sample(args.batch_size))
                        for key, value in update.items():
                            if math.isfinite(value):
                                update_sums[key] = (
                                    update_sums.get(key, 0.0) + value
                                )
                                update_counts[key] = (
                                    update_counts.get(key, 0) + 1
                                )
                reason = str(info.get("reason", "running"))
                if done:
                    break
            if reason == "lap_complete":
                replay.mark_bc_eligible(episode_replay_references)

            train_row = {
                "episode": episode,
                "steps": step,
                "return": total_reward,
                "reason": reason,
                "lap_time_sec": float(info.get("elapsed_sec", float("nan"))),
                "progress_m": float(info.get("cumulative_progress_m", 0.0)),
                "max_abs_cte_m": max_cte,
                "mean_speed_command": (
                    float(np.mean(speed_commands))
                    if speed_commands
                    else float("nan")
                ),
                "max_speed_command": (
                    float(np.max(speed_commands))
                    if speed_commands
                    else float("nan")
                ),
                "replay_size": replay.size,
                "global_step": global_step,
                **{
                    key: update_sums[key] / max(1, update_counts[key])
                    for key in sorted(update_sums)
                },
            }
            append_csv(training_path, [train_row])
            latest_metrics = train_row
            print(
                f"train episode={episode} reason={reason} steps={step} "
                f"return={total_reward:.2f} "
                f"lap={train_row['lap_time_sec']:.2f}s "
                f"speed={train_row['mean_speed_command']:.2f}/"
                f"{train_row['max_speed_command']:.2f} "
                f"cte={max_cte:.3f} replay={replay.size}",
                flush=True,
            )

            if episode % max(1, args.checkpoint_every_episodes) == 0:
                save_agent_checkpoint(
                    output_dir / "camera_speed_lap_time_latest_full.pth",
                    agent,
                    episode=episode,
                    speed_range=target_range,
                    metrics=latest_metrics,
                    train_config=train_config,
                    model_type=model_type,
                    speed_range_expansion=migration,
                )
                save_actor_checkpoint(
                    output_dir
                    / f"camera_speed_lap_time_episode_{episode:04d}.pth",
                    agent,
                    episode=episode,
                    speed_range=target_range,
                    metrics=latest_metrics,
                    model_type=model_type,
                    speed_range_expansion=migration,
                )

            if episode % max(1, args.evaluate_every_episodes) == 0:
                rows, summary = evaluate_actor(
                    agent,
                    env,
                    episode=episode,
                    laps=args.evaluation_laps,
                    seed=args.seed,
                    max_steps=args.max_steps,
                )
                append_csv(evaluations_path, rows)
                with summaries_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(summary, sort_keys=True) + "\n")
                print(
                    f"eval episode={episode} safe={int(summary['eligible'])} "
                    f"laps={summary['completed_laps']}/"
                    f"{summary['episode_count']} "
                    f"departures={summary['lane_departures']} "
                    f"mean_lap={summary['mean_success_lap_time_sec']:.3f}s",
                    flush=True,
                )
                if (
                    summary["eligible"]
                    and float(summary["mean_success_lap_time_sec"])
                    < best_lap_time
                ):
                    best_lap_time = float(
                        summary["mean_success_lap_time_sec"]
                    )
                    save_actor_checkpoint(
                        output_dir / "camera_speed_lap_time_best.pth",
                        agent,
                        episode=episode,
                        speed_range=target_range,
                        metrics=summary,
                        model_type=model_type,
                        speed_range_expansion=migration,
                    )
                    (output_dir / "selected_summary.json").write_text(
                        json.dumps(summary, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
    except KeyboardInterrupt:
        print("training interrupted; saving the latest full state", flush=True)
    finally:
        latest_metrics = {
            **latest_metrics,
            "elapsed_wall_sec": time.monotonic() - started,
            "global_step": global_step,
            "replay_size": replay.size,
        }
        save_agent_checkpoint(
            output_dir / "camera_speed_lap_time_latest_full.pth",
            agent,
            episode=resume_episode
            + max(
                0,
                additional_episode
                if "additional_episode" in locals()
                else 0,
            ),
            speed_range=target_range,
            metrics=latest_metrics,
            train_config=train_config,
            model_type=model_type,
            speed_range_expansion=migration,
        )
        env.close()


if __name__ == "__main__":
    main()
