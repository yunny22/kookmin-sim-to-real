from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from xycar_rl.camera_speed_models import (
    DEFAULT_MAX_SPEED_COMMAND,
    DEFAULT_MIN_SPEED_COMMAND,
    expand_actor_speed_range,
    load_camera_speed_actor,
)
from xycar_rl.reward import (
    RewardWeights,
    lap_time_objective_weights,
    straight_high_speed_objective_weights,
)
from xycar_rl.td3_bc import CameraSpeedTD3BCAgent, TD3BCConfig
from xycar_rl.train_camera_speed_bc import grouped_split
from xycar_rl.train_td3_bc import resolve_device
from xycar_rl.transition_dataset import (
    CameraSpeedTransitionDataset,
    find_transition_csvs,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a camera-only steering and speed TD3+BC policy."
    )
    parser.add_argument("--transitions", action="append", required=True)
    parser.add_argument(
        "--focus-transitions",
        action="append",
        default=[],
        help="Transition directory or CSV sampled more often in the training split.",
    )
    parser.add_argument("--focus-repeat", type=float, default=1.0)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="Resume actor, critics, targets, and optimizers from a full checkpoint.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-speed-command", type=float, default=DEFAULT_MIN_SPEED_COMMAND)
    parser.add_argument("--max-speed-command", type=float, default=DEFAULT_MAX_SPEED_COMMAND)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument(
        "--recompute-rewards",
        action="store_true",
        help="Rebuild every transition reward with the current reward contract.",
    )
    parser.add_argument(
        "--world-sdf",
        type=Path,
        default=Path("worlds/kookmin_xycar_track_final.sdf"),
    )
    parser.add_argument(
        "--reward-objective",
        choices=["legacy", "lap_time", "straight_high_speed"],
        default="legacy",
        help="Reward contract used when --recompute-rewards is enabled.",
    )
    parser.add_argument("--target-right-offset-m", type=float, default=0.0)
    parser.add_argument(
        "--straight-only",
        action="store_true",
        help="Train only on transitions whose current and guarded preview are straight.",
    )
    parser.add_argument("--straight-curvature-threshold", type=float, default=0.10)
    parser.add_argument("--straight-guard-distance-m", type=float, default=0.80)
    parser.add_argument(
        "--straight-bc-target-speed-command",
        type=float,
        default=0.0,
        help=(
            "Override successful straight BC speed targets; zero preserves "
            "the recorded command."
        ),
    )
    parser.add_argument("--off-track-threshold-m", type=float, default=0.38)
    parser.add_argument("--lane-margin-start-m", type=float, default=0.24)
    parser.add_argument(
        "--bc-successful-episodes-only",
        action="store_true",
        help="Use failed episodes for the critic but never imitate their actions.",
    )
    parser.add_argument(
        "--speed-extension-only",
        action="store_true",
        help=(
            "Freeze the approved encoder and steering head, and train only "
            "RangeExpandedCameraSpeedActor.speed_extension."
        ),
    )
    parser.add_argument(
        "--expand-speed-range",
        action="store_true",
        help=(
            "Wrap the initial actor in a range-expanded speed head when its "
            "stored range differs from --min/--max-speed-command."
        ),
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help=(
            "Keep the approved visual encoder and BatchNorm statistics fixed "
            "while fine-tuning the steering/speed control heads."
        ),
    )
    parser.add_argument("--actor-lr", type=float, default=1.0e-5)
    parser.add_argument("--critic-lr", type=float, default=3.0e-4)
    parser.add_argument("--bc-alpha", type=float, default=2.5)
    parser.add_argument("--steering-bc-weight", type=float, default=1.0)
    parser.add_argument("--speed-bc-weight", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy-noise", type=float, default=0.15)
    parser.add_argument("--noise-clip", type=float, default=0.35)
    parser.add_argument("--policy-delay", type=int, default=2)
    parser.add_argument("--checkpoint-every-epochs", type=int, default=5)
    parser.add_argument(
        "--milestone-dir",
        type=Path,
        help="Optional tracked directory for actor-only deployment checkpoints.",
    )
    return parser.parse_args(argv)


@torch.no_grad()
def validation_metrics(agent, loader) -> dict[str, float]:
    if len(loader) == 0:
        return {
            "validation_steering_mse": float("nan"),
            "validation_speed_mse": float("nan"),
            "validation_steering_abs_mean": float("nan"),
        }
    squared_error = np.zeros(2, dtype=np.float64)
    steering_abs_sum = 0.0
    count = 0
    agent.actor.eval()
    for batch in loader:
        image = batch["image"].to(agent.device)
        action = batch.get("bc_action", batch["action"]).to(agent.device)
        sample_weight = batch.get("bc_weight")
        if sample_weight is not None:
            selected = sample_weight.reshape(-1) > 0.0
            if not bool(selected.any()):
                continue
            image = image[selected]
            action = action[selected]
        prediction = agent.actor(image)
        squared_error += (
            torch.sum((prediction - action) ** 2, dim=0).detach().cpu().numpy()
        )
        steering_abs_sum += float(torch.sum(torch.abs(prediction[:, 0])).cpu())
        count += int(action.shape[0])
    agent.actor.train()
    return {
        "validation_steering_mse": float(squared_error[0] / max(1, count)),
        "validation_speed_mse": float(squared_error[1] / max(1, count)),
        "validation_steering_abs_mean": float(
            steering_abs_sum / max(1, count)
        ),
    }


def save_checkpoint(
    path, agent, *, epoch, train_config, metrics, model_type
) -> None:
    torch.save(
        {
            "model_type": str(model_type),
            "temporal_frames": int(agent.temporal_frames),
            "algorithm": "camera_speed_td3_bc",
            "epoch": int(epoch),
            "min_speed_command": train_config["min_speed_command"],
            "max_speed_command": train_config["max_speed_command"],
            "speed_range_expansion": getattr(
                agent.actor,
                "speed_range_expansion",
                None,
            ),
            "train_config": train_config,
            "metrics": metrics,
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


def planned_simulation_cap(epoch: int, epochs: int) -> float:
    del epoch, epochs
    # Zero is the explicit runtime contract for "no additional deployment cap".
    return 0.0


def save_deployment_checkpoint(
    path: Path,
    agent: CameraSpeedTD3BCAgent,
    *,
    epoch: int,
    train_config: dict,
    metrics: dict,
    model_type: str,
    simulation_cap: float,
) -> None:
    torch.save(
        {
            "model_type": str(model_type),
            "temporal_frames": int(agent.temporal_frames),
            "algorithm": "camera_speed_td3_bc",
            "epoch": int(epoch),
            "min_speed_command": train_config["min_speed_command"],
            "max_speed_command": train_config["max_speed_command"],
            "speed_range_expansion": getattr(
                agent.actor,
                "speed_range_expansion",
                None,
            ),
            "planned_simulation_speed_cap": float(simulation_cap),
            "suggested_shadow_speed_cap": 0.0,
            "metrics": metrics,
            "actor_state_dict": agent.actor.state_dict(),
            "update_count": int(agent.update_count),
        },
        path,
    )


def restore_agent_checkpoint(
    agent: CameraSpeedTD3BCAgent,
    checkpoint_path: str | Path,
) -> dict:
    payload = torch.load(
        Path(checkpoint_path).expanduser().resolve(),
        map_location=agent.device,
        weights_only=False,
    )
    required = {
        "actor_state_dict",
        "actor_target_state_dict",
        "critic_state_dict",
        "critic_target_state_dict",
        "actor_optimizer_state_dict",
        "critic_optimizer_state_dict",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(
            "resume checkpoint is not a full TD3+BC checkpoint; missing "
            + ", ".join(missing)
        )
    agent.actor.load_state_dict(payload["actor_state_dict"], strict=True)
    agent.actor_target.load_state_dict(
        payload["actor_target_state_dict"], strict=True
    )
    agent.critic.load_state_dict(payload["critic_state_dict"], strict=True)
    agent.critic_target.load_state_dict(
        payload["critic_target_state_dict"], strict=True
    )
    agent.actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"])
    agent.critic_optimizer.load_state_dict(payload["critic_optimizer_state_dict"])
    for group in agent.actor_optimizer.param_groups:
        group["lr"] = agent.config.actor_lr
    for group in agent.critic_optimizer.param_groups:
        group["lr"] = agent.config.critic_lr
    agent.update_count = int(payload.get("update_count", 0))
    return payload


def transition_root(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved.is_file():
        return resolved.parent
    if (resolved / "transitions.csv").is_file():
        return resolved
    raise FileNotFoundError(f"focus transition path has no transitions.csv: {resolved}")


def transition_roots(paths: list[str | Path]) -> set[Path]:
    if not paths:
        return set()
    return {
        csv_path.parent.resolve()
        for csv_path in find_transition_csvs(paths)
    }


def export_scripted_actor(
    path: Path,
    actor: torch.nn.Module,
    *,
    temporal_frames: int,
) -> None:
    export_actor = deepcopy(actor).to("cpu").eval()
    scripted = torch.jit.trace(
        export_actor,
        torch.zeros(1, 3 * temporal_frames, 90, 160),
    )
    scripted.save(str(path))


def write_milestone_manifest(
    milestone_dir: Path,
    milestones: list[dict],
) -> None:
    with (milestone_dir / "milestones.json").open("w", encoding="utf-8") as handle:
        json.dump(milestones, handle, indent=2, sort_keys=True)
    lines = [
        "# High-speed TD3+BC milestones",
        "",
        "Every checkpoint is actor-only and intended for shadow evaluation first.",
        "A deployment speed cap of 0 means the learned speed is not clipped again.",
        "Shadow mode publishes commands without actuating the vehicle.",
        "",
        "```bash",
        'MODEL_DIR="$(ros2 pkg prefix xycar_rl)/share/xycar_rl/models/'
        f'{milestone_dir.name}"',
        "```",
        "",
    ]
    for item in milestones:
        title = (
            f"Selected best (epoch {item['epoch']:03d})"
            if item.get("selected", False)
            else f"Epoch {item['epoch']:03d}"
        )
        lines.extend(
            [
                f"## {title}",
                "",
                "Additional deployment cap: `disabled (0.0)`",
                "",
                "Real shadow cap: `disabled (0.0)`",
                "",
                "```bash",
                "ros2 launch xycar_rl real_shadow.launch.py \\",
                "  policy_kind:=camera_speed_td3_bc \\",
                f"  checkpoint_path:=$MODEL_DIR/{item['checkpoint']} \\",
                "  min_speed_command:=4.0 max_speed_command:=25.0 \\",
                "  deployment_speed_cap:=0.0 \\",
                "  adaptive_steering_enabled:=false \\",
                "  steering_temporal_alpha:=1.0 speed_temporal_alpha:=1.0 \\",
                "  max_inference_rate_hz:=7.0 \\",
                "  drive_enabled:=false lidar_safety_enabled:=false device:=cpu",
                "```",
                "",
            ]
        )
    (milestone_dir / "README.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main(argv=None) -> None:
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    milestone_dir = (
        None
        if args.milestone_dir is None
        else args.milestone_dir.expanduser().resolve()
    )
    if milestone_dir is not None:
        milestone_dir.mkdir(parents=True, exist_ok=True)
    policy_checkpoint = (
        args.initial_checkpoint
        if args.resume_checkpoint is None
        else args.resume_checkpoint
    )
    actor, initial_payload = load_camera_speed_actor(
        policy_checkpoint.expanduser().resolve(), device=device
    )
    source_range = (
        float(initial_payload.get("min_speed_command", args.min_speed_command)),
        float(initial_payload.get("max_speed_command", args.max_speed_command)),
    )
    target_range = (
        float(args.min_speed_command),
        float(args.max_speed_command),
    )
    model_type = str(
        initial_payload.get("model_type", "camera_speed_resnet18")
    )
    if source_range != target_range:
        if args.resume_checkpoint is not None:
            raise ValueError(
                f"resume checkpoint speed range {source_range} != {target_range}"
            )
        if not args.expand_speed_range:
            raise ValueError(
                "initial policy and transition action ranges differ: "
                f"{source_range} != {target_range}; pass --expand-speed-range"
            )
        actor = expand_actor_speed_range(
            actor,
            source_min_speed_command=source_range[0],
            source_max_speed_command=source_range[1],
            target_min_speed_command=target_range[0],
            target_max_speed_command=target_range[1],
        )
        model_type = f"{model_type}_range_expanded"
    if args.speed_extension_only and args.freeze_encoder:
        raise ValueError(
            "--speed-extension-only already freezes the encoder; "
            "choose only one freeze mode"
        )
    if args.speed_extension_only:
        freeze_base_policy = getattr(actor, "freeze_base_policy", None)
        if freeze_base_policy is None:
            raise ValueError(
                "--speed-extension-only requires a range-expanded actor"
            )
        freeze_base_policy()
    elif args.freeze_encoder:
        freeze_encoder = getattr(actor, "freeze_encoder", None)
        if freeze_encoder is None:
            raise ValueError("--freeze-encoder requires a range-expanded actor")
        freeze_encoder()
    temporal_frames = int(getattr(actor, "temporal_frames", 1))
    if args.reward_objective == "lap_time":
        reward_weights = lap_time_objective_weights(
            lane_margin_start_m=args.lane_margin_start_m,
            lane_departure_threshold_m=args.off_track_threshold_m,
        )
    elif args.reward_objective == "straight_high_speed":
        reward_weights = straight_high_speed_objective_weights(
            target_speed_command=args.max_speed_command,
        )
    else:
        reward_weights = RewardWeights()
    dataset = CameraSpeedTransitionDataset(
        args.transitions,
        min_speed_command=args.min_speed_command,
        max_speed_command=args.max_speed_command,
        temporal_frames=temporal_frames,
        recompute_rewards=args.recompute_rewards,
        world_sdf=args.world_sdf,
        target_right_offset_m=args.target_right_offset_m,
        reward_weights=reward_weights,
        bc_successful_episodes_only=args.bc_successful_episodes_only,
        straight_only=args.straight_only,
        straight_curvature_threshold=args.straight_curvature_threshold,
        straight_guard_distance_m=args.straight_guard_distance_m,
        bc_speed_target_command=(
            args.straight_bc_target_speed_command
            if args.straight_bc_target_speed_command > 0.0
            else None
        ),
    )
    train_indices, validation_indices = grouped_split(
        dataset, args.validation_ratio, args.seed
    )
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    generator = torch.Generator().manual_seed(args.seed)
    train_subset = Subset(dataset, train_indices)
    focus_roots = transition_roots(args.focus_transitions)
    focus_repeat = max(1.0, float(args.focus_repeat))
    if focus_roots:
        sample_weights = [
            (
                focus_repeat
                if dataset.rows[index][0].resolve() in focus_roots
                else 1.0
            )
            for index in train_indices
        ]
        train_sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
        train_loader = DataLoader(
            train_subset,
            sampler=train_sampler,
            **loader_args,
        )
    else:
        train_loader = DataLoader(
            train_subset,
            shuffle=True,
            generator=generator,
            **loader_args,
        )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices), shuffle=False, **loader_args
    )
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
    resume_epoch = 0
    if args.resume_checkpoint is not None:
        resume_payload = restore_agent_checkpoint(
            agent,
            args.resume_checkpoint,
        )
        resume_epoch = int(resume_payload.get("epoch", 0))
    train_config = {
        **vars(args),
        "initial_checkpoint": str(args.initial_checkpoint.expanduser().resolve()),
        "resume_checkpoint": (
            None
            if args.resume_checkpoint is None
            else str(args.resume_checkpoint.expanduser().resolve())
        ),
        "resume_epoch": resume_epoch,
        "output_dir": str(output_dir),
        "device": str(device),
        "dataset_rows": len(dataset),
        "dataset_rows_before_straight_filter": dataset.original_row_count,
        "dataset_rows_after_straight_filter": dataset.straight_row_count,
        "train_rows": len(train_indices),
        "validation_rows": len(validation_indices),
        "td3_bc": asdict(config),
        "model_type": model_type,
        "temporal_frames": temporal_frames,
    }
    for key, value in list(train_config.items()):
        if isinstance(value, Path):
            train_config[key] = str(value)
    with (output_dir / "train_config.json").open("w", encoding="utf-8") as handle:
        json.dump(train_config, handle, indent=2, sort_keys=True)

    fields = [
        "epoch",
        "critic_loss",
        "actor_loss",
        "bc_loss",
        "q_scale",
        "validation_steering_mse",
        "validation_speed_mse",
        "validation_steering_abs_mean",
        "elapsed_sec",
    ]
    best_metric = float("inf")
    milestones: list[dict] = []
    started = time.monotonic()
    with (output_dir / "metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as history_file:
        writer = csv.DictWriter(history_file, fieldnames=fields)
        writer.writeheader()
        for additional_epoch in range(1, args.epochs + 1):
            epoch = resume_epoch + additional_epoch
            sums = {key: 0.0 for key in ("critic_loss", "actor_loss", "bc_loss", "q_scale")}
            counts = {key: 0 for key in sums}
            agent.actor.train()
            for batch in train_loader:
                metrics = agent.update(batch)
                for key, value in metrics.items():
                    if np.isfinite(value):
                        sums[key] += value
                        counts[key] += 1
            row = {
                "epoch": epoch,
                **{key: sums[key] / max(1, counts[key]) for key in sums},
                **validation_metrics(agent, validation_loader),
                "elapsed_sec": time.monotonic() - started,
            }
            writer.writerow(row)
            history_file.flush()
            save_checkpoint(
                output_dir / "camera_speed_td3_bc_latest.pth",
                agent,
                epoch=epoch,
                train_config=train_config,
                metrics=row,
                model_type=model_type,
            )
            selection_metric = (
                row["validation_steering_mse"] + row["validation_speed_mse"]
            )
            if args.straight_only:
                selection_metric += 0.10 * row[
                    "validation_steering_abs_mean"
                ]
            if selection_metric < best_metric:
                best_metric = selection_metric
                save_checkpoint(
                    output_dir / "camera_speed_td3_bc_best.pth",
                    agent,
                    epoch=epoch,
                    train_config=train_config,
                    metrics=row,
                    model_type=model_type,
                )
            checkpoint_period = max(1, int(args.checkpoint_every_epochs))
            if (
                additional_epoch % checkpoint_period == 0
                or additional_epoch == args.epochs
            ):
                full_checkpoint = (
                    output_dir / f"camera_speed_td3_bc_epoch_{epoch:03d}.pth"
                )
                save_checkpoint(
                    full_checkpoint,
                    agent,
                    epoch=epoch,
                    train_config=train_config,
                    metrics=row,
                    model_type=model_type,
                )
                if milestone_dir is not None:
                    simulation_cap = planned_simulation_cap(
                        additional_epoch,
                        args.epochs,
                    )
                    checkpoint_name = f"camera_speed_td3_bc_epoch_{epoch:03d}.pth"
                    save_deployment_checkpoint(
                        milestone_dir / checkpoint_name,
                        agent,
                        epoch=epoch,
                        train_config=train_config,
                        metrics=row,
                        model_type=model_type,
                        simulation_cap=simulation_cap,
                    )
                    milestones.append(
                        {
                            "epoch": epoch,
                            "checkpoint": checkpoint_name,
                            "planned_simulation_speed_cap": simulation_cap,
                            "suggested_shadow_speed_cap": 0.0,
                            "validation_steering_mse": row[
                                "validation_steering_mse"
                            ],
                            "validation_speed_mse": row[
                                "validation_speed_mse"
                            ],
                        }
                    )
                    write_milestone_manifest(milestone_dir, milestones)
            print(
                f"epoch {epoch:03d} (+{additional_epoch:03d}/{args.epochs}): "
                f"critic={row['critic_loss']:.5f} "
                f"actor={row['actor_loss']:.5f} bc={row['bc_loss']:.5f} "
                f"steer={row['validation_steering_mse']:.5f} "
                f"speed={row['validation_speed_mse']:.5f}",
                flush=True,
            )

    best_actor, best_payload = load_camera_speed_actor(
        output_dir / "camera_speed_td3_bc_best.pth", device="cpu"
    )
    if milestone_dir is not None:
        selected_checkpoint = "camera_speed_td3_bc_best.pth"
        torch.save(
            {
                key: best_payload[key]
                for key in (
                    "model_type",
                    "temporal_frames",
                    "algorithm",
                    "epoch",
                    "min_speed_command",
                    "max_speed_command",
                    "speed_range_expansion",
                    "metrics",
                    "actor_state_dict",
                    "update_count",
                )
            }
            | {
                "planned_simulation_speed_cap": 0.0,
                "suggested_shadow_speed_cap": 0.0,
            },
            milestone_dir / selected_checkpoint,
        )
        selected = {
            "epoch": int(best_payload["epoch"]),
            "checkpoint": selected_checkpoint,
            "selected": True,
            "planned_simulation_speed_cap": 0.0,
            "suggested_shadow_speed_cap": 0.0,
            "validation_steering_mse": float(
                best_payload["metrics"]["validation_steering_mse"]
            ),
            "validation_speed_mse": float(
                best_payload["metrics"]["validation_speed_mse"]
            ),
        }
        write_milestone_manifest(milestone_dir, [selected, *milestones])
    export_scripted_actor(
        output_dir / "camera_speed_td3_bc_actor_scripted.pt",
        best_actor,
        temporal_frames=temporal_frames,
    )
    print(f"wrote camera-speed TD3+BC policy: {output_dir}")


if __name__ == "__main__":
    main()
