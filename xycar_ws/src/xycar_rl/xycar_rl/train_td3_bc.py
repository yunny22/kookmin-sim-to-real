from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from xycar_rl.models import load_bc_actor, load_rl_actor
from xycar_rl.td3_bc import TD3BCAgent, TD3BCConfig
from xycar_rl.transition_dataset import RLTransitionDataset


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a steering-only TD3+BC policy from recorded RL transitions."
    )
    parser.add_argument(
        "--transitions",
        action="append",
        required=True,
        help="Session, root directory, or transitions.csv; repeat to combine data.",
    )
    parser.add_argument("--bc-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--actor-lr", type=float, default=1.0e-5)
    parser.add_argument("--critic-lr", type=float, default=3.0e-4)
    parser.add_argument("--bc-alpha", type=float, default=2.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy-noise", type=float, default=0.2)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--policy-delay", type=int, default=2)
    parser.add_argument("--action-max-steer-command", type=float, default=42.0)
    return parser.parse_args(argv)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(value)


def grouped_split(dataset: RLTransitionDataset, ratio: float, seed: int):
    groups: dict[tuple[str, str], list[int]] = {}
    for index, (root, row) in enumerate(dataset.rows):
        key = (str(root), row.get("episode_id", "0"))
        groups.setdefault(key, []).append(index)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    if len(keys) > 1 and ratio > 0.0:
        validation_groups = max(1, round(len(keys) * min(0.5, ratio)))
        validation_keys = set(keys[:validation_groups])
        validation = [i for key in validation_keys for i in groups[key]]
        train = [i for key in keys if key not in validation_keys for i in groups[key]]
    else:
        indices = list(range(len(dataset)))
        random.Random(seed).shuffle(indices)
        validation_count = max(1, round(len(indices) * ratio)) if len(indices) > 9 else 0
        validation = indices[:validation_count]
        train = indices[validation_count:]
    if not train:
        raise ValueError("training split is empty")
    return train, validation


@torch.no_grad()
def validation_metrics(agent: TD3BCAgent, loader: DataLoader) -> dict[str, float]:
    if len(loader) == 0:
        return {"validation_bc_mse": float("nan")}
    squared_error = 0.0
    count = 0
    agent.actor.eval()
    for batch in loader:
        image = batch["image"].to(agent.device)
        lidar = batch["lidar"].to(agent.device)
        action = batch["action"].to(agent.device)
        prediction = agent.actor(image, lidar)
        squared_error += float(torch.sum((prediction - action) ** 2).cpu())
        count += int(action.numel())
    agent.actor.train()
    return {"validation_bc_mse": squared_error / max(1, count)}


def save_checkpoint(
    path: Path,
    agent: TD3BCAgent,
    *,
    epoch: int,
    actor_config: dict,
    train_config: dict,
    metrics: dict,
) -> None:
    torch.save(
        {
            "algorithm": "td3_bc",
            "epoch": int(epoch),
            "actor_config": actor_config,
            "train_config": train_config,
            "metrics": metrics,
            "actor_state_dict": agent.actor.state_dict(),
            "actor_target_state_dict": agent.actor_target.state_dict(),
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
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = RLTransitionDataset(args.transitions)
    train_indices, validation_indices = grouped_split(
        dataset, args.validation_ratio, args.seed
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        shuffle=True,
        generator=generator,
        **loader_options,
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices), shuffle=False, **loader_options
    )
    actor, bc_payload = load_bc_actor(
        args.bc_checkpoint.expanduser().resolve(),
        action_max_steer_command=args.action_max_steer_command,
        device=device,
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
    )
    agent = TD3BCAgent(actor, device=device, config=config)
    actor_config = {
        "model_type": "resnet18_lidar",
        "bc_max_steer_command": float(bc_payload.get("max_steer_deg", 100.0)),
        "action_max_steer_command": args.action_max_steer_command,
        "input_width": 160,
        "input_height": 90,
        "lidar_points": 360,
    }
    train_config = {
        **vars(args),
        "bc_checkpoint": str(args.bc_checkpoint.expanduser().resolve()),
        "output_dir": str(output_dir),
        "device": str(device),
        "dataset_rows": len(dataset),
        "train_rows": len(train_indices),
        "validation_rows": len(validation_indices),
        "td3_bc": asdict(config),
    }
    for key, value in list(train_config.items()):
        if isinstance(value, Path):
            train_config[key] = str(value)
    with (output_dir / "train_config.json").open("w", encoding="utf-8") as handle:
        json.dump(train_config, handle, indent=2, sort_keys=True)

    history_path = output_dir / "metrics.csv"
    history_fields = [
        "epoch",
        "critic_loss",
        "actor_loss",
        "bc_loss",
        "q_scale",
        "validation_bc_mse",
        "elapsed_sec",
    ]
    best_metric = float("inf")
    start = time.monotonic()
    with history_path.open("w", encoding="utf-8", newline="") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=history_fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            sums = {key: 0.0 for key in ("critic_loss", "actor_loss", "bc_loss", "q_scale")}
            counts = {key: 0 for key in sums}
            agent.actor.train()
            agent.actor.disable_dropout()
            for batch in train_loader:
                metrics = agent.update(batch)
                for key, value in metrics.items():
                    if np.isfinite(value):
                        sums[key] += value
                        counts[key] += 1
            row = {
                "epoch": epoch,
                **{
                    key: sums[key] / max(1, counts[key])
                    for key in sums
                },
                **validation_metrics(agent, validation_loader),
                "elapsed_sec": time.monotonic() - start,
            }
            writer.writerow(row)
            history_file.flush()
            save_checkpoint(
                output_dir / "td3_bc_latest.pth",
                agent,
                epoch=epoch,
                actor_config=actor_config,
                train_config=train_config,
                metrics=row,
            )
            selection_metric = row["validation_bc_mse"]
            if not np.isfinite(selection_metric):
                selection_metric = row["critic_loss"]
            if selection_metric < best_metric:
                best_metric = selection_metric
                save_checkpoint(
                    output_dir / "td3_bc_best.pth",
                    agent,
                    epoch=epoch,
                    actor_config=actor_config,
                    train_config=train_config,
                    metrics=row,
                )
            print(
                f"epoch {epoch:03d}/{args.epochs}: "
                f"critic={row['critic_loss']:.5f} "
                f"actor={row['actor_loss']:.5f} "
                f"bc={row['bc_loss']:.5f} "
                f"val_bc={row['validation_bc_mse']:.5f}",
                flush=True,
            )

    example_image = torch.zeros(1, 3, 90, 160)
    example_lidar = torch.zeros(1, 2, 360)
    latest_actor = agent.actor.eval().cpu()
    latest_scripted = torch.jit.trace(
        latest_actor, (example_image, example_lidar)
    )
    latest_scripted.save(
        str(output_dir / "td3_bc_actor_latest_scripted.pt")
    )
    best_actor, _ = load_rl_actor(
        output_dir / "td3_bc_best.pth", device="cpu"
    )
    best_scripted = torch.jit.trace(
        best_actor, (example_image, example_lidar)
    )
    best_scripted.save(str(output_dir / "td3_bc_actor_best_scripted.pt"))
    # Keep the historical filename as the safe default deployment artifact.
    best_scripted.save(str(output_dir / "td3_bc_actor_scripted.pt"))
    print(f"wrote TD3+BC policy: {output_dir}")


if __name__ == "__main__":
    main()
