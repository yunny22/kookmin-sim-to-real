from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from il_data_tools.runtime_preprocessing import preprocess_bgr_image
from xycar_rl.camera_speed_models import (
    CameraSpeedActor,
    CompactCameraSpeedActor,
    DEFAULT_MAX_SPEED_COMMAND,
    DEFAULT_MIN_SPEED_COMMAND,
    normalize_speed_command,
    initialize_temporal_actor,
    load_camera_speed_actor,
)
from xycar_rl.models import load_rl_actor
from xycar_rl.transition_dataset import find_transition_csvs
from xycar_rl.train_td3_bc import resolve_device


def speed_target_from_transition(
    steering_norm: float,
    cross_track_error_m: float,
    heading_error_rad: float,
    *,
    min_speed_command: float = DEFAULT_MIN_SPEED_COMMAND,
    max_speed_command: float = DEFAULT_MAX_SPEED_COMMAND,
) -> float:
    curve = np.clip((abs(float(steering_norm)) - 0.08) / 0.72, 0.0, 1.0)
    curve = curve * curve * (3.0 - 2.0 * curve)
    recovery = max(
        np.clip(abs(float(cross_track_error_m)) / 0.30, 0.0, 1.0),
        np.clip(abs(float(heading_error_rad)) / 1.0, 0.0, 1.0),
    )
    slow_fraction = max(float(curve), float(recovery))
    return float(
        max_speed_command
        - slow_fraction * (max_speed_command - min_speed_command)
    )


def steering_sample_weight(
    steering_norm: float,
    gain: float,
    straight_weight: float = 1.0,
    straight_threshold: float = 0.12,
) -> float:
    """Balance sparse curves while retaining precise straight-line labels."""
    magnitude = abs(float(steering_norm))
    weight = 1.0 + max(0.0, float(gain)) * magnitude**2
    if magnitude <= max(0.0, float(straight_threshold)):
        weight *= max(0.0, float(straight_weight))
    return weight


def transition_rows_are_contiguous(
    previous: dict[str, str], current: dict[str, str]
) -> bool:
    if previous.get("episode_id", "0") != current.get("episode_id", "0"):
        return False
    try:
        return (
            int(previous["step_id"]) + 1 == int(current["step_id"])
            and int(previous["next_timestamp_ns"])
            == int(current["state_timestamp_ns"])
        )
    except (KeyError, TypeError, ValueError):
        return False


class CameraSpeedBCDataset(Dataset):
    def __init__(
        self,
        paths: list[str | Path],
        *,
        min_speed_command: float,
        max_speed_command: float,
        speed_target_source: str = "heuristic",
        steering_weight_gain: float = 0.0,
        straight_steering_weight: float = 1.0,
        straight_steering_threshold: float = 0.12,
        temporal_frames: int = 1,
        label_lookahead_frames: int = 0,
        success_only: bool = False,
        max_abs_cross_track_error_m: float = 0.0,
        max_abs_heading_error_deg: float = 0.0,
        input_width: int = 160,
        input_height: int = 90,
    ) -> None:
        self.input_width = int(input_width)
        self.input_height = int(input_height)
        self.min_speed_command = float(min_speed_command)
        self.max_speed_command = float(max_speed_command)
        self.speed_target_source = str(speed_target_source)
        self.steering_weight_gain = max(0.0, float(steering_weight_gain))
        self.straight_steering_weight = max(
            0.0, float(straight_steering_weight)
        )
        self.straight_steering_threshold = max(
            0.0, float(straight_steering_threshold)
        )
        self.temporal_frames = int(temporal_frames)
        if self.temporal_frames not in {1, 2}:
            raise ValueError("temporal_frames must be 1 or 2")
        self.label_lookahead_frames = max(0, int(label_lookahead_frames))
        self.max_abs_cross_track_error_m = max(
            0.0, float(max_abs_cross_track_error_m)
        )
        self.max_abs_heading_error_rad = math.radians(
            max(0.0, float(max_abs_heading_error_deg))
        )
        self.rows: list[tuple[Path, dict[str, str]]] = []
        self.target_rows: list[dict[str, str]] = []
        self.previous_image_paths: list[str] = []
        for csv_path in find_transition_csvs(paths):
            with csv_path.open(newline="", encoding="utf-8-sig") as handle:
                session_rows = list(csv.DictReader(handle))
            if success_only:
                successful_episodes = {
                    row.get("episode_id", "0")
                    for row in session_rows
                    if row.get("termination_reason") == "lap_complete"
                }
                session_rows = [
                    row
                    for row in session_rows
                    if row.get("episode_id", "0") in successful_episodes
                ]
            if self.max_abs_cross_track_error_m > 0.0:
                session_rows = [
                    row
                    for row in session_rows
                    if abs(float(row.get("cross_track_error_m") or 0.0))
                    <= self.max_abs_cross_track_error_m
                ]
            if self.max_abs_heading_error_rad > 0.0:
                session_rows = [
                    row
                    for row in session_rows
                    if abs(float(row.get("heading_error_rad") or 0.0))
                    <= self.max_abs_heading_error_rad
                ]
            previous_by_episode: dict[str, tuple[dict[str, str], str]] = {}
            for row_index, row in enumerate(session_rows):
                episode_id = row.get("episode_id", "0")
                current_path = row["state_image_path"]
                target_row = row
                target_index = row_index
                for _ in range(self.label_lookahead_frames):
                    next_index = target_index + 1
                    if next_index >= len(session_rows) or not transition_rows_are_contiguous(
                        session_rows[target_index], session_rows[next_index]
                    ):
                        break
                    target_index = next_index
                    target_row = session_rows[target_index]
                self.rows.append((csv_path.parent, row))
                self.target_rows.append(target_row)
                previous_entry = previous_by_episode.get(episode_id)
                previous_path = current_path
                if (
                    previous_entry is not None
                    and transition_rows_are_contiguous(previous_entry[0], row)
                ):
                    previous_path = previous_entry[1]
                self.previous_image_paths.append(previous_path)
                previous_by_episode[episode_id] = (row, current_path)
        if not self.rows:
            raise ValueError("camera-speed BC dataset has no rows")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        root, row = self.rows[index]
        target_row = self.target_rows[index]
        image = cv2.imread(str(root / row["state_image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(root / row["state_image_path"])
        image_array = preprocess_bgr_image(
            image, self.input_width, self.input_height
        )
        if self.temporal_frames == 2:
            previous = cv2.imread(
                str(root / self.previous_image_paths[index]), cv2.IMREAD_COLOR
            )
            if previous is None:
                raise FileNotFoundError(root / self.previous_image_paths[index])
            previous_array = preprocess_bgr_image(
                previous, self.input_width, self.input_height
            )
            image_array = np.concatenate([previous_array, image_array], axis=0)
        image_tensor = torch.from_numpy(image_array)
        steering = float(target_row["action_norm"])
        if self.speed_target_source == "recorded":
            speed_command = float(target_row["speed_command"])
        else:
            speed_command = speed_target_from_transition(
                steering,
                float(target_row.get("cross_track_error_m") or 0.0),
                float(target_row.get("heading_error_rad") or 0.0),
                min_speed_command=self.min_speed_command,
                max_speed_command=self.max_speed_command,
            )
        speed_norm = normalize_speed_command(
            speed_command,
            self.min_speed_command,
            self.max_speed_command,
        )
        return {
            "image": image_tensor,
            "action": torch.tensor([steering, speed_norm], dtype=torch.float32),
            "steering_weight": torch.tensor(
                steering_sample_weight(
                    steering,
                    self.steering_weight_gain,
                    self.straight_steering_weight,
                    self.straight_steering_threshold,
                ),
                dtype=torch.float32,
            ),
        }


def grouped_split(dataset: CameraSpeedBCDataset, ratio: float, seed: int):
    groups: dict[tuple[str, str], list[int]] = {}
    for index, (root, row) in enumerate(dataset.rows):
        groups.setdefault((str(root), row.get("episode_id", "0")), []).append(index)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    validation_count = max(1, round(len(keys) * min(0.5, ratio)))
    validation_keys = set(keys[:validation_count])
    validation = [i for key in validation_keys for i in groups[key]]
    train = [i for key in keys if key not in validation_keys for i in groups[key]]
    if not train or not validation:
        raise ValueError("camera-speed dataset needs at least two episode groups")
    return train, validation


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Distill a camera-only steering and adaptive-speed actor."
    )
    parser.add_argument("--transitions", action="append", required=True)
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--initial-camera-speed-checkpoint", type=Path)
    parser.add_argument("--allow-speed-range-change", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--min-speed-command",
        type=float,
        default=DEFAULT_MIN_SPEED_COMMAND,
        help="Minimum stable command; 4 is the previously validated full-lap speed.",
    )
    parser.add_argument(
        "--max-speed-command",
        type=float,
        default=DEFAULT_MAX_SPEED_COMMAND,
        help=(
            "Actor normalization ceiling, not a deployment cap. "
            "The current high-speed contract uses 24."
        ),
    )
    parser.add_argument(
        "--speed-target-source",
        choices=["heuristic", "recorded"],
        default="heuristic",
    )
    parser.add_argument("--success-only", action="store_true")
    parser.add_argument(
        "--max-abs-cross-track-error-m",
        type=float,
        default=0.0,
        help="Drop states outside this centerline error; zero disables filtering.",
    )
    parser.add_argument(
        "--max-abs-heading-error-deg",
        type=float,
        default=0.0,
        help="Drop states outside this heading error; zero disables filtering.",
    )
    parser.add_argument(
        "--model-architecture",
        choices=("resnet18", "compact"),
        default="resnet18",
    )
    parser.add_argument(
        "--temporal-frames",
        type=int,
        choices=(1, 2),
        default=1,
        help="Use the current image alone or stack the previous and current images.",
    )
    parser.add_argument(
        "--label-lookahead-frames",
        type=int,
        default=0,
        help=(
            "Train the current image against a later action in the same episode "
            "to compensate measured command latency."
        ),
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--speed-loss-weight", type=float, default=0.5)
    parser.add_argument(
        "--steering-weight-gain",
        type=float,
        default=0.0,
        help=(
            "Quadratic weight gain for large steering labels; 4 makes a "
            "full-lock sample five times as important as a straight sample."
        ),
    )
    parser.add_argument(
        "--straight-steering-weight",
        type=float,
        default=1.0,
        help="Extra loss weight for near-zero steering labels.",
    )
    parser.add_argument(
        "--straight-steering-threshold",
        type=float,
        default=0.12,
        help="Absolute normalized steering treated as a straight sample.",
    )
    parser.add_argument(
        "--steering-loss",
        choices=["mse", "smooth_l1"],
        default="mse",
    )
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args(argv)


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    speed_weight: float,
    straight_steering_threshold: float,
):
    model.eval()
    steering_error = 0.0
    weighted_steering_error = 0.0
    steering_weight_total = 0.0
    speed_error = 0.0
    straight_steering_error = 0.0
    straight_prediction_magnitude = 0.0
    straight_count = 0
    count = 0
    for batch in loader:
        image = batch["image"].to(device)
        target = batch["action"].to(device)
        steering_weight = batch["steering_weight"].to(device)
        output = model(image)
        steering_squared_error = (output[:, 0] - target[:, 0]) ** 2
        steering_error += float(torch.sum(steering_squared_error))
        weighted_steering_error += float(
            torch.sum(steering_squared_error * steering_weight)
        )
        steering_weight_total += float(torch.sum(steering_weight))
        speed_error += float(torch.sum((output[:, 1] - target[:, 1]) ** 2))
        straight_mask = torch.abs(target[:, 0]) <= straight_steering_threshold
        if torch.any(straight_mask):
            straight_steering_error += float(
                torch.sum(steering_squared_error[straight_mask])
            )
            straight_prediction_magnitude += float(
                torch.sum(torch.abs(output[straight_mask, 0]))
            )
            straight_count += int(torch.sum(straight_mask))
        count += int(target.shape[0])
    steering_mse = steering_error / max(1, count)
    weighted_steering_mse = weighted_steering_error / max(
        1.0, steering_weight_total
    )
    speed_mse = speed_error / max(1, count)
    return {
        "validation_steering_mse": steering_mse,
        "validation_weighted_steering_mse": weighted_steering_mse,
        "validation_speed_mse": speed_mse,
        "validation_straight_steering_mse": straight_steering_error
        / max(1, straight_count),
        "validation_straight_prediction_abs_mean": straight_prediction_magnitude
        / max(1, straight_count),
        "validation_straight_rows": straight_count,
        "validation_loss": weighted_steering_mse + speed_weight * speed_mse,
    }


def main(argv=None) -> None:
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = CameraSpeedBCDataset(
        args.transitions,
        min_speed_command=args.min_speed_command,
        max_speed_command=args.max_speed_command,
        speed_target_source=args.speed_target_source,
        steering_weight_gain=args.steering_weight_gain,
        straight_steering_weight=args.straight_steering_weight,
        straight_steering_threshold=args.straight_steering_threshold,
        temporal_frames=args.temporal_frames,
        label_lookahead_frames=args.label_lookahead_frames,
        success_only=args.success_only,
        max_abs_cross_track_error_m=args.max_abs_cross_track_error_m,
        max_abs_heading_error_deg=args.max_abs_heading_error_deg,
    )
    train_indices, validation_indices = grouped_split(
        dataset, args.validation_ratio, args.seed
    )
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        Subset(dataset, train_indices), shuffle=True, **loader_args
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices), shuffle=False, **loader_args
    )
    if args.initial_camera_speed_checkpoint is not None:
        actor, teacher_payload = load_camera_speed_actor(
            args.initial_camera_speed_checkpoint.expanduser().resolve(),
            device=device,
        )
        initial_range = (
            float(teacher_payload.get("min_speed_command", args.min_speed_command)),
            float(teacher_payload.get("max_speed_command", args.max_speed_command)),
        )
        if (
            initial_range != (args.min_speed_command, args.max_speed_command)
            and not args.allow_speed_range_change
        ):
            raise ValueError(
                "initial camera-speed checkpoint range differs from training range: "
                f"{initial_range} != {(args.min_speed_command, args.max_speed_command)}"
            )
        loaded_architecture = (
            "compact" if isinstance(actor, CompactCameraSpeedActor) else "resnet18"
        )
        if loaded_architecture != args.model_architecture:
            raise ValueError(
                "initial checkpoint architecture differs from training: "
                f"{loaded_architecture} != {args.model_architecture}"
            )
        if (
            isinstance(actor, CameraSpeedActor)
            and actor.temporal_frames == 1
            and args.temporal_frames == 2
        ):
            actor = initialize_temporal_actor(actor, temporal_frames=2).to(device)
        elif actor.temporal_frames != args.temporal_frames:
            raise ValueError(
                "initial checkpoint temporal frame count differs from training: "
                f"{actor.temporal_frames} != {args.temporal_frames}"
            )
    else:
        if args.model_architecture == "compact":
            actor = CompactCameraSpeedActor(
                temporal_frames=args.temporal_frames
            ).to(device)
            teacher_payload = {}
        elif args.teacher_checkpoint is None:
            raise ValueError(
                "provide --teacher-checkpoint or --initial-camera-speed-checkpoint"
            )
        else:
            actor = CameraSpeedActor().to(device)
            teacher, teacher_payload = load_rl_actor(
                args.teacher_checkpoint.expanduser().resolve(), device=device
            )
            actor.image_encoder.load_state_dict(
                teacher.image_encoder.state_dict(), strict=True
            )
            if args.temporal_frames == 2:
                actor = initialize_temporal_actor(actor, temporal_frames=2).to(device)
    optimizer = torch.optim.AdamW(
        actor.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    best_metric = float("inf")
    best_state = None
    history = []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        actor.train()
        total_loss = 0.0
        batches = 0
        for batch in train_loader:
            image = batch["image"].to(device)
            target = batch["action"].to(device)
            steering_weight = batch["steering_weight"].to(device)
            output = actor(image)
            if args.steering_loss == "mse":
                steering_per_sample = nn.functional.mse_loss(
                    output[:, 0], target[:, 0], reduction="none"
                )
            else:
                steering_per_sample = nn.functional.smooth_l1_loss(
                    output[:, 0], target[:, 0], reduction="none"
                )
            steering_loss = torch.sum(
                steering_per_sample * steering_weight
            ) / torch.sum(steering_weight)
            speed_loss = nn.functional.smooth_l1_loss(
                output[:, 1], target[:, 1]
            )
            loss = steering_loss + args.speed_loss_weight * speed_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        metrics = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, batches),
            **evaluate(
                actor,
                validation_loader,
                device,
                args.speed_loss_weight,
                args.straight_steering_threshold,
            ),
            "elapsed_sec": time.monotonic() - started,
        }
        history.append(metrics)
        if metrics["validation_loss"] < best_metric:
            best_metric = metrics["validation_loss"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in actor.state_dict().items()
            }
        print(
            f"epoch {epoch:03d}/{args.epochs}: train={metrics['train_loss']:.5f} "
            f"steer={metrics['validation_steering_mse']:.5f} "
            f"weighted_steer={metrics['validation_weighted_steering_mse']:.5f} "
            f"straight={metrics['validation_straight_steering_mse']:.5f} "
            f"speed={metrics['validation_speed_mse']:.5f}",
            flush=True,
        )
    if best_state is None:
        raise RuntimeError("camera-speed training produced no checkpoint")
    actor.load_state_dict(best_state, strict=True)
    actor.disable_dropout()
    actor.eval().cpu()
    payload = {
        "model_type": "camera_speed_"
        + ("temporal_" if args.temporal_frames == 2 else "")
        + ("compact" if args.model_architecture == "compact" else "resnet18"),
        "temporal_frames": args.temporal_frames,
        "algorithm": "camera_speed_bc_distillation",
        "state_dict": actor.state_dict(),
        "min_speed_command": args.min_speed_command,
        "max_speed_command": args.max_speed_command,
        "teacher_checkpoint": (
            str(args.teacher_checkpoint.expanduser().resolve())
            if args.teacher_checkpoint is not None
            else None
        ),
        "initial_camera_speed_checkpoint": (
            str(args.initial_camera_speed_checkpoint.expanduser().resolve())
            if args.initial_camera_speed_checkpoint is not None
            else None
        ),
        "teacher_epoch": teacher_payload.get("epoch"),
        "dataset_rows": len(dataset),
        "train_rows": len(train_indices),
        "validation_rows": len(validation_indices),
        "history": history,
        "train_config": vars(args),
    }
    for key, value in list(payload["train_config"].items()):
        if isinstance(value, Path):
            payload["train_config"][key] = str(value)
    checkpoint = output_dir / "camera_speed_bc_best.pth"
    torch.save(payload, checkpoint)
    scripted = torch.jit.trace(
        actor, torch.zeros(1, 3 * args.temporal_frames, 90, 160)
    )
    scripted.save(str(output_dir / "camera_speed_bc_scripted.pt"))
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {key: value for key, value in payload.items() if key != "state_dict"},
            handle,
            indent=2,
            sort_keys=True,
        )
    print(f"wrote camera-speed BC actor: {output_dir}")


if __name__ == "__main__":
    main()
