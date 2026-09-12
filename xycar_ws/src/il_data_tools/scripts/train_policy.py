#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


POLICY_DEFAULT_MODELS = {
    "drive": "resnet18_lidar",
    "cone": "resnet18_lidar",
    "overtake": "pilotnet_phase",
}
SUPPORTED_MODEL_TYPES = (
    "pilotnet",
    "mobilenet_v3_small",
    "resnet18",
    "vit_tiny",
    "pilotnet_phase",
    "mobilenet_v3_small_phase",
    "resnet18_phase",
    "vit_tiny_phase",
    "resnet18_lidar",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified offline trainer for steering-only imitation policies."
    )
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policy-name", choices=["drive", "cone", "overtake"], required=True)
    parser.add_argument("--model-type", choices=SUPPORTED_MODEL_TYPES, default=None)
    parser.add_argument("--input-width", type=int, default=160)
    parser.add_argument("--input-height", type=int, default=90)
    parser.add_argument("--max-steer-deg", type=float, default=100.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--use-phase", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help=(
            "Initialize model weights from a compatible project .pth checkpoint. "
            "Optimizer and scheduler state are intentionally not restored."
        ),
    )
    parser.add_argument("--enable-flip", action="store_true")
    parser.add_argument(
        "--canonical-input",
        action="store_true",
        help="Train on fixed-color canonical BEV images instead of raw camera RGB.",
    )
    parser.add_argument(
        "--lane-dropout-probability",
        type=float,
        default=0.30,
        help="Canonical mode only: probability of hiding one white boundary per sample.",
    )
    parser.add_argument("--recovery-weight", type=float, default=1.5)
    parser.add_argument("--steer-weight-gain", type=float, default=2.0)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scheduler", choices=["plateau", "cosine", "none"], default="plateau")
    parser.add_argument(
        "--mark-final",
        action="store_true",
        help="Also copy scripted output to drive/cone/overtake_policy_scripted.pt.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    report = run_training(args)
    print(f"best_val_loss={report['best_val_loss']:.6f}")
    print(f"checkpoint={report['best_checkpoint']}")
    print(f"torchscript={report['scripted_model']}")


def run_training(args: argparse.Namespace) -> Dict[str, object]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    from policy_dataset import PolicyCsvDataset
    from image_preprocessing import preprocessing_contract
    from policy_models import create_policy_model, model_uses_lidar, model_uses_phase

    if args.pretrained:
        raise ValueError(
            "--pretrained is intentionally disabled: ImageNet normalization is not "
            "implemented consistently in training, evaluation, and runtime inference."
        )
    if not 0.0 <= args.lane_dropout_probability <= 1.0:
        raise ValueError("--lane-dropout-probability must be in [0, 1]")
    seed_everything(args.seed, torch)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_type = args.model_type or POLICY_DEFAULT_MODELS[args.policy_name]
    use_phase = bool(args.use_phase or model_uses_phase(model_type))
    use_lidar = model_uses_lidar(model_type)
    if args.policy_name == "overtake" and not use_phase:
        print("WARN overtake policy is usually phase-conditioned; continuing without phase.")
    if "vit_tiny" in model_type:
        print("WARN vit_tiny is experimental only and must not be the default final model.")

    device = choose_device(args.device, torch)
    if args.device == "cuda" and device.type != "cuda":
        print("WARN CUDA requested but unavailable; falling back to CPU.")

    train_dataset = PolicyCsvDataset(
        args.train_csv,
        input_width=args.input_width,
        input_height=args.input_height,
        max_steer_deg=args.max_steer_deg,
        use_phase=use_phase,
        use_lidar=use_lidar,
        enable_augment=True,
        enable_flip=args.enable_flip and args.policy_name in {"drive", "cone"},
        canonical_input=args.canonical_input,
        lane_dropout_probability=(
            args.lane_dropout_probability if args.canonical_input else 0.0
        ),
    )
    val_dataset = PolicyCsvDataset(
        args.val_csv,
        input_width=args.input_width,
        input_height=args.input_height,
        max_steer_deg=args.max_steer_deg,
        use_phase=use_phase,
        use_lidar=use_lidar,
        enable_augment=False,
        enable_flip=False,
        canonical_input=args.canonical_input,
    )
    shared_sessions = train_dataset.session_ids() & val_dataset.session_ids()
    if shared_sessions:
        print(
            "WARN train and val share session_id values; split should be session-based: "
            + ", ".join(sorted(shared_sessions))
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = create_policy_model(
        model_type,
        input_width=args.input_width,
        input_height=args.input_height,
        use_phase=use_phase,
        pretrained=args.pretrained,
    ).to(device)
    init_checkpoint_path = initialize_from_checkpoint(
        model,
        getattr(args, "init_checkpoint", None),
        policy_name=args.policy_name,
        model_type=model_type,
        input_width=args.input_width,
        input_height=args.input_height,
        max_steer_deg=args.max_steer_deg,
        use_phase=use_phase,
        use_lidar=use_lidar,
        torch_module=torch,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = create_scheduler(args.scheduler, optimizer, args.epochs)
    loss_fn = nn.SmoothL1Loss(reduction="none")

    train_config = vars(args).copy()
    train_config.update(
        {
            "model_type_resolved": model_type,
            "use_phase_resolved": use_phase,
            "use_lidar_resolved": use_lidar,
            "device_resolved": str(device),
            "init_checkpoint_resolved": (
                str(init_checkpoint_path) if init_checkpoint_path is not None else None
            ),
            "selection_warning": (
                "Final model must be selected using offline eval, Jetson latency, "
                "low-speed closed-loop driving, oscillation, and safety compatibility."
            ),
            "safety": "Model outputs steering only. Speed and safety remain rule-based.",
            "preprocessing": preprocessing_contract(
                args.input_width,
                args.input_height,
                canonical_input=args.canonical_input,
            ),
        }
    )
    write_json(output_dir / "train_config.json", train_config)

    best_val = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history: List[Dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            loss_fn,
            use_phase,
            use_lidar,
            args.max_steer_deg,
            args.steer_weight_gain,
            args.recovery_weight,
            optimizer,
        )
        val_metrics, _ = evaluate_loader(
            model,
            val_loader,
            device,
            use_phase,
            use_lidar,
            args.max_steer_deg,
            args.steer_weight_gain,
            args.recovery_weight,
            loss_fn,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["val_loss"],
            "val_mae_deg": val_metrics["val_mae_deg"],
            "val_rmse_deg": val_metrics["val_rmse_deg"],
            "val_max_error_deg": val_metrics["val_max_error_deg"],
            "steering_bin_mae": val_metrics["steering_bin_mae"],
            "phase_bin_mae": val_metrics.get("phase_bin_mae", {}),
        }
        history.append(row)
        print(
            "epoch={epoch} train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
            "val_mae_deg={val_mae_deg:.3f} val_rmse_deg={val_rmse_deg:.3f}".format(
                **row
            )
        )

        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(val_metrics["val_loss"])
            else:
                scheduler.step()

        if val_metrics["val_loss"] < best_val:
            best_val = val_metrics["val_loss"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stop_patience:
                print(f"early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics, predictions = evaluate_loader(
        model,
        val_loader,
        device,
        use_phase,
        use_lidar,
        args.max_steer_deg,
        args.steer_weight_gain,
        args.recovery_weight,
        loss_fn,
    )
    metrics.update(
        {
            "policy_name": args.policy_name,
            "model_type": model_type,
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "history": history,
        }
    )

    checkpoint_path = output_dir / f"{args.policy_name}_{model_type}_best.pth"
    torch.save(
        {
            "policy_name": args.policy_name,
            "model_type": model_type,
            "use_phase": use_phase,
            "use_lidar": use_lidar,
            "input_width": args.input_width,
            "input_height": args.input_height,
            "max_steer_deg": args.max_steer_deg,
            "state_dict": model.state_dict(),
            "train_config": train_config,
            "metrics": metrics,
        },
        checkpoint_path,
    )
    scripted_path = output_dir / f"{args.policy_name}_{model_type}_scripted.pt"
    export_torchscript_model(
        model,
        scripted_path,
        args.input_width,
        args.input_height,
        use_phase,
        use_lidar,
        device,
    )
    if args.mark_final:
        final_path = output_dir / f"{args.policy_name}_policy_scripted.pt"
        shutil.copyfile(scripted_path, final_path)

    write_json(output_dir / "metrics.json", metrics)
    write_predictions(output_dir / "val_predictions.csv", predictions)

    return {
        "best_val_loss": best_val,
        "best_checkpoint": str(checkpoint_path),
        "scripted_model": str(scripted_path),
        "metrics": metrics,
    }


def initialize_from_checkpoint(
    model,
    checkpoint_value: Optional[str],
    *,
    policy_name: str,
    model_type: str,
    input_width: int,
    input_height: int,
    max_steer_deg: float,
    use_phase: bool,
    use_lidar: bool,
    torch_module,
) -> Optional[Path]:
    if not checkpoint_value:
        return None

    checkpoint_path = Path(checkpoint_value).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"initial checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch_module.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch_module.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(
            "--init-checkpoint must be a project .pth checkpoint containing state_dict; "
            "a TorchScript .pt deployment model cannot be fine-tuned"
        )

    expected = {
        "policy_name": policy_name,
        "model_type": model_type,
        "input_width": int(input_width),
        "input_height": int(input_height),
        "use_phase": bool(use_phase),
        "use_lidar": bool(use_lidar),
    }
    mismatches = []
    for key, wanted in expected.items():
        if key in checkpoint and checkpoint[key] != wanted:
            mismatches.append(f"{key}: checkpoint={checkpoint[key]!r}, requested={wanted!r}")
    if "max_steer_deg" in checkpoint and not math.isclose(
        float(checkpoint["max_steer_deg"]),
        float(max_steer_deg),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        mismatches.append(
            "max_steer_deg: "
            f"checkpoint={checkpoint['max_steer_deg']!r}, requested={max_steer_deg!r}"
        )
    if mismatches:
        raise ValueError(
            "initial checkpoint is incompatible with this training run:\n- "
            + "\n- ".join(mismatches)
        )

    model.load_state_dict(checkpoint["state_dict"], strict=True)
    print(f"initialized model weights from checkpoint: {checkpoint_path}")
    return checkpoint_path


def run_epoch(
    model,
    loader,
    device,
    loss_fn,
    use_phase: bool,
    use_lidar: bool,
    max_steer_deg: float,
    steer_weight_gain: float,
    recovery_weight: float,
    optimizer,
) -> Dict[str, float]:
    import torch

    model.train()
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        image = batch["image"].to(device)
        target = batch["target"].to(device)
        phase = batch["phase"].to(device)
        lidar = batch["lidar"].to(device)
        labels = batch["metadata"]["mission_label"]
        pred = run_model(model, image, lidar, phase, use_lidar, use_phase)
        weights = sample_weights(target, labels, steer_weight_gain, recovery_weight, torch).to(device)
        loss_items = loss_fn(pred, target)
        loss = (loss_items * weights).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        count = target.numel()
        total_loss += float(loss.detach().cpu()) * count
        total_count += count
    return {"loss": total_loss / max(1, total_count)}


def evaluate_loader(
    model,
    loader,
    device,
    use_phase: bool,
    use_lidar: bool,
    max_steer_deg: float,
    steer_weight_gain: float,
    recovery_weight: float,
    loss_fn,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    import torch

    model.eval()
    total_loss = 0.0
    total_count = 0
    predictions: List[Dict[str, object]] = []
    errors_deg = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            target = batch["target"].to(device)
            phase = batch["phase"].to(device)
            lidar = batch["lidar"].to(device)
            labels = batch["metadata"]["mission_label"]
            pred = run_model(model, image, lidar, phase, use_lidar, use_phase)
            weights = sample_weights(target, labels, steer_weight_gain, recovery_weight, torch).to(device)
            loss = (loss_fn(pred, target) * weights).mean()
            total_loss += float(loss.detach().cpu()) * target.numel()
            total_count += target.numel()

            pred_np = pred.detach().cpu().numpy().reshape(-1)
            target_np = target.detach().cpu().numpy().reshape(-1)
            phase_np = phase.detach().cpu().numpy().reshape(-1)
            metadata = batch["metadata"]
            for idx, pred_norm in enumerate(pred_np):
                target_norm = float(target_np[idx])
                pred_angle = float(pred_norm) * max_steer_deg
                target_angle = target_norm * max_steer_deg
                error = pred_angle - target_angle
                errors_deg.append(error)
                predictions.append(
                    {
                        "image_path": metadata["image_path"][idx],
                        "target_angle_deg": target_angle,
                        "pred_angle_deg": pred_angle,
                        "error_deg": error,
                        "target_steer_norm": target_norm,
                        "pred_steer_norm": float(pred_norm),
                        "mission_label": metadata["mission_label"][idx],
                        "session_id": metadata["session_id"][idx],
                        "phase": float(phase_np[idx]),
                        "timestamp_ns": metadata["timestamp_ns"][idx],
                    }
                )

    metrics = compute_prediction_metrics(predictions, max_steer_deg, include_phase=use_phase)
    metrics["val_loss"] = total_loss / max(1, total_count)
    return metrics, predictions


def sample_weights(target, labels, steer_weight_gain, recovery_weight, torch_module):
    weights = 1.0 + steer_weight_gain * torch_module.abs(target)
    label_weights = [
        recovery_weight if str(label) == "recovery" else 1.0 for label in labels
    ]
    label_tensor = torch_module.tensor(label_weights, dtype=target.dtype).view(-1, 1)
    return weights.detach().cpu() * label_tensor


def compute_prediction_metrics(
    predictions: List[Dict[str, object]],
    max_steer_deg: float,
    include_phase: bool,
) -> Dict[str, object]:
    abs_errors = [abs(float(row["error_deg"])) for row in predictions]
    squared = [float(row["error_deg"]) ** 2 for row in predictions]
    metrics = {
        "val_mae_deg": mean(abs_errors),
        "val_rmse_deg": math.sqrt(mean(squared)),
        "val_max_error_deg": max(abs_errors) if abs_errors else 0.0,
        "steering_bin_mae": steering_bin_mae(predictions),
    }
    if include_phase:
        metrics["phase_bin_mae"] = phase_bin_mae(predictions)
    return metrics


def steering_bin_mae(predictions: List[Dict[str, object]]) -> Dict[str, float]:
    bins = {
        "left_hard": (-1.0, -0.5),
        "left": (-0.5, -0.15),
        "center": (-0.15, 0.15),
        "right": (0.15, 0.5),
        "right_hard": (0.5, 1.0),
    }
    result = {}
    for name, (low, high) in bins.items():
        vals = [
            abs(float(row["error_deg"]))
            for row in predictions
            if low <= float(row["target_steer_norm"]) <= high
        ]
        result[name] = mean(vals)
    return result


def phase_bin_mae(predictions: List[Dict[str, object]]) -> Dict[str, float]:
    bins = {
        "shift_out_0.0_0.2": (0.0, 0.2),
        "pass_0.2_0.6": (0.2, 0.6),
        "return_0.6_1.0": (0.6, 1.0),
    }
    result = {}
    for name, (low, high) in bins.items():
        vals = [
            abs(float(row["error_deg"]))
            for row in predictions
            if low <= float(row.get("phase", 0.0)) <= high
        ]
        result[name] = mean(vals)
    return result


def export_torchscript_model(
    model,
    output_path: Path,
    input_width: int,
    input_height: int,
    use_phase: bool,
    use_lidar: bool,
    device,
) -> None:
    import torch

    model.eval()
    image = torch.zeros(1, 3, input_height, input_width, device=device)
    lidar = torch.zeros(1, 2, 360, device=device)
    with torch.no_grad():
        if use_lidar:
            scripted = torch.jit.trace(model, (image, lidar))
        elif use_phase:
            phase = torch.zeros(1, 1, device=device)
            scripted = torch.jit.trace(model, (image, phase))
        else:
            scripted = torch.jit.trace(model, image)
    scripted.save(str(output_path))


def run_model(model, image, lidar, phase, use_lidar: bool, use_phase: bool):
    if use_lidar:
        return model(image, lidar)
    if use_phase:
        return model(image, phase)
    return model(image)


def create_scheduler(name: str, optimizer, epochs: int):
    if name == "none":
        return None
    import torch

    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))


def choose_device(name: str, torch_module):
    if name == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    if name == "cuda" and not torch_module.cuda.is_available():
        return torch_module.device("cpu")
    return torch_module.device(name)


def seed_everything(seed: int, torch_module) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def write_predictions(path: Path, predictions: List[Dict[str, object]]) -> None:
    columns = [
        "image_path",
        "target_angle_deg",
        "pred_angle_deg",
        "error_deg",
        "target_steer_norm",
        "pred_steer_norm",
        "mission_label",
        "session_id",
        "phase",
        "timestamp_ns",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in predictions:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_json(path: Path, data: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


if __name__ == "__main__":
    main()
