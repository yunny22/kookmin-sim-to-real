#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

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
    parser = argparse.ArgumentParser(description="Evaluate a TorchScript steering policy.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-type", choices=SUPPORTED_MODEL_TYPES, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-width", type=int, default=160)
    parser.add_argument("--input-height", type=int, default=90)
    parser.add_argument("--max-steer-deg", type=float, default=100.0)
    parser.add_argument("--use-phase", action="store_true")
    parser.add_argument("--canonical-input", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    import torch
    from torch.utils.data import DataLoader

    from policy_dataset import PolicyCsvDataset
    from policy_models import model_uses_lidar, model_uses_phase

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    use_phase = args.use_phase or model_uses_phase(args.model_type)
    use_lidar = model_uses_lidar(args.model_type)
    device = choose_device(args.device, torch)
    if args.device == "cuda" and device.type != "cuda":
        print("WARN CUDA requested but unavailable; falling back to CPU.")

    dataset = PolicyCsvDataset(
        args.csv,
        input_width=args.input_width,
        input_height=args.input_height,
        max_steer_deg=args.max_steer_deg,
        use_phase=use_phase,
        use_lidar=use_lidar,
        enable_augment=False,
        canonical_input=args.canonical_input,
    )
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    model = torch.jit.load(str(Path(args.model).expanduser().resolve()), map_location=device)
    model.eval()

    predictions = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            phase = batch["phase"].to(device)
            lidar = batch["lidar"].to(device)
            target = batch["target"]
            if use_lidar:
                pred = model(image, lidar)
            elif use_phase:
                pred = model(image, phase)
            else:
                pred = model(image)
            pred_np = pred.detach().cpu().numpy().reshape(-1)
            target_np = target.detach().cpu().numpy().reshape(-1)
            phase_np = phase.detach().cpu().numpy().reshape(-1)
            metadata = batch["metadata"]
            for idx, pred_norm in enumerate(pred_np):
                target_norm = float(target_np[idx])
                pred_angle = float(pred_norm) * args.max_steer_deg
                target_angle = target_norm * args.max_steer_deg
                row = {
                    "image_path": metadata["image_path"][idx],
                    "target_angle_deg": target_angle,
                    "pred_angle_deg": pred_angle,
                    "error_deg": pred_angle - target_angle,
                    "target_steer_norm": target_norm,
                    "pred_steer_norm": float(pred_norm),
                    "mission_label": metadata["mission_label"][idx],
                    "session_id": metadata["session_id"][idx],
                    "phase": float(phase_np[idx]),
                    "timestamp_ns": metadata["timestamp_ns"][idx],
                    "speed": float(metadata["speed"][idx]),
                }
                predictions.append(row)

    metrics = compute_metrics(predictions, include_phase=use_phase)
    metrics.update(
        {
            "csv": str(Path(args.csv).expanduser().resolve()),
            "model": str(Path(args.model).expanduser().resolve()),
            "model_type": args.model_type,
            "use_phase": use_phase,
            "use_lidar": use_lidar,
            "canonical_input": args.canonical_input,
            "model_size_mb": Path(args.model).expanduser().resolve().stat().st_size / (1024 * 1024),
        }
    )
    write_predictions(output_dir / "predictions.csv", predictions)
    write_json(output_dir / "eval_metrics.json", metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def compute_metrics(predictions: List[Dict[str, object]], include_phase: bool) -> Dict[str, object]:
    abs_errors = [abs(float(row["error_deg"])) for row in predictions]
    squared = [float(row["error_deg"]) ** 2 for row in predictions]
    return {
        "rows": len(predictions),
        "val_mae_deg": mean(abs_errors),
        "val_rmse_deg": math.sqrt(mean(squared)),
        "max_error_deg": max(abs_errors) if abs_errors else 0.0,
        "steering_bin_mae": steering_bin_mae(predictions),
        "label_wise_mae": grouped_mae(predictions, "mission_label"),
        "session_wise_mae": grouped_mae(predictions, "session_id"),
        "speed_bin_mae": speed_bin_mae(predictions),
        "phase_bin_mae": phase_bin_mae(predictions) if include_phase else {},
    }


def steering_bin_mae(rows):
    bins = {
        "left_hard": (-1.0, -0.5),
        "left": (-0.5, -0.15),
        "center": (-0.15, 0.15),
        "right": (0.15, 0.5),
        "right_hard": (0.5, 1.0),
    }
    return {
        name: mean(
            [
                abs(float(row["error_deg"]))
                for row in rows
                if low <= float(row["target_steer_norm"]) <= high
            ]
        )
        for name, (low, high) in bins.items()
    }


def speed_bin_mae(rows):
    bins = {
        "stopped_or_slow": (0.0, 3.0),
        "medium": (3.0, 10.0),
        "fast": (10.0, 999.0),
    }
    return {
        name: mean(
            [
                abs(float(row["error_deg"]))
                for row in rows
                if low <= abs(float(row.get("speed", 0.0))) < high
            ]
        )
        for name, (low, high) in bins.items()
    }


def phase_bin_mae(rows):
    if not rows or all(float(row.get("phase", 0.0)) == 0.0 for row in rows):
        return {}
    bins = {
        "shift_out_0.0_0.2": (0.0, 0.2),
        "pass_0.2_0.6": (0.2, 0.6),
        "return_0.6_1.0": (0.6, 1.0),
    }
    return {
        name: mean(
            [
                abs(float(row["error_deg"]))
                for row in rows
                if low <= float(row.get("phase", 0.0)) <= high
            ]
        )
        for name, (low, high) in bins.items()
    }


def grouped_mae(rows, key: str):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, ""))].append(abs(float(row["error_deg"])))
    return {name: mean(values) for name, values in sorted(groups.items())}


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


def choose_device(name: str, torch_module):
    if name == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    if name == "cuda" and not torch_module.cuda.is_available():
        return torch_module.device("cpu")
    return torch_module.device(name)


def mean(values) -> float:
    return float(sum(values) / len(values)) if values else 0.0


if __name__ == "__main__":
    main()
