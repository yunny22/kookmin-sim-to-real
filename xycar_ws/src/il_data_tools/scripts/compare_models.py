#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare policy model eval and benchmark reports.")
    parser.add_argument("--eval-metrics", action="append", default=[])
    parser.add_argument("--benchmark-json", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_rows = [load_json(Path(path).expanduser().resolve()) for path in args.eval_metrics]
    bench_rows = [load_json(Path(path).expanduser().resolve()) for path in args.benchmark_json]
    rows = merge_rows(eval_rows, bench_rows)
    add_scores(rows)
    write_csv(output_dir / "model_comparison.csv", rows)
    write_markdown(output_dir / "model_comparison.md", rows)
    print(f"wrote model comparison: {output_dir}")


def merge_rows(eval_rows: List[Dict], bench_rows: List[Dict]) -> List[Dict]:
    by_model = {}
    for item in eval_rows:
        key = Path(item.get("model", item.get("checkpoint_path", "unknown"))).stem
        by_model.setdefault(key, {}).update(
            {
                "policy_name": infer_policy(key),
                "model_type": item.get("model_type", infer_model_type(key)),
                "val_mae_deg": item.get("val_mae_deg", item.get("mae_deg", "")),
                "val_rmse_deg": item.get("val_rmse_deg", ""),
                "max_error_deg": item.get("max_error_deg", item.get("val_max_error_deg", "")),
                "model_size_mb": item.get("model_size_mb", ""),
                "notes": "",
            }
        )
    for item in bench_rows:
        key = Path(item.get("model", "unknown")).stem
        row = by_model.setdefault(
            key,
            {
                "policy_name": infer_policy(key),
                "model_type": infer_model_type(key),
                "val_mae_deg": "",
                "val_rmse_deg": "",
                "max_error_deg": "",
                "model_size_mb": "",
                "notes": "",
            },
        )
        row["p95_latency_ms"] = item.get("p95_latency_ms", item.get("p95_ms", ""))
        row["fps"] = item.get("fps", "")
        row["notes"] = item.get("guidance", row.get("notes", ""))
        if not row.get("model_size_mb"):
            model_path = item.get("model", "")
            if model_path and Path(model_path).is_file():
                row["model_size_mb"] = Path(model_path).stat().st_size / (1024 * 1024)
    return list(by_model.values())


def add_scores(rows: List[Dict]) -> None:
    fields = [
        ("val_mae_deg", 0.35),
        ("p95_latency_ms", 0.25),
        ("max_error_deg", 0.20),
        ("model_size_mb", 0.10),
    ]
    maxima = {
        field: max([to_float(row.get(field)) for row in rows] + [1.0])
        for field, _ in fields
    }
    for row in rows:
        score = 0.0
        for field, weight in fields:
            value = to_float(row.get(field))
            score += weight * (value / maxima[field] if maxima[field] > 0 else 1.0)
        penalty = 1.0 if missing_required(row) else 0.0
        score += 0.10 * penalty
        row["score"] = score
        row["penalty_for_missing_phase_bins_or_failures"] = penalty
    rows.sort(key=lambda row: row["score"])


def write_csv(path: Path, rows: List[Dict]) -> None:
    columns = [
        "policy_name",
        "model_type",
        "val_mae_deg",
        "val_rmse_deg",
        "max_error_deg",
        "p95_latency_ms",
        "fps",
        "model_size_mb",
        "penalty_for_missing_phase_bins_or_failures",
        "score",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_markdown(path: Path, rows: List[Dict]) -> None:
    lines = [
        "# Model Comparison",
        "",
        "Final model should not be selected by validation loss alone.",
        "",
        "Final selection must consider offline accuracy, Jetson Orin Nano latency, "
        "closed-loop low-speed driving stability, steering oscillation, safety "
        "compatibility, and penalty risk.",
        "",
        "| policy_name | model_type | val_mae_deg | val_rmse_deg | max_error_deg | p95_latency_ms | fps | model_size_mb | score | notes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {policy_name} | {model_type} | {val_mae_deg} | {val_rmse_deg} | "
            "{max_error_deg} | {p95_latency_ms} | {fps} | {model_size_mb} | "
            "{score:.4f} | {notes} |".format(
                policy_name=row.get("policy_name", ""),
                model_type=row.get("model_type", ""),
                val_mae_deg=fmt(row.get("val_mae_deg")),
                val_rmse_deg=fmt(row.get("val_rmse_deg")),
                max_error_deg=fmt(row.get("max_error_deg")),
                p95_latency_ms=fmt(row.get("p95_latency_ms")),
                fps=fmt(row.get("fps")),
                model_size_mb=fmt(row.get("model_size_mb")),
                score=float(row.get("score", 0.0)),
                notes=row.get("notes", ""),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def infer_policy(stem: str) -> str:
    for policy in ["drive", "cone", "overtake"]:
        if policy in stem:
            return policy
    return ""


def infer_model_type(stem: str) -> str:
    for model_type in [
        "mobilenet_v3_small_phase",
        "mobilenet_v3_small",
        "pilotnet_phase",
        "resnet18_phase",
        "resnet18_lidar",
        "vit_tiny_phase",
        "pilotnet",
        "resnet18",
        "vit_tiny",
    ]:
        if model_type in stem:
            return model_type
    return ""


def missing_required(row: Dict) -> bool:
    return not row.get("val_mae_deg") or not row.get("p95_latency_ms")


def to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 1e9


def fmt(value) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return ""


if __name__ == "__main__":
    main()
