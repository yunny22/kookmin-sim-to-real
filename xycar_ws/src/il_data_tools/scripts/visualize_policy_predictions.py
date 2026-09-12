#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize policy prediction errors.")
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--random-n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    top_dir = output_dir / "top_errors"
    random_dir = output_dir / "random_examples"
    top_dir.mkdir(parents=True, exist_ok=True)
    random_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(Path(args.predictions_csv).expanduser().resolve())
    rows_sorted = sorted(rows, key=lambda row: abs(float(row["error_deg"])), reverse=True)
    save_debug_images(rows_sorted[: args.top_n], top_dir)

    rng = random.Random(args.seed)
    random_rows = rows[:]
    rng.shuffle(random_rows)
    save_debug_images(random_rows[: args.random_n], random_dir)
    save_plots(rows, output_dir)
    print(f"saved visualizations: {output_dir}")


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def save_debug_images(rows: List[Dict[str, str]], output_dir: Path) -> None:
    import cv2

    for index, row in enumerate(rows, start=1):
        image = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
        if image is None:
            continue
        lines = [
            f"target: {float(row['target_angle_deg']):.2f} deg",
            f"pred:   {float(row['pred_angle_deg']):.2f} deg",
            f"error:  {float(row['error_deg']):.2f} deg",
            f"label:  {row.get('mission_label', '')}",
            f"session:{row.get('session_id', '')}",
        ]
        if row.get("phase", "") not in {"", None}:
            lines.append(f"phase:  {float(row.get('phase') or 0.0):.2f}")
        overlay_text(image, lines)
        safe_session = sanitize(row.get("session_id", "session"))
        out_name = f"{index:04d}_{safe_session}_{abs(float(row['error_deg'])):.2f}.jpg"
        cv2.imwrite(str(output_dir / out_name), image)


def overlay_text(image, lines: List[str]) -> None:
    import cv2

    x, y = 12, 24
    for line in lines:
        cv2.putText(
            image,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 22


def save_plots(rows: List[Dict[str, str]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARN matplotlib is unavailable; skipping plots.")
        return

    target = [float(row["target_angle_deg"]) for row in rows]
    pred = [float(row["pred_angle_deg"]) for row in rows]
    error = [float(row["error_deg"]) for row in rows]

    plt.figure(figsize=(12, 4))
    plt.plot(target, label="target", linewidth=1)
    plt.plot(pred, label="prediction", linewidth=1)
    plt.legend()
    plt.xlabel("sample")
    plt.ylabel("angle deg")
    plt.tight_layout()
    plt.savefig(output_dir / "angle_target_vs_prediction.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.hist(error, bins=60)
    plt.xlabel("error deg")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(output_dir / "error_histogram.png", dpi=160)
    plt.close()

    if any(row.get("phase", "") not in {"", None} for row in rows):
        phase_bins = [
            ("shift_out", 0.0, 0.2),
            ("pass", 0.2, 0.6),
            ("return", 0.6, 1.0),
        ]
        names = []
        values = []
        for name, low, high in phase_bins:
            vals = [
                abs(float(row["error_deg"]))
                for row in rows
                if low <= float(row.get("phase") or 0.0) <= high
            ]
            names.append(name)
            values.append(sum(vals) / len(vals) if vals else 0.0)
        plt.figure(figsize=(6, 4))
        plt.bar(names, values)
        plt.ylabel("MAE deg")
        plt.tight_layout()
        plt.savefig(output_dir / "phase_bin_error.png", dpi=160)
        plt.close()


def sanitize(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)[:80]


if __name__ == "__main__":
    main()
