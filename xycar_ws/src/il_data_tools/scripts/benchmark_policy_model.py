#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark TorchScript steering policy latency on the target device."
    )
    parser.add_argument("--model", required=True, help="TorchScript .pt path.")
    parser.add_argument("--image-height", type=int, default=90)
    parser.add_argument("--image-width", type=int, default=160)
    parser.add_argument("--phase-enabled", action="store_true")
    parser.add_argument("--lidar-enabled", action="store_true")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-json", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    import torch

    device = choose_device(args.device, torch)
    model = torch.jit.load(args.model, map_location=device)
    model.eval()
    image = torch.zeros(1, 3, args.image_height, args.image_width, device=device)
    # Phase-conditioned policies are traced with the runtime contract [B, 1].
    phase = torch.zeros(1, 1, device=device)
    lidar = torch.zeros(1, 2, 360, device=device)

    with torch.no_grad():
        for _ in range(args.warmup):
            run_once(model, image, lidar, phase, args.lidar_enabled, args.phase_enabled)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies = []
        for _ in range(args.iterations):
            started = time.perf_counter()
            run_once(model, image, lidar, phase, args.lidar_enabled, args.phase_enabled)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - started) * 1000.0)

    latencies_sorted = sorted(latencies)
    result = {
        "model": str(Path(args.model).resolve()),
        "device": str(device),
        "iterations": args.iterations,
        "mean_ms": statistics.fmean(latencies),
        "p50_ms": percentile(latencies_sorted, 50),
        "p95_ms": percentile(latencies_sorted, 95),
        "target_p95_ms": "35~50",
        "passes_target_50ms": percentile(latencies_sorted, 95) <= 50.0,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output_json:
        path = Path(args.output_json).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_once(model, image, lidar, phase, lidar_enabled: bool, phase_enabled: bool):
    if lidar_enabled:
        return model(image, lidar)
    if phase_enabled:
        return model(image, phase)
    return model(image)


def choose_device(device_name: str, torch):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def percentile(values, percent: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, int(round((percent / 100.0) * (len(values) - 1)))))
    return values[index]


if __name__ == "__main__":
    main()
