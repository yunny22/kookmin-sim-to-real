from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import time

import cv2
import numpy as np

from xycar_perception.yolo_lane_segmenter import YoloLaneSegmenter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the production lane-segmentation wrapper."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument(
        "--retina-masks",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    return parser.parse_args()


def load_image(source: str, image_size: int) -> np.ndarray:
    if source:
        image = cv2.imread(str(Path(source).expanduser()))
        if image is None:
            raise FileNotFoundError(f"failed to read benchmark image: {source}")
        return image
    return np.zeros((image_size * 2, image_size * 5 // 2, 3), dtype=np.uint8)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
    return ordered[index]


def main() -> None:
    args = parse_args()
    image = load_image(args.source, args.image_size)
    segmenter = YoloLaneSegmenter(
        args.model,
        device=args.device,
        confidence=args.confidence,
        image_size=args.image_size,
        cpu_threads=args.cpu_threads,
        retina_masks=args.retina_masks,
    )

    for _ in range(max(0, args.warmup)):
        segmenter.predict(image, render_debug=False)

    durations_ms: list[float] = []
    for _ in range(max(1, args.iterations)):
        started = time.perf_counter()
        segmenter.predict(image, render_debug=False)
        durations_ms.append((time.perf_counter() - started) * 1000.0)

    mean_ms = statistics.fmean(durations_ms)
    print(f"model={segmenter.model_path}")
    print(
        f"device={args.device} image_size={args.image_size} "
        f"cpu_threads={args.cpu_threads} retina_masks={args.retina_masks}"
    )
    print(f"input_shape={image.shape[1]}x{image.shape[0]}")
    print(
        "latency_ms "
        f"mean={mean_ms:.2f} median={statistics.median(durations_ms):.2f} "
        f"p95={percentile(durations_ms, 0.95):.2f} "
        f"max={max(durations_ms):.2f}"
    )
    print(f"throughput_fps={1000.0 / mean_ms:.2f}")


if __name__ == "__main__":
    main()
