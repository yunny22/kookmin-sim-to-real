#!/usr/bin/env python3
import argparse
import csv
import math
from collections import Counter
from pathlib import Path


def as_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", help="Session directory containing samples.csv")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    csv_path = dataset_dir / "samples.csv"
    if not csv_path.exists():
        raise SystemExit(f"samples.csv not found: {csv_path}")

    rows = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    labels = Counter(row.get("mission_label") or "unknown" for row in rows)
    angles = [as_float(row.get("motor_angle")) for row in rows]
    speeds = [as_float(row.get("motor_speed")) for row in rows]

    missing_front = 0
    missing_scan = 0
    scan_rows = 0
    for row in rows:
        front = row.get("front_image_path") or ""
        scan = row.get("scan_npz_path") or ""
        if front and not (dataset_dir / front).exists():
            missing_front += 1
        if scan:
            scan_rows += 1
            if not (dataset_dir / scan).exists():
                missing_scan += 1

    timestamps = []
    for row in rows:
        try:
            timestamps.append(int(row.get("timestamp_ns") or "0"))
        except ValueError:
            pass
    timestamps.sort()
    duration_sec = 0.0
    if len(timestamps) >= 2:
        duration_sec = (timestamps[-1] - timestamps[0]) / 1e9

    print(f"dataset_dir: {dataset_dir}")
    print(f"samples: {len(rows)}")
    print(f"duration_sec: {duration_sec:.2f}")
    print(f"estimated_rate_hz: {(len(rows) / duration_sec) if duration_sec > 0 else 0.0:.2f}")
    print(f"labels: {dict(labels)}")
    print(f"angle_stats: {stats(angles)}")
    print(f"speed_stats: {stats(speeds)}")
    print(f"front_missing_files: {missing_front}")
    print(f"scan_rows: {scan_rows}")
    print(f"scan_missing_files: {missing_scan}")


if __name__ == "__main__":
    main()
