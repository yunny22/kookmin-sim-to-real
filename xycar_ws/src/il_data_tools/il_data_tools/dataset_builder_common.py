#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_EXCLUDED_LABELS = {
    "bad_data",
    "idle",
    "red_light_wait",
    "pedestrian_wait",
    "parking",
}

COMMON_OUTPUT_COLUMNS = [
    "image_path",
    "steer_norm",
    "angle_deg",
    "speed",
    "mission_label",
    "session_id",
    "timestamp_ns",
]

LIDAR_OUTPUT_COLUMNS = COMMON_OUTPUT_COLUMNS + [
    "scan_npz_path",
    "scan_timestamp_ns",
    "scan_time_offset_ms",
]


@dataclass
class BuildConfig:
    dataset_roots: List[Path]
    session_dirs: List[Path]
    output_dir: Path
    include_labels: List[str]
    excluded_labels: List[str]
    max_steer_deg: float
    min_abs_speed: float
    keep_stopped: bool
    val_ratio: float
    test_ratio: float
    seed: int
    require_scan: bool = False
    max_scan_time_offset_ms: float = 50.0
    image_column: str = "front_image_path"
    scan_column: str = "scan_npz_path"


@dataclass
class Sample:
    image_path: str
    steer_norm: float
    angle_deg: float
    speed: float
    mission_label: str
    session_id: str
    timestamp_ns: int
    scan_npz_path: str = ""
    scan_timestamp_ns: Optional[int] = None
    scan_time_offset_ms: Optional[float] = None
    phase: Optional[float] = None


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-root",
        action="append",
        default=[],
        help="Directory containing session folders, or a single session directory.",
    )
    parser.add_argument(
        "--session-dir",
        action="append",
        default=[],
        help="Explicit session directory containing samples.csv. Can be repeated.",
    )
    parser.add_argument("--output-dir", required=True, help="Processed CSV output dir.")
    parser.add_argument("--max-steer-deg", type=float, required=True)
    parser.add_argument(
        "--min-abs-speed",
        type=float,
        default=1.0,
        help="Stopped frames are excluded when abs(speed) is below this value.",
    )
    parser.add_argument(
        "--keep-stopped",
        action="store_true",
        help="Keep stopped frames instead of applying --min-abs-speed.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--image-column", default="front_image_path")
    parser.add_argument(
        "--require-scan",
        action="store_true",
        help="Reject rows without a valid synchronized scan NPZ file.",
    )
    parser.add_argument(
        "--max-scan-time-offset-ms",
        type=float,
        default=50.0,
        help="Maximum image/LiDAR timestamp difference for --require-scan.",
    )


def parse_config(
    args: argparse.Namespace,
    include_labels: Sequence[str],
    excluded_labels: Iterable[str] = DEFAULT_EXCLUDED_LABELS,
) -> BuildConfig:
    if args.max_steer_deg <= 0:
        raise ValueError("--max-steer-deg must be positive")
    if args.min_abs_speed < 0:
        raise ValueError("--min-abs-speed must be non-negative")
    if args.val_ratio < 0 or args.test_ratio < 0 or args.val_ratio + args.test_ratio >= 1:
        raise ValueError("--val-ratio + --test-ratio must be >= 0 and < 1")
    if args.max_scan_time_offset_ms < 0:
        raise ValueError("--max-scan-time-offset-ms must be non-negative")

    dataset_roots = [Path(value).expanduser().resolve() for value in args.dataset_root]
    session_dirs = [Path(value).expanduser().resolve() for value in args.session_dir]
    if not dataset_roots and not session_dirs:
        raise ValueError("Provide at least one --dataset-root or --session-dir")

    return BuildConfig(
        dataset_roots=dataset_roots,
        session_dirs=session_dirs,
        output_dir=Path(args.output_dir).expanduser().resolve(),
        include_labels=list(include_labels),
        excluded_labels=list(excluded_labels),
        max_steer_deg=float(args.max_steer_deg),
        min_abs_speed=float(args.min_abs_speed),
        keep_stopped=bool(args.keep_stopped),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        seed=int(args.seed),
        require_scan=bool(args.require_scan),
        max_scan_time_offset_ms=float(args.max_scan_time_offset_ms),
        image_column=str(args.image_column),
    )


def discover_sessions(config: BuildConfig) -> List[Path]:
    sessions = []
    for root in config.dataset_roots:
        if (root / "samples.csv").is_file():
            sessions.append(root)
            continue
        if root.is_dir():
            sessions.extend(path.parent for path in root.rglob("samples.csv"))
    sessions.extend(config.session_dirs)

    unique = {}
    for session in sessions:
        resolved = session.expanduser().resolve()
        if (resolved / "samples.csv").is_file():
            unique[str(resolved)] = resolved
    return sorted(unique.values(), key=lambda path: str(path))


def load_samples(config: BuildConfig) -> Tuple[List[Sample], Dict[str, object]]:
    sessions = discover_sessions(config)
    report = {
        "input_sessions": [str(path) for path in sessions],
        "include_labels": list(config.include_labels),
        "excluded_labels": list(config.excluded_labels),
        "max_steer_deg": config.max_steer_deg,
        "min_abs_speed": config.min_abs_speed,
        "keep_stopped": config.keep_stopped,
        "require_scan": config.require_scan,
        "max_scan_time_offset_ms": config.max_scan_time_offset_ms,
        "total_rows_read": 0,
        "samples_kept": 0,
        "filtered": Counter(),
        "label_counts": Counter(),
        "session_counts": Counter(),
        "missing_images": [],
        "missing_scans": [],
        "invalid_rows": [],
    }

    if not sessions:
        searched = [str(path) for path in config.dataset_roots + config.session_dirs]
        raise FileNotFoundError(f"No samples.csv files found under: {searched}")

    samples = []
    for session_dir in sessions:
        with (session_dir / "samples.csv").open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row_number, row in enumerate(reader, start=2):
                report["total_rows_read"] += 1
                parsed = parse_row(row, session_dir, config, row_number, report)
                if parsed is None:
                    continue
                samples.append(parsed)
                report["label_counts"][parsed.mission_label] += 1
                report["session_counts"][parsed.session_id] += 1

    report["samples_kept"] = len(samples)
    report["filtered"] = dict(report["filtered"])
    report["label_counts"] = dict(report["label_counts"])
    report["session_counts"] = dict(report["session_counts"])
    return samples, report


def parse_row(
    row: Dict[str, str],
    session_dir: Path,
    config: BuildConfig,
    row_number: int,
    report: Dict[str, object],
) -> Optional[Sample]:
    label = (row.get("mission_label") or "").strip()
    if label in config.excluded_labels:
        report["filtered"]["excluded_label"] += 1
        return None
    if config.include_labels and label not in config.include_labels:
        report["filtered"]["not_included_label"] += 1
        return None

    try:
        angle_deg = float(row.get("motor_angle", ""))
        speed = float(row.get("motor_speed", ""))
        timestamp_ns = int(row.get("timestamp_ns", ""))
    except (TypeError, ValueError):
        report["filtered"]["invalid_numeric"] += 1
        report["invalid_rows"].append(
            {"session_dir": str(session_dir), "row_number": row_number}
        )
        return None

    if not config.keep_stopped and abs(speed) < config.min_abs_speed:
        report["filtered"]["stopped"] += 1
        return None

    image_value = (row.get(config.image_column) or "").strip()
    image_path = resolve_session_path(session_dir, image_value)
    if not image_value or not image_path.is_file():
        report["filtered"]["missing_image"] += 1
        report["missing_images"].append(
            {
                "session_dir": str(session_dir),
                "row_number": row_number,
                "image_path": image_value,
            }
        )
        return None

    scan_value = (row.get(config.scan_column) or "").strip()
    scan_path = resolve_session_path(session_dir, scan_value) if scan_value else None
    scan_output = str(scan_path) if scan_path and scan_path.is_file() else ""
    if scan_value and not scan_output:
        report["missing_scans"].append(
            {
                "session_dir": str(session_dir),
                "row_number": row_number,
                "scan_npz_path": scan_value,
            }
        )
    if config.require_scan and not scan_output:
        report["filtered"]["missing_scan"] += 1
        return None

    scan_timestamp_ns: Optional[int] = None
    scan_time_offset_ms: Optional[float] = None
    raw_scan_timestamp = (row.get("scan_timestamp_ns") or "").strip()
    raw_scan_offset = (row.get("scan_time_offset_ms") or "").strip()
    if raw_scan_timestamp and raw_scan_offset:
        try:
            scan_timestamp_ns = int(raw_scan_timestamp)
            scan_time_offset_ms = float(raw_scan_offset)
        except (TypeError, ValueError):
            report["filtered"]["invalid_scan_timestamp"] += 1
            if config.require_scan:
                return None
    elif config.require_scan:
        report["filtered"]["missing_scan_timestamp"] += 1
        return None

    if (
        config.require_scan
        and scan_time_offset_ms is not None
        and scan_time_offset_ms > config.max_scan_time_offset_ms
    ):
        report["filtered"]["unsynchronized_scan"] += 1
        return None

    steer_norm = clamp(angle_deg / config.max_steer_deg, -1.0, 1.0)
    session_id = (row.get("session_id") or session_dir.name).strip() or session_dir.name

    return Sample(
        image_path=str(image_path),
        steer_norm=steer_norm,
        angle_deg=angle_deg,
        speed=speed,
        mission_label=label,
        session_id=session_id,
        timestamp_ns=timestamp_ns,
        scan_npz_path=scan_output,
        scan_timestamp_ns=scan_timestamp_ns,
        scan_time_offset_ms=scan_time_offset_ms,
    )


def resolve_session_path(session_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (session_dir / path).resolve()


def split_by_session(
    samples: Sequence[Sample],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[Sample]]:
    sessions = sorted({sample.session_id for sample in samples})
    rng = random.Random(seed)
    rng.shuffle(sessions)

    if len(sessions) <= 1:
        assignment = {session: "train" for session in sessions}
    else:
        test_count = allocation_count(len(sessions), test_ratio)
        val_count = allocation_count(len(sessions) - test_count, val_ratio)
        if len(sessions) >= 3:
            test_count = max(1, test_count) if test_ratio > 0 else 0
            val_count = max(1, val_count) if val_ratio > 0 else 0
        while test_count + val_count >= len(sessions):
            if test_count >= val_count and test_count > 0:
                test_count -= 1
            elif val_count > 0:
                val_count -= 1
            else:
                break

        test_sessions = set(sessions[:test_count])
        val_sessions = set(sessions[test_count : test_count + val_count])
        assignment = {}
        for session in sessions:
            if session in test_sessions:
                assignment[session] = "test"
            elif session in val_sessions:
                assignment[session] = "val"
            else:
                assignment[session] = "train"

    splits = {"train": [], "val": [], "test": []}
    for sample in samples:
        splits[assignment[sample.session_id]].append(sample)
    for split_samples in splits.values():
        split_samples.sort(key=lambda sample: (sample.session_id, sample.timestamp_ns))
    return splits


def allocation_count(total: int, ratio: float) -> int:
    if total <= 0 or ratio <= 0:
        return 0
    return int(round(total * ratio))


def balance_steering(
    samples: Sequence[Sample],
    bins: int,
    max_bin_samples: Optional[int],
    seed: int,
) -> Tuple[List[Sample], Dict[str, object]]:
    if bins <= 0:
        return list(samples), {"enabled": False, "reason": "bins <= 0"}
    grouped: Dict[int, List[Sample]] = defaultdict(list)
    for sample in samples:
        index = min(bins - 1, max(0, int(((sample.steer_norm + 1.0) / 2.0) * bins)))
        grouped[index].append(sample)
    non_empty_sizes = [len(values) for values in grouped.values() if values]
    if not non_empty_sizes:
        return list(samples), {"enabled": True, "before": 0, "after": 0}
    limit = max_bin_samples if max_bin_samples and max_bin_samples > 0 else min(non_empty_sizes)
    rng = random.Random(seed)
    balanced = []
    bin_counts_before = {}
    bin_counts_after = {}
    for index in range(bins):
        values = list(grouped.get(index, []))
        bin_counts_before[str(index)] = len(values)
        if len(values) > limit:
            values = rng.sample(values, limit)
        bin_counts_after[str(index)] = len(values)
        balanced.extend(values)
    balanced.sort(key=lambda sample: (sample.session_id, sample.timestamp_ns))
    return balanced, {
        "enabled": True,
        "bins": bins,
        "max_bin_samples": limit,
        "before": len(samples),
        "after": len(balanced),
        "bin_counts_before": bin_counts_before,
        "bin_counts_after": bin_counts_after,
    }


def oversample_label(
    samples: Sequence[Sample],
    label: str,
    factor: int,
) -> Tuple[List[Sample], Dict[str, object]]:
    if factor <= 1:
        return list(samples), {"enabled": False, "factor": factor}
    output = list(samples)
    added = []
    for sample in samples:
        if sample.mission_label == label:
            added.extend([sample] * (factor - 1))
    output.extend(added)
    output.sort(key=lambda sample: (sample.session_id, sample.timestamp_ns))
    return output, {
        "enabled": True,
        "label": label,
        "factor": factor,
        "added_rows": len(added),
        "before": len(samples),
        "after": len(output),
    }


def write_csv(path: Path, rows: Sequence[Dict[str, object]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(path: Path, report: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def sample_to_row(sample: Sample, columns: Sequence[str]) -> Dict[str, object]:
    values = {
        "image_path": sample.image_path,
        "steer_norm": format_float(sample.steer_norm),
        "angle_deg": format_float(sample.angle_deg),
        "speed": format_float(sample.speed),
        "mission_label": sample.mission_label,
        "session_id": sample.session_id,
        "timestamp_ns": sample.timestamp_ns,
        "scan_npz_path": sample.scan_npz_path,
        "scan_timestamp_ns": "" if sample.scan_timestamp_ns is None else sample.scan_timestamp_ns,
        "scan_time_offset_ms": (
            "" if sample.scan_time_offset_ms is None else format_float(sample.scan_time_offset_ms)
        ),
        "phase": "" if sample.phase is None else format_float(sample.phase),
    }
    return {column: values.get(column, "") for column in columns}


def write_splits(
    output_dir: Path,
    splits: Dict[str, List[Sample]],
    columns: Sequence[str],
) -> Dict[str, Dict[str, object]]:
    output = {}
    for split_name in ["train", "val", "test"]:
        rows = [sample_to_row(sample, columns) for sample in splits[split_name]]
        out_path = output_dir / f"{split_name}.csv"
        write_csv(out_path, rows, columns)
        output[split_name] = {
            "path": str(out_path),
            "rows": len(rows),
            "sessions": sorted({sample.session_id for sample in splits[split_name]}),
        }
    return output


def finalize_report(
    report: Dict[str, object],
    output_dir: Path,
    splits: Dict[str, List[Sample]],
    output_files: Dict[str, Dict[str, object]],
    extra: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    report["output_dir"] = str(output_dir)
    report["output_files"] = output_files
    report["split_counts"] = {name: len(rows) for name, rows in splits.items()}
    report["split_session_counts"] = {
        name: len({sample.session_id for sample in rows}) for name, rows in splits.items()
    }
    report["split_label_counts"] = {
        name: dict(Counter(sample.mission_label for sample in rows))
        for name, rows in splits.items()
    }
    if extra:
        report.update(extra)
    return report


def clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def format_float(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.8g}"
