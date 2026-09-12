#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import defaultdict
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from il_data_tools.dataset_builder_common import (
    add_common_args,
    clamp,
    finalize_report,
    load_samples,
    parse_config,
    split_by_session,
    write_report,
    write_splits,
)


INCLUDE_LABELS = [
    "vehicle_overtake",
    "overtake_start",
    "overtake_end",
    "recovery",
]

OUTPUT_COLUMNS = [
    "image_path",
    "steer_norm",
    "angle_deg",
    "speed",
    "phase",
    "mission_label",
    "session_id",
    "timestamp_ns",
    "scan_npz_path",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build processed CSV files for the steering-only overtake policy."
    )
    add_common_args(parser)
    parser.add_argument(
        "--default-duration-sec",
        type=float,
        default=4.0,
        help="Fallback phase duration when overtake_start/overtake_end is missing.",
    )
    parser.add_argument("--min-speed", type=float, default=None)
    parser.add_argument("--max-speed", type=float, default=None)
    parser.add_argument(
        "--max-speed-std",
        type=float,
        default=None,
        help="Fail when a session's speed standard deviation exceeds this value.",
    )
    parser.add_argument(
        "--require-explicit-phase-boundaries",
        action="store_true",
        help="Require exactly one overtake_start and one later overtake_end per session.",
    )
    return parser


def assign_overtake_phase(
    samples,
    default_duration_sec: float,
    require_explicit_boundaries: bool = False,
) -> Dict[str, object]:
    if default_duration_sec <= 0:
        raise ValueError("--default-duration-sec must be positive")

    by_session = defaultdict(list)
    for sample in samples:
        by_session[sample.session_id].append(sample)

    phase_report = {}
    default_duration_ns = int(default_duration_sec * 1_000_000_000)
    for session_id, session_samples in by_session.items():
        session_samples.sort(key=lambda sample: sample.timestamp_ns)
        label_runs = contiguous_label_runs(session_samples)
        start_runs = [run for run in label_runs if run["label"] == "overtake_start"]
        end_runs = [run for run in label_runs if run["label"] == "overtake_end"]
        explicit_valid = (
            len(start_runs) == 1
            and len(end_runs) == 1
            and end_runs[0]["start_timestamp_ns"] > start_runs[0]["start_timestamp_ns"]
        )
        if require_explicit_boundaries and not explicit_valid:
            raise ValueError(
                f"session {session_id!r} must contain exactly one ordered "
                "overtake_start/overtake_end maneuver"
            )
        start_ts, end_ts, source = find_phase_bounds(session_samples, default_duration_ns)
        duration = max(1, end_ts - start_ts)
        for sample in session_samples:
            sample.phase = clamp((sample.timestamp_ns - start_ts) / duration, 0.0, 1.0)
        phase_report[session_id] = {
            "start_timestamp_ns": start_ts,
            "end_timestamp_ns": end_ts,
            "source": source,
            "duration_sec": duration / 1_000_000_000.0,
            "explicit_boundaries_valid": explicit_valid,
            "used_fallback": source != "overtake_start_to_overtake_end",
            "label_runs": label_runs,
        }
    return phase_report


def contiguous_label_runs(samples) -> List[Dict[str, object]]:
    runs: List[Dict[str, object]] = []
    for sample in samples:
        if not runs or runs[-1]["label"] != sample.mission_label:
            runs.append(
                {
                    "label": sample.mission_label,
                    "start_timestamp_ns": sample.timestamp_ns,
                    "end_timestamp_ns": sample.timestamp_ns,
                    "rows": 1,
                }
            )
        else:
            runs[-1]["end_timestamp_ns"] = sample.timestamp_ns
            runs[-1]["rows"] += 1
    return runs


def validate_speed_consistency(samples, min_speed, max_speed, max_speed_std):
    by_session = defaultdict(list)
    for sample in samples:
        by_session[sample.session_id].append(float(sample.speed))
    report = {}
    for session_id, speeds in by_session.items():
        stats = {
            "count": len(speeds),
            "min": min(speeds),
            "max": max(speeds),
            "mean": statistics.fmean(speeds),
            "std": statistics.pstdev(speeds) if len(speeds) > 1 else 0.0,
        }
        violations = []
        if min_speed is not None and stats["min"] < min_speed:
            violations.append(f"min {stats['min']:.3f} < {min_speed:.3f}")
        if max_speed is not None and stats["max"] > max_speed:
            violations.append(f"max {stats['max']:.3f} > {max_speed:.3f}")
        if max_speed_std is not None and stats["std"] > max_speed_std:
            violations.append(f"std {stats['std']:.3f} > {max_speed_std:.3f}")
        stats["violations"] = violations
        report[session_id] = stats
    failed = {key: value for key, value in report.items() if value["violations"]}
    if failed:
        raise ValueError(f"overtake speed consistency failed: {failed}")
    return report


def find_phase_bounds(samples, default_duration_ns: int) -> Tuple[int, int, str]:
    start_candidates = [
        sample.timestamp_ns for sample in samples if sample.mission_label == "overtake_start"
    ]
    end_candidates = [
        sample.timestamp_ns for sample in samples if sample.mission_label == "overtake_end"
    ]

    if start_candidates and end_candidates:
        start_ts = min(start_candidates)
        valid_end = [timestamp for timestamp in end_candidates if timestamp > start_ts]
        end_ts = min(valid_end) if valid_end else max(end_candidates)
        if end_ts <= start_ts:
            end_ts = start_ts + default_duration_ns
            return start_ts, end_ts, "start_and_invalid_end_fallback_duration"
        return start_ts, end_ts, "overtake_start_to_overtake_end"

    if start_candidates:
        start_ts = min(start_candidates)
        return start_ts, start_ts + default_duration_ns, "start_plus_default_duration"

    if end_candidates:
        end_ts = max(end_candidates)
        return end_ts - default_duration_ns, end_ts, "end_minus_default_duration"

    vehicle_candidates = [
        sample.timestamp_ns
        for sample in samples
        if sample.mission_label == "vehicle_overtake"
    ]
    start_ts = min(vehicle_candidates) if vehicle_candidates else min(
        sample.timestamp_ns for sample in samples
    )
    return start_ts, start_ts + default_duration_ns, "first_overtake_plus_default_duration"


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = parse_config(args, INCLUDE_LABELS)

    samples, report = load_samples(config)
    speed_report = validate_speed_consistency(
        samples, args.min_speed, args.max_speed, args.max_speed_std
    )
    phase_report = assign_overtake_phase(
        samples,
        args.default_duration_sec,
        args.require_explicit_phase_boundaries,
    )
    splits = split_by_session(samples, config.val_ratio, config.test_ratio, config.seed)
    output_files = write_splits(config.output_dir, splits, OUTPUT_COLUMNS)

    final_report = finalize_report(
        report,
        config.output_dir,
        splits,
        output_files,
        extra={
            "policy": "overtake",
            "output_columns": OUTPUT_COLUMNS,
            "default_duration_sec": args.default_duration_sec,
            "phase_by_session": phase_report,
            "speed_by_session": speed_report,
            "speed_contract": {
                "min_speed": args.min_speed,
                "max_speed": args.max_speed,
                "max_speed_std": args.max_speed_std,
            },
        },
    )
    write_report(config.output_dir / "dataset_report.json", final_report)
    write_report(
        config.output_dir / "deployment_contract.json",
        {
            "policy": "overtake",
            "model_output": "steering_only",
            "phase_input_shape": [1, 1],
            "speed_contract": final_report["speed_contract"],
            "rule_based_runtime_must_enforce_speed_contract": True,
        },
    )
    print(f"wrote processed overtake dataset: {config.output_dir}")


if __name__ == "__main__":
    main()
