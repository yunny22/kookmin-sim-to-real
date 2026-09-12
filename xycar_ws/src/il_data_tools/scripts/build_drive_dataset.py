#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from il_data_tools.dataset_builder_common import (
    LIDAR_OUTPUT_COLUMNS,
    add_common_args,
    balance_steering,
    finalize_report,
    load_samples,
    oversample_label,
    parse_config,
    split_by_session,
    write_report,
    write_splits,
)


INCLUDE_LABELS = [
    "general_drive",
    "lane_drive",
    "hill_drive",
    "shortcut",
    "recovery",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build processed CSV files for the steering-only drive policy."
    )
    add_common_args(parser)
    parser.set_defaults(require_scan=True)
    parser.add_argument(
        "--balance-steering",
        action="store_true",
        help="Downsample train split so steering bins are more balanced.",
    )
    parser.add_argument("--steering-bins", type=int, default=21)
    parser.add_argument(
        "--max-bin-samples",
        type=int,
        default=0,
        help="Per-bin cap for --balance-steering. 0 uses the smallest non-empty bin.",
    )
    parser.add_argument(
        "--recovery-oversample-factor",
        type=int,
        default=1,
        help="Duplicate recovery rows in the train split by this integer factor.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = parse_config(args, INCLUDE_LABELS)

    samples, report = load_samples(config)
    splits = split_by_session(samples, config.val_ratio, config.test_ratio, config.seed)

    recovery_samples, recovery_report = oversample_label(
        splits["train"], "recovery", args.recovery_oversample_factor
    )
    splits["train"] = recovery_samples

    balance_report = {"enabled": False}
    if args.balance_steering:
        balanced, balance_report = balance_steering(
            splits["train"],
            bins=args.steering_bins,
            max_bin_samples=args.max_bin_samples,
            seed=config.seed,
        )
        splits["train"] = balanced

    output_files = write_splits(config.output_dir, splits, LIDAR_OUTPUT_COLUMNS)
    final_report = finalize_report(
        report,
        config.output_dir,
        splits,
        output_files,
        extra={
            "policy": "drive",
            "output_columns": LIDAR_OUTPUT_COLUMNS,
            "balance_steering": balance_report,
            "recovery_oversample": recovery_report,
        },
    )
    write_report(config.output_dir / "dataset_report.json", final_report)
    print(f"wrote processed drive dataset: {config.output_dir}")


if __name__ == "__main__":
    main()
