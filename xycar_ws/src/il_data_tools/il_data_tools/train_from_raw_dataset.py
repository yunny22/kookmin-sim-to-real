#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
import shlex
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Dict, List


PACKAGE_DIR = Path(__file__).resolve().parent
SOURCE_SCRIPT_DIR = PACKAGE_DIR.parent / "scripts"
PACKAGE_NAME = "il_data_tools"

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

PROFILE_CONFIG: Dict[str, Dict[str, str]] = {
    "drive": {
        "build_script": "build_drive_dataset.py",
        "train_script": "train_drive_policy.py",
        "default_model": "resnet18_lidar",
    },
    "cone": {
        "build_script": "build_cone_dataset.py",
        "train_script": "train_cone_policy.py",
        "default_model": "resnet18_lidar",
    },
    "overtake": {
        "build_script": "build_overtake_dataset.py",
        "train_script": "train_overtake_policy.py",
        "default_model": "pilotnet_phase",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build processed CSVs from raw recorder sessions, then train the matching "
            "imitation-learning policy."
        )
    )
    parser.add_argument("--profile", choices=sorted(PROFILE_CONFIG), required=True)
    parser.add_argument(
        "--dataset-root",
        action="append",
        default=[],
        help="Raw dataset root. Defaults to ~/xycar_ws/datasets/il/{profile}. Can repeat.",
    )
    parser.add_argument(
        "--session-dir",
        action="append",
        default=[],
        help="Specific raw session directory containing samples.csv. Can repeat.",
    )
    parser.add_argument(
        "--processed-dir",
        default=None,
        help="Processed CSV output dir. Defaults to ~/xycar_ws/datasets/processed/{profile}.",
    )
    parser.add_argument(
        "--model-output-dir",
        default=None,
        help=(
            "Training output dir. Defaults to "
            "~/xycar_ws/models/il_policies/{profile}_{model_type}."
        ),
    )
    parser.add_argument("--model-type", choices=SUPPORTED_MODEL_TYPES, default=None)
    parser.add_argument("--max-steer-deg", type=float, default=100.0)
    parser.add_argument("--min-abs-speed", type=float, default=1.0)
    parser.add_argument("--keep-stopped", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--build-seed", type=int, default=2026)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--image-column", default="front_image_path")
    parser.add_argument("--max-scan-time-offset-ms", type=float, default=50.0)

    parser.add_argument(
        "--balance-steering",
        action="store_true",
        help="Drive only: balance train samples by steering bins.",
    )
    parser.add_argument("--steering-bins", type=int, default=21)
    parser.add_argument("--max-bin-samples", type=int, default=0)
    parser.add_argument("--recovery-oversample-factor", type=int, default=1)
    parser.add_argument(
        "--default-duration-sec",
        type=float,
        default=4.0,
        help="Overtake only: fallback phase duration.",
    )
    parser.add_argument("--overtake-min-speed", type=float, default=None)
    parser.add_argument("--overtake-max-speed", type=float, default=None)
    parser.add_argument("--overtake-max-speed-std", type=float, default=None)
    parser.add_argument("--require-explicit-phase-boundaries", action="store_true")

    parser.add_argument("--input-width", type=int, default=160)
    parser.add_argument("--input-height", type=int, default=90)
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
            "Initialize from an existing project .pth checkpoint. Use this for "
            "low-learning-rate sim-to-real fine-tuning."
        ),
    )
    parser.add_argument("--enable-flip", action="store_true")
    parser.add_argument("--canonical-input", action="store_true")
    parser.add_argument("--lane-dropout-probability", type=float, default=0.30)
    parser.add_argument("--recovery-weight", type=float, default=1.5)
    parser.add_argument("--steer-weight-gain", type=float, default=2.0)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--scheduler", choices=["plateau", "cosine", "none"], default="plateau")
    parser.add_argument("--mark-final", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the build/train commands without running them.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    profile = args.profile
    config = PROFILE_CONFIG[profile]
    model_type = args.model_type or config["default_model"]
    processed_dir = resolve_processed_dir(args, profile)
    model_output_dir = resolve_model_output_dir(args, profile, model_type)

    build_cmd = make_build_command(args, config, profile, processed_dir)
    train_cmd = make_train_command(args, config, model_type, processed_dir, model_output_dir)

    print(f"profile={profile}")
    print(f"processed_dir={processed_dir}")
    print(f"model_output_dir={model_output_dir}")
    print("")
    print("[1/2] build command:")
    print(shlex.join(build_cmd))
    print("")
    print("[2/2] train command:")
    print(shlex.join(train_cmd))
    print("")

    if args.dry_run:
        print("dry-run only; nothing was executed.")
        return

    token = uuid.uuid4().hex[:10]
    processed_stage = processed_dir.with_name(f".{processed_dir.name}.staging-{token}")
    model_stage = model_output_dir.with_name(f".{model_output_dir.name}.staging-{token}")
    build_cmd = make_build_command(args, config, profile, processed_stage)
    train_cmd = make_train_command(args, config, model_type, processed_stage, model_stage)
    try:
        run_command(build_cmd)
        validate_processed_csvs(processed_stage)
        run_command(train_cmd)
        if profile == "overtake":
            contract = processed_stage / "deployment_contract.json"
            if contract.is_file():
                shutil.copy2(contract, model_stage / "deployment_contract.json")
        promote_directory(processed_stage, processed_dir)
        promote_directory(model_stage, model_output_dir)
    except BaseException:
        shutil.rmtree(processed_stage, ignore_errors=True)
        shutil.rmtree(model_stage, ignore_errors=True)
        raise
    print("")
    print("done")
    print(f"processed_csv_dir={processed_dir}")
    print(f"model_output_dir={model_output_dir}")


def resolve_processed_dir(args: argparse.Namespace, profile: str) -> Path:
    if args.processed_dir:
        return Path(args.processed_dir).expanduser().resolve()
    return (Path.home() / "xycar_ws" / "datasets" / "processed" / profile).resolve()


def resolve_model_output_dir(args: argparse.Namespace, profile: str, model_type: str) -> Path:
    if args.model_output_dir:
        return Path(args.model_output_dir).expanduser().resolve()
    return (Path.home() / "xycar_ws" / "models" / "il_policies" / f"{profile}_{model_type}").resolve()


def make_build_command(
    args: argparse.Namespace,
    config: Dict[str, str],
    profile: str,
    processed_dir: Path,
) -> List[str]:
    cmd = [sys.executable, str(find_helper_script(config["build_script"]))]
    dataset_roots = args.dataset_root or [str(Path.home() / "xycar_ws" / "datasets" / "il" / profile)]
    for dataset_root in dataset_roots:
        cmd += ["--dataset-root", str(Path(dataset_root).expanduser())]
    for session_dir in args.session_dir:
        cmd += ["--session-dir", str(Path(session_dir).expanduser())]

    cmd += [
        "--output-dir",
        str(processed_dir),
        "--max-steer-deg",
        str(args.max_steer_deg),
        "--min-abs-speed",
        str(args.min_abs_speed),
        "--val-ratio",
        str(args.val_ratio),
        "--test-ratio",
        str(args.test_ratio),
        "--seed",
        str(args.build_seed),
        "--image-column",
        args.image_column,
    ]
    if args.keep_stopped:
        cmd.append("--keep-stopped")

    if profile == "drive":
        cmd += ["--require-scan", "--max-scan-time-offset-ms", str(args.max_scan_time_offset_ms)]
        if args.balance_steering:
            cmd.append("--balance-steering")
        cmd += [
            "--steering-bins",
            str(args.steering_bins),
            "--max-bin-samples",
            str(args.max_bin_samples),
            "--recovery-oversample-factor",
            str(args.recovery_oversample_factor),
        ]
    elif profile == "cone":
        cmd += ["--require-scan", "--max-scan-time-offset-ms", str(args.max_scan_time_offset_ms)]
    elif profile == "overtake":
        cmd += ["--default-duration-sec", str(args.default_duration_sec)]
        if args.overtake_min_speed is not None:
            cmd += ["--min-speed", str(args.overtake_min_speed)]
        if args.overtake_max_speed is not None:
            cmd += ["--max-speed", str(args.overtake_max_speed)]
        if args.overtake_max_speed_std is not None:
            cmd += ["--max-speed-std", str(args.overtake_max_speed_std)]
        if args.require_explicit_phase_boundaries:
            cmd.append("--require-explicit-phase-boundaries")

    return cmd


def make_train_command(
    args: argparse.Namespace,
    config: Dict[str, str],
    model_type: str,
    processed_dir: Path,
    model_output_dir: Path,
) -> List[str]:
    cmd = [
        sys.executable,
        str(find_helper_script(config["train_script"])),
        "--train-csv",
        str(processed_dir / "train.csv"),
        "--val-csv",
        str(processed_dir / "val.csv"),
        "--output-dir",
        str(model_output_dir),
        "--model-type",
        model_type,
        "--input-width",
        str(args.input_width),
        "--input-height",
        str(args.input_height),
        "--max-steer-deg",
        str(args.max_steer_deg),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--recovery-weight",
        str(args.recovery_weight),
        "--steer-weight-gain",
        str(args.steer_weight_gain),
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--seed",
        str(args.train_seed),
        "--scheduler",
        args.scheduler,
    ]
    if args.use_phase or model_type.endswith("_phase"):
        cmd.append("--use-phase")
    if args.pretrained:
        cmd.append("--pretrained")
    if args.init_checkpoint:
        cmd += [
            "--init-checkpoint",
            str(Path(args.init_checkpoint).expanduser().resolve()),
        ]
    if args.enable_flip:
        cmd.append("--enable-flip")
    if args.canonical_input:
        cmd += [
            "--canonical-input",
            "--lane-dropout-probability",
            str(args.lane_dropout_probability),
        ]
    if args.mark_final:
        cmd.append("--mark-final")
    return cmd


def find_helper_script(script_name: str) -> Path:
    local_candidate = SOURCE_SCRIPT_DIR / script_name
    if local_candidate.is_file():
        return local_candidate

    try:
        from ament_index_python.packages import get_package_share_directory

        share_candidate = (
            Path(get_package_share_directory(PACKAGE_NAME)) / "scripts" / script_name
        )
        if share_candidate.is_file():
            return share_candidate
    except Exception:
        pass

    raise SystemExit(
        f"ERROR: helper script not found: {script_name}. "
        "Rebuild the workspace with colcon build --symlink-install --packages-select il_data_tools."
    )


def run_command(cmd: List[str]) -> None:
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def promote_directory(staging: Path, destination: Path) -> None:
    """Atomically expose successful output while preserving rollback on failure."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = destination.with_name(f".{destination.name}.backup-{uuid.uuid4().hex[:8]}")
    if destination.exists():
        os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except BaseException:
        if backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def validate_processed_csvs(processed_dir: Path) -> None:
    train_csv = processed_dir / "train.csv"
    val_csv = processed_dir / "val.csv"
    train_rows = count_data_rows(train_csv)
    val_rows = count_data_rows(val_csv)
    print(f"processed rows: train={train_rows}, val={val_rows}")

    if train_rows <= 0:
        raise SystemExit(
            f"ERROR: {train_csv} has no training rows. Check raw sessions, labels, "
            "image files, motor values, and --min-abs-speed."
        )
    if val_rows <= 0:
        raise SystemExit(
            f"ERROR: {val_csv} has no validation rows. Collect more than one session "
            "before training, because this project splits by session."
        )


def count_data_rows(csv_path: Path) -> int:
    if not csv_path.is_file():
        raise SystemExit(f"ERROR: missing processed CSV: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return sum(1 for _ in reader)


if __name__ == "__main__":
    main()
