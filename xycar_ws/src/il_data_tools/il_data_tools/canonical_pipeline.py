from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

from il_data_tools.canonical_dataset_converter import convert_session, load_visibility


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect canonical BEV recovery data in independent Gazebo sessions, "
            "train and evaluate a policy, publish it, then optionally power off."
        )
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--total-samples", type=int, default=100_000)
    parser.add_argument("--batch-samples", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--run-name", default="drive_canonical_real_reference_200k_20260715")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--show-gui-first", action="store_true")
    parser.add_argument(
        "--no-canonical-artifacts",
        action="store_true",
        help="Disable real-camera-like canonical artifact injection.",
    )
    parser.add_argument("--max-test-mae-command", type=float, default=8.0)
    parser.add_argument("--max-recovery-mae-command", type=float, default=12.0)
    parser.add_argument("--publish-model", action="store_true")
    parser.add_argument("--git-remotes", default="origin,teamkai")
    parser.add_argument("--poweroff-on-success", action="store_true")
    parser.add_argument(
        "--skip-collection",
        action="store_true",
        help="Train from sessions listed by repeated --session-dir.",
    )
    parser.add_argument("--session-dir", action="append", default=[])
    parser.add_argument(
        "--real-reference-profile",
        default="",
        help="Visibility JSON made from valid real competition and temporary-track frames.",
    )
    parser.add_argument(
        "--include-run",
        action="append",
        default=[],
        help="Existing collection run name whose clean sessions will be converted and reused.",
    )
    parser.add_argument(
        "--existing-visibility-variants",
        type=int,
        default=2,
        help="Number of current-format visibility variants per included clean session.",
    )
    return parser


def run(command: Sequence[str], project_root: Path) -> None:
    print("\n$ " + " ".join(str(part) for part in command), flush=True)
    completed = subprocess.run([str(part) for part in command], cwd=project_root)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}: {command[0]}"
        )


def session_set(dataset_root: Path) -> set[Path]:
    drive_root = dataset_root / "drive"
    return {
        csv_path.parent.resolve()
        for csv_path in drive_root.glob("*/samples.csv")
        if csv_path.is_file()
    }


def sessions_from_run(dataset_root: Path, run_name: str) -> list[Path]:
    manifest = dataset_root / "runs" / run_name / "collection_manifest.json"
    if not manifest.is_file():
        raise RuntimeError(f"included collection manifest not found: {manifest}")
    sessions = [Path(value).expanduser().resolve() for value in load_json(manifest)["sessions"]]
    missing = [str(path) for path in sessions if not (path / "samples.csv").is_file()]
    if missing:
        raise RuntimeError(f"included run has missing sessions: {missing}")
    return sessions


def row_count(sessions: Sequence[Path]) -> int:
    total = 0
    for session in sessions:
        with (session / "samples.csv").open(newline="", encoding="utf-8-sig") as handle:
            total += sum(1 for _ in csv.DictReader(handle))
    return total


def validate_sessions(
    sessions: Sequence[Path],
    expected_samples: int,
    require_canonical_artifacts: bool = False,
) -> Dict[str, object]:
    label_counts: Counter[str] = Counter()
    session_rows: Dict[str, int] = {}
    missing_files: List[str] = []
    stopped_rows = 0
    discarded_bad_preroll = 0
    artifact_event_count = 0
    artifact_modes: Counter[str] = Counter()
    for session in sessions:
        csv_path = session / "samples.csv"
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        metadata = load_json(session / "metadata.json")
        if require_canonical_artifacts:
            artifact_manifest = metadata.get("run_manifest", {}).get(
                "canonical_artifacts", {}
            )
            if not artifact_manifest.get("enabled", False):
                raise RuntimeError(
                    f"{session.name} did not enable canonical artifact injection"
                )
            artifact_mode = str(artifact_manifest.get("mode", ""))
            if artifact_mode != "real_visibility":
                raise RuntimeError(
                    f"{session.name} used unsupported canonical artifact mode: "
                    f"{artifact_mode or 'missing'}"
                )
            artifact_modes[artifact_mode] += 1
            if (
                metadata.get("parameters", {}).get("camera_front_topic")
                != "/perception/canonical_road_image_augmented"
            ):
                raise RuntimeError(
                    f"{session.name} did not record the augmented canonical topic"
                )
            event_log = Path(str(artifact_manifest.get("event_log", "")))
            if not event_log.is_file():
                raise RuntimeError(
                    f"{session.name} canonical artifact event log is missing: "
                    f"{event_log}"
                )
            session_events = sum(
                1
                for line in event_log.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            if session_events <= 0:
                raise RuntimeError(
                    f"{session.name} contains no canonical artifact events"
                )
            artifact_event_count += session_events
        preroll_sec = float(
            metadata.get("parameters", {}).get("bad_data_preroll_sec", 0.0)
        )
        if preroll_sec < 10.0:
            raise RuntimeError(
                f"{session.name} did not use the required 10-second bad-data preroll"
            )
        discarded_bad_preroll += int(
            metadata.get("discarded_bad_data_preroll", 0)
        )
        session_rows[session.name] = len(rows)
        for row in rows:
            label_counts[row.get("mission_label", "")] += 1
            try:
                if abs(float(row.get("motor_speed", "0"))) <= 1.0e-6:
                    stopped_rows += 1
            except ValueError:
                stopped_rows += 1
            for column in ("front_image_path", "scan_npz_path"):
                relative = row.get(column, "")
                if not relative or not (session / relative).is_file():
                    missing_files.append(f"{session.name}:{column}:{relative}")
                    if len(missing_files) >= 20:
                        break
            if len(missing_files) >= 20:
                break
    total_rows = sum(session_rows.values())
    if total_rows != expected_samples:
        raise RuntimeError(
            f"expected {expected_samples} new samples but validated {total_rows}: "
            f"{session_rows}"
        )
    if missing_files:
        raise RuntimeError(f"dataset has missing image/scan files: {missing_files}")
    if stopped_rows:
        raise RuntimeError(
            f"dataset contains {stopped_rows} stopped/invalid-speed rows; "
            "failed recovery frames must not be stored"
        )
    recovery_rows = label_counts.get("recovery", 0)
    recovery_ratio = recovery_rows / max(1, total_rows)
    if recovery_ratio < 0.10:
        raise RuntimeError(
            f"recovery data ratio is too low: {recovery_rows}/{total_rows}"
        )
    return {
        "total_rows": total_rows,
        "session_rows": session_rows,
        "label_counts": dict(label_counts),
        "recovery_ratio": recovery_ratio,
        "stopped_rows": stopped_rows,
        "discarded_bad_data_preroll": discarded_bad_preroll,
        "canonical_artifact_events": artifact_event_count,
        "canonical_artifact_modes": dict(artifact_modes),
    }


def load_json(path: Path) -> Dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_latest_model_doc(
    path: Path,
    model_digest: str,
    collection: Dict[str, object],
    train_metrics: Dict[str, object],
    test_metrics: Dict[str, object],
) -> None:
    labels = collection["label_counts"]
    text = f"""# Latest Canonical BEV Policy

This file is generated only after collection, training, offline evaluation, and
GitHub publication all pass.

- model: `drive_canonical_policy_scripted.pt`
- SHA-256: `{model_digest}`
- collected rows: `{collection['total_rows']}`
- recovery rows: `{labels.get('recovery', 0)}`
- recovery ratio: `{collection['recovery_ratio']:.3f}`
- best epoch: `{train_metrics.get('best_epoch')}`
- validation MAE (Xycar angle command): `{train_metrics.get('val_mae_deg')}`
- held-out test MAE (Xycar angle command): `{test_metrics.get('val_mae_deg')}`

## Simulation

```bash
cd ~/xycar_kookmin_gazebo_track
source /opt/ros/humble/setup.bash
colcon build --packages-select kaiev26_msgs xycar_perception \\
  xycar_gazebo_bridge il_data_tools --symlink-install
source install/setup.bash

MODEL=\"$(ros2 pkg prefix il_data_tools)/share/il_data_tools/models/drive_canonical_policy_scripted.pt\"
ros2 launch il_data_tools sim_policy_drive.launch.py \\
  model_path:=\"$MODEL\" \\
  image_topic:=/perception/canonical_road_image \\
  drive_enabled:=true device:=cuda
```

## Real Car Shadow

```bash
MODEL=\"$(ros2 pkg prefix il_data_tools)/share/il_data_tools/models/drive_canonical_policy_scripted.pt\"
ros2 launch il_data_tools real_canonical_policy_drive.launch.py \\
  model_path:=\"$MODEL\" \\
  source_image_topic:=/wide_camera/rect/image_raw \\
  scan_topic:=/scan max_steer_scale:=140.0 \\
  drive_enabled:=false device:=cpu
```

Only after the shadow steering sign and sensor timeout stop have been checked,
repeat the second command with `drive_enabled:=true speed_command:=3.0`.
The previous 50k policy was more stable at 130--150% steering, so 140% is the
initial calibration candidate rather than a value baked into the model. Recheck
100/130/140/150% with this policy and keep the lowest stable value.
"""
    path.write_text(text, encoding="utf-8")


def publish_model(
    project_root: Path,
    model_path: Path,
    collection: Dict[str, object],
    train_metrics: Dict[str, object],
    test_metrics: Dict[str, object],
    remotes: Sequence[str],
) -> Path:
    package_models = project_root / "xycar_ws" / "src" / "il_data_tools" / "models"
    packaged_model = package_models / "drive_canonical_policy_scripted.pt"
    packaged_metrics = package_models / "drive_canonical_policy_metrics.json"
    latest_doc = project_root / "docs" / "canonical_model_latest.md"
    shutil.copy2(model_path, packaged_model)
    model_digest = sha256(packaged_model)
    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "model_sha256": model_digest,
        "collection": collection,
        "training": train_metrics,
        "test": test_metrics,
    }
    write_json(packaged_metrics, report)
    write_latest_model_doc(
        latest_doc, model_digest, collection, train_metrics, test_metrics
    )

    tracked = [packaged_model, packaged_metrics, latest_doc]
    relative_tracked = [str(path.relative_to(project_root)) for path in tracked]
    run(["git", "add", "--force", *relative_tracked], project_root)
    staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=project_root
    )
    if staged.returncode != 0:
        run(["git", "commit", "-m", "Publish canonical BEV driving policy"], project_root)
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=project_root, text=True
    ).strip()
    for remote in remotes:
        run(["git", "push", remote, branch], project_root)
    return packaged_model


def poweroff() -> None:
    subprocess.run(["sync"], check=False)
    print("Pipeline succeeded. Powering off in 10 seconds.", flush=True)
    time.sleep(10.0)
    first = subprocess.run(["systemctl", "poweroff"])
    if first.returncode != 0:
        fallback = subprocess.run(["shutdown", "-h", "now"])
        if fallback.returncode != 0:
            raise RuntimeError("model succeeded, but both poweroff commands failed")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    project_root = args.project_root.expanduser().resolve()
    if not (project_root / "worlds" / "kookmin_xycar_track_final.sdf").is_file():
        raise SystemExit(f"invalid project root: {project_root}")
    if args.total_samples <= 0 or args.batch_samples <= 0:
        raise SystemExit("sample counts must be positive")

    dataset_root = project_root / "datasets" / "il_canonical"
    run_dir = dataset_root / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    before = session_set(dataset_root)
    if args.skip_collection:
        sessions = [Path(value).expanduser().resolve() for value in args.session_dir]
        if not sessions:
            raise SystemExit("--skip-collection requires at least one --session-dir")
    else:
        command = [
            "ros2",
            "run",
            "il_data_tools",
            "collect_randomized_batches",
            "--project-root",
            str(project_root),
            "--output-root",
            str(dataset_root),
            "--total-samples",
            str(args.total_samples),
            "--batch-samples",
            str(args.batch_samples),
            "--seed",
            str(args.seed),
            "--canonical-input",
            "--scenario-interval-sec",
            "24.0",
            "--recovery-hold-sec",
            "9.0",
            "--lane-offset-from-yellow-m",
            "0.05",
            "--canonical-artifact-mode",
            "real_visibility",
        ]
        if args.real_reference_profile:
            command += ["--real-reference-profile", args.real_reference_profile]
        if not args.no_canonical_artifacts:
            command.append("--canonical-artifacts")
        if args.show_gui_first:
            command.append("--show-gui-first")
        run(command, project_root)
        sessions = sorted(session_set(dataset_root) - before)
        expected_sessions = int(math.ceil(args.total_samples / args.batch_samples))
        if len(sessions) != expected_sessions:
            raise RuntimeError(
                f"expected {expected_sessions} new sessions, found {len(sessions)}"
            )

    collection = validate_sessions(
        sessions,
        args.total_samples,
        require_canonical_artifacts=not args.no_canonical_artifacts,
    )
    collection["sessions"] = [str(path) for path in sessions]
    collection["rejected_sessions"] = [str(path) for path in sorted(before)]
    collection["real_reference_profile"] = args.real_reference_profile or None

    included_clean: list[Path] = []
    for included_run in args.include_run:
        included_clean.extend(sessions_from_run(dataset_root, included_run))
    included_clean = list(dict.fromkeys(included_clean))
    derived_sessions: list[Path] = []
    if included_clean and args.existing_visibility_variants <= 0:
        raise RuntimeError("existing visibility variants must be positive")
    visibility = load_visibility(args.real_reference_profile)
    derived_root = dataset_root / "derived" / "drive"
    for variant_index in range(args.existing_visibility_variants):
        variant = f"v{variant_index + 1}"
        for session_index, source_session in enumerate(included_clean):
            derived_sessions.append(
                convert_session(
                    source_session,
                    derived_root,
                    visibility,
                    args.seed + 100_000 + variant_index * 10_000 + session_index,
                    variant,
                )
            )

    training_sessions = [*sessions, *derived_sessions]
    training_rows = row_count(training_sessions)
    collection["training_dataset"] = {
        "new_sessions": [str(path) for path in sessions],
        "included_clean_sources": [str(path) for path in included_clean],
        "derived_visibility_sessions": [str(path) for path in derived_sessions],
        "training_sessions": [str(path) for path in training_sessions],
        "raw_rows_before_split": training_rows,
        "source_trajectory_rows": args.total_samples + row_count(included_clean),
        "split_leakage_guard": (
            "derived variants preserve source session_id so all views of one drive "
            "remain in the same train/val/test split"
        ),
    }
    write_json(run_dir / "collection_manifest.json", collection)

    processed_dir = project_root / "datasets" / "processed" / args.run_name
    model_dir = project_root / "models" / "il_policies" / args.run_name
    train_command = [
        "ros2",
        "run",
        "il_data_tools",
        "train_from_raw_dataset.py",
        "--profile",
        "drive",
    ]
    for session in training_sessions:
        train_command += ["--session-dir", str(session)]
    train_command += [
        "--processed-dir",
        str(processed_dir),
        "--model-output-dir",
        str(model_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--recovery-oversample-factor",
        "1",
        "--canonical-input",
        "--lane-dropout-probability",
        "0.05",
        "--enable-flip",
        "--mark-final",
    ]
    run(train_command, project_root)

    model_path = model_dir / "drive_policy_scripted.pt"
    train_metrics_path = model_dir / "metrics.json"
    test_csv = processed_dir / "test.csv"
    if not model_path.is_file() or not train_metrics_path.is_file() or not test_csv.is_file():
        raise RuntimeError("training completed without the required model/metrics/test files")

    eval_dir = model_dir / "held_out_test"
    eval_script = (
        project_root
        / "xycar_ws"
        / "src"
        / "il_data_tools"
        / "scripts"
        / "eval_policy.py"
    )
    run(
        [
            sys.executable,
            str(eval_script),
            "--csv",
            str(test_csv),
            "--model",
            str(model_path),
            "--model-type",
            "resnet18_lidar",
            "--output-dir",
            str(eval_dir),
            "--device",
            args.device,
            "--canonical-input",
        ],
        project_root,
    )
    train_metrics = load_json(train_metrics_path)
    test_metrics = load_json(eval_dir / "eval_metrics.json")
    test_mae = float(test_metrics.get("val_mae_deg", float("inf")))
    if not math.isfinite(test_mae) or test_mae > args.max_test_mae_command:
        raise RuntimeError(
            f"held-out test MAE {test_mae:.3f} exceeds quality gate "
            f"{args.max_test_mae_command:.3f}; keeping the computer on"
        )
    recovery_mae = float(
        test_metrics.get("label_wise_mae", {}).get("recovery", float("inf"))
    )
    if (
        not math.isfinite(recovery_mae)
        or recovery_mae > args.max_recovery_mae_command
    ):
        raise RuntimeError(
            f"held-out recovery MAE {recovery_mae:.3f} exceeds quality gate "
            f"{args.max_recovery_mae_command:.3f}; keeping the computer on"
        )
    write_json(
        run_dir / "pipeline_result.json",
        {
            "status": "passed",
            "collection": collection,
            "training": train_metrics,
            "test": test_metrics,
            "model": str(model_path),
            "model_sha256": sha256(model_path),
        },
    )

    if args.publish_model:
        remotes = [value.strip() for value in args.git_remotes.split(",") if value.strip()]
        publish_model(
            project_root,
            model_path,
            collection,
            train_metrics,
            test_metrics,
            remotes,
        )
    if args.poweroff_on_success:
        poweroff()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
