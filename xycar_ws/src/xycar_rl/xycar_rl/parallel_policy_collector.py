from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time

from xycar_rl.models import checkpoint_payload
from xycar_rl.parallel_rule_collector import (
    _launch,
    _stop_processes,
    _wait_for_gazebo,
)


@dataclass(frozen=True)
class WorkerResult:
    worker_id: int
    requested_episodes: int
    recorded_episodes: int
    completed_laps: int
    failed_episodes: int
    transition_count: int
    observed_rate_hz: float
    rollout_return_code: int
    output_dir: str


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Collect camera-speed policy transitions from isolated Gazebo "
            "workers. Failed episodes remain available to the RL critic."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help=(
            "Isolated Gazebo workers. This workstation sustains 10 for long "
            "runs; 12 is intended only for monitored burst collection."
        ),
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=1_200)
    parser.add_argument("--control-rate-hz", type=float, default=7.0)
    parser.add_argument("--base-domain-id", type=int, default=120)
    parser.add_argument("--startup-stagger-sec", type=float, default=0.75)
    parser.add_argument("--episode-timeout-sec", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--action-noise", type=float, default=0.03)
    parser.add_argument("--speed-action-noise", type=float, default=0.03)
    parser.add_argument("--speed-action-bias", type=float, default=0.02)
    parser.add_argument(
        "--raw-steering-actions",
        action="store_true",
        help=(
            "Disable the deployment straight steering stabilizer. Use only "
            "for critic-focused oscillation collection."
        ),
    )
    parser.add_argument("--recovery-probability", type=float, default=0.30)
    parser.add_argument("--recovery-min-lateral-m", type=float, default=0.08)
    parser.add_argument("--recovery-max-lateral-m", type=float, default=0.22)
    parser.add_argument("--recovery-min-yaw-deg", type=float, default=4.0)
    parser.add_argument("--recovery-max-yaw-deg", type=float, default=14.0)
    parser.add_argument("--s-curve-focus-probability", type=float, default=0.40)
    parser.add_argument(
        "--start-progress-fraction",
        type=float,
        default=None,
        help="Use a fixed track start for every episode when provided.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--gui-first-worker", action="store_true")
    parser.add_argument("--target-right-offset-m", type=float, default=0.0)
    parser.add_argument(
        "--speed-cap-command",
        type=float,
        default=25.0,
        help="Cap collected policy actions at the requested real command.",
    )
    parser.add_argument("--off-track-threshold-m", type=float, default=0.38)
    parser.add_argument("--straight-segment-only", action="store_true")
    parser.add_argument("--straight-curvature-threshold", type=float, default=0.10)
    parser.add_argument("--straight-guard-distance-m", type=float, default=0.80)
    parser.add_argument(
        "--trace-privileged-expert",
        action="store_true",
        help=(
            "Record simulator pose-based expert labels alongside the "
            "learner actions for TD3+BC recovery training."
        ),
    )
    return parser.parse_args(argv)


def _episode_allocation(total: int, workers: int) -> list[int]:
    total = max(1, int(total))
    workers = max(1, min(int(workers), total))
    base, remainder = divmod(total, workers)
    return [base + int(index < remainder) for index in range(workers)]


def _session_summary(csv_path: Path) -> tuple[int, int, int, int, float]:
    if not csv_path.is_file():
        return 0, 0, 0, 0, 0.0
    rows = 0
    episode_ids: set[str] = set()
    episode_reasons: dict[str, str] = {}
    episode_stamp_ranges: dict[str, list[float]] = {}
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            episode_id = str(row.get("episode_id") or "0")
            episode_ids.add(episode_id)
            reason = str(row.get("termination_reason") or "")
            if reason:
                episode_reasons[episode_id] = reason
            stamp = float(row.get("state_timestamp_ns") or 0.0) * 1.0e-9
            if stamp > 0.0:
                stamp_range = episode_stamp_ranges.setdefault(
                    episode_id,
                    [stamp, stamp],
                )
                stamp_range[0] = min(stamp_range[0], stamp)
                stamp_range[1] = max(stamp_range[1], stamp)
    completed = sum(reason == "lap_complete" for reason in episode_reasons.values())
    failed = sum(
        reason in {
            "collision",
            "off_track",
            "stuck",
            "observation_timeout",
            "time_limit",
        }
        for reason in episode_reasons.values()
    )
    duration = sum(
        max(0.0, last - first)
        for first, last in episode_stamp_ranges.values()
    )
    rate_hz = rows / duration if duration > 0.0 else 0.0
    return rows, len(episode_ids), completed, failed, rate_hz


def apply_rollout_terminals(csv_path: Path, rollout_log_path: Path) -> None:
    if not csv_path.is_file() or not rollout_log_path.is_file():
        return
    terminal_reasons: list[str] = []
    pattern = re.compile(r"episode=(\d+).*reason=([a-z_]+)\s*$")
    for line in rollout_log_path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines():
        match = pattern.search(line)
        if match is not None:
            reason = match.group(2)
            terminal_reasons.append(
                "time_limit" if reason == "running" else reason
            )
    if not terminal_reasons:
        return
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames or not rows:
        return
    episode_ids = list(dict.fromkeys(row["episode_id"] for row in rows))
    last_indices = {
        episode_id: max(
            index
            for index, row in enumerate(rows)
            if row["episode_id"] == episode_id
        )
        for episode_id in episode_ids
    }
    for episode_id, reason in zip(episode_ids, terminal_reasons):
        final = rows[last_indices[episode_id]]
        final["termination_reason"] = reason
        final["terminated"] = str(
            int(
                reason
                in {
                    "lap_complete",
                    "collision",
                    "off_track",
                    "stuck",
                    "observation_timeout",
                }
            )
        )
        final["truncated"] = str(
            int(reason in {"time_limit", "straight_segment_complete"})
        )
    temporary = csv_path.with_suffix(".terminal.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)


class ParallelPolicyCollector:
    def __init__(self, args) -> None:
        self.args = args
        self.root = args.project_root.expanduser().resolve()
        self.checkpoint = args.checkpoint.expanduser().resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        payload = checkpoint_payload(self.checkpoint)
        self.min_speed = float(payload.get("min_speed_command", 4.0))
        self.max_speed = float(payload.get("max_speed_command", 25.0))
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.output_root = (
            args.output_root.expanduser().resolve()
            if args.output_root is not None
            else self.root
            / "datasets"
            / "rl"
            / f"lap_time_policy_parallel_{timestamp}"
        )
        self.output_root.mkdir(parents=True, exist_ok=False)
        self.results: list[WorkerResult] = []
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.active_processes: dict[
            int,
            list[tuple[subprocess.Popen, object]],
        ] = {}

    def _run_worker(self, worker_id: int, episodes: int) -> None:
        if self.stop_event.is_set():
            return
        time.sleep(max(0.0, self.args.startup_stagger_sec) * worker_id)
        worker_dir = self.output_root / f"worker_{worker_id:02d}"
        transition_dir = worker_dir / "transitions"
        logs = worker_dir / "logs"
        logs.mkdir(parents=True)
        env = os.environ.copy()
        env["ROS_DOMAIN_ID"] = str(self.args.base_domain_id + worker_id)
        env["ROS_LOCALHOST_ONLY"] = "1"
        env["GZ_PARTITION"] = (
            f"kookmin_policy_collect_{os.getpid()}_{worker_id}"
        )
        env["GZ_SIM_RESOURCE_PATH"] = (
            f"{self.root}:{env.get('GZ_SIM_RESOURCE_PATH', '')}"
        )
        env["PYTHONUNBUFFERED"] = "1"
        processes: list[tuple[subprocess.Popen, object]] = []
        with self.lock:
            self.active_processes[worker_id] = processes
        rollout_return_code = -1
        try:
            gazebo_command = ["gz", "sim", "-r"]
            if not (self.args.gui_first_worker and worker_id == 0):
                gazebo_command.append("-s")
            gazebo_command.append(
                str(self.root / "worlds" / "kookmin_xycar_track_final.sdf")
            )
            processes.append(
                _launch(
                    gazebo_command,
                    env=env,
                    log_path=logs / "gazebo.log",
                    project_root=self.root,
                )
            )
            if not _wait_for_gazebo(env, timeout_sec=20.0):
                raise TimeoutError("Gazebo control service did not appear")
            processes.append(
                _launch(
                    [
                        "ros2",
                        "launch",
                        "xycar_gazebo_bridge",
                        "xycar_gazebo_bridge.launch.py",
                        "auto_start:=true",
                    ],
                    env=env,
                    log_path=logs / "bridge.log",
                    project_root=self.root,
                )
            )
            time.sleep(2.0)
            processes.append(
                _launch(
                    [
                        "ros2",
                        "launch",
                        "lane_seg_control",
                        "lane_seg_lraspp_sim_canonical.launch.py",
                    ],
                    env=env,
                    log_path=logs / "perception.log",
                    project_root=self.root,
                )
            )
            time.sleep(3.0)
            processes.append(
                _launch(
                    [
                        "ros2",
                        "run",
                        "xycar_rl",
                        "rl_transition_recorder",
                        "--ros-args",
                        "-p",
                        f"output_dir:={transition_dir}",
                        "-p",
                        "require_lidar:=false",
                        "-p",
                        "require_action_trace:=true",
                        "-p",
                        (
                            "require_expert_action_trace:="
                            f"{str(self.args.trace_privileged_expert).lower()}"
                        ),
                        "-p",
                        "auto_stop_on_terminal:=false",
                        "-p",
                        f"max_rate_hz:={self.args.control_rate_hz}",
                        "-p",
                        (
                            "target_right_offset_m:="
                            f"{self.args.target_right_offset_m}"
                        ),
                        "-p",
                        (
                            "off_track_threshold_m:="
                            f"{self.args.off_track_threshold_m}"
                        ),
                    ],
                    env=env,
                    log_path=logs / "recorder.log",
                    project_root=self.root,
                )
            )
            time.sleep(0.5)
            rollout_command = [
                "ros2",
                "run",
                "xycar_rl",
                "rollout_policy",
                "--project-root",
                str(self.root),
                "--policy-kind",
                "camera_speed_td3_bc",
                "--checkpoint",
                str(self.checkpoint),
                "--episodes",
                str(episodes),
                "--max-steps",
                str(self.args.max_steps),
                "--control-rate-hz",
                str(self.args.control_rate_hz),
                "--min-speed-command",
                str(self.min_speed),
                "--max-speed-command",
                str(self.max_speed),
                "--target-right-offset-m",
                str(self.args.target_right_offset_m),
                "--speed-cap-command",
                str(self.args.speed_cap_command),
                "--action-noise",
                str(self.args.action_noise),
                "--speed-action-noise",
                str(self.args.speed_action_noise),
                "--speed-action-bias",
                str(self.args.speed_action_bias),
                "--steering-temporal-alpha",
                "0.20",
                "--straight-steering-temporal-alpha",
                "0.20",
                "--steering-straight-threshold",
                "1.0",
                "--steering-curve-threshold",
                "1.0",
                "--straight-steering-rate-limit",
                "0.05",
                "--curve-steering-rate-limit",
                "0.05",
                "--steering-deadband",
                "0.02",
                "--turn-in-anticipation-gain",
                "0.0",
                "--speed-temporal-alpha",
                "1.0",
                "--recovery-probability",
                str(self.args.recovery_probability),
                "--recovery-min-lateral-m",
                str(self.args.recovery_min_lateral_m),
                "--recovery-max-lateral-m",
                str(self.args.recovery_max_lateral_m),
                "--recovery-min-yaw-deg",
                str(self.args.recovery_min_yaw_deg),
                "--recovery-max-yaw-deg",
                str(self.args.recovery_max_yaw_deg),
                "--s-curve-focus-probability",
                str(self.args.s_curve_focus_probability),
                "--seed",
                str(self.args.seed + 100_003 * worker_id),
                "--device",
                self.args.device,
            ]
            if self.args.raw_steering_actions:
                rollout_command.append("--disable-adaptive-steering")
            if self.args.straight_segment_only:
                rollout_command.extend(
                    [
                        "--straight-segment-only",
                        "--straight-curvature-threshold",
                        str(self.args.straight_curvature_threshold),
                        "--straight-guard-distance-m",
                        str(self.args.straight_guard_distance_m),
                    ]
                )
            if self.args.trace_privileged_expert:
                rollout_command.append("--trace-privileged-expert")
            if self.args.start_progress_fraction is not None:
                rollout_command.extend(
                    [
                        "--start-progress-fraction",
                        str(self.args.start_progress_fraction),
                    ]
                )
            rollout, rollout_log = _launch(
                rollout_command,
                env=env,
                log_path=logs / "rollout.log",
                project_root=self.root,
            )
            processes.append((rollout, rollout_log))
            rollout.wait(
                timeout=max(
                    30.0,
                    float(self.args.episode_timeout_sec) * episodes,
                )
            )
            rollout_return_code = int(rollout.returncode or 0)
            time.sleep(1.0)
        except (OSError, subprocess.SubprocessError, TimeoutError) as exc:
            (worker_dir / "collector_error.txt").write_text(
                f"{type(exc).__name__}: {exc}\n",
                encoding="utf-8",
            )
        finally:
            _stop_processes(processes)
            with self.lock:
                self.active_processes.pop(worker_id, None)
        apply_rollout_terminals(
            transition_dir / "transitions.csv",
            logs / "rollout.log",
        )
        rows, recorded, complete, failed, rate_hz = _session_summary(
            transition_dir / "transitions.csv"
        )
        result = WorkerResult(
            worker_id=worker_id,
            requested_episodes=episodes,
            recorded_episodes=recorded,
            completed_laps=complete,
            failed_episodes=failed,
            transition_count=rows,
            observed_rate_hz=rate_hz,
            rollout_return_code=rollout_return_code,
            output_dir=str(worker_dir),
        )
        with self.lock:
            self.results.append(result)
            print(
                f"worker={worker_id} episodes={recorded}/{episodes} "
                f"complete={complete} failed={failed} rows={rows} "
                f"rate={rate_hz:.2f}Hz rc={rollout_return_code}",
                flush=True,
            )

    def run(self) -> int:
        started_at = time.monotonic()
        allocation = _episode_allocation(self.args.episodes, self.args.workers)
        threads = [
            threading.Thread(
                target=self._run_worker,
                args=(worker_id, episodes),
                daemon=False,
            )
            for worker_id, episodes in enumerate(allocation)
        ]
        for thread in threads:
            thread.start()
        try:
            for thread in threads:
                thread.join()
        except KeyboardInterrupt:
            self.stop_event.set()
            with self.lock:
                active = list(self.active_processes.values())
            for processes in active:
                _stop_processes(processes)
            for thread in threads:
                thread.join()
        ordered = sorted(self.results, key=lambda item: item.worker_id)
        wall_time_sec = max(0.0, time.monotonic() - started_at)
        transition_count = sum(item.transition_count for item in ordered)
        manifest = {
            "project_root": str(self.root),
            "checkpoint": str(self.checkpoint),
            "min_speed_command": self.min_speed,
            "max_speed_command": self.max_speed,
            "speed_cap_command": self.args.speed_cap_command,
            "raw_steering_actions": self.args.raw_steering_actions,
            "straight_segment_only": self.args.straight_segment_only,
            "straight_curvature_threshold": (
                self.args.straight_curvature_threshold
            ),
            "straight_guard_distance_m": self.args.straight_guard_distance_m,
            "trace_privileged_expert": self.args.trace_privileged_expert,
            "recovery_probability": self.args.recovery_probability,
            "recovery_min_lateral_m": self.args.recovery_min_lateral_m,
            "recovery_max_lateral_m": self.args.recovery_max_lateral_m,
            "recovery_min_yaw_deg": self.args.recovery_min_yaw_deg,
            "recovery_max_yaw_deg": self.args.recovery_max_yaw_deg,
            "straight_steering_stabilizer": {
                "alpha": 0.20,
                "rate_limit": 0.05,
                "deadband": 0.02,
                "zero_crossing_threshold": 0.12,
            },
            "control_rate_hz": self.args.control_rate_hz,
            "workers": len(allocation),
            "requested_episodes": sum(allocation),
            "recorded_episodes": sum(item.recorded_episodes for item in ordered),
            "completed_laps": sum(item.completed_laps for item in ordered),
            "failed_episodes": sum(item.failed_episodes for item in ordered),
            "transition_count": transition_count,
            "steady_aggregate_rate_hz": sum(
                item.observed_rate_hz for item in ordered
            ),
            "wall_time_sec": wall_time_sec,
            "wall_clock_transition_rate_hz": (
                transition_count / wall_time_sec if wall_time_sec > 0.0 else 0.0
            ),
            "failed_episodes_retained_for_critic": True,
            "results": [asdict(item) for item in ordered],
        }
        (self.output_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
        return int(
            manifest["recorded_episodes"] <= 0
            or manifest["transition_count"] <= 0
        )


def main(argv=None) -> None:
    raise SystemExit(ParallelPolicyCollector(parse_args(argv)).run())


if __name__ == "__main__":
    main()
