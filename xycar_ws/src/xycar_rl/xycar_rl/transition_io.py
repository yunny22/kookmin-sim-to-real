from __future__ import annotations

import csv
import json
from pathlib import Path
import time

import cv2
import numpy as np


TRANSITION_COLUMNS = [
    "episode_id",
    "step_id",
    "state_timestamp_ns",
    "next_timestamp_ns",
    "state_image_path",
    "state_scan_path",
    "next_image_path",
    "next_scan_path",
    "action_norm",
    "angle_command",
    "speed_command",
    "action_source",
    "expert_action_norm",
    "expert_angle_command",
    "expert_speed_command",
    "expert_action_source",
    "reward",
    "reward_progress",
    "reward_cross_track",
    "reward_heading",
    "reward_steering_rate",
    "reward_safe_speed",
    "reward_unsafe_speed",
    "reward_time_efficiency",
    "reward_lap_time",
    "reward_lane_margin",
    "reward_large_oscillation",
    "reward_terminal",
    "terminated",
    "truncated",
    "termination_reason",
    "x",
    "y",
    "yaw",
    "next_x",
    "next_y",
    "next_yaw",
    "speed_mps",
    "cross_track_error_m",
    "heading_error_rad",
    "progress_m",
    "progress_delta_m",
    "cumulative_progress_m",
    "collision",
]


class TransitionWriter:
    def __init__(
        self,
        output_dir: str | Path,
        metadata: dict | None = None,
        *,
        allow_overwrite: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.images_dir = self.output_dir / "images"
        self.scans_dir = self.output_dir / "scan"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.scans_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "transitions.csv"
        if self.csv_path.exists() and not allow_overwrite:
            raise FileExistsError(
                f"transition session already exists: {self.csv_path}"
            )
        self.csv_file = self.csv_path.open("w", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.csv_file, fieldnames=TRANSITION_COLUMNS)
        self.writer.writeheader()
        self.count = 0
        self.observation_paths: dict[int, tuple[str, str]] = {}
        self.metadata = dict(metadata or {})
        self.metadata.setdefault("schema_version", 5)
        self.metadata.setdefault("created_unix_sec", time.time())

    def save_observation(
        self,
        timestamp_ns: int,
        image_bgr: np.ndarray,
        scan_payload: dict | None,
    ) -> tuple[str, str]:
        timestamp_ns = int(timestamp_ns)
        cached = self.observation_paths.get(timestamp_ns)
        if cached is not None:
            return cached
        image_relative = Path("images") / f"{timestamp_ns}.png"
        if not cv2.imwrite(str(self.output_dir / image_relative), image_bgr):
            raise OSError(f"failed to write image {image_relative}")
        if scan_payload is None:
            scan_path = ""
        else:
            scan_relative = Path("scan") / f"{timestamp_ns}.npz"
            np.savez_compressed(self.output_dir / scan_relative, **scan_payload)
            scan_path = scan_relative.as_posix()
        result = (image_relative.as_posix(), scan_path)
        self.observation_paths[timestamp_ns] = result
        return result

    def append(self, row: dict) -> None:
        unknown = set(row) - set(TRANSITION_COLUMNS)
        if unknown:
            raise ValueError(f"unknown transition columns: {sorted(unknown)}")
        self.writer.writerow({column: row.get(column, "") for column in TRANSITION_COLUMNS})
        self.count += 1
        if self.count % 20 == 0:
            self.csv_file.flush()

    def close(self) -> None:
        if not self.csv_file.closed:
            self.csv_file.flush()
            self.csv_file.close()
        self.metadata["transition_count"] = self.count
        self.metadata["observation_count"] = len(self.observation_paths)
        self.metadata["finished_unix_sec"] = time.time()
        with (self.output_dir / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(self.metadata, file, ensure_ascii=True, indent=2, sort_keys=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
