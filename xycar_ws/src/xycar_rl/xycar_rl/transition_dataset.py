from __future__ import annotations

from collections import deque
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from il_data_tools.runtime_preprocessing import (
    preprocess_bgr_image,
    preprocess_lidar_ranges,
)
from xycar_rl.camera_speed_models import (
    DEFAULT_MAX_SPEED_COMMAND,
    DEFAULT_MIN_SPEED_COMMAND,
    normalize_speed_command,
)
from xycar_rl.reward import RewardWeights, calculate_reward
from xycar_rl.track_geometry import TrackProjection, TrackReference


def find_transition_csvs(paths: list[str | Path]) -> list[Path]:
    found: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if path.is_file():
            found.append(path)
        elif (path / "transitions.csv").is_file():
            found.append(path / "transitions.csv")
        elif path.is_dir():
            found.extend(sorted(path.rglob("transitions.csv")))
        else:
            raise FileNotFoundError(path)
    unique = list(dict.fromkeys(found))
    if not unique:
        raise ValueError("no transitions.csv files were found")
    return unique


def camera_speed_action_targets(
    row: dict[str, str],
    min_speed_command: float,
    max_speed_command: float,
    *,
    bc_speed_target_command: float | None = None,
) -> tuple[list[float], list[float]]:
    speed_norm = normalize_speed_command(
        float(row["speed_command"]),
        min_speed_command,
        max_speed_command,
    )
    applied_action = [float(row["action_norm"]), speed_norm]
    expert_action_norm = row.get("expert_action_norm", "")
    expert_speed_command = row.get("expert_speed_command", "")
    if expert_action_norm == "" or expert_speed_command == "":
        bc_action = applied_action.copy()
    else:
        bc_action = [
            float(expert_action_norm),
            normalize_speed_command(
                float(expert_speed_command),
                min_speed_command,
                max_speed_command,
            ),
        ]
    if bc_speed_target_command is not None:
        bc_action[1] = normalize_speed_command(
            min(
                max(float(bc_speed_target_command), min_speed_command),
                max_speed_command,
            ),
            min_speed_command,
            max_speed_command,
        )
    return applied_action, bc_action


def successful_bc_episode_keys(
    rows: list[tuple[Path, dict[str, str]]],
) -> set[tuple[Path, str]]:
    return {
        (root, row.get("episode_id", "0"))
        for root, row in rows
        if str(row.get("termination_reason") or "")
        in {"lap_complete", "straight_segment_complete"}
    }


class RLTransitionDataset(Dataset):
    def __init__(
        self,
        paths: list[str | Path],
        *,
        input_width: int = 160,
        input_height: int = 90,
        lidar_points: int = 360,
    ) -> None:
        self.input_width = int(input_width)
        self.input_height = int(input_height)
        self.lidar_points = int(lidar_points)
        self.rows: list[tuple[Path, dict[str, str]]] = []
        for csv_path in find_transition_csvs(paths):
            with csv_path.open(newline="", encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    self.rows.append((csv_path.parent, row))
        if not self.rows:
            raise ValueError("transition dataset has no rows")

    def __len__(self) -> int:
        return len(self.rows)

    def _image(self, root: Path, relative_path: str) -> torch.Tensor:
        image = cv2.imread(str(root / relative_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(root / relative_path)
        array = preprocess_bgr_image(
            image, self.input_width, self.input_height
        )
        return torch.from_numpy(array)

    def _lidar(self, root: Path, relative_path: str) -> torch.Tensor:
        with np.load(root / relative_path) as payload:
            ranges = payload["ranges"]
            range_min = float(payload.get("range_min", 0.1))
            range_max = float(payload.get("range_max", 12.0))
        array = preprocess_lidar_ranges(
            ranges, range_min, range_max, self.lidar_points
        )
        return torch.from_numpy(array)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        root, row = self.rows[index]
        done = bool(int(row.get("terminated", "0"))) or bool(
            int(row.get("truncated", "0"))
        )
        return {
            "image": self._image(root, row["state_image_path"]),
            "lidar": self._lidar(root, row["state_scan_path"]),
            "action": torch.tensor(
                [float(row["action_norm"])], dtype=torch.float32
            ),
            "reward": torch.tensor(
                [float(row["reward"])], dtype=torch.float32
            ),
            "next_image": self._image(root, row["next_image_path"]),
            "next_lidar": self._lidar(root, row["next_scan_path"]),
            "done": torch.tensor([float(done)], dtype=torch.float32),
        }


class CameraSpeedTransitionDataset(RLTransitionDataset):
    """Camera-only transitions with normalized steering and speed actions."""

    def __init__(
        self,
        paths: list[str | Path],
        *,
        min_speed_command: float = DEFAULT_MIN_SPEED_COMMAND,
        max_speed_command: float = DEFAULT_MAX_SPEED_COMMAND,
        temporal_frames: int = 1,
        recompute_rewards: bool = False,
        world_sdf: str | Path | None = None,
        target_right_offset_m: float = 0.0,
        reward_weights: RewardWeights = RewardWeights(),
        bc_successful_episodes_only: bool = False,
        straight_only: bool = False,
        straight_curvature_threshold: float = 0.10,
        straight_guard_distance_m: float = 0.80,
        bc_speed_target_command: float | None = None,
        input_width: int = 160,
        input_height: int = 90,
    ) -> None:
        super().__init__(
            paths,
            input_width=input_width,
            input_height=input_height,
        )
        self.min_speed_command = float(min_speed_command)
        self.max_speed_command = float(max_speed_command)
        self.target_right_offset_m = float(target_right_offset_m)
        self.reward_weights = reward_weights
        self.bc_speed_target_command = bc_speed_target_command
        self.original_row_count = len(self.rows)
        self.successful_episode_keys = successful_bc_episode_keys(self.rows)
        self.straight_only = bool(straight_only)
        self.straight_curvature_threshold = max(
            0.0, float(straight_curvature_threshold)
        )
        self.straight_guard_distance_m = max(
            0.0, float(straight_guard_distance_m)
        )
        if self.straight_only:
            if world_sdf is None:
                raise ValueError("world_sdf is required when straight_only is true")
            straight_track = TrackReference.from_sdf(
                Path(world_sdf).expanduser().resolve(),
                target_right_offset_m=self.target_right_offset_m,
            )
            self.rows = [
                (root, row)
                for root, row in self.rows
                if self._is_straight_transition(row, straight_track)
            ]
            if not self.rows:
                raise ValueError(
                    "straight_only filtering removed every transition row"
                )
        self.straight_row_count = len(self.rows)
        self.bc_successful_episodes_only = bool(
            bc_successful_episodes_only
        )
        self.temporal_frames = int(temporal_frames)
        if self.temporal_frames not in {1, 2}:
            raise ValueError("temporal_frames must be 1 or 2")
        self.previous_image_paths: list[str] = []
        self.recomputed_rewards: list[float] | None = (
            [] if recompute_rewards else None
        )
        previous_by_episode: dict[
            tuple[Path, str], tuple[dict[str, str], str]
        ] = {}
        for root, row in self.rows:
            episode_key = (root, row.get("episode_id", "0"))
            current_path = row["state_image_path"]
            previous_entry = previous_by_episode.get(episode_key)
            previous_path = current_path
            if previous_entry is not None and _rows_are_contiguous(
                previous_entry[0], row
            ):
                previous_path = previous_entry[1]
            self.previous_image_paths.append(previous_path)
            previous_by_episode[episode_key] = (row, current_path)
        if self.recomputed_rewards is not None:
            if world_sdf is None:
                raise ValueError("world_sdf is required when recompute_rewards is true")
            self._recompute_rewards(Path(world_sdf).expanduser().resolve())

    def _is_straight_transition(
        self,
        row: dict[str, str],
        track: TrackReference,
    ) -> bool:
        progress_m = float(row.get("progress_m") or 0.0)
        next_progress_m = progress_m + float(
            row.get("progress_delta_m") or 0.0
        )
        threshold = self.straight_curvature_threshold
        guard = self.straight_guard_distance_m
        return all(
            abs(track.curvature_at(sample, sample_distance_m=0.30))
            <= threshold
            and track.max_abs_curvature_ahead(
                sample,
                preview_distance_m=guard,
            )
            <= threshold
            for sample in (progress_m, next_progress_m)
        )

    def _recompute_rewards(self, world_sdf: Path) -> None:
        track = TrackReference.from_sdf(
            world_sdf,
            target_right_offset_m=self.target_right_offset_m,
        )
        previous_actions: dict[tuple[Path, str], float] = {}
        steering_histories: dict[tuple[Path, str], deque[float]] = {}
        previous_rows: dict[tuple[Path, str], dict[str, str]] = {}
        assert self.recomputed_rewards is not None
        for root, row in self.rows:
            episode_key = (root, row.get("episode_id", "0"))
            action_norm = float(row.get("action_norm") or 0.0)
            history = steering_histories.setdefault(
                episode_key, deque(maxlen=12)
            )
            previous_row = previous_rows.get(episode_key)
            if previous_row is not None and not _rows_are_contiguous(
                previous_row, row
            ):
                history.clear()
                previous_actions.pop(episode_key, None)
            history.append(action_norm)
            progress_m = float(row.get("progress_m") or 0.0)
            projection = TrackProjection(
                x=0.0,
                y=0.0,
                tangent_yaw=0.0,
                progress_m=progress_m,
                progress_fraction=(progress_m / max(track.length_m, 1.0e-6)) % 1.0,
                cross_track_error_m=float(
                    row.get("cross_track_error_m") or 0.0
                ),
                heading_error_rad=float(row.get("heading_error_rad") or 0.0),
                segment_index=0,
            )
            reason = str(row.get("termination_reason") or "")
            try:
                dt_sec = max(
                    0.001,
                    (
                        int(row["next_timestamp_ns"])
                        - int(row["state_timestamp_ns"])
                    )
                    / 1.0e9,
                )
            except (KeyError, TypeError, ValueError):
                dt_sec = 0.1
            reward = calculate_reward(
                projection=projection,
                progress_delta_m=float(row.get("progress_delta_m") or 0.0),
                steering_norm=action_norm,
                previous_steering_norm=previous_actions.get(episode_key, 0.0),
                linear_speed_mps=float(row.get("speed_mps") or 0.0),
                track_curvature=track.curvature_at(
                    progress_m, sample_distance_m=0.30
                ),
                preview_curvature=track.max_abs_curvature_ahead(
                    progress_m,
                    preview_distance_m=1.5,
                ),
                steering_history=tuple(history),
                collision=bool(int(row.get("collision") or 0)),
                off_track=reason == "off_track",
                stuck=reason == "stuck",
                lap_complete=reason == "lap_complete",
                dt_sec=dt_sec,
                weights=self.reward_weights,
            )
            self.recomputed_rewards.append(reward.total)
            previous_actions[episode_key] = action_norm
            previous_rows[episode_key] = row

    def _temporal_image(
        self,
        root: Path,
        previous_path: str,
        current_path: str,
    ) -> torch.Tensor:
        current = self._image(root, current_path)
        if self.temporal_frames == 1:
            return current
        previous = self._image(root, previous_path)
        return torch.cat([previous, current], dim=0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        root, row = self.rows[index]
        done = bool(int(row.get("terminated", "0"))) or bool(
            int(row.get("truncated", "0"))
        )
        applied_action, bc_action = camera_speed_action_targets(
            row,
            self.min_speed_command,
            self.max_speed_command,
            bc_speed_target_command=self.bc_speed_target_command,
        )
        return {
            "image": self._temporal_image(
                root,
                self.previous_image_paths[index],
                row["state_image_path"],
            ),
            "action": torch.tensor(applied_action, dtype=torch.float32),
            "bc_action": torch.tensor(bc_action, dtype=torch.float32),
            "bc_weight": torch.tensor(
                [
                    float(
                        not self.bc_successful_episodes_only
                        or (
                            root,
                            row.get("episode_id", "0"),
                        )
                        in self.successful_episode_keys
                    )
                ],
                dtype=torch.float32,
            ),
            "reward": torch.tensor(
                [
                    self.recomputed_rewards[index]
                    if self.recomputed_rewards is not None
                    else float(row["reward"])
                ],
                dtype=torch.float32,
            ),
            "next_image": self._temporal_image(
                root,
                row["state_image_path"],
                row["next_image_path"],
            ),
            "done": torch.tensor([float(done)], dtype=torch.float32),
        }


def _rows_are_contiguous(
    previous: dict[str, str], current: dict[str, str]
) -> bool:
    if previous.get("episode_id", "0") != current.get("episode_id", "0"):
        return False
    try:
        return (
            int(previous["step_id"]) + 1 == int(current["step_id"])
            and int(previous["next_timestamp_ns"])
            == int(current["state_timestamp_ns"])
        )
    except (KeyError, TypeError, ValueError):
        return False
