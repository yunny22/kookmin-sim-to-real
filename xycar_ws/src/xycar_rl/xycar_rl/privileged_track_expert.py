from __future__ import annotations

import math

import numpy as np

from xycar_rl.camera_speed_models import normalize_speed_command
from xycar_rl.steering_stabilizer import (
    AdaptiveSteeringStabilizer,
    SteeringStabilizerConfig,
)
from xycar_rl.track_geometry import TrackReference, wrap_angle


STEERING_COMMANDS = np.asarray(
    [-42, -40, -35, -30, -20, -10, 0, 10, 20, 30, 35, 40, 42],
    dtype=np.float64,
)
MEASURED_CURVATURES = np.asarray(
    [
        1.502435,
        1.383494,
        1.174860,
        0.922781,
        0.552809,
        0.194230,
        0.0,
        -0.556883,
        -0.959829,
        -1.369323,
        -1.601706,
        -1.853397,
        -1.939236,
    ],
    dtype=np.float64,
)


def command_for_curvature(curvature: float) -> float:
    return float(
        np.interp(
            float(curvature),
            MEASURED_CURVATURES[::-1],
            STEERING_COMMANDS[::-1],
        )
    )


class PrivilegedTrackExpert:
    """Pose-based expert used only to generate high-speed simulation labels."""

    def __init__(
        self,
        track: TrackReference,
        *,
        min_speed_command: float = 4.0,
        max_speed_command: float = 25.0,
        action_min_speed_command: float | None = None,
        action_max_speed_command: float | None = None,
        max_steering_command: float = 42.0,
        base_lookahead_m: float = 0.28,
        speed_lookahead_gain: float = 0.45,
        steering_gain: float = 1.0,
        curvature_preview_m: float = 1.15,
        full_speed_curvature: float = 0.18,
        minimum_speed_curvature: float = 1.10,
        curve_lookahead_reduction: float = 0.0,
        minimum_lookahead_m: float = 0.24,
        feedback_blend: float = 0.35,
        cross_track_gain: float = 2.0,
        heading_gain: float = 1.4,
        stabilize_steering: bool = True,
        steering_stabilizer_config: SteeringStabilizerConfig | None = None,
    ) -> None:
        self.track = track
        self.min_speed_command = float(min_speed_command)
        self.max_speed_command = float(max_speed_command)
        self.action_min_speed_command = float(
            min_speed_command
            if action_min_speed_command is None
            else action_min_speed_command
        )
        self.action_max_speed_command = float(
            max_speed_command
            if action_max_speed_command is None
            else action_max_speed_command
        )
        self.max_steering_command = abs(float(max_steering_command))
        self.base_lookahead_m = float(base_lookahead_m)
        self.speed_lookahead_gain = float(speed_lookahead_gain)
        self.steering_gain = float(steering_gain)
        self.curvature_preview_m = float(curvature_preview_m)
        self.full_speed_curvature = float(full_speed_curvature)
        self.minimum_speed_curvature = float(minimum_speed_curvature)
        self.curve_lookahead_reduction = float(curve_lookahead_reduction)
        self.minimum_lookahead_m = float(minimum_lookahead_m)
        self.feedback_blend = float(np.clip(feedback_blend, 0.0, 1.0))
        self.cross_track_gain = float(cross_track_gain)
        self.heading_gain = float(heading_gain)
        self.stabilize_steering = bool(stabilize_steering)
        self.steering_stabilizer = AdaptiveSteeringStabilizer(
            steering_stabilizer_config
        )

    def reset(self) -> None:
        self.steering_stabilizer.reset()

    def _upcoming_curvature(self, progress_m: float) -> float:
        offsets = np.linspace(0.12, self.curvature_preview_m, 10)
        return max(
            abs(self.track.curvature_at(progress_m + float(offset)))
            for offset in offsets
        )

    def _speed_command(
        self,
        progress_m: float,
        cross_track_error_m: float = 0.0,
        heading_error_rad: float = 0.0,
    ) -> float:
        upcoming = self._upcoming_curvature(progress_m)
        denominator = max(
            1.0e-6,
            self.minimum_speed_curvature - self.full_speed_curvature,
        )
        slow_fraction = float(
            np.clip(
                (upcoming - self.full_speed_curvature) / denominator,
                0.0,
                1.0,
            )
        )
        slow_fraction = slow_fraction * slow_fraction * (3.0 - 2.0 * slow_fraction)
        recovery_fraction = max(
            float(np.clip(abs(cross_track_error_m) / 0.14, 0.0, 1.0)),
            float(
                np.clip(
                    abs(heading_error_rad) / math.radians(12.0),
                    0.0,
                    1.0,
                )
            ),
        )
        slow_fraction = max(slow_fraction, recovery_fraction)
        return float(
            self.max_speed_command
            - slow_fraction * (self.max_speed_command - self.min_speed_command)
        )

    def action(self, info: dict) -> np.ndarray:
        x = float(info["x"])
        y = float(info["y"])
        yaw = float(info["yaw"])
        speed_mps = abs(float(info.get("speed_mps", 0.0)))
        progress_m = float(info["progress_m"])
        upcoming = self._upcoming_curvature(progress_m)
        curve_fraction = float(
            np.clip(upcoming / max(0.1, self.minimum_speed_curvature), 0.0, 1.0)
        )
        nominal_lookahead = (
            self.base_lookahead_m + self.speed_lookahead_gain * speed_mps
        )
        lookahead = max(
            self.minimum_lookahead_m,
            nominal_lookahead
            * (1.0 - self.curve_lookahead_reduction * curve_fraction),
        )
        target, _ = self.track.sample_at_progress(progress_m + lookahead)
        target_yaw = math.atan2(float(target[1] - y), float(target[0] - x))
        alpha = wrap_angle(target_yaw - yaw)
        pursuit_curvature = 2.0 * math.sin(alpha) / max(0.05, lookahead)
        path_curvature = self.track.curvature_at(
            progress_m + min(0.25, 0.5 * lookahead),
            sample_distance_m=0.24,
        )
        feedback_curvature = (
            path_curvature
            - self.cross_track_gain * float(info["cross_track_error_m"])
            - self.heading_gain * float(info["heading_error_rad"])
        )
        desired_curvature = (
            (1.0 - self.feedback_blend) * pursuit_curvature
            + self.feedback_blend * feedback_curvature
        )
        command = command_for_curvature(desired_curvature) * self.steering_gain
        command = float(
            np.clip(command, -self.max_steering_command, self.max_steering_command)
        )
        steering_norm = command / self.max_steering_command
        if self.stabilize_steering:
            steering_norm = self.steering_stabilizer.update(
                steering_norm,
                curve_hint=curve_fraction,
            )
        speed_command = self._speed_command(
            progress_m,
            float(info["cross_track_error_m"]),
            float(info["heading_error_rad"]),
        )
        return np.asarray(
            [
                steering_norm,
                normalize_speed_command(
                    speed_command,
                    self.action_min_speed_command,
                    self.action_max_speed_command,
                ),
            ],
            dtype=np.float32,
        )
