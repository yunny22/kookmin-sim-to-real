from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TerminationConfig:
    off_track_threshold_m: float = 0.38
    stuck_speed_mps: float = 0.03
    stuck_command_threshold: float = 3.0
    stuck_timeout_sec: float = 2.0
    max_episode_sec: float = 120.0
    minimum_lap_fraction: float = 0.95


@dataclass(frozen=True)
class TerminationResult:
    terminated: bool
    truncated: bool
    reason: str
    collision: bool
    off_track: bool
    stuck: bool
    lap_complete: bool


class EpisodeTermination:
    def __init__(self, config: TerminationConfig = TerminationConfig()) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.stuck_duration_sec = 0.0

    def update(
        self,
        *,
        dt_sec: float,
        elapsed_sec: float,
        cross_track_error_m: float,
        linear_speed_mps: float,
        speed_command: float,
        collision: bool,
        cumulative_forward_progress_m: float,
        track_length_m: float,
        observation_timed_out: bool = False,
    ) -> TerminationResult:
        commanded_to_move = abs(speed_command) >= self.config.stuck_command_threshold
        moving = abs(linear_speed_mps) >= self.config.stuck_speed_mps
        if commanded_to_move and not moving:
            self.stuck_duration_sec += max(0.0, float(dt_sec))
        else:
            self.stuck_duration_sec = 0.0

        off_track = abs(cross_track_error_m) > self.config.off_track_threshold_m
        stuck = self.stuck_duration_sec >= self.config.stuck_timeout_sec
        lap_complete = (
            cumulative_forward_progress_m
            >= self.config.minimum_lap_fraction * track_length_m
        )
        truncated = (
            elapsed_sec >= self.config.max_episode_sec
            or observation_timed_out
        )
        terminated = bool(collision or off_track or stuck or lap_complete)
        reason = "running"
        if collision:
            reason = "collision"
        elif off_track:
            reason = "off_track"
        elif stuck:
            reason = "stuck"
        elif lap_complete:
            reason = "lap_complete"
        elif observation_timed_out:
            reason = "observation_timeout"
        elif elapsed_sec >= self.config.max_episode_sec:
            reason = "time_limit"
        return TerminationResult(
            terminated=terminated,
            truncated=bool(truncated and not terminated),
            reason=reason,
            collision=bool(collision),
            off_track=bool(off_track),
            stuck=bool(stuck),
            lap_complete=bool(lap_complete),
        )
