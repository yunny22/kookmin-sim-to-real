from __future__ import annotations

from dataclasses import dataclass
import math

from xycar_rl.track_geometry import TrackProjection


@dataclass(frozen=True)
class RewardWeights:
    progress: float = 8.0
    cross_track: float = 1.5
    heading: float = 0.8
    steering_rate: float = 0.05
    steering_magnitude: float = 0.01
    # Reward speed per metre of safe progress. Unlike a positive reward paid
    # every frame, this cannot be increased by deliberately taking more time.
    safe_speed: float = 6.0
    unsafe_speed: float = 0.80
    curve_overspeed: float = 1.20
    time_efficiency: float = 0.04
    # Optional objective terms. They default to zero so existing checkpoints
    # and reward recomputation remain unchanged.
    lap_time: float = 0.0
    lane_margin: float = 0.0
    lane_margin_start_m: float = 0.24
    lane_departure_threshold_m: float = 0.38
    large_oscillation: float = 0.20
    safe_cross_track_m: float = 0.12
    safe_heading_rad: float = math.radians(12.0)
    large_steering_threshold: float = 0.18
    straight_curvature_threshold: float = 0.35
    full_speed_curvature: float = 0.18
    minimum_speed_curvature: float = 0.90
    # Measured conversion is approximately 0.080612 m/s per command.
    # Straight target command 25 at the measured 0.080612 m/s per command.
    # The full-course legacy objective still slows sharp curves to command 17.
    straight_target_speed_mps: float = 2.015300
    curve_target_speed_mps: float = 1.370404
    reverse_progress: float = 4.0
    collision: float = 50.0
    off_track: float = 30.0
    stuck: float = 10.0
    lap_complete: float = 20.0


def lap_time_objective_weights(
    *,
    lane_margin_start_m: float = 0.24,
    lane_departure_threshold_m: float = 0.38,
) -> RewardWeights:
    """Prioritize safe completion, then minimize elapsed simulation time."""
    threshold = max(0.01, float(lane_departure_threshold_m))
    margin_start = min(
        max(0.0, float(lane_margin_start_m)),
        threshold - 1.0e-3,
    )
    return RewardWeights(
        progress=4.0,
        cross_track=0.0,
        heading=0.0,
        steering_rate=0.0,
        steering_magnitude=0.0,
        safe_speed=0.0,
        unsafe_speed=0.0,
        curve_overspeed=0.0,
        time_efficiency=0.0,
        lap_time=1.0,
        lane_margin=20.0,
        lane_margin_start_m=margin_start,
        lane_departure_threshold_m=threshold,
        large_oscillation=0.0,
        reverse_progress=20.0,
        collision=300.0,
        off_track=300.0,
        stuck=100.0,
        lap_complete=300.0,
    )


def straight_high_speed_objective_weights(
    *,
    target_speed_command: float = 25.0,
    speed_gain_mps_per_command: float = 0.080612,
) -> RewardWeights:
    """Reward fast, centered straight driving while strongly rejecting weave."""
    target_speed_mps = max(
        0.0,
        float(target_speed_command) * float(speed_gain_mps_per_command),
    )
    return RewardWeights(
        progress=8.0,
        cross_track=3.0,
        heading=2.0,
        steering_rate=0.40,
        steering_magnitude=0.08,
        safe_speed=8.0,
        unsafe_speed=1.5,
        curve_overspeed=0.0,
        time_efficiency=0.04,
        large_oscillation=2.0,
        straight_curvature_threshold=0.16,
        straight_target_speed_mps=target_speed_mps,
        curve_target_speed_mps=target_speed_mps,
        reverse_progress=8.0,
        collision=100.0,
        off_track=100.0,
        stuck=30.0,
        lap_complete=0.0,
    )


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    progress: float
    cross_track: float
    heading: float
    steering_rate: float
    steering_magnitude: float
    safe_speed: float
    unsafe_speed: float
    time_efficiency: float
    lap_time: float
    lane_margin: float
    large_oscillation: float
    terminal: float


def large_steering_oscillation(
    steering_history: tuple[float, ...] | list[float],
    *,
    threshold: float = 0.18,
    minimum_reversals: int = 2,
) -> float:
    """Return amplitude only for repeated large left-right-left motion."""
    strong = [
        float(value)
        for value in steering_history
        if abs(float(value)) >= max(0.0, float(threshold))
    ]
    if len(strong) < 3:
        return 0.0
    signs: list[int] = []
    for value in strong:
        sign = 1 if value > 0.0 else -1
        if not signs or sign != signs[-1]:
            signs.append(sign)
    reversals = len(signs) - 1
    if reversals < max(1, int(minimum_reversals)):
        return 0.0
    positive_peak = max((value for value in strong if value > 0.0), default=0.0)
    negative_peak = max((-value for value in strong if value < 0.0), default=0.0)
    bilateral_amplitude = 2.0 * min(positive_peak, negative_peak)
    return bilateral_amplitude * (reversals - minimum_reversals + 1)


def calculate_reward(
    *,
    projection: TrackProjection,
    progress_delta_m: float,
    steering_norm: float,
    previous_steering_norm: float,
    linear_speed_mps: float = 0.0,
    track_curvature: float = 0.0,
    preview_curvature: float | None = None,
    steering_history: tuple[float, ...] | list[float] = (),
    collision: bool = False,
    off_track: bool = False,
    stuck: bool = False,
    lap_complete: bool = False,
    dt_sec: float = 0.1,
    weights: RewardWeights = RewardWeights(),
) -> RewardBreakdown:
    forward_progress = max(0.0, float(progress_delta_m))
    reverse_progress = max(0.0, -float(progress_delta_m))
    progress_term = (
        weights.progress * forward_progress
        - weights.reverse_progress * reverse_progress
    )
    cross_track_term = -weights.cross_track * abs(
        projection.cross_track_error_m
    )
    heading_term = -weights.heading * abs(projection.heading_error_rad) / math.pi
    steering_rate_term = -weights.steering_rate * abs(
        float(steering_norm) - float(previous_steering_norm)
    )
    steering_magnitude_term = -weights.steering_magnitude * abs(
        float(steering_norm)
    )
    cross_track_ratio = abs(projection.cross_track_error_m) / max(
        1.0e-6, weights.safe_cross_track_m
    )
    heading_ratio = abs(projection.heading_error_rad) / max(
        1.0e-6, weights.safe_heading_rad
    )
    safe_factor = math.exp(
        -0.5 * (cross_track_ratio * cross_track_ratio + heading_ratio * heading_ratio)
    )
    forward_speed = max(0.0, float(linear_speed_mps))
    upcoming_curvature = abs(
        float(track_curvature)
        if preview_curvature is None
        else float(preview_curvature)
    )
    curvature_denominator = max(
        1.0e-6,
        weights.minimum_speed_curvature - weights.full_speed_curvature,
    )
    curve_fraction = max(
        0.0,
        min(
            1.0,
            (upcoming_curvature - weights.full_speed_curvature)
            / curvature_denominator,
        ),
    )
    curve_fraction = curve_fraction * curve_fraction * (3.0 - 2.0 * curve_fraction)
    target_speed_mps = (
        weights.straight_target_speed_mps
        - curve_fraction
        * (weights.straight_target_speed_mps - weights.curve_target_speed_mps)
    )
    speed_fraction = min(
        1.0,
        forward_speed / max(1.0e-6, target_speed_mps),
    )
    safe_speed_term = (
        weights.safe_speed
        * forward_progress
        * speed_fraction
        * safe_factor
    )
    tracking_risk = forward_speed * (1.0 - safe_factor)
    curve_speed_excess = max(0.0, forward_speed - target_speed_mps)
    unsafe_speed_term = -(
        weights.unsafe_speed * tracking_risk
        + weights.curve_overspeed * curve_speed_excess
    )
    time_efficiency_term = -weights.time_efficiency
    elapsed_step_sec = max(0.0, float(dt_sec))
    lap_time_term = -weights.lap_time * elapsed_step_sec
    margin_denominator = max(
        1.0e-6,
        weights.lane_departure_threshold_m - weights.lane_margin_start_m,
    )
    lane_margin_ratio = max(
        0.0,
        min(
            1.0,
            (
                abs(projection.cross_track_error_m)
                - weights.lane_margin_start_m
            )
            / margin_denominator,
        ),
    )
    lane_margin_term = (
        -weights.lane_margin
        * lane_margin_ratio
        * lane_margin_ratio
        * elapsed_step_sec
    )
    oscillation_measure = 0.0
    if abs(float(track_curvature)) <= weights.straight_curvature_threshold:
        oscillation_measure = large_steering_oscillation(
            steering_history,
            threshold=weights.large_steering_threshold,
        )
    large_oscillation_term = -weights.large_oscillation * oscillation_measure
    terminal_term = 0.0
    terminal_term -= weights.collision if collision else 0.0
    terminal_term -= weights.off_track if off_track else 0.0
    terminal_term -= weights.stuck if stuck else 0.0
    terminal_term += weights.lap_complete if lap_complete else 0.0
    total = (
        progress_term
        + cross_track_term
        + heading_term
        + steering_rate_term
        + steering_magnitude_term
        + safe_speed_term
        + unsafe_speed_term
        + time_efficiency_term
        + lap_time_term
        + lane_margin_term
        + large_oscillation_term
        + terminal_term
    )
    return RewardBreakdown(
        total=float(total),
        progress=float(progress_term),
        cross_track=float(cross_track_term),
        heading=float(heading_term),
        steering_rate=float(steering_rate_term),
        steering_magnitude=float(steering_magnitude_term),
        safe_speed=float(safe_speed_term),
        unsafe_speed=float(unsafe_speed_term),
        time_efficiency=float(time_efficiency_term),
        lap_time=float(lap_time_term),
        lane_margin=float(lane_margin_term),
        large_oscillation=float(large_oscillation_term),
        terminal=float(terminal_term),
    )
