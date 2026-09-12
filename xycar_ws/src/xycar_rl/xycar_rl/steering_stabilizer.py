from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SteeringStabilizerConfig:
    """Normalized steering filter that preserves curve turn-in response."""

    straight_alpha: float = 0.20
    curve_alpha: float = 0.90
    straight_threshold: float = 0.08
    curve_threshold: float = 0.35
    straight_rate_limit: float = 0.05
    curve_rate_limit: float = 0.38
    deadband: float = 0.02
    zero_crossing_threshold: float = 0.12
    turn_in_anticipation_gain: float = 0.35
    turn_in_anticipation_threshold: float = 0.04


def _smoothstep(value: float) -> float:
    clipped = float(np.clip(value, 0.0, 1.0))
    return clipped * clipped * (3.0 - 2.0 * clipped)


class AdaptiveSteeringStabilizer:
    """Damp small straight-line corrections without delaying large turns."""

    def __init__(self, config: SteeringStabilizerConfig | None = None) -> None:
        self.config = config or SteeringStabilizerConfig()
        self.value = 0.0
        self.raw_value = 0.0

    def reset(self, value: float = 0.0) -> None:
        self.value = float(np.clip(value, -1.0, 1.0))
        self.raw_value = self.value

    def update(self, raw_value: float, *, curve_hint: float = 0.0) -> float:
        config = self.config
        raw_target = float(np.clip(raw_value, -1.0, 1.0))
        if abs(raw_target) < max(0.0, config.deadband):
            raw_target = 0.0

        target = raw_target
        if (
            target * self.raw_value >= 0.0
            and abs(target) > abs(self.raw_value)
            and abs(target) >= max(0.0, config.turn_in_anticipation_threshold)
        ):
            target += float(config.turn_in_anticipation_gain) * (
                target - self.raw_value
            )
            target = float(np.clip(target, -1.0, 1.0))
        self.raw_value = raw_target

        denominator = max(
            1.0e-6,
            float(config.curve_threshold) - float(config.straight_threshold),
        )
        magnitude = max(abs(target), abs(self.value))
        command_blend = _smoothstep(
            (magnitude - float(config.straight_threshold)) / denominator
        )
        curve_blend = max(command_blend, float(np.clip(curve_hint, 0.0, 1.0)))
        alpha = float(config.straight_alpha) + curve_blend * (
            float(config.curve_alpha) - float(config.straight_alpha)
        )
        rate_limit = float(config.straight_rate_limit) + curve_blend * (
            float(config.curve_rate_limit) - float(config.straight_rate_limit)
        )
        alpha = float(np.clip(alpha, 0.0, 1.0))
        rate_limit = max(0.0, rate_limit)

        if (
            self.value * target < 0.0
            and max(abs(self.value), abs(target))
            < max(0.0, float(config.zero_crossing_threshold))
        ):
            target = 0.0

        filtered = alpha * target + (1.0 - alpha) * self.value
        delta = float(np.clip(filtered - self.value, -rate_limit, rate_limit))
        self.value = float(np.clip(self.value + delta, -1.0, 1.0))
        if abs(self.value) < 1.0e-6:
            self.value = 0.0
        return self.value
