from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PreviewSteeringEstimate:
    steering_norm: float
    curve_hint: float
    confidence: float


class CanonicalPreviewSteering:
    """Estimate anticipatory steering from the yellow line in canonical BEV."""

    def __init__(
        self,
        *,
        minimum_band_pixels: int = 3,
        hold_frames: int = 5,
        hold_decay: float = 0.90,
    ) -> None:
        self.minimum_band_pixels = max(1, int(minimum_band_pixels))
        self.hold_frames = max(0, int(hold_frames))
        self.hold_decay = float(np.clip(hold_decay, 0.0, 1.0))
        self.last_steering = 0.0
        self.last_curve_hint = 0.0
        self.missing_frames = self.hold_frames + 1

    def reset(self) -> None:
        self.last_steering = 0.0
        self.last_curve_hint = 0.0
        self.missing_frames = self.hold_frames + 1

    def update(self, image_chw: np.ndarray) -> PreviewSteeringEstimate:
        image = np.asarray(image_chw, dtype=np.float32)
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("canonical model image must have shape (3, height, width)")
        red, green, blue = image
        yellow = (
            (red > 0.67)
            & (green > 0.51)
            & (blue < 0.47)
            & ((red + green - 2.0 * blue) > 0.70)
        )
        height, width = yellow.shape
        bands = (
            (round(0.69 * height), height),
            (round(0.42 * height), round(0.69 * height)),
            (round(0.14 * height), round(0.42 * height)),
        )
        positions: list[float | None] = []
        for low, high in bands:
            _, x_values = np.where(yellow[low:high])
            positions.append(
                float(np.median(x_values))
                if x_values.size >= self.minimum_band_pixels
                else None
            )

        if sum(position is not None for position in positions) < 2:
            return self._held_estimate()
        near, middle, far = self._fill_missing(positions)
        center = 0.5 * float(width - 1)
        half_width = max(1.0, center)
        offsets = np.asarray(
            [(near - center) / half_width, (middle - center) / half_width, (far - center) / half_width],
            dtype=np.float64,
        )
        # Calibrated on the 2026-07-17 stabilized expert set without an
        # intercept, so a centered straight line always commands zero.
        weights = np.asarray([0.5603, 0.4298, 0.4191], dtype=np.float64)
        steering = float(np.clip(np.dot(weights, offsets), -1.0, 1.0))
        if abs(steering) < 0.015:
            steering = 0.0
        curve_hint = float(np.clip(abs(far - near) / (0.32 * width), 0.0, 1.0))
        valid_bands = sum(position is not None for position in positions)
        confidence = 1.0 if valid_bands == 3 else 0.72
        self.last_steering = steering
        self.last_curve_hint = curve_hint
        self.missing_frames = 0
        return PreviewSteeringEstimate(steering, curve_hint, confidence)

    @staticmethod
    def _fill_missing(positions: list[float | None]) -> tuple[float, float, float]:
        near, middle, far = positions
        if near is None:
            near = 2.0 * float(middle) - float(far)
        elif middle is None:
            middle = 0.5 * (float(near) + float(far))
        elif far is None:
            far = 2.0 * float(middle) - float(near)
        return float(near), float(middle), float(far)

    def _held_estimate(self) -> PreviewSteeringEstimate:
        self.missing_frames += 1
        if self.missing_frames > self.hold_frames:
            self.last_steering = 0.0
            self.last_curve_hint = 0.0
            return PreviewSteeringEstimate(0.0, 0.0, 0.0)
        self.last_steering *= self.hold_decay
        self.last_curve_hint *= self.hold_decay
        confidence = 0.60 * self.hold_decay**self.missing_frames
        return PreviewSteeringEstimate(
            self.last_steering,
            self.last_curve_hint,
            confidence,
        )
