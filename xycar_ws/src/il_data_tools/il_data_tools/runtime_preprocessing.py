from __future__ import annotations

import cv2
import numpy as np


VERTICAL_CROP_BIAS = 2.0 / 3.0


def crop_to_target_aspect(
    image_bgr: np.ndarray,
    target_width: int,
    target_height: int,
    vertical_bias: float = VERTICAL_CROP_BIAS,
) -> np.ndarray:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("image is empty")
    if target_width <= 0 or target_height <= 0:
        raise ValueError("target dimensions must be positive")
    if not 0.0 <= vertical_bias <= 1.0:
        raise ValueError("vertical_bias must be in [0, 1]")

    source_height, source_width = image_bgr.shape[:2]
    source_aspect = source_width / source_height
    target_aspect = target_width / target_height
    if abs(source_aspect - target_aspect) < 1.0e-6:
        return image_bgr
    if source_aspect < target_aspect:
        crop_height = min(source_height, max(1, round(source_width / target_aspect)))
        excess = source_height - crop_height
        top = min(excess, max(0, round(excess * vertical_bias)))
        return image_bgr[top : top + crop_height, :]

    crop_width = min(source_width, max(1, round(source_height * target_aspect)))
    left = max(0, (source_width - crop_width) // 2)
    return image_bgr[:, left : left + crop_width]


def preprocess_bgr_image(
    image_bgr: np.ndarray,
    input_width: int,
    input_height: int,
) -> np.ndarray:
    roi = crop_to_target_aspect(image_bgr, input_width, input_height)
    resized = cv2.resize(
        roi,
        (input_width, input_height),
        interpolation=cv2.INTER_AREA,
    )
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    return np.transpose(normalized, (2, 0, 1))


def model_input_to_bgr(image_chw: np.ndarray) -> np.ndarray:
    """Convert normalized RGB CHW model input into a viewable BGR image."""
    image = np.asarray(image_chw, dtype=np.float32)
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("model image must have shape (3, height, width)")
    rgb = np.transpose(np.clip(image, 0.0, 1.0), (1, 2, 0))
    return np.ascontiguousarray((rgb[:, :, ::-1] * 255.0).round().astype(np.uint8))


def preprocess_lidar_ranges(
    ranges,
    range_min: float,
    range_max: float,
    lidar_points: int = 360,
) -> np.ndarray:
    if lidar_points <= 0:
        raise ValueError("lidar_points must be positive")
    values = np.asarray(ranges, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("LiDAR ranges are empty")

    max_range = float(range_max) if np.isfinite(range_max) and range_max > 0.0 else 12.0
    min_range = float(range_min) if np.isfinite(range_min) and range_min >= 0.0 else 0.0
    valid = np.isfinite(values) & (values >= min_range) & (values <= max_range)
    cleaned = np.where(valid, values, max_range).astype(np.float32)
    cleaned = np.clip(cleaned, 0.0, max_range) / max_range

    source_x = np.arange(values.size, dtype=np.float32)
    target_x = np.linspace(0.0, float(values.size - 1), lidar_points, dtype=np.float32)
    normalized = np.interp(target_x, source_x, cleaned).astype(np.float32)
    valid_mask = np.interp(
        target_x,
        source_x,
        valid.astype(np.float32),
    ).astype(np.float32)
    valid_mask = (valid_mask >= 0.5).astype(np.float32)
    return np.stack([normalized, valid_mask], axis=0)
