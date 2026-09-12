#!/usr/bin/env python3

from __future__ import annotations

from typing import Dict

import cv2
import numpy as np


# When a 4:3 image is cropped to 16:9, 120 vertical pixels are removed.
# A 2/3 bias places 80 px above and 40 px below the ROI: y=80:440 at 640x480.
VERTICAL_CROP_BIAS = 2.0 / 3.0


def crop_to_target_aspect(
    image_bgr: np.ndarray,
    target_width: int,
    target_height: int,
    vertical_bias: float = VERTICAL_CROP_BIAS,
) -> np.ndarray:
    """Crop without distortion to the target aspect ratio.

    For the Xycar 640x480 camera and a 160x90 target this returns image[80:440, :].
    The proportional implementation also behaves safely if the camera resolution changes.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("image is empty")
    if target_width <= 0 or target_height <= 0:
        raise ValueError("target dimensions must be positive")
    if not 0.0 <= vertical_bias <= 1.0:
        raise ValueError("vertical_bias must be in [0, 1]")

    source_height, source_width = image_bgr.shape[:2]
    source_aspect = source_width / source_height
    target_aspect = target_width / target_height

    if abs(source_aspect - target_aspect) < 1e-6:
        return image_bgr

    if source_aspect < target_aspect:
        # Source is relatively tall: remove vertical pixels with a road-oriented bias.
        crop_height = min(source_height, max(1, round(source_width / target_aspect)))
        excess = source_height - crop_height
        top = min(excess, max(0, round(excess * vertical_bias)))
        return image_bgr[top : top + crop_height, :]

    # Source is relatively wide: center-crop horizontally.
    crop_width = min(source_width, max(1, round(source_height * target_aspect)))
    left = max(0, (source_width - crop_width) // 2)
    return image_bgr[:, left : left + crop_width]


def preprocess_bgr_image(
    image_bgr: np.ndarray,
    input_width: int,
    input_height: int,
) -> np.ndarray:
    """Return normalized RGB CHW data shared by training/evaluation/runtime."""
    roi = crop_to_target_aspect(image_bgr, input_width, input_height)
    resized = cv2.resize(
        roi,
        (input_width, input_height),
        interpolation=cv2.INTER_AREA,
    )
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    return np.transpose(normalized, (2, 0, 1))


def preprocessing_contract(
    input_width: int,
    input_height: int,
    canonical_input: bool = False,
) -> Dict[str, object]:
    """Machine-readable contract that runtime code must reproduce exactly."""
    source_width, source_height = (256, 144) if canonical_input else (640, 480)
    reference_crop = [0, 0, 256, 144] if canonical_input else [0, 80, 640, 440]
    return {
        "input_representation": "canonical_bev" if canonical_input else "raw_camera_bgr",
        "source_camera_expected": {"width": source_width, "height": source_height},
        "reference_crop_xyxy": reference_crop,
        "crop_strategy": "target_aspect_with_vertical_bias",
        "vertical_crop_bias": VERTICAL_CROP_BIAS,
        "input_width": int(input_width),
        "input_height": int(input_height),
        "resize_interpolation": "cv2.INTER_AREA",
        "source_color_order": "BGR",
        "model_color_order": "RGB",
        "normalization": "float32_divide_by_255",
        "tensor_layout": "CHW",
    }
