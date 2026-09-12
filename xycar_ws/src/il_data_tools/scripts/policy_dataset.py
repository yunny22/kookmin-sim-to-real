#!/usr/bin/env python3

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from image_preprocessing import preprocess_bgr_image


class PolicyCsvDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        input_width: int = 160,
        input_height: int = 90,
        max_steer_deg: float = 100.0,
        use_phase: bool = False,
        use_lidar: bool = False,
        lidar_points: int = 360,
        enable_augment: bool = False,
        enable_flip: bool = False,
        canonical_input: bool = False,
        lane_dropout_probability: float = 0.0,
    ) -> None:
        self.csv_path = Path(csv_path).expanduser().resolve()
        self.input_width = input_width
        self.input_height = input_height
        self.max_steer_deg = max_steer_deg
        self.use_phase = use_phase
        self.use_lidar = use_lidar
        self.lidar_points = lidar_points
        self.enable_augment = enable_augment
        self.enable_flip = enable_flip
        self.canonical_input = canonical_input
        self.lane_dropout_probability = lane_dropout_probability

        with self.csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"empty dataset CSV: {self.csv_path}")

        self.rows: List[Dict[str, str]] = []
        missing = []
        for index, row in enumerate(rows, start=2):
            image_path = Path(row.get("image_path", "")).expanduser()
            if not image_path.is_absolute():
                image_path = (self.csv_path.parent / image_path).resolve()
            if not image_path.is_file():
                missing.append({"row": index, "image_path": str(image_path)})
                continue
            row = dict(row)
            row["image_path"] = str(image_path)
            if use_lidar:
                scan_path = Path(row.get("scan_npz_path", "")).expanduser()
                if not scan_path.is_absolute():
                    scan_path = (self.csv_path.parent / scan_path).resolve()
                if not scan_path.is_file():
                    missing.append({"row": index, "scan_npz_path": str(scan_path)})
                    continue
                row["scan_npz_path"] = str(scan_path)
            self.rows.append(row)

        if missing:
            preview = missing[:5]
            raise FileNotFoundError(
                f"{len(missing)} image files are missing in {self.csv_path}: {preview}"
            )
        if use_phase and "phase" not in self.rows[0]:
            raise ValueError(f"phase column is required for phase model: {self.csv_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.rows[index]
        image = load_image_bgr(row["image_path"])
        lidar = (
            load_lidar_tensor(row["scan_npz_path"], self.lidar_points)
            if self.use_lidar
            else torch.empty(0, dtype=torch.float32)
        )
        steer_norm = float(row["steer_norm"])

        if self.enable_augment:
            image, steer_norm, flipped = augment_image(
                image,
                steer_norm,
                enable_flip=self.enable_flip,
                canonical_input=self.canonical_input,
                lane_dropout_probability=self.lane_dropout_probability,
            )
            if flipped and self.use_lidar:
                lidar = torch.flip(lidar, dims=[1])

        image = preprocess_image(image, self.input_width, self.input_height)
        phase = float(row.get("phase") or 0.0)
        angle_deg = float(row.get("angle_deg") or steer_norm * self.max_steer_deg)
        speed = float(row.get("speed") or 0.0)
        metadata = {
            "image_path": row.get("image_path", ""),
            "scan_npz_path": row.get("scan_npz_path", ""),
            "angle_deg": angle_deg,
            "speed": speed,
            "mission_label": row.get("mission_label", ""),
            "session_id": row.get("session_id", ""),
            "timestamp_ns": row.get("timestamp_ns", ""),
            "phase": phase,
        }
        return {
            "image": image,
            "lidar": lidar,
            "target": torch.tensor([steer_norm], dtype=torch.float32),
            "phase": torch.tensor([phase], dtype=torch.float32),
            "metadata": metadata,
        }

    def session_ids(self) -> set:
        return {row.get("session_id", "") for row in self.rows}


def load_image_bgr(path: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"failed to read image: {path}")
    return image


def preprocess_image(
    image_bgr: np.ndarray,
    input_width: int,
    input_height: int,
) -> torch.Tensor:
    # Preserve geometry: 640x480 becomes image[80:440, :] (640x360),
    # then scales to 160x90.  The same helper must be used at runtime.
    return torch.from_numpy(
        preprocess_bgr_image(image_bgr, input_width, input_height)
    )


def load_lidar_tensor(path: str, lidar_points: int = 360) -> torch.Tensor:
    """Load LaserScan NPZ as normalized range plus validity-mask channels."""
    if lidar_points <= 0:
        raise ValueError("lidar_points must be positive")
    with np.load(str(path), allow_pickle=False) as data:
        ranges = np.asarray(data["ranges"], dtype=np.float32).reshape(-1)
        raw_max = float(np.asarray(data.get("range_max", 12.0)).reshape(-1)[0])
        raw_min = float(np.asarray(data.get("range_min", 0.0)).reshape(-1)[0])
    if ranges.size == 0:
        raise ValueError(f"empty LiDAR ranges: {path}")
    max_range = raw_max if np.isfinite(raw_max) and raw_max > 0.0 else 12.0
    min_range = raw_min if np.isfinite(raw_min) and raw_min >= 0.0 else 0.0
    valid = np.isfinite(ranges) & (ranges >= min_range) & (ranges <= max_range)
    cleaned = np.where(valid, ranges, max_range).astype(np.float32)
    cleaned = np.clip(cleaned, 0.0, max_range) / max_range

    source_x = np.arange(ranges.size, dtype=np.float32)
    target_x = np.linspace(0.0, float(ranges.size - 1), lidar_points, dtype=np.float32)
    normalized = np.interp(target_x, source_x, cleaned).astype(np.float32)
    valid_mask = np.interp(target_x, source_x, valid.astype(np.float32)).astype(np.float32)
    valid_mask = (valid_mask >= 0.5).astype(np.float32)
    return torch.from_numpy(np.stack([normalized, valid_mask], axis=0))


def augment_image(
    image_bgr: np.ndarray,
    steer_norm: float,
    enable_flip: bool = False,
    canonical_input: bool = False,
    lane_dropout_probability: float = 0.0,
) -> Tuple[np.ndarray, float, bool]:
    image = image_bgr.copy()
    if canonical_input:
        image = random_canonical_boundary_dropout(image, lane_dropout_probability)
    else:
        image = random_brightness_contrast_gamma(image)
        image = random_noise_blur(image)
    flipped = enable_flip and random.random() < 0.5
    if flipped:
        image = cv2.flip(image, 1)
        steer_norm = -steer_norm
    return image, steer_norm, flipped


def random_canonical_boundary_dropout(
    image: np.ndarray,
    probability: float,
) -> np.ndarray:
    """Hide one white boundary while preserving yellow markings and the target."""
    if probability <= 0.0 or random.random() >= probability:
        return image

    height, width = image.shape[:2]
    white = np.all(image >= 235, axis=2)
    columns = np.arange(width)[None, :]
    side_masks = {
        "left": white & (columns < width // 2),
        "right": white & (columns >= width // 2),
    }
    min_pixels = max(8, height // 8)
    available = [
        name for name, mask in side_masks.items() if int(np.count_nonzero(mask)) >= min_pixels
    ]
    if len(available) < 2:
        return image

    output = image.copy()
    output[side_masks[random.choice(available)]] = (36, 36, 36)
    return output


def random_brightness_contrast_gamma(image: np.ndarray) -> np.ndarray:
    alpha = random.uniform(0.85, 1.15)
    beta = random.uniform(-18.0, 18.0)
    out = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    gamma = random.uniform(0.85, 1.15)
    table = np.array(
        [((i / 255.0) ** (1.0 / gamma)) * 255.0 for i in range(256)]
    ).astype("uint8")
    return cv2.LUT(out, table)


def random_noise_blur(image: np.ndarray) -> np.ndarray:
    if random.random() < 0.35:
        sigma = random.uniform(1.0, 4.0)
        noise = np.random.normal(0.0, sigma, image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if random.random() < 0.20:
        image = cv2.GaussianBlur(image, (3, 3), 0)
    return image
