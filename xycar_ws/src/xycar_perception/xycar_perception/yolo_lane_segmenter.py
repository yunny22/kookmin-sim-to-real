from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def merge_lane_instance_masks(
    instance_masks: Any,
    class_ids: Any,
    output_shape: tuple[int, int],
    *,
    white_class_id: int = 0,
    yellow_class_id: int = 1,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge YOLO instance masks into one binary mask per lane class."""
    output_height, output_width = output_shape
    white = np.zeros((output_height, output_width), dtype=np.uint8)
    yellow = np.zeros_like(white)
    masks = np.asarray(instance_masks)
    classes = np.asarray(class_ids).reshape(-1)
    if masks.size == 0 or classes.size == 0:
        return white, yellow
    if masks.ndim == 2:
        masks = masks[np.newaxis, ...]

    for mask, class_id in zip(masks, classes):
        class_id = int(round(float(class_id)))
        if class_id not in (white_class_id, yellow_class_id):
            continue
        if mask.shape != output_shape:
            mask = cv2.resize(
                mask.astype(np.float32),
                (output_width, output_height),
                interpolation=cv2.INTER_LINEAR,
            )
        binary = mask >= float(threshold)
        target = white if class_id == white_class_id else yellow
        target[binary] = 255

    # The centerline class takes precedence where mask edges overlap.
    white[yellow > 0] = 0
    return white, yellow


class YoloLaneSegmenter:
    """Small lazy-import wrapper around an Ultralytics segmentation model."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cpu",
        confidence: float = 0.25,
        iou: float = 0.50,
        image_size: int = 640,
        max_detections: int = 30,
        white_class_id: int = 0,
        yellow_class_id: int = 1,
        cpu_threads: int = 0,
        retina_masks: bool = True,
    ) -> None:
        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"YOLO lane model does not exist: {path}")
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is required for lane_segmentation_backend=yolo. "
                "Install it with: python3 -m pip install --user ultralytics"
            ) from exc

        if str(device).lower() == "cpu" and int(cpu_threads) > 0:
            torch.set_num_threads(int(cpu_threads))

        self.model_path = path
        self.model = YOLO(str(path), task="segment")
        self.device = str(device)
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.image_size = int(image_size)
        self.max_detections = int(max_detections)
        self.white_class_id = int(white_class_id)
        self.yellow_class_id = int(yellow_class_id)
        self.cpu_threads = int(cpu_threads)
        self.retina_masks = bool(retina_masks)
        warmup_image = np.zeros(
            (self.image_size, self.image_size, 3), dtype=np.uint8
        )
        self._predict(warmup_image)

    @property
    def class_names(self) -> dict[int, str]:
        names = self.model.names
        if isinstance(names, dict):
            return {int(index): str(name) for index, name in names.items()}
        return {index: str(name) for index, name in enumerate(names)}

    def predict(
        self,
        image_bgr: np.ndarray,
        *,
        render_debug: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("YOLO input image is empty")
        result = self._predict(image_bgr)
        height, width = image_bgr.shape[:2]
        if result.masks is None or result.boxes is None:
            empty = np.zeros((height, width), dtype=np.uint8)
            debug = image_bgr.copy() if render_debug else None
            return empty, empty.copy(), debug

        masks = result.masks.data.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy()
        white, yellow = merge_lane_instance_masks(
            masks,
            classes,
            (height, width),
            white_class_id=self.white_class_id,
            yellow_class_id=self.yellow_class_id,
        )
        debug = result.plot() if render_debug else None
        return white, yellow, debug

    def _predict(self, image_bgr: np.ndarray):
        return self.model.predict(
            source=image_bgr,
            imgsz=self.image_size,
            conf=self.confidence,
            iou=self.iou,
            max_det=self.max_detections,
            device=self.device,
            retina_masks=self.retina_masks,
            verbose=False,
        )[0]
