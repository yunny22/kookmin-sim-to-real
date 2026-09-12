from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CanonicalRoadStages:
    road_image: np.ndarray
    white_mask: np.ndarray
    yellow_mask: np.ndarray
    pre_geometry_white_mask: np.ndarray
    pre_geometry_yellow_mask: np.ndarray
    post_geometry_white_mask: np.ndarray
    post_geometry_yellow_mask: np.ndarray
    valid_mask: np.ndarray


def _filter_components(
    mask: np.ndarray,
    min_area_px: int,
    max_thickness_px: float = 0.0,
) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    if min_area_px <= 1 and max_thickness_px <= 0.0:
        return binary * 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    cleaned = np.zeros_like(binary)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) < min_area_px:
            continue
        component = labels == label
        if max_thickness_px > 0.0:
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            roi = component[y:y + height, x:x + width].astype(np.uint8)
            padded = cv2.copyMakeBorder(roi, 1, 1, 1, 1, cv2.BORDER_CONSTANT)
            distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
            thickness = 2.0 * float(distance.max())
            if thickness > max_thickness_px:
                continue
        cleaned[component] = 1
    return cleaned * 255


def _metric_crop(
    image: np.ndarray,
    *,
    lateral_m_per_px: float,
    forward_m_per_px: float,
    lateral_range_m: float,
    forward_range_m: float,
) -> np.ndarray:
    if lateral_m_per_px <= 0.0 or forward_m_per_px <= 0.0:
        raise ValueError("BEV metric scales must be positive")
    if lateral_range_m <= 0.0 or forward_range_m <= 0.0:
        raise ValueError("canonical metric ranges must be positive")

    source_height, source_width = image.shape[:2]
    crop_width = max(1, int(round(lateral_range_m / lateral_m_per_px)))
    crop_height = max(1, int(round(forward_range_m / forward_m_per_px)))
    center_x = source_width // 2
    left = center_x - crop_width // 2
    right = left + crop_width
    top = source_height - crop_height
    bottom = source_height

    source_left = max(0, left)
    source_right = min(source_width, right)
    source_top = max(0, top)
    source_bottom = min(source_height, bottom)
    cropped = image[source_top:source_bottom, source_left:source_right]

    pad_left = max(0, -left)
    pad_right = max(0, right - source_width)
    pad_top = max(0, -top)
    pad_bottom = max(0, bottom - source_height)
    if any((pad_left, pad_right, pad_top, pad_bottom)):
        border_value = 0 if image.ndim == 2 else [0] * image.shape[2]
        cropped = cv2.copyMakeBorder(
            cropped,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=border_value,
        )
    return cropped


def _fixed_width_mask(mask: np.ndarray, line_width_px: int) -> np.ndarray:
    work = (mask > 0).astype(np.uint8) * 255
    if not np.any(work):
        return work

    skeleton = np.zeros_like(work)
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while np.any(work):
        eroded = cv2.erode(work, cross)
        opened = cv2.dilate(eroded, cross)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(work, opened))
        work = eroded

    width = max(1, int(line_width_px))
    if width == 1:
        return skeleton
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width))
    return cv2.dilate(skeleton, kernel)


def _filter_lane_geometry(
    mask: np.ndarray,
    *,
    brightness: np.ndarray | None = None,
    min_component_median_brightness: float = 0.0,
    min_span_px: int,
    min_elongation: float,
    min_verticality: float,
    max_components: int = 0,
    max_components_per_side: int = 0,
    clutter_component_limit: int = 0,
    max_mask_fraction: float = 0.0,
    max_fit_rmse_px: float = 0.0,
    redraw_fitted_lines: bool = True,
) -> np.ndarray:
    """Keep sparse, longitudinal lane-like components in canonical space."""
    binary = (mask > 0).astype(np.uint8)
    if not np.any(binary):
        return binary * 255
    if max_mask_fraction > 0.0 and float(np.mean(binary)) > max_mask_fraction:
        return np.zeros_like(mask)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    candidates: list[tuple[float, int, float]] = []
    for label in range(1, count):
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if max(width, height) < max(1, int(min_span_px)):
            continue

        ys, xs = np.nonzero(labels == label)
        if xs.size < 2:
            continue
        if (
            brightness is not None
            and min_component_median_brightness > 0.0
            and float(np.median(brightness[ys, xs]))
            < float(min_component_median_brightness)
        ):
            continue
        points = np.column_stack((ys, xs)).astype(np.float32)
        covariance = np.cov(points, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        major_index = int(np.argmax(eigenvalues))
        major = max(float(eigenvalues[major_index]), 0.0)
        minor = max(float(eigenvalues[1 - major_index]), 0.0)
        elongation = float(np.sqrt((major + 1.0) / (minor + 1.0)))
        verticality = abs(float(eigenvectors[0, major_index]))
        if elongation < float(min_elongation):
            continue
        if verticality < float(min_verticality):
            continue

        center_x = float(centroids[label, 0])
        side_distance = abs(center_x - mask.shape[1] * 0.5)
        score = (
            float(max(width, height)) * 2.0
            + min(elongation, 20.0) * 5.0
            + float(area) * 0.15
            + side_distance * 0.25
        )
        candidates.append((score, label, center_x))

    if (
        clutter_component_limit > 0
        and len(candidates) > clutter_component_limit
    ):
        return np.zeros_like(mask)

    selected: list[int] = []
    if max_components_per_side > 0:
        center_x = mask.shape[1] * 0.5
        for is_left in (True, False):
            side = [
                item
                for item in candidates
                if (item[2] < center_x) == is_left
            ]
            side.sort(reverse=True)
            selected.extend(item[1] for item in side[:max_components_per_side])
    else:
        candidates.sort(reverse=True)
        limit = len(candidates) if max_components <= 0 else max_components
        selected.extend(item[1] for item in candidates[:limit])

    cleaned = np.zeros_like(binary)
    for label in selected:
        component = labels == label
        if max_fit_rmse_px <= 0.0:
            cleaned[component] = 1
            continue

        ys, xs = np.nonzero(component)
        rows = np.unique(ys)
        if rows.size < 2:
            continue
        centers = np.array(
            [float(np.median(xs[ys == row])) for row in rows], dtype=np.float64
        )
        degree = min(2, int(rows.size) - 1)
        keep = np.ones(rows.size, dtype=bool)
        for _ in range(3):
            if int(np.count_nonzero(keep)) <= degree:
                break
            coefficients = np.polyfit(rows[keep], centers[keep], degree)
            residuals = np.abs(centers - np.polyval(coefficients, rows))
            median = float(np.median(residuals[keep]))
            mad = float(np.median(np.abs(residuals[keep] - median)))
            robust_limit = max(
                float(max_fit_rmse_px) * 1.5,
                2.5 * 1.4826 * mad,
            )
            next_keep = residuals <= robust_limit
            if np.array_equal(next_keep, keep):
                break
            keep = next_keep
        if int(np.count_nonzero(keep)) <= degree:
            continue
        coefficients = np.polyfit(rows[keep], centers[keep], degree)
        fitted = np.polyval(coefficients, rows[keep])
        rmse = float(np.sqrt(np.mean((centers[keep] - fitted) ** 2)))
        if rmse > float(max_fit_rmse_px):
            continue
        if not redraw_fitted_lines:
            cleaned[component] = 1
            continue
        fitted_x = np.clip(
            np.rint(fitted), 0, mask.shape[1] - 1
        ).astype(np.int32)
        points = np.column_stack(
            (fitted_x, rows[keep].astype(np.int32))
        )
        cv2.polylines(
            cleaned,
            [points],
            False,
            1,
            thickness=1,
            lineType=cv2.LINE_8,
        )
    return cleaned * 255


def make_canonical_road_image(
    bev_bgr: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    lateral_m_per_px: float,
    forward_m_per_px: float,
    lateral_range_m: float = 1.4,
    forward_range_m: float = 1.5,
    output_width: int = 256,
    output_height: int = 144,
    background_gray: int = 36,
    line_width_px: int = 5,
    white_s_max: int = 120,
    white_v_min: int = 145,
    white_v_floor: int = 70,
    white_relative_delta: float = 9.0,
    yellow_h_min: int = 15,
    yellow_h_max: int = 42,
    yellow_s_min: int = 55,
    yellow_v_min: int = 90,
    min_component_area_px: int = 8,
    white_max_component_thickness_px: float = 0.0,
    yellow_max_component_thickness_px: float = 0.0,
    geometry_filter_enabled: bool = False,
    white_min_line_span_px: int = 14,
    yellow_min_line_span_px: int = 7,
    min_line_elongation: float = 1.8,
    min_line_verticality: float = 0.30,
    white_max_components_per_side: int = 1,
    yellow_max_components: int = 5,
    white_clutter_component_limit: int = 5,
    white_max_mask_fraction: float = 0.0,
    yellow_max_mask_fraction: float = 0.0,
    max_line_fit_rmse_px: float = 0.0,
    geometry_redraw_fitted_lines: bool = True,
    yellow_geometry_filter_enabled: bool = True,
    white_min_component_median_v: float = 0.0,
    top_ignore_m: float = 0.0,
    bottom_ignore_m: float = 0.08,
    preserve_white_mask: bool = False,
    return_stages: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | CanonicalRoadStages:
    """Build a fixed metric, fixed-color road representation from a BEV."""
    if bev_bgr is None or bev_bgr.size == 0:
        raise ValueError("BEV image is empty")
    if output_width <= 0 or output_height <= 0:
        raise ValueError("canonical output dimensions must be positive")

    metric_bev = _metric_crop(
        bev_bgr,
        lateral_m_per_px=lateral_m_per_px,
        forward_m_per_px=forward_m_per_px,
        lateral_range_m=lateral_range_m,
        forward_range_m=forward_range_m,
    )
    metric_valid = None
    if valid_mask is not None:
        if valid_mask.shape[:2] != bev_bgr.shape[:2]:
            raise ValueError("valid mask dimensions must match the BEV image")
        metric_valid = _metric_crop(
            valid_mask,
            lateral_m_per_px=lateral_m_per_px,
            forward_m_per_px=forward_m_per_px,
            lateral_range_m=lateral_range_m,
            forward_range_m=forward_range_m,
        )
    hsv = cv2.cvtColor(metric_bev, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(value)
    local_background = cv2.GaussianBlur(clahe, (0, 0), sigmaX=5.0, sigmaY=5.0)
    relative_brightness = (
        clahe.astype(np.float32) - local_background.astype(np.float32)
    )
    low_saturation = saturation <= int(white_s_max)
    absolute_white = value >= int(white_v_min)
    relative_white = (
        (value >= int(white_v_floor))
        & (relative_brightness >= float(white_relative_delta))
    )
    white_mask = (
        (low_saturation & (absolute_white | relative_white)).astype(np.uint8)
        * 255
    )

    yellow_mask = cv2.inRange(
        hsv,
        np.array([yellow_h_min, yellow_s_min, yellow_v_min], dtype=np.uint8),
        np.array([yellow_h_max, 255, 255], dtype=np.uint8),
    )
    if metric_valid is not None:
        valid_binary = (metric_valid > 0).astype(np.uint8) * 255
        white_mask = cv2.bitwise_and(white_mask, valid_binary)
        yellow_mask = cv2.bitwise_and(yellow_mask, valid_binary)
    ignore_rows = max(0, int(round(bottom_ignore_m / forward_m_per_px)))
    if ignore_rows > 0:
        white_mask[-ignore_rows:, :] = 0
        yellow_mask[-ignore_rows:, :] = 0
    top_ignore_rows = max(0, int(round(top_ignore_m / forward_m_per_px)))
    if top_ignore_rows > 0:
        white_mask[:top_ignore_rows, :] = 0
        yellow_mask[:top_ignore_rows, :] = 0

    close_kernel = np.ones((3, 3), dtype=np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, close_kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, close_kernel)
    white_mask = _filter_components(
        white_mask,
        1 if preserve_white_mask else min_component_area_px,
        0.0 if preserve_white_mask else white_max_component_thickness_px,
    )
    yellow_mask = _filter_components(
        yellow_mask,
        min_component_area_px,
        yellow_max_component_thickness_px,
    )

    output_size = (int(output_width), int(output_height))
    white_mask = cv2.resize(
        white_mask, output_size, interpolation=cv2.INTER_NEAREST
    )
    yellow_mask = cv2.resize(
        yellow_mask, output_size, interpolation=cv2.INTER_NEAREST
    )
    output_value = cv2.resize(
        value, output_size, interpolation=cv2.INTER_AREA
    )
    output_valid = None
    if metric_valid is not None:
        output_valid = cv2.resize(
            (metric_valid > 0).astype(np.uint8) * 255,
            output_size,
            interpolation=cv2.INTER_NEAREST,
        )
    pre_geometry_white = white_mask.copy()
    pre_geometry_yellow = yellow_mask.copy()
    if geometry_filter_enabled:
        if not preserve_white_mask:
            white_mask = _filter_lane_geometry(
                white_mask,
                brightness=output_value,
                min_component_median_brightness=white_min_component_median_v,
                min_span_px=white_min_line_span_px,
                min_elongation=min_line_elongation,
                min_verticality=min_line_verticality,
                max_components_per_side=white_max_components_per_side,
                clutter_component_limit=white_clutter_component_limit,
                max_mask_fraction=white_max_mask_fraction,
                max_fit_rmse_px=max_line_fit_rmse_px,
                redraw_fitted_lines=geometry_redraw_fitted_lines,
            )
        if yellow_geometry_filter_enabled:
            yellow_mask = _filter_lane_geometry(
                yellow_mask,
                min_span_px=yellow_min_line_span_px,
                min_elongation=min_line_elongation,
                min_verticality=min_line_verticality,
                max_components=yellow_max_components,
                max_mask_fraction=yellow_max_mask_fraction,
                max_fit_rmse_px=max_line_fit_rmse_px,
                redraw_fitted_lines=geometry_redraw_fitted_lines,
            )
    post_geometry_white = white_mask.copy()
    post_geometry_yellow = yellow_mask.copy()
    white_mask = _fixed_width_mask(white_mask, line_width_px)
    yellow_mask = _fixed_width_mask(yellow_mask, line_width_px)
    if output_valid is not None:
        white_mask = cv2.bitwise_and(white_mask, output_valid)
        yellow_mask = cv2.bitwise_and(yellow_mask, output_valid)
    white_mask[yellow_mask > 0] = 0

    gray = int(np.clip(background_gray, 0, 255))
    canonical = np.full((output_height, output_width, 3), gray, dtype=np.uint8)
    canonical[white_mask > 0] = (255, 255, 255)
    canonical[yellow_mask > 0] = (0, 220, 255)
    if return_stages:
        if output_valid is None:
            output_valid = np.full(
                (output_height, output_width), 255, dtype=np.uint8
            )
        return CanonicalRoadStages(
            road_image=canonical,
            white_mask=white_mask,
            yellow_mask=yellow_mask,
            pre_geometry_white_mask=pre_geometry_white,
            pre_geometry_yellow_mask=pre_geometry_yellow,
            post_geometry_white_mask=post_geometry_white,
            post_geometry_yellow_mask=post_geometry_yellow,
            valid_mask=output_valid,
        )
    return canonical, white_mask, yellow_mask


def make_canonical_road_image_from_masks(
    bev_white_mask: np.ndarray,
    bev_yellow_mask: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    lateral_m_per_px: float,
    forward_m_per_px: float,
    lateral_range_m: float = 1.4,
    forward_range_m: float = 1.5,
    output_width: int = 256,
    output_height: int = 144,
    background_gray: int = 36,
    line_width_px: int = 5,
    min_component_area_px: int = 8,
    white_max_component_thickness_px: float = 0.0,
    yellow_max_component_thickness_px: float = 0.0,
    geometry_filter_enabled: bool = False,
    preserve_white_mask: bool = False,
    white_min_line_span_px: int = 14,
    yellow_min_line_span_px: int = 7,
    min_line_elongation: float = 1.8,
    min_line_verticality: float = 0.30,
    white_max_components_per_side: int = 1,
    yellow_max_components: int = 5,
    white_clutter_component_limit: int = 5,
    white_max_mask_fraction: float = 0.0,
    yellow_max_mask_fraction: float = 0.0,
    max_line_fit_rmse_px: float = 0.0,
    geometry_redraw_fitted_lines: bool = True,
    yellow_geometry_filter_enabled: bool = True,
    top_ignore_m: float = 0.0,
    bottom_ignore_m: float = 0.08,
    return_stages: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | CanonicalRoadStages:
    """Normalize class masks into the shared fixed-color canonical contract."""
    if bev_white_mask.shape != bev_yellow_mask.shape:
        raise ValueError(
            "white and yellow BEV masks must have identical dimensions"
        )
    if bev_white_mask.ndim != 2:
        raise ValueError("lane masks must be single-channel images")

    gray = int(np.clip(background_gray, 0, 255))
    classified_bev = np.full(
        (*bev_white_mask.shape, 3), gray, dtype=np.uint8
    )
    classified_bev[bev_white_mask > 0] = (255, 255, 255)
    classified_bev[bev_yellow_mask > 0] = (0, 220, 255)

    # Exact synthetic class colors let the existing metric crop, geometry
    # filtering, fixed-width normalization, and overlap rules stay shared.
    return make_canonical_road_image(
        classified_bev,
        valid_mask=valid_mask,
        lateral_m_per_px=lateral_m_per_px,
        forward_m_per_px=forward_m_per_px,
        lateral_range_m=lateral_range_m,
        forward_range_m=forward_range_m,
        output_width=output_width,
        output_height=output_height,
        background_gray=background_gray,
        line_width_px=line_width_px,
        white_s_max=10,
        white_v_min=250,
        white_v_floor=250,
        white_relative_delta=255.0,
        yellow_h_min=20,
        yellow_h_max=35,
        yellow_s_min=200,
        yellow_v_min=180,
        min_component_area_px=min_component_area_px,
        white_max_component_thickness_px=white_max_component_thickness_px,
        yellow_max_component_thickness_px=yellow_max_component_thickness_px,
        geometry_filter_enabled=geometry_filter_enabled,
        white_min_line_span_px=white_min_line_span_px,
        yellow_min_line_span_px=yellow_min_line_span_px,
        min_line_elongation=min_line_elongation,
        min_line_verticality=min_line_verticality,
        white_max_components_per_side=white_max_components_per_side,
        yellow_max_components=yellow_max_components,
        white_clutter_component_limit=white_clutter_component_limit,
        white_max_mask_fraction=white_max_mask_fraction,
        yellow_max_mask_fraction=yellow_max_mask_fraction,
        max_line_fit_rmse_px=max_line_fit_rmse_px,
        geometry_redraw_fitted_lines=geometry_redraw_fitted_lines,
        yellow_geometry_filter_enabled=yellow_geometry_filter_enabled,
        white_min_component_median_v=0.0,
        top_ignore_m=top_ignore_m,
        bottom_ignore_m=bottom_ignore_m,
        preserve_white_mask=preserve_white_mask,
        return_stages=return_stages,
    )
