from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class CurveCandidate:
    coefficients: np.ndarray
    mask: np.ndarray
    row_min: int
    row_max: int
    area: int
    span: int

    def x_at(self, rows: np.ndarray | float) -> np.ndarray:
        return np.polyval(self.coefficients, rows)


@dataclass
class LaneTrack:
    name: str
    coefficients: np.ndarray | None = None
    row_min: int = 0
    row_max: int = 0
    last_mask: np.ndarray | None = None
    last_seen_sec: float | None = None
    confidence: float = 0.0
    pending: CurveCandidate | None = None
    pending_hits: int = 0
    observed_this_frame: bool = False
    synthetic_from_width: bool = False

    def reset(self) -> None:
        self.coefficients = None
        self.row_min = 0
        self.row_max = 0
        self.last_mask = None
        self.last_seen_sec = None
        self.confidence = 0.0
        self.pending = None
        self.pending_hits = 0
        self.observed_this_frame = False
        self.synthetic_from_width = False

    def elapsed(self, timestamp_sec: float) -> float:
        if self.last_seen_sec is None:
            return float("inf")
        return max(0.0, timestamp_sec - self.last_seen_sec)

    def status(
        self,
        timestamp_sec: float,
        coast_sec: float,
        search_sec: float,
    ) -> str:
        if self.observed_this_frame:
            return "confirmed"
        if self.coefficients is None:
            return "acquiring" if self.pending_hits > 0 else "lost"
        elapsed = self.elapsed(timestamp_sec)
        if elapsed <= coast_sec:
            return "coasting"
        if elapsed <= search_sec:
            return "searching"
        return "lost"


@dataclass
class TrackingResult:
    road_image: np.ndarray
    white_mask: np.ndarray
    yellow_mask: np.ndarray
    debug_image: np.ndarray
    statuses: dict[str, str] = field(default_factory=dict)


def _curve_distance(
    first: np.ndarray,
    second: np.ndarray,
    row_min: int,
    row_max: int,
) -> float:
    if row_max < row_min:
        rows = np.array([(row_min + row_max) * 0.5], dtype=np.float64)
    else:
        rows = np.linspace(row_min, row_max, 9, dtype=np.float64)
    residuals = np.abs(
        np.polyval(first, rows) - np.polyval(second, rows)
    )
    return float(np.mean(residuals))


def _fit_mask(mask: np.ndarray) -> CurveCandidate | None:
    ys, xs = np.nonzero(mask)
    if xs.size < 6:
        return None
    rows = np.unique(ys)
    if rows.size < 3:
        return None
    centers = np.array(
        [float(np.median(xs[ys == row])) for row in rows],
        dtype=np.float64,
    )
    keep = np.ones(rows.size, dtype=bool)
    for _ in range(3):
        if int(np.count_nonzero(keep)) < 3:
            return None
        coefficients = np.polyfit(rows[keep], centers[keep], 2)
        residuals = np.abs(centers - np.polyval(coefficients, rows))
        median = float(np.median(residuals[keep]))
        mad = float(np.median(np.abs(residuals[keep] - median)))
        limit = max(3.0, 2.5 * 1.4826 * mad)
        next_keep = residuals <= limit
        if np.array_equal(next_keep, keep):
            break
        keep = next_keep
    if int(np.count_nonzero(keep)) < 3:
        return None
    coefficients = np.polyfit(rows[keep], centers[keep], 2)
    return CurveCandidate(
        coefficients=coefficients,
        mask=(mask > 0).astype(np.uint8) * 255,
        row_min=int(rows[keep].min()),
        row_max=int(rows[keep].max()),
        area=int(xs.size),
        span=int(rows[keep].max() - rows[keep].min() + 1),
    )


def _component_candidates(
    mask: np.ndarray,
    min_span_px: int,
) -> list[CurveCandidate]:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    candidates: list[CurveCandidate] = []
    for label in range(1, count):
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        if max(height, width) < max(3, int(min_span_px)):
            continue
        candidate = _fit_mask(labels == label)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _combine_candidates(
    candidates: list[CurveCandidate],
    shape: tuple[int, int],
) -> CurveCandidate | None:
    if not candidates:
        return None
    combined = np.zeros(shape, dtype=np.uint8)
    for candidate in candidates:
        combined[candidate.mask > 0] = 255
    return _fit_mask(combined)


def _row_gap(first: CurveCandidate, second: CurveCandidate) -> int:
    if first.row_max < second.row_min:
        return second.row_min - first.row_max
    if second.row_max < first.row_min:
        return first.row_min - second.row_max
    return 0


def _is_curve_continuation(
    first: CurveCandidate,
    second: CurveCandidate,
    *,
    max_row_gap_px: float,
    lateral_gate_px: float,
    curvature_gate_gain: float = 0.0,
    curvature_max_extra_px: float = 0.0,
) -> bool:
    row_gap = _row_gap(first, second)
    if row_gap > max_row_gap_px:
        return False
    row_min = max(first.row_min, second.row_min)
    row_max = min(first.row_max, second.row_max)
    if row_max >= row_min:
        rows = np.linspace(row_min, row_max, 7, dtype=np.float64)
    elif first.row_max < second.row_min:
        rows = np.array(
            [first.row_max, second.row_min], dtype=np.float64
        )
    else:
        rows = np.array(
            [second.row_max, first.row_min], dtype=np.float64
        )
    separation = np.abs(first.x_at(rows) - second.x_at(rows))
    effective_gate_px = float(lateral_gate_px)
    if row_gap > 0 and curvature_gate_gain > 0.0:
        first_gap_row = (
            first.row_max if first.row_max < second.row_min else first.row_min
        )
        second_gap_row = (
            second.row_min if first.row_max < second.row_min else second.row_max
        )
        first_slope = float(
            np.polyval(np.polyder(first.coefficients), first_gap_row)
        )
        second_slope = float(
            np.polyval(np.polyder(second.coefficients), second_gap_row)
        )
        first_second_derivative = float(
            np.polyval(np.polyder(first.coefficients, 2), first_gap_row)
        )
        second_second_derivative = float(
            np.polyval(np.polyder(second.coefficients, 2), second_gap_row)
        )
        heading_change_px = (
            0.5 * abs(first_slope - second_slope) * float(row_gap)
        )
        bend_px = (
            0.5
            * max(
                abs(first_second_derivative),
                abs(second_second_derivative),
            )
            * float(row_gap) ** 2
        )
        curvature_extra_px = float(curvature_gate_gain) * (
            heading_change_px + bend_px
        )
        effective_gate_px += min(
            max(0.0, float(curvature_max_extra_px)),
            curvature_extra_px,
        )
    return float(np.mean(separation)) <= effective_gate_px


def _continuation_chain(
    candidates: list[CurveCandidate],
    seed: CurveCandidate,
    *,
    shape: tuple[int, int],
    max_row_gap_px: float,
    lateral_gate_px: float,
    curvature_gate_gain: float = 0.0,
    curvature_max_extra_px: float = 0.0,
) -> CurveCandidate:
    selected = [seed]
    remaining = [candidate for candidate in candidates if candidate is not seed]
    changed = True
    while changed:
        changed = False
        for index, candidate in enumerate(remaining):
            if any(
                _is_curve_continuation(
                    anchor,
                    candidate,
                    max_row_gap_px=max_row_gap_px,
                    lateral_gate_px=lateral_gate_px,
                    curvature_gate_gain=curvature_gate_gain,
                    curvature_max_extra_px=curvature_max_extra_px,
                )
                for anchor in selected
            ):
                selected.append(candidate)
                remaining.pop(index)
                changed = True
                break
    return _combine_candidates(selected, shape) or seed


def _prune_curve_components(
    candidate: CurveCandidate,
    max_residual_px: float,
) -> CurveCandidate:
    """Remove chained components that do not follow the robust fitted curve."""
    if max_residual_px <= 0.0:
        return candidate
    components = _component_candidates(candidate.mask, min_span_px=3)
    kept: list[CurveCandidate] = []
    scored: list[tuple[float, CurveCandidate]] = []
    for component in components:
        rows = np.linspace(
            component.row_min,
            component.row_max,
            9,
            dtype=np.float64,
        )
        residuals = np.abs(
            component.x_at(rows) - candidate.x_at(rows)
        )
        median_residual = float(np.median(residuals))
        scored.append((median_residual, component))
        if median_residual > max_residual_px:
            continue
        if float(np.percentile(residuals, 90)) > max_residual_px * 1.5:
            continue
        kept.append(component)
    if not kept and scored:
        kept.append(min(scored, key=lambda item: item[0])[1])
    return _combine_candidates(kept, candidate.mask.shape) or candidate


class CanonicalLaneTracker:
    def __init__(
        self,
        *,
        width: int,
        height: int,
        lateral_range_m: float,
        forward_range_m: float = 1.5,
        expected_half_lane_width_m: float = 0.412,
        lane_width_tolerance_m: float = 0.18,
        min_white_yellow_offset_m: float = 0.18,
        base_search_gate_m: float = 0.08,
        max_search_gate_m: float = 0.25,
        continuation_distance_m: float = 0.38,
        yellow_fit_gate_m: float = 0.0,
        curvature_gate_gain: float = 1.0,
        curvature_max_extra_m: float = 0.10,
        confirmation_frames: int = 1,
        coast_sec: float = 0.30,
        search_sec: float = 1.00,
        smoothing_alpha: float = 0.55,
        width_prediction_enabled: bool = True,
        width_prediction_replacement_frames: int = 1,
        transverse_clutter_row_fraction: float = 0.0,
        transverse_clutter_min_rows: int = 6,
        persistent_prediction_enabled: bool = False,
        allow_unpaired_yellow: bool = False,
        line_width_px: int = 5,
        background_gray: int = 36,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.m_per_px = float(lateral_range_m) / max(1, self.width)
        self.forward_m_per_px = float(forward_range_m) / max(1, self.height)
        self.expected_offset_px = expected_half_lane_width_m / self.m_per_px
        self.width_tolerance_px = lane_width_tolerance_m / self.m_per_px
        self.min_offset_px = min_white_yellow_offset_m / self.m_per_px
        self.base_gate_px = base_search_gate_m / self.m_per_px
        self.max_gate_px = max_search_gate_m / self.m_per_px
        self.max_continuation_gap_px = max(
            1.0,
            float(continuation_distance_m) / self.forward_m_per_px,
        )
        self.yellow_fit_gate_px = max(
            0.0, float(yellow_fit_gate_m) / self.m_per_px
        )
        self.curvature_gate_gain = max(0.0, float(curvature_gate_gain))
        self.curvature_max_extra_px = max(
            0.0,
            float(curvature_max_extra_m) / self.m_per_px,
        )
        self.confirmation_frames = max(1, int(confirmation_frames))
        self.coast_sec = max(0.0, float(coast_sec))
        self.search_sec = max(self.coast_sec, float(search_sec))
        self.smoothing_alpha = float(np.clip(smoothing_alpha, 0.0, 1.0))
        self.width_prediction_enabled = bool(width_prediction_enabled)
        self.width_prediction_replacement_frames = max(
            1, int(width_prediction_replacement_frames)
        )
        self.transverse_clutter_row_fraction = float(
            np.clip(transverse_clutter_row_fraction, 0.0, 1.0)
        )
        self.transverse_clutter_min_rows = max(
            1, int(transverse_clutter_min_rows)
        )
        self.persistent_prediction_enabled = bool(
            persistent_prediction_enabled
        )
        self.allow_unpaired_yellow = bool(allow_unpaired_yellow)
        self.line_width_px = max(1, int(line_width_px))
        self.background_gray = int(np.clip(background_gray, 0, 255))
        self.tracks = {
            "left_white": LaneTrack("left_white"),
            "yellow": LaneTrack("yellow"),
            "right_white": LaneTrack("right_white"),
        }
        self.last_timestamp_sec: float | None = None

    def reset(self) -> None:
        for track in self.tracks.values():
            track.reset()
        self.last_timestamp_sec = None

    def _search_gate(self, track: LaneTrack, timestamp_sec: float) -> float:
        if track.last_seen_sec is None or self.search_sec <= self.coast_sec:
            return self.max_gate_px
        elapsed = track.elapsed(timestamp_sec)
        fraction = np.clip(
            (elapsed - self.coast_sec) / (self.search_sec - self.coast_sec),
            0.0,
            1.0,
        )
        return float(
            self.base_gate_px
            + fraction * (self.max_gate_px - self.base_gate_px)
        )

    def _update_pending(
        self,
        track: LaneTrack,
        candidate: CurveCandidate,
        required_frames: int | None = None,
    ) -> bool:
        required_frames = max(
            1,
            self.confirmation_frames
            if required_frames is None
            else int(required_frames),
        )
        if track.pending is None:
            track.pending = candidate
            track.pending_hits = 1
            return required_frames <= 1
        distance = _curve_distance(
            track.pending.coefficients,
            candidate.coefficients,
            max(track.pending.row_min, candidate.row_min),
            min(track.pending.row_max, candidate.row_max),
        )
        if distance <= self.base_gate_px * 1.5:
            track.pending = candidate
            track.pending_hits += 1
        else:
            track.pending = candidate
            track.pending_hits = 1
        return track.pending_hits >= required_frames

    def _accept(
        self,
        track: LaneTrack,
        candidate: CurveCandidate,
        timestamp_sec: float,
        replace: bool,
    ) -> None:
        previous_row_min = track.row_min
        previous_row_max = track.row_max
        had_coefficients = track.coefficients is not None
        if not had_coefficients or replace:
            track.coefficients = candidate.coefficients.copy()
        else:
            alpha = self.smoothing_alpha
            track.coefficients = (
                alpha * candidate.coefficients
                + (1.0 - alpha) * track.coefficients
            )
        if had_coefficients and not replace:
            track.row_min = min(previous_row_min, candidate.row_min)
            track.row_max = max(previous_row_max, candidate.row_max)
        else:
            track.row_min = candidate.row_min
            track.row_max = candidate.row_max
        track.last_mask = candidate.mask.copy()
        track.last_seen_sec = timestamp_sec
        track.confidence = min(1.0, track.confidence + 0.22)
        track.pending = None
        track.pending_hits = 0
        track.observed_this_frame = True
        track.synthetic_from_width = False

    def _update_track(
        self,
        track: LaneTrack,
        candidate: CurveCandidate | None,
        timestamp_sec: float,
        required_frames: int | None = None,
    ) -> None:
        track.observed_this_frame = False
        if candidate is None:
            track.confidence = max(0.0, track.confidence - 0.08)
        elif track.coefficients is None:
            if self._update_pending(track, candidate, required_frames):
                self._accept(track, candidate, timestamp_sec, replace=True)
        else:
            if (
                not self.persistent_prediction_enabled
                and track.elapsed(timestamp_sec) > self.search_sec
            ):
                if self._update_pending(track, candidate, required_frames):
                    self._accept(
                        track, candidate, timestamp_sec, replace=True
                    )
                return
            tracked_curve = CurveCandidate(
                coefficients=track.coefficients,
                mask=np.zeros_like(candidate.mask),
                row_min=track.row_min,
                row_max=track.row_max,
                area=0,
                span=max(0, track.row_max - track.row_min + 1),
            )
            if not _is_curve_continuation(
                tracked_curve,
                candidate,
                max_row_gap_px=self.max_continuation_gap_px,
                lateral_gate_px=self._search_gate(track, timestamp_sec),
                curvature_gate_gain=self.curvature_gate_gain,
                curvature_max_extra_px=self.curvature_max_extra_px,
            ):
                track.confidence = max(0.0, track.confidence - 0.08)
                if (
                    not self.persistent_prediction_enabled
                    and track.elapsed(timestamp_sec) > self.search_sec
                ):
                    track.reset()
                return
            distance = _curve_distance(
                track.coefficients,
                candidate.coefficients,
                max(track.row_min, candidate.row_min),
                min(track.row_max, candidate.row_max),
            )
            if distance <= self._search_gate(track, timestamp_sec):
                self._accept(track, candidate, timestamp_sec, replace=False)
            elif self._update_pending(track, candidate, required_frames):
                self._accept(track, candidate, timestamp_sec, replace=True)
            else:
                track.confidence = max(0.0, track.confidence - 0.08)

        if (
            not self.persistent_prediction_enabled
            and track.elapsed(timestamp_sec) > self.search_sec
        ):
            track.coefficients = None
            track.last_mask = None
            track.confidence = 0.0

    def _yellow_candidate(
        self,
        mask: np.ndarray,
        timestamp_sec: float,
    ) -> CurveCandidate | None:
        candidates = _component_candidates(mask, min_span_px=6)
        if not candidates:
            return None
        track = self.tracks["yellow"]
        if track.coefficients is not None:
            seed = min(
                candidates,
                key=lambda item: _curve_distance(
                    track.coefficients,
                    item.coefficients,
                    max(track.row_min, item.row_min),
                    min(track.row_max, item.row_max),
                ),
            )
        else:
            center = self.width * 0.5
            seed = max(
                candidates,
                key=lambda item: (
                    item.span * 3.0
                    + item.area * 0.1
                    - abs(
                        float(
                            item.x_at(
                                (item.row_min + item.row_max) * 0.5
                            )
                        )
                        - center
                    )
                    * 0.45
                ),
            )

        merge_gate = max(self.base_gate_px * 1.5, 8.0)
        combined = _continuation_chain(
            candidates,
            seed,
            shape=mask.shape,
            max_row_gap_px=self.max_continuation_gap_px,
            lateral_gate_px=merge_gate,
            curvature_gate_gain=self.curvature_gate_gain,
            curvature_max_extra_px=self.curvature_max_extra_px,
        )
        return _prune_curve_components(combined, self.yellow_fit_gate_px)

    def _yellow_white_offset(
        self,
        yellow: CurveCandidate,
        white_coefficients: np.ndarray,
        white_row_min: int,
        white_row_max: int,
    ) -> float | None:
        row_min = max(yellow.row_min, white_row_min)
        row_max = min(yellow.row_max, white_row_max)
        if row_max < row_min:
            return None
        rows = np.linspace(row_min, row_max, 9, dtype=np.float64)
        offsets = yellow.x_at(rows) - np.polyval(
            white_coefficients, rows
        )
        return float(np.median(offsets))

    def _yellow_matches_side(
        self,
        offset: float | None,
        side: str,
    ) -> bool:
        if offset is None:
            return False
        expected_sign = 1.0 if side == "left_white" else -1.0
        return (
            offset * expected_sign > 0.0
            and abs(offset) >= self.min_offset_px
            and abs(offset)
            <= self.expected_offset_px + self.width_tolerance_px
        )

    def _boundary_side_from_near_field(
        self,
        candidate: CurveCandidate,
    ) -> str | None:
        """Infer a physical boundary side from its closest visible rows."""
        band_rows = np.linspace(
            max(candidate.row_min, candidate.row_max - 5),
            candidate.row_max,
            6,
            dtype=np.float64,
        )
        near_x = float(np.median(candidate.x_at(band_rows)))
        center_x = self.width * 0.5
        margin = max(float(self.line_width_px), self.base_gate_px)
        if near_x < center_x - margin:
            return "left_white"
        if near_x > center_x + margin:
            return "right_white"
        return None

    def _yellow_respects_white_corridor(
        self,
        yellow: CurveCandidate | None,
        white_mask: np.ndarray,
        timestamp_sec: float,
    ) -> bool:
        if yellow is None:
            return False

        supported_sides = 0
        for side in ("left_white", "right_white"):
            track = self.tracks[side]
            if (
                track.coefficients is None
                or track.elapsed(timestamp_sec) > self.search_sec
            ):
                continue
            offset = self._yellow_white_offset(
                yellow,
                track.coefficients,
                track.row_min,
                track.row_max,
            )
            # A short yellow dash often has no longitudinal overlap with one
            # of the visible white components. That side provides no evidence
            # either way and must not erase a valid observation supported by
            # the other boundary.
            if offset is None:
                continue
            supported_sides += 1
            if not self._yellow_matches_side(offset, side):
                return False
        if supported_sides > 0:
            return True

        raw_whites = _component_candidates(white_mask, min_span_px=10)
        if not raw_whites:
            if not self.allow_unpaired_yellow:
                return False
            midpoint_x = float(
                yellow.x_at((yellow.row_min + yellow.row_max) * 0.5)
            )
            return self.width * 0.20 <= midpoint_x <= self.width * 0.80
        has_left_support = False
        has_right_support = False
        for white in raw_whites:
            offset = self._yellow_white_offset(
                yellow,
                white.coefficients,
                white.row_min,
                white.row_max,
            )
            boundary_side = self._boundary_side_from_near_field(white)
            if boundary_side is not None and offset is not None:
                if not self._yellow_matches_side(offset, boundary_side):
                    return False
                if boundary_side == "left_white":
                    has_left_support = True
                else:
                    has_right_support = True
                continue
            if self._yellow_matches_side(offset, "left_white"):
                has_left_support = True
            if self._yellow_matches_side(offset, "right_white"):
                has_right_support = True
        if has_left_support and has_right_support:
            return True
        midpoint_x = float(
            yellow.x_at((yellow.row_min + yellow.row_max) * 0.5)
        )
        return (
            (has_left_support or has_right_support)
            and self.width * 0.20 <= midpoint_x <= self.width * 0.80
        )

    def _yellow_respects_dominant_near_boundaries(
        self,
        yellow: CurveCandidate,
        white_mask: np.ndarray,
    ) -> bool:
        strongest: dict[str, tuple[float, CurveCandidate]] = {}
        for white in _component_candidates(white_mask, min_span_px=10):
            side = self._boundary_side_from_near_field(white)
            if side is None:
                continue
            offset = self._yellow_white_offset(
                yellow,
                white.coefficients,
                white.row_min,
                white.row_max,
            )
            if offset is None:
                continue
            quality = white.span * 3.0 + white.area * 0.1
            previous = strongest.get(side)
            if previous is None or quality > previous[0]:
                strongest[side] = (quality, white)

        for side, (_, white) in strongest.items():
            offset = self._yellow_white_offset(
                yellow,
                white.coefficients,
                white.row_min,
                white.row_max,
            )
            if offset is None:
                continue
            if not self._yellow_matches_side(offset, side):
                return False
        return True

    def _recover_yellow_from_white_corridor(
        self,
        white_mask: np.ndarray,
    ) -> CurveCandidate | None:
        """Recover a washed-out yellow dash classified as bright white."""
        candidates = _component_candidates(white_mask, min_span_px=6)
        if len(candidates) < 3:
            return None

        max_dash_span_px = max(
            6,
            int(round(0.50 / self.forward_m_per_px)),
        )
        eligible: list[tuple[float, CurveCandidate]] = []
        for center_candidate in candidates:
            if center_candidate.span > max_dash_span_px:
                continue
            midpoint_x = float(
                center_candidate.x_at(
                    (center_candidate.row_min + center_candidate.row_max)
                    * 0.5
                )
            )
            if not self.width * 0.20 <= midpoint_x <= self.width * 0.80:
                continue

            side_errors: dict[str, float] = {}
            for side in ("left_white", "right_white"):
                errors = []
                for boundary in candidates:
                    if boundary is center_candidate:
                        continue
                    offset = self._yellow_white_offset(
                        center_candidate,
                        boundary.coefficients,
                        boundary.row_min,
                        boundary.row_max,
                    )
                    if not self._yellow_matches_side(offset, side):
                        continue
                    errors.append(
                        abs(abs(float(offset)) - self.expected_offset_px)
                    )
                if errors:
                    side_errors[side] = min(errors)

            # Color is ambiguous here, so require physical support on both
            # sides before relabeling a white-looking component as yellow.
            if len(side_errors) != 2:
                continue
            score = (
                side_errors["left_white"]
                + side_errors["right_white"]
                + 0.15 * abs(midpoint_x - self.width * 0.5)
            )
            eligible.append((score, center_candidate))
        if not eligible:
            return None
        seed = min(eligible, key=lambda item: item[0])[1]
        combined = _continuation_chain(
            [item[1] for item in eligible],
            seed,
            shape=white_mask.shape,
            max_row_gap_px=self.max_continuation_gap_px,
            lateral_gate_px=max(self.base_gate_px * 1.5, 8.0),
            curvature_gate_gain=self.curvature_gate_gain,
            curvature_max_extra_px=self.curvature_max_extra_px,
        )
        return _prune_curve_components(combined, self.yellow_fit_gate_px)

    def _white_candidates(
        self,
        mask: np.ndarray,
        yellow: CurveCandidate | None,
        timestamp_sec: float,
    ) -> dict[str, CurveCandidate | None]:
        candidates = _component_candidates(mask, min_span_px=10)
        output: dict[str, CurveCandidate | None] = {
            "left_white": None,
            "right_white": None,
        }
        scored: dict[str, list[tuple[float, CurveCandidate]]] = {
            "left_white": [],
            "right_white": [],
        }
        yellow_coefficients = None
        yellow_row_min = 0
        yellow_row_max = self.height - 1
        if yellow is not None:
            yellow_coefficients = yellow.coefficients
            yellow_row_min = yellow.row_min
            yellow_row_max = yellow.row_max
        else:
            yellow_track = self.tracks["yellow"]
            if (
                yellow_track.coefficients is not None
                and yellow_track.elapsed(timestamp_sec) <= self.search_sec
            ):
                yellow_coefficients = yellow_track.coefficients
                yellow_row_min = yellow_track.row_min
                yellow_row_max = yellow_track.row_max

        for candidate in candidates:
            near_field_side = self._boundary_side_from_near_field(candidate)
            sample_row_min = candidate.row_min
            sample_row_max = candidate.row_max
            if yellow_coefficients is not None:
                sample_row_min = max(sample_row_min, yellow_row_min)
                sample_row_max = min(sample_row_max, yellow_row_max)
                if sample_row_max < sample_row_min:
                    continue
            sample_rows = np.linspace(
                sample_row_min,
                sample_row_max,
                7,
                dtype=np.float64,
            )
            candidate_x = candidate.x_at(sample_rows)
            temporal_score = 0.0
            if yellow_coefficients is not None:
                offsets = candidate_x - np.polyval(
                    yellow_coefficients, sample_rows
                )
                median_offset = float(np.median(offsets))
                if abs(median_offset) < self.min_offset_px:
                    continue
                if abs(median_offset) > (
                    self.expected_offset_px + self.width_tolerance_px
                ):
                    continue
                same_side_ratio = float(
                    np.mean(np.sign(offsets) == np.sign(median_offset))
                )
                if same_side_ratio < 0.85:
                    continue
                side = "left_white" if median_offset < 0.0 else "right_white"
                if near_field_side is not None and near_field_side != side:
                    continue
                topology_score = (
                    abs(abs(median_offset) - self.expected_offset_px)
                    / max(1.0, self.width_tolerance_px)
                )
            else:
                median_x = float(np.median(candidate_x))
                side = near_field_side or (
                    "left_white" if median_x < self.width * 0.5
                    else "right_white"
                )
                topology_score = 0.5

            track = self.tracks[side]
            other_side = (
                "right_white" if side == "left_white" else "left_white"
            )
            other_track = self.tracks[other_side]
            if (
                self.width_prediction_enabled
                and yellow_coefficients is not None
                and other_track.coefficients is not None
                and other_track.elapsed(timestamp_sec) <= self.search_sec
                and abs(abs(median_offset) - self.expected_offset_px)
                > self.base_gate_px
            ):
                # Once yellow and one physical boundary establish the corridor,
                # do not let an interior reflection become the missing side.
                continue
            if (
                yellow_coefficients is not None
                and track.coefficients is not None
                and track.elapsed(timestamp_sec) <= self.search_sec
            ):
                tracked_offsets = np.polyval(
                    track.coefficients, sample_rows
                ) - np.polyval(yellow_coefficients, sample_rows)
                tracked_offset = abs(float(np.median(tracked_offsets)))
                if (
                    abs(median_offset) + self.base_gate_px
                    < tracked_offset
                ):
                    continue
            if track.coefficients is not None:
                distance = _curve_distance(
                    track.coefficients,
                    candidate.coefficients,
                    max(track.row_min, candidate.row_min),
                    min(track.row_max, candidate.row_max),
                )
                temporal_score = distance / max(
                    1.0, self._search_gate(track, timestamp_sec)
                )
            quality_bonus = min(1.0, candidate.span / max(1.0, self.height))
            score = (
                topology_score
                + 1.6 * temporal_score
                - 0.35 * quality_bonus
            )
            scored[side].append((score, candidate))

        for side, items in scored.items():
            if items:
                seed = min(items, key=lambda item: item[0])[1]
                output[side] = _continuation_chain(
                    [item[1] for item in items],
                    seed,
                    shape=mask.shape,
                    max_row_gap_px=self.max_continuation_gap_px,
                    lateral_gate_px=max(self.base_gate_px, 5.0),
                    curvature_gate_gain=self.curvature_gate_gain,
                    curvature_max_extra_px=self.curvature_max_extra_px,
                )
        if yellow_coefficients is None:
            left = output["left_white"]
            right = output["right_white"]
            if left is not None and right is not None:
                row_min = max(left.row_min, right.row_min)
                row_max = min(left.row_max, right.row_max)
                if row_max >= row_min:
                    rows = np.linspace(
                        row_min, row_max, 9, dtype=np.float64
                    )
                    separation = right.x_at(rows) - left.x_at(rows)
                    min_pair_width_px = max(
                        2.0 * self.min_offset_px,
                        2.0 * self.expected_offset_px
                        - self.width_tolerance_px,
                    )
                    if (
                        float(np.median(separation))
                        < min_pair_width_px
                        or float(np.mean(separation > 0.0)) < 0.85
                    ):
                        left_quality = left.span * 3.0 + left.area * 0.1
                        right_quality = right.span * 3.0 + right.area * 0.1
                        weaker_side = (
                            "left_white"
                            if left_quality < right_quality
                            else "right_white"
                        )
                        output[weaker_side] = None
        return output

    def _draw_curve(
        self,
        coefficients: np.ndarray,
        row_min: int,
        row_max: int,
    ) -> np.ndarray:
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        row_min = max(0, min(self.height - 1, int(row_min)))
        row_max = max(row_min, min(self.height - 1, int(row_max)))
        rows = np.arange(row_min, row_max + 1, dtype=np.float64)
        xs = np.polyval(coefficients, rows)
        valid = (xs >= 0.0) & (xs < self.width)
        if int(np.count_nonzero(valid)) < 2:
            return mask
        points = np.column_stack(
            (
                np.rint(xs[valid]).astype(np.int32),
                rows[valid].astype(np.int32),
            )
        )
        cv2.polylines(
            mask,
            [points],
            False,
            255,
            thickness=self.line_width_px,
            lineType=cv2.LINE_8,
        )
        return mask

    def _track_output(
        self,
        track: LaneTrack,
        timestamp_sec: float,
    ) -> tuple[np.ndarray, str]:
        status = track.status(timestamp_sec, self.coast_sec, self.search_sec)
        empty = np.zeros((self.height, self.width), dtype=np.uint8)
        if status == "confirmed" and track.last_mask is not None:
            # YOLO boundary masks can be sparse along a physical white tape.
            # Render the robustly fitted white curve so the model receives the
            # same continuous line representation in rosbag and live-camera
            # processing. Yellow dashes retain their observed segmentation.
            if track.name != "yellow" and track.coefficients is not None:
                return (
                    self._draw_curve(
                        track.coefficients,
                        track.row_min,
                        track.row_max,
                    ),
                    status,
                )
            return track.last_mask.copy(), status
        # A short association miss is still a usable observation for the
        # controller. Keep the last accepted curve through the expanding
        # search window instead of publishing an all-background frame.
        if (
            track.coefficients is not None
            and (
                status == "coasting"
                or (
                    status == "searching"
                    and not self.persistent_prediction_enabled
                )
            )
        ):
            return (
                self._draw_curve(
                    track.coefficients,
                    track.row_min,
                    track.row_max,
                ),
                status,
            )
        if (
            self.persistent_prediction_enabled
            and track.coefficients is not None
        ):
            predicted_status = (
                "width_predicted"
                if track.synthetic_from_width
                else "persistent_predicted"
            )
            if track.name == "yellow" and track.last_mask is not None:
                return track.last_mask.copy(), predicted_status
            return (
                self._draw_curve(
                    track.coefficients,
                    track.row_min,
                    track.row_max,
                ),
                predicted_status,
            )
        return empty, status

    def _width_predicted_white(
        self,
        side: str,
        reference_track: LaneTrack,
        yellow_track: LaneTrack,
    ) -> np.ndarray:
        if (
            reference_track.coefficients is None
            or yellow_track.coefficients is None
        ):
            return np.zeros((self.height, self.width), dtype=np.uint8)

        row_min = max(reference_track.row_min, yellow_track.row_min)
        row_max = min(reference_track.row_max, yellow_track.row_max)
        if row_max < row_min:
            return np.zeros((self.height, self.width), dtype=np.uint8)
        rows = np.linspace(row_min, row_max, 9, dtype=np.float64)
        reference_offsets = (
            np.polyval(reference_track.coefficients, rows)
            - np.polyval(yellow_track.coefficients, rows)
        )
        median_offset = float(np.median(reference_offsets))
        reference_is_left = side == "right_white"
        if (reference_is_left and median_offset >= 0.0) or (
            not reference_is_left and median_offset <= 0.0
        ):
            return np.zeros((self.height, self.width), dtype=np.uint8)
        if (
            abs(abs(median_offset) - self.expected_offset_px)
            > self.width_tolerance_px
        ):
            return np.zeros((self.height, self.width), dtype=np.uint8)

        # Keep the observed boundary curvature and translate it by the known
        # white-center spacing. This avoids extrapolating short yellow dashes.
        offset = (
            -2.0 * self.expected_offset_px
            if side == "left_white"
            else 2.0 * self.expected_offset_px
        )
        coefficients = reference_track.coefficients.copy()
        coefficients[-1] += offset
        return self._draw_curve(
            coefficients,
            reference_track.row_min,
            reference_track.row_max,
        )

    def _width_offset_error(
        self,
        track: LaneTrack,
        yellow_track: LaneTrack,
        expected_sign: float,
    ) -> float:
        if track.coefficients is None or yellow_track.coefficients is None:
            return float("inf")
        row_min = max(track.row_min, yellow_track.row_min)
        row_max = min(track.row_max, yellow_track.row_max)
        if row_max < row_min:
            return float("inf")
        rows = np.linspace(row_min, row_max, 9, dtype=np.float64)
        offsets = (
            np.polyval(track.coefficients, rows)
            - np.polyval(yellow_track.coefficients, rows)
        )
        median_offset = float(np.median(offsets))
        if median_offset * expected_sign <= 0.0:
            return float("inf")
        return abs(abs(median_offset) - self.expected_offset_px)

    def _remove_interior_track_when_corridor_is_known(self) -> None:
        if not self.width_prediction_enabled:
            return
        yellow_track = self.tracks["yellow"]
        if yellow_track.coefficients is None:
            return
        errors = {
            "left_white": self._width_offset_error(
                self.tracks["left_white"], yellow_track, -1.0
            ),
            "right_white": self._width_offset_error(
                self.tracks["right_white"], yellow_track, 1.0
            ),
        }
        left_valid = errors["left_white"] <= self.base_gate_px
        right_valid = errors["right_white"] <= self.base_gate_px
        if left_valid and not right_valid:
            self.tracks["right_white"].reset()
        elif right_valid and not left_valid:
            self.tracks["left_white"].reset()

    def _matches_width_prediction(
        self,
        side: str,
        candidate: CurveCandidate | None,
        timestamp_sec: float,
    ) -> bool:
        if not self.width_prediction_enabled or candidate is None:
            return False
        yellow_track = self.tracks["yellow"]
        other_side = (
            "right_white" if side == "left_white" else "left_white"
        )
        other_track = self.tracks[other_side]
        if (
            yellow_track.coefficients is None
            or other_track.coefficients is None
            or other_track.elapsed(timestamp_sec) > self.search_sec
        ):
            return False
        row_min = max(candidate.row_min, yellow_track.row_min)
        row_max = min(candidate.row_max, yellow_track.row_max)
        if row_max < row_min:
            return False
        rows = np.linspace(row_min, row_max, 9, dtype=np.float64)
        offsets = (
            candidate.x_at(rows)
            - np.polyval(yellow_track.coefficients, rows)
        )
        median_offset = float(np.median(offsets))
        expected_sign = -1.0 if side == "left_white" else 1.0
        return (
            median_offset * expected_sign > 0.0
            and abs(abs(median_offset) - self.expected_offset_px)
            <= self.base_gate_px
        )

    def _has_transverse_clutter(self, white_mask: np.ndarray) -> bool:
        if self.transverse_clutter_row_fraction <= 0.0:
            return False
        threshold = max(
            1,
            int(np.ceil(self.width * self.transverse_clutter_row_fraction)),
        )
        row_counts = np.count_nonzero(white_mask > 0, axis=1)
        return (
            int(np.count_nonzero(row_counts >= threshold))
            >= self.transverse_clutter_min_rows
        )

    def _held_track_output(self, track: LaneTrack) -> np.ndarray:
        if track.coefficients is None:
            return np.zeros((self.height, self.width), dtype=np.uint8)
        return self._draw_curve(
            track.coefficients,
            track.row_min,
            track.row_max,
        )

    def update(
        self,
        white_mask: np.ndarray,
        yellow_mask: np.ndarray,
        timestamp_sec: float,
    ) -> TrackingResult:
        if white_mask.shape != (self.height, self.width):
            raise ValueError("white mask dimensions do not match tracker")
        if yellow_mask.shape != (self.height, self.width):
            raise ValueError("yellow mask dimensions do not match tracker")
        timestamp_sec = float(timestamp_sec)
        if (
            self.last_timestamp_sec is not None
            and timestamp_sec + 0.05 < self.last_timestamp_sec
        ):
            self.reset()
        self.last_timestamp_sec = timestamp_sec

        transverse_clutter = self._has_transverse_clutter(white_mask)
        if not transverse_clutter:
            yellow_candidate = self._yellow_candidate(
                yellow_mask, timestamp_sec
            )
            if not self._yellow_respects_white_corridor(
                yellow_candidate,
                white_mask,
                timestamp_sec,
            ):
                yellow_candidate = None
            recovered_yellow = self._recover_yellow_from_white_corridor(
                white_mask
            )
            if yellow_candidate is None:
                yellow_candidate = recovered_yellow
            elif recovered_yellow is not None:
                # Saturation can split one physical dash between the yellow
                # and white masks. Keep the color observation in overlapping
                # rows and use recovered pixels only to supplement other rows.
                recovered_mask = recovered_yellow.mask.copy()
                occupied_rows = np.any(yellow_candidate.mask > 0, axis=1)
                occupied_rows = np.convolve(
                    occupied_rows.astype(np.uint8),
                    np.ones(7, dtype=np.uint8),
                    mode="same",
                ) > 0
                recovered_mask[occupied_rows, :] = 0
                recovered_yellow = _fit_mask(recovered_mask)

            if yellow_candidate is not None and recovered_yellow is not None:
                yellow_candidate = _continuation_chain(
                    [yellow_candidate, recovered_yellow],
                    yellow_candidate,
                    shape=yellow_mask.shape,
                    max_row_gap_px=self.max_continuation_gap_px,
                    lateral_gate_px=max(self.base_gate_px * 1.5, 8.0),
                    curvature_gate_gain=self.curvature_gate_gain,
                    curvature_max_extra_px=self.curvature_max_extra_px,
                )
                yellow_candidate = _prune_curve_components(
                    yellow_candidate, self.yellow_fit_gate_px
                )

            if (
                yellow_candidate is not None
                and not self._yellow_respects_dominant_near_boundaries(
                    yellow_candidate,
                    white_mask,
                )
            ):
                yellow_candidate = None
                recovered_yellow = None

            white_tracking_mask = white_mask
            if recovered_yellow is not None:
                white_tracking_mask = white_mask.copy()
                white_tracking_mask[recovered_yellow.mask > 0] = 0

            white_candidates = self._white_candidates(
                white_tracking_mask, yellow_candidate, timestamp_sec
            )
            for side in ("left_white", "right_white"):
                required_frames = (
                    self.width_prediction_replacement_frames
                    if self._matches_width_prediction(
                        side,
                        white_candidates[side],
                        timestamp_sec,
                    )
                    else self.confirmation_frames
                )
                self._update_track(
                    self.tracks[side],
                    white_candidates[side],
                    timestamp_sec,
                    required_frames,
                )
            self._update_track(
                self.tracks["yellow"], yellow_candidate, timestamp_sec
            )
            self._remove_interior_track_when_corridor_is_known()
        elif not self.persistent_prediction_enabled:
            # Transverse clutter is a rejected observation, not proof that
            # the road disappeared. Keep the last accepted curves so the
            # normal coast/search timeout can handle this short dropout.
            for track in self.tracks.values():
                track.observed_this_frame = False

        if transverse_clutter:
            if self.persistent_prediction_enabled:
                tracked_yellow = self._held_track_output(
                    self.tracks["yellow"]
                )
                yellow_status = (
                    "startline_hold" if np.any(tracked_yellow) else "lost"
                )
            else:
                tracked_yellow, yellow_status = self._track_output(
                    self.tracks["yellow"], timestamp_sec
                )
        else:
            tracked_yellow, yellow_status = self._track_output(
                self.tracks["yellow"], timestamp_sec
            )
        tracked_white = np.zeros_like(white_mask)
        statuses = {"yellow": yellow_status}
        sources: dict[str, str] = {}
        side_masks: dict[str, np.ndarray] = {}
        for side in ("left_white", "right_white"):
            if transverse_clutter:
                if self.persistent_prediction_enabled:
                    side_mask = self._held_track_output(self.tracks[side])
                    status = (
                        "startline_hold" if np.any(side_mask) else "lost"
                    )
                else:
                    side_mask, status = self._track_output(
                        self.tracks[side], timestamp_sec
                    )
            else:
                side_mask, status = self._track_output(
                    self.tracks[side], timestamp_sec
                )
            side_masks[side] = side_mask
            statuses[side] = status
            sources[side] = status

        yellow_active = yellow_status in {
            "confirmed",
            "coasting",
            "startline_hold",
            "persistent_predicted",
        }
        for side, other_side in (
            ("left_white", "right_white"),
            ("right_white", "left_white"),
        ):
            if np.any(side_masks[side]) and statuses[side] in {
                "confirmed",
                "coasting",
                "startline_hold",
            }:
                continue
            other_status = statuses[other_side]
            reference_is_fresher = other_status in {
                "confirmed",
                "coasting",
                "startline_hold",
            }
            if (
                self.width_prediction_enabled
                and yellow_active
                and (
                    reference_is_fresher
                    or not np.any(side_masks[side])
                )
                and other_status
                in {
                    "confirmed",
                    "coasting",
                    "startline_hold",
                    "persistent_predicted",
                }
            ):
                predicted = self._width_predicted_white(
                    side,
                    self.tracks[other_side],
                    self.tracks["yellow"],
                )
                if np.any(predicted):
                    side_masks[side] = predicted
                    statuses[side] = "width_predicted"
                    sources[side] = "width_predicted"
                    predicted_candidate = _fit_mask(predicted)
                    if predicted_candidate is not None:
                        predicted_track = self.tracks[side]
                        predicted_track.coefficients = (
                            predicted_candidate.coefficients.copy()
                        )
                        predicted_track.row_min = predicted_candidate.row_min
                        predicted_track.row_max = predicted_candidate.row_max
                        predicted_track.last_mask = predicted.copy()
                        predicted_track.synthetic_from_width = True
                        predicted_track.confidence = max(
                            predicted_track.confidence,
                            0.35,
                        )

        for side_mask in side_masks.values():
            tracked_white[side_mask > 0] = 255
        tracked_white[tracked_yellow > 0] = 0

        road = np.full(
            (self.height, self.width, 3),
            self.background_gray,
            dtype=np.uint8,
        )
        road[tracked_white > 0] = (255, 255, 255)
        road[tracked_yellow > 0] = (0, 220, 255)
        debug = self._debug_image(
            white_mask,
            yellow_mask,
            side_masks,
            tracked_yellow,
            statuses,
            sources,
        )
        return TrackingResult(
            road_image=road,
            white_mask=tracked_white,
            yellow_mask=tracked_yellow,
            debug_image=debug,
            statuses=statuses,
        )

    def _debug_image(
        self,
        input_white: np.ndarray,
        input_yellow: np.ndarray,
        side_masks: dict[str, np.ndarray],
        tracked_yellow: np.ndarray,
        statuses: dict[str, str],
        sources: dict[str, str],
    ) -> np.ndarray:
        debug = np.full(
            (self.height, self.width, 3),
            self.background_gray,
            dtype=np.uint8,
        )
        selected_white = np.zeros_like(input_white)
        for mask in side_masks.values():
            selected_white[mask > 0] = 255
        rejected_white = (input_white > 0) & (selected_white == 0)
        rejected_yellow = (input_yellow > 0) & (tracked_yellow == 0)
        debug[rejected_white] = (35, 35, 130)
        debug[rejected_yellow] = (80, 45, 120)
        debug[tracked_yellow > 0] = (0, 220, 255)

        colors = {
            "confirmed": (60, 220, 60),
            "coasting": (0, 170, 255),
            "width_predicted": (255, 210, 0),
            "startline_hold": (0, 170, 255),
            "persistent_predicted": (200, 120, 255),
            "searching": (0, 100, 255),
        }
        for side, mask in side_masks.items():
            debug[mask > 0] = colors.get(sources[side], (130, 130, 130))

        text = (
            f"L:{statuses['left_white']} "
            f"Y:{statuses['yellow']} "
            f"R:{statuses['right_white']}"
        )
        cv2.putText(
            debug,
            text,
            (4, 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        return debug
