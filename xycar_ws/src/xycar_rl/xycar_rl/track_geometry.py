from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re
import xml.etree.ElementTree as ET

import numpy as np


YELLOW_DASH_PREFIX = "yellow_centerline_dash_"


def wrap_angle(angle_rad: float) -> float:
    return float((angle_rad + math.pi) % (2.0 * math.pi) - math.pi)


@dataclass(frozen=True)
class TrackProjection:
    x: float
    y: float
    tangent_yaw: float
    progress_m: float
    progress_fraction: float
    cross_track_error_m: float
    heading_error_rad: float
    segment_index: int


@dataclass(frozen=True)
class SpawnPose:
    x: float
    y: float
    yaw: float
    progress_m: float


def _numeric_suffix(name: str) -> int:
    match = re.search(r"(\d+)$", name)
    return int(match.group(1)) if match else 0


def _ordered_nearest_neighbours(
    named_points: list[tuple[str, np.ndarray]],
) -> np.ndarray:
    if len(named_points) < 4:
        raise ValueError("at least four CAD centerline points are required")
    points_by_name = {name: point for name, point in named_points}
    start_name = min(points_by_name, key=_numeric_suffix)
    remaining = set(points_by_name) - {start_name}
    ordered = [points_by_name[start_name]]
    current_name = start_name
    while remaining:
        current = points_by_name[current_name]
        current_name = min(
            remaining,
            key=lambda name: float(
                np.sum((points_by_name[name] - current) ** 2)
            ),
        )
        ordered.append(points_by_name[current_name])
        remaining.remove(current_name)
    result = np.asarray(ordered, dtype=np.float64)
    neighbour_lengths = np.linalg.norm(
        np.roll(result, -1, axis=0) - result,
        axis=1,
    )
    if float(neighbour_lengths.max()) > 1.25:
        raise ValueError("CAD centerline ordering contains a gap larger than 1.25m")
    return result


def _sample_closed_catmull_rom(
    anchors: np.ndarray,
    samples_per_anchor: int,
) -> np.ndarray:
    samples_per_anchor = max(2, int(samples_per_anchor))
    sampled: list[np.ndarray] = []
    count = int(anchors.shape[0])
    for index in range(count):
        p0 = anchors[(index - 1) % count]
        p1 = anchors[index]
        p2 = anchors[(index + 1) % count]
        p3 = anchors[(index + 2) % count]
        for step in range(samples_per_anchor):
            t = step / samples_per_anchor
            t2 = t * t
            t3 = t2 * t
            point = 0.5 * (
                2.0 * p1
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )
            sampled.append(point)
    return np.asarray(sampled, dtype=np.float64)


def _offset_polyline_left(points: np.ndarray, offset_m: float) -> np.ndarray:
    if abs(offset_m) < 1.0e-12:
        return points.copy()
    tangent = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    norms = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent = tangent / np.maximum(norms, 1.0e-9)
    left_normals = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    return points + float(offset_m) * left_normals


class TrackReference:
    """Closed CAD-derived driving reference with fast point projection."""

    def __init__(self, points: np.ndarray) -> None:
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 8:
            raise ValueError("track points must have shape [N, 2], N >= 8")
        self.points = points
        self.segment_starts = points
        self.segment_vectors = np.roll(points, -1, axis=0) - points
        self.segment_lengths = np.linalg.norm(self.segment_vectors, axis=1)
        if np.any(self.segment_lengths <= 1.0e-6):
            raise ValueError("track contains zero-length segments")
        self.segment_length_sq = self.segment_lengths ** 2
        self.cumulative_lengths = np.concatenate(
            ([0.0], np.cumsum(self.segment_lengths))
        )
        self.length_m = float(self.cumulative_lengths[-1])

    @classmethod
    def from_sdf(
        cls,
        sdf_path: str | Path,
        *,
        target_right_offset_m: float = 0.05,
        samples_per_anchor: int = 20,
        reverse_direction: bool = True,
    ) -> "TrackReference":
        path = Path(sdf_path).expanduser().resolve()
        world = ET.parse(path).getroot().find("world")
        if world is None:
            raise ValueError(f"SDF contains no world: {path}")
        named_points: list[tuple[str, np.ndarray]] = []
        for model in world.findall("model"):
            name = model.attrib.get("name", "")
            if not name.startswith(YELLOW_DASH_PREFIX):
                continue
            pose = model.findtext("pose")
            if not pose:
                continue
            values = [float(value) for value in pose.split()]
            if len(values) < 2:
                continue
            named_points.append((name, np.asarray(values[:2], dtype=np.float64)))
        anchors = _ordered_nearest_neighbours(named_points)
        yellow_centerline = _sample_closed_catmull_rom(
            anchors,
            samples_per_anchor,
        )
        # The CAD dash numbering is clockwise, while the competition's driving
        # direction is counter-clockwise.
        if reverse_direction:
            yellow_centerline = yellow_centerline[::-1].copy()
        # The local left normal is recomputed after reversing the samples, so a
        # right-side target is always a negative left-normal offset.
        target = _offset_polyline_left(
            yellow_centerline,
            -abs(float(target_right_offset_m)),
        )
        return cls(target)

    def project(
        self,
        x: float,
        y: float,
        yaw: float,
        *,
        hint_segment_index: int | None = None,
        search_radius_segments: int = 140,
    ) -> TrackProjection:
        position = np.asarray([x, y], dtype=np.float64)
        candidate_indices = np.arange(len(self.segment_starts), dtype=np.int64)
        if hint_segment_index is not None:
            radius = max(2, int(search_radius_segments))
            offsets = np.arange(-radius, radius + 1, dtype=np.int64)
            candidate_indices = (
                int(hint_segment_index) + offsets
            ) % len(self.segment_starts)
        starts = self.segment_starts[candidate_indices]
        vectors = self.segment_vectors[candidate_indices]
        length_sq = self.segment_length_sq[candidate_indices]
        relative = position - starts
        fractions = np.einsum(
            "ij,ij->i", relative, vectors
        ) / length_sq
        fractions = np.clip(fractions, 0.0, 1.0)
        projected = starts + fractions[:, None] * vectors
        squared_distance = np.sum((projected - position) ** 2, axis=1)
        local_index = int(np.argmin(squared_distance))
        index = int(candidate_indices[local_index])
        tangent = self.segment_vectors[index] / self.segment_lengths[index]
        delta = position - projected[local_index]
        signed_error = float(tangent[0] * delta[1] - tangent[1] * delta[0])
        tangent_yaw = math.atan2(float(tangent[1]), float(tangent[0]))
        progress_m = float(
            self.cumulative_lengths[index]
            + fractions[local_index] * self.segment_lengths[index]
        )
        return TrackProjection(
            x=float(projected[local_index, 0]),
            y=float(projected[local_index, 1]),
            tangent_yaw=tangent_yaw,
            progress_m=progress_m,
            progress_fraction=progress_m / self.length_m,
            cross_track_error_m=signed_error,
            heading_error_rad=wrap_angle(float(yaw) - tangent_yaw),
            segment_index=index,
        )

    def progress_delta(self, previous_m: float, current_m: float) -> float:
        delta = float(current_m) - float(previous_m)
        half = 0.5 * self.length_m
        if delta > half:
            delta -= self.length_m
        elif delta < -half:
            delta += self.length_m
        return delta

    def sample_at_progress(self, progress_m: float) -> tuple[np.ndarray, float]:
        progress = float(progress_m) % self.length_m
        index = min(
            len(self.segment_lengths) - 1,
            int(
                np.searchsorted(
                    self.cumulative_lengths, progress, side="right"
                )
                - 1
            ),
        )
        fraction = (
            progress - self.cumulative_lengths[index]
        ) / self.segment_lengths[index]
        point = (
            self.segment_starts[index]
            + fraction * self.segment_vectors[index]
        )
        tangent = self.segment_vectors[index] / self.segment_lengths[index]
        yaw = math.atan2(float(tangent[1]), float(tangent[0]))
        return point.copy(), yaw

    def curvature_at(self, progress_m: float, sample_distance_m: float = 0.12) -> float:
        distance = max(0.02, float(sample_distance_m))
        _, yaw_before = self.sample_at_progress(progress_m - 0.5 * distance)
        _, yaw_after = self.sample_at_progress(progress_m + 0.5 * distance)
        return wrap_angle(yaw_after - yaw_before) / distance

    def max_abs_curvature_ahead(
        self,
        progress_m: float,
        *,
        preview_distance_m: float = 1.5,
        sample_count: int = 12,
        curvature_sample_distance_m: float = 0.30,
    ) -> float:
        """Return the strongest upcoming curvature for anticipatory speed control."""
        preview = max(0.12, float(preview_distance_m))
        offsets = np.linspace(0.12, preview, max(2, int(sample_count)))
        return max(
            abs(
                self.curvature_at(
                    float(progress_m) + float(offset),
                    sample_distance_m=curvature_sample_distance_m,
                )
            )
            for offset in offsets
        )

    def pose_at(
        self,
        progress_fraction: float,
        *,
        lateral_error_m: float = 0.0,
        yaw_error_rad: float = 0.0,
    ) -> SpawnPose:
        fraction = float(progress_fraction) % 1.0
        progress_m = fraction * self.length_m
        index = min(
            len(self.segment_lengths) - 1,
            int(np.searchsorted(self.cumulative_lengths, progress_m, side="right") - 1),
        )
        segment_fraction = (
            progress_m - self.cumulative_lengths[index]
        ) / self.segment_lengths[index]
        tangent = self.segment_vectors[index] / self.segment_lengths[index]
        point = self.segment_starts[index] + segment_fraction * self.segment_vectors[index]
        left = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
        point = point + float(lateral_error_m) * left
        yaw = math.atan2(float(tangent[1]), float(tangent[0]))
        return SpawnPose(
            x=float(point[0]),
            y=float(point[1]),
            yaw=wrap_angle(yaw + float(yaw_error_rad)),
            progress_m=progress_m,
        )
