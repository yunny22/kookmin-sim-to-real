from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import mean


@dataclass(frozen=True)
class CandidateSummary:
    name: str
    episode_count: int
    completed_laps: int
    lane_departures: int
    other_failures: int
    mean_success_lap_time_sec: float
    fastest_success_lap_time_sec: float
    eligible: bool


def summarize_candidate(
    name: str,
    episodes: list[dict],
    *,
    required_safe_laps: int,
) -> CandidateSummary:
    completed = [
        row for row in episodes if str(row.get("reason")) == "lap_complete"
    ]
    departures = sum(
        str(row.get("reason")) in {"off_track", "collision"} for row in episodes
    )
    other_failures = len(episodes) - len(completed) - departures
    lap_times = [
        float(row["lap_time_sec"])
        for row in completed
        if math.isfinite(float(row.get("lap_time_sec", float("nan"))))
    ]
    required = max(1, int(required_safe_laps))
    eligible = (
        len(episodes) >= required
        and len(completed) == len(episodes)
        and departures == 0
    )
    return CandidateSummary(
        name=str(name),
        episode_count=len(episodes),
        completed_laps=len(completed),
        lane_departures=int(departures),
        other_failures=int(other_failures),
        mean_success_lap_time_sec=(
            float(mean(lap_times)) if lap_times else float("inf")
        ),
        fastest_success_lap_time_sec=(
            float(min(lap_times)) if lap_times else float("inf")
        ),
        eligible=bool(eligible),
    )


def select_fastest_safe_candidate(
    summaries: list[CandidateSummary],
) -> CandidateSummary | None:
    """Select only among perfect-completion candidates, then minimize lap time."""
    eligible = [summary for summary in summaries if summary.eligible]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda summary: (
            summary.mean_success_lap_time_sec,
            summary.fastest_success_lap_time_sec,
            summary.name,
        ),
    )


def diagnostic_rank_key(summary: CandidateSummary) -> tuple:
    """Rank unsafe candidates for training diagnostics, never for deployment."""
    return (
        summary.completed_laps,
        -summary.lane_departures,
        -summary.other_failures,
        -summary.mean_success_lap_time_sec,
    )
