from __future__ import annotations

import hashlib
from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contracts import NormalizedTrack


class WindowingPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    window_duration_s: float = Field(gt=0.0)
    stride_s: float = Field(gt=0.0)
    nominal_sample_interval_s: float = Field(default=0.1, gt=0.0)
    gap_multiplier: float = Field(default=2.5, gt=1.0)
    minimum_coverage_fraction: float = Field(default=0.8, gt=0.0, le=1.0)
    minimum_samples: int = Field(default=2, ge=2)
####


class TrackWindow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    window_id: str = Field(min_length=1)
    split_group_id: str = Field(min_length=1)
    segment_index: int = Field(ge=0)
    start_index: int = Field(ge=0)
    end_index_exclusive: int = Field(ge=2)
    start_time_s: float
    end_time_s: float
    requested_duration_s: float = Field(gt=0.0)
    sample_count: int = Field(ge=2)
    expected_sample_count: int = Field(ge=2)
    coverage_fraction: float = Field(gt=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if self.end_index_exclusive <= self.start_index:
            raise ValueError("window end index must follow the start index")
        if self.sample_count != self.end_index_exclusive - self.start_index:
            raise ValueError("sample_count must match the index span")
        if self.end_time_s < self.start_time_s:
            raise ValueError("window end time must not precede the start time")
        return self
    ####
####


class TrackWindowingSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    segment_count: int = Field(ge=0)
    candidate_window_count: int = Field(ge=0)
    accepted_window_count: int = Field(ge=0)
    rejected_low_coverage_count: int = Field(ge=0)
    rejected_short_segment_count: int = Field(ge=0)
####


class TrackWindowingResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    split_group_id: str = Field(min_length=1)
    windows: tuple[TrackWindow, ...]
    summary: TrackWindowingSummary

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        if any(window.split_group_id != self.split_group_id for window in self.windows):
            raise ValueError("all windows must belong to the declared split group")
        if len(self.windows) != self.summary.accepted_window_count:
            raise ValueError("accepted window count must match the windows tuple")
        return self
    ####
####


def _segment_bounds(track: NormalizedTrack, policy: WindowingPolicy) -> tuple[tuple[int, int], ...]:
    timestamps = track.timestamps_s
    gap_threshold_s = policy.nominal_sample_interval_s * policy.gap_multiplier
    split_points = np.flatnonzero(np.diff(timestamps) > gap_threshold_s) + 1
    bounds: list[tuple[int, int]] = []
    start = 0
    for split_point in split_points:
        end = int(split_point)
        bounds.append((start, end))
        start = end
    bounds.append((start, int(timestamps.shape[0])))
    return tuple(bounds)
####


def _stable_window_id(
    track: NormalizedTrack,
    *,
    segment_index: int,
    start_index: int,
    end_index_exclusive: int,
    policy: WindowingPolicy,
) -> str:
    payload = "|".join(
        (
            track.provenance.dataset_id,
            track.provenance.split_group_id,
            str(segment_index),
            str(start_index),
            str(end_index_exclusive),
            f"{policy.window_duration_s:.12g}",
            f"{policy.stride_s:.12g}",
            f"{policy.nominal_sample_interval_s:.12g}",
        )
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"{track.provenance.dataset_id}:window:{digest}"
####


def window_track(
    track: NormalizedTrack,
    *,
    policy: WindowingPolicy,
) -> TrackWindowingResult:
    timestamps = track.timestamps_s
    expected_sample_count = max(
        policy.minimum_samples,
        int(round(policy.window_duration_s / policy.nominal_sample_interval_s)),
    )
    segments = _segment_bounds(track, policy)

    windows: list[TrackWindow] = []
    candidate_window_count = 0
    rejected_low_coverage_count = 0
    rejected_short_segment_count = 0

    for segment_index, (segment_start, segment_end) in enumerate(segments):
        segment_sample_count = segment_end - segment_start
        if segment_sample_count < policy.minimum_samples:
            rejected_short_segment_count += 1
            continue

        segment_start_time = float(timestamps[segment_start])
        segment_end_exclusive_time = (
            float(timestamps[segment_end - 1]) + policy.nominal_sample_interval_s
        )
        candidate_index = 0

        while True:
            requested_start_time = segment_start_time + candidate_index * policy.stride_s
            requested_end_time = requested_start_time + policy.window_duration_s
            if requested_end_time > segment_end_exclusive_time + 1e-9:
                break

            start_index = max(
                segment_start,
                int(np.searchsorted(timestamps, requested_start_time, side="left")),
            )
            end_index_exclusive = min(
                segment_end,
                int(np.searchsorted(timestamps, requested_end_time, side="left")),
            )
            sample_count = end_index_exclusive - start_index
            candidate_window_count += 1
            coverage_fraction = min(1.0, sample_count / expected_sample_count)

            if (
                sample_count < policy.minimum_samples
                or coverage_fraction < policy.minimum_coverage_fraction
            ):
                rejected_low_coverage_count += 1
                candidate_index += 1
                continue

            windows.append(
                TrackWindow(
                    window_id=_stable_window_id(
                        track,
                        segment_index=segment_index,
                        start_index=start_index,
                        end_index_exclusive=end_index_exclusive,
                        policy=policy,
                    ),
                    split_group_id=track.provenance.split_group_id,
                    segment_index=segment_index,
                    start_index=start_index,
                    end_index_exclusive=end_index_exclusive,
                    start_time_s=float(timestamps[start_index]),
                    end_time_s=float(timestamps[end_index_exclusive - 1]),
                    requested_duration_s=policy.window_duration_s,
                    sample_count=sample_count,
                    expected_sample_count=expected_sample_count,
                    coverage_fraction=coverage_fraction,
                )
            )
            candidate_index += 1

    summary = TrackWindowingSummary(
        segment_count=len(segments),
        candidate_window_count=candidate_window_count,
        accepted_window_count=len(windows),
        rejected_low_coverage_count=rejected_low_coverage_count,
        rejected_short_segment_count=rejected_short_segment_count,
    )
    return TrackWindowingResult(
        split_group_id=track.provenance.split_group_id,
        windows=tuple(windows),
        summary=summary,
    )
####


def window_tracks(
    tracks: tuple[NormalizedTrack, ...],
    *,
    policy: WindowingPolicy,
) -> tuple[TrackWindow, ...]:
    windows: list[TrackWindow] = []
    for track in tracks:
        windows.extend(window_track(track, policy=policy).windows)
    return tuple(windows)
####


__all__ = [
    "TrackWindow",
    "TrackWindowingResult",
    "TrackWindowingSummary",
    "WindowingPolicy",
    "window_track",
    "window_tracks",
]
