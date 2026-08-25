from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...common_experiment.contracts import ExecutablePairSpec
from .contracts import NormalizedTrack
from .projection import ProjectionKind, ProjectionResult, project_pair_windows
from .splits import (
    DatasetPartition,
    DatasetSplit,
    GroupSplitPolicy,
    SplitRatios,
    assign_grouped_split,
)
from .windowing import (
    TrackWindow,
    TrackWindowingResult,
    WindowingPolicy,
    window_track,
)


class RoadVehicleStudyConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    class_a: str = Field(default="passenger_car", min_length=1)
    class_b: str = Field(default="truck", min_length=1)
    pair_id: str = Field(default="passenger_car_vs_truck", min_length=1)
    expected_difficulty: str = Field(default="unknown", min_length=1)
    window_durations_s: tuple[float, ...] = (10.0, 20.0, 30.0)
    stride_fraction: float = Field(default=0.5, gt=0.0, le=1.0)
    nominal_sample_interval_s: float = Field(default=0.1, gt=0.0)
    gap_multiplier: float = Field(default=2.5, gt=1.0)
    minimum_coverage_fraction: float = Field(default=0.8, gt=0.0, le=1.0)
    split_seed: str = Field(default="tgsim-foggy-bottom-road-v1", min_length=1)
    split_ratios: SplitRatios = Field(default_factory=SplitRatios)

    @model_validator(mode="after")
    def validate_classes_and_durations(self) -> Self:
        if self.class_a == self.class_b:
            raise ValueError("road-vehicle study classes must be distinct")
        if not self.window_durations_s:
            raise ValueError("at least one window duration is required")
        if any(duration <= 0.0 for duration in self.window_durations_s):
            raise ValueError("window durations must be positive")
        if len(self.window_durations_s) != len(set(self.window_durations_s)):
            raise ValueError("window durations must be unique")
        return self
    ####
####


class RoadVehiclePartitionSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    partition: DatasetPartition
    normalized_class: str = Field(min_length=1)
    track_count: int = Field(ge=0)
    source_duration_s: float = Field(ge=0.0)
    candidate_window_count: int = Field(ge=0)
    accepted_window_count: int = Field(ge=0)
    accepted_window_duration_s: float = Field(ge=0.0)
    rejected_low_coverage_count: int = Field(ge=0)
    rejected_short_segment_count: int = Field(ge=0)
####


@dataclass(frozen=True, slots=True)
class RoadVehicleDurationStudy:
    window_duration_s: float
    windows: tuple[TrackWindow, ...]
    split: DatasetSplit
    partition_summary: tuple[RoadVehiclePartitionSummary, ...]
    speed_profile: tuple[ProjectionResult, ...]
    cumulative_path_length: tuple[ProjectionResult, ...]
####


@dataclass(frozen=True, slots=True)
class RoadVehicleStudyResult:
    pair_spec: ExecutablePairSpec
    tracks: tuple[NormalizedTrack, ...]
    duration_studies: tuple[RoadVehicleDurationStudy, ...]
####


def _filter_pair_tracks(
    tracks: tuple[NormalizedTrack, ...],
    *,
    class_a: str,
    class_b: str,
) -> tuple[NormalizedTrack, ...]:
    selected = tuple(
        track
        for track in tracks
        if track.labels.normalized_class in {class_a, class_b}
    )
    classes_present = {track.labels.normalized_class for track in selected}
    missing = {class_a, class_b} - classes_present
    if missing:
        raise ValueError(
            "road-vehicle study is missing class track(s): "
            + ", ".join(sorted(missing))
        )
    return selected
####


def _partition_summary(
    tracks: tuple[NormalizedTrack, ...],
    windowing_results: tuple[TrackWindowingResult, ...],
    *,
    split: DatasetSplit,
    window_duration_s: float,
) -> tuple[RoadVehiclePartitionSummary, ...]:
    track_by_group = {
        track.provenance.split_group_id: track
        for track in tracks
    }
    result_by_group = {
        result.split_group_id: result
        for result in windowing_results
    }
    classes = sorted({track.labels.normalized_class for track in tracks})

    rows: list[RoadVehiclePartitionSummary] = []
    for partition in DatasetPartition:
        partition_groups = {
            assignment.split_group_id
            for assignment in split.assignments
            if assignment.partition is partition
        }
        for normalized_class in classes:
            matching_groups = tuple(
                split_group_id
                for split_group_id in sorted(partition_groups)
                if track_by_group[split_group_id].labels.normalized_class
                == normalized_class
            )
            source_duration_s = sum(
                float(
                    track_by_group[split_group_id].timestamps_s[-1]
                    - track_by_group[split_group_id].timestamps_s[0]
                )
                for split_group_id in matching_groups
            )
            matching_results = tuple(
                result_by_group[split_group_id]
                for split_group_id in matching_groups
            )
            accepted_window_count = sum(
                result.summary.accepted_window_count
                for result in matching_results
            )
            rows.append(
                RoadVehiclePartitionSummary(
                    partition=partition,
                    normalized_class=normalized_class,
                    track_count=len(matching_groups),
                    source_duration_s=source_duration_s,
                    candidate_window_count=sum(
                        result.summary.candidate_window_count
                        for result in matching_results
                    ),
                    accepted_window_count=accepted_window_count,
                    accepted_window_duration_s=(
                        accepted_window_count * window_duration_s
                    ),
                    rejected_low_coverage_count=sum(
                        result.summary.rejected_low_coverage_count
                        for result in matching_results
                    ),
                    rejected_short_segment_count=sum(
                        result.summary.rejected_short_segment_count
                        for result in matching_results
                    ),
                )
            )
    return tuple(rows)
####


def build_road_vehicle_study(
    tracks: tuple[NormalizedTrack, ...],
    *,
    config: RoadVehicleStudyConfig | None = None,
) -> RoadVehicleStudyResult:
    effective_config = config or RoadVehicleStudyConfig()
    selected_tracks = _filter_pair_tracks(
        tracks,
        class_a=effective_config.class_a,
        class_b=effective_config.class_b,
    )
    pair_spec = ExecutablePairSpec(
        pair_id=effective_config.pair_id,
        class_a=effective_config.class_a,
        class_b=effective_config.class_b,
        expected_difficulty=effective_config.expected_difficulty,
    )
    split_policy = GroupSplitPolicy(
        seed=effective_config.split_seed,
        ratios=effective_config.split_ratios,
        stratify_by_class=True,
    )

    duration_studies: list[RoadVehicleDurationStudy] = []
    expected_assignments: tuple[tuple[str, str], ...] | None = None
    for window_duration_s in effective_config.window_durations_s:
        window_policy = WindowingPolicy(
            window_duration_s=window_duration_s,
            stride_s=window_duration_s * effective_config.stride_fraction,
            nominal_sample_interval_s=effective_config.nominal_sample_interval_s,
            gap_multiplier=effective_config.gap_multiplier,
            minimum_coverage_fraction=effective_config.minimum_coverage_fraction,
        )
        windowing_results = tuple(
            window_track(track, policy=window_policy)
            for track in selected_tracks
        )
        windows = tuple(
            window
            for result in windowing_results
            for window in result.windows
        )
        split = assign_grouped_split(
            selected_tracks,
            windows=windows,
            policy=split_policy,
        )
        assignment_signature = tuple(
            (assignment.split_group_id, assignment.partition.value)
            for assignment in split.assignments
        )
        if expected_assignments is None:
            expected_assignments = assignment_signature
        elif assignment_signature != expected_assignments:
            raise RuntimeError("group split assignment changed across window durations")

        speed_profile = project_pair_windows(
            selected_tracks,
            windows,
            pair_spec=pair_spec,
            projection_kind=ProjectionKind.SPEED_PROFILE,
            split=split,
        )
        cumulative_path_length = project_pair_windows(
            selected_tracks,
            windows,
            pair_spec=pair_spec,
            projection_kind=ProjectionKind.CUMULATIVE_PATH_LENGTH,
            split=split,
        )
        duration_studies.append(
            RoadVehicleDurationStudy(
                window_duration_s=window_duration_s,
                windows=windows,
                split=split,
                partition_summary=_partition_summary(
                    selected_tracks,
                    windowing_results,
                    split=split,
                    window_duration_s=window_duration_s,
                ),
                speed_profile=speed_profile,
                cumulative_path_length=cumulative_path_length,
            )
        )

    return RoadVehicleStudyResult(
        pair_spec=pair_spec,
        tracks=selected_tracks,
        duration_studies=tuple(duration_studies),
    )
####


__all__ = [
    "RoadVehicleDurationStudy",
    "RoadVehiclePartitionSummary",
    "RoadVehicleStudyConfig",
    "RoadVehicleStudyResult",
    "build_road_vehicle_study",
]
