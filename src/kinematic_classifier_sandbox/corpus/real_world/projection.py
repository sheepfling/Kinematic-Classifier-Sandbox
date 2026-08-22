from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from ...common_experiment.contracts import ExecutablePairSpec, ExecutableTrajectory
from .contracts import NormalizedTrack
from .splits import DatasetPartition, DatasetSplit
from .windowing import TrackWindow


class ProjectionKind(StrEnum):
    SPEED_PROFILE = "speed_profile"
    CUMULATIVE_PATH_LENGTH = "cumulative_path_length"
####


@dataclass(frozen=True, slots=True)
class ProjectedTrajectoryMetadata:
    trajectory_id: str
    window_id: str
    split_group_id: str
    dataset_id: str
    recording_id: str
    run_id: str
    track_id: str
    source_asset_id: str
    native_label: str
    normalized_class: str
    partition: DatasetPartition | None
    projection_kind: ProjectionKind
    source_start_time_s: float
    source_end_time_s: float
####


@dataclass(frozen=True, slots=True)
class ProjectionResult:
    trajectory: ExecutableTrajectory
    metadata: ProjectedTrajectoryMetadata
####


def _validate_window(track: NormalizedTrack, window: TrackWindow) -> None:
    if window.split_group_id != track.provenance.split_group_id:
        raise ValueError("window and track split groups do not match")
    if window.end_index_exclusive > track.timestamps_s.shape[0]:
        raise ValueError("window extends beyond the source track")
    if window.start_index < 0:
        raise ValueError("window start index must be nonnegative")
####


def _slice_window(
    track: NormalizedTrack,
    window: TrackWindow,
) -> tuple[np.ndarray, np.ndarray]:
    _validate_window(track, window)
    sample_slice = slice(window.start_index, window.end_index_exclusive)
    timestamps = np.asarray(track.timestamps_s[sample_slice], dtype=np.float64)
    position = np.asarray(track.position_m[sample_slice], dtype=np.float64)
    relative_time = timestamps - timestamps[0]
    return relative_time, position
####


def _first_derivative(values: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    if values.shape[0] != timestamps.shape[0]:
        raise ValueError("values and timestamps must have matching lengths")
    if values.shape[0] < 2:
        raise ValueError("at least two samples are required for a derivative")
    return np.asarray(np.gradient(values, timestamps, edge_order=1), dtype=np.float64)
####


def _cumulative_path_length(position_m: np.ndarray) -> np.ndarray:
    steps = np.linalg.norm(np.diff(position_m, axis=0), axis=1)
    return np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(steps)))
####


def _speed_projection(
    track: NormalizedTrack,
    window: TrackWindow,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    sample_slice = slice(window.start_index, window.end_index_exclusive)
    times = np.asarray(track.timestamps_s[sample_slice], dtype=np.float64)
    times = times - times[0]
    velocity = np.asarray(track.derived_velocity_mps[sample_slice], dtype=np.float64)
    speed = np.linalg.norm(velocity[:, :2], axis=1)
    speed_rate = _first_derivative(speed, times)
    speed_acceleration = _first_derivative(speed_rate, times)
    return speed, speed, speed_rate, speed_acceleration, "scalar_speed_mps"
####


def _path_length_projection(
    track: NormalizedTrack,
    window: TrackWindow,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    times, position = _slice_window(track, window)
    path_length = _cumulative_path_length(position)
    path_speed = _first_derivative(path_length, times)
    path_acceleration = _first_derivative(path_speed, times)
    return (
        path_length,
        path_length,
        path_speed,
        path_acceleration,
        "scalar_cumulative_path_m",
    )
####


def project_window(
    track: NormalizedTrack,
    window: TrackWindow,
    *,
    pair_spec: ExecutablePairSpec,
    projection_kind: ProjectionKind,
    partition: DatasetPartition | None = None,
) -> ProjectionResult:
    _validate_window(track, window)
    true_class = track.labels.normalized_class
    if true_class not in {pair_spec.class_a, pair_spec.class_b}:
        raise ValueError(
            f"track class {true_class!r} is not part of pair {pair_spec.pair_id!r}"
        )

    if projection_kind is ProjectionKind.SPEED_PROFILE:
        measurement, truth, velocity, acceleration, coordinate_frame = _speed_projection(
            track,
            window,
        )
    elif projection_kind is ProjectionKind.CUMULATIVE_PATH_LENGTH:
        (
            measurement,
            truth,
            velocity,
            acceleration,
            coordinate_frame,
        ) = _path_length_projection(track, window)
    else:
        raise ValueError(f"unsupported projection kind: {projection_kind}")

    times = np.asarray(
        track.timestamps_s[window.start_index : window.end_index_exclusive],
        dtype=np.float64,
    )
    times = times - times[0]
    trajectory_id = f"{window.window_id}:{projection_kind.value}"
    trajectory = ExecutableTrajectory(
        trajectory_id=trajectory_id,
        class_pair_id=pair_spec.pair_id,
        class_a=pair_spec.class_a,
        class_b=pair_spec.class_b,
        true_class=true_class,
        scenario_id=f"real_world:{track.provenance.dataset_id}:{projection_kind.value}",
        seed=0,
        times=tuple(float(value) for value in times),
        measurements=tuple(float(value) for value in measurement),
        true_position=tuple(float(value) for value in truth),
        true_velocity=tuple(float(value) for value in velocity),
        true_acceleration=tuple(float(value) for value in acceleration),
        measurement_dim=1,
        coordinate_frame=coordinate_frame,
    )
    metadata = ProjectedTrajectoryMetadata(
        trajectory_id=trajectory_id,
        window_id=window.window_id,
        split_group_id=track.provenance.split_group_id,
        dataset_id=track.provenance.dataset_id,
        recording_id=track.provenance.recording_id,
        run_id=track.provenance.run_id,
        track_id=track.provenance.track_id,
        source_asset_id=track.provenance.source_asset_id,
        native_label=track.labels.native_label,
        normalized_class=true_class,
        partition=partition,
        projection_kind=projection_kind,
        source_start_time_s=window.start_time_s,
        source_end_time_s=window.end_time_s,
    )
    return ProjectionResult(trajectory=trajectory, metadata=metadata)
####


def project_pair_windows(
    tracks: tuple[NormalizedTrack, ...],
    windows: tuple[TrackWindow, ...],
    *,
    pair_spec: ExecutablePairSpec,
    projection_kind: ProjectionKind,
    split: DatasetSplit | None = None,
) -> tuple[ProjectionResult, ...]:
    tracks_by_group = {
        track.provenance.split_group_id: track
        for track in tracks
        if track.labels.normalized_class in {pair_spec.class_a, pair_spec.class_b}
    }
    results: list[ProjectionResult] = []
    for window in windows:
        track = tracks_by_group.get(window.split_group_id)
        if track is None:
            continue
        partition = None
        if split is not None:
            partition = split.partition_for(window.split_group_id)
        results.append(
            project_window(
                track,
                window,
                pair_spec=pair_spec,
                projection_kind=projection_kind,
                partition=partition,
            )
        )
    return tuple(results)
####


__all__ = [
    "ProjectedTrajectoryMetadata",
    "ProjectionKind",
    "ProjectionResult",
    "project_pair_windows",
    "project_window",
]
