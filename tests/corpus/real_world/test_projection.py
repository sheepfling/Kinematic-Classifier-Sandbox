from __future__ import annotations

import pytest

from kinematic_classifier_sandbox.common_experiment.contracts import ExecutablePairSpec
from kinematic_classifier_sandbox.corpus.real_world.projection import (
    ProjectionKind,
    project_window,
)
from kinematic_classifier_sandbox.corpus.real_world.splits import DatasetPartition
from kinematic_classifier_sandbox.corpus.real_world.windowing import (
    WindowingPolicy,
    window_track,
)

from ._helpers import make_real_world_track


def test_speed_and_path_projections_preserve_pair_and_provenance() -> None:
    track = make_real_world_track(
        split_group_id="car-1",
        normalized_class="passenger_car",
        duration_s=20.0,
        speed_mps=3.0,
    )
    window = window_track(
        track,
        policy=WindowingPolicy(window_duration_s=10.0, stride_s=10.0),
    ).windows[0]
    pair_spec = ExecutablePairSpec(
        pair_id="passenger_car_vs_truck",
        class_a="passenger_car",
        class_b="truck",
        expected_difficulty="unknown",
    )

    speed = project_window(
        track,
        window,
        pair_spec=pair_spec,
        projection_kind=ProjectionKind.SPEED_PROFILE,
        partition=DatasetPartition.TRAIN,
    )
    path = project_window(
        track,
        window,
        pair_spec=pair_spec,
        projection_kind=ProjectionKind.CUMULATIVE_PATH_LENGTH,
        partition=DatasetPartition.TRAIN,
    )

    assert speed.trajectory.true_class == "passenger_car"
    assert speed.trajectory.measurement_dim == 1
    assert speed.trajectory.coordinate_frame == "scalar_speed_mps"
    assert all(abs(value - 3.0) < 1e-12 for value in speed.trajectory.measurements)
    assert speed.metadata.window_id == window.window_id
    assert speed.metadata.split_group_id == track.provenance.split_group_id
    assert speed.metadata.partition is DatasetPartition.TRAIN

    assert path.trajectory.coordinate_frame == "scalar_cumulative_path_m"
    assert path.trajectory.measurements[0] == 0.0
    assert path.trajectory.measurements[-1] > path.trajectory.measurements[0]
    assert path.metadata.projection_kind is ProjectionKind.CUMULATIVE_PATH_LENGTH
####


def test_speed_projection_uses_declared_three_dimensional_speed_axes() -> None:
    track = make_real_world_track(
        split_group_id="rocket-1",
        normalized_class="passenger_car",
        duration_s=20.0,
        speed_mps=3.0,
        vertical_speed_mps=4.0,
        speed_axis_count=3,
    )
    window = window_track(
        track,
        policy=WindowingPolicy(window_duration_s=10.0, stride_s=10.0),
    ).windows[0]
    pair_spec = ExecutablePairSpec(
        pair_id="passenger_car_vs_truck",
        class_a="passenger_car",
        class_b="truck",
        expected_difficulty="unknown",
    )

    speed = project_window(
        track,
        window,
        pair_spec=pair_spec,
        projection_kind=ProjectionKind.SPEED_PROFILE,
    )

    assert all(abs(value - 5.0) < 1e-12 for value in speed.trajectory.measurements)
####


def test_projection_rejects_track_outside_declared_pair() -> None:
    track = make_real_world_track(
        split_group_id="bus-1",
        normalized_class="bus",
        duration_s=20.0,
    )
    window = window_track(
        track,
        policy=WindowingPolicy(window_duration_s=10.0, stride_s=10.0),
    ).windows[0]
    pair_spec = ExecutablePairSpec(
        pair_id="passenger_car_vs_truck",
        class_a="passenger_car",
        class_b="truck",
        expected_difficulty="unknown",
    )

    with pytest.raises(ValueError, match="is not part of pair"):
        project_window(
            track,
            window,
            pair_spec=pair_spec,
            projection_kind=ProjectionKind.SPEED_PROFILE,
        )
####
