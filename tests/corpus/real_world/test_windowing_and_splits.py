from __future__ import annotations

from kinematic_classifier_sandbox.corpus.real_world.splits import (
    DatasetPartition,
    GroupSplitPolicy,
    assign_grouped_split,
)
from kinematic_classifier_sandbox.corpus.real_world.windowing import (
    WindowingPolicy,
    window_track,
    window_tracks,
)

from ._helpers import make_real_world_track


def test_windowing_breaks_at_real_time_gaps_and_is_stable() -> None:
    track = make_real_world_track(
        split_group_id="car-gap",
        normalized_class="passenger_car",
        duration_s=40.0,
        gap_after_s=20.0,
        gap_duration_s=1.0,
    )
    policy = WindowingPolicy(
        window_duration_s=10.0,
        stride_s=5.0,
        nominal_sample_interval_s=0.1,
        gap_multiplier=2.5,
    )

    first = window_track(track, policy=policy)
    second = window_track(track, policy=policy)

    assert first.summary.segment_count == 2
    assert first.windows
    assert tuple(window.window_id for window in first.windows) == tuple(
        window.window_id for window in second.windows
    )

    for window in first.windows:
        timestamp_slice = track.timestamps_s[
            window.start_index : window.end_index_exclusive
        ]
        time_steps = timestamp_slice[1:] - timestamp_slice[:-1]
        assert all(time_step <= 0.25 + 1e-12 for time_step in time_steps)
####


def test_windowing_reports_segments_shorter_than_requested_duration() -> None:
    track = make_real_world_track(
        split_group_id="car-short",
        normalized_class="passenger_car",
        duration_s=5.0,
    )
    result = window_track(
        track,
        policy=WindowingPolicy(window_duration_s=10.0, stride_s=5.0),
    )

    assert result.windows == ()
    assert result.summary.candidate_window_count == 0
    assert result.summary.rejected_short_segment_count == 1
####


def test_grouped_split_is_deterministic_balanced_and_window_safe() -> None:
    tracks = tuple(
        make_real_world_track(
            split_group_id=f"car-{index}",
            normalized_class="passenger_car",
        )
        for index in range(8)
    ) + tuple(
        make_real_world_track(
            split_group_id=f"truck-{index}",
            normalized_class="truck",
        )
        for index in range(8)
    )
    windows = window_tracks(
        tracks,
        policy=WindowingPolicy(
            window_duration_s=20.0,
            stride_s=10.0,
        ),
    )
    policy = GroupSplitPolicy(seed="unit-test-road-split")

    first = assign_grouped_split(tracks, windows=windows, policy=policy)
    second = assign_grouped_split(tuple(reversed(tracks)), windows=windows, policy=policy)

    first_map = {
        assignment.split_group_id: assignment.partition
        for assignment in first.assignments
    }
    second_map = {
        assignment.split_group_id: assignment.partition
        for assignment in second.assignments
    }
    assert first_map == second_map

    for window in windows:
        assert first.partition_for(window.split_group_id) is first_map[window.split_group_id]

    for normalized_class in ("passenger_car", "truck"):
        class_partitions = {
            assignment.partition
            for assignment in first.assignments
            if assignment.normalized_class == normalized_class
        }
        assert class_partitions == {
            DatasetPartition.TRAIN,
            DatasetPartition.VALIDATION,
            DatasetPartition.TEST,
        }

    summary = {
        (row.partition, row.normalized_class): row
        for row in first.summary_rows
    }
    assert summary[(DatasetPartition.TRAIN, "passenger_car")].group_count == 6
    assert summary[(DatasetPartition.VALIDATION, "passenger_car")].group_count == 1
    assert summary[(DatasetPartition.TEST, "passenger_car")].group_count == 1
####
