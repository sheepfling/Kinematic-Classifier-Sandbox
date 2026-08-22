from __future__ import annotations

from kinematic_classifier_sandbox.corpus.real_world.road_vehicle_study import (
    RoadVehicleStudyConfig,
    build_road_vehicle_study,
)

from ._helpers import make_real_world_track


def test_road_vehicle_study_builds_duration_sensitivity_and_two_projections() -> None:
    tracks = tuple(
        make_real_world_track(
            split_group_id=f"car-{index}",
            normalized_class="passenger_car",
            duration_s=40.0,
        )
        for index in range(8)
    ) + tuple(
        make_real_world_track(
            split_group_id=f"truck-{index}",
            normalized_class="truck",
            duration_s=40.0,
        )
        for index in range(8)
    )
    result = build_road_vehicle_study(
        tracks,
        config=RoadVehicleStudyConfig(
            window_durations_s=(10.0, 20.0, 30.0),
            split_seed="road-study-test",
        ),
    )

    assert result.pair_spec.pair_id == "passenger_car_vs_truck"
    assert tuple(study.window_duration_s for study in result.duration_studies) == (
        10.0,
        20.0,
        30.0,
    )

    assignment_signatures = []
    for study in result.duration_studies:
        assert study.windows
        assert len(study.speed_profile) == len(study.windows)
        assert len(study.cumulative_path_length) == len(study.windows)
        assert all(
            item.metadata.split_group_id == window.split_group_id
            for item, window in zip(study.speed_profile, study.windows)
        )
        summary_track_count = sum(row.track_count for row in study.partition_summary)
        assert summary_track_count == len(result.tracks)
        accepted_window_count = sum(
            row.accepted_window_count for row in study.partition_summary
        )
        assert accepted_window_count == len(study.windows)
        assignment_signatures.append(
            tuple(
                (assignment.split_group_id, assignment.partition)
                for assignment in study.split.assignments
            )
        )

    assert assignment_signatures[0] == assignment_signatures[1]
    assert assignment_signatures[1] == assignment_signatures[2]
####


def test_road_vehicle_study_requires_both_classes() -> None:
    tracks = (
        make_real_world_track(
            split_group_id="car-only",
            normalized_class="passenger_car",
        ),
    )

    try:
        build_road_vehicle_study(tracks)
    except ValueError as exc:
        assert "missing class track" in str(exc)
    else:
        raise AssertionError("expected a missing-class validation error")
####
