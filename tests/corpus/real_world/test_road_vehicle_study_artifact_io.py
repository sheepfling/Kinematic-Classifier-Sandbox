from __future__ import annotations

import csv
import json

from kinematic_classifier_sandbox.corpus.real_world.road_vehicle_study import (
    RoadVehicleStudyConfig,
    build_road_vehicle_study,
)
from kinematic_classifier_sandbox.corpus.real_world.road_vehicle_study_artifact_io import (
    write_road_vehicle_study_artifacts,
)

from ._helpers import make_real_world_track


def test_road_vehicle_study_artifacts_are_machine_readable(tmp_path) -> None:
    tracks = tuple(
        make_real_world_track(
            split_group_id=f"car-{index}",
            normalized_class="passenger_car",
            duration_s=20.0,
        )
        for index in range(3)
    ) + tuple(
        make_real_world_track(
            split_group_id=f"truck-{index}",
            normalized_class="truck",
            duration_s=20.0,
        )
        for index in range(3)
    )
    result = build_road_vehicle_study(
        tracks,
        config=RoadVehicleStudyConfig(
            window_durations_s=(10.0,),
            split_seed="artifact-test",
        ),
    )
    artifacts = write_road_vehicle_study_artifacts(result, tmp_path)

    assert artifacts.manifest_path.exists()
    assert artifacts.tracks_path.exists()
    assert artifacts.report_path.exists()
    assert len(artifacts.duration_dirs) == 1

    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["pair_id"] == "passenger_car_vs_truck"
    assert manifest["track_count"] == 6
    assert manifest["projection_kinds"] == [
        "speed_profile",
        "cumulative_path_length",
    ]

    duration_dir = artifacts.duration_dirs[0]
    for filename in (
        "split_assignments.csv",
        "partition_summary.csv",
        "windows.csv",
        "projection_metadata.csv",
    ):
        assert (duration_dir / filename).exists()

    with (duration_dir / "windows.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert {row["partition"] for row in rows} <= {
        "train",
        "validation",
        "test",
    }

    report = artifacts.report_path.read_text(encoding="utf-8")
    assert "does not claim classifier performance" in report
####
