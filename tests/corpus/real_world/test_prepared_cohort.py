from __future__ import annotations

import json

from kinematic_classifier_sandbox.corpus.real_world.episode_contracts import (
    GroupingNamespace,
)
from kinematic_classifier_sandbox.corpus.real_world.portfolio import audit_split_assignments
from kinematic_classifier_sandbox.corpus.real_world.prepared_cohort import (
    PreparedClassifierCohortConfig,
    build_prepared_classifier_cohort,
)
from kinematic_classifier_sandbox.corpus.real_world.projection import ProjectionKind
from kinematic_classifier_sandbox.corpus.real_world.splits import GroupSplitPolicy
from kinematic_classifier_sandbox.corpus.real_world.windowing import WindowingPolicy

from ._helpers import make_real_world_track


def _config() -> PreparedClassifierCohortConfig:
    return PreparedClassifierCohortConfig(
        corpus_snapshot_id="prepared-land-test-v1",
        dataset_id="tgsim_foggy_bottom",
        source_artifact_id="tgsim-balanced-csv",
        target_label_namespace="platform_class",
        pair_id="passenger_car_vs_truck",
        class_a="passenger_car",
        class_b="truck",
        projection_kind=ProjectionKind.SPEED_PROFILE,
        window_policy=WindowingPolicy(
            window_duration_s=10.0,
            stride_s=5.0,
            nominal_sample_interval_s=0.1,
        ),
        split_policy=GroupSplitPolicy(seed="prepared-cohort-test"),
    )


def test_prepared_cohort_keeps_labels_and_identity_out_of_classifier_assets(tmp_path) -> None:
    tracks = tuple(
        make_real_world_track(
            split_group_id=f"car-{index}",
            normalized_class="passenger_car",
            duration_s=20.0,
        )
        for index in range(4)
    ) + tuple(
        make_real_world_track(
            split_group_id=f"truck-{index}",
            normalized_class="truck",
            duration_s=20.0,
        )
        for index in range(4)
    )

    cohort = build_prepared_classifier_cohort(
        tracks,
        output_root=tmp_path,
        config=_config(),
    )

    assert cohort.episodes
    assert len(cohort.episodes) == len(cohort.projections)
    assert {assignment.split.value for assignment in cohort.episode_assignments()} == {
        "train",
        "validation",
        "test",
    }
    for episode in cohort.episodes:
        classifier = episode.classifier_trajectory_view
        assert classifier is not None
        assert classifier.target_labels_stored_outside_asset is True
        assert classifier.identity_and_grouping_values_excluded is True
        payload = json.loads((tmp_path / classifier.asset.path).read_text(encoding="utf-8"))
        assert set(payload) == {"measurements", "timestamps_s"}
        assert "passenger_car" not in json.dumps(payload)
        assert "tgsim_foggy_bottom" not in json.dumps(payload)


def test_task_grouping_policy_can_scope_physical_platform_holdout_explicitly(tmp_path) -> None:
    tracks = tuple(
        make_real_world_track(
            split_group_id=f"car-{index}",
            normalized_class="passenger_car",
            duration_s=20.0,
        )
        for index in range(4)
    ) + tuple(
        make_real_world_track(
            split_group_id=f"truck-{index}",
            normalized_class="truck",
            duration_s=20.0,
        )
        for index in range(4)
    )
    cohort = build_prepared_classifier_cohort(
        tracks,
        output_root=tmp_path,
        config=_config(),
    )
    assignments = cohort.episode_assignments()

    conservative = audit_split_assignments(cohort.episodes, assignments)
    assert conservative.passes is False
    physical_only = audit_split_assignments(
        cohort.episodes,
        assignments,
        grouping_namespaces=(GroupingNamespace.PHYSICAL_PLATFORM,),
    )
    assert physical_only.passes is True
