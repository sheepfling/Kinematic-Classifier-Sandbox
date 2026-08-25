from __future__ import annotations

from kinematic_classifier_sandbox.corpus.real_world.portfolio import (
    EpisodeSplitAssignment,
    SnapshotSplit,
)
from kinematic_classifier_sandbox.corpus.real_world.portfolio_matrix import (
    evaluate_product4_lane_matrix,
)

from .test_portfolio import _episode, _prepared_registry, _snapshot


def test_lane_matrix_scopes_gate_results_without_masking_lane_readiness(tmp_path) -> None:
    episodes = tuple(
        _episode(
            episode_id=f"{lane}-{split.value}",
            lane=lane,
            platform_group=f"{lane}-{split.value}",
            classifier=lane == "land_surface",
        )
        for lane in ("land_surface", "air_atmospheric")
        for split in SnapshotSplit
    )
    snapshot = _snapshot(tmp_path, episodes)
    split_cycle = tuple(SnapshotSplit)
    assignments = tuple(
        EpisodeSplitAssignment(episode_id=episode.episode_id, split=split)
        for episode, split in zip(
            episodes,
            tuple(split_cycle[index % len(split_cycle)] for index in range(len(episodes))),
            strict=True,
        )
    )

    report = evaluate_product4_lane_matrix(
        _prepared_registry(),
        snapshot=snapshot,
        episodes=episodes,
        assignments=assignments,
        expected_lanes=("land_surface", "air_atmospheric"),
    )

    by_lane = {lane_report.lane: lane_report for lane_report in report.lane_reports}
    assert report.all_lanes_pass is False
    assert by_lane["land_surface"].gate.passes is True
    assert by_lane["land_surface"].episode_count == 3
    assert by_lane["land_surface"].classifier_ready_episode_count == 3
    assert by_lane["air_atmospheric"].gate.passes is False
    assert by_lane["air_atmospheric"].gate.classifier_projection_passes is False
    assert by_lane["air_atmospheric"].episode_count == 3


def test_lane_matrix_rejects_empty_grouping_policy(tmp_path) -> None:
    episodes = tuple(
        _episode(
            episode_id=f"land-{split.value}",
            platform_group=f"land-{split.value}",
        )
        for split in SnapshotSplit
    )
    snapshot = _snapshot(tmp_path, episodes)
    assignments = tuple(
        EpisodeSplitAssignment(episode_id=episode.episode_id, split=split)
        for episode, split in zip(episodes, SnapshotSplit, strict=True)
    )

    try:
        evaluate_product4_lane_matrix(
            _prepared_registry(),
            snapshot=snapshot,
            episodes=episodes,
            assignments=assignments,
            expected_lanes=("land_surface",),
            split_grouping_namespaces=(),
        )
    except ValueError as error:
        assert "grouping policy" in str(error)
    else:
        raise AssertionError("empty grouping policy should be rejected")
