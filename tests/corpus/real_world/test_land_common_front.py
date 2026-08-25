from __future__ import annotations

import json
from pathlib import Path

from kinematic_classifier_sandbox.corpus.real_world.episode_contracts import (
    StateViewKind,
)
from kinematic_classifier_sandbox.corpus.real_world.land.common_front import (
    build_tgsim_fixture_episodes,
)

FIXTURE = Path(__file__).parent / "fixtures" / "tgsim_foggy_bottom_minimal.csv"


def test_tgsim_common_front_keeps_land_fixture_validation_only(tmp_path: Path) -> None:
    episodes = build_tgsim_fixture_episodes(
        FIXTURE,
        output_root=tmp_path,
        corpus_snapshot_id="land-validation",
    )

    assert len(episodes) == 3
    assert {episode.corpus_sublane for episode in episodes} == {"land_surface"}
    assert all(episode.classifier_trajectory_view is None for episode in episodes)
    assert all(
        episode.domain_extension is not None
        and episode.domain_extension.payload["fixture_status"] == "validation_only"
        for episode in episodes
    )
    assert len({episode.platform_group_id for episode in episodes}) == 3
    for episode in episodes:
        assert next(
            view for view in episode.state_views if view.view_kind is StateViewKind.SOURCE_NATIVE
        ).sample_count == episode.quality_summary.sample_count
        analysis_path = tmp_path / next(
            view for view in episode.state_views if view.view_kind is StateViewKind.ANALYSIS
        ).sample_asset.path
        payload = json.loads(analysis_path.read_text(encoding="utf-8"))
        assert len(payload["timestamps_s"]) == episode.quality_summary.sample_count
