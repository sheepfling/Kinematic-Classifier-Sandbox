from __future__ import annotations

import json
from pathlib import Path

from kinematic_classifier_sandbox.corpus.real_world.episode_contracts import (
    StateViewKind,
)
from kinematic_classifier_sandbox.corpus.real_world.sea_subsurface.common_front import (
    build_ioos_anchor_episode,
)

FIXTURE = (
    Path(__file__).parents[3]
    / "docs/research/product4/sea_subsurface/fixtures/ioos_uaf_unit_191_profile_1709942882.csv"
)


def test_ioos_common_front_preserves_asynchronous_channel_events(tmp_path: Path) -> None:
    episode = build_ioos_anchor_episode(
        FIXTURE,
        output_root=tmp_path,
        corpus_snapshot_id="sea-sub-validation",
    )

    assert episode.corpus_sublane == "sea_subsurface"
    assert episode.quality_summary.sample_count == 99
    assert episode.quality_summary.duplicate_timestamp_count == 5
    assert episode.classifier_trajectory_view is None
    assert {
        finding.code for finding in episode.quality_summary.findings
    } >= {
        "SEA_SUB_ASYNCHRONOUS_CHANNEL_EVENTS",
        "SEA_SUB_GPS_PHASE_NOT_PROVEN",
    }
    analysis_view = next(
        view for view in episode.state_views if view.view_kind is StateViewKind.ANALYSIS
    )
    payload = json.loads((tmp_path / analysis_view.sample_asset.path).read_text())
    elapsed = [sample["elapsed_s"] for sample in payload["samples"]]
    assert sum(left == right for left, right in zip(elapsed, elapsed[1:])) == 5
    assert any(sample["onboard_gps_valid"][0] for sample in payload["samples"])
