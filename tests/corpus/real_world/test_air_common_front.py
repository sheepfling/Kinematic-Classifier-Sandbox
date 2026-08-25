from __future__ import annotations

import json
from pathlib import Path

from kinematic_classifier_sandbox.corpus.real_world.air.common_front import (
    build_readsb_fixture_episodes,
)
from kinematic_classifier_sandbox.corpus.real_world.episode_contracts import (
    StateViewKind,
)

FIXTURE = Path(__file__).parent / "fixtures" / "readsb_documented_a320_trace.json"


def test_readsb_common_front_preserves_vertical_basis_and_blocks_classifier(tmp_path: Path) -> None:
    episodes = build_readsb_fixture_episodes(
        FIXTURE,
        output_root=tmp_path,
        corpus_snapshot_id="air-validation",
    )

    assert len(episodes) == 1
    episode = episodes[0]
    assert episode.corpus_sublane == "air_atmospheric"
    assert episode.classifier_trajectory_view is None
    assert episode.domain_extension is not None
    assert episode.domain_extension.payload["common_front_contract_validation"] == "passed"
    assert episode.domain_extension.payload["fixture_status"] == (
        "documented_parser_fixture_only"
    )
    assert episode.quality_summary.disposition == "usable_with_restrictions"
    analysis_view = next(
        view for view in episode.state_views if view.view_kind is StateViewKind.ANALYSIS
    )
    payload = json.loads((tmp_path / analysis_view.sample_asset.path).read_text())
    assert len(payload["samples"]) == 7
    assert payload["samples"][0]["altitude_basis"] == "barometric"
    assert payload["samples"][0]["vertical_rate_basis"] == "barometric"
    assert payload["normalization"]["vertical_missing_policy"] == "null_with_validity_false"
