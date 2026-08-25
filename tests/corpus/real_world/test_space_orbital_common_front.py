from __future__ import annotations

import json
from pathlib import Path

from kinematic_classifier_sandbox.corpus.real_world.episode_contracts import (
    StateViewKind,
)
from kinematic_classifier_sandbox.corpus.real_world.space_orbital.common_front import (
    build_nasa_iss_oem_episode,
)

FIXTURE = Path(__file__).parent / "fixtures" / "nasa_iss_oem_20220427_excerpt.kvn"


def test_nasa_iss_common_front_preserves_eci_source_velocity(tmp_path: Path) -> None:
    episode = build_nasa_iss_oem_episode(
        FIXTURE,
        output_root=tmp_path,
        corpus_snapshot_id="space-orb-validation",
    )

    assert episode.corpus_sublane == "space_orbital"
    assert episode.quality_summary.sample_count == 13
    assert episode.classifier_trajectory_view is None
    assert episode.state_views[0].frame.crs_or_datum == "EME2000"
    analysis_view = next(
        view for view in episode.state_views if view.view_kind is StateViewKind.ANALYSIS
    )
    payload = json.loads((tmp_path / analysis_view.sample_asset.path).read_text())
    assert len(payload["elapsed_s"]) == 13
    assert payload["elapsed_s"][1] - payload["elapsed_s"][0] == 240.0
    assert payload["source_velocity_mps"] != payload["derived_velocity_mps"]
