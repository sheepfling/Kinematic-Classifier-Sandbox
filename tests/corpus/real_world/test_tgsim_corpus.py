from __future__ import annotations

from datetime import date
from pathlib import Path

from kinematic_classifier_sandbox.corpus.real_world.adapters.tgsim_contracts import (
    TgsimFoggyBottomAdapterConfig,
)
from kinematic_classifier_sandbox.corpus.real_world.adapters.tgsim_corpus import (
    TgsimFoggyBottomCorpusAdapter,
)
from kinematic_classifier_sandbox.corpus.real_world.corpus_contracts import (
    ObservationModality,
    PhysicalDomain,
    TimeBasis,
)


_FIXTURE = Path(__file__).parent / "fixtures" / "tgsim_foggy_bottom_minimal.csv"


def test_tgsim_exposes_domain_neutral_corpus_interface() -> None:
    adapter = TgsimFoggyBottomCorpusAdapter(
        config=TgsimFoggyBottomAdapterConfig(accessed_on=date(2026, 8, 22))
    )

    trajectories = adapter.load_corpus(_FIXTURE)

    assert trajectories
    assert adapter.corpus_metadata.domains == (PhysicalDomain.LAND,)
    assert adapter.corpus_metadata.time_basis is TimeBasis.RELATIVE_SECONDS
    assert adapter.corpus_metadata.observation_modalities == (
        ObservationModality.OPTICAL_TRACKING,
    )
    assert all(item.metadata.domain is PhysicalDomain.LAND for item in trajectories)
    assert all(
        item.metadata.trajectory_id == item.trajectory.provenance.split_group_id
        for item in trajectories
    )
####
