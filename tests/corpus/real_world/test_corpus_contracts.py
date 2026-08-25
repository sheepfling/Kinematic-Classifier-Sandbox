from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from kinematic_classifier_sandbox.corpus.real_world.contracts import DatasetManifest, SourceAsset
from kinematic_classifier_sandbox.corpus.real_world.corpus_contracts import (
    CoordinateFrameKind,
    CoordinateFrameMetadata,
    CorpusDatasetMetadata,
    CorpusTrajectory,
    CorpusTrajectoryMetadata,
    ObservationModality,
    PhysicalDomain,
    TimeBasis,
)

from ._helpers import make_real_world_track


_FRAME_ID = "foggy_bottom_local_metric_xy"


def _dataset(domain: PhysicalDomain) -> CorpusDatasetMetadata:
    manifest = DatasetManifest(
        dataset_id="tgsim_foggy_bottom",
        title="Fixture dataset",
        version="1",
        publisher="Fixture publisher",
        citation="Fixture citation",
        license_id="fixture-license",
        license_url="https://example.invalid/license",
        landing_page_url="https://example.invalid/dataset",
        accessed_on=date(2026, 8, 22),
        adapter_id="fixture_adapter",
        adapter_version="1.0.0",
        coordinate_frame=_FRAME_ID,
        nominal_sample_interval_s=0.1,
        source_assets=(
            SourceAsset(
                asset_id="trajectory_csv",
                title="Fixture asset",
                download_url="https://example.invalid/data",
                media_type="text/csv",
            ),
        ),
    )
    return CorpusDatasetMetadata(
        dataset_manifest=manifest,
        domains=(domain,),
        observation_modalities=(ObservationModality.TELEMETRY,),
        native_coordinate_frame="fixture-native",
        canonical_frame=CoordinateFrameMetadata(
            frame_id=_FRAME_ID,
            kind=CoordinateFrameKind.LOCAL_CARTESIAN,
            axes_description="fixture x/y/z metric Cartesian axes",
        ),
        time_basis=TimeBasis.RELATIVE_SECONDS,
        source_type="fixture",
    )
####


@pytest.mark.parametrize("domain", tuple(PhysicalDomain))
def test_common_contract_supports_all_major_domains(domain: PhysicalDomain) -> None:
    dataset = _dataset(domain)
    track = make_real_world_track(
        split_group_id=f"{domain.value}-fixture",
        normalized_class="fixture_platform",
    )
    metadata = CorpusTrajectoryMetadata(
        trajectory_id=f"trajectory-{domain.value}",
        domain=domain,
        time_basis=TimeBasis.RELATIVE_SECONDS,
        frame=dataset.canonical_frame,
        observation_modalities=(ObservationModality.TELEMETRY,),
        platform_type="fixture_platform",
        domain_extensions={"domain_specific_example": domain.value},
    )

    wrapped = CorpusTrajectory(dataset=dataset, metadata=metadata, trajectory=track)

    assert wrapped.metadata.domain is domain
    assert wrapped.model_dump(mode="json")["metadata"]["domain"] == domain.value
####


def test_common_contract_rejects_domain_mismatch() -> None:
    dataset = _dataset(PhysicalDomain.LAND)
    track = make_real_world_track(
        split_group_id="air-fixture",
        normalized_class="aircraft",
    )
    metadata = CorpusTrajectoryMetadata(
        trajectory_id="trajectory-air",
        domain=PhysicalDomain.AIR,
        time_basis=TimeBasis.RELATIVE_SECONDS,
        frame=dataset.canonical_frame,
        observation_modalities=(ObservationModality.TELEMETRY,),
    )

    with pytest.raises(ValidationError, match="trajectory domain"):
        CorpusTrajectory(dataset=dataset, metadata=metadata, trajectory=track)
####
