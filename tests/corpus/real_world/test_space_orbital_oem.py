from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from kinematic_classifier_sandbox.corpus.real_world.adapters.space_orbital_oem import (
    NASA_ISS_FIXTURE_URI,
    NASA_ISS_ORIGINAL_SOURCE_ASSET_ID,
    NASA_ISS_ORIGINAL_SOURCE_URL,
    NASA_ISS_SOURCE_ASSET_ID,
    NASA_ISS_SOURCE_SHA256,
    NasaIssOemCorpusAdapter,
    canonicalize_cospar_id,
)
from kinematic_classifier_sandbox.corpus.real_world.adapters.space_orbital_oem_parsing import (
    parse_oem_file,
    parse_oem_text,
    records_to_si_arrays as oem_records_to_si_arrays,
)
from kinematic_classifier_sandbox.corpus.real_world.contracts import LabelEvidence
from kinematic_classifier_sandbox.corpus.real_world.corpus_contracts import (
    CoordinateFrameKind,
    PhysicalDomain,
    TimeBasis,
)


FIXTURE = Path(__file__).with_name("fixtures") / "nasa_iss_oem_20220427_excerpt.kvn"
EVIDENCE = (
    Path(__file__).parents[3] / "docs" / "methods" / "space_orbital" / "evidence.yaml"
)


def test_evidence_yaml_preserves_roles_and_release_boundaries() -> None:
    evidence = yaml.safe_load(EVIDENCE.read_text(encoding="utf-8"))
    assert isinstance(evidence, dict)

    nasa = evidence["nasa_iss_oem_source_card"]
    assert nasa["research_evidence_state"] == "fixture_validated"
    assert nasa["repository_release_state"] == "bounded_fixture_validation_only"
    assert nasa["observation"]["state_roles_present"] == ["estimate"]
    assert nasa["artifacts"]["committed_fixture_uri"] == NASA_ISS_FIXTURE_URI
    assert nasa["artifacts"]["fixture_sha256"] == NASA_ISS_SOURCE_SHA256
    assert nasa["artifacts"]["original_source_url"] == NASA_ISS_ORIGINAL_SOURCE_URL
    assert nasa["study_eligibility"]["classifier_eligible"] is False

    igs = evidence["igs_final_sp3_source_card"]
    assert igs["research_evidence_state"] == "fixture_validated_restricted"
    assert igs["repository_release_state"] == "parser_only_no_source_fixture"
    assert igs["observation"]["state_roles_present"] == ["reference_solution"]

    matrix = evidence["independent_validation_matrix"]
    assert matrix["independence_checks"]["disposition"] == (
        "independent_validation_eligible"
    )
    assert matrix["non_independent_comparators"][0]["physical_object_id"] == "1998-067A"
####


def test_parse_nasa_iss_oem_source_semantics() -> None:
    extract = parse_oem_file(FIXTURE)

    assert extract.header.version == "2.0"
    assert extract.header.originator == "NASA/JSC/FOD/TOPO"
    assert extract.metadata.object_name == "ISS"
    assert extract.metadata.source_object_id == "1998-067-A"
    assert extract.metadata.center_name == "Earth"
    assert extract.metadata.reference_frame == "EME2000"
    assert extract.metadata.time_system == "UTC"
    assert len(extract.records) == 13

    timestamps_s, position_m, source_velocity_mps = oem_records_to_si_arrays(extract.records)
    assert timestamps_s[0] == pytest.approx(1_651_060_800.0)
    assert np.diff(timestamps_s) == pytest.approx(np.full(12, 240.0))
    assert position_m[0] == pytest.approx(
        (-4_954_469.912645210, -4_603_679.573832890, -648_822.018902234)
    )
    assert source_velocity_mps[0] == pytest.approx(
        (2_845.24123202163, -3_888.05749279550, 5_960.01055905332)
    )
####


def test_adapter_builds_validation_only_product4_space_trajectory() -> None:
    adapter = NasaIssOemCorpusAdapter()
    corpus = adapter.load_corpus(FIXTURE)

    assert len(corpus) == 1
    item = corpus[0]
    track = item.trajectory
    assets = {
        asset.asset_id: asset
        for asset in item.dataset.dataset_manifest.source_assets
    }

    assert item.dataset.domains == (PhysicalDomain.SPACE,)
    assert item.dataset.time_basis is TimeBasis.UNIX_UTC_SECONDS
    assert item.dataset.canonical_frame.kind is CoordinateFrameKind.ECI
    assert item.dataset.canonical_frame.frame_id == "EME2000"
    assert item.dataset.dataset_manifest.license_id == "US-PD"

    bounded_asset = assets[NASA_ISS_SOURCE_ASSET_ID]
    assert bounded_asset.download_url == NASA_ISS_FIXTURE_URI
    assert bounded_asset.sha256 == NASA_ISS_SOURCE_SHA256
    assert bounded_asset.required is True

    original_asset = assets[NASA_ISS_ORIGINAL_SOURCE_ASSET_ID]
    assert original_asset.download_url == NASA_ISS_ORIGINAL_SOURCE_URL
    assert original_asset.sha256 is None
    assert original_asset.required is False

    assert item.metadata.subject_id == "1998-067A"
    assert item.metadata.domain is PhysicalDomain.SPACE
    assert item.metadata.time_basis is TimeBasis.UNIX_UTC_SECONDS
    assert item.metadata.source_metadata["source_fixture_uri"] == NASA_ISS_FIXTURE_URI
    assert track.provenance.split_group_id == "physical_object:1998-067A"
    assert track.labels.normalized_class == "persistent_orbit"
    assert track.labels.evidence is LabelEvidence.DERIVED
    assert track.labels.is_proxy is False
    assert track.source_velocity_mps is not None
    assert not np.allclose(track.source_velocity_mps, track.derived_velocity_mps)
    assert track.source_acceleration_mps2 is None
    assert track.numeric_channels == ()
    assert track.categorical_channels == ()
    assert track.metadata["identity_access_class"] == "grouping_only"
    assert track.metadata["classifier_eligible"] is False
    assert track.metadata["validation_fixture_only"] is True
    assert track.quality is not None
    assert track.quality.sample_count == 13
    assert track.quality.gap_count == 0
    assert track.quality.source_velocity_rmse_mps is not None
    assert np.isfinite(track.quality.source_velocity_rmse_mps)

    serialized = item.model_dump(mode="json")
    assert serialized["metadata"]["frame"]["frame_id"] == "EME2000"
    assert serialized["trajectory"]["provenance"]["split_group_id"] == (
        "physical_object:1998-067A"
    )
####


def test_adapter_rejects_mutated_fixture_hash(tmp_path: Path) -> None:
    mutated = tmp_path / "mutated.kvn"
    mutated.write_text(
        FIXTURE.read_text(encoding="ascii").replace("OBJECT_NAME = ISS", "OBJECT_NAME = XSS"),
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        NasaIssOemCorpusAdapter().load_corpus(mutated)
####


def test_oem_parser_rejects_duplicate_epoch() -> None:
    source = FIXTURE.read_text(encoding="ascii")
    duplicated = source.replace(
        "2022-04-27T12:04:00.000",
        "2022-04-27T12:00:00.000",
        1,
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        parse_oem_text(duplicated)
####


def test_oem_parser_rejects_non_utc_time_system() -> None:
    source = FIXTURE.read_text(encoding="ascii").replace(
        "TIME_SYSTEM = UTC",
        "TIME_SYSTEM = TAI",
        1,
    )

    with pytest.raises(ValueError, match="requires TIME_SYSTEM = UTC"):
        parse_oem_text(source)
####


def test_oem_parser_rejects_multiple_metadata_segments() -> None:
    source = FIXTURE.read_text(encoding="ascii").replace(
        "META_STOP",
        "META_STOP\nMETA_START\nOBJECT_NAME = OTHER\nMETA_STOP",
        1,
    )

    with pytest.raises(ValueError, match="exactly one OEM metadata segment"):
        parse_oem_text(source)
####


def test_oem_parser_rejects_state_outside_useable_interval() -> None:
    source = FIXTURE.read_text(encoding="ascii").replace(
        "2022-04-27T12:00:00.000 -4954",
        "2022-04-27T11:56:00.000 -4954",
        1,
    )

    with pytest.raises(ValueError, match="outside the declared useable interval"):
        parse_oem_text(source)
####


def test_oem_parser_rejects_invalid_metadata_interval() -> None:
    source = FIXTURE.read_text(encoding="ascii").replace(
        "USEABLE_STOP_TIME = 2022-05-12T12:00:00.000",
        "USEABLE_STOP_TIME = 2022-04-27T11:00:00.000",
        1,
    )

    with pytest.raises(ValueError, match="START <= USEABLE_START"):
        parse_oem_text(source)
####


def test_canonicalize_cospar_id() -> None:
    assert canonicalize_cospar_id("1998-067-A") == "1998-067A"
    assert canonicalize_cospar_id("1998-067A") == "1998-067A"
####
