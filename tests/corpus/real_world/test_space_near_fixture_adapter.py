from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from kinematic_classifier_sandbox.corpus.real_world.space_near.common_front import (
    build_fixture_episode_manifest,
)
from kinematic_classifier_sandbox.corpus.real_world.space_near.fixture_adapter import (
    load_space_near_fixture_definitions,
    load_space_near_fixture_portfolio,
    validate_embedded_fixture,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "space_near"


def test_repository_fixture_gzip_members_are_intact_and_parseable() -> None:
    fixture_paths = tuple(sorted(FIXTURE_PATH.glob("*.json.gz")))

    assert len(fixture_paths) == 6
    for fixture_path in fixture_paths:
        with gzip.open(fixture_path, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        assert payload["portfolio_version"] == "space-near-repository-fixtures-v0.1"
        assert len(payload["fixtures"]) == 1
    ####
####


def test_space_near_portfolio_loads_six_real_missions() -> None:
    trajectories = load_space_near_fixture_portfolio(FIXTURE_PATH)

    assert len(trajectories) == 6
    assert sum(track.trajectory.timestamps_s.shape[0] for track in trajectories) == 206
    assert {track.metadata.domain.value for track in trajectories} == {"space"}
    assert {track.metadata.domain_extensions["corpus_sublane"] for track in trajectories} == {
        "space_near"
    }
    assert len({track.trajectory.provenance.split_group_id for track in trajectories}) == 6
    assert {track.metadata.source_metadata["provider"] for track in trajectories} == {
        "ISAS/JAXA DARTS",
        "NASA GSFC Space Physics Data Facility / CDAWeb",
    }
####


def test_space_near_tracks_keep_identity_out_of_numeric_channels() -> None:
    trajectories = load_space_near_fixture_portfolio(FIXTURE_PATH)
    forbidden = {
        "mission_id",
        "object_id",
        "provider",
        "source_dataset_id",
        "source_asset_id",
    }

    for corpus_trajectory in trajectories:
        track = corpus_trajectory.trajectory
        assert not forbidden.intersection(channel.name for channel in track.numeric_channels)
        assert not forbidden.intersection(channel.name for channel in track.categorical_channels)
        assert track.source_velocity_mps is None
        assert track.source_acceleration_mps2 is None
        assert track.quality is not None
        assert np.all(np.isfinite(track.position_m))
        assert np.all(np.diff(track.timestamps_s) > 0.0)
    ####
####


def test_endurance_remains_reference_solution_without_native_gps_velocity() -> None:
    trajectories = load_space_near_fixture_portfolio(FIXTURE_PATH)
    endurance = next(
        trajectory
        for trajectory in trajectories
        if trajectory.metadata.platform_subtype == "Endurance 47.001"
    )

    assert endurance.trajectory.metadata["state_role"] == "reference_solution"
    assert endurance.trajectory.source_velocity_mps is None
    assert endurance.trajectory.labels.evidence.value == "derived"
    assert endurance.metadata.source_metadata["source_type"] == (
        "postflight_reference_solution"
    )
####


def test_each_fixture_passes_integrity_and_semantic_validation() -> None:
    fixtures = load_space_near_fixture_definitions(FIXTURE_PATH)
    validations = tuple(validate_embedded_fixture(fixture) for fixture in fixtures)

    assert len(validations) == 6
    assert sum(validation.sample_count for validation in validations) == 206
    assert sum(validation.state_view_count for validation in validations) == 12
    assert sum(validation.label_assertion_count for validation in validations) == 16
####


def test_fixture_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    source_path = next(FIXTURE_PATH.glob("*.json.gz"))
    with gzip.open(source_path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    payload["fixtures"][0]["episode"]["analysis_view"]["rows"][0][2] += 1.0
    damaged_path = tmp_path / "portfolio.json"
    damaged_path.write_text(json.dumps(payload), encoding="utf-8")
    fixture = load_space_near_fixture_definitions(damaged_path)[0]

    with pytest.raises(ValueError, match="hash mismatch"):
        validate_embedded_fixture(fixture)


def test_space_near_common_front_preserves_validation_boundary(tmp_path: Path) -> None:
    fixtures = load_space_near_fixture_definitions(FIXTURE_PATH)
    manifests = tuple(
        build_fixture_episode_manifest(
            fixture,
            output_root=tmp_path,
            corpus_snapshot_id="space-near-validation-v0.1",
            source_artifact_id=f"fixture:{fixture.source.source_dataset_id}",
        )
        for fixture in fixtures
    )

    assert len(manifests) == 6
    assert all(manifest.classifier_trajectory_view is None for manifest in manifests)
    assert {manifest.corpus_sublane for manifest in manifests} == {"space_near"}
    assert sum(manifest.quality_summary.sample_count for manifest in manifests) == 206
    assert sum(len(manifest.labels) for manifest in manifests) == 16
    assert all(
        manifest.domain_extension is not None
        and manifest.domain_extension.payload["classifier_view_status"]
        == "intentionally_blocked"
        for manifest in manifests
    )
    for manifest in manifests:
        for view in manifest.state_views:
            assert (tmp_path / view.sample_asset.path).is_file()
####
