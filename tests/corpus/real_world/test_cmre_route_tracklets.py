from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kinematic_classifier_sandbox.corpus.real_world.adapters.cmre_route_tracklets import (
    build_fixture,
    parse_tracklets,
    protected_identity_group_id,
    sha256_file,
    write_fixture_index,
)
from kinematic_classifier_sandbox.corpus.real_world.episode_contracts import (
    GroupingNamespace,
    LabelEvidenceKind,
    StateViewKind,
    TrajectoryEpisodeManifest,
)


_TEST_IDENTITY_KEY = b"unit-test-identity-key-material-32-bytes"


def _tracklet_row(
    *,
    tracklet_id: int,
    mmsi: int,
    route: str,
    timestamps: tuple[int, int, int, int, int],
    heading: float = 90.0,
    course: float = 90.0,
    speed: float | None = None,
) -> str:
    fields: list[str] = [str(tracklet_id)]
    for index, timestamp in enumerate(timestamps, start=1):
        speed_value = speed if speed is not None else 8.0 + index / 10.0
        fields.extend(
            (
                str(tracklet_id * 100 + index),
                str(mmsi),
                f"{speed_value:.1f}",
                f"{course:.1f}",
                f"{heading:.1f}",
                f"{-4.70 + index * 0.001:.6f}",
                f"{48.30 + index * 0.0001:.6f}",
                str(timestamp),
            )
        )
    fields.append(route)
    return "|".join(fields)
####


def _header() -> list[str]:
    header = ["idtracklet"]
    for index in range(1, 6):
        header.extend(
            (
                f"id{index}",
                f"mmsi{index}",
                f"speed{index}",
                f"course{index}",
                f"heading{index}",
                f"lon{index}",
                f"lat{index}",
                f"ts{index}",
            )
        )
    header.append("route")
    return header
####


def _write_sources(root: Path) -> tuple[Path, Path]:
    tracklets = root / "tracklets.csv"
    tracklets.write_text(
        "\n".join(
            (
                "|".join(_header()),
                _tracklet_row(
                    tracklet_id=1,
                    mmsi=111_111_111,
                    route="R_TEST_A",
                    timestamps=(100, 110, 120, 130, 140),
                ),
                _tracklet_row(
                    tracklet_id=2,
                    mmsi=111_111_111,
                    route="R_TEST_B",
                    timestamps=(200, 210, 210, 230, 240),
                    heading=511.0,
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    nomenclature = root / "nomenclature.csv"
    nomenclature.write_text(
        "route|originport|destinationport|length\n"
        "R_TEST_A|PORT_A|PORT_B|1000\n"
        "R_TEST_B|PORT_B|PORT_C|2000\n",
        encoding="utf-8",
    )
    return tracklets, nomenclature
####


def _build(root: Path):
    tracklets, nomenclature = _write_sources(root)
    output = root / "prepared"
    result = build_fixture(
        tracklets_path=tracklets,
        nomenclature_path=nomenclature,
        output_root=output,
        source_artifact_id="synthetic-contract-fixture",
        nomenclature_artifact_id="synthetic-route-nomenclature",
        corpus_snapshot_id="synthetic-sea-surface-fixture",
        identity_key=_TEST_IDENTITY_KEY,
    )
    return result, output
####


def test_adapter_builds_valid_manifests_and_repeated_platform_group(tmp_path: Path) -> None:
    result, output = _build(tmp_path)
    assert len(result.manifests) == 2
    assert result.physical_platform_group_count == 1
    assert len(result.repeated_physical_platform_groups) == 1
    for manifest in result.manifests:
        manifest_path = output / "episodes" / f"{manifest.episode_id}.json"
        reloaded = TrajectoryEpisodeManifest.model_validate_json(manifest_path.read_text())
        assert reloaded == manifest
        assert manifest.corpus_sublane == "sea_surface"
        assert manifest.source_artifact_ids == (
            "synthetic-contract-fixture",
            "synthetic-route-nomenclature",
        )
    ####
####


def test_vertical_is_nan_and_invalid_not_measured_zero(tmp_path: Path) -> None:
    result, output = _build(tmp_path)
    for manifest in result.manifests:
        analysis = next(
            view for view in manifest.state_views if view.view_kind is StateViewKind.ANALYSIS
        )
        with np.load(output / analysis.sample_asset.path, allow_pickle=False) as arrays:
            assert np.all(np.isnan(arrays["position_enu_m"][:, 2]))
            assert np.all(~arrays["position_valid"][:, 2])
            assert np.all(np.isnan(arrays["reported_velocity_enu_mps"][:, 2]))
            assert np.all(~arrays["velocity_valid"][:, 2])
        ####
    ####
####


def test_source_duplicate_is_preserved_and_classifier_time_is_strict(tmp_path: Path) -> None:
    result, output = _build(tmp_path)
    manifest = result.manifests[1]
    source = next(
        view for view in manifest.state_views if view.view_kind is StateViewKind.SOURCE_NATIVE
    )
    classifier = manifest.classifier_trajectory_view
    assert classifier is not None
    with np.load(output / source.sample_asset.path, allow_pickle=False) as arrays:
        assert np.any(np.diff(arrays["elapsed_s"]) == 0.0)
    ####
    with np.load(output / classifier.asset.path, allow_pickle=False) as arrays:
        assert np.all(np.diff(arrays["elapsed_s"]) > 0.0)
        assert len(arrays["elapsed_s"]) == 4
    ####
    assert "duplicate_timestamp" in {
        finding.code for finding in manifest.quality_summary.findings
    }
####


def test_non_adjacent_duplicate_is_counted_and_removed_from_classifier(
    tmp_path: Path,
) -> None:
    tracklets = tmp_path / "tracklets_non_adjacent_duplicate.csv"
    tracklets.write_text(
        "|".join(_header())
        + "\n"
        + _tracklet_row(
            tracklet_id=5,
            mmsi=555_555_555,
            route="R_TEST_A",
            timestamps=(100, 110, 100, 120, 130),
        )
        + "\n",
        encoding="utf-8",
    )
    nomenclature = tmp_path / "nomenclature_non_adjacent_duplicate.csv"
    nomenclature.write_text(
        "route|originport|destinationport|length\nR_TEST_A|PORT_A|PORT_B|1000\n",
        encoding="utf-8",
    )
    output = tmp_path / "prepared_non_adjacent_duplicate"
    result = build_fixture(
        tracklets_path=tracklets,
        nomenclature_path=nomenclature,
        output_root=output,
        source_artifact_id="non-adjacent-duplicate-contract-fixture",
        corpus_snapshot_id="non-adjacent-duplicate-sea-surface-fixture",
        identity_key=_TEST_IDENTITY_KEY,
    )
    manifest = result.manifests[0]
    assert manifest.quality_summary.duplicate_timestamp_count == 1
    duplicate_finding = next(
        finding
        for finding in manifest.quality_summary.findings
        if finding.code == "duplicate_timestamp"
    )
    assert duplicate_finding.value == 1
    classifier = manifest.classifier_trajectory_view
    assert classifier is not None
    assert classifier.sample_count == 4
    with np.load(output / classifier.asset.path, allow_pickle=False) as arrays:
        assert arrays["elapsed_s"].tolist() == [0.0, 10.0, 20.0, 30.0]
    ####
####


def test_classifier_asset_excludes_identity_route_and_target_labels(tmp_path: Path) -> None:
    result, output = _build(tmp_path)
    for manifest in result.manifests:
        assert "111111111" not in manifest.model_dump_json()
        assert manifest.labels[0].namespace == "route"
        assert manifest.labels[0].evidence_kind is LabelEvidenceKind.NATIVE
        classifier = manifest.classifier_trajectory_view
        assert classifier is not None
        with np.load(output / classifier.asset.path, allow_pickle=False) as arrays:
            assert set(arrays.files) == {
                "elapsed_s",
                "position_xy_m",
                "position_valid_xy",
                "reported_velocity_xy_mps",
                "reported_velocity_valid_xy",
            }
        ####
    ####
####


def test_grouping_key_links_repeated_platform_without_raw_mmsi(tmp_path: Path) -> None:
    result, _ = _build(tmp_path)
    values = []
    for manifest in result.manifests:
        key = next(
            item
            for item in manifest.grouping_keys
            if item.namespace is GroupingNamespace.PHYSICAL_PLATFORM
        )
        values.append(key.opaque_value)
        assert key.opaque_value == manifest.platform_group_id
    ####
    assert len(set(values)) == 1
####


def test_platform_group_is_keyed_and_changes_with_identity_key() -> None:
    first = protected_identity_group_id(
        dataset_id="dataset",
        namespace="physical_platform",
        raw_value="111111111",
        identity_key=b"first-key-material-for-tests",
    )
    second = protected_identity_group_id(
        dataset_id="dataset",
        namespace="physical_platform",
        raw_value="111111111",
        identity_key=b"second-key-material-for-tests",
    )
    assert first != second
    assert "111111111" not in first
    with pytest.raises(ValueError, match="at least 16 bytes"):
        protected_identity_group_id(
            dataset_id="dataset",
            namespace="physical_platform",
            raw_value="111111111",
            identity_key=b"short",
        )
    ####
####


def test_heading_sentinel_is_retained_and_marked_invalid(tmp_path: Path) -> None:
    result, output = _build(tmp_path)
    manifest = result.manifests[1]
    source = next(
        view for view in manifest.state_views if view.view_kind is StateViewKind.SOURCE_NATIVE
    )
    with np.load(output / source.sample_asset.path, allow_pickle=False) as arrays:
        assert np.all(arrays["heading_deg"] == 511.0)
        assert np.all(~arrays["heading_valid"])
    ####
    assert "invalid_heading" in {
        finding.code for finding in manifest.quality_summary.findings
    }
####


def test_invalid_reported_motion_is_masked_in_classifier_asset(tmp_path: Path) -> None:
    tracklets = tmp_path / "invalid_motion.csv"
    tracklets.write_text(
        "|".join(_header())
        + "\n"
        + _tracklet_row(
            tracklet_id=4,
            mmsi=444_444_444,
            route="R_TEST_A",
            timestamps=(100, 110, 120, 130, 140),
            course=360.0,
        )
        + "\n",
        encoding="utf-8",
    )
    nomenclature = tmp_path / "nomenclature_invalid_motion.csv"
    nomenclature.write_text(
        "route|originport|destinationport|length\nR_TEST_A|PORT_A|PORT_B|1000\n",
        encoding="utf-8",
    )
    output = tmp_path / "prepared_invalid_motion"
    result = build_fixture(
        tracklets_path=tracklets,
        nomenclature_path=nomenclature,
        output_root=output,
        source_artifact_id="invalid-motion-contract-fixture",
        corpus_snapshot_id="invalid-motion-sea-surface-fixture",
        identity_key=_TEST_IDENTITY_KEY,
    )
    manifest = result.manifests[0]
    classifier = manifest.classifier_trajectory_view
    assert classifier is not None
    with np.load(output / classifier.asset.path, allow_pickle=False) as arrays:
        assert np.all(np.isnan(arrays["reported_velocity_xy_mps"]))
        assert np.all(~arrays["reported_velocity_valid_xy"])
    ####
    assert "invalid_reported_motion" in {
        finding.code for finding in manifest.quality_summary.findings
    }
####


def test_nomenclature_artifact_is_identified_and_hashed(tmp_path: Path) -> None:
    tracklets, nomenclature = _write_sources(tmp_path)
    source_sha = sha256_file(tracklets)
    first_nomenclature_sha = sha256_file(nomenclature)
    first_result = build_fixture(
        tracklets_path=tracklets,
        nomenclature_path=nomenclature,
        output_root=tmp_path / "prepared_nomenclature_first",
        source_artifact_id="source-tracklets",
        nomenclature_artifact_id="route-nomenclature",
        corpus_snapshot_id="nomenclature-provenance-first",
        identity_key=_TEST_IDENTITY_KEY,
    )
    first_manifest = first_result.manifests[0]
    assert first_manifest.source_artifact_ids == (
        "source-tracklets",
        "route-nomenclature",
    )
    assert first_manifest.domain_extension is not None
    first_artifacts = first_manifest.domain_extension.payload["source_artifacts"]
    assert first_artifacts == {
        "tracklets": {
            "artifact_id": "source-tracklets",
            "sha256": source_sha,
        },
        "route_nomenclature": {
            "artifact_id": "route-nomenclature",
            "sha256": first_nomenclature_sha,
        },
    }

    nomenclature.write_text(
        nomenclature.read_text(encoding="utf-8").replace("|1000\n", "|1001\n"),
        encoding="utf-8",
    )
    second_nomenclature_sha = sha256_file(nomenclature)
    second_result = build_fixture(
        tracklets_path=tracklets,
        nomenclature_path=nomenclature,
        output_root=tmp_path / "prepared_nomenclature_second",
        source_artifact_id="source-tracklets",
        nomenclature_artifact_id="route-nomenclature",
        corpus_snapshot_id="nomenclature-provenance-second",
        identity_key=_TEST_IDENTITY_KEY,
    )
    second_manifest = second_result.manifests[0]
    assert second_manifest.domain_extension is not None
    second_artifacts = second_manifest.domain_extension.payload["source_artifacts"]
    assert second_artifacts["tracklets"]["sha256"] == source_sha
    assert second_artifacts["route_nomenclature"]["sha256"] == second_nomenclature_sha
    assert second_nomenclature_sha != first_nomenclature_sha
####


def test_fixture_index_is_hashed_and_machine_readable(tmp_path: Path) -> None:
    result, output = _build(tmp_path)
    path = write_fixture_index(output_root=output, result=result)
    payload = json.loads(path.read_text())
    assert payload["episode_count"] == 2
    assert payload["physical_platform_group_count"] == 1
    assert all(len(item["manifest_sha256"]) == 64 for item in payload["episodes"])
####


def test_parser_rejects_mixed_platform_identity(tmp_path: Path) -> None:
    tracklets, _ = _write_sources(tmp_path)
    text = tracklets.read_text()
    text = text.replace("111111111|8.3", "222222222|8.3", 1)
    tracklets.write_text(text)
    with pytest.raises(ValueError, match="mixes MMSI"):
        parse_tracklets(tracklets)
    ####
####


def test_parser_rejects_duplicate_tracklet_id(tmp_path: Path) -> None:
    tracklets, _ = _write_sources(tmp_path)
    rows = tracklets.read_text().splitlines()
    tracklets.write_text("\n".join((rows[0], rows[1], rows[1])) + "\n")
    with pytest.raises(ValueError, match="duplicate tracklet ID"):
        parse_tracklets(tracklets)
    ####
####


def test_parser_rejects_missing_selected_tracklet_id(tmp_path: Path) -> None:
    tracklets, _ = _write_sources(tmp_path)
    with pytest.raises(ValueError, match="were not found"):
        parse_tracklets(tracklets, selected_tracklet_ids={1, 99})
    ####
####


def test_classifier_projection_sorts_and_deduplicates_source_time(tmp_path: Path) -> None:
    tracklets = tmp_path / "tracklets_out_of_order.csv"
    tracklets.write_text(
        "|".join(_header())
        + "\n"
        + _tracklet_row(
            tracklet_id=3,
            mmsi=333_333_333,
            route="R_TEST_A",
            timestamps=(100, 110, 105, 106, 120),
        )
        + "\n",
        encoding="utf-8",
    )
    nomenclature = tmp_path / "nomenclature_out_of_order.csv"
    nomenclature.write_text(
        "route|originport|destinationport|length\nR_TEST_A|PORT_A|PORT_B|1000\n",
        encoding="utf-8",
    )
    output = tmp_path / "prepared_out_of_order"
    result = build_fixture(
        tracklets_path=tracklets,
        nomenclature_path=nomenclature,
        output_root=output,
        source_artifact_id="out-of-order-contract-fixture",
        corpus_snapshot_id="out-of-order-sea-surface-fixture",
        identity_key=_TEST_IDENTITY_KEY,
    )
    manifest = result.manifests[0]
    classifier = manifest.classifier_trajectory_view
    assert classifier is not None
    with np.load(output / classifier.asset.path, allow_pickle=False) as arrays:
        assert np.all(np.diff(arrays["elapsed_s"]) > 0.0)
        assert arrays["elapsed_s"].tolist() == [0.0, 5.0, 6.0, 10.0, 20.0]
    ####
    assert "out_of_order_timestamp" in {
        finding.code for finding in manifest.quality_summary.findings
    }
    assert manifest.quality_summary.median_sample_interval_s == pytest.approx(4.5)
    assert manifest.quality_summary.maximum_gap_s == pytest.approx(10.0)
####
