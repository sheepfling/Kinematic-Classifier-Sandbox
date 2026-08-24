from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kinematic_classifier_sandbox.corpus.real_world.adapters.cmre_route_tracklets import (
    build_fixture,
    parse_tracklets,
    write_fixture_index,
)
from kinematic_classifier_sandbox.corpus.real_world.episode_contracts import (
    GroupingNamespace,
    LabelEvidenceKind,
    StateViewKind,
    TrajectoryEpisodeManifest,
)


def _tracklet_row(
    *,
    tracklet_id: int,
    mmsi: int,
    route: str,
    timestamps: tuple[int, int, int, int, int],
    heading: float = 90.0,
) -> str:
    fields: list[str] = [str(tracklet_id)]
    for index, timestamp in enumerate(timestamps, start=1):
        fields.extend(
            (
                str(tracklet_id * 100 + index),
                str(mmsi),
                f"{8.0 + index / 10.0:.1f}",
                "90.0",
                f"{heading:.1f}",
                f"{-4.70 + index * 0.001:.6f}",
                f"{48.30 + index * 0.0001:.6f}",
                str(timestamp),
            )
        )
    fields.append(route)
    return "|".join(fields)
####


def _write_sources(root: Path) -> tuple[Path, Path]:
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
    tracklets = root / "tracklets.csv"
    tracklets.write_text(
        "\n".join(
            (
                "|".join(header),
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
        corpus_snapshot_id="synthetic-sea-surface-fixture",
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
                "reported_velocity_xy_mps",
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
