from __future__ import annotations

import csv
import hashlib
import json
import math
from itertools import pairwise
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
LANE_ROOT = REPO_ROOT / "docs/research/product4/sea_subsurface"


def _read_yaml(relative_path: str) -> dict[str, Any]:
    payload = yaml.safe_load((LANE_ROOT / relative_path).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


####


def _read_json(relative_path: str) -> dict[str, Any]:
    payload = json.loads((LANE_ROOT / relative_path).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


####


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


####


def _extract_query_url(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        candidate = line.strip()
        if candidate.startswith("https://"):
            return candidate
    raise AssertionError(f"no HTTPS query found in {path}")


####


def _state_view(fixture: dict[str, Any], view_id: str) -> dict[str, Any]:
    for view in fixture["state_views"]:
        if view["state_view_id"] == view_id:
            return view
    raise AssertionError(f"missing state view {view_id!r}")


####


def _parse_dbd_ascii(
    path: Path,
) -> tuple[dict[str, str], tuple[str, ...], tuple[str, ...], list[dict[str, float | None]]]:
    lines = [line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()]
    header_index = next(index for index, line in enumerate(lines) if line.startswith("c_battpos "))

    metadata: dict[str, str] = {}
    for line in lines[:header_index]:
        key, separator, value = line.partition(":")
        if separator:
            metadata[key.strip()] = value.strip()

    columns = tuple(lines[header_index].split())
    units = tuple(lines[header_index + 1].split())
    storage_widths = tuple(lines[header_index + 2].split())
    assert len(columns) == len(units) == len(storage_widths)

    rows: list[dict[str, float | None]] = []
    for raw_line in lines[header_index + 3 :]:
        tokens = raw_line.split()
        assert len(tokens) == len(columns)
        rows.append(
            {
                column: None if token == "NaN" else float(token)
                for column, token in zip(columns, tokens, strict=True)
            }
        )
    return metadata, columns, units, rows


####


def _ddmm_to_decimal(value: float) -> float:
    sign = -1.0 if value < 0.0 else 1.0
    absolute = abs(value)
    degrees = int(absolute // 100.0)
    minutes = absolute - degrees * 100.0
    return sign * (degrees + minutes / 60.0)


####


def test_all_committed_structured_files_parse() -> None:
    yaml_paths = sorted(LANE_ROOT.rglob("*.yaml"))
    json_paths = sorted(LANE_ROOT.rglob("*.json"))

    assert len(yaml_paths) == 11
    assert len(json_paths) == 2
    for path in yaml_paths:
        assert yaml.safe_load(path.read_text(encoding="utf-8")) is not None
    for path in json_paths:
        assert json.loads(path.read_text(encoding="utf-8")) is not None


####


def test_scorecards_have_valid_weighted_totals() -> None:
    cases = (
        ("scorecards/ioos_ngdac_uaf_unit_191-20240309T1200.yaml", 89),
        ("scorecards/whoi_ndsf_sentry_at26-09.yaml", 87),
    )

    for relative_path, expected_total in cases:
        scorecard = _read_yaml(relative_path)
        weighted_scores = scorecard["weighted_scores"]
        assert sum(item["weight"] for item in weighted_scores.values()) == 100
        assert sum(item["score"] for item in weighted_scores.values()) == expected_total
        assert scorecard["total_score_0_to_100"] == expected_total


####


def test_portfolio_keeps_anchor_and_holdout_at_access_verified() -> None:
    anchor = _read_yaml("source_cards/ioos_ngdac_uaf_unit_191-20240309T1200.yaml")
    holdout = _read_yaml("source_cards/whoi_ndsf_sentry_at26-09.yaml")
    status = _read_yaml("agent_status.yaml")

    assert anchor["portfolio_role"] == "anchor"
    assert anchor["access"]["evidence_state"] == "access_verified"
    assert anchor["access"]["artifact_acquired"] is False
    assert holdout["portfolio_role"] == "independent_validation"
    assert holdout["access"]["evidence_state"] == "access_verified"
    assert holdout["access"]["artifact_acquired"] is False
    assert status["gates"]["G1_source_portfolio"] == "complete"
    assert status["gates"]["G2_selected_anchor_fixture"] == "open"


####


def test_retained_artifact_hashes_and_sizes_match_manifests() -> None:
    ioos_fixture = _read_json("fixtures/ioos_glider_dac_murphy_profile_1.json")
    ooi_fixture = _read_json("fixtures/ooi_unit_364_mixed_provenance.json")

    cases = (
        (ioos_fixture["source_artifact"], "fixtures/ioos_glider_dac_murphy_profile_1.csv"),
        (ooi_fixture["source_artifact"], "fixtures/ooi_unit_364_2013_192_1_0.mrg"),
    )
    for manifest, relative_path in cases:
        path = LANE_ROOT / relative_path
        assert path.stat().st_size == manifest["byte_size"]
        assert _sha256(path) == manifest["sha256"]

    license_path = LANE_ROOT / ooi_fixture["source_artifact"]["license_path"]
    assert license_path.stat().st_size == ooi_fixture["source_artifact"]["license_byte_size"]
    assert _sha256(license_path) == ooi_fixture["source_artifact"]["license_sha256"]


####


def test_anchor_query_is_exact_and_bounded_to_one_profile() -> None:
    query_url = _extract_query_url(LANE_ROOT / "acquisition/ioos_profile_query.txt")
    parsed = urlparse(query_url)
    query = parse_qs(parsed.query)

    assert parsed.scheme == "https"
    assert parsed.netloc == "gliders.ioos.us"
    assert parsed.path.endswith("/unit_191-20240309T1200.csv")
    assert query["profile_id"] == ["1709942882"]

    requested_fields = set(parsed.query.split("&", maxsplit=1)[0].split(","))
    assert {"precise_time", "m_gps_lat", "m_gps_lon", "m_lat", "m_lon"}.issubset(requested_fields)
    assert {"m_depth", "pressure", "m_heading", "m_pitch", "m_roll"}.issubset(requested_fields)


####


def test_ioos_fixture_matches_retained_rows_and_preserves_roles() -> None:
    fixture = _read_json("fixtures/ioos_glider_dac_murphy_profile_1.json")
    csv_path = LANE_ROOT / "fixtures/ioos_glider_dac_murphy_profile_1.csv"
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    epochs = [float(row["time_epoch_s"]) for row in rows]
    assert len(rows) == 8
    assert all(later > earlier for earlier, later in pairwise(epochs))
    assert {row["trajectory"] for row in rows} == {"murphy-20150809T135508Z"}
    assert {row["profile_id"] for row in rows} == {"1"}

    horizontal = _state_view(fixture, "murphy-profile-1:provider-horizontal")
    pressure = _state_view(fixture, "murphy-profile-1:pressure-observation")
    depth = _state_view(fixture, "murphy-profile-1:derived-depth")

    assert horizontal["state_role"] == "estimate"
    assert "interpolated" in horizontal["value_basis"]
    assert pressure["state_role"] == "source_observation"
    assert pressure["value_basis"] == "measured"
    assert depth["state_role"] == "estimate"
    assert depth["value_basis"] == "calculated_from_pressure"
    assert depth["dependency_channels"] == ["sea_water_pressure"]
    assert depth["frame"] == {
        "vertical_reference": "sea_surface",
        "positive_direction": "down",
    }

    assert [sample["pressure_dbar"] for sample in pressure["samples"]] == [
        float(row["pressure_dbar"]) for row in rows
    ]
    assert [sample["depth_m"] for sample in depth["samples"]] == [
        float(row["depth_m"]) for row in rows
    ]


####


def test_ioos_qc_zero_is_not_reinterpreted_as_good_data() -> None:
    fixture = _read_json("fixtures/ioos_glider_dac_murphy_profile_1.json")
    findings = {finding["code"]: finding for finding in fixture["quality_findings"]}
    pressure = _state_view(fixture, "murphy-profile-1:pressure-observation")

    assert {sample["source_qc"] for sample in pressure["samples"]} == {0}
    assert "SEA_SUB_QC_ZERO_MEANS_NO_QC_PERFORMED" in findings
    assert (
        "must not be promoted to good_data"
        in findings["SEA_SUB_QC_ZERO_MEANS_NO_QC_PERFORMED"]["message"]
    )


####


def test_ooi_raw_parser_finds_expected_identity_and_shape() -> None:
    raw_path = LANE_ROOT / "fixtures/ooi_unit_364_2013_192_1_0.mrg"
    metadata, columns, units, rows = _parse_dbd_ascii(raw_path)

    assert metadata["filename"] == "unit_364-2013-192-1-0"
    assert metadata["mission_name"] == "BAVMSIM.MI"
    assert metadata["encoding_ver"] == "2"
    assert len(columns) == len(units) == 28
    assert len(rows) == 4
    assert rows[0]["sci_water_pressure"] == 0.007
    assert rows[1]["m_depth"] == 25.227
    assert rows[1]["m_gps_lat"] == 3233.7903
    assert rows[1]["m_lat"] == 3233.84081482324


####


def test_ooi_fixture_separates_gps_and_dead_reckoning() -> None:
    fixture = _read_json("fixtures/ooi_unit_364_mixed_provenance.json")
    surface_gps = _state_view(fixture, "unit-364:surface-gps")
    dead_reckoning = _state_view(fixture, "unit-364:underwater-dead-reckoning")

    gps_sample = surface_gps["samples"][0]
    dr_sample = dead_reckoning["samples"][0]
    assert surface_gps["state_role"] == "source_observation"
    assert surface_gps["value_basis"] == "measured"
    assert dead_reckoning["state_role"] == "estimate"
    assert dead_reckoning["value_basis"] == "dead_reckoned"
    assert math.isclose(gps_sample["position"][0], _ddmm_to_decimal(3233.7903))
    assert math.isclose(gps_sample["position"][1], _ddmm_to_decimal(-11802.6978))
    assert math.isclose(dr_sample["position"][0], _ddmm_to_decimal(3233.84081482324))
    assert math.isclose(dr_sample["position"][1], _ddmm_to_decimal(-11802.6863209312))
    assert gps_sample["position"][:2] != dr_sample["position"][:2]
    assert math.isclose(gps_sample["elapsed_s"], 0.3656001091003418)
    assert math.isclose(surface_gps["samples"][1]["elapsed_s"], 41.34756016731262)


####


def test_ooi_fixture_uses_null_and_validity_for_missing_components() -> None:
    fixture = _read_json("fixtures/ooi_unit_364_mixed_provenance.json")
    surface_gps = _state_view(fixture, "unit-364:surface-gps")
    dead_reckoning = _state_view(fixture, "unit-364:underwater-dead-reckoning")
    depth = _state_view(fixture, "unit-364:source-depth")

    for view in (surface_gps, dead_reckoning):
        for sample in view["samples"]:
            assert sample["position"][2] is None
            assert sample["position_valid"] == [True, True, False]

    depth_sample = depth["samples"][0]
    assert depth_sample["position"] == [None, None, 25.227]
    assert depth_sample["position_valid"] == [False, False, True]
    assert depth["frame"]["vertical_reference"] == "unresolved"
    assert depth["frame"]["positive_direction"] == "unresolved"


####


def test_grouping_keys_are_identity_only() -> None:
    fixtures = (
        _read_json("fixtures/ioos_glider_dac_murphy_profile_1.json"),
        _read_json("fixtures/ooi_unit_364_mixed_provenance.json"),
    )

    for fixture in fixtures:
        grouping_keys = fixture["episode"]["grouping_keys"]
        assert grouping_keys
        assert {key["access_class"] for key in grouping_keys} == {"identity_grouping_only"}


####


def test_restricted_fixtures_block_classifier_view_and_do_not_satisfy_g2() -> None:
    fixtures = (
        _read_json("fixtures/ioos_glider_dac_murphy_profile_1.json"),
        _read_json("fixtures/ooi_unit_364_mixed_provenance.json"),
    )

    for fixture in fixtures:
        assert fixture["classifier_view"]["status"] == "intentionally_blocked"
        assert fixture["classifier_view"]["blocking_reasons"]
        assert fixture["acceptance"]["restricted_fixture_validated"] is True
        assert fixture["acceptance"]["lane_g2_satisfied"] is False
        assert fixture["acceptance"]["independent_validation_satisfied"] is False


####


def test_registry_patch_preserves_honest_lifecycle_states() -> None:
    registry = _read_yaml("registry_updates/source_registry_patch.yaml")
    updates = {update["source_dataset_id"]: update for update in registry["updates"]}

    assert len(updates) == 4
    assert updates["ioos-ngdac-uaf-unit_191-20240309T1200"]["evidence_state"] == ("access_verified")
    assert updates["ioos-ngdac-uaf-unit_191-20240309T1200"]["fixture_validated"] is False
    assert updates["mgds-whoi-sentry-at26-09-navigation"]["evidence_state"] == ("access_verified")
    assert updates["mgds-whoi-sentry-at26-09-navigation"]["artifact_acquired"] is False
    assert (
        updates["ioos-glider-dac-murphy-official-regression-artifact"]["evidence_state"]
        == "fixture_validated"
    )
    assert (
        updates["ooici-marine-integrations-glider-parser-resource"]["evidence_state"]
        == "fixture_validated"
    )
    assert all(update["lane_g2_satisfied"] is False for update in updates.values())


####


def test_depth_semantics_change_request_requires_no_schema_fork() -> None:
    request = _read_yaml("change_requests/SCR-SEA-SUB-001_depth_semantics.yaml")
    anchor_mapping = _read_yaml("mappings/ioos_ngdac_uaf_unit_191-20240309T1200.yaml")

    assert request["request_id"] == "SCR-SEA-SUB-001"
    assert request["scope"] == "common_contract_acceptance_clarification"
    assert request["backward_compatibility"].startswith("No root-schema field change")
    assert request["rejected_alternative"].startswith("Relabeling source-calculated depth")

    depth_mapping = next(
        mapping
        for mapping in anchor_mapping["field_mappings"]
        if mapping["target_channel"] == "source_depth"
    )
    pressure_mapping = next(
        mapping
        for mapping in anchor_mapping["field_mappings"]
        if mapping["target_channel"] == "sea_water_pressure"
    )
    assert pressure_mapping["state_role"] == "source_observation"
    assert pressure_mapping["value_basis"] == "measured"
    assert depth_mapping["state_role"] == "estimate"
    assert depth_mapping["value_basis"] == "calculated_from_pressure"
    assert depth_mapping["dependency_channels"] == ["sea_water_pressure"]


####
