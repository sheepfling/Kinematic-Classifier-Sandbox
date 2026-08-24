from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
LANE_ROOT = REPO_ROOT / "docs/research/product4/sea_subsurface"
ARTIFACT_PATH = LANE_ROOT / "fixtures/ioos_uaf_unit_191_profile_1709942882.csv"
INSPECTION_PATH = LANE_ROOT / "acquisition/ioos_profile_1709942882_inspection.json"
MISSING_VALUES = {"", "NaN", "nan", "null", "-999", "-9999.9"}


def _read_inspection() -> dict[str, Any]:
    payload = json.loads(INSPECTION_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload
####


def _read_anchor_rows() -> tuple[dict[str, str], list[dict[str, str]]]:
    with ARTIFACT_PATH.open(newline="", encoding="iso-8859-1") as stream:
        reader = csv.DictReader(stream)
        raw_rows = [
            {str(key): "" if value is None else str(value) for key, value in row.items()}
            for row in reader
        ]
    assert raw_rows
    units = raw_rows[0]
    data_rows = [row for row in raw_rows[1:] if row["profile_id"].isdigit()]
    return units, data_rows
####


def _present(value: str) -> bool:
    return value.strip() not in MISSING_VALUES
####


def _populated_source_channels(row: dict[str, str]) -> set[str]:
    identity_and_context = {
        "trajectory",
        "wmo_id",
        "profile_id",
        "precise_time",
        "latitude",
        "longitude",
    }
    return {
        field
        for field, value in row.items()
        if field not in identity_and_context and _present(value)
    }
####


def test_selected_anchor_bytes_match_retained_inspection() -> None:
    inspection = _read_inspection()
    artifact = inspection["source_artifact"]

    assert ARTIFACT_PATH.stat().st_size == 15465 == artifact["byte_size"]
    assert hashlib.sha256(ARTIFACT_PATH.read_bytes()).hexdigest() == (
        "29114562885c844dec7440148a4dbe8bfbc21efcc4f793190bdd0c5ff2a6d13a"
    )
    assert artifact["sha256"] == (
        "29114562885c844dec7440148a4dbe8bfbc21efcc4f793190bdd0c5ff2a6d13a"
    )
    assert artifact["http_status"] == 200
    assert artifact["erddap_version"] == "2.30.0"
    assert inspection["shape"] == {
        "column_count": 16,
        "data_row_count": 99,
        "units_row_count": 1,
    }
####


def test_selected_anchor_preserves_sparse_mixed_provenance_channels() -> None:
    _, rows = _read_anchor_rows()
    inspection = _read_inspection()
    counts = inspection["channel_presence"]

    assert {row["trajectory"] for row in rows} == {"unit_191-20240309T1200"}
    assert {row["wmo_id"] for row in rows} == {"4902987"}
    assert {row["profile_id"] for row in rows} == {"1709942882"}
    assert counts == {
        "coincident_gps_dead_reckoned_rows": 0,
        "dead_reckoned_rows": 2,
        "heading_rows": 4,
        "onboard_gps_rows": 1,
        "pitch_rows": 4,
        "pressure_rows": 63,
        "roll_rows": 4,
        "source_depth_rows": 24,
        "standardized_depth_rows": 63,
    }

    gps_rows = [row for row in rows if _present(row["m_gps_lat"])]
    dead_reckoned_rows = [row for row in rows if _present(row["m_lat"])]
    assert len(gps_rows) == 1
    assert len(dead_reckoned_rows) == 2
    assert all(not _present(row["m_lat"]) for row in gps_rows)
    assert all(not _present(row["m_gps_lat"]) for row in dead_reckoned_rows)
####


def test_duplicate_timestamps_are_channel_events_not_duplicate_states() -> None:
    _, rows = _read_anchor_rows()
    by_time: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_time.setdefault(row["precise_time"], []).append(row)

    duplicate_groups = {time: group for time, group in by_time.items() if len(group) > 1}
    assert len(duplicate_groups) == 5
    channel_patterns = {
        tuple(tuple(sorted(_populated_source_channels(row))) for row in group)
        for group in duplicate_groups.values()
    }
    assert channel_patterns == {(("m_depth",), ("depth", "pressure"))}

    inspection = _read_inspection()
    assert inspection["time"]["duplicate_timestamp_count"] == 5
    assert inspection["time"]["out_of_order_count"] == 0
    assert inspection["time"]["maximum_positive_gap_s"] == 9.0
    finding_codes = {finding["code"] for finding in inspection["quality_findings"]}
    assert "SEA_SUB_ASYNCHRONOUS_CHANNEL_EVENTS" in finding_codes
####


def test_gps_units_mismatch_is_pinned_without_unsafe_ddmm_conversion() -> None:
    units, rows = _read_anchor_rows()
    gps_row = next(row for row in rows if _present(row["m_gps_lat"]))
    latitude = float(gps_row["m_gps_lat"])
    longitude = float(gps_row["m_gps_lon"])
    inspection = _read_inspection()
    decision = inspection["mapping_decisions"]["m_gps_lat_lon"]

    assert units["m_gps_lat"] == "degrees_minutes_north"
    assert units["m_gps_lon"] == "degrees_minutes_east"
    assert -90.0 <= latitude <= 90.0
    assert -180.0 <= longitude <= 180.0
    assert abs(latitude - float(gps_row["latitude"])) < 0.001
    assert abs(longitude - float(gps_row["longitude"])) < 0.001
    assert decision["artifact_encoding"] == "decimal_degrees_as_retained"
    assert decision["surface_phase_inferred"] is False
    assert decision["target_channel"] == "onboard_gps_horizontal_position"

    finding_codes = {finding["code"] for finding in inspection["quality_findings"]}
    assert "SEA_SUB_GPS_UNITS_VALUE_MISMATCH" in finding_codes
    assert "SEA_SUB_GPS_PHASE_NOT_PROVEN" in finding_codes
####


def test_selected_anchor_is_mapping_complete_but_not_g2_validated() -> None:
    inspection = _read_inspection()
    acceptance = inspection["acceptance"]

    assert acceptance == {
        "artifact_acquired": True,
        "canonical_common_front_validated": False,
        "lane_g2_satisfied": False,
        "mapping_complete": True,
        "schema_inspected": True,
        "selected_anchor_fixture_validated": False,
    }
####
