from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from kinematic_classifier_sandbox.corpus.real_world.adapters.adsblol.readsb_trace import (
    ReadsbVerticalBasis,
    load_readsb_trace,
    parse_readsb_trace,
    split_readsb_legs,
    trace_time_findings,
)


_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "readsb_documented_a320_trace.json"


def _load_documented_fixture() -> dict[str, object]:
    payload = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload
####


def test_parse_preserves_documented_time_vertical_and_source_semantics() -> None:
    trace = parse_readsb_trace(_load_documented_fixture())

    assert trace.icao == "3c66b0"
    assert trace.registration == "D-AIUP"
    assert trace.type_code == "A320"
    assert trace.military_database_flag is False
    assert len(trace.points) == 7

    first = trace.points[0]
    assert first.event_time_unix_s == pytest.approx(1663266869.606)
    assert first.altitude_ft == 25125.0
    assert first.altitude_basis is ReadsbVerticalBasis.BAROMETRIC
    assert first.geometric_altitude_ft == 25875.0
    assert first.vertical_rate_basis is ReadsbVerticalBasis.BAROMETRIC
    assert first.geometric_vertical_rate_fpm == -2208.0
    assert first.source_type == "adsb_icao"
    assert first.track_or_ground_heading_deg == 309.0
####


def test_flag_decoding_keeps_ground_stale_and_geometric_semantics_distinct() -> None:
    payload: dict[str, object] = {
        "icao": "abc123",
        "timestamp": 1000.0,
        "trace": [
            [0.0, 34.0, -86.0, "ground", 0.0, 90.0, 0, None],
            [1.0, 34.1, -86.1, 1500, 100.0, 91.0, 15, 256],
        ],
    }

    trace = parse_readsb_trace(payload)

    assert trace.points[0].on_ground is True
    assert trace.points[0].altitude_ft is None
    assert trace.points[0].altitude_basis is None

    flagged = trace.points[1]
    assert flagged.stale_position is True
    assert flagged.starts_new_leg is True
    assert flagged.altitude_basis is ReadsbVerticalBasis.GEOMETRIC
    assert flagged.vertical_rate_basis is ReadsbVerticalBasis.GEOMETRIC
####


@pytest.mark.parametrize(
    ("flags", "expected_altitude_basis", "expected_vertical_rate_basis"),
    (
        (4, ReadsbVerticalBasis.GEOMETRIC, ReadsbVerticalBasis.BAROMETRIC),
        (8, ReadsbVerticalBasis.BAROMETRIC, ReadsbVerticalBasis.GEOMETRIC),
    ),
)
def test_single_vertical_basis_flags_decode_their_declared_channels(
    flags: int,
    expected_altitude_basis: ReadsbVerticalBasis,
    expected_vertical_rate_basis: ReadsbVerticalBasis,
) -> None:
    payload: dict[str, object] = {
        "icao": "abc123",
        "timestamp": 1000.0,
        "trace": [[0.0, 34.0, -86.0, 1500, 100.0, 91.0, flags, 256]],
    }

    point = parse_readsb_trace(payload).points[0]

    assert point.altitude_basis is expected_altitude_basis
    assert point.vertical_rate_basis is expected_vertical_rate_basis
####


def test_split_uses_source_new_leg_evidence_without_rewriting_parent_trace() -> None:
    payload: dict[str, object] = {
        "icao": "abc123",
        "timestamp": 1000.0,
        "trace": [
            [0.0, 34.0, -86.0, 1000, 80.0, 90.0, 0, 100],
            [1.0, 34.1, -86.1, 1100, 82.0, 90.0, 0, 100],
            [2.0, 34.2, -86.2, 1200, 84.0, 90.0, 2, 100],
            [3.0, 34.3, -86.3, 1300, 86.0, 90.0, 0, 100],
        ],
    }

    trace = parse_readsb_trace(payload)
    legs = split_readsb_legs(trace, minimum_samples=2)

    assert len(legs) == 2
    assert legs[0].leg_ordinal == 0
    assert (legs[0].start_source_index, legs[0].end_source_index) == (0, 1)
    assert (legs[1].start_source_index, legs[1].end_source_index) == (2, 3)
    assert trace.icao == "abc123"
####


def test_duplicate_and_out_of_order_times_remain_explicit_findings() -> None:
    payload = _load_documented_fixture()
    trace_rows = payload["trace"]
    assert isinstance(trace_rows, list)
    assert isinstance(trace_rows[0], list)
    assert isinstance(trace_rows[1], list)
    assert isinstance(trace_rows[2], list)
    trace_rows[1][0] = trace_rows[0][0]
    trace_rows[2][0] = float(trace_rows[0][0]) - 1.0

    findings = trace_time_findings(parse_readsb_trace(payload))

    assert [finding.code for finding in findings] == [
        "AIR_DUPLICATE_TIMESTAMP",
        "AIR_OUT_OF_ORDER_TIMESTAMP",
    ]
    assert findings[0].source_index == 1
    assert findings[1].source_index == 2
####


def test_loader_accepts_plain_and_gzip_json(tmp_path: Path) -> None:
    raw = _FIXTURE_PATH.read_text(encoding="utf-8")
    plain_path = tmp_path / "trace.json"
    gzip_path = tmp_path / "trace.json.gz"
    plain_path.write_text(raw, encoding="utf-8")
    with gzip.open(gzip_path, mode="wt", encoding="utf-8") as handle:
        handle.write(raw)

    assert load_readsb_trace(plain_path) == load_readsb_trace(gzip_path)
####


def test_parser_rejects_invalid_coordinates_without_silent_clamping() -> None:
    payload = _load_documented_fixture()
    trace_rows = payload["trace"]
    assert isinstance(trace_rows, list)
    assert isinstance(trace_rows[0], list)
    trace_rows[0][1] = 91.0

    with pytest.raises(ValueError, match=r"trace\[0\]\[1\].*\[-90.0, 90.0\]"):
        parse_readsb_trace(payload)
####


def test_filtered_short_leg_does_not_renumber_source_leg_ordinals() -> None:
    payload: dict[str, object] = {
        "icao": "abc123",
        "timestamp": 1000.0,
        "trace": [
            [0.0, 34.0, -86.0, 1000, 80.0, 90.0, 0, 100],
            [1.0, 34.1, -86.1, 1100, 82.0, 90.0, 2, 100],
            [2.0, 34.2, -86.2, 1200, 84.0, 90.0, 0, 100],
        ],
    }

    legs = split_readsb_legs(parse_readsb_trace(payload), minimum_samples=2)

    assert len(legs) == 1
    assert legs[0].leg_ordinal == 1
    assert (legs[0].start_source_index, legs[0].end_source_index) == (1, 2)
####


def test_parser_rejects_non_numeric_optional_values_instead_of_dropping_them() -> None:
    payload = _load_documented_fixture()
    trace_rows = payload["trace"]
    assert isinstance(trace_rows, list)
    assert isinstance(trace_rows[0], list)
    trace_rows[0][4] = "fast"

    with pytest.raises(ValueError, match=r"trace\[0\]\[4\].*finite number or null"):
        parse_readsb_trace(payload)
####
