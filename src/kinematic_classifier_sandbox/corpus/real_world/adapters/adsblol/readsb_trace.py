from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping, Sequence


STALE_POSITION_FLAG = 1
NEW_LEG_FLAG = 2
GEOMETRIC_VERTICAL_RATE_FLAG = 4
GEOMETRIC_ALTITUDE_FLAG = 8


class ReadsbVerticalBasis(StrEnum):
    BAROMETRIC = "barometric"
    GEOMETRIC = "geometric"


####


@dataclass(frozen=True, slots=True)
class ReadsbTracePoint:
    source_index: int
    event_time_unix_s: float
    latitude_deg: float | None
    longitude_deg: float | None
    altitude_ft: float | None
    altitude_basis: ReadsbVerticalBasis | None
    on_ground: bool
    ground_speed_kt: float | None
    track_deg: float | None
    flags: int
    stale_position: bool
    starts_new_leg: bool
    vertical_rate_fpm: float | None
    vertical_rate_basis: ReadsbVerticalBasis | None
    source_type: str | None
    geometric_altitude_ft: float | None
    geometric_vertical_rate_fpm: float | None
    indicated_airspeed_kt: float | None
    roll_deg: float | None


####


@dataclass(frozen=True, slots=True)
class ReadsbTrace:
    icao: str
    timestamp_unix_s: float
    registration: str | None
    type_code: str | None
    database_flags: int | None
    description: str | None
    points: tuple[ReadsbTracePoint, ...]

    @property
    def military_database_flag(self) -> bool | None:
        if self.database_flags is None:
            return None
        return bool(self.database_flags & 1)
    ####


####


@dataclass(frozen=True, slots=True)
class ReadsbTraceLeg:
    leg_ordinal: int
    points: tuple[ReadsbTracePoint, ...]

    @property
    def start_source_index(self) -> int:
        return self.points[0].source_index
    ####

    @property
    def end_source_index(self) -> int:
        return self.points[-1].source_index
    ####


####


@dataclass(frozen=True, slots=True)
class ReadsbTraceFinding:
    code: str
    source_index: int
    previous_source_index: int | None = None


####


def _optional_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        converted = float(value)
        return converted if math.isfinite(converted) else None
    return None
####


def _optional_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None
####


def _row_value(row: Sequence[object], index: int) -> object:
    return row[index] if index < len(row) else None
####


def _require_finite_number(value: object, *, location: str) -> float:
    converted = _optional_float(value)
    if converted is None:
        raise ValueError(f"{location} must be a finite number")
    return converted
####


def _optional_bounded_float(
    value: object,
    *,
    location: str,
    minimum: float,
    maximum: float,
    maximum_inclusive: bool = True,
) -> float | None:
    converted = _optional_float(value)
    if converted is None:
        return None
    maximum_valid = converted <= maximum if maximum_inclusive else converted < maximum
    if converted < minimum or not maximum_valid:
        interval = f"[{minimum}, {maximum}{']' if maximum_inclusive else ')'}"
        raise ValueError(f"{location} must be in {interval}")
    return converted
####


def _vertical_basis(flags: int, flag: int, value: float | None) -> ReadsbVerticalBasis | None:
    if value is None:
        return None
    if flags & flag:
        return ReadsbVerticalBasis.GEOMETRIC
    return ReadsbVerticalBasis.BAROMETRIC
####


def parse_readsb_trace(payload: Mapping[str, Any]) -> ReadsbTrace:
    icao = _optional_string(payload.get("icao"))
    if icao is None:
        raise ValueError("readsb trace requires non-empty icao")

    timestamp = _require_finite_number(payload.get("timestamp"), location="timestamp")

    raw_trace = payload.get("trace")
    if not isinstance(raw_trace, list) or not raw_trace:
        raise ValueError("readsb trace requires a non-empty trace array")

    points: list[ReadsbTracePoint] = []
    for source_index, raw_row in enumerate(raw_trace):
        if not isinstance(raw_row, list):
            raise ValueError(f"trace[{source_index}] must be an array")
        if len(raw_row) < 8:
            raise ValueError(f"trace[{source_index}] must contain at least eight fields")

        offset_s = _require_finite_number(
            _row_value(raw_row, 0),
            location=f"trace[{source_index}][0]",
        )

        flags_value = _row_value(raw_row, 6)
        if not isinstance(flags_value, int) or isinstance(flags_value, bool):
            raise ValueError(f"trace[{source_index}][6] must be an integer bitfield")
        flags = int(flags_value)

        raw_altitude = _row_value(raw_row, 3)
        on_ground = raw_altitude == "ground"
        altitude_ft = None if on_ground else _optional_float(raw_altitude)
        if raw_altitude not in (None, "ground") and altitude_ft is None:
            raise ValueError(
                f"trace[{source_index}][3] must be numeric, 'ground', or null"
            )

        ground_speed_kt = _optional_float(_row_value(raw_row, 4))
        if ground_speed_kt is not None and ground_speed_kt < 0.0:
            raise ValueError(f"trace[{source_index}][4] must be non-negative")

        indicated_airspeed_kt = _optional_float(_row_value(raw_row, 12))
        if indicated_airspeed_kt is not None and indicated_airspeed_kt < 0.0:
            raise ValueError(f"trace[{source_index}][12] must be non-negative")

        vertical_rate_fpm = _optional_float(_row_value(raw_row, 7))
        points.append(
            ReadsbTracePoint(
                source_index=source_index,
                event_time_unix_s=timestamp + offset_s,
                latitude_deg=_optional_bounded_float(
                    _row_value(raw_row, 1),
                    location=f"trace[{source_index}][1]",
                    minimum=-90.0,
                    maximum=90.0,
                ),
                longitude_deg=_optional_bounded_float(
                    _row_value(raw_row, 2),
                    location=f"trace[{source_index}][2]",
                    minimum=-180.0,
                    maximum=180.0,
                ),
                altitude_ft=altitude_ft,
                altitude_basis=_vertical_basis(
                    flags,
                    GEOMETRIC_ALTITUDE_FLAG,
                    altitude_ft,
                ),
                on_ground=on_ground,
                ground_speed_kt=ground_speed_kt,
                track_deg=_optional_bounded_float(
                    _row_value(raw_row, 5),
                    location=f"trace[{source_index}][5]",
                    minimum=0.0,
                    maximum=360.0,
                    maximum_inclusive=False,
                ),
                flags=flags,
                stale_position=bool(flags & STALE_POSITION_FLAG),
                starts_new_leg=bool(flags & NEW_LEG_FLAG),
                vertical_rate_fpm=vertical_rate_fpm,
                vertical_rate_basis=_vertical_basis(
                    flags,
                    GEOMETRIC_VERTICAL_RATE_FLAG,
                    vertical_rate_fpm,
                ),
                source_type=_optional_string(_row_value(raw_row, 9)),
                geometric_altitude_ft=_optional_float(_row_value(raw_row, 10)),
                geometric_vertical_rate_fpm=_optional_float(_row_value(raw_row, 11)),
                indicated_airspeed_kt=indicated_airspeed_kt,
                roll_deg=_optional_float(_row_value(raw_row, 13)),
            )
        )

    database_flags_value = payload.get("dbFlags")
    database_flags = (
        int(database_flags_value)
        if isinstance(database_flags_value, int)
        and not isinstance(database_flags_value, bool)
        else None
    )

    return ReadsbTrace(
        icao=icao.lower(),
        timestamp_unix_s=timestamp,
        registration=_optional_string(payload.get("r")),
        type_code=_optional_string(payload.get("t")),
        database_flags=database_flags,
        description=_optional_string(payload.get("desc")),
        points=tuple(points),
    )
####


def load_readsb_trace(path: str | Path) -> ReadsbTrace:
    source_path = Path(path)
    if source_path.suffix == ".gz":
        with gzip.open(source_path, mode="rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        with source_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError("readsb trace file must contain one JSON object")
    return parse_readsb_trace(payload)
####


def split_readsb_legs(
    trace: ReadsbTrace,
    *,
    minimum_samples: int = 1,
) -> tuple[ReadsbTraceLeg, ...]:
    if minimum_samples < 1:
        raise ValueError("minimum_samples must be at least one")

    accepted: list[ReadsbTraceLeg] = []
    current: list[ReadsbTracePoint] = []

    def append_current() -> None:
        if len(current) >= minimum_samples:
            accepted.append(
                ReadsbTraceLeg(
                    leg_ordinal=len(accepted),
                    points=tuple(current),
                )
            )
    ####

    for point in trace.points:
        if point.starts_new_leg and current:
            append_current()
            current = []
        current.append(point)

    append_current()
    return tuple(accepted)
####


def trace_time_findings(trace: ReadsbTrace) -> tuple[ReadsbTraceFinding, ...]:
    findings: list[ReadsbTraceFinding] = []
    previous: ReadsbTracePoint | None = None
    for point in trace.points:
        if previous is not None:
            if point.event_time_unix_s == previous.event_time_unix_s:
                findings.append(
                    ReadsbTraceFinding(
                        code="AIR_DUPLICATE_TIMESTAMP",
                        source_index=point.source_index,
                        previous_source_index=previous.source_index,
                    )
                )
            elif point.event_time_unix_s < previous.event_time_unix_s:
                findings.append(
                    ReadsbTraceFinding(
                        code="AIR_OUT_OF_ORDER_TIMESTAMP",
                        source_index=point.source_index,
                        previous_source_index=previous.source_index,
                    )
                )
        previous = point
    return tuple(findings)
####
