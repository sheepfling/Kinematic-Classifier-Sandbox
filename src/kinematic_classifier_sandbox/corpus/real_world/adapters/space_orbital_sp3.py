"""Strict SP3-c position parsing for SPACE-ORB source inspection."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np

from .space_orbital_sp3_records import records_to_si_arrays, select_satellite_records
from .space_orbital_sp3_types import (
    GPS_EPOCH,
    GPS_WEEK_SECONDS,
    SP3_MISSING_CLOCK_ABS_MIN,
    SP3_NONSTANDARD_POSITION_SENTINEL,
    Sp3Extract,
    Sp3Header,
    Sp3PositionRecord,
)


_VALID_TIME_SYSTEMS: Final[frozenset[str]] = frozenset(
    {"GPS", "GLO", "GAL", "TAI", "UTC", "QZS"}
)
_SATELLITE_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Z][0-9]{2}$")
_HEADER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^#(?P<version>[a-z])(?P<data_type>[A-Z])"
    r"(?P<year>\d{4})\s+(?P<month>\d{1,2})\s+(?P<day>\d{1,2})\s+"
    r"(?P<hour>\d{1,2})\s+(?P<minute>\d{1,2})\s+"
    r"(?P<second>\d+(?:\.\d+)?)\s+"
    r"(?P<epoch_count>\d+)\s+\S+\s+"
    r"(?P<coordinate_system>\S+)\s+"
    r"(?P<orbit_type>\S+)\s+"
    r"(?P<agency>\S+)\s*$"
)


def _datetime_from_fields(
    *,
    year: int,
    month: int,
    day: int,
    hour: int,
    minute: int,
    second: float,
    field_name: str,
) -> datetime:
    if not np.isfinite(second) or second < 0.0 or second >= 61.0:
        raise ValueError(f"invalid seconds value in {field_name}: {second!r}")
    try:
        base = datetime(year, month, day, hour, minute)
    except ValueError as exc:
        raise ValueError(f"invalid date fields in {field_name}") from exc
    return base + timedelta(seconds=second)
####


def _parse_epoch_line(line: str) -> datetime:
    fields = line[1:].split()
    if len(fields) != 6:
        raise ValueError(f"invalid SP3 epoch line: {line!r}")
    return _datetime_from_fields(
        year=int(fields[0]),
        month=int(fields[1]),
        day=int(fields[2]),
        hour=int(fields[3]),
        minute=int(fields[4]),
        second=float(fields[5]),
        field_name="SP3 epoch line",
    )
####


def _declared_time_system(lines: list[str]) -> str:
    candidates: list[str] = []
    for line in lines:
        if not line.startswith("%c"):
            continue
        fields = line.split()
        if len(fields) < 4:
            continue
        candidate = fields[3].upper()
        if candidate not in {"CC", "CCC", "CCCC"}:
            candidates.append(candidate)
    unique = tuple(dict.fromkeys(candidates))
    if len(unique) > 1:
        raise ValueError(f"conflicting SP3 time-system declarations: {unique}")
    if not unique:
        raise ValueError("SP3-c requires an explicit time-system declaration")
    if unique[0] not in _VALID_TIME_SYSTEMS:
        raise ValueError(f"unsupported SP3-c time system: {unique[0]!r}")
    return unique[0]
####


def _declared_satellite_roster(lines: list[str]) -> tuple[int, tuple[str, ...]]:
    roster_lines = [
        line for line in lines if line.startswith("+") and not line.startswith("++")
    ]
    if not roster_lines:
        raise ValueError("SP3-c header contains no satellite roster")
    try:
        declared_count = int(roster_lines[0][3:6])
    except (ValueError, IndexError) as exc:
        raise ValueError("invalid SP3-c satellite count") from exc
    if declared_count < 1:
        raise ValueError("SP3-c declared satellite count must be positive")

    satellite_ids: list[str] = []
    for line in roster_lines:
        payload = line[9:60].ljust(51)
        for offset in range(0, 51, 3):
            candidate = payload[offset : offset + 3].strip().upper()
            if candidate in {"", "0"}:
                continue
            if _SATELLITE_ID_PATTERN.fullmatch(candidate) is None:
                raise ValueError(f"invalid SP3 satellite ID in header: {candidate!r}")
            satellite_ids.append(candidate)
    if len(satellite_ids) != declared_count:
        raise ValueError(
            "SP3 declared satellite count does not match header roster: "
            f"declared {declared_count}, parsed {len(satellite_ids)}"
        )
    if len(satellite_ids) != len(set(satellite_ids)):
        raise ValueError("SP3 satellite roster contains duplicate IDs")
    return declared_count, tuple(satellite_ids)
####


def _parse_header(lines: list[str]) -> Sp3Header:
    if len(lines) < 2:
        raise ValueError("SP3 input must contain at least two header lines")

    match = _HEADER_PATTERN.fullmatch(lines[0])
    if match is None:
        raise ValueError(f"invalid SP3 primary header line: {lines[0]!r}")
    if not lines[1].startswith("##"):
        raise ValueError(f"invalid SP3 secondary header line: {lines[1]!r}")

    second_fields = lines[1][2:].split()
    if len(second_fields) < 3:
        raise ValueError(f"invalid SP3 secondary header line: {lines[1]!r}")

    start_epoch = _datetime_from_fields(
        year=int(match.group("year")),
        month=int(match.group("month")),
        day=int(match.group("day")),
        hour=int(match.group("hour")),
        minute=int(match.group("minute")),
        second=float(match.group("second")),
        field_name="SP3 primary header",
    )
    declared_satellite_count, satellite_ids = _declared_satellite_roster(lines)
    comments = tuple(line[2:].strip() for line in lines if line.startswith("/*"))
    header = Sp3Header(
        version=match.group("version"),
        data_type=match.group("data_type"),
        start_epoch=start_epoch,
        declared_epoch_count=int(match.group("epoch_count")),
        coordinate_system=match.group("coordinate_system"),
        orbit_type=match.group("orbit_type"),
        agency=match.group("agency"),
        gps_week=int(second_fields[0]),
        seconds_of_week=float(second_fields[1]),
        sampling_period_s=float(second_fields[2]),
        time_system=_declared_time_system(lines),
        declared_satellite_count=declared_satellite_count,
        satellite_ids=satellite_ids,
        comments=comments,
    )
    if header.version != "c":
        raise ValueError(f"this parser requires SP3-c; got version {header.version!r}")
    if header.data_type != "P":
        raise ValueError(
            f"this position parser requires data type 'P'; got {header.data_type!r}"
        )
    if header.declared_epoch_count < 1:
        raise ValueError("SP3 declared epoch count must be positive")
    if not np.isfinite(header.sampling_period_s) or header.sampling_period_s <= 0.0:
        raise ValueError("SP3 sampling period must be finite and positive")
    if header.time_system == "GPS":
        declared_start = GPS_EPOCH + timedelta(
            seconds=(header.gps_week * GPS_WEEK_SECONDS) + header.seconds_of_week
        )
        if abs((declared_start - header.start_epoch).total_seconds()) > 1.0e-6:
            raise ValueError(
                "SP3 GPS week/seconds do not match the primary-header start epoch"
            )
    return header
####


def _parse_position_record(line: str, epoch: datetime) -> Sp3PositionRecord:
    fields = line.split()
    if len(fields) < 5:
        raise ValueError(f"invalid SP3 position record: {line!r}")
    satellite_id = fields[0][1:].upper()
    if _SATELLITE_ID_PATTERN.fullmatch(satellite_id) is None:
        raise ValueError(f"invalid SP3 satellite ID: {satellite_id!r}")

    x_km, y_km, z_km = (float(value) for value in fields[1:4])
    position = np.asarray((x_km, y_km, z_km), dtype=np.float64)
    if not np.all(np.isfinite(position)):
        raise ValueError(f"non-finite position for {satellite_id} at {epoch.isoformat()}")
    if np.all(position == 0.0) or np.any(
        np.abs(position) >= SP3_NONSTANDARD_POSITION_SENTINEL
    ):
        raise ValueError(f"position missing for {satellite_id} at {epoch.isoformat()}")

    clock_value = float(fields[4])
    if not np.isfinite(clock_value):
        raise ValueError(f"non-finite clock value for {satellite_id} at {epoch.isoformat()}")
    clock_offset_microseconds = (
        None if abs(clock_value) >= SP3_MISSING_CLOCK_ABS_MIN else clock_value
    )
    return Sp3PositionRecord(
        epoch=epoch,
        satellite_id=satellite_id,
        position_km=(x_km, y_km, z_km),
        clock_offset_microseconds=clock_offset_microseconds,
    )
####


def parse_sp3_text(text: str) -> Sp3Extract:
    """Parse a complete regular-cadence SP3-c position product."""

    lines = [line.rstrip("\r\n") for line in text.splitlines() if line.strip()]
    header = _parse_header(lines)
    current_epoch: datetime | None = None
    epochs: list[datetime] = []
    records: list[Sp3PositionRecord] = []
    record_ids_by_epoch: dict[datetime, list[str]] = {}
    seen_record_keys: set[tuple[datetime, str]] = set()

    for line in lines:
        if line.startswith("*"):
            epoch = _parse_epoch_line(line)
            if epochs and epoch <= epochs[-1]:
                raise ValueError("SP3 epochs must be strictly increasing")
            epochs.append(epoch)
            record_ids_by_epoch[epoch] = []
            current_epoch = epoch
        elif line.startswith("P"):
            if current_epoch is None:
                raise ValueError("SP3 position record appears before an epoch line")
            record = _parse_position_record(line, current_epoch)
            record_key = (record.epoch, record.satellite_id)
            if record_key in seen_record_keys:
                raise ValueError(
                    "duplicate SP3 position record for "
                    f"{record.satellite_id} at {record.epoch.isoformat()}"
                )
            seen_record_keys.add(record_key)
            record_ids_by_epoch[current_epoch].append(record.satellite_id)
            records.append(record)
        elif line.startswith("V"):
            raise ValueError("velocity records are not permitted in an SP3-c P product")

    if not epochs:
        raise ValueError("SP3 input contains no epochs")
    if not records:
        raise ValueError("SP3 input contains no position records")
    if len(epochs) != header.declared_epoch_count:
        raise ValueError(
            "SP3 declared epoch count does not match parsed epoch count: "
            f"declared {header.declared_epoch_count}, parsed {len(epochs)}"
        )
    if epochs[0] != header.start_epoch:
        raise ValueError("SP3 first epoch does not match the primary-header start epoch")
    for epoch in epochs:
        actual_roster = tuple(record_ids_by_epoch[epoch])
        if actual_roster != header.satellite_ids:
            raise ValueError(
                "SP3 position-record roster does not match the header roster at "
                f"{epoch.isoformat()}"
            )
    if len(epochs) > 1:
        epoch_seconds = np.asarray(
            [(epoch - epochs[0]).total_seconds() for epoch in epochs],
            dtype=np.float64,
        )
        deltas = np.diff(epoch_seconds)
        if not np.allclose(
            deltas,
            header.sampling_period_s,
            rtol=0.0,
            atol=1.0e-6,
        ):
            raise ValueError(
                "SP3 epoch cadence does not match the declared sampling period"
            )
    return Sp3Extract(header=header, records=tuple(records))
####


def parse_sp3_file(path: str | Path) -> Sp3Extract:
    return parse_sp3_text(Path(path).read_text(encoding="ascii"))
####


__all__ = [
    "GPS_EPOCH",
    "GPS_WEEK_SECONDS",
    "SP3_MISSING_CLOCK_ABS_MIN",
    "SP3_NONSTANDARD_POSITION_SENTINEL",
    "Sp3Extract",
    "Sp3Header",
    "Sp3PositionRecord",
    "parse_sp3_file",
    "parse_sp3_text",
    "records_to_si_arrays",
    "select_satellite_records",
]
