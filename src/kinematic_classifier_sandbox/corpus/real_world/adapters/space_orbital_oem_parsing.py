"""Bounded single-segment UTC CCSDS OEM parsing primitives."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
_EXPECTED_STATE_FIELD_COUNT: Final[int] = 7


@dataclass(frozen=True, slots=True)
class OemHeader:
    version: str
    creation_date_utc: datetime
    originator: str
####


@dataclass(frozen=True, slots=True)
class OemMetadata:
    object_name: str
    source_object_id: str
    center_name: str
    reference_frame: str
    time_system: str
    start_time_utc: datetime
    useable_start_time_utc: datetime
    useable_stop_time_utc: datetime
    stop_time_utc: datetime
####


@dataclass(frozen=True, slots=True)
class OemStateRecord:
    epoch_utc: datetime
    position_km: tuple[float, float, float]
    velocity_kmps: tuple[float, float, float]
####


@dataclass(frozen=True, slots=True)
class OemExtract:
    header: OemHeader
    metadata: OemMetadata
    comments: tuple[str, ...]
    records: tuple[OemStateRecord, ...]
####


def _parse_utc_timestamp(value: str, *, field_name: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"invalid UTC timestamp in {field_name}: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
####


def _parse_key_value(line: str) -> tuple[str, str]:
    if "=" not in line:
        raise ValueError(f"expected KEY = VALUE line, got {line!r}")
    key, value = line.split("=", maxsplit=1)
    normalized_key = key.strip()
    normalized_value = value.strip()
    if not normalized_key or not normalized_value:
        raise ValueError(f"invalid KEY = VALUE line: {line!r}")
    return normalized_key, normalized_value
####


def _parse_state_line(line: str) -> OemStateRecord:
    fields = line.split()
    if len(fields) != _EXPECTED_STATE_FIELD_COUNT:
        raise ValueError(f"invalid OEM state line: {line!r}")
    values = tuple(float(value) for value in fields[1:])
    if not np.all(np.isfinite(np.asarray(values, dtype=np.float64))):
        raise ValueError(f"OEM state line contains non-finite values: {line!r}")
    return OemStateRecord(
        epoch_utc=_parse_utc_timestamp(fields[0], field_name="OEM state epoch"),
        position_km=(values[0], values[1], values[2]),
        velocity_kmps=(values[3], values[4], values[5]),
    )
####


def _validate_metadata_interval(metadata: OemMetadata) -> None:
    ordered = (
        metadata.start_time_utc,
        metadata.useable_start_time_utc,
        metadata.useable_stop_time_utc,
        metadata.stop_time_utc,
    )
    if any(current < previous for previous, current in zip(ordered, ordered[1:])):
        raise ValueError(
            "OEM metadata times must satisfy START <= USEABLE_START <= "
            "USEABLE_STOP <= STOP"
        )
####


def _validate_record_interval(
    metadata: OemMetadata,
    records: tuple[OemStateRecord, ...],
) -> None:
    for record in records:
        if not (
            metadata.useable_start_time_utc
            <= record.epoch_utc
            <= metadata.useable_stop_time_utc
        ):
            raise ValueError(
                "OEM state epoch falls outside the declared useable interval: "
                f"{record.epoch_utc.isoformat()}"
            )
####


def parse_oem_text(text: str) -> OemExtract:
    """Parse one state-only OEM 2.0 KVN segment whose declared time system is UTC."""

    lines = [line.rstrip("\r\n") for line in text.splitlines()]
    header_values: dict[str, str] = {}
    metadata_values: dict[str, str] = {}
    comments: list[str] = []
    raw_state_lines: list[str] = []
    in_metadata = False
    metadata_segment_count = 0
    state_data_started = False

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line == "META_START":
            if in_metadata:
                raise ValueError("nested META_START is invalid")
            if state_data_started:
                raise ValueError("metadata after state data is not supported")
            metadata_segment_count += 1
            if metadata_segment_count > 1:
                raise ValueError("exactly one OEM metadata segment is supported")
            in_metadata = True
            continue
        if line == "META_STOP":
            if not in_metadata:
                raise ValueError("META_STOP appears before META_START")
            in_metadata = False
            continue
        if line.startswith("COMMENT"):
            comments.append(line[len("COMMENT") :].strip())
            continue
        if line[0].isdigit():
            if in_metadata:
                raise ValueError("OEM state appears inside metadata block")
            if metadata_segment_count != 1:
                raise ValueError("OEM state appears before a complete metadata segment")
            state_data_started = True
            raw_state_lines.append(line)
            continue

        key, value = _parse_key_value(line)
        if metadata_segment_count == 1 and not in_metadata:
            raise ValueError(
                f"unsupported OEM content outside the metadata block: {line!r}"
            )
        target = metadata_values if in_metadata else header_values
        if key in target:
            raise ValueError(f"duplicate OEM field {key!r}")
        target[key] = value

    if in_metadata:
        raise ValueError("OEM metadata block was not closed")
    if metadata_segment_count != 1:
        raise ValueError("exactly one OEM metadata segment is required")
    if len(raw_state_lines) < 2:
        raise ValueError("OEM extract must contain at least two state records")

    required_header = {"CCSDS_OEM_VERS", "CREATION_DATE", "ORIGINATOR"}
    missing_header = required_header - header_values.keys()
    if missing_header:
        raise ValueError(f"missing OEM header fields: {sorted(missing_header)}")

    required_metadata = {
        "OBJECT_NAME",
        "OBJECT_ID",
        "CENTER_NAME",
        "REF_FRAME",
        "TIME_SYSTEM",
        "START_TIME",
        "USEABLE_START_TIME",
        "USEABLE_STOP_TIME",
        "STOP_TIME",
    }
    missing_metadata = required_metadata - metadata_values.keys()
    if missing_metadata:
        raise ValueError(f"missing OEM metadata fields: {sorted(missing_metadata)}")

    time_system = metadata_values["TIME_SYSTEM"].strip().upper()
    if time_system != "UTC":
        raise ValueError(
            "this bounded OEM parser requires TIME_SYSTEM = UTC; "
            f"got {metadata_values['TIME_SYSTEM']!r}"
        )

    header = OemHeader(
        version=header_values["CCSDS_OEM_VERS"],
        creation_date_utc=_parse_utc_timestamp(
            header_values["CREATION_DATE"],
            field_name="CREATION_DATE",
        ),
        originator=header_values["ORIGINATOR"],
    )
    metadata = OemMetadata(
        object_name=metadata_values["OBJECT_NAME"],
        source_object_id=metadata_values["OBJECT_ID"],
        center_name=metadata_values["CENTER_NAME"],
        reference_frame=metadata_values["REF_FRAME"],
        time_system=time_system,
        start_time_utc=_parse_utc_timestamp(
            metadata_values["START_TIME"],
            field_name="START_TIME",
        ),
        useable_start_time_utc=_parse_utc_timestamp(
            metadata_values["USEABLE_START_TIME"],
            field_name="USEABLE_START_TIME",
        ),
        useable_stop_time_utc=_parse_utc_timestamp(
            metadata_values["USEABLE_STOP_TIME"],
            field_name="USEABLE_STOP_TIME",
        ),
        stop_time_utc=_parse_utc_timestamp(
            metadata_values["STOP_TIME"],
            field_name="STOP_TIME",
        ),
    )
    _validate_metadata_interval(metadata)

    records = tuple(_parse_state_line(line) for line in raw_state_lines)
    epochs = tuple(record.epoch_utc for record in records)
    if any(current <= previous for previous, current in zip(epochs, epochs[1:])):
        raise ValueError("OEM epochs must be strictly increasing")
    _validate_record_interval(metadata, records)

    return OemExtract(
        header=header,
        metadata=metadata,
        comments=tuple(comments),
        records=records,
    )
####


def parse_oem_file(path: str | Path) -> OemExtract:
    return parse_oem_text(Path(path).read_text(encoding="ascii"))
####


def records_to_si_arrays(
    records: tuple[OemStateRecord, ...],
) -> tuple[FloatArray, FloatArray, FloatArray]:
    if len(records) < 2:
        raise ValueError("at least two OEM state records are required")
    unix_utc_seconds = np.asarray(
        [record.epoch_utc.timestamp() for record in records],
        dtype=np.float64,
    )
    position_m = np.asarray(
        [[component * 1_000.0 for component in record.position_km] for record in records],
        dtype=np.float64,
    )
    source_velocity_mps = np.asarray(
        [
            [component * 1_000.0 for component in record.velocity_kmps]
            for record in records
        ],
        dtype=np.float64,
    )
    if not np.all(np.diff(unix_utc_seconds) > 0.0):
        raise ValueError("OEM UTC seconds must be strictly increasing")
    if not np.all(np.isfinite(position_m)):
        raise ValueError("OEM positions must be finite")
    if not np.all(np.isfinite(source_velocity_mps)):
        raise ValueError("OEM source velocities must be finite")
    return unix_utc_seconds, position_m, source_velocity_mps
####


__all__ = [
    "FloatArray",
    "OemExtract",
    "OemHeader",
    "OemMetadata",
    "OemStateRecord",
    "parse_oem_file",
    "parse_oem_text",
    "records_to_si_arrays",
]
