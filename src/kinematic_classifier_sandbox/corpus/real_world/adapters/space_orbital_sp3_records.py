"""Selection and SI conversion for strict SP3-c position records."""

from __future__ import annotations

import re
from typing import Final, Iterable

import numpy as np

from .space_orbital_sp3_types import (
    FloatArray,
    GPS_WEEK_SECONDS,
    Sp3Extract,
    Sp3Header,
    Sp3PositionRecord,
)


_SATELLITE_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Z][0-9]{2}$")


def select_satellite_records(
    extract: Sp3Extract,
    satellite_id: str,
) -> tuple[Sp3PositionRecord, ...]:
    normalized_id = satellite_id.strip().upper()
    if _SATELLITE_ID_PATTERN.fullmatch(normalized_id) is None:
        raise ValueError(f"invalid SP3 satellite ID: {satellite_id!r}")
    selected = tuple(
        record for record in extract.records if record.satellite_id == normalized_id
    )
    if len(selected) < 2:
        raise ValueError(f"expected at least two records for {normalized_id!r}")
    epochs = tuple(record.epoch for record in selected)
    if any(current <= previous for previous, current in zip(epochs, epochs[1:])):
        raise ValueError(f"epochs for {normalized_id!r} must be strictly increasing")
    return selected
####


def records_to_si_arrays(
    header: Sp3Header,
    records: Iterable[Sp3PositionRecord],
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Convert one satellite's GPS-time SP3 positions and clocks to SI arrays."""

    if header.time_system != "GPS":
        raise ValueError(
            "GPS-second conversion requires an SP3 header declaring time system GPS; "
            f"got {header.time_system!r}"
        )
    materialized = tuple(records)
    if len(materialized) < 2:
        raise ValueError("at least two records are required")
    satellite_ids = {record.satellite_id for record in materialized}
    if len(satellite_ids) != 1:
        raise ValueError("records_to_si_arrays requires records from one satellite")
    epochs = tuple(record.epoch for record in materialized)
    if any(current <= previous for previous, current in zip(epochs, epochs[1:])):
        raise ValueError("SP3 record epochs must be strictly increasing")

    gps_seconds = np.asarray(
        [
            (header.gps_week * GPS_WEEK_SECONDS)
            + header.seconds_of_week
            + (record.epoch - header.start_epoch).total_seconds()
            for record in materialized
        ],
        dtype=np.float64,
    )
    position_m = np.asarray(
        [
            [component * 1_000.0 for component in record.position_km]
            for record in materialized
        ],
        dtype=np.float64,
    )
    clock_offset_s = np.asarray(
        [
            np.nan
            if record.clock_offset_microseconds is None
            else record.clock_offset_microseconds * 1.0e-6
            for record in materialized
        ],
        dtype=np.float64,
    )

    if not np.all(np.diff(gps_seconds) > 0.0):
        raise ValueError("GPS seconds must be strictly increasing")
    if not np.all(np.isfinite(position_m)):
        raise ValueError("positions must be finite")
    return gps_seconds, position_m, clock_offset_s
####


__all__ = ["records_to_si_arrays", "select_satellite_records"]
