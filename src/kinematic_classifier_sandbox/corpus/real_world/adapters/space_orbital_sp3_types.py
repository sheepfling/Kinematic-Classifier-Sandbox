"""Typed SP3-c position-product records and constants."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Final

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
GPS_WEEK_SECONDS: Final[float] = 604_800.0
GPS_EPOCH: Final[datetime] = datetime(1980, 1, 6)
SP3_MISSING_CLOCK_ABS_MIN: Final[float] = 999_999.0
SP3_NONSTANDARD_POSITION_SENTINEL: Final[float] = 999_999.999_999


@dataclass(frozen=True, slots=True)
class Sp3Header:
    version: str
    data_type: str
    start_epoch: datetime
    declared_epoch_count: int
    coordinate_system: str
    orbit_type: str
    agency: str
    gps_week: int
    seconds_of_week: float
    sampling_period_s: float
    time_system: str
    declared_satellite_count: int
    satellite_ids: tuple[str, ...]
    comments: tuple[str, ...]
####


@dataclass(frozen=True, slots=True)
class Sp3PositionRecord:
    epoch: datetime
    satellite_id: str
    position_km: tuple[float, float, float]
    clock_offset_microseconds: float | None
####


@dataclass(frozen=True, slots=True)
class Sp3Extract:
    header: Sp3Header
    records: tuple[Sp3PositionRecord, ...]
####


__all__ = [
    "FloatArray",
    "GPS_EPOCH",
    "GPS_WEEK_SECONDS",
    "SP3_MISSING_CLOCK_ABS_MIN",
    "SP3_NONSTANDARD_POSITION_SENTINEL",
    "Sp3Extract",
    "Sp3Header",
    "Sp3PositionRecord",
]
