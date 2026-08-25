from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..contracts import DatasetManifest, NormalizedTrack
from ..quality import TrackQualityPolicy


class UnknownLabelPolicy(StrEnum):
    ERROR = "error"
    SKIP = "skip"
    PRESERVE = "preserve"
####


class InvalidRowPolicy(StrEnum):
    ERROR = "error"
    SKIP = "skip"
####


class DuplicateTimestampPolicy(StrEnum):
    ERROR = "error"
    KEEP_FIRST = "keep_first"
    KEEP_LAST = "keep_last"
####


class TgsimFoggyBottomAdapterConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    accessed_on: date = Field(default_factory=date.today)
    recording_id: str = "foggy_bottom_2023-05-04"
    location_id: str = "foggy_bottom_washington_dc"
    source_asset_id: str = "trajectory_csv"
    minimum_samples: int = Field(default=2, ge=2)
    minimum_heading_speed_mps: float = Field(default=0.05, gt=0.0)
    unknown_label_policy: UnknownLabelPolicy = UnknownLabelPolicy.ERROR
    invalid_row_policy: InvalidRowPolicy = InvalidRowPolicy.ERROR
    duplicate_timestamp_policy: DuplicateTimestampPolicy = DuplicateTimestampPolicy.ERROR
    require_consistent_track_labels: bool = True
    quality_policy: TrackQualityPolicy = Field(default_factory=TrackQualityPolicy)
####


class TgsimLabelCount(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    native_label: str
    normalized_class: str
    track_count: int = Field(ge=0)
####


class TgsimParseSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    rows_read: int = Field(ge=0)
    rows_skipped_invalid: int = Field(ge=0)
    track_groups_seen: int = Field(ge=0)
    tracks_loaded: int = Field(ge=0)
    tracks_skipped_short: int = Field(ge=0)
    tracks_skipped_unknown_label: int = Field(ge=0)
    duplicate_timestamps_resolved: int = Field(ge=0)
    inconsistent_label_tracks: int = Field(ge=0)
    label_counts: tuple[TgsimLabelCount, ...] = Field(default_factory=tuple)
####


class TgsimLoadResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    manifest: DatasetManifest
    tracks: tuple[NormalizedTrack, ...]
    summary: TgsimParseSummary

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        if len(self.tracks) != self.summary.tracks_loaded:
            raise ValueError("summary tracks_loaded must match the returned tracks")
        return self
    ####
####


@dataclass(frozen=True, slots=True)
class _TgsimRow:
    source_row_number: int
    track_id: str
    run_id: str
    time_s: float
    x_m: float
    y_m: float
    lane_id: float
    velocity_x_mps: float
    velocity_y_mps: float
    acceleration_x_mps2: float
    acceleration_y_mps2: float
    length_m: float
    width_m: float
    native_label: str
####


__all__ = [
    "DuplicateTimestampPolicy",
    "InvalidRowPolicy",
    "TgsimFoggyBottomAdapterConfig",
    "TgsimLabelCount",
    "TgsimLoadResult",
    "TgsimParseSummary",
    "UnknownLabelPolicy",
]
