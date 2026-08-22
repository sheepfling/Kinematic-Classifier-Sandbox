from __future__ import annotations

from datetime import date
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Mapping, Self, TypeAlias

import numpy as np
from numpy.typing import NDArray
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    WithJsonSchema,
    field_serializer,
    field_validator,
    model_validator,
)


FloatArray = NDArray[np.float64]


def _freeze_float_array_1d(value: object) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("expected a one-dimensional numeric array")
    if not np.all(np.isfinite(array)):
        raise ValueError("numeric arrays must contain only finite values")
    frozen = np.array(array, dtype=np.float64, copy=True)
    frozen.setflags(write=False)
    return frozen
####


def _freeze_float_array_nx3(value: object) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("expected an N x 3 numeric array")
    if not np.all(np.isfinite(array)):
        raise ValueError("numeric arrays must contain only finite values")
    frozen = np.array(array, dtype=np.float64, copy=True)
    frozen.setflags(write=False)
    return frozen
####


def _serialize_float_array(value: FloatArray) -> list[float] | list[list[float]]:
    serialized = value.tolist()
    if value.ndim == 1:
        return [float(item) for item in serialized]
    return [[float(item) for item in row] for row in serialized]
####


FloatArray1D: TypeAlias = Annotated[
    FloatArray,
    BeforeValidator(_freeze_float_array_1d),
    PlainSerializer(_serialize_float_array, return_type=list[float]),
    WithJsonSchema({"type": "array", "items": {"type": "number"}}),
]

FloatArrayNx3: TypeAlias = Annotated[
    FloatArray,
    BeforeValidator(_freeze_float_array_nx3),
    PlainSerializer(_serialize_float_array, return_type=list[list[float]]),
    WithJsonSchema(
        {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 3,
                "maxItems": 3,
            },
        }
    ),
]


class LabelEvidence(StrEnum):
    NATIVE = "native"
    DERIVED = "derived"
    PROXY = "proxy"
    WEAK = "weak"
####


class ChannelRole(StrEnum):
    SOURCE = "source"
    DERIVED = "derived"
    CONTEXT = "context"
    AUDIT_ONLY = "audit_only"
####


class SourceAsset(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    asset_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    download_url: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")
    required: bool = True
####


class DatasetManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    version: str = Field(min_length=1)
    publisher: str = Field(min_length=1)
    citation: str = Field(min_length=1)
    doi: str | None = None
    license_id: str = Field(min_length=1)
    license_url: str = Field(min_length=1)
    landing_page_url: str = Field(min_length=1)
    accessed_on: date
    adapter_id: str = Field(min_length=1)
    adapter_version: str = Field(min_length=1)
    coordinate_frame: str = Field(min_length=1)
    nominal_sample_interval_s: float = Field(gt=0.0)
    source_assets: tuple[SourceAsset, ...]
    notes: tuple[str, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def validate_assets(self) -> Self:
        asset_ids = tuple(asset.asset_id for asset in self.source_assets)
        if not asset_ids:
            raise ValueError("a dataset manifest must declare at least one source asset")
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("source asset IDs must be unique")
        return self
    ####
####


class TrackProvenance(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str = Field(min_length=1)
    source_asset_id: str = Field(min_length=1)
    recording_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    track_id: str = Field(min_length=1)
    location_id: str = Field(min_length=1)
    split_group_id: str = Field(min_length=1)
    source_row_start: int | None = Field(default=None, ge=1)
    source_row_end: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_row_bounds(self) -> Self:
        if (
            self.source_row_start is not None
            and self.source_row_end is not None
            and self.source_row_end < self.source_row_start
        ):
            raise ValueError("source_row_end must not precede source_row_start")
        return self
    ####
####


class TrackLabels(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    native_label: str = Field(min_length=1)
    normalized_class: str = Field(min_length=1)
    mobility_family: str = Field(min_length=1)
    operating_domain: str = Field(min_length=1)
    evidence: LabelEvidence
    is_proxy: bool = False
    notes: tuple[str, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def validate_proxy_marker(self) -> Self:
        if self.evidence is LabelEvidence.PROXY and not self.is_proxy:
            raise ValueError("proxy evidence must set is_proxy=True")
        if self.is_proxy and self.evidence is not LabelEvidence.PROXY:
            raise ValueError("is_proxy=True requires proxy evidence")
        return self
    ####
####


class NumericChannel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    name: str = Field(min_length=1)
    units: str | None = None
    role: ChannelRole
    values: FloatArray1D
####


class CategoricalChannel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    role: ChannelRole
    values: tuple[str, ...]
####


class TrackQuality(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_count: int = Field(ge=2)
    duration_s: float = Field(gt=0.0)
    median_dt_s: float = Field(gt=0.0)
    max_dt_s: float = Field(gt=0.0)
    gap_count: int = Field(ge=0)
    max_position_step_m: float = Field(ge=0.0)
    source_velocity_rmse_mps: float | None = Field(default=None, ge=0.0)
    source_acceleration_rmse_mps2: float | None = Field(default=None, ge=0.0)
    findings: tuple[str, ...] = Field(default_factory=tuple)
####


class NormalizedTrack(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    provenance: TrackProvenance
    labels: TrackLabels
    coordinate_frame: str = Field(min_length=1)
    timestamps_s: FloatArray1D
    position_m: FloatArrayNx3
    derived_velocity_mps: FloatArrayNx3
    derived_acceleration_mps2: FloatArrayNx3
    source_velocity_mps: FloatArrayNx3 | None = None
    source_acceleration_mps2: FloatArrayNx3 | None = None
    numeric_channels: tuple[NumericChannel, ...] = Field(default_factory=tuple)
    categorical_channels: tuple[CategoricalChannel, ...] = Field(default_factory=tuple)
    quality: TrackQuality | None = None
    metadata: Mapping[str, str | int | float | bool | None] = Field(default_factory=dict)

    @field_validator("metadata", mode="after")
    @classmethod
    def freeze_metadata(
        cls,
        value: Mapping[str, str | int | float | bool | None],
    ) -> Mapping[str, str | int | float | bool | None]:
        return MappingProxyType(dict(value))
    ####

    @field_serializer("metadata")
    def serialize_metadata(
        self,
        value: Mapping[str, str | int | float | bool | None],
    ) -> dict[str, str | int | float | bool | None]:
        return dict(value)
    ####

    @model_validator(mode="after")
    def validate_track_shape(self) -> Self:
        sample_count = int(self.timestamps_s.shape[0])
        if sample_count < 2:
            raise ValueError("normalized tracks require at least two samples")
        if not np.all(np.diff(self.timestamps_s) > 0.0):
            raise ValueError("timestamps_s must be strictly increasing")

        vector_fields = (
            ("position_m", self.position_m),
            ("derived_velocity_mps", self.derived_velocity_mps),
            ("derived_acceleration_mps2", self.derived_acceleration_mps2),
            ("source_velocity_mps", self.source_velocity_mps),
            ("source_acceleration_mps2", self.source_acceleration_mps2),
        )
        for field_name, values in vector_fields:
            if values is not None and values.shape[0] != sample_count:
                raise ValueError(f"{field_name} must have one row per timestamp")

        channel_names: list[str] = []
        for channel in self.numeric_channels:
            if channel.values.shape[0] != sample_count:
                raise ValueError(f"numeric channel {channel.name!r} has the wrong sample count")
            channel_names.append(channel.name)
        for channel in self.categorical_channels:
            if len(channel.values) != sample_count:
                raise ValueError(f"categorical channel {channel.name!r} has the wrong sample count")
            channel_names.append(channel.name)
        if len(channel_names) != len(set(channel_names)):
            raise ValueError("channel names must be unique")

        if self.quality is not None and self.quality.sample_count != sample_count:
            raise ValueError("quality sample_count must match the track")
        return self
    ####
####


__all__ = [
    "CategoricalChannel",
    "ChannelRole",
    "DatasetManifest",
    "FloatArray",
    "FloatArray1D",
    "FloatArrayNx3",
    "LabelEvidence",
    "NormalizedTrack",
    "NumericChannel",
    "SourceAsset",
    "TrackLabels",
    "TrackProvenance",
    "TrackQuality",
]
