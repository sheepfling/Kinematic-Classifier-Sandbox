from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Protocol, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from .contracts import DatasetManifest, NormalizedTrack


ScalarMetadataValue = str | int | float | bool | None
CORPUS_TRAJECTORY_CONTRACT_VERSION = "1.0.0"


class PhysicalDomain(StrEnum):
    LAND = "land"
    SEA = "sea"
    AIR = "air"
    SPACE = "space"
####


class TimeBasis(StrEnum):
    RELATIVE_SECONDS = "relative_seconds"
    UNIX_UTC_SECONDS = "unix_utc_seconds"
    GPS_SECONDS = "gps_seconds"
    SOURCE_NATIVE_SECONDS = "source_native_seconds"
####


class CoordinateFrameKind(StrEnum):
    LOCAL_CARTESIAN = "local_cartesian"
    ENU = "enu"
    NED = "ned"
    ECEF = "ecef"
    ECI = "eci"
    PROJECTED_GEODETIC = "projected_geodetic"
    OTHER_CARTESIAN = "other_cartesian"
####


class ObservationModality(StrEnum):
    GNSS = "gnss"
    ADS_B = "ads_b"
    AIS = "ais"
    RADAR = "radar"
    OPTICAL_TRACKING = "optical_tracking"
    INERTIAL = "inertial"
    ODOMETRY = "odometry"
    TELEMETRY = "telemetry"
    MULTI_SENSOR_FUSION = "multi_sensor_fusion"
    OTHER = "other"
####


class CoordinateFrameMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    frame_id: str = Field(min_length=1)
    kind: CoordinateFrameKind
    axes_description: str = Field(min_length=1)
    origin_description: str | None = None
    authority: str | None = None
    epoch: str | None = None
    vertical_datum: str | None = None
    notes: tuple[str, ...] = Field(default_factory=tuple)
####


class CorpusDatasetMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    contract_version: str = CORPUS_TRAJECTORY_CONTRACT_VERSION
    dataset_manifest: DatasetManifest
    domains: tuple[PhysicalDomain, ...]
    observation_modalities: tuple[ObservationModality, ...]
    native_coordinate_frame: str = Field(min_length=1)
    canonical_frame: CoordinateFrameMetadata
    time_basis: TimeBasis
    source_type: str = Field(min_length=1)
    extensions: Mapping[str, ScalarMetadataValue] = Field(default_factory=dict)

    @field_validator("domains")
    @classmethod
    def validate_domains(cls, value: tuple[PhysicalDomain, ...]) -> tuple[PhysicalDomain, ...]:
        if not value:
            raise ValueError("a corpus dataset must declare at least one physical domain")
        if len(value) != len(set(value)):
            raise ValueError("physical domains must be unique")
        return value
    ####

    @field_validator("observation_modalities")
    @classmethod
    def validate_modalities(
        cls,
        value: tuple[ObservationModality, ...],
    ) -> tuple[ObservationModality, ...]:
        if not value:
            raise ValueError("a corpus dataset must declare at least one observation modality")
        if len(value) != len(set(value)):
            raise ValueError("observation modalities must be unique")
        return value
    ####

    @field_validator("extensions", mode="after")
    @classmethod
    def freeze_extensions(
        cls,
        value: Mapping[str, ScalarMetadataValue],
    ) -> Mapping[str, ScalarMetadataValue]:
        return MappingProxyType(dict(value))
    ####

    @field_serializer("extensions")
    def serialize_extensions(
        self,
        value: Mapping[str, ScalarMetadataValue],
    ) -> dict[str, ScalarMetadataValue]:
        return dict(value)
    ####

    @model_validator(mode="after")
    def validate_frame(self) -> Self:
        if self.canonical_frame.frame_id != self.dataset_manifest.coordinate_frame:
            raise ValueError(
                "canonical_frame.frame_id must match dataset_manifest.coordinate_frame"
            )
        return self
    ####
####


class CorpusTrajectoryMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    trajectory_id: str = Field(min_length=1)
    domain: PhysicalDomain
    time_basis: TimeBasis
    frame: CoordinateFrameMetadata
    observation_modalities: tuple[ObservationModality, ...]
    subject_id: str | None = None
    platform_type: str | None = None
    platform_subtype: str | None = None
    source_metadata: Mapping[str, ScalarMetadataValue] = Field(default_factory=dict)
    domain_extensions: Mapping[str, ScalarMetadataValue] = Field(default_factory=dict)

    @field_validator("observation_modalities")
    @classmethod
    def validate_modalities(
        cls,
        value: tuple[ObservationModality, ...],
    ) -> tuple[ObservationModality, ...]:
        if not value:
            raise ValueError("a corpus trajectory must declare at least one observation modality")
        if len(value) != len(set(value)):
            raise ValueError("observation modalities must be unique")
        return value
    ####

    @field_validator("source_metadata", "domain_extensions", mode="after")
    @classmethod
    def freeze_metadata(
        cls,
        value: Mapping[str, ScalarMetadataValue],
    ) -> Mapping[str, ScalarMetadataValue]:
        return MappingProxyType(dict(value))
    ####

    @field_serializer("source_metadata", "domain_extensions")
    def serialize_metadata(
        self,
        value: Mapping[str, ScalarMetadataValue],
    ) -> dict[str, ScalarMetadataValue]:
        return dict(value)
    ####
####


class CorpusTrajectory(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    dataset: CorpusDatasetMetadata
    metadata: CorpusTrajectoryMetadata
    trajectory: NormalizedTrack

    @model_validator(mode="after")
    def validate_cross_contract_identity(self) -> Self:
        if self.trajectory.provenance.dataset_id != self.dataset.dataset_manifest.dataset_id:
            raise ValueError("trajectory dataset_id must match corpus dataset manifest")
        if self.metadata.domain not in self.dataset.domains:
            raise ValueError("trajectory domain must be declared by corpus dataset metadata")
        if self.metadata.time_basis is not self.dataset.time_basis:
            raise ValueError("trajectory time basis must match corpus dataset metadata")
        if self.metadata.frame.frame_id != self.trajectory.coordinate_frame:
            raise ValueError("trajectory coordinate frame must match trajectory metadata frame")
        if self.metadata.frame.frame_id != self.dataset.canonical_frame.frame_id:
            raise ValueError("trajectory coordinate frame must match dataset canonical frame")
        dataset_modalities = set(self.dataset.observation_modalities)
        if not set(self.metadata.observation_modalities).issubset(dataset_modalities):
            raise ValueError(
                "trajectory observation modalities must be declared by the dataset"
            )
        return self
    ####
####


class RealWorldCorpusAdapter(Protocol):
    adapter_id: str
    adapter_version: str

    @property
    def corpus_metadata(self) -> CorpusDatasetMetadata: ...

    def load_corpus(self, path: str | Path) -> tuple[CorpusTrajectory, ...]: ...
####


__all__ = [
    "CORPUS_TRAJECTORY_CONTRACT_VERSION",
    "CoordinateFrameKind",
    "CoordinateFrameMetadata",
    "CorpusDatasetMetadata",
    "CorpusTrajectory",
    "CorpusTrajectoryMetadata",
    "ObservationModality",
    "PhysicalDomain",
    "RealWorldCorpusAdapter",
    "ScalarMetadataValue",
    "TimeBasis",
]
