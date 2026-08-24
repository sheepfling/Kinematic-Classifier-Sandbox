from __future__ import annotations

from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


TRAJECTORY_EPISODE_CONTRACT_VERSION = "trajectory-corpus-v0.1"


class StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
####


class ProgramDomain(StrEnum):
    LAND = "land"
    SEA = "sea"
    AIR = "air"
    SPACE = "space"
####


class StateViewKind(StrEnum):
    SOURCE_NATIVE = "source_native"
    REFERENCE = "reference"
    ANALYSIS = "analysis"
    SENSOR_TRACK = "sensor_track"
    SIMULATION_TRUTH = "simulation_truth"
    SYNTHETIC_OBSERVATION = "synthetic_observation"
####


class StateRole(StrEnum):
    OBSERVATION = "observation"
    ESTIMATE = "estimate"
    RECONSTRUCTION = "reconstruction"
    REFERENCE = "reference"
    PROPAGATED = "propagated"
    SIMULATION_TRUTH = "simulation_truth"
    SYNTHETIC_OBSERVATION = "synthetic_observation"
####


class ValueBasis(StrEnum):
    MEASURED = "measured"
    REPORTED = "reported"
    DERIVED = "derived"
    INTERPOLATED = "interpolated"
    DEAD_RECKONED = "dead_reckoned"
    POSTPROCESSED = "postprocessed"
    PROPAGATED = "propagated"
    SIMULATED = "simulated"
    UNKNOWN = "unknown"
####


class AccessClass(StrEnum):
    CLASSIFIER_CANDIDATE = "classifier_candidate"
    CONTEXT = "context"
    AUDIT_ONLY = "audit_only"
    IDENTITY_GROUPING_ONLY = "identity_grouping_only"
    RESTRICTED = "restricted"
####


class EvidenceStrength(StrEnum):
    WEAK = "weak"
    MEDIUM = "medium"
    STRONG = "strong"
####


class LabelEvidenceKind(StrEnum):
    NATIVE = "native"
    RECONCILED = "reconciled"
    DERIVED = "derived"
    HUMAN_VERIFIED = "human_verified"
    PROXY = "proxy"
####


class GroupingNamespace(StrEnum):
    PHYSICAL_PLATFORM = "physical_platform"
    MISSION_EVENT = "mission_event"
    SOURCE_RECORDING = "source_recording"
    ROUTE = "route"
    SOURCE_DATASET = "source_dataset"
    GEOGRAPHY = "geography"
    TEMPORAL_COLLECTION = "temporal_collection"
    OPERATOR_OR_FLEET = "operator_or_fleet"
####


class QualitySeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
####


class AssetReference(StrictFrozenModel):
    path: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path == PurePosixPath("."):
            raise ValueError("asset paths must identify a file")
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("asset paths must be relative and may not traverse parents")
        return value
    ####
####


class FrameDescriptor(StrictFrozenModel):
    frame_id: str = Field(min_length=1)
    frame_kind: str = Field(min_length=1)
    axes: tuple[str, ...] = Field(min_length=1)
    axis_units: tuple[str, ...] = Field(min_length=1)
    center_or_origin: str | dict[str, float]
    vertical_reference: str = Field(min_length=1)
    vertical_positive_direction: str = Field(min_length=1)
    crs_or_datum: str | None = None
    reference_epoch: str | None = None
    local_origin: dict[str, float] | None = None
    earth_orientation_or_transform_model: str | None = None
    body_attitude_convention: str | None = None
    orbital_local_definition: str | None = None

    @model_validator(mode="after")
    def validate_axes_and_units(self) -> Self:
        if len(self.axes) != len(self.axis_units):
            raise ValueError("frame axes and axis_units must have equal length")
        if len(set(self.axes)) != len(self.axes):
            raise ValueError("frame axes must be unique")
        return self
    ####
####


class TimeAxisDescriptor(StrictFrozenModel):
    source_time_system: str = Field(min_length=1)
    normalized_time_system: str | None = None
    absolute_time_available: bool
    source_epoch_or_reference: str | None = None
    elapsed_origin: str = Field(min_length=1)
    precision_or_resolution: str = Field(min_length=1)
    rollover_policy: str = Field(min_length=1)
    leap_second_policy: str = Field(min_length=1)
####


class ChannelDescriptor(StrictFrozenModel):
    channel_id: str = Field(min_length=1)
    semantic_role: str = Field(min_length=1)
    component_names: tuple[str, ...] = Field(min_length=1)
    units: tuple[str, ...] = Field(min_length=1)
    frame_id: str | None = None
    state_role: StateRole
    value_basis: ValueBasis
    access_class: AccessClass
    source_fields: tuple[str, ...] = Field(default_factory=tuple)
    lineage_step_ids: tuple[str, ...] = Field(default_factory=tuple)
    uncertainty_reference: str | None = None
    validity_reference: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def validate_components_and_units(self) -> Self:
        if len(self.component_names) != len(self.units):
            raise ValueError("channel component_names and units must have equal length")
        if len(set(self.component_names)) != len(self.component_names):
            raise ValueError("channel component names must be unique")
        return self
    ####
####


class TrajectoryStateViewManifest(StrictFrozenModel):
    state_view_id: str = Field(min_length=1)
    view_kind: StateViewKind
    state_role: StateRole
    value_basis: ValueBasis
    frame: FrameDescriptor
    source_time_axis: TimeAxisDescriptor
    sample_count: int = Field(gt=0)
    sample_asset: AssetReference
    channel_descriptors: tuple[ChannelDescriptor, ...] = Field(min_length=1)
    normalization_assumptions: tuple[str, ...] = Field(default_factory=tuple)
    processing_step_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_channel_ids(self) -> Self:
        channel_ids = tuple(channel.channel_id for channel in self.channel_descriptors)
        if len(channel_ids) != len(set(channel_ids)):
            raise ValueError("state-view channel IDs must be unique")
        if len(self.processing_step_ids) != len(set(self.processing_step_ids)):
            raise ValueError("state-view processing step IDs must be unique")
        for channel in self.channel_descriptors:
            if channel.frame_id is not None and channel.frame_id != self.frame.frame_id:
                raise ValueError("channel frame_id must match the owning state view")
        return self
    ####
####


class TrajectorySegment(StrictFrozenModel):
    segment_id: str = Field(min_length=1)
    start_offset_s: float = Field(ge=0.0)
    end_offset_s: float = Field(ge=0.0)
    operating_environment: str = Field(min_length=1)
    motion_regime: str = Field(min_length=1)
    evidence_kind: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if self.end_offset_s < self.start_offset_s:
            raise ValueError("segment end_offset_s must not precede start_offset_s")
        return self
    ####
####


class LabelAssertion(StrictFrozenModel):
    assertion_id: str = Field(min_length=1)
    namespace: str = Field(min_length=1)
    value: str = Field(min_length=1)
    evidence_kind: LabelEvidenceKind
    evidence_strength: EvidenceStrength
    source_reference: str = Field(min_length=1)
    proxy: bool
    start_offset_s: float | None = Field(default=None, ge=0.0)
    end_offset_s: float | None = Field(default=None, ge=0.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    dependency_channel_ids: tuple[str, ...] = Field(default_factory=tuple)
    vocabulary_version: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def validate_assertion(self) -> Self:
        if self.end_offset_s is not None and self.start_offset_s is None:
            raise ValueError("end_offset_s requires start_offset_s")
        if (
            self.start_offset_s is not None
            and self.end_offset_s is not None
            and self.end_offset_s < self.start_offset_s
        ):
            raise ValueError("label assertion end_offset_s must not precede start_offset_s")
        if self.evidence_kind is LabelEvidenceKind.PROXY and not self.proxy:
            raise ValueError("proxy evidence must set proxy=True")
        if self.proxy and self.evidence_kind is not LabelEvidenceKind.PROXY:
            raise ValueError("proxy=True requires proxy evidence")
        return self
    ####
####


class GroupingKey(StrictFrozenModel):
    namespace: GroupingNamespace
    opaque_value: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    evidence_strength: EvidenceStrength
    access_class: AccessClass = AccessClass.IDENTITY_GROUPING_ONLY

    @model_validator(mode="after")
    def validate_access_class(self) -> Self:
        if self.access_class is not AccessClass.IDENTITY_GROUPING_ONLY:
            raise ValueError("grouping keys must remain identity_grouping_only")
        return self
    ####
####


class QualityFinding(StrictFrozenModel):
    code: str = Field(min_length=1)
    severity: QualitySeverity
    message: str = Field(min_length=1)
    metric: str | None = None
    value: float | int | str | None = None
    unit: str | None = None
    start_offset_s: float | None = Field(default=None, ge=0.0)
    end_offset_s: float | None = Field(default=None, ge=0.0)
    source_reference: str | None = None

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if self.end_offset_s is not None and self.start_offset_s is None:
            raise ValueError("quality finding end_offset_s requires start_offset_s")
        if (
            self.start_offset_s is not None
            and self.end_offset_s is not None
            and self.end_offset_s < self.start_offset_s
        ):
            raise ValueError("quality finding end_offset_s must not precede start_offset_s")
        return self
    ####
####


class QualitySummary(StrictFrozenModel):
    disposition: str = Field(min_length=1)
    sample_count: int = Field(gt=0)
    duration_s: float = Field(ge=0.0)
    median_sample_interval_s: float = Field(ge=0.0)
    maximum_gap_s: float = Field(ge=0.0)
    duplicate_timestamp_count: int = Field(ge=0)
    out_of_order_timestamp_count: int = Field(ge=0)
    findings: tuple[QualityFinding, ...] = Field(default_factory=tuple)
####


class DomainExtension(StrictFrozenModel):
    schema_id: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    payload: dict[str, Any]
####


class ClassifierTrajectoryView(StrictFrozenModel):
    episode_id: str = Field(min_length=1)
    state_view_id: str = Field(min_length=1)
    asset: AssetReference
    sample_count: int = Field(gt=0)
    frame_id: str = Field(min_length=1)
    permitted_extra_channels: tuple[str, ...] = Field(default_factory=tuple)
    processing_step_ids: tuple[str, ...] = Field(min_length=1)
    target_labels_stored_outside_asset: bool
    identity_and_grouping_values_excluded: bool

    @model_validator(mode="after")
    def validate_leakage_boundary(self) -> Self:
        if not self.target_labels_stored_outside_asset:
            raise ValueError("classifier assets must not contain target labels")
        if not self.identity_and_grouping_values_excluded:
            raise ValueError("classifier assets must exclude identity and grouping values")
        if len(self.permitted_extra_channels) != len(set(self.permitted_extra_channels)):
            raise ValueError("classifier permitted_extra_channels must be unique")
        if len(self.processing_step_ids) != len(set(self.processing_step_ids)):
            raise ValueError("classifier processing step IDs must be unique")
        return self
    ####
####


class TrajectoryEpisodeManifest(StrictFrozenModel):
    schema_version: str = TRAJECTORY_EPISODE_CONTRACT_VERSION
    corpus_snapshot_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    primary_program_domain: ProgramDomain
    corpus_sublane: str = Field(min_length=1)
    default_operating_environment: str = Field(min_length=1)
    default_motion_regime: str = Field(min_length=1)
    source_dataset_id: str = Field(min_length=1)
    source_artifact_ids: tuple[str, ...] = Field(min_length=1)
    observation_modality: str = Field(min_length=1)
    platform_group_id: str | None = None
    mission_id: str | None = None
    object_id: str | None = None
    parent_episode_id: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    state_views: tuple[TrajectoryStateViewManifest, ...] = Field(min_length=1)
    segments: tuple[TrajectorySegment, ...] = Field(default_factory=tuple)
    labels: tuple[LabelAssertion, ...] = Field(default_factory=tuple)
    grouping_keys: tuple[GroupingKey, ...] = Field(min_length=1)
    quality_summary: QualitySummary
    processing_step_ids: tuple[str, ...] = Field(min_length=1)
    domain_extension: DomainExtension | None = None
    classifier_trajectory_view: ClassifierTrajectoryView | None = None

    @model_validator(mode="after")
    def validate_internal_references(self) -> Self:
        if self.schema_version != TRAJECTORY_EPISODE_CONTRACT_VERSION:
            raise ValueError("unsupported trajectory episode contract version")

        if len(self.source_artifact_ids) != len(set(self.source_artifact_ids)):
            raise ValueError("source artifact IDs must be unique")
        if len(self.processing_step_ids) != len(set(self.processing_step_ids)):
            raise ValueError("episode processing step IDs must be unique")

        state_view_ids = tuple(view.state_view_id for view in self.state_views)
        if len(state_view_ids) != len(set(state_view_ids)):
            raise ValueError("state view IDs must be unique")
        state_views_by_id = {view.state_view_id: view for view in self.state_views}

        root_steps = set(self.processing_step_ids)
        channel_ids: set[str] = set()
        for view in self.state_views:
            if not set(view.processing_step_ids).issubset(root_steps):
                raise ValueError("state-view processing steps must be declared by the episode")
            for channel in view.channel_descriptors:
                channel_ids.add(channel.channel_id)
                if not set(channel.lineage_step_ids).issubset(root_steps):
                    raise ValueError("channel lineage steps must be declared by the episode")

        assertion_ids = tuple(label.assertion_id for label in self.labels)
        if len(assertion_ids) != len(set(assertion_ids)):
            raise ValueError("label assertion IDs must be unique")
        for label in self.labels:
            if not set(label.dependency_channel_ids).issubset(channel_ids):
                raise ValueError("label dependency channels must exist in an episode state view")

        classifier = self.classifier_trajectory_view
        if classifier is not None:
            if classifier.episode_id != self.episode_id:
                raise ValueError("classifier view episode_id must match episode_id")
            if classifier.state_view_id not in state_view_ids:
                raise ValueError("classifier view must reference an episode state view")
            if not set(classifier.processing_step_ids).issubset(root_steps):
                raise ValueError("classifier processing steps must be declared by the episode")
            referenced_view = state_views_by_id[classifier.state_view_id]
            if classifier.frame_id != referenced_view.frame.frame_id:
                raise ValueError("classifier frame_id must match the referenced state view")
            if classifier.sample_count > referenced_view.sample_count:
                raise ValueError("classifier sample_count must not exceed the referenced state view")

        grouping_pairs = tuple(
            (key.namespace, key.opaque_value) for key in self.grouping_keys
        )
        if len(grouping_pairs) != len(set(grouping_pairs)):
            raise ValueError("grouping keys must be unique by namespace and value")

        split_capable = {
            GroupingNamespace.PHYSICAL_PLATFORM,
            GroupingNamespace.SOURCE_RECORDING,
            GroupingNamespace.MISSION_EVENT,
        }
        if not any(key.namespace in split_capable for key in self.grouping_keys):
            raise ValueError("episode requires at least one split-capable grouping key")

        if self.platform_group_id is not None:
            platform_values = {
                key.opaque_value
                for key in self.grouping_keys
                if key.namespace is GroupingNamespace.PHYSICAL_PLATFORM
            }
            if self.platform_group_id not in platform_values:
                raise ValueError("platform_group_id must match a physical-platform grouping key")

        source_native_counts = [
            view.sample_count
            for view in self.state_views
            if view.view_kind is StateViewKind.SOURCE_NATIVE
        ]
        if source_native_counts and self.quality_summary.sample_count not in source_native_counts:
            raise ValueError("quality sample_count must match a source-native state view")
        return self
    ####
####


__all__ = [
    "AccessClass",
    "AssetReference",
    "ChannelDescriptor",
    "ClassifierTrajectoryView",
    "DomainExtension",
    "EvidenceStrength",
    "FrameDescriptor",
    "GroupingKey",
    "GroupingNamespace",
    "LabelAssertion",
    "LabelEvidenceKind",
    "ProgramDomain",
    "QualityFinding",
    "QualitySeverity",
    "QualitySummary",
    "StateRole",
    "StateViewKind",
    "TRAJECTORY_EPISODE_CONTRACT_VERSION",
    "TimeAxisDescriptor",
    "TrajectoryEpisodeManifest",
    "TrajectorySegment",
    "TrajectoryStateViewManifest",
    "ValueBasis",
]
