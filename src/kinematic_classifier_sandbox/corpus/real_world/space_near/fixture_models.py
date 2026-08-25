from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


JsonScalar: TypeAlias = str | int | float | bool | None


class SpaceNearSourceSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_dataset_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    publisher: str = Field(min_length=1)
    citation: str = Field(min_length=1)
    version: str = Field(min_length=1)
    license_id: str = Field(min_length=1)
    license_url: str = Field(min_length=1)
    landing_page_url: str = Field(min_length=1)
    accessed_on: date
    source_asset_id: str = Field(min_length=1)
    source_asset_title: str = Field(min_length=1)
    source_asset_url: str = Field(min_length=1)
    source_asset_media_type: str = Field(min_length=1)
    source_asset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    nominal_sample_interval_s: float = Field(gt=0.0)
    platform_subtype: str = Field(min_length=1)
    portfolio_role: str = Field(min_length=1)
    launch_region_id: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    claim_boundary: str = Field(min_length=1)
####


class EmbeddedStateView(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    state_view_id: str = Field(min_length=1)
    view_kind: str = Field(min_length=1)
    state_role: str = Field(min_length=1)
    channel_ids: tuple[str, ...]
    processing_step_ids: tuple[str, ...]
    columns: tuple[str, ...]
    rows: tuple[tuple[JsonScalar, ...], ...]
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_rows(self) -> Self:
        if len(self.rows) < 2:
            raise ValueError("embedded state views require at least two rows")
        if len(self.columns) != len(set(self.columns)):
            raise ValueError("embedded state-view columns must be unique")
        if any(len(row) != len(self.columns) for row in self.rows):
            raise ValueError("embedded state-view rows must match the declared columns")
        return self
    ####
####


class AnalysisFrame(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    frame_id: str = Field(min_length=1)
    frame_kind: Literal["cartesian_earth_fixed"]
    axes: tuple[str, str, str]
    axis_units: tuple[Literal["m"], Literal["m"], Literal["m"]]
    crs_or_datum: str | None = None
    unresolved_ambiguities: tuple[str, ...] = ()
####


class GroupingKey(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    namespace: str = Field(min_length=1)
    opaque_value: str = Field(min_length=1)
    access_class: Literal["identity_grouping_only"]
####


class LabelAssertion(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    assertion_id: str = Field(min_length=1)
    namespace: str = Field(min_length=1)
    value: str = Field(min_length=1)
    dependency_channel_ids: tuple[str, ...] = ()
    start_offset_s: float
    end_offset_s: float

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if self.end_offset_s < self.start_offset_s:
            raise ValueError("label assertion interval is reversed")
        return self
    ####
####


class ProcessingStep(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    step_id: str | None = None
    processing_step_id: str | None = None

    @model_validator(mode="after")
    def validate_identifier(self) -> Self:
        if not (self.step_id or self.processing_step_id):
            raise ValueError("processing step requires an identifier")
        return self
    ####

    @property
    def identifier(self) -> str:
        return self.step_id or self.processing_step_id or ""
    ####
####


class EmbeddedEpisode(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["trajectory-corpus-v0.1"]
    episode_id: str = Field(min_length=1)
    mission_id: str = Field(min_length=1)
    object_id: str = Field(min_length=1)
    primary_program_domain: Literal["space"]
    corpus_sublane: Literal["space_near"]
    default_motion_regime: str = Field(min_length=1)
    source_dataset_id: str = Field(min_length=1)
    observation_modality: str = Field(min_length=1)
    absolute_start_time: str = Field(min_length=1)
    absolute_end_time: str = Field(min_length=1)
    analysis_frame: AnalysisFrame
    source_native_view: EmbeddedStateView
    analysis_view: EmbeddedStateView
    label_assertions: tuple[LabelAssertion, ...]
    grouping_keys: tuple[GroupingKey, ...]
    quality_disposition: str = Field(min_length=1)
    quality_findings: tuple[dict[str, object], ...]
    processing_steps: tuple[ProcessingStep, ...]

    @model_validator(mode="after")
    def validate_views(self) -> Self:
        if self.analysis_view.view_kind != "analysis":
            raise ValueError("analysis_view must declare view_kind='analysis'")
        if self.source_native_view.view_kind not in {"source_native", "reference"}:
            raise ValueError("source_native_view has an unsupported view kind")
        if len(self.source_native_view.rows) != len(self.analysis_view.rows):
            raise ValueError("source and analysis state views must align by row")
        return self
    ####
####


class EmbeddedFixture(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    fixture_id: str = Field(min_length=1)
    source: SpaceNearSourceSpec
    episode: EmbeddedEpisode

    @model_validator(mode="after")
    def validate_source_identity(self) -> Self:
        if self.source.source_dataset_id != self.episode.source_dataset_id:
            raise ValueError("fixture source_dataset_id values do not match")
        return self
    ####
####


class SpaceNearFixturePortfolio(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    portfolio_version: Literal["space-near-repository-fixtures-v0.1"]
    contract_version: Literal["trajectory-corpus-v0.1"]
    program_domain: Literal["space"]
    corpus_sublane: Literal["space_near"]
    authoritative_common_front_validation: Literal["pending", "passed"]
    fixtures: tuple[EmbeddedFixture, ...]

    @field_validator("fixtures")
    @classmethod
    def validate_unique_fixtures(
        cls,
        fixtures: tuple[EmbeddedFixture, ...],
    ) -> tuple[EmbeddedFixture, ...]:
        fixture_ids = tuple(fixture.fixture_id for fixture in fixtures)
        episode_ids = tuple(fixture.episode.episode_id for fixture in fixtures)
        if len(fixture_ids) != len(set(fixture_ids)):
            raise ValueError("fixture_id values must be unique")
        if len(episode_ids) != len(set(episode_ids)):
            raise ValueError("episode_id values must be unique")
        return fixtures
    ####
####


@dataclass(frozen=True, slots=True)
class SpaceNearFixtureValidation:
    episode_id: str
    source_dataset_id: str
    sample_count: int
    state_view_count: int
    label_assertion_count: int
    quality_disposition: str
####


