"""Governed source registries, snapshots, selection, and split audits."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Iterable, Mapping

import yaml
from pydantic import Field, field_validator, model_validator

from .episode_contracts import (
    AssetReference,
    GroupingNamespace,
    ProgramDomain,
    StateRole,
    StrictFrozenModel,
    TrajectoryEpisodeManifest,
)

SOURCE_REGISTRY_VERSION = "real-world-source-registry-v0.1"
CORPUS_SNAPSHOT_VERSION = "corpus-snapshot-v0.1"
REAL_WORLD_CORPUS_LANES = (
    "land_surface",
    "sea_surface",
    "sea_subsurface",
    "air_atmospheric",
    "space_near",
    "space_orbital",
)


class SourceEvidenceState(StrEnum):
    LEAD_ONLY = "lead_only"
    ACCESS_VERIFIED = "access_verified"
    ARTIFACT_ACQUIRED = "artifact_acquired"
    SCHEMA_INSPECTED = "schema_inspected"
    MAPPING_COMPLETE = "mapping_complete"
    FIXTURE_VALIDATED = "fixture_validated"
    PREPARED = "prepared"
    RELEASED = "released"


_EVIDENCE_RANK = {
    SourceEvidenceState.LEAD_ONLY: 0,
    SourceEvidenceState.ACCESS_VERIFIED: 1,
    SourceEvidenceState.ARTIFACT_ACQUIRED: 2,
    SourceEvidenceState.SCHEMA_INSPECTED: 3,
    SourceEvidenceState.MAPPING_COMPLETE: 4,
    SourceEvidenceState.FIXTURE_VALIDATED: 5,
    SourceEvidenceState.PREPARED: 6,
    SourceEvidenceState.RELEASED: 7,
}


class SnapshotSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class SourceArtifactRecord(StrictFrozenModel):
    artifact_id: str = Field(min_length=1)
    uri_or_query: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    accessed_on: date | None = None
    license_id: str | None = None
    redistribution_allowed: bool = False
    derived_data_redistribution_allowed: bool = False
    notes: tuple[str, ...] = ()


class SourceRegistryEntry(StrictFrozenModel):
    source_dataset_id: str = Field(min_length=1)
    lane: str = Field(min_length=1)
    domain: ProgramDomain
    title: str = Field(min_length=1)
    adapter_id: str = Field(min_length=1)
    adapter_version: str = Field(min_length=1)
    portfolio_role: str = Field(default="candidate", min_length=1)
    evidence_state: SourceEvidenceState
    artifacts: tuple[SourceArtifactRecord, ...] = Field(min_length=1)
    grouping_namespaces: tuple[GroupingNamespace, ...] = Field(min_length=1)
    claim_boundary: tuple[str, ...] = ()
    promotion_blockers: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_artifacts(self) -> SourceRegistryEntry:
        artifact_ids = tuple(artifact.artifact_id for artifact in self.artifacts)
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("source artifact IDs must be unique within a registry entry")
        if len(self.grouping_namespaces) != len(set(self.grouping_namespaces)):
            raise ValueError("source grouping namespaces must be unique")
        split_capable = {
            GroupingNamespace.PHYSICAL_PLATFORM,
            GroupingNamespace.SOURCE_RECORDING,
            GroupingNamespace.MISSION_EVENT,
        }
        if not any(namespace in split_capable for namespace in self.grouping_namespaces):
            raise ValueError("source requires at least one split-capable grouping namespace")
        return self


class SourceRegistry(StrictFrozenModel):
    registry_version: str = SOURCE_REGISTRY_VERSION
    registry_id: str = Field(min_length=1)
    sources: tuple[SourceRegistryEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_sources(self) -> SourceRegistry:
        dataset_ids = tuple(source.source_dataset_id for source in self.sources)
        if len(dataset_ids) != len(set(dataset_ids)):
            raise ValueError("source dataset IDs must be unique in a registry")
        artifact_ids = tuple(
            artifact.artifact_id for source in self.sources for artifact in source.artifacts
        )
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("source artifact IDs must be unique in a registry")
        return self

    def source(self, source_dataset_id: str) -> SourceRegistryEntry:
        for source in self.sources:
            if source.source_dataset_id == source_dataset_id:
                return source
        raise KeyError(f"unknown source dataset: {source_dataset_id!r}")


class SnapshotEpisodeReference(StrictFrozenModel):
    episode_id: str = Field(min_length=1)
    lane: str = Field(min_length=1)
    source_dataset_id: str = Field(min_length=1)
    manifest: AssetReference


class CorpusSnapshotManifest(StrictFrozenModel):
    schema_version: str = CORPUS_SNAPSHOT_VERSION
    snapshot_id: str = Field(min_length=1)
    registry_id: str = Field(min_length=1)
    created_at: datetime
    episodes: tuple[SnapshotEpisodeReference, ...] = Field(min_length=1)
    source_artifact_ids: tuple[str, ...] = Field(min_length=1)
    adapter_versions: tuple[str, ...] = Field(min_length=1)
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_references(self) -> CorpusSnapshotManifest:
        episode_ids = tuple(reference.episode_id for reference in self.episodes)
        if len(episode_ids) != len(set(episode_ids)):
            raise ValueError("snapshot episode IDs must be unique")
        if len(self.source_artifact_ids) != len(set(self.source_artifact_ids)):
            raise ValueError("snapshot source artifact IDs must be unique")
        if len(self.adapter_versions) != len(set(self.adapter_versions)):
            raise ValueError("snapshot adapter versions must be unique")
        return self

    def content_sha256(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class SnapshotSelectionPolicy(StrictFrozenModel):
    lanes: tuple[str, ...] = ()
    domains: tuple[ProgramDomain, ...] = ()
    source_dataset_ids: tuple[str, ...] = ()
    state_roles: tuple[StateRole, ...] = ()
    require_classifier_view: bool = False

    @field_validator("lanes", "source_dataset_ids")
    @classmethod
    def validate_unique_strings(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("selection values must be unique")
        return value


class SnapshotEvaluationReport(StrictFrozenModel):
    snapshot_id: str
    passes: bool
    episode_count: int = Field(ge=0)
    classifier_ready_episode_count: int = Field(ge=0)
    lane_episode_counts: Mapping[str, int]
    domain_episode_counts: Mapping[str, int]
    source_dataset_counts: Mapping[str, int] = Field(default_factory=dict)
    state_role_counts: Mapping[str, int] = Field(default_factory=dict)
    observation_modality_counts: Mapping[str, int] = Field(default_factory=dict)
    quality_disposition_counts: Mapping[str, int] = Field(default_factory=dict)
    quality_finding_severity_counts: Mapping[str, int] = Field(default_factory=dict)
    label_evidence_counts: Mapping[str, int] = Field(default_factory=dict)
    group_namespace_counts: Mapping[str, int] = Field(default_factory=dict)
    proxy_label_count: int = Field(default=0, ge=0)
    issues: tuple[str, ...] = ()


class EpisodeSplitAssignment(StrictFrozenModel):
    episode_id: str = Field(min_length=1)
    split: SnapshotSplit


class SplitAuditReport(StrictFrozenModel):
    passes: bool
    assignment_count: int = Field(ge=0)
    group_count: int = Field(ge=0)
    issues: tuple[str, ...] = ()


class SourceRegistryEvaluationReport(StrictFrozenModel):
    registry_id: str
    passes: bool
    required_lanes: tuple[str, ...]
    covered_lanes: tuple[str, ...]
    missing_lanes: tuple[str, ...]
    lane_source_counts: Mapping[str, int]
    lane_best_evidence_states: Mapping[str, str]
    evidence_state_counts: Mapping[str, int]
    fixture_validated_lanes: tuple[str, ...]
    prepared_lanes: tuple[str, ...]
    classifier_ready: bool
    open_gates: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()


def load_source_registry(path: str | Path) -> SourceRegistry:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return SourceRegistry.model_validate(payload)


def load_snapshot_manifest(path: str | Path) -> CorpusSnapshotManifest:
    return CorpusSnapshotManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))


def write_snapshot_manifest(manifest: CorpusSnapshotManifest, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_snapshot_episodes(
    manifest: CorpusSnapshotManifest,
    manifest_path: str | Path,
) -> tuple[TrajectoryEpisodeManifest, ...]:
    root = Path(manifest_path).parent
    episodes: list[TrajectoryEpisodeManifest] = []
    for reference in manifest.episodes:
        episode_path = root / reference.manifest.path
        actual_sha256 = _sha256_file(episode_path)
        if actual_sha256 != reference.manifest.sha256:
            raise ValueError(
                f"snapshot episode hash mismatch for {reference.episode_id}: "
                f"expected {reference.manifest.sha256}, got {actual_sha256}"
            )
        episode = TrajectoryEpisodeManifest.model_validate_json(
            episode_path.read_text(encoding="utf-8")
        )
        if episode.episode_id != reference.episode_id:
            raise ValueError(
                f"snapshot episode ID mismatch: expected {reference.episode_id}, "
                f"got {episode.episode_id}"
            )
        if episode.corpus_snapshot_id != manifest.snapshot_id:
            raise ValueError(
                f"snapshot ID mismatch for {reference.episode_id}: "
                f"expected {manifest.snapshot_id}, got {episode.corpus_snapshot_id}"
            )
        episodes.append(episode)
    return tuple(episodes)


def select_snapshot_episodes(
    episodes: Iterable[TrajectoryEpisodeManifest],
    policy: SnapshotSelectionPolicy,
) -> tuple[TrajectoryEpisodeManifest, ...]:
    selected: list[TrajectoryEpisodeManifest] = []
    for episode in episodes:
        if policy.lanes and episode.corpus_sublane not in policy.lanes:
            continue
        if policy.domains and episode.primary_program_domain not in policy.domains:
            continue
        if policy.source_dataset_ids and episode.source_dataset_id not in policy.source_dataset_ids:
            continue
        if policy.state_roles and not any(
            view.state_role in policy.state_roles for view in episode.state_views
        ):
            continue
        if policy.require_classifier_view and episode.classifier_trajectory_view is None:
            continue
        selected.append(episode)
    return tuple(selected)


def evaluate_snapshot(
    manifest: CorpusSnapshotManifest,
    episodes: Iterable[TrajectoryEpisodeManifest],
    *,
    expected_lanes: Iterable[str] = (),
) -> SnapshotEvaluationReport:
    materialized = tuple(episodes)
    issues: list[str] = []
    expected_references = {reference.episode_id: reference for reference in manifest.episodes}
    actual_ids = tuple(episode.episode_id for episode in materialized)
    if len(actual_ids) != len(set(actual_ids)):
        issues.append("duplicate_episode_id")
    if set(actual_ids) != set(expected_references):
        issues.append("snapshot_episode_reference_mismatch")
    for episode in materialized:
        reference = expected_references.get(episode.episode_id)
        if reference is None:
            continue
        if episode.corpus_snapshot_id != manifest.snapshot_id:
            issues.append(f"episode_snapshot_mismatch:{episode.episode_id}")
        if episode.source_dataset_id != reference.source_dataset_id:
            issues.append(f"episode_source_mismatch:{episode.episode_id}")
        if episode.corpus_sublane != reference.lane:
            issues.append(f"episode_lane_mismatch:{episode.episode_id}")
    lane_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    state_role_counts: dict[str, int] = {}
    modality_counts: dict[str, int] = {}
    quality_disposition_counts: dict[str, int] = {}
    quality_severity_counts: dict[str, int] = {}
    label_evidence_counts: dict[str, int] = {}
    group_namespace_counts: dict[str, int] = {}
    proxy_label_count = 0
    classifier_ready = 0
    for episode in materialized:
        lane_counts[episode.corpus_sublane] = lane_counts.get(episode.corpus_sublane, 0) + 1
        domain = episode.primary_program_domain.value
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        source_counts[episode.source_dataset_id] = (
            source_counts.get(episode.source_dataset_id, 0) + 1
        )
        modality_counts[episode.observation_modality] = (
            modality_counts.get(episode.observation_modality, 0) + 1
        )
        quality_disposition_counts[episode.quality_summary.disposition] = (
            quality_disposition_counts.get(episode.quality_summary.disposition, 0) + 1
        )
        for view in episode.state_views:
            state_role = view.state_role.value
            state_role_counts[state_role] = state_role_counts.get(state_role, 0) + 1
        for finding in episode.quality_summary.findings:
            severity = finding.severity.value
            quality_severity_counts[severity] = quality_severity_counts.get(severity, 0) + 1
        for label in episode.labels:
            evidence_kind = label.evidence_kind.value
            label_evidence_counts[evidence_kind] = label_evidence_counts.get(evidence_kind, 0) + 1
            proxy_label_count += int(label.proxy)
        for grouping_key in episode.grouping_keys:
            namespace = grouping_key.namespace.value
            group_namespace_counts[namespace] = group_namespace_counts.get(namespace, 0) + 1
        classifier_ready += episode.classifier_trajectory_view is not None
    for lane in expected_lanes:
        if lane not in lane_counts:
            issues.append(f"missing_expected_lane:{lane}")
    return SnapshotEvaluationReport(
        snapshot_id=manifest.snapshot_id,
        passes=not issues,
        episode_count=len(materialized),
        classifier_ready_episode_count=classifier_ready,
        lane_episode_counts=lane_counts,
        domain_episode_counts=domain_counts,
        source_dataset_counts=source_counts,
        state_role_counts=state_role_counts,
        observation_modality_counts=modality_counts,
        quality_disposition_counts=quality_disposition_counts,
        quality_finding_severity_counts=quality_severity_counts,
        label_evidence_counts=label_evidence_counts,
        group_namespace_counts=group_namespace_counts,
        proxy_label_count=proxy_label_count,
        issues=tuple(issues),
    )


def evaluate_source_registry(
    registry: SourceRegistry,
    *,
    expected_lanes: Iterable[str] = REAL_WORLD_CORPUS_LANES,
) -> SourceRegistryEvaluationReport:
    required_lanes = tuple(dict.fromkeys(expected_lanes))
    issues: list[str] = []
    lane_source_counts: dict[str, int] = {}
    lane_best_evidence_states: dict[str, str] = {}
    evidence_state_counts: dict[str, int] = {}
    for source in registry.sources:
        lane_source_counts[source.lane] = lane_source_counts.get(source.lane, 0) + 1
        state = source.evidence_state.value
        evidence_state_counts[state] = evidence_state_counts.get(state, 0) + 1
        prior_state = lane_best_evidence_states.get(source.lane)
        if (
            prior_state is None
            or _EVIDENCE_RANK[source.evidence_state]
            > _EVIDENCE_RANK[SourceEvidenceState(prior_state)]
        ):
            lane_best_evidence_states[source.lane] = state

    covered_lanes = tuple(lane for lane in required_lanes if lane in lane_source_counts)
    missing_lanes = tuple(lane for lane in required_lanes if lane not in lane_source_counts)
    issues.extend(f"missing_required_lane:{lane}" for lane in missing_lanes)
    fixture_validated_lanes = tuple(
        lane
        for lane in covered_lanes
        if _EVIDENCE_RANK[SourceEvidenceState(lane_best_evidence_states[lane])]
        >= _EVIDENCE_RANK[SourceEvidenceState.FIXTURE_VALIDATED]
    )
    prepared_lanes = tuple(
        lane
        for lane in covered_lanes
        if _EVIDENCE_RANK[SourceEvidenceState(lane_best_evidence_states[lane])]
        >= _EVIDENCE_RANK[SourceEvidenceState.PREPARED]
    )
    classifier_ready = len(prepared_lanes) == len(required_lanes)
    open_gates = tuple(
        f"{lane}:requires_prepared_or_released_source"
        for lane in required_lanes
        if lane not in prepared_lanes
    )
    return SourceRegistryEvaluationReport(
        registry_id=registry.registry_id,
        passes=not issues,
        required_lanes=required_lanes,
        covered_lanes=covered_lanes,
        missing_lanes=missing_lanes,
        lane_source_counts=lane_source_counts,
        lane_best_evidence_states=lane_best_evidence_states,
        evidence_state_counts=evidence_state_counts,
        fixture_validated_lanes=fixture_validated_lanes,
        prepared_lanes=prepared_lanes,
        classifier_ready=classifier_ready,
        open_gates=open_gates,
        issues=tuple(issues),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_split_assignments(
    episodes: Iterable[TrajectoryEpisodeManifest],
    assignments: Iterable[EpisodeSplitAssignment],
) -> SplitAuditReport:
    materialized = tuple(episodes)
    assignment_rows = tuple(assignments)
    issues: list[str] = []
    assignment_by_episode: dict[str, SnapshotSplit] = {}
    for assignment in assignment_rows:
        if assignment.episode_id in assignment_by_episode:
            issues.append(f"duplicate_split_assignment:{assignment.episode_id}")
        assignment_by_episode[assignment.episode_id] = assignment.split
    group_to_split: dict[tuple[GroupingNamespace, str], SnapshotSplit] = {}
    for episode in materialized:
        split = assignment_by_episode.get(episode.episode_id)
        if split is None:
            issues.append(f"missing_split_assignment:{episode.episode_id}")
            continue
        for grouping_key in episode.grouping_keys:
            if grouping_key.namespace not in {
                GroupingNamespace.PHYSICAL_PLATFORM,
                GroupingNamespace.SOURCE_RECORDING,
                GroupingNamespace.MISSION_EVENT,
            }:
                continue
            identity = (grouping_key.namespace, grouping_key.opaque_value)
            prior_split = group_to_split.get(identity)
            if prior_split is not None and prior_split is not split:
                issues.append(
                    "group_split_collision:"
                    f"{grouping_key.namespace.value}:{grouping_key.opaque_value}"
                )
            group_to_split[identity] = split
    extra_assignments = set(assignment_by_episode) - {
        episode.episode_id for episode in materialized
    }
    issues.extend(
        f"unknown_split_assignment:{episode_id}" for episode_id in sorted(extra_assignments)
    )
    return SplitAuditReport(
        passes=not issues,
        assignment_count=len(assignment_rows),
        group_count=len(group_to_split),
        issues=tuple(issues),
    )


__all__ = [
    "CorpusSnapshotManifest",
    "CORPUS_SNAPSHOT_VERSION",
    "EpisodeSplitAssignment",
    "REAL_WORLD_CORPUS_LANES",
    "SOURCE_REGISTRY_VERSION",
    "SnapshotEpisodeReference",
    "SnapshotEvaluationReport",
    "SnapshotSelectionPolicy",
    "SnapshotSplit",
    "SourceArtifactRecord",
    "SourceEvidenceState",
    "SourceRegistry",
    "SourceRegistryEvaluationReport",
    "SourceRegistryEntry",
    "SplitAuditReport",
    "audit_split_assignments",
    "evaluate_snapshot",
    "evaluate_source_registry",
    "load_snapshot_episodes",
    "load_snapshot_manifest",
    "load_source_registry",
    "select_snapshot_episodes",
    "write_snapshot_manifest",
]
