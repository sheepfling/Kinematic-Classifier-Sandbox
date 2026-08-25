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
    LabelEvidenceKind,
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


class Product4GateReport(StrictFrozenModel):
    """Composed Product 4 decision across portfolio and study-eligibility gates."""

    registry_id: str
    passes: bool
    decision: str
    registry_passes: bool
    provenance_passes: bool
    rights_passes: bool
    rights_release_ready: bool
    snapshot_present: bool
    snapshot_passes: bool
    coverage_passes: bool
    quality_passes: bool
    leakage_passes: bool
    classifier_projection_passes: bool
    classifier_ready: bool
    registry_report: SourceRegistryEvaluationReport
    snapshot_report: SnapshotEvaluationReport | None = None
    split_report: SplitAuditReport | None = None
    selected_source_dataset_ids: tuple[str, ...] = ()
    selected_source_artifact_ids: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()
    open_gates: tuple[str, ...] = ()


def load_source_registry(path: str | Path) -> SourceRegistry:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return SourceRegistry.model_validate(payload)


def load_snapshot_manifest(path: str | Path) -> CorpusSnapshotManifest:
    return CorpusSnapshotManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _verify_episode_assets(root: Path, episode: TrajectoryEpisodeManifest) -> None:
    assets = [view.sample_asset for view in episode.state_views]
    if episode.classifier_trajectory_view is not None:
        assets.append(episode.classifier_trajectory_view.asset)
    for asset in assets:
        asset_path = root / asset.path
        if not asset_path.is_file():
            raise ValueError(
                f"snapshot asset is missing for {episode.episode_id}: {asset.path}"
            )
        actual_sha256 = _sha256_file(asset_path)
        if actual_sha256 != asset.sha256:
            raise ValueError(
                f"snapshot asset hash mismatch for {episode.episode_id}: "
                f"{asset.path}; expected {asset.sha256}, got {actual_sha256}"
            )


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
        _verify_episode_assets(root, episode)
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


def _registry_provenance_issues(
    registry: SourceRegistry,
    *,
    source_dataset_ids: set[str] | None,
) -> tuple[str, ...]:
    issues: list[str] = []
    for source in registry.sources:
        if source_dataset_ids is not None and source.source_dataset_id not in source_dataset_ids:
            continue
        evidence_rank = _EVIDENCE_RANK[source.evidence_state]
        for artifact in source.artifacts:
            if (
                evidence_rank >= _EVIDENCE_RANK[SourceEvidenceState.ARTIFACT_ACQUIRED]
                and artifact.sha256 is None
            ):
                issues.append(f"missing_artifact_sha256:{artifact.artifact_id}")
            if (
                evidence_rank >= _EVIDENCE_RANK[SourceEvidenceState.SCHEMA_INSPECTED]
                and artifact.license_id is None
            ):
                issues.append(f"missing_license_terms:{artifact.artifact_id}")
    return tuple(issues)


def _snapshot_source_integrity_issues(
    registry: SourceRegistry,
    snapshot: CorpusSnapshotManifest,
    episodes: tuple[TrajectoryEpisodeManifest, ...],
) -> tuple[str, ...]:
    issues: list[str] = []
    if snapshot.registry_id != registry.registry_id:
        issues.append("snapshot_registry_id_mismatch")

    registry_artifacts = {
        artifact.artifact_id: source
        for source in registry.sources
        for artifact in source.artifacts
    }
    snapshot_artifact_ids = set(snapshot.source_artifact_ids)
    for artifact_id in snapshot.source_artifact_ids:
        if artifact_id not in registry_artifacts:
            issues.append(f"snapshot_unknown_source_artifact:{artifact_id}")

    registry_sources = {source.source_dataset_id: source for source in registry.sources}
    references = {reference.episode_id: reference for reference in snapshot.episodes}
    snapshot_adapter_versions = set(snapshot.adapter_versions)
    for reference in snapshot.episodes:
        source = registry_sources.get(reference.source_dataset_id)
        if source is None:
            issues.append(f"snapshot_unknown_source_dataset:{reference.source_dataset_id}")
            continue
        if source.lane != reference.lane:
            issues.append(f"snapshot_lane_source_mismatch:{reference.episode_id}")
        adapter_token = f"{source.adapter_id}:{source.adapter_version}"
        if adapter_token not in snapshot_adapter_versions:
            issues.append(f"snapshot_missing_adapter_version:{adapter_token}")

    for episode in episodes:
        reference = references.get(episode.episode_id)
        if reference is None:
            continue
        if not set(episode.source_artifact_ids).issubset(snapshot_artifact_ids):
            issues.append(f"episode_artifact_not_in_snapshot:{episode.episode_id}")
        source = registry_sources.get(episode.source_dataset_id)
        if source is None:
            issues.append(f"episode_unknown_source_dataset:{episode.episode_id}")
        elif source.lane != episode.corpus_sublane:
            issues.append(f"episode_lane_source_mismatch:{episode.episode_id}")
        elif not set(episode.source_artifact_ids).issubset(
            {artifact.artifact_id for artifact in source.artifacts}
        ):
            issues.append(f"episode_artifact_not_declared_by_source:{episode.episode_id}")
    return tuple(issues)


def evaluate_product4_gates(
    registry: SourceRegistry,
    *,
    snapshot: CorpusSnapshotManifest | None = None,
    episodes: Iterable[TrajectoryEpisodeManifest] = (),
    assignments: Iterable[EpisodeSplitAssignment] = (),
    expected_lanes: Iterable[str] = REAL_WORLD_CORPUS_LANES,
) -> Product4GateReport:
    """Evaluate the complete Product 4 promotion boundary.

    The registry can be coherent while the classifier gate remains blocked. A
    source registry report therefore is not treated as a snapshot, leakage,
    quality, or classifier-readiness result. This composed report makes every
    missing layer explicit and accepts restricted local use when terms are
    declared, while separately reporting whether derived assets are releasable.
    """

    expected_lane_values = tuple(dict.fromkeys(expected_lanes))
    materialized_episodes = tuple(episodes)
    materialized_assignments = tuple(assignments)
    registry_report = evaluate_source_registry(
        registry,
        expected_lanes=expected_lane_values,
    )
    issues: list[str] = list(registry_report.issues)

    selected_source_dataset_ids: tuple[str, ...] = ()
    selected_source_artifact_ids: tuple[str, ...] = ()
    snapshot_integrity_issues: tuple[str, ...] = ()
    if snapshot is not None:
        selected_source_dataset_ids = tuple(
            dict.fromkeys(reference.source_dataset_id for reference in snapshot.episodes)
        )
        selected_source_artifact_ids = tuple(snapshot.source_artifact_ids)
        snapshot_integrity_issues = _snapshot_source_integrity_issues(
            registry,
            snapshot,
            materialized_episodes,
        )
        issues.extend(snapshot_integrity_issues)

    provenance_issues = _registry_provenance_issues(
        registry,
        source_dataset_ids=(
            set(selected_source_dataset_ids) if snapshot is not None else None
        ),
    )
    issues.extend(provenance_issues)
    provenance_passes = not provenance_issues

    snapshot_report: SnapshotEvaluationReport | None = None
    snapshot_passes = False
    coverage_passes = False
    quality_passes = False
    if snapshot is None:
        issues.append("snapshot_missing")
    else:
        snapshot_integrity_report = evaluate_snapshot(
            snapshot,
            materialized_episodes,
        )
        snapshot_report = evaluate_snapshot(
            snapshot,
            materialized_episodes,
            expected_lanes=expected_lane_values,
        )
        issues.extend(snapshot_report.issues)
        snapshot_passes = snapshot_integrity_report.passes and not snapshot_integrity_issues
        coverage_passes = snapshot_integrity_report.passes and all(
            snapshot_report.lane_episode_counts.get(lane, 0) > 0
            for lane in expected_lane_values
        )
        if not coverage_passes:
            issues.append("snapshot_lane_coverage_incomplete")
        quality_passes = (
            snapshot_report.quality_finding_severity_counts.get("error", 0) == 0
            and not any(
                disposition.casefold() in {"reject", "rejected", "invalid"}
                for disposition in snapshot_report.quality_disposition_counts
            )
        )
        if not quality_passes:
            issues.append("snapshot_quality_gate_failed")

    rights_issues: list[str] = []
    rights_release_ready = False
    if snapshot is None:
        rights_issues.append("rights_not_evaluable_without_snapshot")
    else:
        artifact_index = {
            artifact.artifact_id: artifact
            for source in registry.sources
            for artifact in source.artifacts
        }
        rights_release_ready = True
        for artifact_id in selected_source_artifact_ids:
            artifact = artifact_index.get(artifact_id)
            if artifact is None:
                rights_issues.append(f"rights_unknown_artifact:{artifact_id}")
                rights_release_ready = False
                continue
            if artifact.license_id is None:
                rights_issues.append(f"rights_missing_license_terms:{artifact_id}")
                rights_release_ready = False
            if not (
                artifact.redistribution_allowed
                or artifact.derived_data_redistribution_allowed
            ):
                rights_release_ready = False
    issues.extend(rights_issues)
    rights_passes = not rights_issues

    classifier_projection_passes = True
    for episode in materialized_episodes:
        if episode.classifier_trajectory_view is None:
            classifier_projection_passes = False
            issues.append(f"classifier_view_missing:{episode.episode_id}")
        if not episode.labels:
            classifier_projection_passes = False
            issues.append(f"label_assertion_missing:{episode.episode_id}")
        if any(label.proxy or label.evidence_kind is LabelEvidenceKind.PROXY for label in episode.labels):
            classifier_projection_passes = False
            issues.append(f"proxy_label_present:{episode.episode_id}")
    if snapshot is None or not materialized_episodes:
        classifier_projection_passes = False

    split_report: SplitAuditReport | None = None
    leakage_passes = False
    if snapshot is None:
        issues.append("split_audit_not_evaluable_without_snapshot")
    else:
        split_report = audit_split_assignments(materialized_episodes, materialized_assignments)
        issues.extend(split_report.issues)
        assigned_splits = {assignment.split for assignment in materialized_assignments}
        required_splits = set(SnapshotSplit)
        leakage_passes = split_report.passes and assigned_splits == required_splits
        if assigned_splits != required_splits:
            missing_splits = sorted(split.value for split in required_splits - assigned_splits)
            issues.append("missing_required_split:" + ",".join(missing_splits))
        if not split_report.passes:
            issues.append("split_leakage_gate_failed")

    selected_sources_prepared = snapshot is not None
    if snapshot is None:
        selected_sources_prepared = False
    else:
        registry_sources = {source.source_dataset_id: source for source in registry.sources}
        for dataset_id in selected_source_dataset_ids:
            source = registry_sources.get(dataset_id)
            if source is None or _EVIDENCE_RANK[source.evidence_state] < _EVIDENCE_RANK[SourceEvidenceState.PREPARED]:
                selected_sources_prepared = False
                issues.append(f"source_not_prepared:{dataset_id}")

    classifier_ready = all(
        (
            registry_report.classifier_ready,
            selected_sources_prepared,
            provenance_passes,
            rights_passes,
            snapshot_passes,
            coverage_passes,
            quality_passes,
            leakage_passes,
            classifier_projection_passes,
        )
    )
    if classifier_ready:
        decision = "real_world_evidence_supported"
    elif snapshot is None or not coverage_passes:
        decision = "insufficient_real_world_evidence"
    elif not registry_report.classifier_ready or not selected_sources_prepared:
        decision = "revise_source_portfolio"
    elif not leakage_passes:
        decision = "revise_grouping_policy"
    elif not classifier_projection_passes:
        decision = "revise_label_claim"
    else:
        decision = "supported_with_limits"

    open_gates: list[str] = []
    if not registry_report.passes:
        open_gates.append("registry:required_lane_coverage")
    if not provenance_passes:
        open_gates.append("provenance:pin_artifact_hashes_and_terms")
    if not snapshot_passes:
        open_gates.append("snapshot:immutable_manifest_integrity")
    if not coverage_passes:
        open_gates.append("coverage:one_or_more_episodes_per_required_lane")
    if not quality_passes:
        open_gates.append("quality:no_error_or_rejected_episodes")
    if not leakage_passes:
        open_gates.append("leakage:grouped_train_validation_test_assignments")
    if not classifier_projection_passes:
        open_gates.append("classifier_view:identity_free_non_proxy_projection")
    if not rights_passes:
        open_gates.append("rights:declared_terms_for_selected_artifacts")
    if not selected_sources_prepared:
        open_gates.append("source:promote_selected_sources_to_prepared")
    if not rights_release_ready and snapshot is not None:
        open_gates.append("rights:release_boundary_is_restricted")

    unique_issues = tuple(dict.fromkeys(issues))
    unique_open_gates = tuple(dict.fromkeys(open_gates))
    return Product4GateReport(
        registry_id=registry.registry_id,
        passes=classifier_ready,
        decision=decision,
        registry_passes=registry_report.passes,
        provenance_passes=provenance_passes,
        rights_passes=rights_passes,
        rights_release_ready=rights_release_ready,
        snapshot_present=snapshot is not None,
        snapshot_passes=snapshot_passes,
        coverage_passes=coverage_passes,
        quality_passes=quality_passes,
        leakage_passes=leakage_passes,
        classifier_projection_passes=classifier_projection_passes,
        classifier_ready=classifier_ready,
        registry_report=registry_report,
        snapshot_report=snapshot_report,
        split_report=split_report,
        selected_source_dataset_ids=selected_source_dataset_ids,
        selected_source_artifact_ids=selected_source_artifact_ids,
        issues=unique_issues,
        open_gates=unique_open_gates,
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
    "Product4GateReport",
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
    "evaluate_product4_gates",
    "evaluate_snapshot",
    "evaluate_source_registry",
    "load_snapshot_episodes",
    "load_snapshot_manifest",
    "load_source_registry",
    "select_snapshot_episodes",
    "write_snapshot_manifest",
]
