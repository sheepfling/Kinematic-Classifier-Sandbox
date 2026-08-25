"""Lane-isolated Product 4 portfolio evaluation and promotion reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from pydantic import Field

from .episode_contracts import (
    GroupingNamespace,
    StrictFrozenModel,
    TrajectoryEpisodeManifest,
)
from .portfolio import (
    REAL_WORLD_CORPUS_LANES,
    CorpusSnapshotManifest,
    EpisodeSplitAssignment,
    Product4GateReport,
    SourceRegistry,
    evaluate_product4_gates,
)

PORTFOLIO_MATRIX_VERSION = "product4-lane-evaluation-matrix-v0.1"


class Product4LaneEvaluation(StrictFrozenModel):
    """One lane's gate result, scoped to only that lane's snapshot episodes."""

    lane: str = Field(min_length=1)
    best_evidence_state: str | None = None
    source_dataset_ids: tuple[str, ...] = ()
    episode_count: int = Field(ge=0)
    classifier_ready_episode_count: int = Field(ge=0)
    registry_promotion_blockers: tuple[str, ...] = ()
    gate: Product4GateReport


class Product4LaneEvaluationMatrix(StrictFrozenModel):
    """Comparable per-lane decisions plus the source snapshot identity."""

    schema_version: str = PORTFOLIO_MATRIX_VERSION
    registry_id: str = Field(min_length=1)
    snapshot_id: str | None = None
    snapshot_content_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    expected_lanes: tuple[str, ...] = Field(min_length=1)
    split_grouping_namespaces: tuple[GroupingNamespace, ...] = Field(min_length=1)
    all_lanes_pass: bool
    lane_reports: tuple[Product4LaneEvaluation, ...] = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class _LaneScope:
    snapshot: CorpusSnapshotManifest | None
    episodes: tuple[TrajectoryEpisodeManifest, ...]
    assignments: tuple[EpisodeSplitAssignment, ...]


def _adapter_token(adapter_id: str, adapter_version: str) -> str:
    return f"{adapter_id}:{adapter_version}"


def _scope_snapshot(
    snapshot: CorpusSnapshotManifest,
    episodes: tuple[TrajectoryEpisodeManifest, ...],
    registry: SourceRegistry,
) -> CorpusSnapshotManifest | None:
    episode_ids = {episode.episode_id for episode in episodes}
    references = tuple(
        reference for reference in snapshot.episodes if reference.episode_id in episode_ids
    )
    if not references:
        return None
    source_ids = tuple(sorted({reference.source_dataset_id for reference in references}))
    source_artifact_ids = tuple(
        sorted(
            {
                artifact_id
                for episode in episodes
                for artifact_id in episode.source_artifact_ids
            }
        )
    )
    adapter_versions = tuple(
        sorted(
            {
                _adapter_token(
                    registry.source(source_id).adapter_id,
                    registry.source(source_id).adapter_version,
                )
                for source_id in source_ids
            }
        )
    )
    return snapshot.model_copy(
        update={
            "episodes": references,
            "source_artifact_ids": source_artifact_ids,
            "adapter_versions": adapter_versions,
        }
    )


def _lane_scope(
    lane: str,
    *,
    snapshot: CorpusSnapshotManifest,
    episodes: tuple[TrajectoryEpisodeManifest, ...],
    assignments: tuple[EpisodeSplitAssignment, ...],
    registry: SourceRegistry,
) -> _LaneScope:
    lane_episodes = tuple(
        episode for episode in episodes if episode.corpus_sublane == lane
    )
    lane_ids = {episode.episode_id for episode in lane_episodes}
    lane_assignments = tuple(
        assignment for assignment in assignments if assignment.episode_id in lane_ids
    )
    return _LaneScope(
        snapshot=_scope_snapshot(snapshot, lane_episodes, registry),
        episodes=lane_episodes,
        assignments=lane_assignments,
    )


def evaluate_product4_lane_matrix(
    registry: SourceRegistry,
    *,
    snapshot: CorpusSnapshotManifest,
    episodes: Iterable[TrajectoryEpisodeManifest],
    assignments: Iterable[EpisodeSplitAssignment],
    expected_lanes: Iterable[str] = REAL_WORLD_CORPUS_LANES,
    split_grouping_namespaces: Iterable[GroupingNamespace] = (
        GroupingNamespace.PHYSICAL_PLATFORM,
        GroupingNamespace.SOURCE_RECORDING,
        GroupingNamespace.MISSION_EVENT,
    ),
) -> Product4LaneEvaluationMatrix:
    """Evaluate each lane independently while preserving the full snapshot identity.

    A lane report is scoped to that lane's episode references before invoking the composed
    Product 4 gate. This prevents a blocked source in one domain from masking which other lane
    is ready for a bounded task, while the matrix's ``all_lanes_pass`` field still represents the
    full cross-domain claim.
    """

    lane_values = tuple(dict.fromkeys(expected_lanes))
    if not lane_values:
        raise ValueError("lane evaluation matrix requires at least one expected lane")
    grouping_values = tuple(dict.fromkeys(split_grouping_namespaces))
    if not grouping_values:
        raise ValueError("lane evaluation matrix requires a grouping policy")
    materialized_episodes = tuple(episodes)
    materialized_assignments = tuple(assignments)
    lane_reports: list[Product4LaneEvaluation] = []
    for lane in lane_values:
        scope = _lane_scope(
            lane,
            snapshot=snapshot,
            episodes=materialized_episodes,
            assignments=materialized_assignments,
            registry=registry,
        )
        gate = evaluate_product4_gates(
            registry,
            snapshot=scope.snapshot,
            episodes=scope.episodes,
            assignments=scope.assignments,
            expected_lanes=(lane,),
            split_grouping_namespaces=grouping_values,
        )
        source_dataset_ids = tuple(
            sorted({episode.source_dataset_id for episode in scope.episodes})
        )
        classifier_ready_episode_count = (
            0
            if gate.snapshot_report is None
            else gate.snapshot_report.classifier_ready_episode_count
        )
        blockers = tuple(
            blocker
            for source in registry.sources
            if source.lane == lane
            for blocker in source.promotion_blockers
        )
        lane_reports.append(
            Product4LaneEvaluation(
                lane=lane,
                best_evidence_state=gate.registry_report.lane_best_evidence_states.get(lane),
                source_dataset_ids=source_dataset_ids,
                episode_count=len(scope.episodes),
                classifier_ready_episode_count=classifier_ready_episode_count,
                registry_promotion_blockers=blockers,
                gate=gate,
            )
        )
    return Product4LaneEvaluationMatrix(
        registry_id=registry.registry_id,
        snapshot_id=snapshot.snapshot_id,
        snapshot_content_sha256=snapshot.content_sha256(),
        expected_lanes=lane_values,
        split_grouping_namespaces=grouping_values,
        all_lanes_pass=all(report.gate.passes for report in lane_reports),
        lane_reports=tuple(lane_reports),
    )


__all__ = [
    "PORTFOLIO_MATRIX_VERSION",
    "Product4LaneEvaluation",
    "Product4LaneEvaluationMatrix",
    "evaluate_product4_lane_matrix",
]
