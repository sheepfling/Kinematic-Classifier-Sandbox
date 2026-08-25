"""Explicit consumer boundaries for Product 4 analysis products."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Iterable, Mapping

from pydantic import Field, model_validator

from .episode_contracts import (
    AssetReference,
    StateViewKind,
    StrictFrozenModel,
    TrajectoryEpisodeManifest,
)
from .portfolio import (
    _EVIDENCE_RANK,
    CorpusSnapshotManifest,
    SourceEvidenceState,
    SourceRegistry,
)

ANALYSIS_PRODUCT_CONTRACT_VERSION = "real-world-analysis-product-v0.1"


class AnalysisProductId(StrEnum):
    """Named consumers with deliberately different evidence depth."""

    SOURCE_AUDIT = "source_audit"
    KINEMATIC_ANALYSIS = "kinematic_analysis"
    CLASSIFIER_LADDER = "classifier_ladder"


class AnalysisProductPolicy(StrictFrozenModel):
    """The state and promotion boundary for one analysis consumer."""

    product_id: AnalysisProductId
    title: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    selected_state_view_kinds: tuple[StateViewKind, ...] = ()
    includes_classifier_view: bool = False
    requires_prepared_sources: bool = False
    requires_target_labels: bool = False
    target_label_namespace: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_boundary(self) -> AnalysisProductPolicy:
        if len(self.selected_state_view_kinds) != len(set(self.selected_state_view_kinds)):
            raise ValueError("analysis-product state-view kinds must be unique")
        if self.product_id is AnalysisProductId.CLASSIFIER_LADDER:
            if self.selected_state_view_kinds:
                raise ValueError("classifier-ladder policy must not select analysis state views")
            if not self.includes_classifier_view:
                raise ValueError("classifier-ladder policy must include the classifier view")
            if not self.requires_prepared_sources:
                raise ValueError("classifier-ladder policy must require prepared sources")
            if not self.requires_target_labels:
                raise ValueError("classifier-ladder policy must require target labels")
            if not self.target_label_namespace:
                raise ValueError(
                    "classifier-ladder policy must declare a target label namespace"
                )
        elif self.includes_classifier_view:
            raise ValueError("non-classifier analysis products cannot include classifier views")
        elif self.requires_prepared_sources or self.requires_target_labels:
            raise ValueError(
                "source and kinematic analysis products cannot require classifier promotion gates"
            )
        elif self.target_label_namespace is not None:
            raise ValueError(
                "source and kinematic analysis products cannot declare a classifier target label"
            )
        return self


class AnalysisProductEpisodeSelection(StrictFrozenModel):
    """One episode's explicitly selected assets and non-feature label boundary."""

    episode_id: str = Field(min_length=1)
    lane: str = Field(min_length=1)
    source_dataset_id: str = Field(min_length=1)
    state_view_ids: tuple[str, ...] = ()
    state_assets: tuple[AssetReference, ...] = ()
    classifier_asset: AssetReference | None = None
    target_label_available: bool = False

    @model_validator(mode="after")
    def validate_asset_alignment(self) -> AnalysisProductEpisodeSelection:
        if len(self.state_view_ids) != len(set(self.state_view_ids)):
            raise ValueError("analysis-product state-view IDs must be unique")
        if len(self.state_view_ids) != len(self.state_assets):
            raise ValueError("each selected state view must have exactly one selected asset")
        return self


class AnalysisProductManifest(StrictFrozenModel):
    """Hash-bound selection manifest consumed by one analysis product."""

    schema_version: str = ANALYSIS_PRODUCT_CONTRACT_VERSION
    snapshot_id: str = Field(min_length=1)
    snapshot_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    registry_id: str = Field(min_length=1)
    policy: AnalysisProductPolicy
    episodes: tuple[AnalysisProductEpisodeSelection, ...] = Field(min_length=1)
    lane_episode_counts: Mapping[str, int]
    source_dataset_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_selection_boundary(self) -> AnalysisProductManifest:
        episode_ids = tuple(episode.episode_id for episode in self.episodes)
        if len(episode_ids) != len(set(episode_ids)):
            raise ValueError("analysis-product episode IDs must be unique")
        if tuple(sorted(self.source_dataset_ids)) != self.source_dataset_ids:
            raise ValueError("analysis-product source dataset IDs must be sorted")
        if any(count < 0 for count in self.lane_episode_counts.values()):
            raise ValueError("analysis-product lane counts must be nonnegative")
        actual_lane_counts: dict[str, int] = {}
        actual_source_ids: set[str] = set()
        for episode in self.episodes:
            actual_lane_counts[episode.lane] = actual_lane_counts.get(episode.lane, 0) + 1
            actual_source_ids.add(episode.source_dataset_id)
        if dict(self.lane_episode_counts) != dict(sorted(actual_lane_counts.items())):
            raise ValueError("analysis-product lane counts must match selected episodes")
        if set(self.source_dataset_ids) != actual_source_ids:
            raise ValueError("analysis-product source IDs must match selected episodes")

        if self.policy.includes_classifier_view:
            for episode in self.episodes:
                if episode.state_view_ids or episode.state_assets:
                    raise ValueError(
                        "classifier-ladder manifest must not select analysis state assets"
                    )
                if episode.classifier_asset is None:
                    raise ValueError(
                        "classifier-ladder manifest requires one classifier asset per episode"
                    )
                if not episode.target_label_available:
                    raise ValueError(
                        "classifier-ladder manifest requires an out-of-band target label"
                    )
        else:
            for episode in self.episodes:
                if episode.classifier_asset is not None:
                    raise ValueError(
                        "source/kinematic analysis manifests must not select classifier assets"
                    )
                if not episode.state_view_ids:
                    raise ValueError(
                        "source/kinematic analysis manifests require selected state views"
                    )
        return self


def analysis_product_policy(
    product_id: AnalysisProductId,
    *,
    target_label_namespace: str | None = None,
) -> AnalysisProductPolicy:
    """Return the explicit boundary for a named analysis consumer."""

    product_id = AnalysisProductId(product_id)
    if product_id is AnalysisProductId.SOURCE_AUDIT:
        return AnalysisProductPolicy(
            product_id=product_id,
            title="Source and provenance audit",
            purpose=(
                "Inspect every state view, source artifact, quality finding, and domain "
                "extension without making a classifier claim."
            ),
            selected_state_view_kinds=tuple(StateViewKind),
        )
    if product_id is AnalysisProductId.KINEMATIC_ANALYSIS:
        return AnalysisProductPolicy(
            product_id=product_id,
            title="Kinematic analysis",
            purpose=(
                "Analyze normalized or derived state without carrying source-native audit "
                "assets into a classifier study."
            ),
            selected_state_view_kinds=(StateViewKind.ANALYSIS,),
        )
    return AnalysisProductPolicy(
        product_id=product_id,
        title="Product 2 classifier-ladder bridge",
        purpose=(
            "Evaluate only identity-free classifier assets from a prepared held-out snapshot; "
            "labels and grouping remain outside the feature asset."
        ),
        includes_classifier_view=True,
        requires_prepared_sources=True,
        requires_target_labels=True,
        target_label_namespace=target_label_namespace,
    )


def build_analysis_product_manifest(
    snapshot: CorpusSnapshotManifest,
    episodes: Iterable[TrajectoryEpisodeManifest],
    registry: SourceRegistry,
    *,
    product_id: AnalysisProductId,
    target_label_namespace: str | None = None,
) -> AnalysisProductManifest:
    """Build a bounded analysis selection without widening its consumer boundary."""

    policy = analysis_product_policy(
        product_id,
        target_label_namespace=target_label_namespace,
    )
    if registry.registry_id != snapshot.registry_id:
        raise ValueError(
            "analysis-product registry does not match snapshot: "
            f"{registry.registry_id!r} != {snapshot.registry_id!r}"
        )
    references = {reference.episode_id: reference for reference in snapshot.episodes}
    registry_sources = {source.source_dataset_id: source for source in registry.sources}
    materialized = tuple(sorted(episodes, key=lambda episode: episode.episode_id))
    if not materialized:
        raise ValueError("analysis-product selection requires at least one episode")

    selections: list[AnalysisProductEpisodeSelection] = []
    for episode in materialized:
        reference = references.get(episode.episode_id)
        if reference is None:
            raise ValueError(f"episode is not declared by snapshot: {episode.episode_id}")
        if episode.corpus_snapshot_id != snapshot.snapshot_id:
            raise ValueError(f"episode snapshot mismatch: {episode.episode_id}")
        if episode.source_dataset_id != reference.source_dataset_id:
            raise ValueError(f"episode source mismatch: {episode.episode_id}")
        if episode.corpus_sublane != reference.lane:
            raise ValueError(f"episode lane mismatch: {episode.episode_id}")
        source = registry_sources.get(episode.source_dataset_id)
        if source is None:
            raise ValueError(f"episode source is not in registry: {episode.source_dataset_id}")
        if source.lane != episode.corpus_sublane:
            raise ValueError(f"episode lane/source mismatch: {episode.episode_id}")
        if (
            policy.requires_prepared_sources
            and _EVIDENCE_RANK[source.evidence_state]
            < _EVIDENCE_RANK[SourceEvidenceState.PREPARED]
        ):
            raise ValueError(
                "classifier-ladder analysis requires prepared source: "
                f"{source.source_dataset_id}"
            )

        selected_views = tuple(
            view
            for view in episode.state_views
            if view.view_kind in policy.selected_state_view_kinds
        )
        classifier = episode.classifier_trajectory_view
        if policy.includes_classifier_view:
            if classifier is None:
                raise ValueError(
                    "classifier-ladder analysis requires a classifier view: "
                    f"{episode.episode_id}"
                )
            if not classifier.identity_and_grouping_values_excluded:
                raise ValueError(
                    "classifier-ladder analysis rejects identity-bearing classifier assets: "
                    f"{episode.episode_id}"
                )
            if not classifier.target_labels_stored_outside_asset:
                raise ValueError(
                    "classifier-ladder analysis requires labels outside the asset: "
                    f"{episode.episode_id}"
                )
            target_labels = tuple(
                label
                for label in episode.labels
                if label.namespace == policy.target_label_namespace
            )
            if policy.requires_target_labels and not target_labels:
                raise ValueError(
                    "classifier-ladder analysis requires target label namespace "
                    f"{policy.target_label_namespace!r}: {episode.episode_id}"
                )
            selections.append(
                AnalysisProductEpisodeSelection(
                    episode_id=episode.episode_id,
                    lane=episode.corpus_sublane,
                    source_dataset_id=episode.source_dataset_id,
                    classifier_asset=classifier.asset,
                    target_label_available=bool(target_labels),
                )
            )
            continue

        if not selected_views:
            raise ValueError(
                f"analysis product {policy.product_id.value!r} has no selected state view: "
                f"{episode.episode_id}"
            )
        selections.append(
            AnalysisProductEpisodeSelection(
                episode_id=episode.episode_id,
                lane=episode.corpus_sublane,
                source_dataset_id=episode.source_dataset_id,
                state_view_ids=tuple(view.state_view_id for view in selected_views),
                state_assets=tuple(view.sample_asset for view in selected_views),
            )
        )

    lane_counts: dict[str, int] = {}
    for selection in selections:
        lane_counts[selection.lane] = lane_counts.get(selection.lane, 0) + 1
    return AnalysisProductManifest(
        snapshot_id=snapshot.snapshot_id,
        snapshot_content_sha256=snapshot.content_sha256(),
        registry_id=snapshot.registry_id,
        policy=policy,
        episodes=tuple(selections),
        lane_episode_counts=dict(sorted(lane_counts.items())),
        source_dataset_ids=tuple(sorted({selection.source_dataset_id for selection in selections})),
    )


def write_analysis_product_manifest(
    manifest: AnalysisProductManifest,
    path: str | Path,
) -> None:
    """Write a selection manifest; the caller controls whether the destination is external."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")


__all__ = [
    "ANALYSIS_PRODUCT_CONTRACT_VERSION",
    "AnalysisProductEpisodeSelection",
    "AnalysisProductId",
    "AnalysisProductManifest",
    "AnalysisProductPolicy",
    "analysis_product_policy",
    "build_analysis_product_manifest",
    "write_analysis_product_manifest",
]
