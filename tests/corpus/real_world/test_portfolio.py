from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from kinematic_classifier_sandbox.corpus.real_world.episode_contracts import (
    AssetReference,
    ChannelDescriptor,
    ClassifierTrajectoryView,
    EvidenceStrength,
    FrameDescriptor,
    GroupingKey,
    GroupingNamespace,
    LabelAssertion,
    LabelEvidenceKind,
    ProgramDomain,
    QualitySummary,
    StateRole,
    StateViewKind,
    TimeAxisDescriptor,
    TrajectoryEpisodeManifest,
    TrajectoryStateViewManifest,
    ValueBasis,
)
from kinematic_classifier_sandbox.corpus.real_world.portfolio import (
    REAL_WORLD_CORPUS_LANES,
    CorpusSnapshotManifest,
    EpisodeSplitAssignment,
    SnapshotEpisodeReference,
    SnapshotSelectionPolicy,
    SnapshotSplit,
    SourceArtifactRecord,
    SourceEvidenceState,
    SourceRegistry,
    SourceRegistryEntry,
    audit_split_assignments,
    evaluate_snapshot,
    evaluate_source_registry,
    load_snapshot_episodes,
    load_snapshot_manifest,
    load_source_registry,
    select_snapshot_episodes,
    write_snapshot_manifest,
)

_SHA = "0" * 64
_REGISTRY = Path(__file__).parents[3] / "docs" / "product4" / "real_world_source_registry.yaml"


def _episode(
    *,
    snapshot_id: str = "snapshot-v0.1",
    episode_id: str = "land-episode-1",
    lane: str = "land_surface",
    domain: ProgramDomain = ProgramDomain.LAND,
    platform_group: str = "platform-1",
    classifier: bool = True,
) -> TrajectoryEpisodeManifest:
    frame = FrameDescriptor(
        frame_id=f"{episode_id}-frame",
        frame_kind="local_cartesian",
        axes=("x", "y"),
        axis_units=("m", "m"),
        center_or_origin="fixture origin",
        vertical_reference="unavailable",
        vertical_positive_direction="unavailable",
    )
    time_axis = TimeAxisDescriptor(
        source_time_system="relative seconds",
        normalized_time_system="elapsed SI seconds",
        absolute_time_available=False,
        elapsed_origin="fixture start",
        precision_or_resolution="0.1 s",
        rollover_policy="none",
        leap_second_policy="not applicable",
    )
    view = TrajectoryStateViewManifest(
        state_view_id=f"{episode_id}-observed",
        view_kind=StateViewKind.ANALYSIS,
        state_role=StateRole.OBSERVATION,
        value_basis=ValueBasis.REPORTED,
        frame=frame,
        source_time_axis=time_axis,
        sample_count=2,
        sample_asset=AssetReference(
            path=f"states/{episode_id}.json",
            media_type="application/json",
            sha256=_SHA,
        ),
        channel_descriptors=(
            ChannelDescriptor(
                channel_id=f"{episode_id}-position",
                semantic_role="position",
                component_names=("x", "y"),
                units=("m", "m"),
                frame_id=frame.frame_id,
                state_role=StateRole.OBSERVATION,
                value_basis=ValueBasis.REPORTED,
                access_class="classifier_candidate",
            ),
        ),
        processing_step_ids=("load",),
    )
    classifier_view = None
    if classifier:
        classifier_view = ClassifierTrajectoryView(
            episode_id=episode_id,
            state_view_id=view.state_view_id,
            asset=AssetReference(
                path=f"classifier/{episode_id}.npz",
                media_type="application/x-npz",
                sha256=_SHA,
            ),
            sample_count=2,
            frame_id=frame.frame_id,
            processing_step_ids=("load",),
            target_labels_stored_outside_asset=True,
            identity_and_grouping_values_excluded=True,
        )
    return TrajectoryEpisodeManifest(
        corpus_snapshot_id=snapshot_id,
        episode_id=episode_id,
        primary_program_domain=domain,
        corpus_sublane=lane,
        default_operating_environment="fixture",
        default_motion_regime="steady",
        source_dataset_id=f"{lane}-dataset",
        source_artifact_ids=(f"{lane}-artifact",),
        observation_modality="fixture",
        platform_group_id=platform_group,
        start_time=None,
        end_time=None,
        state_views=(view,),
        labels=(
            LabelAssertion(
                assertion_id=f"{episode_id}-label",
                namespace="class",
                value="class_a",
                evidence_kind=LabelEvidenceKind.NATIVE,
                evidence_strength=EvidenceStrength.STRONG,
                source_reference="fixture",
                proxy=False,
            ),
        ),
        grouping_keys=(
            GroupingKey(
                namespace=GroupingNamespace.PHYSICAL_PLATFORM,
                opaque_value=platform_group,
                scope="fixture",
                evidence_strength=EvidenceStrength.STRONG,
            ),
            GroupingKey(
                namespace=GroupingNamespace.SOURCE_RECORDING,
                opaque_value=f"recording-{episode_id}",
                scope="fixture",
                evidence_strength=EvidenceStrength.STRONG,
            ),
        ),
        quality_summary=QualitySummary(
            disposition="accept_with_findings",
            sample_count=2,
            duration_s=1.0,
            median_sample_interval_s=1.0,
            maximum_gap_s=1.0,
            duplicate_timestamp_count=0,
            out_of_order_timestamp_count=0,
        ),
        processing_step_ids=("load",),
        classifier_trajectory_view=classifier_view,
    )


def _snapshot(tmp_path, episodes: tuple[TrajectoryEpisodeManifest, ...]) -> CorpusSnapshotManifest:
    for episode in episodes:
        path = tmp_path / "episodes" / f"{episode.episode_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(episode.model_dump_json(indent=2), encoding="utf-8")

    def episode_sha256(episode: TrajectoryEpisodeManifest) -> str:
        path = tmp_path / "episodes" / f"{episode.episode_id}.json"
        return hashlib.sha256(path.read_bytes()).hexdigest()

    refs = tuple(
        SnapshotEpisodeReference(
            episode_id=episode.episode_id,
            lane=episode.corpus_sublane,
            source_dataset_id=episode.source_dataset_id,
            manifest=AssetReference(
                path=f"episodes/{episode.episode_id}.json",
                media_type="application/json",
                sha256=episode_sha256(episode),
            ),
        )
        for episode in episodes
    )
    manifest = CorpusSnapshotManifest(
        snapshot_id="snapshot-v0.1",
        registry_id="registry-v0.1",
        created_at=datetime(2026, 8, 25, tzinfo=timezone.utc),
        episodes=refs,
        source_artifact_ids=tuple(episode.source_artifact_ids[0] for episode in episodes),
        adapter_versions=("fixture:0.1.0",),
    )
    write_snapshot_manifest(manifest, tmp_path / "snapshot.json")
    return manifest


def test_registry_rejects_duplicate_source_ids() -> None:
    source = SourceRegistryEntry(
        source_dataset_id="land-dataset",
        lane="land_surface",
        domain=ProgramDomain.LAND,
        title="Land fixture",
        adapter_id="land-fixture",
        adapter_version="0.1.0",
        evidence_state=SourceEvidenceState.FIXTURE_VALIDATED,
        artifacts=(
            SourceArtifactRecord(
                artifact_id="land-artifact",
                uri_or_query="fixture://land",
                media_type="application/json",
            ),
        ),
        grouping_namespaces=(GroupingNamespace.PHYSICAL_PLATFORM,),
    )

    with pytest.raises(ValueError, match="source dataset IDs must be unique"):
        SourceRegistry(registry_id="registry-v0.1", sources=(source, source))


def test_snapshot_round_trip_selects_by_lane_and_evaluates_coverage(tmp_path) -> None:
    land = _episode(episode_id="land-episode-1")
    air = _episode(
        episode_id="air-episode-1",
        lane="air_atmospheric",
        domain=ProgramDomain.AIR,
        platform_group="aircraft-1",
        classifier=False,
    )
    manifest = _snapshot(tmp_path, (land, air))
    loaded_manifest = load_snapshot_manifest(tmp_path / "snapshot.json")
    loaded_episodes = load_snapshot_episodes(loaded_manifest, tmp_path / "snapshot.json")

    selected = select_snapshot_episodes(
        loaded_episodes,
        SnapshotSelectionPolicy(lanes=("land_surface",), require_classifier_view=True),
    )
    report = evaluate_snapshot(
        manifest,
        loaded_episodes,
        expected_lanes=("land_surface", "air_atmospheric"),
    )

    assert manifest.content_sha256() == loaded_manifest.content_sha256()
    assert [episode.episode_id for episode in selected] == ["land-episode-1"]
    assert report.passes is True
    assert report.lane_episode_counts == {"land_surface": 1, "air_atmospheric": 1}
    assert report.classifier_ready_episode_count == 1
    assert report.source_dataset_counts == {
        "land_surface-dataset": 1,
        "air_atmospheric-dataset": 1,
    }
    assert report.state_role_counts == {"observation": 2}
    assert report.label_evidence_counts == {"native": 2}
    assert report.group_namespace_counts == {
        "physical_platform": 2,
        "source_recording": 2,
    }


def test_snapshot_loader_rejects_mutated_episode_asset(tmp_path) -> None:
    episode = _episode(episode_id="episode-1")
    _snapshot(tmp_path, (episode,))
    episode_path = tmp_path / "episodes" / "episode-1.json"
    episode_path.write_text(episode_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    manifest = load_snapshot_manifest(tmp_path / "snapshot.json")
    with pytest.raises(ValueError, match="snapshot episode hash mismatch"):
        load_snapshot_episodes(manifest, tmp_path / "snapshot.json")


def test_split_audit_rejects_physical_platform_cross_split_collision() -> None:
    first = _episode(episode_id="episode-1", platform_group="shared-platform")
    second = _episode(episode_id="episode-2", platform_group="shared-platform")
    report = audit_split_assignments(
        (first, second),
        (
            EpisodeSplitAssignment(episode_id="episode-1", split=SnapshotSplit.TRAIN),
            EpisodeSplitAssignment(episode_id="episode-2", split=SnapshotSplit.TEST),
        ),
    )

    assert report.passes is False
    assert any(issue.startswith("group_split_collision") for issue in report.issues)


def test_canonical_registry_covers_six_lanes_and_reports_open_promotion_gates() -> None:
    registry = load_source_registry(_REGISTRY)
    report = evaluate_source_registry(registry)

    assert report.passes is True
    assert report.covered_lanes == REAL_WORLD_CORPUS_LANES
    assert report.missing_lanes == ()
    assert report.fixture_validated_lanes == (
        "sea_surface",
        "space_near",
        "space_orbital",
    )
    assert report.prepared_lanes == ()
    assert report.classifier_ready is False
    assert len(report.open_gates) == len(REAL_WORLD_CORPUS_LANES)
    assert report.lane_best_evidence_states["sea_subsurface"] == "mapping_complete"
    assert report.lane_best_evidence_states["air_atmospheric"] == "access_verified"
