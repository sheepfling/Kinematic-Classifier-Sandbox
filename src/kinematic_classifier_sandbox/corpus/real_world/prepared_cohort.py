"""Build task-scoped, identity-free classifier cohorts from normalized tracks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from pydantic import Field, model_validator

from ...common_experiment.contracts import ExecutablePairSpec
from .common_front_utils import opaque_group_id, quality_summary_from_elapsed, write_json_asset
from .contracts import LabelEvidence, NormalizedTrack
from .episode_contracts import (
    AccessClass,
    ChannelDescriptor,
    ClassifierTrajectoryView,
    DomainExtension,
    EvidenceStrength,
    FrameDescriptor,
    GroupingKey,
    GroupingNamespace,
    LabelAssertion,
    LabelEvidenceKind,
    ProgramDomain,
    StateRole,
    StateViewKind,
    StrictFrozenModel,
    TimeAxisDescriptor,
    TrajectoryEpisodeManifest,
    TrajectorySegment,
    TrajectoryStateViewManifest,
    ValueBasis,
)
from .portfolio import EpisodeSplitAssignment, SnapshotSplit
from .projection import ProjectionKind, ProjectionResult, project_pair_windows
from .splits import DatasetPartition, DatasetSplit, GroupSplitPolicy, assign_grouped_split
from .windowing import TrackWindow, WindowingPolicy, window_track


class PreparedClassifierCohortConfig(StrictFrozenModel):
    """Explicit policy for one prepared classifier task, not a whole portfolio."""

    corpus_snapshot_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    native_dataset_id: str | None = None
    source_artifact_id: str = Field(min_length=1)
    target_label_namespace: str = Field(min_length=1)
    pair_id: str = Field(min_length=1)
    class_a: str = Field(min_length=1)
    class_b: str = Field(min_length=1)
    expected_difficulty: str = Field(default="unknown", min_length=1)
    projection_kind: ProjectionKind = ProjectionKind.SPEED_PROFILE
    window_policy: WindowingPolicy
    split_policy: GroupSplitPolicy = Field(default_factory=GroupSplitPolicy)
    primary_program_domain: ProgramDomain = ProgramDomain.LAND
    corpus_sublane: str = Field(default="land_surface", min_length=1)
    operating_environment: str = Field(default="road_surface", min_length=1)
    motion_regime: str = Field(default="road_vehicle_motion", min_length=1)
    observation_modality: str = Field(default="optical_tracking", min_length=1)
    parse_step_id: str = Field(default="prepared-source-parse-v1", min_length=1)
    window_step_id: str = Field(default="prepared-task-window-v1", min_length=1)
    projection_step_id: str = Field(default="prepared-task-projection-v1", min_length=1)
    classifier_step_id: str = Field(default="prepared-classifier-view-v1", min_length=1)
    require_all_splits: bool = True
    require_each_class_per_split: bool = True

    @model_validator(mode="after")
    def validate_task(self) -> PreparedClassifierCohortConfig:
        if self.class_a == self.class_b:
            raise ValueError("prepared classifier task classes must be distinct")
        step_ids = (
            self.parse_step_id,
            self.window_step_id,
            self.projection_step_id,
            self.classifier_step_id,
        )
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("prepared classifier processing step IDs must be unique")
        return self

    def pair_spec(self) -> ExecutablePairSpec:
        return ExecutablePairSpec(
            pair_id=self.pair_id,
            class_a=self.class_a,
            class_b=self.class_b,
            expected_difficulty=self.expected_difficulty,
        )
    ####
####


@dataclass(frozen=True, slots=True)
class PreparedClassifierCohort:
    config: PreparedClassifierCohortConfig
    tracks: tuple[NormalizedTrack, ...]
    split: DatasetSplit
    projections: tuple[ProjectionResult, ...]
    episodes: tuple[TrajectoryEpisodeManifest, ...]

    def episode_assignments(self) -> tuple[EpisodeSplitAssignment, ...]:
        """Return snapshot assignments while preserving the track-level split groups."""

        partition_by_group = {
            assignment.split_group_id: assignment.partition
            for assignment in self.split.assignments
        }
        assignments: list[EpisodeSplitAssignment] = []
        for episode, projection in zip(self.episodes, self.projections, strict=True):
            partition = partition_by_group[projection.metadata.split_group_id]
            assignments.append(
                EpisodeSplitAssignment(
                    episode_id=episode.episode_id,
                    split=SnapshotSplit(partition.value),
                )
            )
        return tuple(assignments)
    ####
####


def _selected_tracks(
    tracks: tuple[NormalizedTrack, ...],
    *,
    config: PreparedClassifierCohortConfig,
) -> tuple[NormalizedTrack, ...]:
    allowed_dataset_ids = {config.dataset_id}
    if config.native_dataset_id is not None:
        allowed_dataset_ids.add(config.native_dataset_id)
    if any(track.provenance.dataset_id not in allowed_dataset_ids for track in tracks):
        raise ValueError("all prepared cohort tracks must belong to the declared source dataset")
    selected = tuple(
        track
        for track in tracks
        if track.labels.normalized_class in {config.class_a, config.class_b}
    )
    classes_present = {track.labels.normalized_class for track in selected}
    missing = {config.class_a, config.class_b} - classes_present
    if missing:
        raise ValueError(
            "prepared classifier cohort is missing class track(s): "
            + ", ".join(sorted(missing))
        )
    if any(track.labels.is_proxy for track in selected):
        raise ValueError("prepared classifier cohort cannot use proxy labels")
    return selected
####


def _active_partitions(policy: GroupSplitPolicy) -> tuple[DatasetPartition, ...]:
    ratios = policy.ratios
    return tuple(
        partition
        for partition in DatasetPartition
        if getattr(ratios, partition.value) > 0.0
    )
####


def _validate_split_coverage(
    tracks: tuple[NormalizedTrack, ...],
    windows: tuple[TrackWindow, ...],
    split: DatasetSplit,
    *,
    config: PreparedClassifierCohortConfig,
) -> None:
    window_groups = {window.split_group_id for window in windows}
    missing_window_groups = {
        track.provenance.split_group_id for track in tracks
    } - window_groups
    if missing_window_groups:
        raise ValueError(
            "prepared classifier cohort has no accepted windows for split group(s): "
            + ", ".join(sorted(missing_window_groups))
        )

    class_pair = {config.class_a, config.class_b}
    assignments_by_partition: dict[DatasetPartition, set[str]] = {
        partition: set() for partition in DatasetPartition
    }
    for assignment in split.assignments:
        assignments_by_partition[assignment.partition].add(assignment.normalized_class)

    if config.require_all_splits:
        missing_partitions = {
            partition
            for partition in _active_partitions(config.split_policy)
            if not assignments_by_partition[partition]
        }
        if missing_partitions:
            raise ValueError(
                "prepared classifier cohort is missing split partition(s): "
                + ", ".join(sorted(partition.value for partition in missing_partitions))
            )
    if config.require_each_class_per_split:
        for partition in _active_partitions(config.split_policy):
            if assignments_by_partition[partition] != class_pair:
                raise ValueError(
                    f"prepared classifier cohort requires both classes in {partition.value}"
                )
    ####
####


def _projection_units(projection_kind: ProjectionKind) -> str:
    if projection_kind is ProjectionKind.SPEED_PROFILE:
        return "m/s"
    if projection_kind is ProjectionKind.CUMULATIVE_PATH_LENGTH:
        return "m"
    raise ValueError(f"unsupported projection kind: {projection_kind}")
####


def _projection_source_fields(projection_kind: ProjectionKind) -> tuple[str, ...]:
    if projection_kind is ProjectionKind.SPEED_PROFILE:
        return ("derived_velocity_mps",)
    if projection_kind is ProjectionKind.CUMULATIVE_PATH_LENGTH:
        return ("position_m",)
    raise ValueError(f"unsupported projection kind: {projection_kind}")
####


def _frame(*, episode_id: str, projection_kind: ProjectionKind, units: str) -> FrameDescriptor:
    return FrameDescriptor(
        frame_id=f"{episode_id}:scalar-projection",
        frame_kind="scalar_kinematic_projection",
        axes=("measurement",),
        axis_units=(units,),
        center_or_origin="window-relative scalar projection origin",
        vertical_reference="not applicable to scalar projection",
        vertical_positive_direction="not applicable to scalar projection",
        crs_or_datum=None,
    )
####


def _time_axis() -> TimeAxisDescriptor:
    return TimeAxisDescriptor(
        source_time_system="relative seconds",
        normalized_time_system="elapsed SI seconds",
        absolute_time_available=False,
        elapsed_origin="first sample in the prepared window",
        precision_or_resolution="adapter-normalized source sample resolution",
        rollover_policy="none",
        leap_second_policy="not applicable",
    )
####


def _label_evidence_kind(evidence: LabelEvidence) -> LabelEvidenceKind:
    return {
        LabelEvidence.NATIVE: LabelEvidenceKind.NATIVE,
        LabelEvidence.DERIVED: LabelEvidenceKind.DERIVED,
        LabelEvidence.PROXY: LabelEvidenceKind.PROXY,
        LabelEvidence.WEAK: LabelEvidenceKind.RECONCILED,
    }[evidence]
####


def _grouping_keys(
    track: NormalizedTrack,
    *,
    dataset_id: str,
) -> tuple[GroupingKey, ...]:
    provenance = track.provenance
    return (
        GroupingKey(
            namespace=GroupingNamespace.PHYSICAL_PLATFORM,
            opaque_value=opaque_group_id(
                dataset_id=dataset_id,
                namespace=GroupingNamespace.PHYSICAL_PLATFORM.value,
                raw_value=f"{provenance.run_id}:{provenance.track_id}",
            ),
            scope="opaque physical platform grouping retained across task windows",
            evidence_strength=EvidenceStrength.STRONG,
        ),
        GroupingKey(
            namespace=GroupingNamespace.SOURCE_RECORDING,
            opaque_value=opaque_group_id(
                dataset_id=dataset_id,
                namespace=GroupingNamespace.SOURCE_RECORDING.value,
                raw_value=provenance.recording_id,
            ),
            scope="opaque source recording grouping",
            evidence_strength=EvidenceStrength.STRONG,
        ),
        GroupingKey(
            namespace=GroupingNamespace.SOURCE_DATASET,
            opaque_value=opaque_group_id(
                dataset_id=dataset_id,
                namespace=GroupingNamespace.SOURCE_DATASET.value,
                raw_value=dataset_id,
            ),
            scope="source dataset grouping",
            evidence_strength=EvidenceStrength.STRONG,
        ),
        GroupingKey(
            namespace=GroupingNamespace.GEOGRAPHY,
            opaque_value=opaque_group_id(
                dataset_id=dataset_id,
                namespace=GroupingNamespace.GEOGRAPHY.value,
                raw_value=provenance.location_id,
            ),
            scope="source geography retained for audit only",
            evidence_strength=EvidenceStrength.MEDIUM,
        ),
    )
####


def _episode_from_projection(
    track: NormalizedTrack,
    projection: ProjectionResult,
    window: TrackWindow,
    partition: DatasetPartition,
    *,
    output_root: str | Path,
    config: PreparedClassifierCohortConfig,
) -> TrajectoryEpisodeManifest:
    episode_id = (
        f"{config.dataset_id}:prepared:{config.projection_kind.value}:{window.window_id}"
    )
    units = _projection_units(config.projection_kind)
    frame = _frame(
        episode_id=episode_id,
        projection_kind=config.projection_kind,
        units=units,
    )
    payload = {
        "timestamps_s": list(projection.trajectory.times),
        "measurements": list(projection.trajectory.measurements),
    }
    asset_stem = hashlib.sha256(episode_id.encode("utf-8")).hexdigest()[:24]
    analysis_asset = write_json_asset(
        output_root,
        Path("assets/analysis") / f"{asset_stem}.json",
        payload,
    )
    classifier_asset = write_json_asset(
        output_root,
        Path("assets/classifier") / f"{asset_stem}.json",
        payload,
    )
    timestamps = projection.trajectory.times
    duration_s = float(timestamps[-1] - timestamps[0])
    projection_step = config.projection_step_id
    classifier_step = config.classifier_step_id
    state_view_id = f"{episode_id}:classifier-projection"
    state_view = TrajectoryStateViewManifest(
        state_view_id=state_view_id,
        view_kind=StateViewKind.ANALYSIS,
        state_role=StateRole.RECONSTRUCTION,
        value_basis=ValueBasis.DERIVED,
        frame=frame,
        source_time_axis=_time_axis(),
        sample_count=len(timestamps),
        sample_asset=analysis_asset,
        channel_descriptors=(
            ChannelDescriptor(
                channel_id="scalar-measurement",
                semantic_role=config.projection_kind.value,
                component_names=("measurement",),
                units=(units,),
                frame_id=frame.frame_id,
                state_role=StateRole.RECONSTRUCTION,
                value_basis=ValueBasis.DERIVED,
                access_class=AccessClass.CLASSIFIER_CANDIDATE,
                source_fields=_projection_source_fields(config.projection_kind),
                lineage_step_ids=(projection_step,),
                notes="Task-scoped scalar projection; identity and target labels are excluded.",
            ),
        ),
        processing_step_ids=(projection_step,),
    )
    provenance = track.provenance
    platform_group = next(
        key.opaque_value
        for key in _grouping_keys(track, dataset_id=config.dataset_id)
        if key.namespace is GroupingNamespace.PHYSICAL_PLATFORM
    )
    label = LabelAssertion(
        assertion_id=f"{episode_id}:target-label",
        namespace=config.target_label_namespace,
        value=track.labels.normalized_class,
        evidence_kind=_label_evidence_kind(track.labels.evidence),
        evidence_strength=EvidenceStrength.STRONG,
        source_reference=config.source_artifact_id,
        proxy=False,
        start_offset_s=0.0,
        end_offset_s=duration_s,
        vocabulary_version="tgsim-normalized-vehicle-v1",
        notes="Out-of-band target label for the declared task; excluded from classifier assets.",
    )
    return TrajectoryEpisodeManifest(
        corpus_snapshot_id=config.corpus_snapshot_id,
        episode_id=episode_id,
        primary_program_domain=config.primary_program_domain,
        corpus_sublane=config.corpus_sublane,
        default_operating_environment=config.operating_environment,
        default_motion_regime=config.motion_regime,
        source_dataset_id=config.dataset_id,
        source_artifact_ids=(config.source_artifact_id,),
        observation_modality=config.observation_modality,
        platform_group_id=platform_group,
        mission_id=provenance.run_id,
        object_id=provenance.track_id,
        start_time=str(window.start_time_s),
        end_time=str(window.end_time_s),
        state_views=(state_view,),
        segments=(
            TrajectorySegment(
                segment_id=f"{episode_id}:window",
                start_offset_s=0.0,
                end_offset_s=duration_s,
                operating_environment=config.operating_environment,
                motion_regime=config.motion_regime,
                evidence_kind="prepared_task_window",
            ),
        ),
        labels=(label,),
        grouping_keys=_grouping_keys(track, dataset_id=config.dataset_id),
        quality_summary=quality_summary_from_elapsed(
            list(timestamps),
            disposition="accept_with_findings",
        ),
        processing_step_ids=(
            config.parse_step_id,
            config.window_step_id,
            config.projection_step_id,
            config.classifier_step_id,
        ),
        domain_extension=DomainExtension(
            schema_id="real_world_prepared_classifier_cohort_v0.1",
            schema_version="0.1.0",
            payload={
                "task_pair_id": config.pair_id,
                "target_label_namespace": config.target_label_namespace,
                "projection_kind": config.projection_kind.value,
                "window_duration_s": window.requested_duration_s,
                "partition": partition.value,
                "claim_boundary": (
                    "Bounded task-scoped held-out cohort; no source-shift or six-lane "
                    "generalization claim."
                ),
            },
        ),
        classifier_trajectory_view=ClassifierTrajectoryView(
            episode_id=episode_id,
            state_view_id=state_view_id,
            asset=classifier_asset,
            sample_count=len(timestamps),
            frame_id=frame.frame_id,
            processing_step_ids=(classifier_step,),
            target_labels_stored_outside_asset=True,
            identity_and_grouping_values_excluded=True,
        ),
    )
####


def build_prepared_classifier_cohort(
    tracks: tuple[NormalizedTrack, ...],
    *,
    output_root: str | Path,
    config: PreparedClassifierCohortConfig,
) -> PreparedClassifierCohort:
    """Window, split, project, and materialize a governed classifier cohort.

    The source track is the split unit. Each window inherits its physical-platform and
    source-recording grouping keys, while the classifier asset contains only relative time and
    scalar measurements. Labels, provenance, and audit metadata remain in the episode manifest.
    """

    selected_tracks = _selected_tracks(tracks, config=config)
    pair_spec = config.pair_spec()
    windowing_results = tuple(
        window_track(track, policy=config.window_policy) for track in selected_tracks
    )
    windows = tuple(
        window
        for result in windowing_results
        for window in result.windows
    )
    if not windows:
        raise ValueError("prepared classifier cohort has no accepted windows")
    split = assign_grouped_split(
        selected_tracks,
        windows=windows,
        policy=config.split_policy,
    )
    _validate_split_coverage(selected_tracks, windows, split, config=config)
    projections = project_pair_windows(
        selected_tracks,
        windows,
        pair_spec=pair_spec,
        projection_kind=config.projection_kind,
        split=split,
    )
    if len(projections) != len(windows):
        raise RuntimeError("prepared classifier projection count does not match window count")
    windows_by_id = {window.window_id: window for window in windows}
    tracks_by_group = {
        track.provenance.split_group_id: track for track in selected_tracks
    }
    partition_by_group = {
        assignment.split_group_id: assignment.partition
        for assignment in split.assignments
    }
    episodes = tuple(
        _episode_from_projection(
            tracks_by_group[projection.metadata.split_group_id],
            projection,
            windows_by_id[projection.metadata.window_id],
            partition_by_group[projection.metadata.split_group_id],
            output_root=output_root,
            config=config,
        )
        for projection in projections
    )
    return PreparedClassifierCohort(
        config=config,
        tracks=selected_tracks,
        split=split,
        projections=projections,
        episodes=episodes,
    )


__all__ = [
    "PreparedClassifierCohort",
    "PreparedClassifierCohortConfig",
    "build_prepared_classifier_cohort",
]
