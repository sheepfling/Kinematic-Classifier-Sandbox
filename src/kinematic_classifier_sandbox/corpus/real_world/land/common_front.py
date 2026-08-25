"""Convert the bounded TGSIM adapter output into validation-only episodes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..adapters.tgsim import load_tgsim_foggy_bottom_csv
from ..adapters.tgsim_contracts import TgsimLoadResult
from ..common_front_utils import (
    opaque_group_id,
    quality_summary_from_elapsed,
    sha256_file,
    write_json_asset,
)
from ..contracts import LabelEvidence, NormalizedTrack
from ..episode_contracts import (
    AccessClass,
    ChannelDescriptor,
    DomainExtension,
    EvidenceStrength,
    FrameDescriptor,
    GroupingKey,
    GroupingNamespace,
    LabelAssertion,
    LabelEvidenceKind,
    ProgramDomain,
    QualityFinding,
    QualitySeverity,
    StateRole,
    StateViewKind,
    TimeAxisDescriptor,
    TrajectoryEpisodeManifest,
    TrajectorySegment,
    TrajectoryStateViewManifest,
    ValueBasis,
)

DATASET_ID = "fhwa_tgsim_foggy_bottom"
SOURCE_ARTIFACT_ID = "fhwa_tgsim_foggy_bottom_trajectory_csv"
PARSE_STEP_ID = "tgsim-common-front-parse-v1"
ANALYSIS_STEP_ID = "tgsim-common-front-analysis-v1"


def _frame(episode_id: str) -> FrameDescriptor:
    return FrameDescriptor(
        frame_id=f"{episode_id}:tgsim-planar",
        frame_kind="local_cartesian",
        axes=("x", "y", "z"),
        axis_units=("m", "m", "m"),
        center_or_origin="top-left of the TGSIM Foggy Bottom reference image",
        vertical_reference="source fixture has no measured vertical state; z=0 normalization",
        vertical_positive_direction="source-defined planar up",
        crs_or_datum="source-defined image-referenced metric plane",
    )


def _time_axis() -> TimeAxisDescriptor:
    return TimeAxisDescriptor(
        source_time_system="relative seconds",
        normalized_time_system="elapsed SI seconds",
        absolute_time_available=False,
        elapsed_origin="first source sample in the track",
        precision_or_resolution="source-declared subsecond samples",
        rollover_policy="none",
        leap_second_policy="not applicable",
    )


def _channel(
    *,
    channel_id: str,
    semantic_role: str,
    components: tuple[str, ...],
    units: tuple[str, ...],
    frame_id: str,
    state_role: StateRole,
    value_basis: ValueBasis,
    access_class: AccessClass,
    source_fields: tuple[str, ...],
    lineage: tuple[str, ...],
    notes: str | None = None,
) -> ChannelDescriptor:
    return ChannelDescriptor(
        channel_id=channel_id,
        semantic_role=semantic_role,
        component_names=components,
        units=units,
        frame_id=frame_id,
        state_role=state_role,
        value_basis=value_basis,
        access_class=access_class,
        source_fields=source_fields,
        lineage_step_ids=lineage,
        notes=notes,
    )


def _track_payload(track: NormalizedTrack, *, analysis: bool) -> dict[str, Any]:
    channels = {
        channel.name: {
            "units": channel.units,
            "role": channel.role.value,
            "values": channel.values.tolist(),
        }
        for channel in track.numeric_channels
    }
    payload: dict[str, Any] = {
        "timestamps_s": track.timestamps_s.tolist(),
        "position_m": track.position_m.tolist(),
        "metadata": dict(track.metadata),
    }
    if analysis:
        payload.update(
            {
                "derived_velocity_mps": track.derived_velocity_mps.tolist(),
                "derived_acceleration_mps2": track.derived_acceleration_mps2.tolist(),
                "channels": channels,
            }
        )
    else:
        payload.update(
            {
                "source_velocity_mps": (
                    track.source_velocity_mps.tolist()
                    if track.source_velocity_mps is not None
                    else None
                ),
                "source_acceleration_mps2": (
                    track.source_acceleration_mps2.tolist()
                    if track.source_acceleration_mps2 is not None
                    else None
                ),
                "channels": channels,
            }
        )
    return payload


def _label(track: NormalizedTrack, *, source_artifact_id: str) -> LabelAssertion:
    evidence_kind = {
        LabelEvidence.NATIVE: LabelEvidenceKind.NATIVE,
        LabelEvidence.DERIVED: LabelEvidenceKind.DERIVED,
        LabelEvidence.PROXY: LabelEvidenceKind.PROXY,
        LabelEvidence.WEAK: LabelEvidenceKind.RECONCILED,
    }[track.labels.evidence]
    strength = (
        EvidenceStrength.WEAK
        if track.labels.evidence is LabelEvidence.WEAK
        else EvidenceStrength.STRONG
    )
    return LabelAssertion(
        assertion_id=f"{track.provenance.track_id}:platform-class",
        namespace="platform_class",
        value=track.labels.normalized_class,
        evidence_kind=evidence_kind,
        evidence_strength=strength,
        source_reference=source_artifact_id,
        proxy=track.labels.is_proxy,
        vocabulary_version="tgsim-normalized-vehicle-v1",
        notes="Source label is retained for validation; no classifier projection is emitted.",
    )


def _quality(track: NormalizedTrack, *, source_artifact_id: str):
    quality = track.quality
    if quality is None:
        return quality_summary_from_elapsed(track.timestamps_s.tolist())
    findings = [
        QualityFinding(
            code="land_adapter_note",
            severity=QualitySeverity.INFO,
            message=message,
            source_reference=source_artifact_id,
        )
        for message in quality.findings
    ]
    if quality.gap_count:
        findings.append(
            QualityFinding(
                code="land_sampling_gap",
                severity=QualitySeverity.WARNING,
                message="TGSIM track contains one or more policy-defined sampling gaps.",
                value=quality.gap_count,
                source_reference=source_artifact_id,
            )
        )
    summary = quality_summary_from_elapsed(
        track.timestamps_s.tolist(),
        findings=findings,
        disposition="accept_with_findings",
    )
    return summary.model_copy(
        update={
            "maximum_gap_s": quality.max_dt_s,
            "median_sample_interval_s": quality.median_dt_s,
        }
    )


def _build_episode(
    track: NormalizedTrack,
    *,
    output_root: str | Path,
    corpus_snapshot_id: str,
    dataset_id: str,
    source_artifact_id: str,
    source_sha256: str,
) -> TrajectoryEpisodeManifest:
    provenance = track.provenance
    episode_id = f"land:tgsim:{provenance.run_id}:{provenance.track_id}"
    frame = _frame(episode_id)
    source_asset = write_json_asset(
        output_root,
        Path("assets/source_native") / f"{episode_id}.json",
        _track_payload(track, analysis=False),
    )
    analysis_asset = write_json_asset(
        output_root,
        Path("assets/analysis") / f"{episode_id}.json",
        _track_payload(track, analysis=True),
    )
    platform_group = opaque_group_id(
        dataset_id=dataset_id,
        namespace=GroupingNamespace.PHYSICAL_PLATFORM.value,
        raw_value=f"{provenance.run_id}:{provenance.track_id}",
    )
    recording_group = opaque_group_id(
        dataset_id=dataset_id,
        namespace=GroupingNamespace.SOURCE_RECORDING.value,
        raw_value=provenance.recording_id,
    )
    source_dataset_group = opaque_group_id(
        dataset_id=dataset_id,
        namespace=GroupingNamespace.SOURCE_DATASET.value,
        raw_value=dataset_id,
    )
    timestamps = track.timestamps_s.tolist()
    return TrajectoryEpisodeManifest(
        corpus_snapshot_id=corpus_snapshot_id,
        episode_id=episode_id,
        primary_program_domain=ProgramDomain.LAND,
        corpus_sublane="land_surface",
        default_operating_environment="road_surface",
        default_motion_regime="road_vehicle_motion",
        source_dataset_id=dataset_id,
        source_artifact_ids=(source_artifact_id,),
        observation_modality="optical_tracking",
        platform_group_id=platform_group,
        mission_id=provenance.run_id,
        object_id=None,
        state_views=(
            TrajectoryStateViewManifest(
                state_view_id=f"{episode_id}:source-native",
                view_kind=StateViewKind.SOURCE_NATIVE,
                state_role=StateRole.OBSERVATION,
                value_basis=ValueBasis.REPORTED,
                frame=frame,
                source_time_axis=_time_axis(),
                sample_count=len(timestamps),
                sample_asset=source_asset,
                channel_descriptors=(
                    _channel(
                        channel_id="position",
                        semantic_role="position",
                        components=("x", "y", "z"),
                        units=("m", "m", "m"),
                        frame_id=frame.frame_id,
                        state_role=StateRole.OBSERVATION,
                        value_basis=ValueBasis.REPORTED,
                        access_class=AccessClass.CONTEXT,
                        source_fields=("xloc_kf", "yloc_kf"),
                        lineage=(PARSE_STEP_ID,),
                    ),
                    _channel(
                        channel_id="source-velocity",
                        semantic_role="velocity",
                        components=("x", "y", "z"),
                        units=("m/s", "m/s", "m/s"),
                        frame_id=frame.frame_id,
                        state_role=StateRole.OBSERVATION,
                        value_basis=ValueBasis.REPORTED,
                        access_class=AccessClass.AUDIT_ONLY,
                        source_fields=("speed_kf_x", "speed_kf_y"),
                        lineage=(PARSE_STEP_ID,),
                    ),
                    _channel(
                        channel_id="source-acceleration",
                        semantic_role="acceleration",
                        components=("x", "y", "z"),
                        units=("m/s^2", "m/s^2", "m/s^2"),
                        frame_id=frame.frame_id,
                        state_role=StateRole.OBSERVATION,
                        value_basis=ValueBasis.REPORTED,
                        access_class=AccessClass.AUDIT_ONLY,
                        source_fields=("acceleration_kf_x", "acceleration_kf_y"),
                        lineage=(PARSE_STEP_ID,),
                    ),
                ),
                processing_step_ids=(PARSE_STEP_ID,),
            ),
            TrajectoryStateViewManifest(
                state_view_id=f"{episode_id}:analysis",
                view_kind=StateViewKind.ANALYSIS,
                state_role=StateRole.RECONSTRUCTION,
                value_basis=ValueBasis.DERIVED,
                frame=frame,
                source_time_axis=_time_axis(),
                sample_count=len(timestamps),
                sample_asset=analysis_asset,
                channel_descriptors=(
                    _channel(
                        channel_id="derived-kinematics",
                        semantic_role="kinematics",
                        components=("position", "velocity", "acceleration"),
                        units=("m", "m/s", "m/s^2"),
                        frame_id=frame.frame_id,
                        state_role=StateRole.RECONSTRUCTION,
                        value_basis=ValueBasis.DERIVED,
                        access_class=AccessClass.CLASSIFIER_CANDIDATE,
                        source_fields=("position_m", "derived_velocity_mps", "derived_acceleration_mps2"),
                        lineage=(ANALYSIS_STEP_ID,),
                        notes="Validation-only analysis view; no classifier asset is emitted.",
                    ),
                ),
                processing_step_ids=(ANALYSIS_STEP_ID,),
            ),
        ),
        segments=(
            TrajectorySegment(
                segment_id=f"{episode_id}:full-track",
                start_offset_s=float(timestamps[0]),
                end_offset_s=float(timestamps[-1]),
                operating_environment="road_surface",
                motion_regime="road_vehicle_motion",
                evidence_kind="source_track_extent",
            ),
        ),
        labels=(_label(track, source_artifact_id=source_artifact_id),),
        grouping_keys=(
            GroupingKey(
                namespace=GroupingNamespace.PHYSICAL_PLATFORM,
                opaque_value=platform_group,
                scope="opaque TGSIM run and track grouping",
                evidence_strength=EvidenceStrength.STRONG,
            ),
            GroupingKey(
                namespace=GroupingNamespace.SOURCE_RECORDING,
                opaque_value=recording_group,
                scope="TGSIM source recording",
                evidence_strength=EvidenceStrength.STRONG,
            ),
            GroupingKey(
                namespace=GroupingNamespace.SOURCE_DATASET,
                opaque_value=source_dataset_group,
                scope="source dataset",
                evidence_strength=EvidenceStrength.STRONG,
            ),
            GroupingKey(
                namespace=GroupingNamespace.GEOGRAPHY,
                opaque_value=opaque_group_id(
                    dataset_id=dataset_id,
                    namespace=GroupingNamespace.GEOGRAPHY.value,
                    raw_value=provenance.location_id,
                ),
                scope="source location",
                evidence_strength=EvidenceStrength.MEDIUM,
            ),
        ),
        quality_summary=_quality(track, source_artifact_id=source_artifact_id),
        processing_step_ids=(PARSE_STEP_ID, ANALYSIS_STEP_ID),
        domain_extension=DomainExtension(
            schema_id="land_tgsim_common_front_v0.1",
            schema_version="0.1.0",
            payload={
                "fixture_status": "validation_only",
                "source_asset_sha256": source_sha256,
                "source_asset_id": source_artifact_id,
                "source_claim_boundary": (
                    "Bounded validation input exercises the common contract; it does not "
                    "represent a complete acquired TGSIM source artifact."
                ),
                "classifier_view_status": "intentionally_blocked",
            },
        ),
        classifier_trajectory_view=None,
    )


def build_tgsim_fixture_episodes(
    source_path: str | Path,
    *,
    output_root: str | Path,
    corpus_snapshot_id: str,
    dataset_id: str = DATASET_ID,
    source_artifact_id: str = SOURCE_ARTIFACT_ID,
) -> tuple[TrajectoryEpisodeManifest, ...]:
    """Materialize bounded TGSIM validation episodes under an external root."""

    source = Path(source_path)
    result: TgsimLoadResult = load_tgsim_foggy_bottom_csv(source)
    source_sha256 = sha256_file(source)
    episodes = tuple(
        _build_episode(
            track,
            output_root=output_root,
            corpus_snapshot_id=corpus_snapshot_id,
            dataset_id=dataset_id,
            source_artifact_id=source_artifact_id,
            source_sha256=source_sha256,
        )
        for track in result.tracks
    )
    episodes_root = Path(output_root) / "episodes"
    episodes_root.mkdir(parents=True, exist_ok=True)
    for episode in episodes:
        (episodes_root / f"{episode.episode_id}.json").write_text(
            episode.model_dump_json(indent=2, exclude_none=True) + "\n",
            encoding="utf-8",
        )
    return episodes


__all__ = ["build_tgsim_fixture_episodes"]
