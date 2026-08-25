"""Convert the bounded NASA ISS OEM fixture into a validation-only episode."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..adapters.space_orbital_oem import NasaIssOemCorpusAdapter, canonicalize_cospar_id
from ..adapters.space_orbital_oem_parsing import parse_oem_file, records_to_si_arrays
from ..common_front_utils import (
    iso_utc_from_unix,
    opaque_group_id,
    quality_summary_from_elapsed,
    sha256_file,
    write_json_asset,
)
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

DATASET_ID = "nasa-topo-iss-oem-2022-04-27-bounded-fixture"
SOURCE_ARTIFACT_ID = "nasa_topo_iss_oem_20220427_bounded_extract"
PARSE_STEP_ID = "nasa-iss-oem-parse-v1"
ANALYSIS_STEP_ID = "nasa-iss-oem-common-front-analysis-v1"


def _frame(episode_id: str, *, analysis: bool) -> FrameDescriptor:
    return FrameDescriptor(
        frame_id=f"{episode_id}:eme2000-{'analysis' if analysis else 'source'}",
        frame_kind="eci",
        axes=("x", "y", "z"),
        axis_units=("m", "m", "m") if analysis else ("km", "km", "km"),
        center_or_origin="Earth center",
        vertical_reference="not applicable for ECI state",
        vertical_positive_direction="right-handed EME2000 axes",
        crs_or_datum="EME2000",
        reference_epoch="J2000-compatible EME2000 convention",
        earth_orientation_or_transform_model="source-declared EME2000; no additional transform",
    )


def _time_axis(first_epoch: float) -> TimeAxisDescriptor:
    return TimeAxisDescriptor(
        source_time_system="UTC",
        normalized_time_system="elapsed SI seconds",
        absolute_time_available=True,
        source_epoch_or_reference=iso_utc_from_unix(first_epoch),
        elapsed_origin=iso_utc_from_unix(first_epoch),
        precision_or_resolution="OEM epoch millisecond precision",
        rollover_policy="none",
        leap_second_policy="UTC source timestamps; no leap-second rewrite",
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


def _quality(track: Any, *, source_artifact_id: str):
    findings = tuple(
        QualityFinding(
            code=f"SPACE_ORB_NOTE_{index}",
            severity=QualitySeverity.INFO,
            message=message,
            source_reference=source_artifact_id,
        )
        for index, message in enumerate(track.quality.findings, start=1)
    )
    return quality_summary_from_elapsed(
        track.timestamps_s.tolist(),
        findings=findings,
        disposition="usable_with_restrictions",
    )


def build_nasa_iss_oem_episode(
    source_path: str | Path,
    *,
    output_root: str | Path,
    corpus_snapshot_id: str,
    source_artifact_id: str = SOURCE_ARTIFACT_ID,
) -> TrajectoryEpisodeManifest:
    """Write a source-faithful orbital episode with classifier projection blocked."""

    source = Path(source_path)
    adapter = NasaIssOemCorpusAdapter()
    corpus = adapter.load_corpus(source)
    if len(corpus) != 1:
        raise ValueError("NASA ISS OEM fixture must produce exactly one corpus trajectory")
    track = corpus[0].trajectory
    extract = parse_oem_file(source)
    timestamps_s, position_m, source_velocity_mps = records_to_si_arrays(extract.records)
    canonical_object_id = canonicalize_cospar_id(extract.metadata.source_object_id)
    platform_group = opaque_group_id(
        dataset_id=DATASET_ID,
        namespace=GroupingNamespace.PHYSICAL_PLATFORM.value,
        raw_value=canonical_object_id,
    )
    episode_id = f"space_orb:nasa_iss:{platform_group}:bounded_arc"
    source_frame = _frame(episode_id, analysis=False)
    analysis_frame = _frame(episode_id, analysis=True)
    first_epoch = float(timestamps_s[0])
    elapsed = (timestamps_s - first_epoch).tolist()
    source_asset = write_json_asset(
        output_root,
        Path("assets/source_native") / f"{episode_id}.json",
        {
            "header": {
                "version": extract.header.version,
                "creation_date_utc": extract.header.creation_date_utc.isoformat(),
                "originator": extract.header.originator,
            },
            "metadata": {
                "object_name": extract.metadata.object_name,
                "source_object_id": extract.metadata.source_object_id,
                "center_name": extract.metadata.center_name,
                "reference_frame": extract.metadata.reference_frame,
                "time_system": extract.metadata.time_system,
            },
            "records": [
                {
                    "epoch_utc": record.epoch_utc.isoformat(),
                    "position_km": list(record.position_km),
                    "velocity_kmps": list(record.velocity_kmps),
                }
                for record in extract.records
            ],
        },
    )
    analysis_asset = write_json_asset(
        output_root,
        Path("assets/analysis") / f"{episode_id}.json",
        {
            "elapsed_s": elapsed,
            "unix_time_s": timestamps_s.tolist(),
            "position_m": position_m.tolist(),
            "source_velocity_mps": source_velocity_mps.tolist(),
            "derived_velocity_mps": track.derived_velocity_mps.tolist(),
            "derived_acceleration_mps2": track.derived_acceleration_mps2.tolist(),
        },
    )
    mission_group = opaque_group_id(
        dataset_id=DATASET_ID,
        namespace=GroupingNamespace.MISSION_EVENT.value,
        raw_value=extract.header.creation_date_utc.isoformat(),
    )
    recording_group = opaque_group_id(
        dataset_id=DATASET_ID,
        namespace=GroupingNamespace.SOURCE_RECORDING.value,
        raw_value=track.provenance.recording_id,
    )
    dataset_group = opaque_group_id(
        dataset_id=DATASET_ID,
        namespace=GroupingNamespace.SOURCE_DATASET.value,
        raw_value=DATASET_ID,
    )
    return TrajectoryEpisodeManifest(
        corpus_snapshot_id=corpus_snapshot_id,
        episode_id=episode_id,
        primary_program_domain=ProgramDomain.SPACE,
        corpus_sublane="space_orbital",
        default_operating_environment="orbital_space",
        default_motion_regime="persistent_orbit",
        source_dataset_id=DATASET_ID,
        source_artifact_ids=(source_artifact_id,),
        observation_modality="operational_predicted_oem",
        platform_group_id=platform_group,
        mission_id="ISS.OEM_J2K_EPH",
        object_id=None,
        start_time=iso_utc_from_unix(float(timestamps_s[0])),
        end_time=iso_utc_from_unix(float(timestamps_s[-1])),
        state_views=(
            TrajectoryStateViewManifest(
                state_view_id=f"{episode_id}:source-native",
                view_kind=StateViewKind.SOURCE_NATIVE,
                state_role=StateRole.ESTIMATE,
                value_basis=ValueBasis.REPORTED,
                frame=source_frame,
                source_time_axis=_time_axis(first_epoch),
                sample_count=len(extract.records),
                sample_asset=source_asset,
                channel_descriptors=(
                    _channel(
                        channel_id="oem-position",
                        semantic_role="position",
                        components=("x", "y", "z"),
                        units=("km", "km", "km"),
                        frame_id=source_frame.frame_id,
                        state_role=StateRole.ESTIMATE,
                        value_basis=ValueBasis.REPORTED,
                        access_class=AccessClass.CONTEXT,
                        source_fields=("X", "Y", "Z"),
                        lineage=(PARSE_STEP_ID,),
                    ),
                    _channel(
                        channel_id="oem-source-velocity",
                        semantic_role="velocity",
                        components=("vx", "vy", "vz"),
                        units=("km/s", "km/s", "km/s"),
                        frame_id=source_frame.frame_id,
                        state_role=StateRole.ESTIMATE,
                        value_basis=ValueBasis.REPORTED,
                        access_class=AccessClass.CLASSIFIER_CANDIDATE,
                        source_fields=("VX", "VY", "VZ"),
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
                frame=analysis_frame,
                source_time_axis=_time_axis(first_epoch),
                sample_count=len(extract.records),
                sample_asset=analysis_asset,
                channel_descriptors=(
                    _channel(
                        channel_id="eci-analysis-kinematics",
                        semantic_role="orbital_state",
                        components=("position", "source_velocity", "derived_velocity", "derived_acceleration"),
                        units=("m", "m/s", "m/s", "m/s^2"),
                        frame_id=analysis_frame.frame_id,
                        state_role=StateRole.RECONSTRUCTION,
                        value_basis=ValueBasis.DERIVED,
                        access_class=AccessClass.CLASSIFIER_CANDIDATE,
                        source_fields=("position_m", "source_velocity_mps", "derived_velocity_mps", "derived_acceleration_mps2"),
                        lineage=(ANALYSIS_STEP_ID,),
                        notes="Validation-only analysis view; no classifier asset is emitted.",
                    ),
                ),
                normalization_assumptions=(
                    "OEM kilometer state values are converted to SI meters and meters per second.",
                    "EME2000 and UTC semantics are retained without an additional transform.",
                    "Source velocity remains distinct from finite-difference velocity.",
                ),
                processing_step_ids=(ANALYSIS_STEP_ID,),
            ),
        ),
        segments=(
            TrajectorySegment(
                segment_id=f"{episode_id}:bounded-arc",
                start_offset_s=elapsed[0],
                end_offset_s=elapsed[-1],
                operating_environment="orbital_space",
                motion_regime="persistent_orbit",
                evidence_kind="bounded_oem_arc",
            ),
        ),
        labels=(
            LabelAssertion(
                assertion_id=f"{episode_id}:orbital-regime",
                namespace="motion_regime",
                value="persistent_orbit",
                evidence_kind=LabelEvidenceKind.DERIVED,
                evidence_strength=EvidenceStrength.STRONG,
                source_reference=source_artifact_id,
                proxy=False,
                vocabulary_version="product4-motion-regime-v1",
                notes="Lane-normalized from the operational OEM context; not measurement truth.",
            ),
        ),
        grouping_keys=(
            GroupingKey(
                namespace=GroupingNamespace.PHYSICAL_PLATFORM,
                opaque_value=platform_group,
                scope="opaque persistent object grouping",
                evidence_strength=EvidenceStrength.STRONG,
            ),
            GroupingKey(
                namespace=GroupingNamespace.MISSION_EVENT,
                opaque_value=mission_group,
                scope="OEM creation event",
                evidence_strength=EvidenceStrength.STRONG,
            ),
            GroupingKey(
                namespace=GroupingNamespace.SOURCE_RECORDING,
                opaque_value=recording_group,
                scope="OEM recording segment",
                evidence_strength=EvidenceStrength.STRONG,
            ),
            GroupingKey(
                namespace=GroupingNamespace.SOURCE_DATASET,
                opaque_value=dataset_group,
                scope="source dataset",
                evidence_strength=EvidenceStrength.STRONG,
            ),
        ),
        quality_summary=_quality(track, source_artifact_id=source_artifact_id),
        processing_step_ids=(PARSE_STEP_ID, ANALYSIS_STEP_ID),
        domain_extension=DomainExtension(
            schema_id="space_orbital_nasa_iss_oem_common_front_v0.1",
            schema_version="0.1.0",
            payload={
                "fixture_status": "validation_only",
                "source_fixture_sha256": sha256_file(source),
                "source_object_id": extract.metadata.source_object_id,
                "classifier_eligible": False,
                "classifier_view_status": "intentionally_blocked",
                "source_claim_boundary": (
                    "Supports CCSDS OEM ingestion, EME2000/UTC semantics, and persistent-object "
                    "grouping; it does not support measurement truth or population-level orbital claims."
                ),
            },
        ),
        classifier_trajectory_view=None,
    )


def build_nasa_iss_oem_episodes(
    source_path: str | Path,
    *,
    output_root: str | Path,
    corpus_snapshot_id: str,
    source_artifact_id: str = SOURCE_ARTIFACT_ID,
) -> tuple[TrajectoryEpisodeManifest, ...]:
    episode = build_nasa_iss_oem_episode(
        source_path,
        output_root=output_root,
        corpus_snapshot_id=corpus_snapshot_id,
        source_artifact_id=source_artifact_id,
    )
    episodes_root = Path(output_root) / "episodes"
    episodes_root.mkdir(parents=True, exist_ok=True)
    (episodes_root / f"{episode.episode_id}.json").write_text(
        episode.model_dump_json(indent=2, exclude_none=True) + "\n",
        encoding="utf-8",
    )
    return (episode,)


__all__ = ["build_nasa_iss_oem_episode", "build_nasa_iss_oem_episodes"]
