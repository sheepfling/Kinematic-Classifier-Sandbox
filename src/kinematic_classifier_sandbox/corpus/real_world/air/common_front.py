"""Convert a documented readsb trace into validation-only AIR episodes."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..adapters.adsblol.readsb_trace import (
    ReadsbTrace,
    ReadsbTraceLeg,
    load_readsb_trace,
    split_readsb_legs,
    trace_time_findings,
)
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

DATASET_ID = "adsblol_globe_history"
SOURCE_ARTIFACT_ID = "adsblol_globe_history_release"
PARSE_STEP_ID = "adsblol-readsb-parse-v1"
ANALYSIS_STEP_ID = "adsblol-readsb-common-front-analysis-v1"
FT_TO_M = 0.3048
FPM_TO_MPS = 0.00508
KNOT_TO_MPS = 0.514444


def _source_frame(episode_id: str) -> FrameDescriptor:
    return FrameDescriptor(
        frame_id=f"{episode_id}:readsb-geodetic",
        frame_kind="geodetic_with_vertical",
        axes=("latitude", "longitude", "altitude"),
        axis_units=("degree", "degree", "ft"),
        center_or_origin="WGS 84 Earth",
        vertical_reference="readsb primary altitude basis varies by source flag",
        vertical_positive_direction="upward",
        crs_or_datum="WGS 84 geodetic; altitude source-declared",
    )


def _analysis_frame(episode_id: str) -> FrameDescriptor:
    return FrameDescriptor(
        frame_id=f"{episode_id}:readsb-normalized",
        frame_kind="geodetic_with_vertical",
        axes=("latitude", "longitude", "altitude"),
        axis_units=("degree", "degree", "m"),
        center_or_origin="WGS 84 Earth",
        vertical_reference="source primary altitude converted to meters; basis retained per sample",
        vertical_positive_direction="upward",
        crs_or_datum="WGS 84 geodetic; no ECEF conversion in this tranche",
    )


def _time_axis(first_epoch: float) -> TimeAxisDescriptor:
    return TimeAxisDescriptor(
        source_time_system="Unix UTC seconds",
        normalized_time_system="elapsed SI seconds",
        absolute_time_available=True,
        source_epoch_or_reference=iso_utc_from_unix(first_epoch),
        elapsed_origin=iso_utc_from_unix(first_epoch),
        precision_or_resolution="source trace offset precision",
        rollover_policy="none",
        leap_second_policy="Unix/POSIX convention",
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
    validity_reference: str | None = None,
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
        validity_reference=validity_reference,
        notes=notes,
    )


def _point_payload(point: Any) -> dict[str, Any]:
    payload = asdict(point)
    for key in ("altitude_basis", "vertical_rate_basis"):
        value = payload.get(key)
        if value is not None:
            payload[key] = value.value
    return payload


def _analysis_samples(leg: ReadsbTraceLeg, first_epoch: float) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for point in leg.points:
        samples.append(
            {
                "elapsed_s": point.event_time_unix_s - first_epoch,
                "unix_time_s": point.event_time_unix_s,
                "position": [point.latitude_deg, point.longitude_deg],
                "position_valid": [
                    point.latitude_deg is not None,
                    point.longitude_deg is not None,
                ],
                "altitude_m": (
                    point.altitude_ft * FT_TO_M if point.altitude_ft is not None else None
                ),
                "altitude_valid": point.altitude_ft is not None,
                "altitude_basis": (
                    point.altitude_basis.value if point.altitude_basis is not None else None
                ),
                "ground_speed_mps": (
                    point.ground_speed_kt * KNOT_TO_MPS
                    if point.ground_speed_kt is not None
                    else None
                ),
                "track_deg": point.track_or_ground_heading_deg,
                "vertical_rate_mps": (
                    point.vertical_rate_fpm * FPM_TO_MPS
                    if point.vertical_rate_fpm is not None
                    else None
                ),
                "vertical_rate_basis": (
                    point.vertical_rate_basis.value
                    if point.vertical_rate_basis is not None
                    else None
                ),
                "flags": point.flags,
                "stale_position": point.stale_position,
                "on_ground": point.on_ground,
            }
        )
    return samples


def _quality(
    trace: ReadsbTrace,
    leg: ReadsbTraceLeg,
    *,
    source_artifact_id: str,
):
    first_epoch = leg.points[0].event_time_unix_s
    elapsed = tuple(point.event_time_unix_s - first_epoch for point in leg.points)
    findings: list[QualityFinding] = [
        QualityFinding(
            code="AIR_VALIDATION_FIXTURE_ONLY",
            severity=QualitySeverity.INFO,
            message="The committed readsb trace documents parser semantics but is not a historical release trace.",
            source_reference=source_artifact_id,
        )
    ]
    for finding in trace_time_findings(trace):
        if any(point.source_index == finding.source_index for point in leg.points):
            severity = QualitySeverity.WARNING
            findings.append(
                QualityFinding(
                    code=finding.code,
                    severity=severity,
                    message="Readsb source-time finding is preserved in the common quality summary.",
                    value=finding.source_index,
                    source_reference=source_artifact_id,
                )
            )
    stale_count = sum(point.stale_position for point in leg.points)
    if stale_count:
        findings.append(
            QualityFinding(
                code="AIR_STALE_POSITION",
                severity=QualitySeverity.WARNING,
                message="One or more points carry the readsb stale-position flag.",
                value=stale_count,
                source_reference=source_artifact_id,
            )
        )
    return quality_summary_from_elapsed(
        elapsed,
        findings=findings,
        disposition="usable_with_restrictions",
    )


def _build_episode(
    trace: ReadsbTrace,
    leg: ReadsbTraceLeg,
    *,
    output_root: str | Path,
    corpus_snapshot_id: str,
    source_artifact_id: str,
    source_sha256: str,
) -> TrajectoryEpisodeManifest:
    platform_group = opaque_group_id(
        dataset_id=DATASET_ID,
        namespace=GroupingNamespace.PHYSICAL_PLATFORM.value,
        raw_value=trace.icao,
    )
    episode_id = f"air:adsblol:{platform_group}:leg:{leg.leg_ordinal}"
    source_frame = _source_frame(episode_id)
    analysis_frame = _analysis_frame(episode_id)
    first_epoch = leg.points[0].event_time_unix_s
    source_asset = write_json_asset(
        output_root,
        Path("assets/source_native") / f"{episode_id}.json",
        {
            "icao": trace.icao,
            "registration": trace.registration,
            "type_code": trace.type_code,
            "database_flags": trace.database_flags,
            "description": trace.description,
            "points": [_point_payload(point) for point in leg.points],
        },
    )
    analysis_asset = write_json_asset(
        output_root,
        Path("assets/analysis") / f"{episode_id}.json",
        {
            "samples": _analysis_samples(leg, first_epoch),
            "normalization": {
                "feet_to_meters": FT_TO_M,
                "feet_per_minute_to_meters_per_second": FPM_TO_MPS,
                "knots_to_meters_per_second": KNOT_TO_MPS,
                "vertical_missing_policy": "null_with_validity_false",
            },
        },
    )
    recording_group = opaque_group_id(
        dataset_id=DATASET_ID,
        namespace=GroupingNamespace.SOURCE_RECORDING.value,
        raw_value=f"{trace.timestamp_unix_s}:{trace.icao}",
    )
    mission_group = opaque_group_id(
        dataset_id=DATASET_ID,
        namespace=GroupingNamespace.MISSION_EVENT.value,
        raw_value=f"{trace.timestamp_unix_s}:leg:{leg.leg_ordinal}",
    )
    dataset_group = opaque_group_id(
        dataset_id=DATASET_ID,
        namespace=GroupingNamespace.SOURCE_DATASET.value,
        raw_value=DATASET_ID,
    )
    temporal_group = opaque_group_id(
        dataset_id=DATASET_ID,
        namespace=GroupingNamespace.TEMPORAL_COLLECTION.value,
        raw_value=iso_utc_from_unix(first_epoch)[:10],
    )
    elapsed_end = leg.points[-1].event_time_unix_s - first_epoch
    return TrajectoryEpisodeManifest(
        corpus_snapshot_id=corpus_snapshot_id,
        episode_id=episode_id,
        primary_program_domain=ProgramDomain.AIR,
        corpus_sublane="air_atmospheric",
        default_operating_environment="atmosphere",
        default_motion_regime="aircraft_flight",
        source_dataset_id=DATASET_ID,
        source_artifact_ids=(source_artifact_id,),
        observation_modality="readsb_adsb_trace",
        platform_group_id=platform_group,
        mission_id=f"readsb-leg-{leg.leg_ordinal}",
        object_id=None,
        start_time=iso_utc_from_unix(first_epoch),
        end_time=iso_utc_from_unix(leg.points[-1].event_time_unix_s),
        state_views=(
            TrajectoryStateViewManifest(
                state_view_id=f"{episode_id}:source-native",
                view_kind=StateViewKind.SOURCE_NATIVE,
                state_role=StateRole.OBSERVATION,
                value_basis=ValueBasis.REPORTED,
                frame=source_frame,
                source_time_axis=_time_axis(first_epoch),
                sample_count=len(leg.points),
                sample_asset=source_asset,
                channel_descriptors=(
                    _channel(
                        channel_id="geodetic-position",
                        semantic_role="position",
                        components=("latitude", "longitude"),
                        units=("degree", "degree"),
                        frame_id=source_frame.frame_id,
                        state_role=StateRole.OBSERVATION,
                        value_basis=ValueBasis.REPORTED,
                        access_class=AccessClass.CONTEXT,
                        source_fields=("trace[*][1]", "trace[*][2]"),
                        lineage=(PARSE_STEP_ID,),
                    ),
                    _channel(
                        channel_id="primary-altitude",
                        semantic_role="altitude",
                        components=("altitude",),
                        units=("ft",),
                        frame_id=source_frame.frame_id,
                        state_role=StateRole.OBSERVATION,
                        value_basis=ValueBasis.REPORTED,
                        access_class=AccessClass.CLASSIFIER_CANDIDATE,
                        source_fields=("trace[*][3]", "trace[*][6]"),
                        lineage=(PARSE_STEP_ID,),
                        validity_reference="altitude_valid",
                        notes="Barometric/geometric basis remains per-sample metadata.",
                    ),
                    _channel(
                        channel_id="ground-motion",
                        semantic_role="horizontal_motion",
                        components=("speed", "track"),
                        units=("kt", "degree"),
                        frame_id=source_frame.frame_id,
                        state_role=StateRole.OBSERVATION,
                        value_basis=ValueBasis.REPORTED,
                        access_class=AccessClass.CLASSIFIER_CANDIDATE,
                        source_fields=("trace[*][4]", "trace[*][5]"),
                        lineage=(PARSE_STEP_ID,),
                    ),
                    _channel(
                        channel_id="vertical-rate",
                        semantic_role="vertical_motion",
                        components=("rate",),
                        units=("ft/min",),
                        frame_id=source_frame.frame_id,
                        state_role=StateRole.OBSERVATION,
                        value_basis=ValueBasis.REPORTED,
                        access_class=AccessClass.CLASSIFIER_CANDIDATE,
                        source_fields=("trace[*][7]", "trace[*][6]"),
                        lineage=(PARSE_STEP_ID,),
                        validity_reference="vertical_rate_valid",
                    ),
                ),
                processing_step_ids=(PARSE_STEP_ID,),
            ),
            TrajectoryStateViewManifest(
                state_view_id=f"{episode_id}:analysis",
                view_kind=StateViewKind.ANALYSIS,
                state_role=StateRole.RECONSTRUCTION,
                value_basis=ValueBasis.POSTPROCESSED,
                frame=analysis_frame,
                source_time_axis=_time_axis(first_epoch),
                sample_count=len(leg.points),
                sample_asset=analysis_asset,
                channel_descriptors=(
                    _channel(
                        channel_id="normalized-flight-state",
                        semantic_role="normalized_motion_state",
                        components=("latitude", "longitude", "altitude_m", "speed_mps", "vertical_rate_mps"),
                        units=("degree", "degree", "m", "m/s", "m/s"),
                        frame_id=analysis_frame.frame_id,
                        state_role=StateRole.RECONSTRUCTION,
                        value_basis=ValueBasis.POSTPROCESSED,
                        access_class=AccessClass.CLASSIFIER_CANDIDATE,
                        source_fields=("readsb source trace",),
                        lineage=(ANALYSIS_STEP_ID,),
                        notes="Validation-only normalized view; no classifier asset is emitted.",
                    ),
                ),
                normalization_assumptions=(
                    "Altitude basis is retained per sample; barometric and geometric values are not coalesced.",
                    "Missing values remain null with validity flags.",
                    "No ECEF or local-ENU transform is claimed in this tranche.",
                ),
                processing_step_ids=(ANALYSIS_STEP_ID,),
            ),
        ),
        segments=(
            TrajectorySegment(
                segment_id=f"{episode_id}:leg",
                start_offset_s=0.0,
                end_offset_s=elapsed_end,
                operating_environment="atmosphere",
                motion_regime="aircraft_flight",
                evidence_kind="source_derived_leg_candidate",
            ),
        ),
        labels=(
            LabelAssertion(
                assertion_id=f"{episode_id}:platform-type",
                namespace="platform_type",
                value="aircraft",
                evidence_kind=LabelEvidenceKind.RECONCILED,
                evidence_strength=EvidenceStrength.MEDIUM,
                source_reference=source_artifact_id,
                proxy=False,
                vocabulary_version="product4-platform-domain-v1",
                notes="The fixture does not establish authoritative flight identity or aircraft-family labels.",
            ),
        ),
        grouping_keys=(
            GroupingKey(
                namespace=GroupingNamespace.PHYSICAL_PLATFORM,
                opaque_value=platform_group,
                scope="opaque readsb aircraft grouping",
                evidence_strength=EvidenceStrength.STRONG,
            ),
            GroupingKey(
                namespace=GroupingNamespace.MISSION_EVENT,
                opaque_value=mission_group,
                scope="source new-leg event",
                evidence_strength=EvidenceStrength.MEDIUM,
            ),
            GroupingKey(
                namespace=GroupingNamespace.SOURCE_RECORDING,
                opaque_value=recording_group,
                scope="readsb collection trace",
                evidence_strength=EvidenceStrength.STRONG,
            ),
            GroupingKey(
                namespace=GroupingNamespace.SOURCE_DATASET,
                opaque_value=dataset_group,
                scope="source dataset",
                evidence_strength=EvidenceStrength.STRONG,
            ),
            GroupingKey(
                namespace=GroupingNamespace.TEMPORAL_COLLECTION,
                opaque_value=temporal_group,
                scope="collection day",
                evidence_strength=EvidenceStrength.MEDIUM,
            ),
        ),
        quality_summary=_quality(trace, leg, source_artifact_id=source_artifact_id),
        processing_step_ids=(PARSE_STEP_ID, ANALYSIS_STEP_ID),
        domain_extension=DomainExtension(
            schema_id="air_adsblol_readsb_common_front_v0.1",
            schema_version="0.1.0",
            payload={
                "fixture_status": "documented_parser_fixture_only",
                "common_front_contract_validation": "passed",
                "source_fixture_sha256": source_sha256,
                "raw_identity_access": "adapter_only",
                "classifier_view_status": "intentionally_blocked",
                "source_claim_boundary": (
                    "Supports native readsb parsing and explicit altitude/vertical-rate "
                    "semantics; it does not support physical truth or historical-flight claims."
                ),
            },
        ),
        classifier_trajectory_view=None,
    )


def build_readsb_fixture_episodes(
    source_path: str | Path,
    *,
    output_root: str | Path,
    corpus_snapshot_id: str,
    source_artifact_id: str = SOURCE_ARTIFACT_ID,
) -> tuple[TrajectoryEpisodeManifest, ...]:
    source = Path(source_path)
    trace = load_readsb_trace(source)
    legs = split_readsb_legs(trace, minimum_samples=2)
    source_sha256 = sha256_file(source)
    episodes = tuple(
        _build_episode(
            trace,
            leg,
            output_root=output_root,
            corpus_snapshot_id=corpus_snapshot_id,
            source_artifact_id=source_artifact_id,
            source_sha256=source_sha256,
        )
        for leg in legs
    )
    episodes_root = Path(output_root) / "episodes"
    episodes_root.mkdir(parents=True, exist_ok=True)
    for episode in episodes:
        (episodes_root / f"{episode.episode_id}.json").write_text(
            episode.model_dump_json(indent=2, exclude_none=True) + "\n",
            encoding="utf-8",
        )
    return episodes


__all__ = ["build_readsb_fixture_episodes"]
