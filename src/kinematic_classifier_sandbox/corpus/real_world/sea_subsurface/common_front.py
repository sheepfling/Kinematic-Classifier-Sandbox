"""Convert the acquired IOOS SEA-SUB anchor into a validation-only episode."""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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

DATASET_ID = "ioos-ngdac-uaf-unit_191-20240309T1200"
SOURCE_ARTIFACT_ID = "ioos_uaf_unit_191_profile_1709942882"
PARSE_STEP_ID = "ioos-sea-sub-parse-v1"
ANALYSIS_STEP_ID = "ioos-sea-sub-channel-aware-analysis-v1"
MISSING_VALUES = {"", "NaN", "nan", "null", "-999", "-9999.9"}


def _read_source(path: str | Path) -> tuple[dict[str, str], tuple[dict[str, str], ...]]:
    with Path(path).open(newline="", encoding="iso-8859-1") as stream:
        reader = csv.DictReader(stream)
        try:
            units_row = next(reader)
        except StopIteration as error:
            raise ValueError("SEA-SUB source has no units row") from error
        units = {
            str(key): "" if value is None else str(value)
            for key, value in units_row.items()
            if key is not None
        }
        rows = tuple(
            {
                str(key): "" if value is None else str(value)
                for key, value in row.items()
                if key is not None
            }
            for row in reader
        )
    if not rows:
        raise ValueError("SEA-SUB source has no data rows")
    return units, rows


def _optional_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "").strip()
    if value in MISSING_VALUES:
        return None
    return float(value)


def _epoch(row: dict[str, str]) -> float:
    value = row.get("precise_time", "").strip()
    if not value:
        raise ValueError("SEA-SUB row has no precise_time")
    normalized = value.removesuffix("Z") + "+00:00"
    parsed = datetime.fromisoformat(normalized).astimezone(UTC)
    return parsed.timestamp()


def _frame(episode_id: str) -> FrameDescriptor:
    return FrameDescriptor(
        frame_id=f"{episode_id}:glider-depth-frame",
        frame_kind="geodetic_with_vertical",
        axes=("latitude", "longitude", "depth"),
        axis_units=("degree", "degree", "m"),
        center_or_origin="Earth surface geodetic coordinates with water-column depth",
        vertical_reference="sea_surface",
        vertical_positive_direction="downward",
        crs_or_datum="WGS 84 geodetic; source-declared sea-surface depth",
    )


def _time_axis(first_epoch: float) -> TimeAxisDescriptor:
    return TimeAxisDescriptor(
        source_time_system="UTC",
        normalized_time_system="elapsed SI seconds",
        absolute_time_available=True,
        source_epoch_or_reference=iso_utc_from_unix(first_epoch),
        elapsed_origin=iso_utc_from_unix(first_epoch),
        precision_or_resolution="source precise_time resolution",
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


def _analysis_samples(rows: tuple[dict[str, str], ...], epochs: tuple[float, ...]) -> list[dict[str, Any]]:
    origin = epochs[0]
    samples: list[dict[str, Any]] = []
    for row, epoch in zip(rows, epochs, strict=True):
        latitude = _optional_float(row, "latitude")
        longitude = _optional_float(row, "longitude")
        source_depth = _optional_float(row, "m_depth")
        standardized_depth = _optional_float(row, "depth")
        pressure = _optional_float(row, "pressure")
        gps_latitude = _optional_float(row, "m_gps_lat")
        gps_longitude = _optional_float(row, "m_gps_lon")
        dead_reckoned_latitude = _optional_float(row, "m_lat")
        dead_reckoned_longitude = _optional_float(row, "m_lon")
        samples.append(
            {
                "elapsed_s": epoch - origin,
                "unix_time_s": epoch,
                "position": [latitude, longitude, standardized_depth],
                "position_valid": [
                    latitude is not None,
                    longitude is not None,
                    standardized_depth is not None,
                ],
                "provider_horizontal": [latitude, longitude],
                "provider_horizontal_valid": [latitude is not None, longitude is not None],
                "onboard_gps_horizontal": [gps_latitude, gps_longitude],
                "onboard_gps_valid": [gps_latitude is not None, gps_longitude is not None],
                "dead_reckoned_horizontal": [dead_reckoned_latitude, dead_reckoned_longitude],
                "dead_reckoned_valid": [
                    dead_reckoned_latitude is not None,
                    dead_reckoned_longitude is not None,
                ],
                "pressure_dbar": pressure,
                "source_depth_m": source_depth,
                "standardized_depth_m": standardized_depth,
            }
        )
    return samples


def _quality(epochs: tuple[float, ...], *, source_artifact_id: str):
    deltas = tuple(right - left for left, right in zip(epochs, epochs[1:]))
    duplicate_count = len(epochs) - len(set(epochs))
    findings = [
        QualityFinding(
            code="SEA_SUB_GPS_PHASE_NOT_PROVEN",
            severity=QualitySeverity.WARNING,
            message="Sparse GPS fixes are preserved without inferring surface phase.",
            source_reference=source_artifact_id,
        ),
        QualityFinding(
            code="SEA_SUB_QC_ZERO_MEANS_NO_QC_PERFORMED",
            severity=QualitySeverity.INFO,
            message="The retained mapping treats source QC zero as no QC performed, not good data.",
            source_reference=source_artifact_id,
        ),
    ]
    if duplicate_count:
        findings.append(
            QualityFinding(
                code="SEA_SUB_ASYNCHRONOUS_CHANNEL_EVENTS",
                severity=QualitySeverity.WARNING,
                message=(
                    "Duplicate timestamps are retained as channel events rather than "
                    "collapsed into a single state."
                ),
                value=duplicate_count,
                source_reference=source_artifact_id,
            )
        )
    if any(row_delta < 0.0 for row_delta in deltas):
        findings.append(
            QualityFinding(
                code="SEA_SUB_OUT_OF_ORDER_TIMESTAMP",
                severity=QualitySeverity.WARNING,
                message="Source timestamps are not strictly chronological.",
                source_reference=source_artifact_id,
            )
        )
    return quality_summary_from_elapsed(
        tuple(epoch - epochs[0] for epoch in epochs),
        findings=findings,
        disposition="degraded_but_usable",
    )


def build_ioos_anchor_episode(
    source_path: str | Path,
    *,
    output_root: str | Path,
    corpus_snapshot_id: str,
    source_artifact_id: str = SOURCE_ARTIFACT_ID,
) -> TrajectoryEpisodeManifest:
    """Write a channel-aware SEA-SUB episode without creating a classifier view."""

    source = Path(source_path)
    units, rows = _read_source(source)
    epochs = tuple(_epoch(row) for row in rows)
    if any(current < previous for previous, current in zip(epochs, epochs[1:])):
        raise ValueError("SEA-SUB source rows must be nondecreasing in time")
    episode_id = "sea_sub:ioos:unit_191:profile_1709942882"
    frame = _frame(episode_id)
    source_payload = {
        "units": units,
        "rows": rows,
        "source_artifact_sha256": sha256_file(source),
        "missing_values": sorted(MISSING_VALUES),
    }
    analysis_payload = {
        "samples": _analysis_samples(rows, epochs),
        "channel_semantics": {
            "provider_horizontal": "provider standardized latitude/longitude; mixed measured/interpolated",
            "onboard_gps_horizontal": "sparse measured GPS; surface phase not proven",
            "dead_reckoned_horizontal": "source dead-reckoned latitude/longitude",
            "pressure_dbar": "measured sea-water pressure; QC zero is not good-data evidence",
            "standardized_depth_m": "provider depth derived from pressure",
        },
    }
    source_asset = write_json_asset(
        output_root,
        Path("assets/source_native") / f"{episode_id}.json",
        source_payload,
    )
    analysis_asset = write_json_asset(
        output_root,
        Path("assets/analysis") / f"{episode_id}.json",
        analysis_payload,
    )
    platform_group = opaque_group_id(
        dataset_id=DATASET_ID,
        namespace=GroupingNamespace.PHYSICAL_PLATFORM.value,
        raw_value="wmo:4902987",
    )
    mission_group = opaque_group_id(
        dataset_id=DATASET_ID,
        namespace=GroupingNamespace.MISSION_EVENT.value,
        raw_value="unit_191-20240309T1200",
    )
    recording_group = opaque_group_id(
        dataset_id=DATASET_ID,
        namespace=GroupingNamespace.SOURCE_RECORDING.value,
        raw_value="unit_191-20240309T1200:1709942882",
    )
    source_dataset_group = opaque_group_id(
        dataset_id=DATASET_ID,
        namespace=GroupingNamespace.SOURCE_DATASET.value,
        raw_value=DATASET_ID,
    )
    timestamps = tuple(epoch - epochs[0] for epoch in epochs)
    start_time = iso_utc_from_unix(epochs[0])
    end_time = iso_utc_from_unix(epochs[-1])
    return TrajectoryEpisodeManifest(
        corpus_snapshot_id=corpus_snapshot_id,
        episode_id=episode_id,
        primary_program_domain=ProgramDomain.SEA,
        corpus_sublane="sea_subsurface",
        default_operating_environment="underwater",
        default_motion_regime="glider_profile_descent",
        source_dataset_id=DATASET_ID,
        source_artifact_ids=(source_artifact_id,),
        observation_modality="glider_mixed_navigation_and_water_column",
        platform_group_id=platform_group,
        mission_id="unit_191-20240309T1200",
        object_id="profile_1709942882",
        start_time=start_time,
        end_time=end_time,
        state_views=(
            TrajectoryStateViewManifest(
                state_view_id=f"{episode_id}:source-native",
                view_kind=StateViewKind.SOURCE_NATIVE,
                state_role=StateRole.OBSERVATION,
                value_basis=ValueBasis.MEASURED,
                frame=frame,
                source_time_axis=_time_axis(epochs[0]),
                sample_count=len(rows),
                sample_asset=source_asset,
                channel_descriptors=(
                    _channel(
                        channel_id="provider-horizontal",
                        semantic_role="position",
                        components=("latitude", "longitude"),
                        units=("degree", "degree"),
                        frame_id=frame.frame_id,
                        state_role=StateRole.ESTIMATE,
                        value_basis=ValueBasis.POSTPROCESSED,
                        access_class=AccessClass.CONTEXT,
                        source_fields=("latitude", "longitude"),
                        lineage=(PARSE_STEP_ID,),
                        notes="Provider-standardized horizontal position may be interpolated.",
                    ),
                    _channel(
                        channel_id="onboard-gps-horizontal",
                        semantic_role="position",
                        components=("latitude", "longitude"),
                        units=("degree", "degree"),
                        frame_id=frame.frame_id,
                        state_role=StateRole.OBSERVATION,
                        value_basis=ValueBasis.MEASURED,
                        access_class=AccessClass.AUDIT_ONLY,
                        source_fields=("m_gps_lat", "m_gps_lon"),
                        lineage=(PARSE_STEP_ID,),
                        notes="Sparse GPS channel; no surface-phase inference.",
                    ),
                    _channel(
                        channel_id="dead-reckoned-horizontal",
                        semantic_role="position",
                        components=("latitude", "longitude"),
                        units=("degree", "degree"),
                        frame_id=frame.frame_id,
                        state_role=StateRole.ESTIMATE,
                        value_basis=ValueBasis.DEAD_RECKONED,
                        access_class=AccessClass.AUDIT_ONLY,
                        source_fields=("m_lat", "m_lon"),
                        lineage=(PARSE_STEP_ID,),
                    ),
                    _channel(
                        channel_id="sea-water-pressure",
                        semantic_role="pressure",
                        components=("pressure",),
                        units=("dbar",),
                        frame_id=frame.frame_id,
                        state_role=StateRole.OBSERVATION,
                        value_basis=ValueBasis.MEASURED,
                        access_class=AccessClass.CLASSIFIER_CANDIDATE,
                        source_fields=("pressure",),
                        lineage=(PARSE_STEP_ID,),
                        validity_reference="pressure_valid",
                    ),
                    _channel(
                        channel_id="source-depth",
                        semantic_role="depth",
                        components=("depth",),
                        units=("m",),
                        frame_id=frame.frame_id,
                        state_role=StateRole.OBSERVATION,
                        value_basis=ValueBasis.REPORTED,
                        access_class=AccessClass.AUDIT_ONLY,
                        source_fields=("m_depth",),
                        lineage=(PARSE_STEP_ID,),
                        validity_reference="source_depth_valid",
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
                source_time_axis=_time_axis(epochs[0]),
                sample_count=len(rows),
                sample_asset=analysis_asset,
                channel_descriptors=(
                    _channel(
                        channel_id="channel-aware-depth-state",
                        semantic_role="position_and_depth",
                        components=("latitude", "longitude", "depth"),
                        units=("degree", "degree", "m"),
                        frame_id=frame.frame_id,
                        state_role=StateRole.RECONSTRUCTION,
                        value_basis=ValueBasis.DERIVED,
                        access_class=AccessClass.CLASSIFIER_CANDIDATE,
                        source_fields=("latitude", "longitude", "depth", "pressure"),
                        lineage=(ANALYSIS_STEP_ID,),
                        notes="Asynchronous measured, estimated, and derived channels remain explicit.",
                    ),
                ),
                normalization_assumptions=(
                    "Provider latitude/longitude are retained in EPSG:4326 semantics.",
                    "Depth is positive downward from sea surface.",
                    "Missing channel values remain null with validity masks; no zero fill is applied.",
                ),
                processing_step_ids=(ANALYSIS_STEP_ID,),
            ),
        ),
        segments=(
            TrajectorySegment(
                segment_id=f"{episode_id}:profile",
                start_offset_s=timestamps[0],
                end_offset_s=timestamps[-1],
                operating_environment="underwater",
                motion_regime="glider_profile_descent",
                evidence_kind="selected_profile_extent",
            ),
        ),
        labels=(
            LabelAssertion(
                assertion_id=f"{episode_id}:platform",
                namespace="platform_type",
                value="underwater_glider",
                evidence_kind=LabelEvidenceKind.RECONCILED,
                evidence_strength=EvidenceStrength.STRONG,
                source_reference=source_artifact_id,
                proxy=False,
                vocabulary_version="product4-platform-domain-v1",
                notes="Platform type is source metadata, not a classifier projection.",
            ),
        ),
        grouping_keys=(
            GroupingKey(
                namespace=GroupingNamespace.PHYSICAL_PLATFORM,
                opaque_value=platform_group,
                scope="opaque WMO platform grouping",
                evidence_strength=EvidenceStrength.STRONG,
            ),
            GroupingKey(
                namespace=GroupingNamespace.MISSION_EVENT,
                opaque_value=mission_group,
                scope="deployment event",
                evidence_strength=EvidenceStrength.STRONG,
            ),
            GroupingKey(
                namespace=GroupingNamespace.SOURCE_RECORDING,
                opaque_value=recording_group,
                scope="profile recording",
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
                    dataset_id=DATASET_ID,
                    namespace=GroupingNamespace.GEOGRAPHY.value,
                    raw_value="alaska_kenai_fjord",
                ),
                scope="source geography",
                evidence_strength=EvidenceStrength.MEDIUM,
            ),
        ),
        quality_summary=_quality(epochs, source_artifact_id=source_artifact_id),
        processing_step_ids=(PARSE_STEP_ID, ANALYSIS_STEP_ID),
        domain_extension=DomainExtension(
            schema_id="sea_subsurface_ioos_channel_aware_common_front_v0.1",
            schema_version="0.1.0",
            payload={
                "fixture_status": "validation_only",
                "source_artifact_sha256": sha256_file(source),
                "canonical_common_front_validation": "pending_g2_review",
                "classifier_view_status": "intentionally_blocked",
                "source_claim_boundary": (
                    "Supports measured GPS, dead-reckoned, pressure, and derived-depth "
                    "semantics; it does not support independent navigation truth."
                ),
            },
        ),
        classifier_trajectory_view=None,
    )


__all__ = ["build_ioos_anchor_episode"]
