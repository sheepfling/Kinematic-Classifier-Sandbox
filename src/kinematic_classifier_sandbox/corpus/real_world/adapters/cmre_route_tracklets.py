from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from kinematic_classifier_sandbox.corpus.real_world.episode_contracts import (
    AccessClass,
    AssetReference,
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
    QualityFinding,
    QualitySeverity,
    QualitySummary,
    StateRole,
    StateViewKind,
    TimeAxisDescriptor,
    TrajectoryEpisodeManifest,
    TrajectorySegment,
    TrajectoryStateViewManifest,
    ValueBasis,
)


EARTH_RADIUS_M = 6_378_137.0
KNOT_TO_MPS = 0.514444
DEFAULT_DATASET_ID = "cmre_brest_maritime_routes_tracklets_v1_0"

PARSE_STEP_ID = "cmre-parse-tracklets-v1"
LOCAL_TANGENT_STEP_ID = "cmre-local-tangent-v1"
REPORTED_VELOCITY_STEP_ID = "cmre-reported-velocity-v1"
CLASSIFIER_DEDUPLICATE_STEP_ID = "cmre-classifier-deduplicate-v1"

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class AisContact:
    source_contact_id: int
    mmsi: int
    speed_knots: float
    course_deg: float
    heading_deg: float
    longitude_deg: float
    latitude_deg: float
    unix_time_s: int
####


@dataclass(frozen=True, slots=True)
class RouteTracklet:
    tracklet_id: int
    contacts: tuple[AisContact, ...]
    route_id: str
####


@dataclass(frozen=True, slots=True)
class RouteDefinition:
    route_id: str
    origin_port: str
    destination_port: str
    nominal_length_m: int
####


@dataclass(frozen=True, slots=True)
class FixtureBuildResult:
    manifests: tuple[TrajectoryEpisodeManifest, ...]
    physical_platform_group_count: int
    repeated_physical_platform_groups: tuple[str, ...]
####


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        ####
    return digest.hexdigest()
####


def stable_group_id(*, dataset_id: str, namespace: str, raw_value: str) -> str:
    material = f"{dataset_id}:{namespace}:{raw_value}".encode()
    return hashlib.sha256(material).hexdigest()[:24]
####


def _require_columns(fieldnames: Sequence[str] | None, expected: Iterable[str]) -> None:
    if fieldnames is None:
        raise ValueError("source has no header")
    missing = sorted(set(expected) - set(fieldnames))
    if missing:
        raise ValueError(f"source is missing required columns: {missing}")
####


def _contact_columns(index: int) -> tuple[str, ...]:
    return (
        f"id{index}",
        f"mmsi{index}",
        f"speed{index}",
        f"course{index}",
        f"heading{index}",
        f"lon{index}",
        f"lat{index}",
        f"ts{index}",
    )
####


def parse_tracklets(
    path: str | Path,
    *,
    selected_tracklet_ids: set[int] | None = None,
) -> tuple[RouteTracklet, ...]:
    expected = {"idtracklet", "route"}
    for index in range(1, 6):
        expected.update(_contact_columns(index))
    ####

    result: list[RouteTracklet] = []
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="|")
        _require_columns(reader.fieldnames, expected)
        for row in reader:
            tracklet_id = int(row["idtracklet"])
            if selected_tracklet_ids is not None and tracklet_id not in selected_tracklet_ids:
                continue
            ####
            contacts = tuple(
                AisContact(
                    source_contact_id=int(row[f"id{index}"]),
                    mmsi=int(row[f"mmsi{index}"]),
                    speed_knots=float(row[f"speed{index}"]),
                    course_deg=float(row[f"course{index}"]),
                    heading_deg=float(row[f"heading{index}"]),
                    longitude_deg=float(row[f"lon{index}"]),
                    latitude_deg=float(row[f"lat{index}"]),
                    unix_time_s=int(row[f"ts{index}"]),
                )
                for index in range(1, 6)
            )
            identities = {contact.mmsi for contact in contacts}
            if len(identities) != 1:
                raise ValueError(f"tracklet {tracklet_id} mixes MMSI values")
            result.append(RouteTracklet(tracklet_id, contacts, row["route"]))
        ####
    return tuple(result)
####


def parse_route_nomenclature(path: str | Path) -> dict[str, RouteDefinition]:
    routes: dict[str, RouteDefinition] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="|")
        _require_columns(reader.fieldnames, {"route", "originport", "destinationport", "length"})
        for row in reader:
            route = RouteDefinition(
                route_id=row["route"],
                origin_port=row["originport"],
                destination_port=row["destinationport"],
                nominal_length_m=int(row["length"]),
            )
            if route.route_id in routes:
                raise ValueError(f"duplicate route {route.route_id!r}")
            routes[route.route_id] = route
        ####
    return routes
####


def local_enu_position(latitude_deg: FloatArray, longitude_deg: FloatArray) -> FloatArray:
    if latitude_deg.ndim != 1 or longitude_deg.ndim != 1:
        raise ValueError("latitude and longitude must be one-dimensional")
    if latitude_deg.shape != longitude_deg.shape or latitude_deg.size == 0:
        raise ValueError("latitude and longitude must be nonempty and aligned")
    latitude_rad = np.deg2rad(latitude_deg)
    longitude_rad = np.deg2rad(longitude_deg)
    latitude_origin = float(latitude_rad[0])
    longitude_origin = float(longitude_rad[0])
    east = (longitude_rad - longitude_origin) * EARTH_RADIUS_M * math.cos(latitude_origin)
    north = (latitude_rad - latitude_origin) * EARTH_RADIUS_M
    up = np.full(latitude_deg.shape, np.nan, dtype=np.float64)
    return np.column_stack((east, north, up)).astype(np.float64)
####


def reported_velocity_enu(speed_knots: FloatArray, course_deg: FloatArray) -> FloatArray:
    if speed_knots.ndim != 1 or course_deg.ndim != 1:
        raise ValueError("speed and course must be one-dimensional")
    if speed_knots.shape != course_deg.shape:
        raise ValueError("speed and course must be aligned")
    speed_mps = speed_knots * KNOT_TO_MPS
    course_rad = np.deg2rad(course_deg)
    east = speed_mps * np.sin(course_rad)
    north = speed_mps * np.cos(course_rad)
    up = np.full(speed_knots.shape, np.nan, dtype=np.float64)
    return np.column_stack((east, north, up)).astype(np.float64)
####


def _npz(
    *,
    output_root: Path,
    relative_path: Path,
    arrays: Mapping[str, NDArray[Any]],
) -> AssetReference:
    path = output_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return AssetReference(
        path=relative_path.as_posix(),
        media_type="application/x-npz",
        sha256=sha256_file(path),
    )
####


def _source_arrays(tracklet: RouteTracklet) -> dict[str, NDArray[Any]]:
    timestamp = np.array([item.unix_time_s for item in tracklet.contacts], dtype=np.int64)
    latitude = np.array([item.latitude_deg for item in tracklet.contacts], dtype=np.float64)
    longitude = np.array([item.longitude_deg for item in tracklet.contacts], dtype=np.float64)
    speed = np.array([item.speed_knots for item in tracklet.contacts], dtype=np.float64)
    course = np.array([item.course_deg for item in tracklet.contacts], dtype=np.float64)
    heading = np.array([item.heading_deg for item in tracklet.contacts], dtype=np.float64)
    position = np.column_stack((latitude, longitude, np.full(latitude.shape, np.nan)))
    position_valid = np.column_stack(
        (np.ones(latitude.shape, dtype=np.bool_), np.ones(latitude.shape, dtype=np.bool_), np.zeros(latitude.shape, dtype=np.bool_))
    )
    return {
        "elapsed_s": (timestamp - timestamp[0]).astype(np.float64),
        "unix_time_s": timestamp,
        "position_geodetic": position.astype(np.float64),
        "position_valid": position_valid,
        "speed_over_ground_knots": speed,
        "speed_valid": (speed >= 0.0) & (speed < 102.3),
        "course_over_ground_deg": course,
        "course_valid": (course >= 0.0) & (course < 360.0),
        "heading_deg": heading,
        "heading_valid": (heading >= 0.0) & (heading < 360.0),
    }
####


def _analysis_arrays(source: Mapping[str, NDArray[Any]]) -> dict[str, NDArray[Any]]:
    position = np.asarray(source["position_geodetic"], dtype=np.float64)
    speed = np.asarray(source["speed_over_ground_knots"], dtype=np.float64)
    course = np.asarray(source["course_over_ground_deg"], dtype=np.float64)
    speed_valid = np.asarray(source["speed_valid"], dtype=np.bool_)
    course_valid = np.asarray(source["course_valid"], dtype=np.bool_)
    position_enu = local_enu_position(position[:, 0], position[:, 1])
    velocity = reported_velocity_enu(speed, course)
    velocity_valid = np.column_stack(
        (speed_valid & course_valid, speed_valid & course_valid, np.zeros(speed.shape, dtype=np.bool_))
    )
    return {
        "elapsed_s": np.asarray(source["elapsed_s"], dtype=np.float64),
        "position_enu_m": position_enu,
        "position_valid": np.column_stack(
            (np.ones(speed.shape, dtype=np.bool_), np.ones(speed.shape, dtype=np.bool_), np.zeros(speed.shape, dtype=np.bool_))
        ),
        "reported_velocity_enu_mps": velocity,
        "velocity_valid": velocity_valid,
    }
####


def _classifier_arrays(analysis: Mapping[str, NDArray[Any]]) -> dict[str, NDArray[Any]]:
    elapsed = np.asarray(analysis["elapsed_s"], dtype=np.float64)
    keep = np.ones(elapsed.shape, dtype=np.bool_)
    if elapsed.size > 1:
        keep[1:] = np.diff(elapsed) > 0.0
    return {
        "elapsed_s": elapsed[keep],
        "position_xy_m": np.asarray(analysis["position_enu_m"], dtype=np.float64)[keep, :2],
        "reported_velocity_xy_mps": np.asarray(
            analysis["reported_velocity_enu_mps"], dtype=np.float64
        )[keep, :2],
    }
####


def _time_axis() -> TimeAxisDescriptor:
    return TimeAxisDescriptor(
        source_time_system="Unix UTC seconds",
        normalized_time_system="elapsed SI seconds",
        absolute_time_available=True,
        source_epoch_or_reference="1970-01-01T00:00:00Z",
        elapsed_origin="first source contact",
        precision_or_resolution="whole seconds",
        rollover_policy="none",
        leap_second_policy="Unix/POSIX convention",
    )
####


def _quality(source: Mapping[str, NDArray[Any]]) -> QualitySummary:
    elapsed = np.asarray(source["elapsed_s"], dtype=np.float64)
    delta = np.diff(elapsed)
    duplicate_count = int(np.count_nonzero(delta == 0.0))
    out_of_order_count = int(np.count_nonzero(delta < 0.0))
    positive = delta[delta > 0.0]
    findings = [
        QualityFinding(
            code="vertical_position_unavailable",
            severity=QualitySeverity.INFO,
            message="AIS tracklet has no measured vertical position; vertical validity is false.",
        )
    ]
    if duplicate_count:
        findings.append(
            QualityFinding(
                code="duplicate_timestamp",
                severity=QualitySeverity.WARNING,
                message="Source view preserves exact duplicate timestamps; classifier view keeps the first.",
                value=duplicate_count,
            )
        )
    invalid_heading = int(np.count_nonzero(~np.asarray(source["heading_valid"], dtype=np.bool_)))
    if invalid_heading:
        findings.append(
            QualityFinding(
                code="invalid_heading",
                severity=QualitySeverity.WARNING,
                message="Heading contains AIS sentinel or out-of-range values.",
                value=invalid_heading,
            )
        )
    return QualitySummary(
        disposition="accept_with_findings",
        sample_count=int(elapsed.size),
        duration_s=float(elapsed[-1] if elapsed.size else 0.0),
        median_sample_interval_s=float(np.median(positive) if positive.size else 0.0),
        maximum_gap_s=float(np.max(positive) if positive.size else 0.0),
        duplicate_timestamp_count=duplicate_count,
        out_of_order_timestamp_count=out_of_order_count,
        findings=tuple(findings),
    )
####


def build_episode(
    *,
    output_root: str | Path,
    tracklet: RouteTracklet,
    route: RouteDefinition,
    source_artifact_id: str,
    source_artifact_sha256: str,
    corpus_snapshot_id: str,
    dataset_id: str = DEFAULT_DATASET_ID,
) -> TrajectoryEpisodeManifest:
    root = Path(output_root)
    episode_id = f"cmre-route-tracklet-{tracklet.tracklet_id:06d}"
    source = _source_arrays(tracklet)
    analysis = _analysis_arrays(source)
    classifier = _classifier_arrays(analysis)

    source_rel = Path("assets/source") / f"{episode_id}.npz"
    analysis_rel = Path("assets/analysis") / f"{episode_id}.npz"
    classifier_rel = Path("assets/classifier") / f"{episode_id}.npz"
    source_asset = _npz(output_root=root, relative_path=source_rel, arrays=source)
    analysis_asset = _npz(output_root=root, relative_path=analysis_rel, arrays=analysis)
    classifier_asset = _npz(
        output_root=root, relative_path=classifier_rel, arrays=classifier
    )

    raw_mmsi = str(tracklet.contacts[0].mmsi)
    platform_group = stable_group_id(
        dataset_id=dataset_id,
        namespace=GroupingNamespace.PHYSICAL_PLATFORM.value,
        raw_value=raw_mmsi,
    )
    start_unix = tracklet.contacts[0].unix_time_s
    end_unix = tracklet.contacts[-1].unix_time_s
    duration_s = float(end_unix - start_unix)
    start_time = datetime.fromtimestamp(start_unix, tz=UTC).isoformat()
    end_time = datetime.fromtimestamp(end_unix, tz=UTC).isoformat()

    source_frame = FrameDescriptor(
        frame_id=f"{episode_id}:source-geodetic",
        frame_kind="geodetic",
        axes=("latitude", "longitude", "vertical_placeholder"),
        axis_units=("degree", "degree", "m"),
        center_or_origin="WGS 84 Earth",
        vertical_reference="unavailable",
        vertical_positive_direction="unavailable",
        crs_or_datum="EPSG:4326",
    )
    analysis_frame = FrameDescriptor(
        frame_id=f"{episode_id}:local-enu",
        frame_kind="local_tangent",
        axes=("east", "north", "up"),
        axis_units=("m", "m", "m"),
        center_or_origin="first valid AIS position",
        vertical_reference="unavailable",
        vertical_positive_direction="unavailable",
        crs_or_datum="WGS 84 local tangent approximation",
        local_origin={
            "latitude_deg": tracklet.contacts[0].latitude_deg,
            "longitude_deg": tracklet.contacts[0].longitude_deg,
        },
    )
    source_view = TrajectoryStateViewManifest(
        state_view_id=f"{episode_id}:source-native",
        view_kind=StateViewKind.SOURCE_NATIVE,
        state_role=StateRole.OBSERVATION,
        value_basis=ValueBasis.REPORTED,
        frame=source_frame,
        source_time_axis=_time_axis(),
        sample_count=len(tracklet.contacts),
        sample_asset=source_asset,
        channel_descriptors=(
            ChannelDescriptor(
                channel_id="position_geodetic",
                semantic_role="position",
                component_names=("latitude", "longitude", "vertical_placeholder"),
                units=("degree", "degree", "m"),
                frame_id=source_frame.frame_id,
                state_role=StateRole.OBSERVATION,
                value_basis=ValueBasis.REPORTED,
                access_class=AccessClass.CONTEXT,
                source_fields=("lat1..lat5", "lon1..lon5"),
                lineage_step_ids=(PARSE_STEP_ID,),
                validity_reference="position_valid",
                notes="Vertical is NaN with validity false.",
            ),
            ChannelDescriptor(
                channel_id="speed_over_ground",
                semantic_role="speed",
                component_names=("speed_over_ground",),
                units=("kn",),
                state_role=StateRole.OBSERVATION,
                value_basis=ValueBasis.REPORTED,
                access_class=AccessClass.CLASSIFIER_CANDIDATE,
                source_fields=("speed1..speed5",),
                lineage_step_ids=(PARSE_STEP_ID,),
                validity_reference="speed_valid",
            ),
            ChannelDescriptor(
                channel_id="course_over_ground",
                semantic_role="course",
                component_names=("course_over_ground",),
                units=("degree",),
                state_role=StateRole.OBSERVATION,
                value_basis=ValueBasis.REPORTED,
                access_class=AccessClass.CLASSIFIER_CANDIDATE,
                source_fields=("course1..course5",),
                lineage_step_ids=(PARSE_STEP_ID,),
                validity_reference="course_valid",
            ),
            ChannelDescriptor(
                channel_id="heading",
                semantic_role="heading",
                component_names=("heading",),
                units=("degree",),
                state_role=StateRole.OBSERVATION,
                value_basis=ValueBasis.REPORTED,
                access_class=AccessClass.CLASSIFIER_CANDIDATE,
                source_fields=("heading1..heading5",),
                lineage_step_ids=(PARSE_STEP_ID,),
                validity_reference="heading_valid",
            ),
        ),
        processing_step_ids=(PARSE_STEP_ID,),
    )
    analysis_view = TrajectoryStateViewManifest(
        state_view_id=f"{episode_id}:analysis-enu",
        view_kind=StateViewKind.ANALYSIS,
        state_role=StateRole.RECONSTRUCTION,
        value_basis=ValueBasis.DERIVED,
        frame=analysis_frame,
        source_time_axis=_time_axis(),
        sample_count=len(tracklet.contacts),
        sample_asset=analysis_asset,
        channel_descriptors=(
            ChannelDescriptor(
                channel_id="position_enu",
                semantic_role="position",
                component_names=("east", "north", "up"),
                units=("m", "m", "m"),
                frame_id=analysis_frame.frame_id,
                state_role=StateRole.RECONSTRUCTION,
                value_basis=ValueBasis.DERIVED,
                access_class=AccessClass.CLASSIFIER_CANDIDATE,
                source_fields=("position_geodetic",),
                lineage_step_ids=(LOCAL_TANGENT_STEP_ID,),
                validity_reference="position_valid",
                notes="Up is NaN with validity false.",
            ),
            ChannelDescriptor(
                channel_id="reported_velocity_enu",
                semantic_role="velocity",
                component_names=("east", "north", "up"),
                units=("m/s", "m/s", "m/s"),
                frame_id=analysis_frame.frame_id,
                state_role=StateRole.RECONSTRUCTION,
                value_basis=ValueBasis.DERIVED,
                access_class=AccessClass.CLASSIFIER_CANDIDATE,
                source_fields=("speed_over_ground", "course_over_ground"),
                lineage_step_ids=(REPORTED_VELOCITY_STEP_ID,),
                validity_reference="velocity_valid",
                notes="Up is NaN with validity false.",
            ),
        ),
        processing_step_ids=(LOCAL_TANGENT_STEP_ID, REPORTED_VELOCITY_STEP_ID),
    )

    route_group = stable_group_id(dataset_id=dataset_id, namespace="route", raw_value=tracklet.route_id)
    recording_group = stable_group_id(
        dataset_id=dataset_id,
        namespace="source_recording",
        raw_value=f"{source_artifact_id}:{tracklet.tracklet_id}",
    )
    temporal_group = stable_group_id(
        dataset_id=dataset_id,
        namespace="temporal_collection",
        raw_value=datetime.fromtimestamp(start_unix, tz=UTC).strftime("%Y-%m"),
    )
    processing_steps = (
        PARSE_STEP_ID,
        LOCAL_TANGENT_STEP_ID,
        REPORTED_VELOCITY_STEP_ID,
        CLASSIFIER_DEDUPLICATE_STEP_ID,
    )
    manifest = TrajectoryEpisodeManifest(
        corpus_snapshot_id=corpus_snapshot_id,
        episode_id=episode_id,
        primary_program_domain=ProgramDomain.SEA,
        corpus_sublane="sea_surface",
        default_operating_environment="water_surface",
        default_motion_regime="surface_navigation",
        source_dataset_id=dataset_id,
        source_artifact_ids=(source_artifact_id,),
        observation_modality="cooperative_ais",
        platform_group_id=platform_group,
        start_time=start_time,
        end_time=end_time,
        state_views=(source_view, analysis_view),
        segments=(
            TrajectorySegment(
                segment_id=f"{episode_id}:full",
                start_offset_s=0.0,
                end_offset_s=duration_s,
                operating_environment="water_surface",
                motion_regime="surface_navigation",
                evidence_kind="source_tracklet_extent",
            ),
        ),
        labels=(
            LabelAssertion(
                assertion_id=f"{episode_id}:route",
                namespace="route",
                value=tracklet.route_id,
                evidence_kind=LabelEvidenceKind.NATIVE,
                evidence_strength=EvidenceStrength.STRONG,
                source_reference="route",
                proxy=False,
                start_offset_s=0.0,
                end_offset_s=duration_s,
                vocabulary_version="cmre-maritime-routes-v1",
                notes="Route association is not a vessel-family label.",
            ),
        ),
        grouping_keys=(
            GroupingKey(
                namespace=GroupingNamespace.PHYSICAL_PLATFORM,
                opaque_value=platform_group,
                scope="source MMSI within the dataset",
                evidence_strength=EvidenceStrength.STRONG,
            ),
            GroupingKey(
                namespace=GroupingNamespace.SOURCE_RECORDING,
                opaque_value=recording_group,
                scope="source artifact plus tracklet",
                evidence_strength=EvidenceStrength.STRONG,
            ),
            GroupingKey(
                namespace=GroupingNamespace.ROUTE,
                opaque_value=route_group,
                scope="source route",
                evidence_strength=EvidenceStrength.STRONG,
            ),
            GroupingKey(
                namespace=GroupingNamespace.SOURCE_DATASET,
                opaque_value=stable_group_id(dataset_id=dataset_id, namespace="dataset", raw_value=dataset_id),
                scope="source dataset",
                evidence_strength=EvidenceStrength.STRONG,
            ),
            GroupingKey(
                namespace=GroupingNamespace.GEOGRAPHY,
                opaque_value=stable_group_id(dataset_id=dataset_id, namespace="geography", raw_value="brest_france_routes"),
                scope="Brest route network",
                evidence_strength=EvidenceStrength.STRONG,
            ),
            GroupingKey(
                namespace=GroupingNamespace.TEMPORAL_COLLECTION,
                opaque_value=temporal_group,
                scope="calendar month",
                evidence_strength=EvidenceStrength.STRONG,
            ),
        ),
        quality_summary=_quality(source),
        processing_step_ids=processing_steps,
        domain_extension=DomainExtension(
            schema_id="sea_surface_ais_route_tracklet_v0.1",
            schema_version="0.1.0",
            payload={
                "source_tracklet_id": tracklet.tracklet_id,
                "route_nomenclature": {
                    "route_id": route.route_id,
                    "origin_port": route.origin_port,
                    "destination_port": route.destination_port,
                    "nominal_length_m": route.nominal_length_m,
                },
                "source_artifact_sha256": source_artifact_sha256,
                "raw_identity_access": "adapter_only",
            },
        ),
        classifier_trajectory_view=ClassifierTrajectoryView(
            episode_id=episode_id,
            state_view_id=analysis_view.state_view_id,
            asset=classifier_asset,
            sample_count=int(np.asarray(classifier["elapsed_s"]).size),
            frame_id=analysis_frame.frame_id,
            processing_step_ids=(CLASSIFIER_DEDUPLICATE_STEP_ID,),
            target_labels_stored_outside_asset=True,
            identity_and_grouping_values_excluded=True,
        ),
    )
    episode_path = root / "episodes" / f"{episode_id}.json"
    episode_path.parent.mkdir(parents=True, exist_ok=True)
    episode_path.write_text(manifest.model_dump_json(indent=2, exclude_none=True) + "\n")
    return manifest
####


def build_fixture(
    *,
    tracklets_path: str | Path,
    nomenclature_path: str | Path,
    output_root: str | Path,
    source_artifact_id: str,
    corpus_snapshot_id: str,
    selected_tracklet_ids: set[int] | None = None,
    dataset_id: str = DEFAULT_DATASET_ID,
) -> FixtureBuildResult:
    tracklets = parse_tracklets(tracklets_path, selected_tracklet_ids=selected_tracklet_ids)
    if not tracklets:
        raise ValueError("selected source contains no tracklets")
    routes = parse_route_nomenclature(nomenclature_path)
    source_sha = sha256_file(tracklets_path)
    manifests = tuple(
        build_episode(
            output_root=output_root,
            tracklet=tracklet,
            route=routes[tracklet.route_id],
            source_artifact_id=source_artifact_id,
            source_artifact_sha256=source_sha,
            corpus_snapshot_id=corpus_snapshot_id,
            dataset_id=dataset_id,
        )
        for tracklet in tracklets
    )
    counts = Counter(item.platform_group_id for item in manifests)
    repeated = tuple(sorted(key for key, count in counts.items() if key is not None and count > 1))
    return FixtureBuildResult(
        manifests=manifests,
        physical_platform_group_count=len(counts),
        repeated_physical_platform_groups=repeated,
    )
####


def write_fixture_index(*, output_root: str | Path, result: FixtureBuildResult) -> Path:
    root = Path(output_root)
    payload = {
        "fixture_version": "sea-surface-real-fixture-v0.3",
        "contract_version": "trajectory-corpus-v0.1",
        "episode_count": len(result.manifests),
        "physical_platform_group_count": result.physical_platform_group_count,
        "repeated_physical_platform_groups": list(result.repeated_physical_platform_groups),
        "episodes": [
            {
                "episode_id": item.episode_id,
                "manifest_path": f"episodes/{item.episode_id}.json",
                "manifest_sha256": sha256_file(root / "episodes" / f"{item.episode_id}.json"),
            }
            for item in result.manifests
        ],
    }
    path = root / "fixture_index.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
####


__all__ = [
    "AisContact",
    "FixtureBuildResult",
    "RouteDefinition",
    "RouteTracklet",
    "build_episode",
    "build_fixture",
    "local_enu_position",
    "parse_route_nomenclature",
    "parse_tracklets",
    "reported_velocity_enu",
    "sha256_file",
    "stable_group_id",
    "write_fixture_index",
]
