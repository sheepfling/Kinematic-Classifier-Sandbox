from __future__ import annotations

import pytest
from pydantic import ValidationError

from kinematic_classifier_sandbox.corpus.real_world.episode_contracts import (
    AccessClass,
    AssetReference,
    ChannelDescriptor,
    ClassifierTrajectoryView,
    EvidenceStrength,
    FrameDescriptor,
    GroupingKey,
    GroupingNamespace,
    ProgramDomain,
    QualitySummary,
    StateRole,
    StateViewKind,
    TimeAxisDescriptor,
    TrajectoryEpisodeManifest,
    TrajectoryStateViewManifest,
    ValueBasis,
)


_SHA256 = "0" * 64


def _state_view() -> TrajectoryStateViewManifest:
    frame = FrameDescriptor(
        frame_id="air-local-enu",
        frame_kind="cartesian_local_tangent",
        axes=("east", "north", "up"),
        axis_units=("m", "m", "m"),
        center_or_origin="first observation",
        vertical_reference="mean_sea_level",
        vertical_positive_direction="up",
    )
    return TrajectoryStateViewManifest(
        state_view_id="flight-1:analysis",
        view_kind=StateViewKind.ANALYSIS,
        state_role=StateRole.OBSERVATION,
        value_basis=ValueBasis.DERIVED,
        frame=frame,
        source_time_axis=TimeAxisDescriptor(
            source_time_system="UTC",
            normalized_time_system="relative_seconds",
            absolute_time_available=True,
            source_epoch_or_reference="Unix epoch",
            elapsed_origin="first observation",
            precision_or_resolution="1 second",
            rollover_policy="none",
            leap_second_policy="source-defined",
        ),
        sample_count=2,
        sample_asset=AssetReference(
            path="state_views/flight-1.npz",
            media_type="application/x-npz",
            sha256=_SHA256,
        ),
        channel_descriptors=(
            ChannelDescriptor(
                channel_id="position",
                semantic_role="position",
                component_names=("east", "north", "up"),
                units=("m", "m", "m"),
                frame_id=frame.frame_id,
                state_role=StateRole.OBSERVATION,
                value_basis=ValueBasis.DERIVED,
                access_class=AccessClass.CLASSIFIER_CANDIDATE,
                lineage_step_ids=("normalize",),
                validity_reference="position_valid",
            ),
        ),
        processing_step_ids=("normalize",),
    )
####


def _manifest() -> TrajectoryEpisodeManifest:
    state_view = _state_view()
    return TrajectoryEpisodeManifest(
        corpus_snapshot_id="cross-domain-contract-test",
        episode_id="flight-1",
        primary_program_domain=ProgramDomain.AIR,
        corpus_sublane="air_atmospheric",
        default_operating_environment="atmosphere",
        default_motion_regime="flight",
        source_dataset_id="test-air-source",
        source_artifact_ids=("artifact-1",),
        observation_modality="ads_b",
        platform_group_id="aircraft-group-1",
        state_views=(state_view,),
        grouping_keys=(
            GroupingKey(
                namespace=GroupingNamespace.PHYSICAL_PLATFORM,
                opaque_value="aircraft-group-1",
                scope="aircraft identity",
                evidence_strength=EvidenceStrength.STRONG,
            ),
        ),
        quality_summary=QualitySummary(
            disposition="usable",
            sample_count=2,
            duration_s=1.0,
            median_sample_interval_s=1.0,
            maximum_gap_s=1.0,
            duplicate_timestamp_count=0,
            out_of_order_timestamp_count=0,
        ),
        processing_step_ids=("normalize", "project-classifier"),
        classifier_trajectory_view=ClassifierTrajectoryView(
            episode_id="flight-1",
            state_view_id=state_view.state_view_id,
            asset=AssetReference(
                path="classifier_views/flight-1.npz",
                media_type="application/x-npz",
                sha256=_SHA256,
            ),
            sample_count=2,
            frame_id=state_view.frame.frame_id,
            processing_step_ids=("project-classifier",),
            target_labels_stored_outside_asset=True,
            identity_and_grouping_values_excluded=True,
        ),
    )
####


def test_episode_contract_is_cross_domain_not_sea_specific() -> None:
    manifest = _manifest()
    assert manifest.primary_program_domain is ProgramDomain.AIR
    assert manifest.corpus_sublane == "air_atmospheric"
####


def test_asset_reference_rejects_path_traversal() -> None:
    with pytest.raises(ValidationError, match="may not traverse parents"):
        AssetReference(
            path="../secret.bin",
            media_type="application/octet-stream",
            sha256=_SHA256,
        )
    ####
####


def test_classifier_view_requires_leakage_boundary_flags() -> None:
    with pytest.raises(ValidationError, match="must exclude identity"):
        ClassifierTrajectoryView(
            episode_id="flight-1",
            state_view_id="flight-1:analysis",
            asset=AssetReference(
                path="classifier_views/flight-1.npz",
                media_type="application/x-npz",
                sha256=_SHA256,
            ),
            sample_count=2,
            frame_id="air-local-enu",
            processing_step_ids=("project-classifier",),
            target_labels_stored_outside_asset=True,
            identity_and_grouping_values_excluded=False,
        )
    ####
####


def test_episode_requires_split_capable_grouping() -> None:
    payload = _manifest().model_dump()
    payload["platform_group_id"] = None
    payload["grouping_keys"] = (
        GroupingKey(
            namespace=GroupingNamespace.ROUTE,
            opaque_value="route-1",
            scope="route",
            evidence_strength=EvidenceStrength.STRONG,
        ),
    )
    with pytest.raises(ValidationError, match="split-capable grouping key"):
        TrajectoryEpisodeManifest.model_validate(payload)
    ####
####
