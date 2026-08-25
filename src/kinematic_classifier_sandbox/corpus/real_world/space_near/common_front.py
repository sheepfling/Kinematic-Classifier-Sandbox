"""Convert bounded SPACE-NEAR fixture evidence into the common episode contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from ..episode_contracts import (
    AccessClass,
    AssetReference,
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
    QualitySummary,
    StateRole,
    StateViewKind,
    TimeAxisDescriptor,
    TrajectoryEpisodeManifest,
    TrajectorySegment,
    TrajectoryStateViewManifest,
    ValueBasis,
)
from .fixture_models import EmbeddedFixture, EmbeddedStateView

_STATE_ROLE_BY_SOURCE_VALUE = {
    "observation": StateRole.OBSERVATION,
    "estimate": StateRole.ESTIMATE,
    "reference": StateRole.REFERENCE,
    "reference_solution": StateRole.REFERENCE,
}
_EVIDENCE_KIND_BY_SOURCE_VALUE = {
    "trajectory_derived_from_reference_solution": LabelEvidenceKind.DERIVED,
    "trajectory_derived_monotonic_height": LabelEvidenceKind.DERIVED,
    "source_documented_interval_semantics": LabelEvidenceKind.RECONCILED,
    "source_documented_event": LabelEvidenceKind.RECONCILED,
    "publisher_documentation": LabelEvidenceKind.RECONCILED,
    "publisher_documentation_and_native_trajectory": LabelEvidenceKind.RECONCILED,
}
_EVIDENCE_STRENGTH_BY_SOURCE_VALUE = {
    "weak": EvidenceStrength.WEAK,
    "moderate": EvidenceStrength.MEDIUM,
    "medium": EvidenceStrength.MEDIUM,
    "strong": EvidenceStrength.STRONG,
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _state_role(source_value: str) -> StateRole:
    return _STATE_ROLE_BY_SOURCE_VALUE.get(source_value, StateRole.REFERENCE)


def _time_axis(fixture: EmbeddedFixture) -> TimeAxisDescriptor:
    source = fixture.source
    return TimeAxisDescriptor(
        source_time_system="UTC",
        normalized_time_system="unix_utc_seconds",
        absolute_time_available=True,
        source_epoch_or_reference=fixture.episode.absolute_start_time,
        elapsed_origin=fixture.episode.absolute_start_time,
        precision_or_resolution=f"{source.nominal_sample_interval_s:g} s nominal",
        rollover_policy="none",
        leap_second_policy="source-declared UTC; no leap-second rewrite",
    )


def _frame_descriptors(fixture: EmbeddedFixture) -> tuple[FrameDescriptor, FrameDescriptor]:
    episode = fixture.episode
    source_frame = FrameDescriptor(
        frame_id=f"{episode.episode_id}:source-native-frame",
        frame_kind="source_native",
        axes=("source_record",),
        axis_units=("source_defined",),
        center_or_origin="source-defined mission frame",
        vertical_reference="source-defined or unresolved",
        vertical_positive_direction="source-defined or unresolved",
        crs_or_datum=episode.analysis_frame.crs_or_datum,
    )
    analysis_frame = FrameDescriptor(
        frame_id=episode.analysis_frame.frame_id,
        frame_kind=episode.analysis_frame.frame_kind,
        axes=episode.analysis_frame.axes,
        axis_units=episode.analysis_frame.axis_units,
        center_or_origin="Earth center",
        vertical_reference=episode.analysis_frame.crs_or_datum or "unresolved",
        vertical_positive_direction="outward from Earth center",
        crs_or_datum=episode.analysis_frame.crs_or_datum,
    )
    return source_frame, analysis_frame


def _channel_descriptors(
    view: EmbeddedStateView,
    *,
    frame: FrameDescriptor,
    state_role: StateRole,
    value_basis: ValueBasis,
    access_class: AccessClass,
    additional_channel_ids: Iterable[str] = (),
) -> tuple[ChannelDescriptor, ...]:
    channel_ids = tuple(dict.fromkeys((*view.channel_ids, *additional_channel_ids)))
    component_names = (
        ("x", "y", "z")
        if view.view_kind == "analysis" and "position" in view.channel_ids[0]
        else ("value",)
    )
    units = ("m", "m", "m") if len(component_names) == 3 else ("source_defined",)
    return tuple(
        ChannelDescriptor(
            channel_id=channel_id,
            semantic_role="trajectory_state_or_label_dependency",
            component_names=component_names,
            units=units,
            frame_id=frame.frame_id,
            state_role=state_role,
            value_basis=value_basis,
            access_class=access_class,
            source_fields=view.columns,
            lineage_step_ids=view.processing_step_ids,
            notes=(
                "Source fixture channel retained for semantic audit; classifier eligibility "
                "remains blocked pending authoritative common-front validation."
            ),
        )
        for channel_id in channel_ids
    )


def _state_view(
    fixture: EmbeddedFixture,
    view: EmbeddedStateView,
    *,
    frame: FrameDescriptor,
    state_view_id: str,
    view_kind: StateViewKind,
    state_role: StateRole,
    value_basis: ValueBasis,
    additional_channel_ids: Iterable[str] = (),
) -> TrajectoryStateViewManifest:
    return TrajectoryStateViewManifest(
        state_view_id=state_view_id,
        view_kind=view_kind,
        state_role=state_role,
        value_basis=value_basis,
        frame=frame,
        source_time_axis=_time_axis(fixture),
        sample_count=len(view.rows),
        sample_asset=AssetReference(
            path=f"assets/{view_kind.value}/{fixture.episode.episode_id}.json",
            media_type="application/json",
            sha256=_sha256_bytes(
                json.dumps(
                    view.model_dump(mode="json"),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ),
        ),
        channel_descriptors=_channel_descriptors(
            view,
            frame=frame,
            state_role=state_role,
            value_basis=value_basis,
            access_class=(
                AccessClass.AUDIT_ONLY
                if view_kind is not StateViewKind.ANALYSIS
                else AccessClass.CLASSIFIER_CANDIDATE
            ),
            additional_channel_ids=additional_channel_ids,
        ),
        normalization_assumptions=tuple(fixture.episode.analysis_frame.unresolved_ambiguities),
        processing_step_ids=view.processing_step_ids,
    )


def _label_assertions(fixture: EmbeddedFixture) -> tuple[LabelAssertion, ...]:
    labels: list[LabelAssertion] = []
    for source_label in fixture.episode.label_assertions:
        evidence_kind = _EVIDENCE_KIND_BY_SOURCE_VALUE.get(
            source_label.model_extra.get("evidence_kind", ""),
            LabelEvidenceKind.RECONCILED,
        )
        strength = _EVIDENCE_STRENGTH_BY_SOURCE_VALUE.get(
            source_label.model_extra.get("evidence_strength", ""),
            EvidenceStrength.MEDIUM,
        )
        labels.append(
            LabelAssertion(
                assertion_id=source_label.assertion_id,
                namespace=source_label.namespace,
                value=source_label.value,
                evidence_kind=evidence_kind,
                evidence_strength=strength,
                source_reference=(
                    source_label.model_extra.get("source_reference")
                    or fixture.source.citation
                ),
                proxy=source_label.model_extra.get("proxy", False),
                start_offset_s=source_label.start_offset_s,
                end_offset_s=source_label.end_offset_s,
                confidence=source_label.model_extra.get("confidence"),
                dependency_channel_ids=source_label.dependency_channel_ids,
                vocabulary_version=source_label.model_extra.get("vocabulary_version"),
                notes="; ".join(
                    value
                    for value in (
                        source_label.model_extra.get("claim_boundary"),
                        source_label.model_extra.get("notes"),
                        "Original fixture evidence kind preserved in domain_extension",
                    )
                    if value
                ),
            )
        )
    return tuple(labels)


def _quality_summary(fixture: EmbeddedFixture) -> QualitySummary:
    view = fixture.episode.analysis_view
    elapsed_index = view.columns.index("elapsed_s")
    elapsed = tuple(float(row[elapsed_index]) for row in view.rows)
    deltas = tuple(right - left for left, right in zip(elapsed, elapsed[1:]))
    positive = tuple(delta for delta in deltas if delta > 0.0)
    sorted_positive = sorted(positive)
    midpoint = len(sorted_positive) // 2
    median = (
        sorted_positive[midpoint]
        if len(sorted_positive) % 2
        else (sorted_positive[midpoint - 1] + sorted_positive[midpoint]) / 2
    ) if sorted_positive else 0.0
    findings: list[QualityFinding] = []
    for raw in fixture.episode.quality_findings:
        interval = raw.get("interval")
        findings.append(
            QualityFinding(
                code=str(raw["code"]),
                severity=QualitySeverity(str(raw["severity"])),
                message=str(raw["message"]),
                metric=str(raw["metric"]) if raw.get("metric") is not None else None,
                value=raw.get("value"),
                unit=str(raw["unit"]) if raw.get("unit") is not None else None,
                start_offset_s=(
                    float(interval["start_offset_s"])
                    if isinstance(interval, dict) and interval.get("start_offset_s") is not None
                    else None
                ),
                end_offset_s=(
                    float(interval["end_offset_s"])
                    if isinstance(interval, dict) and interval.get("end_offset_s") is not None
                    else None
                ),
                source_reference=(
                    str(raw["source_reference"])
                    if raw.get("source_reference") is not None
                    else None
                ),
            )
        )
    return QualitySummary(
        disposition=fixture.episode.quality_disposition,
        sample_count=len(elapsed),
        duration_s=max(0.0, elapsed[-1] - elapsed[0]),
        median_sample_interval_s=median,
        maximum_gap_s=max((abs(delta) for delta in deltas), default=0.0),
        duplicate_timestamp_count=sum(delta == 0.0 for delta in deltas),
        out_of_order_timestamp_count=sum(delta < 0.0 for delta in deltas),
        findings=tuple(findings),
    )


def _write_state_asset(root: Path, view: EmbeddedStateView, episode_id: str) -> None:
    path = root / "assets" / view.view_kind / f"{episode_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(view.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    path.write_text(payload, encoding="utf-8")


def build_fixture_episode_manifest(
    fixture: EmbeddedFixture,
    *,
    output_root: str | Path,
    corpus_snapshot_id: str,
    source_artifact_id: str,
) -> TrajectoryEpisodeManifest:
    """Write a validation-only common-front episode for one bounded fixture.

    The returned manifest intentionally has no classifier trajectory view. Contract construction
    is validated here, while the source fixture portfolio's authoritative semantic sign-off remains
    pending; this preserves coverage and quality evidence without converting semantic fixtures into
    classifier data.
    """

    root = Path(output_root)
    episode = fixture.episode
    source_frame, analysis_frame = _frame_descriptors(fixture)
    dependency_ids = tuple(
        dict.fromkeys(
            dependency
            for label in episode.label_assertions
            for dependency in label.dependency_channel_ids
        )
    )
    source_view = _state_view(
        fixture,
        episode.source_native_view,
        frame=source_frame,
        state_view_id=f"{episode.episode_id}:source-native",
        view_kind=(
            StateViewKind.REFERENCE
            if episode.source_native_view.view_kind == "reference"
            else StateViewKind.SOURCE_NATIVE
        ),
        state_role=_state_role(episode.source_native_view.state_role),
        value_basis=ValueBasis.REPORTED,
        additional_channel_ids=dependency_ids,
    )
    analysis_view = _state_view(
        fixture,
        episode.analysis_view,
        frame=analysis_frame,
        state_view_id=f"{episode.episode_id}:analysis",
        view_kind=StateViewKind.ANALYSIS,
        state_role=StateRole.RECONSTRUCTION,
        value_basis=ValueBasis.DERIVED,
    )
    _write_state_asset(root, episode.source_native_view, episode.episode_id)
    _write_state_asset(root, episode.analysis_view, episode.episode_id)
    processing_steps = tuple(
        dict.fromkeys(step.identifier for step in episode.processing_steps)
    )
    elapsed_index = episode.analysis_view.columns.index("elapsed_s")
    elapsed = tuple(float(row[elapsed_index]) for row in episode.analysis_view.rows)
    grouping_keys = tuple(
        GroupingKey(
            namespace=GroupingNamespace(source_key.namespace),
            opaque_value=source_key.opaque_value,
            scope="space-near fixture portfolio",
            evidence_strength=EvidenceStrength.MEDIUM,
        )
        for source_key in episode.grouping_keys
    )
    return TrajectoryEpisodeManifest(
        corpus_snapshot_id=corpus_snapshot_id,
        episode_id=episode.episode_id,
        primary_program_domain=ProgramDomain.SPACE,
        corpus_sublane="space_near",
        default_operating_environment="near_space",
        default_motion_regime=episode.default_motion_regime,
        source_dataset_id=episode.source_dataset_id,
        source_artifact_ids=(source_artifact_id,),
        observation_modality=episode.observation_modality,
        platform_group_id=next(
            key.opaque_value
            for key in grouping_keys
            if key.namespace is GroupingNamespace.PHYSICAL_PLATFORM
        ),
        mission_id=episode.mission_id,
        object_id=episode.object_id,
        start_time=episode.absolute_start_time,
        end_time=episode.absolute_end_time,
        state_views=(source_view, analysis_view),
        segments=(
            TrajectorySegment(
                segment_id=f"{episode.episode_id}:fixture-window",
                start_offset_s=elapsed[0],
                end_offset_s=elapsed[-1],
                operating_environment="near_space",
                motion_regime=episode.default_motion_regime,
                evidence_kind="bounded_fixture_window",
            ),
        ),
        labels=_label_assertions(fixture),
        grouping_keys=grouping_keys,
        quality_summary=_quality_summary(fixture),
        processing_step_ids=processing_steps,
        domain_extension=DomainExtension(
            schema_id="space_near_fixture_common_front_v0.1",
            schema_version="0.1.0",
            payload={
                "fixture_id": fixture.fixture_id,
                "common_front_contract_validation": "passed",
                "authoritative_common_front_validation": "pending",
                "source_asset_id": fixture.source.source_asset_id,
                "source_asset_sha256": fixture.source.source_asset_sha256,
                "source_claim_boundary": fixture.source.claim_boundary,
                "source_view_content_sha256": episode.source_native_view.content_sha256,
                "analysis_view_content_sha256": episode.analysis_view.content_sha256,
                "classifier_view_status": "intentionally_blocked",
            },
        ),
        classifier_trajectory_view=None,
    )


__all__ = ["build_fixture_episode_manifest"]
