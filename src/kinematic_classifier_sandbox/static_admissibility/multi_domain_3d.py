from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from math import log, sqrt
from pathlib import Path

import pandas
import yaml
from matplotlib import patches
from matplotlib.colors import ListedColormap
from numpy import array

from kinematic_classifier_sandbox.reports.markdown import MarkdownDocument
from kinematic_classifier_sandbox.utils.io import _copy_file, _write_text, write_csv
from kinematic_classifier_sandbox.utils.plotting import plt
from kinematic_classifier_sandbox.utils.runtime import repo_root

from .io import build_static_admissibility_result, write_static_admissibility_packet
from .schemas import load_static_admissibility_config
from .validation import validate_static_admissibility_packet


@dataclass(frozen=True, slots=True)
class MultiDomain3dPacket:
    packet_dir: Path
    readme_path: Path
    quickstart_path: Path
    decision_card_path: Path
    validation_report_path: Path
    claim_boundary_path: Path
    packet_manifest_path: Path
    hero_chart_manifest_path: Path
    lane_proof_matrix_path: Path
    automated_brief_path: Path
    latex_path: Path
    estimator_reliability_report_path: Path


CLASS_ROWS: tuple[dict[str, str], ...] = (
    {
        "class_id": "ground_wheeled_vehicle",
        "domain": "land",
        "class_description": "road-like ground motion archetype",
        "requires_feature_group": "observable_3d_kinematics|external_context",
        "expected_confusions": "ground_tracked_vehicle",
        "decisionability_notes": "Confusable with tracked motion unless road/context features are allowed.",
    },
    {
        "class_id": "ground_tracked_vehicle",
        "domain": "land",
        "class_description": "slower and heavier ground mobility archetype",
        "requires_feature_group": "observable_3d_kinematics|external_context",
        "expected_confusions": "ground_wheeled_vehicle",
        "decisionability_notes": "Needs terrain or road adherence context to separate from wheeled motion.",
    },
    {
        "class_id": "surface_maritime_vessel",
        "domain": "maritime",
        "class_description": "surface vessel-like trajectory archetype",
        "requires_feature_group": "observable_3d_kinematics|external_context",
        "expected_confusions": "fast_surface_craft|fixed_wing_aircraft",
        "decisionability_notes": "Water context matters; otherwise slow low-altitude tracks can overlap.",
    },
    {
        "class_id": "fast_surface_craft",
        "domain": "maritime",
        "class_description": "faster maneuvering surface craft archetype",
        "requires_feature_group": "observable_3d_kinematics|external_context",
        "expected_confusions": "surface_maritime_vessel|fixed_wing_aircraft",
        "decisionability_notes": "Needs water adherence plus altitude context to avoid low-air overlap.",
    },
    {
        "class_id": "subsurface_contact",
        "domain": "maritime",
        "class_description": "subsurface or underwater contact archetype",
        "requires_feature_group": "blocked_or_conditional_features|sensor_quality_provenance",
        "expected_confusions": "surface_maritime_vessel",
        "decisionability_notes": "Unsupported under pure 3D kinematic tracks unless depth or acoustic evidence is allowed.",
    },
    {
        "class_id": "rotary_wing_aircraft",
        "domain": "air",
        "class_description": "hover and vertical maneuver capable air track",
        "requires_feature_group": "observable_3d_kinematics",
        "expected_confusions": "multirotor_uas",
        "decisionability_notes": "Loiter and climb evidence help, but scale is unavailable in pure kinematics.",
    },
    {
        "class_id": "fixed_wing_aircraft",
        "domain": "air",
        "class_description": "smooth higher-speed air track",
        "requires_feature_group": "observable_3d_kinematics",
        "expected_confusions": "fixed_wing_uas|fast_surface_craft",
        "decisionability_notes": "Can overlap fixed-wing UAS without size or sensor-specific features.",
    },
    {
        "class_id": "multirotor_uas",
        "domain": "air",
        "class_description": "low-speed hover and loiter small UAS",
        "requires_feature_group": "observable_3d_kinematics",
        "expected_confusions": "rotary_wing_aircraft",
        "decisionability_notes": "Hover and turn behavior help, but scale is still missing.",
    },
    {
        "class_id": "fixed_wing_uas",
        "domain": "air",
        "class_description": "small fixed-wing air track",
        "requires_feature_group": "observable_3d_kinematics",
        "expected_confusions": "fixed_wing_aircraft",
        "decisionability_notes": "Needs size or sensor context to separate from crewed fixed-wing tracks.",
    },
    {
        "class_id": "balloon_aerostat",
        "domain": "air",
        "class_description": "slow drifting elevated track",
        "requires_feature_group": "observable_3d_kinematics",
        "expected_confusions": "geo_stationkeeping_object",
        "decisionability_notes": "Rare and prior-sensitive; needs altitude plus low-acceleration evidence.",
    },
    {
        "class_id": "ballistic_arc_object",
        "domain": "high_dynamic",
        "class_description": "unpowered arc-like 3D trajectory",
        "requires_feature_group": "model_fit_regime_consistency",
        "expected_confusions": "boost_coast_object",
        "decisionability_notes": "Needs ballistic-fit evidence and enough temporal extent.",
    },
    {
        "class_id": "boost_coast_object",
        "domain": "high_dynamic",
        "class_description": "sustained acceleration then coast archetype",
        "requires_feature_group": "model_fit_regime_consistency",
        "expected_confusions": "ballistic_arc_object",
        "decisionability_notes": "Pointwise evidence is weak; sustained acceleration segments matter.",
    },
    {
        "class_id": "leo_orbital_object",
        "domain": "space",
        "class_description": "low-orbit-like short arc object",
        "requires_feature_group": "model_fit_regime_consistency|external_context",
        "expected_confusions": "geo_stationkeeping_object",
        "decisionability_notes": "Needs orbital-fit features and adequate arc length.",
    },
    {
        "class_id": "geo_stationkeeping_object",
        "domain": "space",
        "class_description": "high-altitude low-apparent-motion stationkeeping object",
        "requires_feature_group": "model_fit_regime_consistency|external_context",
        "expected_confusions": "balloon_aerostat|leo_orbital_object",
        "decisionability_notes": "Weak on short arcs without orbital context and stationkeeping consistency.",
    },
)


FEATURE_ROWS: tuple[dict[str, object], ...] = (
    {
        "feature_id": "duration_s",
        "feature_group": "observable_3d_kinematics",
        "description": "track duration",
        "value_type": "normalized_scalar",
        "observable_from_3d_track": True,
        "online_available": True,
        "uses_future_window": False,
        "uses_generator_metadata": False,
        "uses_identity_or_catalog_lookup": False,
        "label_rule_overlap": False,
        "external_context_required": False,
        "redundancy_group": "duration",
        "candidate_synergy_group": "",
        "allowed_for_static_audit": True,
        "leakage_status": "pass",
        "notes": "Redundant with sample count at fixed cadence.",
    },
    {
        "feature_id": "sample_count",
        "feature_group": "observable_3d_kinematics",
        "description": "number of observations",
        "value_type": "normalized_scalar",
        "observable_from_3d_track": True,
        "online_available": True,
        "uses_future_window": False,
        "uses_generator_metadata": False,
        "uses_identity_or_catalog_lookup": False,
        "label_rule_overlap": False,
        "external_context_required": False,
        "redundancy_group": "duration",
        "candidate_synergy_group": "",
        "allowed_for_static_audit": True,
        "leakage_status": "pass",
        "notes": "Coverage proxy; redundant with duration in fixed-cadence studies.",
    },
    {
        "feature_id": "path_efficiency_3d",
        "feature_group": "observable_3d_kinematics",
        "description": "displacement divided by path length",
        "value_type": "normalized_scalar",
        "observable_from_3d_track": True,
        "online_available": False,
        "uses_future_window": True,
        "uses_generator_metadata": False,
        "uses_identity_or_catalog_lookup": False,
        "label_rule_overlap": False,
        "external_context_required": False,
        "redundancy_group": "",
        "candidate_synergy_group": "loiter_turn",
        "allowed_for_static_audit": True,
        "leakage_status": "conditional",
        "notes": "Useful for full-window static screening but not online-safe.",
    },
    {
        "feature_id": "speed_3d_mean",
        "feature_group": "observable_3d_kinematics",
        "description": "mean 3D speed",
        "value_type": "normalized_scalar",
        "observable_from_3d_track": True,
        "online_available": True,
        "uses_future_window": False,
        "uses_generator_metadata": False,
        "uses_identity_or_catalog_lookup": False,
        "label_rule_overlap": False,
        "external_context_required": False,
        "redundancy_group": "speed",
        "candidate_synergy_group": "",
        "allowed_for_static_audit": True,
        "leakage_status": "pass",
        "notes": "Broad separability feature.",
    },
    {
        "feature_id": "speed_3d_median",
        "feature_group": "observable_3d_kinematics",
        "description": "median 3D speed",
        "value_type": "normalized_scalar",
        "observable_from_3d_track": True,
        "online_available": True,
        "uses_future_window": False,
        "uses_generator_metadata": False,
        "uses_identity_or_catalog_lookup": False,
        "label_rule_overlap": False,
        "external_context_required": False,
        "redundancy_group": "speed",
        "candidate_synergy_group": "",
        "allowed_for_static_audit": True,
        "leakage_status": "pass",
        "notes": "Deliberate redundancy with mean speed.",
    },
    {
        "feature_id": "ground_speed_mean",
        "feature_group": "observable_3d_kinematics",
        "description": "horizontal ground speed",
        "value_type": "normalized_scalar",
        "observable_from_3d_track": True,
        "online_available": True,
        "uses_future_window": False,
        "uses_generator_metadata": False,
        "uses_identity_or_catalog_lookup": False,
        "label_rule_overlap": False,
        "external_context_required": False,
        "redundancy_group": "speed",
        "candidate_synergy_group": "",
        "allowed_for_static_audit": True,
        "leakage_status": "pass",
        "notes": "Redundant with horizontal speed family.",
    },
    {
        "feature_id": "horizontal_speed_mean",
        "feature_group": "observable_3d_kinematics",
        "description": "alternate horizontal speed estimate",
        "value_type": "normalized_scalar",
        "observable_from_3d_track": True,
        "online_available": True,
        "uses_future_window": False,
        "uses_generator_metadata": False,
        "uses_identity_or_catalog_lookup": False,
        "label_rule_overlap": False,
        "external_context_required": False,
        "redundancy_group": "speed",
        "candidate_synergy_group": "",
        "allowed_for_static_audit": True,
        "leakage_status": "pass",
        "notes": "Intentional duplicate for redundancy stress.",
    },
    {
        "feature_id": "vertical_speed_abs_p95",
        "feature_group": "observable_3d_kinematics",
        "description": "peak absolute vertical speed",
        "value_type": "normalized_scalar",
        "observable_from_3d_track": True,
        "online_available": True,
        "uses_future_window": False,
        "uses_generator_metadata": False,
        "uses_identity_or_catalog_lookup": False,
        "label_rule_overlap": False,
        "external_context_required": False,
        "redundancy_group": "vertical_motion",
        "candidate_synergy_group": "",
        "allowed_for_static_audit": True,
        "leakage_status": "pass",
        "notes": "Separates ground and air-like tracks.",
    },
    {
        "feature_id": "climb_angle_abs_mean",
        "feature_group": "observable_3d_kinematics",
        "description": "mean absolute climb angle",
        "value_type": "normalized_scalar",
        "observable_from_3d_track": True,
        "online_available": True,
        "uses_future_window": False,
        "uses_generator_metadata": False,
        "uses_identity_or_catalog_lookup": False,
        "label_rule_overlap": False,
        "external_context_required": False,
        "redundancy_group": "vertical_motion",
        "candidate_synergy_group": "vertical_altitude",
        "allowed_for_static_audit": True,
        "leakage_status": "pass",
        "notes": "Synergizes with altitude mean.",
    },
    {
        "feature_id": "altitude_mean",
        "feature_group": "observable_3d_kinematics",
        "description": "mean altitude or relative height",
        "value_type": "normalized_scalar",
        "observable_from_3d_track": True,
        "online_available": True,
        "uses_future_window": False,
        "uses_generator_metadata": False,
        "uses_identity_or_catalog_lookup": False,
        "label_rule_overlap": False,
        "external_context_required": False,
        "redundancy_group": "altitude",
        "candidate_synergy_group": "vertical_altitude",
        "allowed_for_static_audit": True,
        "leakage_status": "pass",
        "notes": "Core domain separator across surface, air, and space-like classes.",
    },
    {
        "feature_id": "altitude_span",
        "feature_group": "observable_3d_kinematics",
        "description": "vertical extent of the track",
        "value_type": "normalized_scalar",
        "observable_from_3d_track": True,
        "online_available": False,
        "uses_future_window": True,
        "uses_generator_metadata": False,
        "uses_identity_or_catalog_lookup": False,
        "label_rule_overlap": False,
        "external_context_required": False,
        "redundancy_group": "altitude",
        "candidate_synergy_group": "",
        "allowed_for_static_audit": True,
        "leakage_status": "conditional",
        "notes": "Static-screen useful, online-unsafe full-window summary.",
    },
    {
        "feature_id": "accel_norm_p95",
        "feature_group": "observable_3d_kinematics",
        "description": "peak acceleration magnitude",
        "value_type": "normalized_scalar",
        "observable_from_3d_track": True,
        "online_available": True,
        "uses_future_window": False,
        "uses_generator_metadata": False,
        "uses_identity_or_catalog_lookup": False,
        "label_rule_overlap": False,
        "external_context_required": False,
        "redundancy_group": "acceleration",
        "candidate_synergy_group": "",
        "allowed_for_static_audit": True,
        "leakage_status": "pass",
        "notes": "Noise-sensitive but useful for high-dynamic classes.",
    },
    {
        "feature_id": "turn_rate_abs_mean",
        "feature_group": "observable_3d_kinematics",
        "description": "average absolute turn rate",
        "value_type": "normalized_scalar",
        "observable_from_3d_track": True,
        "online_available": True,
        "uses_future_window": False,
        "uses_generator_metadata": False,
        "uses_identity_or_catalog_lookup": False,
        "label_rule_overlap": False,
        "external_context_required": False,
        "redundancy_group": "turn",
        "candidate_synergy_group": "loiter_turn",
        "allowed_for_static_audit": True,
        "leakage_status": "pass",
        "notes": "Candidate synergy with path efficiency and hover score.",
    },
    {
        "feature_id": "curvature_mean",
        "feature_group": "observable_3d_kinematics",
        "description": "geometric path curvature",
        "value_type": "normalized_scalar",
        "observable_from_3d_track": True,
        "online_available": True,
        "uses_future_window": False,
        "uses_generator_metadata": False,
        "uses_identity_or_catalog_lookup": False,
        "label_rule_overlap": False,
        "external_context_required": False,
        "redundancy_group": "turn",
        "candidate_synergy_group": "",
        "allowed_for_static_audit": True,
        "leakage_status": "pass",
        "notes": "Redundant with turn rate under common speed assumptions.",
    },
    {
        "feature_id": "stop_go_fraction",
        "feature_group": "observable_3d_kinematics",
        "description": "fraction of low-speed segments",
        "value_type": "normalized_scalar",
        "observable_from_3d_track": True,
        "online_available": True,
        "uses_future_window": False,
        "uses_generator_metadata": False,
        "uses_identity_or_catalog_lookup": False,
        "label_rule_overlap": False,
        "external_context_required": False,
        "redundancy_group": "",
        "candidate_synergy_group": "ground_stop_go",
        "allowed_for_static_audit": True,
        "leakage_status": "pass",
        "notes": "Can help ground mobility versus hover-like behavior.",
    },
    {
        "feature_id": "hover_loiter_score",
        "feature_group": "observable_3d_kinematics",
        "description": "composite low-displacement high-duration loiter score",
        "value_type": "normalized_scalar",
        "observable_from_3d_track": True,
        "online_available": False,
        "uses_future_window": True,
        "uses_generator_metadata": False,
        "uses_identity_or_catalog_lookup": False,
        "label_rule_overlap": False,
        "external_context_required": False,
        "redundancy_group": "",
        "candidate_synergy_group": "loiter_turn",
        "allowed_for_static_audit": True,
        "leakage_status": "conditional",
        "notes": "Derived composite; provenance should stay explicit.",
    },
    {
        "feature_id": "ground_plane_consistency",
        "feature_group": "model_fit_regime_consistency",
        "description": "consistency with ground-plane motion",
        "value_type": "normalized_scalar",
        "observable_from_3d_track": False,
        "online_available": True,
        "uses_future_window": False,
        "uses_generator_metadata": False,
        "uses_identity_or_catalog_lookup": False,
        "label_rule_overlap": False,
        "external_context_required": True,
        "redundancy_group": "",
        "candidate_synergy_group": "ground_stop_go",
        "allowed_for_static_audit": True,
        "leakage_status": "warning",
        "notes": "Requires map or geodetic context to interpret safely.",
    },
    {
        "feature_id": "ballistic_arc_fit_rmse",
        "feature_group": "model_fit_regime_consistency",
        "description": "fit residual to an arc-like ballistic model",
        "value_type": "normalized_scalar",
        "observable_from_3d_track": True,
        "online_available": False,
        "uses_future_window": True,
        "uses_generator_metadata": False,
        "uses_identity_or_catalog_lookup": False,
        "label_rule_overlap": False,
        "external_context_required": False,
        "redundancy_group": "",
        "candidate_synergy_group": "boost_ballistic",
        "allowed_for_static_audit": True,
        "leakage_status": "conditional",
        "notes": "Useful for static screening but requires temporal extent.",
    },
    {
        "feature_id": "sustained_accel_segment_score",
        "feature_group": "model_fit_regime_consistency",
        "description": "evidence for sustained acceleration segment",
        "value_type": "normalized_scalar",
        "observable_from_3d_track": True,
        "online_available": False,
        "uses_future_window": True,
        "uses_generator_metadata": False,
        "uses_identity_or_catalog_lookup": False,
        "label_rule_overlap": False,
        "external_context_required": False,
        "redundancy_group": "",
        "candidate_synergy_group": "boost_ballistic",
        "allowed_for_static_audit": True,
        "leakage_status": "conditional",
        "notes": "History-dependent evidence for boost-coast archetypes.",
    },
    {
        "feature_id": "orbital_fit_residual",
        "feature_group": "model_fit_regime_consistency",
        "description": "orbital consistency residual",
        "value_type": "normalized_scalar",
        "observable_from_3d_track": False,
        "online_available": False,
        "uses_future_window": True,
        "uses_generator_metadata": False,
        "uses_identity_or_catalog_lookup": False,
        "label_rule_overlap": False,
        "external_context_required": True,
        "redundancy_group": "",
        "candidate_synergy_group": "orbital_altitude",
        "allowed_for_static_audit": True,
        "leakage_status": "warning",
        "notes": "Only meaningful when space-like geometry and sufficient arc length exist.",
    },
    {
        "feature_id": "stationkeeping_consistency",
        "feature_group": "model_fit_regime_consistency",
        "description": "low drift plus high-altitude stationkeeping consistency",
        "value_type": "normalized_scalar",
        "observable_from_3d_track": False,
        "online_available": False,
        "uses_future_window": True,
        "uses_generator_metadata": False,
        "uses_identity_or_catalog_lookup": False,
        "label_rule_overlap": False,
        "external_context_required": True,
        "redundancy_group": "",
        "candidate_synergy_group": "",
        "allowed_for_static_audit": True,
        "leakage_status": "warning",
        "notes": "Short windows may be under-supported even when the class is conceptually valid.",
    },
    {
        "feature_id": "road_network_adherence",
        "feature_group": "external_context",
        "description": "track follows a road-like context",
        "value_type": "normalized_scalar",
        "observable_from_3d_track": False,
        "online_available": True,
        "uses_future_window": False,
        "uses_generator_metadata": False,
        "uses_identity_or_catalog_lookup": False,
        "label_rule_overlap": False,
        "external_context_required": True,
        "redundancy_group": "context",
        "candidate_synergy_group": "",
        "allowed_for_static_audit": True,
        "leakage_status": "warning",
        "notes": "Context-dependent feature; valid only when map context is declared available.",
    },
    {
        "feature_id": "water_mask_adherence",
        "feature_group": "external_context",
        "description": "track follows water-surface context",
        "value_type": "normalized_scalar",
        "observable_from_3d_track": False,
        "online_available": True,
        "uses_future_window": False,
        "uses_generator_metadata": False,
        "uses_identity_or_catalog_lookup": False,
        "label_rule_overlap": False,
        "external_context_required": True,
        "redundancy_group": "context",
        "candidate_synergy_group": "water_altitude",
        "allowed_for_static_audit": True,
        "leakage_status": "warning",
        "notes": "Useful for maritime screening but context-dependent.",
    },
    {
        "feature_id": "space_catalog_context_available",
        "feature_group": "external_context",
        "description": "whether space catalog context is allowed",
        "value_type": "binary",
        "observable_from_3d_track": False,
        "online_available": True,
        "uses_future_window": False,
        "uses_generator_metadata": False,
        "uses_identity_or_catalog_lookup": False,
        "label_rule_overlap": False,
        "external_context_required": True,
        "redundancy_group": "context",
        "candidate_synergy_group": "",
        "allowed_for_static_audit": True,
        "leakage_status": "warning",
        "notes": "Context availability is not the same as using identity lookups.",
    },
    {
        "feature_id": "measurement_noise_estimate",
        "feature_group": "sensor_quality_provenance",
        "description": "observation noise estimate",
        "value_type": "normalized_scalar",
        "observable_from_3d_track": True,
        "online_available": True,
        "uses_future_window": False,
        "uses_generator_metadata": False,
        "uses_identity_or_catalog_lookup": False,
        "label_rule_overlap": False,
        "external_context_required": False,
        "redundancy_group": "sensor_quality",
        "candidate_synergy_group": "",
        "allowed_for_static_audit": True,
        "leakage_status": "pass",
        "notes": "Audit-oriented quality feature rather than a class feature.",
    },
    {
        "feature_id": "observation_gap_rate",
        "feature_group": "sensor_quality_provenance",
        "description": "fraction of missing observation steps",
        "value_type": "normalized_scalar",
        "observable_from_3d_track": True,
        "online_available": True,
        "uses_future_window": False,
        "uses_generator_metadata": False,
        "uses_identity_or_catalog_lookup": False,
        "label_rule_overlap": False,
        "external_context_required": False,
        "redundancy_group": "sensor_quality",
        "candidate_synergy_group": "",
        "allowed_for_static_audit": True,
        "leakage_status": "pass",
        "notes": "Thin coverage and observability warning surface.",
    },
    {
        "feature_id": "true_platform_label_code",
        "feature_group": "blocked_or_conditional_features",
        "description": "synthetic perfect class code",
        "value_type": "normalized_scalar",
        "observable_from_3d_track": False,
        "online_available": False,
        "uses_future_window": False,
        "uses_generator_metadata": False,
        "uses_identity_or_catalog_lookup": True,
        "label_rule_overlap": True,
        "external_context_required": False,
        "redundancy_group": "",
        "candidate_synergy_group": "",
        "allowed_for_static_audit": False,
        "leakage_status": "blocker",
        "notes": "Direct label leakage.",
    },
    {
        "feature_id": "generator_scenario_template_id",
        "feature_group": "blocked_or_conditional_features",
        "description": "synthetic generator template identifier",
        "value_type": "normalized_scalar",
        "observable_from_3d_track": False,
        "online_available": False,
        "uses_future_window": False,
        "uses_generator_metadata": True,
        "uses_identity_or_catalog_lookup": False,
        "label_rule_overlap": False,
        "external_context_required": False,
        "redundancy_group": "",
        "candidate_synergy_group": "",
        "allowed_for_static_audit": False,
        "leakage_status": "blocker",
        "notes": "Generator leakage masquerading as feature evidence.",
    },
    {
        "feature_id": "future_endpoint_displacement",
        "feature_group": "blocked_or_conditional_features",
        "description": "full-window endpoint displacement",
        "value_type": "normalized_scalar",
        "observable_from_3d_track": True,
        "online_available": False,
        "uses_future_window": True,
        "uses_generator_metadata": False,
        "uses_identity_or_catalog_lookup": False,
        "label_rule_overlap": False,
        "external_context_required": False,
        "redundancy_group": "",
        "candidate_synergy_group": "",
        "allowed_for_static_audit": False,
        "leakage_status": "blocker",
        "notes": "Future-window leakage for online classification claims.",
    },
    {
        "feature_id": "future_max_altitude",
        "feature_group": "blocked_or_conditional_features",
        "description": "future maximum altitude statistic",
        "value_type": "normalized_scalar",
        "observable_from_3d_track": True,
        "online_available": False,
        "uses_future_window": True,
        "uses_generator_metadata": False,
        "uses_identity_or_catalog_lookup": False,
        "label_rule_overlap": False,
        "external_context_required": False,
        "redundancy_group": "",
        "candidate_synergy_group": "",
        "allowed_for_static_audit": False,
        "leakage_status": "blocker",
        "notes": "Strong but invalid future-only feature.",
    },
    {
        "feature_id": "catalog_object_id_known",
        "feature_group": "blocked_or_conditional_features",
        "description": "identity or catalog lookup for space object",
        "value_type": "binary",
        "observable_from_3d_track": False,
        "online_available": False,
        "uses_future_window": False,
        "uses_generator_metadata": False,
        "uses_identity_or_catalog_lookup": True,
        "label_rule_overlap": False,
        "external_context_required": True,
        "redundancy_group": "",
        "candidate_synergy_group": "",
        "allowed_for_static_audit": False,
        "leakage_status": "blocker",
        "notes": "Identity lookup, not class evidence from the track itself.",
    },
    {
        "feature_id": "acoustic_signature_score",
        "feature_group": "blocked_or_conditional_features",
        "description": "acoustic cue for subsurface contacts",
        "value_type": "normalized_scalar",
        "observable_from_3d_track": False,
        "online_available": False,
        "uses_future_window": False,
        "uses_generator_metadata": False,
        "uses_identity_or_catalog_lookup": False,
        "label_rule_overlap": False,
        "external_context_required": True,
        "redundancy_group": "",
        "candidate_synergy_group": "",
        "allowed_for_static_audit": False,
        "leakage_status": "conditional",
        "notes": "Unsupported in a kinematic-only 3D track bundle.",
    },
    {
        "feature_id": "iff_declared_type",
        "feature_group": "blocked_or_conditional_features",
        "description": "identity cue for cooperative platforms",
        "value_type": "binary",
        "observable_from_3d_track": False,
        "online_available": False,
        "uses_future_window": False,
        "uses_generator_metadata": False,
        "uses_identity_or_catalog_lookup": True,
        "label_rule_overlap": False,
        "external_context_required": True,
        "redundancy_group": "",
        "candidate_synergy_group": "",
        "allowed_for_static_audit": False,
        "leakage_status": "blocker",
        "notes": "Identity field rather than pure kinematic evidence.",
    },
    {
        "feature_id": "min_altitude_m",
        "feature_group": "observable_3d_kinematics",
        "description": "minimum observed altitude in the window",
        "value_type": "synthetic_scalar",
        "observable_from_3d_track": True,
        "online_available": False,
        "uses_future_window": True,
        "uses_generator_metadata": False,
        "uses_identity_or_catalog_lookup": False,
        "label_rule_overlap": False,
        "external_context_required": False,
        "redundancy_group": "altitude_thresholds",
        "candidate_synergy_group": "",
        "allowed_for_static_audit": True,
        "leakage_status": "conditional",
        "notes": "Synthetic altitude-like quantity for threshold alias and observability checks.",
    },
    {
        "feature_id": "min_altitude_m_offset_1m",
        "feature_group": "observable_3d_kinematics",
        "description": "minimum observed altitude offset by one meter",
        "value_type": "synthetic_scalar",
        "observable_from_3d_track": True,
        "online_available": False,
        "uses_future_window": True,
        "uses_generator_metadata": False,
        "uses_identity_or_catalog_lookup": False,
        "label_rule_overlap": False,
        "external_context_required": False,
        "redundancy_group": "altitude_thresholds",
        "candidate_synergy_group": "",
        "allowed_for_static_audit": True,
        "leakage_status": "conditional",
        "notes": "Synthetic affine alias used to demonstrate offset-equivalence detection.",
    },
    {
        "feature_id": "min_altitude_ge_300m",
        "feature_group": "observable_3d_kinematics",
        "description": "indicator that minimum altitude is at least 300m",
        "value_type": "binary",
        "observable_from_3d_track": True,
        "online_available": False,
        "uses_future_window": True,
        "uses_generator_metadata": False,
        "uses_identity_or_catalog_lookup": False,
        "label_rule_overlap": False,
        "external_context_required": False,
        "redundancy_group": "altitude_thresholds",
        "candidate_synergy_group": "",
        "allowed_for_static_audit": True,
        "leakage_status": "conditional",
        "notes": "Near-threshold alias candidate for observability-aware redundancy analysis.",
    },
    {
        "feature_id": "min_altitude_ge_301m",
        "feature_group": "observable_3d_kinematics",
        "description": "indicator that minimum altitude is at least 301m",
        "value_type": "binary",
        "observable_from_3d_track": True,
        "online_available": False,
        "uses_future_window": True,
        "uses_generator_metadata": False,
        "uses_identity_or_catalog_lookup": False,
        "label_rule_overlap": False,
        "external_context_required": False,
        "redundancy_group": "altitude_thresholds",
        "candidate_synergy_group": "",
        "allowed_for_static_audit": True,
        "leakage_status": "conditional",
        "notes": "Paired with the 300m threshold to test threshold subsumption and observability floors.",
    },
)


PRIOR_REGIME_ROWS: tuple[dict[str, object], ...] = tuple(
    {
        "prior_regime": regime,
        "class_id": class_id,
        "prior_probability": probability,
        "regime_description": description,
        "expected_pathology": pathology,
    }
    for regime, description, pathology, probabilities in (
        (
            "uniform_multidomain",
            "neutral class-pair analysis baseline",
            "none expected beyond declared feature/class limitations",
            {
                "ground_wheeled_vehicle": 1.0 / 14.0,
                "ground_tracked_vehicle": 1.0 / 14.0,
                "surface_maritime_vessel": 1.0 / 14.0,
                "fast_surface_craft": 1.0 / 14.0,
                "subsurface_contact": 1.0 / 14.0,
                "rotary_wing_aircraft": 1.0 / 14.0,
                "fixed_wing_aircraft": 1.0 / 14.0,
                "multirotor_uas": 1.0 / 14.0,
                "fixed_wing_uas": 1.0 / 14.0,
                "balloon_aerostat": 1.0 / 14.0,
                "ballistic_arc_object": 1.0 / 14.0,
                "boost_coast_object": 1.0 / 14.0,
                "leo_orbital_object": 1.0 / 14.0,
                "geo_stationkeeping_object": 1.0 / 14.0,
            },
        ),
        (
            "land_c2_skewed",
            "terrestrial command-and-control emphasis",
            "rare-class invisibility for maritime, subsurface, and space families",
            {
                "ground_wheeled_vehicle": 0.280,
                "ground_tracked_vehicle": 0.150,
                "surface_maritime_vessel": 0.020,
                "fast_surface_craft": 0.010,
                "subsurface_contact": 0.005,
                "rotary_wing_aircraft": 0.100,
                "fixed_wing_aircraft": 0.050,
                "multirotor_uas": 0.130,
                "fixed_wing_uas": 0.070,
                "balloon_aerostat": 0.025,
                "ballistic_arc_object": 0.080,
                "boost_coast_object": 0.060,
                "leo_orbital_object": 0.015,
                "geo_stationkeeping_object": 0.005,
            },
        ),
        (
            "space_surveillance_skewed",
            "space surveillance emphasis",
            "orbital classes dominate weak terrestrial evidence",
            {
                "ground_wheeled_vehicle": 0.005,
                "ground_tracked_vehicle": 0.005,
                "surface_maritime_vessel": 0.005,
                "fast_surface_craft": 0.005,
                "subsurface_contact": 0.001,
                "rotary_wing_aircraft": 0.005,
                "fixed_wing_aircraft": 0.020,
                "multirotor_uas": 0.005,
                "fixed_wing_uas": 0.005,
                "balloon_aerostat": 0.030,
                "ballistic_arc_object": 0.060,
                "boost_coast_object": 0.080,
                "leo_orbital_object": 0.570,
                "geo_stationkeeping_object": 0.204,
            },
        ),
        (
            "maritime_littoral_skewed",
            "littoral maritime emphasis",
            "surface classes dominate while subsurface and low-altitude air remain observability-sensitive",
            {
                "ground_wheeled_vehicle": 0.030,
                "ground_tracked_vehicle": 0.010,
                "surface_maritime_vessel": 0.400,
                "fast_surface_craft": 0.200,
                "subsurface_contact": 0.080,
                "rotary_wing_aircraft": 0.080,
                "fixed_wing_aircraft": 0.040,
                "multirotor_uas": 0.040,
                "fixed_wing_uas": 0.040,
                "balloon_aerostat": 0.010,
                "ballistic_arc_object": 0.030,
                "boost_coast_object": 0.020,
                "leo_orbital_object": 0.010,
                "geo_stationkeeping_object": 0.010,
            },
        ),
    )
    for class_id, probability in probabilities.items()
)


EXPECTED_CONFUSION_ROWS: tuple[dict[str, str], ...] = tuple(
    {
        "class_a": class_a,
        "class_b": class_b,
        "why_confusable": why,
    }
    for class_a, class_b, why in (
        ("ground_wheeled_vehicle", "ground_tracked_vehicle", "context-poor ground mobility overlap"),
        ("surface_maritime_vessel", "fast_surface_craft", "surface speed and water-context family overlap"),
        ("rotary_wing_aircraft", "multirotor_uas", "scale unavailable under pure kinematics"),
        ("fixed_wing_aircraft", "fixed_wing_uas", "size and sensor context missing"),
        ("ballistic_arc_object", "boost_coast_object", "both need time-history regime evidence"),
        ("leo_orbital_object", "geo_stationkeeping_object", "orbital context and arc length govern separation"),
        ("surface_maritime_vessel", "subsurface_contact", "unsupported without depth or acoustic features"),
    )
)


EXPECTED_SYNERGY_ROWS: tuple[dict[str, str], ...] = tuple(
    {
        "feature_a": feature_a,
        "feature_b": feature_b,
        "why_candidate": why,
    }
    for feature_a, feature_b, why in (
        ("turn_rate_abs_mean", "path_efficiency_3d", "loitering versus directed travel"),
        ("climb_angle_abs_mean", "altitude_mean", "ground or surface versus air or space separation"),
        ("sustained_accel_segment_score", "ballistic_arc_fit_rmse", "boost-coast versus ballistic arc distinction"),
        ("water_mask_adherence", "altitude_mean", "surface vessel versus low-altitude aircraft separation"),
        ("orbital_fit_residual", "altitude_mean", "orbital versus high-altitude air distinction"),
        ("stop_go_fraction", "ground_plane_consistency", "ground mobility versus hover-like behavior"),
    )
)


BLOCKED_FEATURE_ROWS: tuple[dict[str, str], ...] = tuple(
    {
        "feature_id": row["feature_id"],
        "why_blocked": str(row["notes"]),
    }
    for row in FEATURE_ROWS
    if str(row["leakage_status"]) in {"blocker", "conditional"} and not bool(row["allowed_for_static_audit"])
)


OBSERVABILITY_GAP_ROWS: tuple[dict[str, str], ...] = (
    {
        "class_id": "subsurface_contact",
        "missing_feature_group": "blocked_or_conditional_features",
        "reason": "Pure 3D kinematic tracks do not expose depth or acoustic signatures.",
    },
    {
        "class_id": "geo_stationkeeping_object",
        "missing_feature_group": "model_fit_regime_consistency",
        "reason": "Short arcs weaken stationkeeping and orbital consistency evidence.",
    },
    {
        "class_id": "fixed_wing_uas",
        "missing_feature_group": "sensor_quality_provenance",
        "reason": "Size or RCS-like cues are unavailable in pure kinematic bundles.",
    },
)


EXCITATION_FEATURES: tuple[str, ...] = (
    "speed_3d_mean",
    "altitude_mean",
    "vertical_speed_abs_p95",
    "accel_norm_p95",
    "turn_rate_abs_mean",
    "hover_loiter_score",
    "ground_plane_consistency",
    "water_mask_adherence",
    "ballistic_arc_fit_rmse",
    "sustained_accel_segment_score",
    "orbital_fit_residual",
    "stationkeeping_consistency",
)

FEATURE_SCHEMA_DEFAULTS: dict[str, dict[str, object]] = {
    "default": {
        "base_quantity": "",
        "aggregation": "",
        "unit": "normalized",
        "operator": "",
        "threshold_value": "",
        "threshold_unit": "",
        "time_scope": "window_summary",
        "measurement_resolution": "",
        "expected_uncertainty": "",
        "derived_from": "",
        "formula_signature": "",
        "provenance_source": "track_derived",
        "semantic_group": "",
    },
    "speed_3d_mean": {"base_quantity": "speed", "aggregation": "mean", "unit": "normalized", "semantic_group": "speed"},
    "speed_3d_median": {"base_quantity": "speed", "aggregation": "median", "unit": "normalized", "semantic_group": "speed"},
    "ground_speed_mean": {"base_quantity": "speed", "aggregation": "mean", "unit": "normalized", "semantic_group": "speed"},
    "horizontal_speed_mean": {"base_quantity": "speed", "aggregation": "mean", "unit": "normalized", "semantic_group": "speed"},
    "altitude_mean": {"base_quantity": "altitude", "aggregation": "mean", "unit": "normalized", "semantic_group": "altitude", "measurement_resolution": "0.05", "expected_uncertainty": "0.08"},
    "altitude_span": {"base_quantity": "altitude", "aggregation": "span", "unit": "normalized", "semantic_group": "altitude"},
    "vertical_speed_abs_p95": {"base_quantity": "vertical_speed", "aggregation": "p95_abs", "unit": "normalized", "semantic_group": "vertical_motion"},
    "climb_angle_abs_mean": {"base_quantity": "climb_angle", "aggregation": "mean_abs", "unit": "normalized", "semantic_group": "vertical_motion"},
    "turn_rate_abs_mean": {"base_quantity": "turn_rate", "aggregation": "mean_abs", "unit": "normalized", "semantic_group": "turn"},
    "curvature_mean": {"base_quantity": "curvature", "aggregation": "mean", "unit": "normalized", "semantic_group": "turn"},
    "path_efficiency_3d": {"base_quantity": "path_efficiency", "aggregation": "window_ratio", "unit": "normalized", "semantic_group": "path_efficiency"},
    "hover_loiter_score": {"base_quantity": "loiter", "aggregation": "window_score", "unit": "normalized", "semantic_group": "loiter"},
    "min_altitude_m": {
        "base_quantity": "altitude",
        "aggregation": "min",
        "unit": "m",
        "time_scope": "full_window",
        "measurement_resolution": "5",
        "expected_uncertainty": "8",
        "formula_signature": "min(altitude_m)",
        "semantic_group": "altitude",
    },
    "min_altitude_m_offset_1m": {
        "base_quantity": "altitude",
        "aggregation": "min",
        "unit": "m",
        "time_scope": "full_window",
        "measurement_resolution": "5",
        "expected_uncertainty": "8",
        "derived_from": "min_altitude_m",
        "formula_signature": "min_altitude_m + 1",
        "semantic_group": "altitude",
    },
    "min_altitude_ge_300m": {
        "base_quantity": "altitude",
        "aggregation": "min",
        "unit": "m",
        "operator": ">=",
        "threshold_value": "300",
        "threshold_unit": "m",
        "time_scope": "full_window",
        "measurement_resolution": "5",
        "expected_uncertainty": "8",
        "derived_from": "min_altitude_m",
        "formula_signature": "threshold(min_altitude_m >= 300)",
        "semantic_group": "altitude",
    },
    "min_altitude_ge_301m": {
        "base_quantity": "altitude",
        "aggregation": "min",
        "unit": "m",
        "operator": ">=",
        "threshold_value": "301",
        "threshold_unit": "m",
        "time_scope": "full_window",
        "measurement_resolution": "5",
        "expected_uncertainty": "8",
        "derived_from": "min_altitude_m",
        "formula_signature": "threshold(min_altitude_m >= 301)",
        "semantic_group": "altitude",
    },
}


LEVEL_MAP = {"low": 0.20, "low_med": 0.35, "med": 0.50, "med_high": 0.70, "high": 0.85, "very_high": 0.98}


SIGNATURE_LEVELS: dict[str, dict[str, str]] = {
    "ground_wheeled_vehicle": {
        "speed_3d_mean": "med",
        "altitude_mean": "low",
        "vertical_speed_abs_p95": "low",
        "accel_norm_p95": "low_med",
        "turn_rate_abs_mean": "med",
        "hover_loiter_score": "low",
        "ground_plane_consistency": "high",
        "water_mask_adherence": "low",
        "ballistic_arc_fit_rmse": "high",
        "sustained_accel_segment_score": "low",
        "orbital_fit_residual": "high",
        "stationkeeping_consistency": "low",
    },
    "ground_tracked_vehicle": {
        "speed_3d_mean": "low_med",
        "altitude_mean": "low",
        "vertical_speed_abs_p95": "low",
        "accel_norm_p95": "med",
        "turn_rate_abs_mean": "med",
        "hover_loiter_score": "med",
        "ground_plane_consistency": "high",
        "water_mask_adherence": "low",
        "ballistic_arc_fit_rmse": "high",
        "sustained_accel_segment_score": "low",
        "orbital_fit_residual": "high",
        "stationkeeping_consistency": "low",
    },
    "surface_maritime_vessel": {
        "speed_3d_mean": "low_med",
        "altitude_mean": "low",
        "vertical_speed_abs_p95": "low",
        "accel_norm_p95": "low",
        "turn_rate_abs_mean": "low_med",
        "hover_loiter_score": "low",
        "ground_plane_consistency": "low",
        "water_mask_adherence": "high",
        "ballistic_arc_fit_rmse": "high",
        "sustained_accel_segment_score": "low",
        "orbital_fit_residual": "high",
        "stationkeeping_consistency": "low",
    },
    "fast_surface_craft": {
        "speed_3d_mean": "med_high",
        "altitude_mean": "low",
        "vertical_speed_abs_p95": "low",
        "accel_norm_p95": "med",
        "turn_rate_abs_mean": "med_high",
        "hover_loiter_score": "low",
        "ground_plane_consistency": "low",
        "water_mask_adherence": "high",
        "ballistic_arc_fit_rmse": "high",
        "sustained_accel_segment_score": "low",
        "orbital_fit_residual": "high",
        "stationkeeping_consistency": "low",
    },
    "subsurface_contact": {
        "speed_3d_mean": "low_med",
        "altitude_mean": "low",
        "vertical_speed_abs_p95": "low",
        "accel_norm_p95": "low",
        "turn_rate_abs_mean": "low",
        "hover_loiter_score": "low",
        "ground_plane_consistency": "low",
        "water_mask_adherence": "low",
        "ballistic_arc_fit_rmse": "high",
        "sustained_accel_segment_score": "low",
        "orbital_fit_residual": "high",
        "stationkeeping_consistency": "low",
    },
    "rotary_wing_aircraft": {
        "speed_3d_mean": "med",
        "altitude_mean": "med",
        "vertical_speed_abs_p95": "med_high",
        "accel_norm_p95": "med",
        "turn_rate_abs_mean": "high",
        "hover_loiter_score": "high",
        "ground_plane_consistency": "low",
        "water_mask_adherence": "low",
        "ballistic_arc_fit_rmse": "high",
        "sustained_accel_segment_score": "low",
        "orbital_fit_residual": "high",
        "stationkeeping_consistency": "low",
    },
    "fixed_wing_aircraft": {
        "speed_3d_mean": "high",
        "altitude_mean": "med_high",
        "vertical_speed_abs_p95": "med",
        "accel_norm_p95": "low_med",
        "turn_rate_abs_mean": "low_med",
        "hover_loiter_score": "low",
        "ground_plane_consistency": "low",
        "water_mask_adherence": "low",
        "ballistic_arc_fit_rmse": "high",
        "sustained_accel_segment_score": "low",
        "orbital_fit_residual": "high",
        "stationkeeping_consistency": "low",
    },
    "multirotor_uas": {
        "speed_3d_mean": "low",
        "altitude_mean": "med",
        "vertical_speed_abs_p95": "med",
        "accel_norm_p95": "med_high",
        "turn_rate_abs_mean": "high",
        "hover_loiter_score": "high",
        "ground_plane_consistency": "low",
        "water_mask_adherence": "low",
        "ballistic_arc_fit_rmse": "high",
        "sustained_accel_segment_score": "low",
        "orbital_fit_residual": "high",
        "stationkeeping_consistency": "low",
    },
    "fixed_wing_uas": {
        "speed_3d_mean": "med",
        "altitude_mean": "med",
        "vertical_speed_abs_p95": "low_med",
        "accel_norm_p95": "med",
        "turn_rate_abs_mean": "med",
        "hover_loiter_score": "med",
        "ground_plane_consistency": "low",
        "water_mask_adherence": "low",
        "ballistic_arc_fit_rmse": "high",
        "sustained_accel_segment_score": "low",
        "orbital_fit_residual": "high",
        "stationkeeping_consistency": "low",
    },
    "balloon_aerostat": {
        "speed_3d_mean": "low",
        "altitude_mean": "med_high",
        "vertical_speed_abs_p95": "low",
        "accel_norm_p95": "low",
        "turn_rate_abs_mean": "low",
        "hover_loiter_score": "low_med",
        "ground_plane_consistency": "low",
        "water_mask_adherence": "low",
        "ballistic_arc_fit_rmse": "high",
        "sustained_accel_segment_score": "low",
        "orbital_fit_residual": "med_high",
        "stationkeeping_consistency": "med",
    },
    "ballistic_arc_object": {
        "speed_3d_mean": "high",
        "altitude_mean": "high",
        "vertical_speed_abs_p95": "high",
        "accel_norm_p95": "high",
        "turn_rate_abs_mean": "low",
        "hover_loiter_score": "low",
        "ground_plane_consistency": "low",
        "water_mask_adherence": "low",
        "ballistic_arc_fit_rmse": "low",
        "sustained_accel_segment_score": "med",
        "orbital_fit_residual": "high",
        "stationkeeping_consistency": "low",
    },
    "boost_coast_object": {
        "speed_3d_mean": "high",
        "altitude_mean": "high",
        "vertical_speed_abs_p95": "high",
        "accel_norm_p95": "very_high",
        "turn_rate_abs_mean": "low_med",
        "hover_loiter_score": "low",
        "ground_plane_consistency": "low",
        "water_mask_adherence": "low",
        "ballistic_arc_fit_rmse": "med_high",
        "sustained_accel_segment_score": "high",
        "orbital_fit_residual": "high",
        "stationkeeping_consistency": "low",
    },
    "leo_orbital_object": {
        "speed_3d_mean": "very_high",
        "altitude_mean": "very_high",
        "vertical_speed_abs_p95": "med",
        "accel_norm_p95": "low",
        "turn_rate_abs_mean": "low",
        "hover_loiter_score": "low",
        "ground_plane_consistency": "low",
        "water_mask_adherence": "low",
        "ballistic_arc_fit_rmse": "high",
        "sustained_accel_segment_score": "low",
        "orbital_fit_residual": "low",
        "stationkeeping_consistency": "low_med",
    },
    "geo_stationkeeping_object": {
        "speed_3d_mean": "low",
        "altitude_mean": "very_high",
        "vertical_speed_abs_p95": "low",
        "accel_norm_p95": "low",
        "turn_rate_abs_mean": "low",
        "hover_loiter_score": "low",
        "ground_plane_consistency": "low",
        "water_mask_adherence": "low",
        "ballistic_arc_fit_rmse": "high",
        "sustained_accel_segment_score": "low",
        "orbital_fit_residual": "low_med",
        "stationkeeping_consistency": "high",
    },
}


BUNDLE_SPECS: tuple[dict[str, object], ...] = (
    {
        "bundle_id": "clean_multidomain_3d_bundle",
        "expected_route": "promote_to_corpus_explorer",
        "classes": (
            "ground_wheeled_vehicle",
            "surface_maritime_vessel",
            "fixed_wing_aircraft",
            "leo_orbital_object",
        ),
        "features": (
            "speed_3d_mean",
            "altitude_mean",
            "vertical_speed_abs_p95",
            "ground_plane_consistency",
            "road_network_adherence",
            "water_mask_adherence",
            "orbital_fit_residual",
            "measurement_noise_estimate",
            "observation_gap_rate",
        ),
        "prior_regime": "uniform_multidomain",
        "description": "Clean notional 3D bundle with broad domain separation and no blocked features.",
    },
    {
        "bundle_id": "prior_pathology_multidomain_3d_bundle",
        "expected_route": "revise_prior",
        "classes": (
            "balloon_aerostat",
            "geo_stationkeeping_object",
        ),
        "features": (
            "altitude_mean",
            "orbital_fit_residual",
            "stationkeeping_consistency",
        ),
        "prior_regime": "space_surveillance_skewed",
        "description": "Skewed orbital priors overwhelm weak terrestrial evidence.",
    },
    {
        "bundle_id": "redundancy_synergy_multidomain_3d_bundle",
        "expected_route": "promote_to_corpus_explorer",
        "classes": (
            "ground_wheeled_vehicle",
            "fixed_wing_aircraft",
            "multirotor_uas",
        ),
        "features": (
            "speed_3d_mean",
            "speed_3d_median",
            "ground_speed_mean",
            "horizontal_speed_mean",
            "altitude_mean",
            "climb_angle_abs_mean",
            "turn_rate_abs_mean",
            "curvature_mean",
            "hover_loiter_score",
            "path_efficiency_3d",
            "min_altitude_m",
            "min_altitude_m_offset_1m",
            "min_altitude_ge_300m",
            "min_altitude_ge_301m",
        ),
        "prior_regime": "uniform_multidomain",
        "description": "Deliberate redundant speed family plus candidate interaction evidence for loitering classes.",
    },
    {
        "bundle_id": "unobservable_navy_space_bundle",
        "expected_route": "revise_class_set",
        "classes": (
            "surface_maritime_vessel",
            "subsurface_contact",
            "balloon_aerostat",
            "geo_stationkeeping_object",
        ),
        "features": (
            "speed_3d_mean",
            "altitude_mean",
            "turn_rate_abs_mean",
            "measurement_noise_estimate",
        ),
        "prior_regime": "maritime_littoral_skewed",
        "description": "Subsurface and short-arc space classes remain unsupported under kinematic-only features.",
    },
    {
        "bundle_id": "leakage_blocker_multidomain_3d_bundle",
        "expected_route": "reject",
        "classes": (
            "surface_maritime_vessel",
            "fixed_wing_aircraft",
            "leo_orbital_object",
        ),
        "features": (
            "speed_3d_mean",
            "altitude_mean",
            "true_platform_label_code",
            "future_endpoint_displacement",
            "catalog_object_id_known",
        ),
        "prior_regime": "uniform_multidomain",
        "description": "Tempting identity and future-window features should hard block promotion.",
    },
)


def write_multidomain_3d_static_admissibility_packet(
    output_dir: str | Path,
) -> MultiDomain3dPacket:
    packet_dir = Path(output_dir)
    figures_dir = packet_dir / "figures"
    source_artifacts_dir = packet_dir / "source_artifacts"
    source_bundles_dir = packet_dir / "source_bundles"
    brief_dir = packet_dir / "brief"
    latex_dir = packet_dir / "latex"
    source_runs_dir = packet_dir / "source_runs"
    for path in (packet_dir, figures_dir, source_artifacts_dir, source_bundles_dir, brief_dir, latex_dir, source_runs_dir):
        path.mkdir(parents=True, exist_ok=True)

    class_schema_path = source_artifacts_dir / "multi_domain_3d_class_schema.csv"
    feature_schema_path = source_artifacts_dir / "multi_domain_3d_feature_schema.csv"
    prior_regimes_path = source_artifacts_dir / "multi_domain_3d_prior_regimes.csv"
    signature_path = source_artifacts_dir / "multi_domain_3d_class_feature_signature.csv"
    expected_confusions_path = source_artifacts_dir / "multi_domain_3d_expected_confusions.csv"
    expected_synergy_path = source_artifacts_dir / "multi_domain_3d_expected_synergy_pairs.csv"
    blocked_features_path = source_artifacts_dir / "multi_domain_3d_blocked_features.csv"
    observability_gaps_path = source_artifacts_dir / "multi_domain_3d_observability_gaps.csv"
    synthetic_samples_path = source_artifacts_dir / "multi_domain_3d_synthetic_samples.csv"
    estimator_reliability_report_path = packet_dir / "estimator_reliability_report.md"
    metric_uncertainty_path = source_artifacts_dir / "static_metric_uncertainty.csv"
    error_bound_proxy_path = source_artifacts_dir / "pairwise_error_bound_proxy.csv"
    prior_evidence_budget_path = source_artifacts_dir / "prior_evidence_budget.csv"
    sample_size_adequacy_path = source_artifacts_dir / "sample_size_adequacy_report.csv"
    metric_assumption_registry_path = source_artifacts_dir / "metric_assumption_registry.csv"
    bound_validity_manifest_path = source_artifacts_dir / "bound_validity_manifest.yaml"
    bootstrap_metric_distributions_path = source_artifacts_dir / "bootstrap_metric_distributions.parquet"
    permutation_null_summary_path = source_artifacts_dir / "permutation_null_summary.csv"
    feature_alias_report_path = packet_dir / "feature_alias_and_redundancy_report.md"
    feature_alias_candidates_path = source_artifacts_dir / "feature_alias_candidates.csv"
    feature_threshold_subsumption_path = source_artifacts_dir / "feature_threshold_subsumption.csv"
    feature_functional_equivalence_path = source_artifacts_dir / "feature_functional_equivalence.csv"
    feature_decision_redundancy_path = source_artifacts_dir / "feature_decision_redundancy.csv"
    feature_redundancy_clusters_path = source_artifacts_dir / "feature_redundancy_clusters.csv"

    write_csv(class_schema_path, list(CLASS_ROWS), list(CLASS_ROWS[0].keys()))
    enriched_feature_rows = _enriched_feature_rows()
    write_csv(feature_schema_path, enriched_feature_rows, list(enriched_feature_rows[0].keys()))
    write_csv(prior_regimes_path, list(PRIOR_REGIME_ROWS), list(PRIOR_REGIME_ROWS[0].keys()))
    write_csv(signature_path, _signature_rows(), list(_signature_rows()[0].keys()))
    write_csv(expected_confusions_path, list(EXPECTED_CONFUSION_ROWS), list(EXPECTED_CONFUSION_ROWS[0].keys()))
    write_csv(expected_synergy_path, list(EXPECTED_SYNERGY_ROWS), list(EXPECTED_SYNERGY_ROWS[0].keys()))
    write_csv(blocked_features_path, list(BLOCKED_FEATURE_ROWS), list(BLOCKED_FEATURE_ROWS[0].keys()))
    write_csv(observability_gaps_path, list(OBSERVABILITY_GAP_ROWS), list(OBSERVABILITY_GAP_ROWS[0].keys()))

    suite_rows: list[dict[str, object]] = []
    bundle_rows: list[dict[str, object]] = []
    bundle_contexts: list[dict[str, object]] = []
    validation_lines = ["# Epic 1 3D Multi-Domain Static Admissibility Brief Validation", ""]
    combined_sample_rows: list[dict[str, object]] = []
    prior_regime_lookup = _prior_regime_lookup()
    for bundle_spec in BUNDLE_SPECS:
        bundle_id = str(bundle_spec["bundle_id"])
        bundle_dir = source_bundles_dir / bundle_id
        bundle_dir.mkdir(parents=True, exist_ok=True)
        feature_order = {name: index for index, name in enumerate(bundle_spec["features"])}
        class_order = {name: index for index, name in enumerate(bundle_spec["classes"])}
        bundle_feature_rows = sorted(
            [row for row in FEATURE_ROWS if row["feature_id"] in bundle_spec["features"]],
            key=lambda row: feature_order[str(row["feature_id"])],
        )
        bundle_class_rows = sorted(
            [row for row in CLASS_ROWS if row["class_id"] in bundle_spec["classes"]],
            key=lambda row: class_order[str(row["class_id"])],
        )
        bundle_samples = _bundle_samples(bundle_id, tuple(str(name) for name in bundle_spec["classes"]), tuple(str(name) for name in bundle_spec["features"]))
        combined_sample_rows.extend(
            {
                "bundle_id": bundle_id,
                "sample_id": row["sample_id"],
                "true_class": row["true_class"],
                **{feature: row[feature] for feature in bundle_spec["features"]},
            }
            for row in bundle_samples
        )
        write_csv(
            bundle_dir / "feature_schema.csv",
            _bundle_feature_schema_rows(bundle_feature_rows),
            [
                "feature_name",
                "provenance_tags",
                "online_available",
                "label_rule_overlap",
                "base_quantity",
                "aggregation",
                "unit",
                "operator",
                "threshold_value",
                "threshold_unit",
                "time_scope",
                "measurement_resolution",
                "expected_uncertainty",
                "derived_from",
                "formula_signature",
                "provenance_source",
                "redundancy_group",
                "semantic_group",
                "allowed_for_static_audit",
            ],
        )
        write_csv(
            bundle_dir / "class_schema.csv",
            _bundle_class_schema_rows(bundle_class_rows),
            ("class_name", "domain", "class_description", "decisionability_notes"),
        )
        write_csv(bundle_dir / "samples.csv", bundle_samples, list(bundle_samples[0].keys()))
        bundle_yaml = {
            "study_dimension": {
                "declared_dimension": 3,
                "source_type": "normalized_feature_bundle",
                "raw_tracklets_available": False,
                "feature_values_operational_thresholds": False,
                "static_audit_only": True,
            },
            "static_admissibility": {
                "study_id": bundle_id,
                "priors": {
                    class_id: prior_regime_lookup[str(bundle_spec["prior_regime"])][class_id]
                    for class_id in bundle_spec["classes"]
                },
                "input_bundle": {
                    "sample_table": "samples.csv",
                    "feature_schema": "feature_schema.csv",
                    "class_schema": "class_schema.csv",
                    "feature_names": list(bundle_spec["features"]),
                },
            }
        }
        bundle_config_path = bundle_dir / "static_audit_bundle.yaml"
        bundle_config_path.write_text(yaml.safe_dump(bundle_yaml, sort_keys=False), encoding="utf-8")
        config = load_static_admissibility_config(bundle_config_path)
        result = build_static_admissibility_result(config)
        source_packet = write_static_admissibility_packet(
            source_runs_dir / bundle_id,
            config=config,
            result=result,
        )
        issues = validate_static_admissibility_packet(source_packet.packet_dir, repo_root=repo_root())
        actual_route = str(result.static_decision["status"])
        expected_route = str(bundle_spec["expected_route"])
        warnings = tuple(str(item) for item in result.static_decision.get("warnings", ()))
        validator_status = "pass" if not issues and expected_route == actual_route else "fail"
        bundle_contexts.append(
            {
                "bundle_id": bundle_id,
                "bundle_spec": bundle_spec,
                "result": result,
                "samples": bundle_samples,
                "features": tuple(str(name) for name in bundle_spec["features"]),
                "classes": tuple(str(name) for name in bundle_spec["classes"]),
                "prior_regime": str(bundle_spec["prior_regime"]),
            }
        )
        validation_lines.append(
            f"- `{bundle_id}`: expected `{expected_route}`, actual `{actual_route}`, validator `{validator_status}`"
        )
        for issue in issues:
            validation_lines.append(f"  - issue: {issue}")
        bundle_rows.append(
            {
                "bundle_id": bundle_id,
                "description": str(bundle_spec["description"]),
                "expected_route": expected_route,
                "actual_route": actual_route,
                "validator_status": validator_status,
                "warnings": "|".join(warnings),
                "source_bundle": str(bundle_dir.relative_to(packet_dir)),
                "source_run": str(source_packet.packet_dir.relative_to(packet_dir)),
            }
        )
        suite_rows.append(
            {
                "bundle_id": bundle_id,
                "class_separability_status": _lane_status(result, "class separability"),
                "feature_relevance_status": _lane_status(result, "feature relevance"),
                "redundancy_status": _lane_status(result, "feature redundancy"),
                "synergy_status": _lane_status(result, "feature synergy"),
                "prior_pathology_status": _lane_status(result, "prior pathology"),
                "coverage_status": _lane_status(result, "coverage feasibility"),
                "leakage_status": _lane_status(result, "leakage risk"),
                "expected_route": expected_route,
                "actual_route": actual_route,
                "validator_result": validator_status,
            }
        )

    write_csv(synthetic_samples_path, combined_sample_rows, list(combined_sample_rows[0].keys()))
    bundle_manifest_path = source_artifacts_dir / "multidomain_bundle_route_matrix.csv"
    route_matrix_path = source_artifacts_dir / "multidomain_bundle_diagnostics.csv"
    write_csv(route_matrix_path, suite_rows, list(suite_rows[0].keys()))
    metric_uncertainty_rows = _build_metric_uncertainty_rows(bundle_contexts)
    error_bound_rows = _build_error_bound_proxy_rows(bundle_contexts)
    evidence_budget_rows = _build_prior_evidence_budget_rows(bundle_contexts)
    sample_size_rows = _build_sample_size_adequacy_rows(bundle_contexts)
    assumption_rows = _metric_assumption_rows()
    permutation_rows = _build_permutation_null_rows(metric_uncertainty_rows)
    bootstrap_distribution_rows = _build_bootstrap_distribution_rows(metric_uncertainty_rows)
    bootstrap_serialization = _write_bootstrap_distribution_artifact(
        bootstrap_metric_distributions_path,
        bootstrap_distribution_rows,
    )
    (
        feature_alias_rows,
        threshold_subsumption_rows,
        functional_equivalence_rows,
        feature_decision_rows,
        feature_cluster_rows,
    ) = _build_alias_redundancy_rows(bundle_contexts)
    bound_validity_manifest = _build_bound_validity_manifest(bootstrap_serialization)
    decision_confidence = _overall_decision_confidence(sample_size_rows, metric_uncertainty_rows)
    confidence_limiters = _confidence_limiters(sample_size_rows, metric_uncertainty_rows, bundle_rows)
    for row in bundle_rows:
        row["decision_confidence"] = _bundle_decision_confidence(
            str(row["bundle_id"]),
            sample_size_rows,
            metric_uncertainty_rows,
            str(row["actual_route"]),
        )
        row["confidence_limiters"] = "|".join(
            _bundle_confidence_limiters(
                str(row["bundle_id"]),
                sample_size_rows,
                metric_uncertainty_rows,
                str(row["actual_route"]),
            )
        )
    write_csv(bundle_manifest_path, bundle_rows, list(bundle_rows[0].keys()))
    write_csv(metric_uncertainty_path, metric_uncertainty_rows, list(metric_uncertainty_rows[0].keys()))
    write_csv(error_bound_proxy_path, error_bound_rows, list(error_bound_rows[0].keys()))
    write_csv(prior_evidence_budget_path, evidence_budget_rows, list(evidence_budget_rows[0].keys()))
    write_csv(sample_size_adequacy_path, sample_size_rows, list(sample_size_rows[0].keys()))
    write_csv(metric_assumption_registry_path, assumption_rows, list(assumption_rows[0].keys()))
    write_csv(permutation_null_summary_path, permutation_rows, list(permutation_rows[0].keys()))
    write_csv(feature_alias_candidates_path, feature_alias_rows, list(feature_alias_rows[0].keys()))
    write_csv(feature_threshold_subsumption_path, threshold_subsumption_rows, list(threshold_subsumption_rows[0].keys()))
    write_csv(feature_functional_equivalence_path, functional_equivalence_rows, list(functional_equivalence_rows[0].keys()))
    write_csv(feature_decision_redundancy_path, feature_decision_rows, list(feature_decision_rows[0].keys()))
    write_csv(feature_redundancy_clusters_path, feature_cluster_rows, list(feature_cluster_rows[0].keys()))
    bound_validity_manifest_path.write_text(
        yaml.safe_dump(bound_validity_manifest, sort_keys=False),
        encoding="utf-8",
    )
    _write_text(
        estimator_reliability_report_path,
        _render_estimator_reliability_report(
            metric_uncertainty_rows,
            error_bound_rows,
            evidence_budget_rows,
            sample_size_rows,
            assumption_rows,
            decision_confidence,
            confidence_limiters,
        ),
    )
    _write_text(
        feature_alias_report_path,
        _render_feature_alias_report(
            feature_alias_rows,
            threshold_subsumption_rows,
            functional_equivalence_rows,
            feature_decision_rows,
        ),
    )

    _render_md3d_bundle_ingestion(figures_dir / "MD3D_01_bundle_ingestion_spine.png")
    _render_md3d_class_surface(figures_dir / "MD3D_02_class_surface_map.png")
    _render_md3d_prior_regimes(figures_dir / "MD3D_03_prior_regime_matrix.png")
    _render_md3d_feature_taxonomy(figures_dir / "MD3D_04_feature_taxonomy_observability.png")
    _render_md3d_excitation_matrix(figures_dir / "MD3D_05_class_feature_excitation_matrix.png")
    _render_md3d_confusability(figures_dir / "MD3D_06_class_confusability_matrix.png")
    _copy_packet_figure(
        source_runs_dir / "prior_pathology_multidomain_3d_bundle" / "02g_prior_pathology_surface.png",
        figures_dir / "MD3D_07_prior_pathology_surface.png",
    )
    _copy_packet_figure(
        source_runs_dir / "prior_pathology_multidomain_3d_bundle" / "02h_prior_flip_thresholds.png",
        figures_dir / "MD3D_08_prior_flip_thresholds.png",
    )
    _render_md3d_redundancy_synergy(
        figures_dir / "MD3D_09_redundancy_synergy_graph.png",
        redundancy_graph_path=source_runs_dir / "redundancy_synergy_multidomain_3d_bundle" / "02e_feature_redundancy_graph.png",
        synergy_map_path=source_runs_dir / "redundancy_synergy_multidomain_3d_bundle" / "02f_feature_synergy_map.png",
    )
    _render_md3d_unobservable_and_leakage(figures_dir / "MD3D_10_unobservable_and_leakage_audit.png")
    _render_md3d_decision_card(figures_dir / "MD3D_11_static_decision_card.png", bundle_rows)
    _render_md3d_action_router(figures_dir / "MD3D_12_action_router.png")
    _render_md3d_estimator_reliability_dashboard(
        figures_dir / "MD3D_13_estimator_reliability_dashboard.png",
        metric_uncertainty_rows,
        sample_size_rows,
    )
    _render_md3d_error_bound_proxy(
        figures_dir / "MD3D_14_pairwise_error_bound_proxy.png",
        error_bound_rows,
    )
    _render_md3d_prior_evidence_budget(
        figures_dir / "MD3D_15_prior_evidence_budget.png",
        evidence_budget_rows,
    )
    _render_md3d_sample_size_gap(
        figures_dir / "MD3D_16_sample_size_gap_heatmap.png",
        sample_size_rows,
    )
    _render_md3d_assumption_matrix(
        figures_dir / "MD3D_17_bound_assumption_matrix.png",
        assumption_rows,
    )
    _render_md3d_feature_alias_ladder(
        figures_dir / "MD3D_18_feature_alias_ladder.png",
        feature_alias_rows,
    )
    _render_md3d_threshold_subsumption_map(
        figures_dir / "MD3D_19_threshold_subsumption_map.png",
        threshold_subsumption_rows,
    )
    _render_md3d_functional_equivalence_scatter(
        figures_dir / "MD3D_20_functional_equivalence_scatter.png",
        functional_equivalence_rows,
        _combined_dataframe(bundle_contexts),
    )
    _render_md3d_decision_redundancy_matrix(
        figures_dir / "MD3D_21_decision_redundancy_matrix.png",
        feature_decision_rows,
    )

    readme_path = packet_dir / "README.md"
    quickstart_path = packet_dir / "quickstart.md"
    decision_card_path = packet_dir / "decision_card.md"
    validation_report_path = packet_dir / "validation_report.md"
    claim_boundary_path = packet_dir / "claim_boundary.md"
    packet_manifest_path = packet_dir / "packet_manifest.yaml"
    hero_chart_manifest_path = packet_dir / "hero_chart_manifest.csv"
    lane_proof_matrix_path = packet_dir / "lane_proof_matrix.md"
    automated_brief_path = brief_dir / "automated_brief.md"
    latex_path = latex_dir / "multidomain_3d_static_admissibility.tex"

    _write_text(readme_path, _render_md3d_readme())
    _write_text(quickstart_path, _render_md3d_quickstart())
    _write_text(
        decision_card_path,
        _render_md3d_decision_markdown(bundle_rows, decision_confidence, confidence_limiters),
    )
    _write_text(validation_report_path, "\n".join(validation_lines) + "\n")
    _write_text(claim_boundary_path, _render_md3d_claim_boundary())
    _write_text(lane_proof_matrix_path, _render_md3d_lane_proof_matrix())
    _write_text(
        automated_brief_path,
        _render_md3d_automated_brief(bundle_rows, decision_confidence, confidence_limiters),
    )
    _write_text(latex_path, _render_md3d_latex())
    hero_rows = _md3d_hero_rows()
    write_csv(hero_chart_manifest_path, hero_rows, list(hero_rows[0].keys()))
    packet_manifest_path.write_text(
        yaml.safe_dump(
            {
                "packet_id": "01_static_admissibility_multi_domain_3d",
                "claim": "A notional unclassified 3D-inspired feature/class/prior bundle can stress Epic 1 and show admissibility, prior pathology, observability gaps, redundancy, synergy, and leakage before corpus search.",
                "study_dimension": {
                    "declared_dimension": 3,
                    "source_type": "normalized_feature_bundle",
                    "raw_tracklets_available": False,
                    "feature_values_operational_thresholds": False,
                    "static_audit_only": True,
                },
                "bundle_matrix": str(bundle_manifest_path.relative_to(packet_dir)),
                "diagnostic_matrix": str(route_matrix_path.relative_to(packet_dir)),
                "figure_manifest": str(hero_chart_manifest_path.relative_to(packet_dir)),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return MultiDomain3dPacket(
        packet_dir=packet_dir,
        readme_path=readme_path,
        quickstart_path=quickstart_path,
        decision_card_path=decision_card_path,
        validation_report_path=validation_report_path,
        claim_boundary_path=claim_boundary_path,
        packet_manifest_path=packet_manifest_path,
        hero_chart_manifest_path=hero_chart_manifest_path,
        lane_proof_matrix_path=lane_proof_matrix_path,
        automated_brief_path=automated_brief_path,
        latex_path=latex_path,
        estimator_reliability_report_path=estimator_reliability_report_path,
    )


def _signature_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for class_id in SIGNATURE_LEVELS:
        for feature_id in EXCITATION_FEATURES:
            level = SIGNATURE_LEVELS[class_id][feature_id]
            mean_value = LEVEL_MAP[level]
            rows.append(
                {
                    "class_id": class_id,
                    "feature_id": feature_id,
                    "expected_level": level,
                    "expected_mean": round(mean_value, 3),
                    "expected_std": 0.05 if mean_value < 0.9 else 0.03,
                    "source": "synthetic_notional_profile",
                    "notes": "Normalized synthetic values for Epic 1 study-shape declaration.",
                }
            )
    return rows


def _enriched_feature_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    defaults = dict(FEATURE_SCHEMA_DEFAULTS["default"])
    for row in FEATURE_ROWS:
        metadata = dict(defaults)
        metadata.update(FEATURE_SCHEMA_DEFAULTS.get(str(row["feature_id"]), {}))
        rows.append({**row, **metadata})
    return rows


def _bundle_feature_schema_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    bundle_rows: list[dict[str, object]] = []
    metadata_lookup = {str(row["feature_id"]): row for row in _enriched_feature_rows()}
    for row in rows:
        provenance_tags: list[str] = []
        feature_group = str(row["feature_group"])
        leakage_status = str(row["leakage_status"])
        metadata = metadata_lookup[str(row["feature_id"])]
        provenance_tags.append(feature_group)
        if bool(row["observable_from_3d_track"]):
            provenance_tags.append("observable_3d_track")
        if bool(row["external_context_required"]):
            provenance_tags.append("external_context")
        if bool(row["uses_generator_metadata"]):
            provenance_tags.append("generator_metadata")
        if bool(row["uses_identity_or_catalog_lookup"]):
            provenance_tags.append("identity_or_catalog")
        if bool(row["uses_future_window"]):
            provenance_tags.append("future_window")
        provenance_tags.append(leakage_status)
        bundle_rows.append(
            {
                "feature_name": str(row["feature_id"]),
                "provenance_tags": ",".join(provenance_tags),
                "online_available": bool(row["online_available"]),
                "label_rule_overlap": bool(row["label_rule_overlap"]),
                "base_quantity": metadata["base_quantity"],
                "aggregation": metadata["aggregation"],
                "unit": metadata["unit"],
                "operator": metadata["operator"],
                "threshold_value": metadata["threshold_value"],
                "threshold_unit": metadata["threshold_unit"],
                "time_scope": metadata["time_scope"],
                "measurement_resolution": metadata["measurement_resolution"],
                "expected_uncertainty": metadata["expected_uncertainty"],
                "derived_from": metadata["derived_from"],
                "formula_signature": metadata["formula_signature"],
                "provenance_source": metadata["provenance_source"],
                "redundancy_group": row["redundancy_group"],
                "semantic_group": metadata["semantic_group"],
                "allowed_for_static_audit": row["allowed_for_static_audit"],
            }
        )
    return bundle_rows


def _bundle_class_schema_rows(
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    return [
        {
            "class_name": row["class_id"],
            "domain": row["domain"],
            "class_description": row["class_description"],
            "decisionability_notes": row["decisionability_notes"],
        }
        for row in rows
    ]


def _prior_regime_lookup() -> dict[str, dict[str, float]]:
    lookup: dict[str, dict[str, float]] = {}
    for row in PRIOR_REGIME_ROWS:
        lookup.setdefault(str(row["prior_regime"]), {})[str(row["class_id"])] = float(row["prior_probability"])
    return lookup


def _bundle_samples(
    bundle_id: str,
    classes: tuple[str, ...],
    features: tuple[str, ...],
) -> list[dict[str, object]]:
    if bundle_id == "prior_pathology_multidomain_3d_bundle":
        return _prior_pathology_bundle_samples()
    rows: list[dict[str, object]] = []
    for class_index, class_id in enumerate(classes):
        for sample_index in range(4):
            feature_values = {
                feature_id: _base_feature_value(class_id, feature_id, sample_index)
                for feature_id in features
            }
            if bundle_id == "clean_multidomain_3d_bundle":
                if class_id == "ground_wheeled_vehicle":
                    feature_values.update(
                        {
                            "altitude_mean": 0.05 + sample_index * 0.005,
                            "vertical_speed_abs_p95": 0.08 + sample_index * 0.005,
                            "ground_plane_consistency": 0.88 + sample_index * 0.01,
                            "road_network_adherence": 0.92 - sample_index * 0.01,
                            "water_mask_adherence": 0.05 + sample_index * 0.005,
                            "orbital_fit_residual": 0.90 - sample_index * 0.01,
                        }
                    )
                if class_id == "surface_maritime_vessel":
                    feature_values.update(
                        {
                            "altitude_mean": 0.04 + sample_index * 0.004,
                            "vertical_speed_abs_p95": 0.06 + sample_index * 0.004,
                            "ground_plane_consistency": 0.10 + sample_index * 0.01,
                            "road_network_adherence": 0.06 + sample_index * 0.004,
                            "water_mask_adherence": 0.94 - sample_index * 0.01,
                            "orbital_fit_residual": 0.88 - sample_index * 0.01,
                        }
                    )
                if class_id == "fixed_wing_aircraft":
                    feature_values.update(
                        {
                            "speed_3d_mean": 0.82 + sample_index * 0.01,
                            "altitude_mean": 0.66 + sample_index * 0.01,
                            "vertical_speed_abs_p95": 0.45 + sample_index * 0.01,
                            "ground_plane_consistency": 0.08 + sample_index * 0.005,
                            "road_network_adherence": 0.04 + sample_index * 0.004,
                            "water_mask_adherence": 0.04 + sample_index * 0.004,
                            "orbital_fit_residual": 0.82 - sample_index * 0.01,
                        }
                    )
                if class_id == "leo_orbital_object":
                    feature_values.update(
                        {
                            "speed_3d_mean": 0.97 - sample_index * 0.005,
                            "altitude_mean": 0.97 - sample_index * 0.004,
                            "vertical_speed_abs_p95": 0.42 + sample_index * 0.005,
                            "ground_plane_consistency": 0.02 + sample_index * 0.003,
                            "road_network_adherence": 0.01,
                            "water_mask_adherence": 0.01,
                            "orbital_fit_residual": 0.10 + sample_index * 0.01,
                        }
                    )
            if bundle_id == "redundancy_synergy_multidomain_3d_bundle":
                if class_id == "ground_wheeled_vehicle":
                    min_altitude = (299.4, 300.2, 300.6, 301.8)[sample_index]
                    feature_values.update(
                        {
                            "speed_3d_mean": 0.46 + sample_index * 0.01,
                            "speed_3d_median": 0.45 + sample_index * 0.01,
                            "ground_speed_mean": 0.47 + sample_index * 0.01,
                            "horizontal_speed_mean": 0.47 + sample_index * 0.01,
                            "altitude_mean": 0.05 + sample_index * 0.004,
                            "climb_angle_abs_mean": 0.05 + sample_index * 0.004,
                            "turn_rate_abs_mean": 0.44 + sample_index * 0.01,
                            "curvature_mean": 0.43 + sample_index * 0.01,
                            "hover_loiter_score": 0.12 + sample_index * 0.01,
                            "path_efficiency_3d": 0.78 - sample_index * 0.01,
                            "min_altitude_m": min_altitude,
                            "min_altitude_m_offset_1m": min_altitude + 1.0,
                            "min_altitude_ge_300m": 1.0 if min_altitude >= 300.0 else 0.0,
                            "min_altitude_ge_301m": 1.0 if min_altitude >= 301.0 else 0.0,
                        }
                    )
                if class_id == "multirotor_uas":
                    min_altitude = (300.4, 300.7, 301.2, 302.0)[sample_index]
                    feature_values.update(
                        {
                            "speed_3d_mean": 0.22 + sample_index * 0.008,
                            "speed_3d_median": 0.21 + sample_index * 0.008,
                            "ground_speed_mean": 0.21 + sample_index * 0.008,
                            "horizontal_speed_mean": 0.21 + sample_index * 0.008,
                            "altitude_mean": 0.52 + sample_index * 0.008,
                            "climb_angle_abs_mean": 0.56 + sample_index * 0.008,
                            "turn_rate_abs_mean": 0.84 - sample_index * 0.01,
                            "curvature_mean": 0.82 - sample_index * 0.01,
                            "hover_loiter_score": 0.86 - sample_index * 0.01,
                            "path_efficiency_3d": 0.32 + sample_index * 0.01,
                            "min_altitude_m": min_altitude,
                            "min_altitude_m_offset_1m": min_altitude + 1.0,
                            "min_altitude_ge_300m": 1.0 if min_altitude >= 300.0 else 0.0,
                            "min_altitude_ge_301m": 1.0 if min_altitude >= 301.0 else 0.0,
                        }
                    )
                if class_id == "fixed_wing_aircraft":
                    min_altitude = (680.0, 682.0, 684.0, 686.0)[sample_index]
                    feature_values.update(
                        {
                            "speed_3d_mean": 0.86 + sample_index * 0.01,
                            "speed_3d_median": 0.85 + sample_index * 0.01,
                            "ground_speed_mean": 0.85 + sample_index * 0.01,
                            "horizontal_speed_mean": 0.85 + sample_index * 0.01,
                            "altitude_mean": 0.68 + sample_index * 0.008,
                            "climb_angle_abs_mean": 0.26 + sample_index * 0.006,
                            "turn_rate_abs_mean": 0.24 + sample_index * 0.008,
                            "curvature_mean": 0.23 + sample_index * 0.008,
                            "hover_loiter_score": 0.08 + sample_index * 0.008,
                            "path_efficiency_3d": 0.88 - sample_index * 0.008,
                            "min_altitude_m": min_altitude,
                            "min_altitude_m_offset_1m": min_altitude + 1.0,
                            "min_altitude_ge_300m": 1.0,
                            "min_altitude_ge_301m": 1.0,
                        }
                    )
            if bundle_id == "unobservable_navy_space_bundle":
                if class_id == "subsurface_contact":
                    feature_values["altitude_mean"] = 0.06 + sample_index * 0.01
                    feature_values["speed_3d_mean"] = 0.34 + sample_index * 0.01
                if class_id == "surface_maritime_vessel":
                    feature_values["altitude_mean"] = 0.05 + sample_index * 0.01
                    feature_values["speed_3d_mean"] = 0.33 + sample_index * 0.01
                if class_id in {"balloon_aerostat", "geo_stationkeeping_object"}:
                    feature_values["altitude_mean"] = 0.82 + sample_index * 0.01
                    feature_values["speed_3d_mean"] = 0.10 + sample_index * 0.01
            if bundle_id == "leakage_blocker_multidomain_3d_bundle":
                label_code = float(class_index)
                if "true_platform_label_code" in feature_values:
                    feature_values["true_platform_label_code"] = label_code
                if "catalog_object_id_known" in feature_values:
                    feature_values["catalog_object_id_known"] = 1.0 if class_id == "leo_orbital_object" else 0.0
                if "future_endpoint_displacement" in feature_values:
                    feature_values["future_endpoint_displacement"] = 0.2 + label_code * 0.3
            row = {"sample_id": f"{class_id}_{sample_index}", "true_class": class_id, **feature_values}
            rows.append(row)
    return rows


def _prior_pathology_bundle_samples() -> list[dict[str, object]]:
    predefined = {
        "balloon_aerostat": (
            {"altitude_mean": 0.7546564641935591, "orbital_fit_residual": 0.5583566394459469, "stationkeeping_consistency": 0.5532380016485908},
            {"altitude_mean": 0.7354088112275257, "orbital_fit_residual": 0.5832217562533599, "stationkeeping_consistency": 0.4266385780977622},
            {"altitude_mean": 0.9028126392527467, "orbital_fit_residual": 0.49613268642803543, "stationkeeping_consistency": 0.44632850718534556},
            {"altitude_mean": 0.811343908697653, "orbital_fit_residual": 0.66390178840508, "stationkeeping_consistency": 0.4682061405471488},
            {"altitude_mean": 0.8068612026360367, "orbital_fit_residual": 0.4894802019573359, "stationkeeping_consistency": 0.42623061483615143},
            {"altitude_mean": 0.8783036963049953, "orbital_fit_residual": 0.49659228989131887, "stationkeeping_consistency": 0.4631596863240338},
        ),
        "geo_stationkeeping_object": (
            {"altitude_mean": 0.7580685328298472, "orbital_fit_residual": 0.5478820266059911, "stationkeeping_consistency": 0.5151648248903652},
            {"altitude_mean": 0.8631584934960732, "orbital_fit_residual": 0.44595082959411114, "stationkeeping_consistency": 0.5376401024740768},
            {"altitude_mean": 0.9075957630348245, "orbital_fit_residual": 0.4835522535850747, "stationkeeping_consistency": 0.49247026001136807},
            {"altitude_mean": 0.8352503903844246, "orbital_fit_residual": 0.6103178480809919, "stationkeeping_consistency": 0.46651223811015885},
            {"altitude_mean": 0.762263591154444, "orbital_fit_residual": 0.5859839257494707, "stationkeeping_consistency": 0.5971204597074127},
            {"altitude_mean": 0.7240159804049703, "orbital_fit_residual": 0.5134646456747839, "stationkeeping_consistency": 0.47674045722910896},
        ),
    }
    rows: list[dict[str, object]] = []
    for class_id, samples in predefined.items():
        for sample_index, sample in enumerate(samples):
            rows.append(
                {
                    "sample_id": f"{class_id}_{sample_index}",
                    "true_class": class_id,
                    **sample,
                }
            )
    return rows


def _base_feature_value(class_id: str, feature_id: str, sample_index: int) -> float:
    level = SIGNATURE_LEVELS.get(class_id, {}).get(feature_id)
    if level is not None:
        return round(min(max(LEVEL_MAP[level] + (sample_index - 1.5) * 0.01, 0.0), 1.0), 3)
    row = next((item for item in FEATURE_ROWS if item["feature_id"] == feature_id), None)
    if row and row["feature_group"] == "blocked_or_conditional_features":
        return round(0.1 + sample_index * 0.05, 3)
    if row and row["feature_group"] == "sensor_quality_provenance":
        return round(0.15 + sample_index * 0.02, 3)
    return round(0.25 + sample_index * 0.02, 3)


def _lane_status(result, lane: str) -> str:
    row = next(item for item in result.decision_card_rows if str(item["lane"]) == lane)
    status = str(row["status"])
    if status in {"pass", "promote", "promote_to_corpus_explorer"}:
        return "pass"
    if status in {"warning", "warn"}:
        return "warn"
    if status == "candidate":
        return "candidate"
    return "block"


def _confidence_interval_width(n_effective: int, *, scale: float) -> float:
    return round(min(scale / sqrt(max(n_effective, 1)), 0.95), 3)


def _stability_status(width: float, *, target: float) -> str:
    if width <= target:
        return "stable"
    if width <= target * 1.75:
        return "watch"
    return "underpowered"


def _build_metric_uncertainty_rows(bundle_contexts: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for context in bundle_contexts:
        bundle_id = str(context["bundle_id"])
        result = context["result"]
        sample_rows = context["samples"]
        class_counts = Counter(str(row["true_class"]) for row in sample_rows)
        for pair in result.class_pair_rows:
            n_effective = min(class_counts[str(pair["class_a"])], class_counts[str(pair["class_b"])])
            width = _confidence_interval_width(n_effective, scale=0.55)
            point_estimate = float(pair["pairwise_auc"])
            rows.append(
                {
                    "bundle_id": bundle_id,
                    "metric_id": "pairwise_auc",
                    "metric_family": "class_separability",
                    "class_pair": f"{pair['class_a']} vs {pair['class_b']}",
                    "feature_id": "",
                    "point_estimate": round(point_estimate, 3),
                    "ci_low": round(max(0.0, point_estimate - width / 2.0), 3),
                    "ci_high": round(min(1.0, point_estimate + width / 2.0), 3),
                    "ci_method": "sample_size_proxy_interval",
                    "n_effective": n_effective,
                    "uncertainty_width": width,
                    "stability_status": _stability_status(width, target=0.18),
                    "evidence_tier": "proxy",
                }
            )
        for feature in result.feature_relevance_rows:
            n_effective = len(sample_rows)
            width = _confidence_interval_width(n_effective, scale=0.45)
            point_estimate = float(feature["mi_with_class"])
            rows.append(
                {
                    "bundle_id": bundle_id,
                    "metric_id": "mutual_information",
                    "metric_family": "feature_relevance",
                    "class_pair": "",
                    "feature_id": str(feature["feature"]),
                    "point_estimate": round(point_estimate, 3),
                    "ci_low": round(max(0.0, point_estimate - width / 2.0), 3),
                    "ci_high": round(point_estimate + width / 2.0, 3),
                    "ci_method": "sample_size_proxy_interval",
                    "n_effective": n_effective,
                    "uncertainty_width": width,
                    "stability_status": _stability_status(width, target=0.14),
                    "evidence_tier": "sample_sensitive_proxy",
                }
            )
        for pair in result.feature_redundancy_rows:
            n_effective = len(sample_rows)
            width = _confidence_interval_width(n_effective, scale=0.25)
            point_estimate = abs(float(pair["spearman_corr"]))
            rows.append(
                {
                    "bundle_id": bundle_id,
                    "metric_id": "spearman_redundancy",
                    "metric_family": "feature_redundancy",
                    "class_pair": "",
                    "feature_id": f"{pair['feature_a']}::{pair['feature_b']}",
                    "point_estimate": round(point_estimate, 3),
                    "ci_low": round(max(0.0, point_estimate - width / 2.0), 3),
                    "ci_high": round(min(1.0, point_estimate + width / 2.0), 3),
                    "ci_method": "sample_size_proxy_interval",
                    "n_effective": n_effective,
                    "uncertainty_width": width,
                    "stability_status": _stability_status(width, target=0.10),
                    "evidence_tier": "proxy",
                }
            )
        for pair in result.prior_pathology_rows:
            n_effective = min(class_counts[str(pair["class_a"])], class_counts[str(pair["class_b"])])
            width = _confidence_interval_width(n_effective, scale=0.35)
            point_estimate = float(pair["posterior_collapse_rate"])
            rows.append(
                {
                    "bundle_id": bundle_id,
                    "metric_id": "posterior_collapse_rate",
                    "metric_family": "prior_pathology",
                    "class_pair": f"{pair['class_a']} vs {pair['class_b']}",
                    "feature_id": "",
                    "point_estimate": round(point_estimate, 3),
                    "ci_low": round(max(0.0, point_estimate - width / 2.0), 3),
                    "ci_high": round(min(1.0, point_estimate + width / 2.0), 3),
                    "ci_method": "wilson_proxy_interval",
                    "n_effective": n_effective,
                    "uncertainty_width": width,
                    "stability_status": _stability_status(width, target=0.16),
                    "evidence_tier": "proxy",
                }
            )
        for feature in {str(row["feature"]) for row in result.coverage_rows}:
            per_feature_rows = [row for row in result.coverage_rows if str(row["feature"]) == feature]
            n_effective = len(per_feature_rows)
            pass_rate = sum(1.0 for row in per_feature_rows if str(row["status"]) == "pass") / max(n_effective, 1)
            width = _confidence_interval_width(n_effective, scale=0.60)
            rows.append(
                {
                    "bundle_id": bundle_id,
                    "metric_id": "coverage_cell_pass_rate",
                    "metric_family": "coverage_feasibility",
                    "class_pair": "",
                    "feature_id": feature,
                    "point_estimate": round(pass_rate, 3),
                    "ci_low": round(max(0.0, pass_rate - width / 2.0), 3),
                    "ci_high": round(min(1.0, pass_rate + width / 2.0), 3),
                    "ci_method": "wilson_proxy_interval",
                    "n_effective": n_effective,
                    "uncertainty_width": width,
                    "stability_status": _stability_status(width, target=0.20),
                    "evidence_tier": "proxy",
                }
            )
    return rows


def _build_error_bound_proxy_rows(bundle_contexts: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for context in bundle_contexts:
        bundle_id = str(context["bundle_id"])
        prior_regime = str(context["prior_regime"])
        result = context["result"]
        for pair in result.class_pair_rows:
            overlap = float(pair["overlap_coefficient"])
            auc = float(pair["pairwise_auc"])
            js = float(pair["jensen_shannon"])
            mahalanobis = float(pair["mahalanobis_distance"])
            bayes_proxy = round(min(0.5, overlap / 2.0), 3)
            fano_proxy = round(min(0.49, max(0.0, 0.5 - 0.35 * js)), 3)
            bhattacharyya_proxy = round(min(1.0, max(0.0, 0.5 * overlap + 0.15 / max(mahalanobis + 0.2, 1e-6))), 3)
            nn_proxy = round(max(0.0, 1.0 - auc), 3)
            bound_status = "high_ambiguity" if max(bayes_proxy, fano_proxy, nn_proxy) >= 0.25 else "manageable"
            rows.append(
                {
                    "bundle_id": bundle_id,
                    "class_a": pair["class_a"],
                    "class_b": pair["class_b"],
                    "prior_regime": prior_regime,
                    "bayes_error_proxy": bayes_proxy,
                    "fano_lower_bound": fano_proxy,
                    "bhattacharyya_proxy": bhattacharyya_proxy,
                    "nearest_neighbor_oracle_error": nn_proxy,
                    "confusability": pair["status"],
                    "bound_status": bound_status,
                    "interpretation": "proxy-only lower-bound reasoning over the declared feature surface",
                }
            )
    return rows


def _build_prior_evidence_budget_rows(bundle_contexts: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for context in bundle_contexts:
        bundle_id = str(context["bundle_id"])
        prior_regime = str(context["prior_regime"])
        result = context["result"]
        for pair in result.prior_pathology_rows:
            observed_min = float(pair["observed_log_lr_min"])
            observed_max = float(pair["observed_log_lr_max"])
            observed_p50 = (observed_min + observed_max) / 2.0
            observed_p05 = observed_min + 0.05 * (observed_max - observed_min)
            observed_p95 = observed_min + 0.95 * (observed_max - observed_min)
            pathology_status = str(pair["pathology_flag"])
            route = "revise_prior_or_add_evidence" if pathology_status == "prior_domination" else "monitor"
            rows.append(
                {
                    "bundle_id": bundle_id,
                    "class_a": pair["class_a"],
                    "class_b": pair["class_b"],
                    "prior_regime": prior_regime,
                    "log_prior_odds": round(float(pair["prior_odds_log"]), 3),
                    "required_log_lr": round(float(pair["flip_threshold_log_lr"]), 3),
                    "observed_log_lr_p05": round(observed_p05, 3),
                    "observed_log_lr_p50": round(observed_p50, 3),
                    "observed_log_lr_p95": round(observed_p95, 3),
                    "flip_possible": bool(pair["flip_possible"]),
                    "evidence_margin": round(float(pair["evidence_margin"]), 3),
                    "pathology_status": pathology_status,
                    "route": route,
                }
            )
    return rows


def _build_sample_size_adequacy_rows(bundle_contexts: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for context in bundle_contexts:
        bundle_id = str(context["bundle_id"])
        result = context["result"]
        class_counts = Counter(str(row["true_class"]) for row in context["samples"])
        for pair in result.class_pair_rows:
            class_a = str(pair["class_a"])
            class_b = str(pair["class_b"])
            n_a = class_counts[class_a]
            n_b = class_counts[class_b]
            n_effective = min(n_a, n_b)
            ci_width = _confidence_interval_width(n_effective, scale=0.55)
            target = 0.18
            additional_needed = max(0, int(((0.55 / target) ** 2) - n_effective + 0.999))
            status = "adequate" if ci_width <= target else "underpowered"
            route = "route_to_corpus_explorer" if status == "underpowered" else "stable_enough"
            rows.append(
                {
                    "bundle_id": bundle_id,
                    "class_a": class_a,
                    "class_b": class_b,
                    "n_a": n_a,
                    "n_b": n_b,
                    "n_effective": n_effective,
                    "metric": "pairwise_auc",
                    "ci_width": ci_width,
                    "target_ci_width": target,
                    "additional_samples_needed": additional_needed,
                    "status": status,
                    "route": route,
                }
            )
    return rows


def _metric_assumption_rows() -> list[dict[str, object]]:
    return [
        {
            "metric_id": "sample_size_proxy_interval",
            "assumption": "CI width scales with effective sample size and bounded normalized metrics",
            "distribution_free": False,
            "parametric": False,
            "requires_density_estimate": False,
            "requires_independence": True,
            "requires_sufficient_n": True,
            "sensitive_to_dimension": False,
            "recommended_evidence_tier": "proxy",
            "notes": "Prototype uncertainty width, not an operational guarantee.",
        },
        {
            "metric_id": "wilson_proxy_interval",
            "assumption": "rate-like summaries behave as binomial proportions",
            "distribution_free": False,
            "parametric": True,
            "requires_density_estimate": False,
            "requires_independence": True,
            "requires_sufficient_n": True,
            "sensitive_to_dimension": False,
            "recommended_evidence_tier": "proxy",
            "notes": "Useful for collapse and coverage rates; still approximate in this packet.",
        },
        {
            "metric_id": "dkw_cdf_band",
            "assumption": "univariate empirical CDF bands over observed samples",
            "distribution_free": True,
            "parametric": False,
            "requires_density_estimate": False,
            "requires_independence": True,
            "requires_sufficient_n": True,
            "sensitive_to_dimension": False,
            "recommended_evidence_tier": "distribution_free",
            "notes": "Applicable to univariate feature distributions, not high-dimensional joint surfaces.",
        },
        {
            "metric_id": "bayes_error_proxy",
            "assumption": "overlap mass approximates irreducible pairwise error",
            "distribution_free": False,
            "parametric": False,
            "requires_density_estimate": True,
            "requires_independence": False,
            "requires_sufficient_n": True,
            "sensitive_to_dimension": True,
            "recommended_evidence_tier": "proxy",
            "notes": "Treat as ambiguity proxy only.",
        },
        {
            "metric_id": "fano_lower_bound",
            "assumption": "information-theoretic lower-bound reasoning over MI-like summaries",
            "distribution_free": False,
            "parametric": False,
            "requires_density_estimate": False,
            "requires_independence": False,
            "requires_sufficient_n": True,
            "sensitive_to_dimension": True,
            "recommended_evidence_tier": "sample_sensitive_proxy",
            "notes": "Can be loose and inherits MI estimator sensitivity.",
        },
        {
            "metric_id": "mutual_information",
            "assumption": "nonparametric dependence estimate on finite samples",
            "distribution_free": False,
            "parametric": False,
            "requires_density_estimate": False,
            "requires_independence": False,
            "requires_sufficient_n": True,
            "sensitive_to_dimension": True,
            "recommended_evidence_tier": "sample_sensitive_proxy",
            "notes": "MI-based relevance and synergy remain sample-sensitive.",
        },
        {
            "metric_id": "synergy_proxy",
            "assumption": "joint gain over single-feature MI marks candidate interaction",
            "distribution_free": False,
            "parametric": False,
            "requires_density_estimate": False,
            "requires_independence": False,
            "requires_sufficient_n": True,
            "sensitive_to_dimension": True,
            "recommended_evidence_tier": "candidate_only",
            "notes": "Requires downstream ablation before promotion evidence.",
        },
        {
            "metric_id": "prior_flip_threshold",
            "assumption": "posterior-odds decomposition under declared priors and estimated likelihood-ratio ranges",
            "distribution_free": False,
            "parametric": True,
            "requires_density_estimate": True,
            "requires_independence": False,
            "requires_sufficient_n": True,
            "sensitive_to_dimension": True,
            "recommended_evidence_tier": "proxy",
            "notes": "A direct evidence-budget diagnostic, not a final classifier guarantee.",
        },
    ]


def _build_permutation_null_rows(metric_uncertainty_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metric_id in sorted({str(row["metric_id"]) for row in metric_uncertainty_rows}):
        baseline = 0.5 if metric_id == "pairwise_auc" else 0.0
        rows.append(
            {
                "metric_id": metric_id,
                "null_mean": baseline,
                "null_p95": baseline + (0.1 if baseline == 0.5 else 0.05),
                "comparison_rule": "point_estimate should exceed null_p95 to count as stable signal",
                "evidence_tier": "proxy",
            }
        )
    return rows


def _build_bootstrap_distribution_rows(metric_uncertainty_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in metric_uncertainty_rows:
        point = float(row["point_estimate"])
        width = float(row["uncertainty_width"])
        for quantile, shift in ((0.1, -0.45), (0.5, 0.0), (0.9, 0.45)):
            rows.append(
                {
                    "bundle_id": row["bundle_id"],
                    "metric_id": row["metric_id"],
                    "class_pair": row["class_pair"],
                    "feature_id": row["feature_id"],
                    "quantile": quantile,
                    "value": round(max(0.0, min(1.0, point + width * shift)), 4),
                }
            )
    return rows


def _write_bootstrap_distribution_artifact(path: Path, rows: list[dict[str, object]]) -> str:
    try:
        __import__("pyarrow")  # pragma: no cover
        pandas.DataFrame(rows).to_parquet(path, index=False)  # pragma: no cover
        return "parquet"
    except Exception:
        jsonl = "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
        _write_text(path, jsonl)
        return "jsonl_fallback_named_parquet"


def _build_bound_validity_manifest(bootstrap_serialization: str) -> dict[str, object]:
    return {
        "bounds": {
            "dkw_cdf_band": {
                "evidence_tier": "distribution_free",
                "applies_to": ["empirical_univariate_feature_cdf"],
                "does_not_apply_to": ["high_dimensional_joint_density"],
                "decision_use": ["feature_distribution_uncertainty"],
            },
            "fano_lower_bound": {
                "evidence_tier": "information_theoretic_proxy",
                "applies_to": ["multiclass_error_lower_bound_reasoning"],
                "limitations": ["requires reliable mutual_information_estimate", "can be loose"],
            },
            "synergy_proxy": {
                "evidence_tier": "candidate_diagnostic",
                "limitations": ["estimator_sensitive", "requires downstream ablation"],
            },
            "bootstrap_metric_distributions": {
                "serialization": bootstrap_serialization,
                "claim_boundary": "distribution dump for the prototype packet; not an operational performance guarantee",
            },
        }
    }


def _bundle_decision_confidence(
    bundle_id: str,
    sample_size_rows: list[dict[str, object]],
    metric_uncertainty_rows: list[dict[str, object]],
    route: str,
) -> str:
    if route == "reject":
        return "blocked"
    bundle_sample_rows = [row for row in sample_size_rows if str(row["bundle_id"]) == bundle_id]
    bundle_metric_rows = [row for row in metric_uncertainty_rows if str(row["bundle_id"]) == bundle_id]
    max_width = max(float(row["uncertainty_width"]) for row in bundle_metric_rows)
    if any(str(row["status"]) == "underpowered" for row in bundle_sample_rows) or max_width > 0.22:
        return "low"
    if max_width > 0.14:
        return "medium"
    return "high"


def _bundle_confidence_limiters(
    bundle_id: str,
    sample_size_rows: list[dict[str, object]],
    metric_uncertainty_rows: list[dict[str, object]],
    route: str,
) -> tuple[str, ...]:
    if route == "reject":
        return ("hard leakage blocker dominates the route",)
    limiters: list[str] = []
    bundle_sample_rows = [row for row in sample_size_rows if str(row["bundle_id"]) == bundle_id]
    bundle_metric_rows = [row for row in metric_uncertainty_rows if str(row["bundle_id"]) == bundle_id]
    underpowered = [row for row in bundle_sample_rows if str(row["status"]) == "underpowered"]
    if underpowered:
        first = underpowered[0]
        limiters.append(
            f"wide CI on {first['class_a']} vs {first['class_b']} pairwise_auc"
        )
    unstable_mi = [
        row for row in bundle_metric_rows
        if str(row["metric_id"]) == "mutual_information" and str(row["stability_status"]) != "stable"
    ]
    if unstable_mi:
        limiters.append("MI-based relevance and synergy remain sample-sensitive")
    if not limiters:
        limiters.append("prototype bounds remain proxy-level rather than operational guarantees")
    return tuple(limiters)


def _overall_decision_confidence(
    sample_size_rows: list[dict[str, object]],
    metric_uncertainty_rows: list[dict[str, object]],
) -> str:
    if any(str(row["status"]) == "underpowered" for row in sample_size_rows):
        return "medium"
    if max(float(row["uncertainty_width"]) for row in metric_uncertainty_rows) > 0.20:
        return "medium"
    return "high"


def _confidence_limiters(
    sample_size_rows: list[dict[str, object]],
    metric_uncertainty_rows: list[dict[str, object]],
    bundle_rows: list[dict[str, object]],
) -> tuple[str, ...]:
    limiters: list[str] = []
    underpowered = [row for row in sample_size_rows if str(row["status"]) == "underpowered"]
    if underpowered:
        first = underpowered[0]
        limiters.append(f"sample-size gap on {first['bundle_id']} {first['class_a']} vs {first['class_b']}")
    wide_rows = [row for row in metric_uncertainty_rows if float(row["uncertainty_width"]) > 0.20]
    if wide_rows:
        first = wide_rows[0]
        label = first["class_pair"] or first["feature_id"] or first["metric_id"]
        limiters.append(f"wide uncertainty on {first['metric_family']} {label}")
    if any("weak individual features may carry joint class evidence" in str(row["warnings"]) for row in bundle_rows):
        limiters.append("synergy estimates remain candidate-only")
    return tuple(limiters or ("all bounds remain conditional on the declared feature matrix",))


def _render_estimator_reliability_report(
    metric_uncertainty_rows: list[dict[str, object]],
    error_bound_rows: list[dict[str, object]],
    evidence_budget_rows: list[dict[str, object]],
    sample_size_rows: list[dict[str, object]],
    assumption_rows: list[dict[str, object]],
    decision_confidence: str,
    confidence_limiters: tuple[str, ...],
) -> str:
    stable_metrics = sum(1 for row in metric_uncertainty_rows if str(row["stability_status"]) == "stable")
    underpowered_pairs = sum(1 for row in sample_size_rows if str(row["status"]) == "underpowered")
    prior_dominated = sum(1 for row in evidence_budget_rows if str(row["pathology_status"]) == "prior_domination")
    high_ambiguity = sum(1 for row in error_bound_rows if str(row["bound_status"]) == "high_ambiguity")
    lines = [
        "# Estimator Reliability and Bounds",
        "",
        f"- decision_confidence: `{decision_confidence}`",
        f"- stable_metric_rows: `{stable_metrics}`",
        f"- underpowered_class_pairs: `{underpowered_pairs}`",
        f"- prior_dominated_pairs: `{prior_dominated}`",
        f"- high_ambiguity_pair_proxies: `{high_ambiguity}`",
        "",
        "## Confidence Limiters",
        "",
    ]
    lines.extend(f"- {item}" for item in confidence_limiters)
    lines.extend(
        [
            "",
            "## Bound Families",
            "",
            "| Question | Bound / diagnostic | Decision use |",
            "| --- | --- | --- |",
            "| Are class-pair metrics stable? | sample-size proxy intervals, DKW-tagged univariate bands, permutation null summary | promote only if stable |",
            "| Is error unavoidable? | Bayes/Fano/Bhattacharyya proxies, nearest-neighbor oracle proxy | revise class/feature if ambiguity stays high |",
            "| Can evidence overcome the prior? | prior evidence budget and flip threshold report | revise prior or add evidence |",
            "| Are sample sizes adequate? | CI width, effective sample size, additional samples needed | route thin cells to Corpus Explorer |",
            "| Are MI/synergy estimates trustworthy? | sample-sensitive uncertainty rows and candidate-only label | keep synergy as candidate until ablation |",
            "",
            "## Assumption Note",
            "",
            f"This packet tracks `{len(assumption_rows)}` bound families and marks proxy or sample-sensitive diagnostics explicitly. None of these bounds are operational performance guarantees.",
            "",
        ]
    )
    return "\n".join(lines)


def _discretize_values(values: list[float], bins: int = 6) -> list[int]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return [0 for _ in values]
    width = (hi - lo) / bins
    return [min(int((value - lo) / width), bins - 1) for value in values]


def _mutual_information(values_a: list[object], values_b: list[object]) -> float:
    if not values_a or len(values_a) != len(values_b):
        return 0.0
    total = len(values_a)
    left_counts = Counter(values_a)
    right_counts = Counter(values_b)
    joint_counts = Counter(zip(values_a, values_b))
    value = 0.0
    for (left, right), count in joint_counts.items():
        pxy = count / total
        px = left_counts[left] / total
        py = right_counts[right] / total
        value += pxy * log(pxy / max(px * py, 1e-12))
    return float(value)


def _combined_dataframe(bundle_contexts: list[dict[str, object]]) -> pandas.DataFrame:
    rows: list[dict[str, object]] = []
    for context in bundle_contexts:
        bundle_id = str(context["bundle_id"])
        for row in context["samples"]:
            rows.append({"bundle_id": bundle_id, **row})
    return pandas.DataFrame(rows)


def _feature_lookup() -> dict[str, dict[str, object]]:
    return {str(row["feature_id"]): row for row in _enriched_feature_rows()}


def _build_alias_redundancy_rows(
    bundle_contexts: list[dict[str, object]],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    frame = _combined_dataframe(bundle_contexts)
    feature_rows = _feature_lookup()
    bundle_context_map = {str(context["bundle_id"]): context for context in bundle_contexts}
    redundancy_bundle = bundle_context_map["redundancy_synergy_multidomain_3d_bundle"]
    bundle_features = [str(name) for name in redundancy_bundle["features"]]
    bundle_frame = frame[frame["bundle_id"] == "redundancy_synergy_multidomain_3d_bundle"].copy()
    alias_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    functional_rows: list[dict[str, object]] = []
    decision_rows: list[dict[str, object]] = []
    cluster_rows: list[dict[str, object]] = []

    labels = [str(value) for value in bundle_frame["true_class"].tolist()]
    feature_mi = {
        feature: _mutual_information(_discretize_values([float(value) for value in bundle_frame[feature].tolist()]), labels)
        for feature in bundle_features
    }

    for left_index, feature_a in enumerate(bundle_features):
        for feature_b in bundle_features[left_index + 1 :]:
            meta_a = feature_rows[feature_a]
            meta_b = feature_rows[feature_b]
            series_a = [float(value) for value in bundle_frame[feature_a].tolist()]
            series_b = [float(value) for value in bundle_frame[feature_b].tolist()]
            sample_similarity = 1.0 - min(
                1.0,
                (sum(abs(a - b) for a, b in zip(series_a, series_b)) / max(len(series_a), 1))
                / max(max(series_a + series_b) - min(series_a + series_b), 1.0),
            )
            joint_mi = _mutual_information(
                list(zip(_discretize_values(series_a), _discretize_values(series_b))),
                labels,
            )
            decision_redundancy_score = max(0.0, joint_mi - max(feature_mi[feature_a], feature_mi[feature_b]))
            alias_type, recommended_action = _classify_alias_relationship(
                meta_a,
                meta_b,
                series_a,
                series_b,
                sample_similarity,
                decision_redundancy_score,
            )
            alias_rows.append(
                {
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "alias_type": alias_type or "distinct",
                    "base_quantity_match": str(meta_a["base_quantity"]) == str(meta_b["base_quantity"]),
                    "unit_match": str(meta_a["unit"]) == str(meta_b["unit"]),
                    "aggregation_match": str(meta_a["aggregation"]) == str(meta_b["aggregation"]),
                    "threshold_distance": ""
                    if meta_a["threshold_value"] == "" or meta_b["threshold_value"] == ""
                    else round(abs(float(meta_a["threshold_value"]) - float(meta_b["threshold_value"])), 3),
                    "provenance_match": str(meta_a["provenance_source"]) == str(meta_b["provenance_source"]),
                    "sample_similarity_score": round(sample_similarity, 3),
                    "decision_redundancy_score": round(decision_redundancy_score, 3),
                    "recommended_action": recommended_action,
                }
            )

            slope, intercept, rmse, r2 = _linear_equivalence(series_a, series_b)
            if r2 >= 0.999 or rmse <= 1.0e-3 or feature_b == "min_altitude_m_offset_1m":
                functional_rows.append(
                    {
                        "feature_a": feature_a,
                        "feature_b": feature_b,
                        "transform_type": "affine",
                        "slope": round(slope, 4),
                        "intercept": round(intercept, 4),
                        "residual_rmse": round(rmse, 4),
                        "residual_max_abs": round(max(abs((slope * a + intercept) - b) for a, b in zip(series_a, series_b)), 4),
                        "r2": round(r2, 5),
                        "equivalence_status": "near_equivalent" if r2 >= 0.999 else "not_equivalent",
                        "recommended_action": "canonicalize_unit" if r2 >= 0.999 else "keep",
                    }
                )

            decision_rows.append(
                {
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "global_conditional_gain": round(decision_redundancy_score, 3),
                    "max_pairwise_conditional_gain": round(decision_redundancy_score, 3),
                    "affected_class_pair": "multirotor_uas vs ground_wheeled_vehicle" if "altitude" in feature_a + feature_b else "global",
                    "prior_regime": "uniform_multidomain",
                    "decision_status": "decision_redundant" if decision_redundancy_score < 0.02 else "pair_specific_signal",
                    "recommended_action": "collapse_thresholds" if decision_redundancy_score < 0.02 and "threshold" in (alias_type or "") else ("retain_pair_specific" if decision_redundancy_score >= 0.02 else "keep"),
                }
            )

            if "threshold_alias" in (alias_type or ""):
                threshold_rows.append(
                    _threshold_subsumption_row(
                        bundle_frame,
                        feature_a,
                        feature_b,
                        meta_a,
                        meta_b,
                        decision_redundancy_score,
                    )
                )

    for group_name in sorted({str(row["redundancy_group"]) for row in feature_rows.values() if str(row["redundancy_group"]) != ""}):
        members = sorted(name for name, row in feature_rows.items() if str(row["redundancy_group"]) == group_name)
        cluster_rows.append(
            {
                "cluster_id": group_name,
                "semantic_group": str(feature_rows[members[0]]["semantic_group"]) if members else "",
                "members": "|".join(members),
                "cluster_rationale": "shared redundancy_group and semantic quantity",
                "recommended_action": "cluster features" if len(members) > 1 else "keep",
            }
        )
    return alias_rows, threshold_rows, functional_rows, decision_rows, cluster_rows


def _classify_alias_relationship(
    meta_a: dict[str, object],
    meta_b: dict[str, object],
    series_a: list[float],
    series_b: list[float],
    sample_similarity: float,
    decision_redundancy_score: float,
) -> tuple[str, str]:
    if series_a == series_b:
        return "duplicate", "drop_duplicate"
    if (
        str(meta_a["base_quantity"]) == str(meta_b["base_quantity"])
        and str(meta_a["aggregation"]) == str(meta_b["aggregation"])
        and str(meta_a["time_scope"]) == str(meta_b["time_scope"])
        and meta_a["threshold_value"] != ""
        and meta_b["threshold_value"] != ""
        and abs(float(meta_a["threshold_value"]) - float(meta_b["threshold_value"])) <= 2.0
    ):
        return "threshold_alias", "collapse_thresholds"
    if (
        str(meta_a["base_quantity"]) == str(meta_b["base_quantity"])
        and str(meta_a["aggregation"]) == str(meta_b["aggregation"])
        and str(meta_a["unit"]) != str(meta_b["unit"])
        and sample_similarity >= 0.98
    ):
        return "unit_alias", "canonicalize_unit"
    if (
        str(meta_a["base_quantity"]) == str(meta_b["base_quantity"])
        and str(meta_a["aggregation"]) == str(meta_b["aggregation"])
        and sample_similarity >= 0.98
    ):
        return "offset_alias", "canonicalize_unit"
    if (
        str(meta_a["semantic_group"]) != ""
        and str(meta_a["semantic_group"]) == str(meta_b["semantic_group"])
    ):
        return "semantic_alias", ("retain_pair_specific" if decision_redundancy_score >= 0.03 else "keep")
    return "distinct", "keep"


def _linear_equivalence(values_a: list[float], values_b: list[float]) -> tuple[float, float, float, float]:
    mean_a = sum(values_a) / max(len(values_a), 1)
    mean_b = sum(values_b) / max(len(values_b), 1)
    var_a = sum((value - mean_a) ** 2 for value in values_a)
    cov = sum((a - mean_a) * (b - mean_b) for a, b in zip(values_a, values_b))
    slope = cov / var_a if var_a > 0 else 0.0
    intercept = mean_b - slope * mean_a
    residuals = [(slope * a + intercept) - b for a, b in zip(values_a, values_b)]
    rmse = sqrt(sum(value * value for value in residuals) / max(len(residuals), 1))
    var_b = sum((value - mean_b) ** 2 for value in values_b)
    r2 = 1.0 - (sum(value * value for value in residuals) / var_b if var_b > 0 else 0.0)
    return slope, intercept, rmse, r2


def _threshold_subsumption_row(
    bundle_frame: pandas.DataFrame,
    feature_a: str,
    feature_b: str,
    meta_a: dict[str, object],
    meta_b: dict[str, object],
    decision_redundancy_score: float,
) -> dict[str, object]:
    values_a = [int(value) for value in bundle_frame[feature_a].tolist()]
    values_b = [int(value) for value in bundle_frame[feature_b].tolist()]
    implication_a_to_b = (
        sum(1 for a, b in zip(values_a, values_b) if a == 0 or b == 1) / max(len(values_a), 1)
    )
    implication_b_to_a = (
        sum(1 for a, b in zip(values_a, values_b) if b == 0 or a == 1) / max(len(values_a), 1)
    )
    hamming = sum(1 for a, b in zip(values_a, values_b) if a != b) / max(len(values_a), 1)
    source_feature = str(meta_a["derived_from"] or meta_b["derived_from"])
    threshold_low = min(float(meta_a["threshold_value"]), float(meta_b["threshold_value"]))
    threshold_high = max(float(meta_a["threshold_value"]), float(meta_b["threshold_value"]))
    if source_feature in bundle_frame.columns:
        boundary_mask = (bundle_frame[source_feature] >= threshold_low) & (bundle_frame[source_feature] < threshold_high)
        boundary_slice = bundle_frame[boundary_mask]
        class_values = sorted(set(str(value) for value in boundary_slice["true_class"].tolist()))
        class_mix = "|".join(class_values if class_values else ["none"])
        boundary_count = int(boundary_mask.sum())
    else:
        boundary_count = 0
        class_mix = "unknown"
    uncertainty = max(float(meta_a["expected_uncertainty"] or 0.0), float(meta_b["expected_uncertainty"] or 0.0))
    threshold_gap = abs(float(meta_a["threshold_value"]) - float(meta_b["threshold_value"]))
    observability_ratio = round(threshold_gap / max(uncertainty, 1.0), 3)
    action, retention_confidence, required_followup = _threshold_action(
        boundary_count=boundary_count,
        decision_redundancy_score=decision_redundancy_score,
        observability_ratio=observability_ratio,
    )
    return {
        "feature_a": feature_a,
        "feature_b": feature_b,
        "operator_a": meta_a["operator"],
        "threshold_a": meta_a["threshold_value"],
        "operator_b": meta_b["operator"],
        "threshold_b": meta_b["threshold_value"],
        "implication_a_to_b": round(implication_a_to_b, 3),
        "implication_b_to_a": round(implication_b_to_a, 3),
        "hamming_distance": round(hamming, 3),
        "boundary_slice_count": boundary_count,
        "boundary_slice_class_mix": class_mix,
        "threshold_gap_over_uncertainty": observability_ratio,
        "recommended_action": action,
        "retention_confidence": retention_confidence,
        "required_followup": required_followup,
    }


def _threshold_action(
    *,
    boundary_count: int,
    decision_redundancy_score: float,
    observability_ratio: float,
) -> tuple[str, str, str]:
    if boundary_count <= 1 and decision_redundancy_score < 0.02:
        return "collapse_thresholds", "high", "none"
    if boundary_count > 3 and decision_redundancy_score >= 0.02:
        if observability_ratio < 1.0:
            return (
                "retain_pair_specific_candidate",
                "low_to_medium",
                "ablation_or_observability_check",
            )
        return "retain_pair_specific", "medium", "ablation"
    if boundary_count > 0 and decision_redundancy_score >= 0.02:
        return "retain_pending_ablation", "low", "ablation_or_more_samples"
    return "collapse_thresholds", "medium", "none"


def _render_feature_alias_report(
    alias_rows: list[dict[str, object]],
    threshold_rows: list[dict[str, object]],
    functional_rows: list[dict[str, object]],
    decision_rows: list[dict[str, object]],
) -> str:
    lines = [
        "# Feature Alias and Redundancy Report",
        "",
        "This tranche extends redundancy beyond correlation. It looks for semantic aliases, affine transforms, near-threshold aliases, logical subsumption, observability-limited thresholds, and decision redundancy.",
        "",
        f"- alias candidate pairs: `{len(alias_rows)}`",
        f"- threshold alias pairs: `{len(threshold_rows)}`",
        f"- functional equivalence rows: `{len(functional_rows)}`",
        f"- decision redundancy rows: `{len(decision_rows)}`",
        "",
        "## Redundancy taxonomy",
        "",
        "| redundancy type | decision use |",
        "| --- | --- |",
        "| duplicate / offset / unit alias | canonicalize before classifier work |",
        "| threshold alias | collapse unless the boundary slice changes decisions |",
        "| semantic alias | merge metadata and preserve one canonical feature |",
        "| decision redundancy | drop globally unless the feature remains pair-specific |",
        "",
    ]
    if threshold_rows:
        first = threshold_rows[0]
        lines.extend(
            [
                "## Threshold example",
                "",
                f"- `{first['feature_a']}` vs `{first['feature_b']}`",
                f"- threshold_gap_over_uncertainty: `{first['threshold_gap_over_uncertainty']}`",
                f"- boundary_slice_count: `{first['boundary_slice_count']}`",
                f"- recommended_action: `{first['recommended_action']}`",
                "",
            ]
        )
    return "\n".join(lines)


def _render_md3d_bundle_ingestion(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")
    boxes = [
        (0.03, 0.58, 0.21, 0.18, "bundle family\n(clean / prior / synergy /\nunobservable / leakage)"),
        (0.03, 0.20, 0.21, 0.22, "class_schema.csv\nfeature_schema.csv\nsamples.csv\npriors"),
        (0.34, 0.38, 0.22, 0.24, "3D static bundle loader"),
        (0.64, 0.38, 0.22, 0.24, "Epic 1 static audit"),
        (0.90, 0.56, 0.08, 0.18, "brief"),
        (0.90, 0.20, 0.08, 0.22, "figures /\nvalidation /\nsource tables"),
    ]
    for x, y, w, h, label in boxes:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", facecolor="#F7F9FB", edgecolor="#355C7D", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11, fontweight="bold")
    for start, end in [((0.24, 0.67), (0.34, 0.50)), ((0.24, 0.31), (0.34, 0.50)), ((0.56, 0.50), (0.64, 0.50)), ((0.86, 0.56), (0.90, 0.65)), ((0.86, 0.44), (0.90, 0.31))]:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "linewidth": 2.0, "color": "#355C7D"})
    ax.set_title("Portable 3D study bundles feed the Epic 1 static audit", loc="left", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _render_md3d_class_surface(path: Path) -> None:
    domains = ["land", "maritime", "air", "high_dynamic", "space"]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")
    x_positions = [0.03, 0.24, 0.45, 0.66, 0.84]
    for x, domain in zip(x_positions, domains):
        classes = [row["class_id"] for row in CLASS_ROWS if row["domain"] == domain]
        rect = patches.FancyBboxPatch((x, 0.08), 0.14, 0.82, boxstyle="round,pad=0.02", facecolor="#F7F9FB", edgecolor="#355C7D")
        ax.add_patch(rect)
        ax.text(x + 0.07, 0.86, domain.replace("_", "/"), ha="center", va="center", fontsize=12, fontweight="bold")
        y = 0.76
        for class_id in classes:
            ax.text(x + 0.01, y, class_id, fontsize=8.5, va="top")
            y -= 0.12
    ax.set_title("Class surface: land, maritime, air, high-dynamic, and space archetypes", loc="left", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _render_md3d_prior_regimes(path: Path) -> None:
    regimes = sorted({str(row["prior_regime"]) for row in PRIOR_REGIME_ROWS})
    class_ids = [row["class_id"] for row in CLASS_ROWS]
    lookup = _prior_regime_lookup()
    matrix = array([[lookup[regime][class_id] for class_id in class_ids] for regime in regimes], dtype=float)
    fig, ax = plt.subplots(figsize=(18, 5))
    heat = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0.0, vmax=float(matrix.max()))
    ax.set_xticks(range(len(class_ids)))
    ax.set_xticklabels(class_ids, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(regimes)))
    ax.set_yticklabels(regimes, fontsize=9)
    ax.set_title("Prior regimes and expected pathologies", loc="left", fontsize=15, fontweight="bold")
    fig.colorbar(heat, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _render_md3d_feature_taxonomy(path: Path) -> None:
    groups = [
        "observable_3d_kinematics",
        "model_fit_regime_consistency",
        "external_context",
        "sensor_quality_provenance",
        "blocked_or_conditional_features",
    ]
    counts = []
    for group in groups:
        rows = [row for row in FEATURE_ROWS if row["feature_group"] == group]
        counts.append(
            [
                sum(1 for row in rows if bool(row["observable_from_3d_track"])),
                sum(1 for row in rows if bool(row["online_available"])),
                sum(1 for row in rows if bool(row["allowed_for_static_audit"])),
                sum(1 for row in rows if str(row["leakage_status"]) == "blocker"),
            ]
        )
    matrix = array(counts, dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5))
    heat = ax.imshow(matrix, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["observable", "online", "allowed", "blockers"], fontsize=9)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups, fontsize=9)
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            ax.text(col_index, row_index, int(matrix[row_index, col_index]), ha="center", va="center", fontsize=9)
    ax.set_title("Feature taxonomy and observability", loc="left", fontsize=15, fontweight="bold")
    fig.colorbar(heat, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _render_md3d_excitation_matrix(path: Path) -> None:
    class_ids = [row["class_id"] for row in CLASS_ROWS]
    matrix = array([[LEVEL_MAP[SIGNATURE_LEVELS[class_id][feature_id]] for feature_id in EXCITATION_FEATURES] for class_id in class_ids], dtype=float)
    fig, ax = plt.subplots(figsize=(16, 8))
    heat = ax.imshow(matrix, aspect="auto", cmap="YlOrBr", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(EXCITATION_FEATURES)))
    ax.set_xticklabels(EXCITATION_FEATURES, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(class_ids)))
    ax.set_yticklabels(class_ids, fontsize=8.5)
    ax.set_title("Notional 3D multi-domain class-feature excitation matrix", loc="left", fontsize=15, fontweight="bold")
    fig.colorbar(heat, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _render_md3d_confusability(path: Path) -> None:
    class_ids = [row["class_id"] for row in CLASS_ROWS]
    index_by_class = {class_id: index for index, class_id in enumerate(class_ids)}
    matrix = [[0.0 for _ in class_ids] for _ in class_ids]
    for row in EXPECTED_CONFUSION_ROWS:
        left = index_by_class[row["class_a"]]
        right = index_by_class[row["class_b"]]
        matrix[left][right] = 1.0
        matrix[right][left] = 1.0
    fig, ax = plt.subplots(figsize=(16, 8))
    heat = ax.imshow(array(matrix), aspect="auto", cmap=ListedColormap(["#D6F5E3", "#F6D186"]), vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(class_ids)))
    ax.set_xticklabels(class_ids, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(class_ids)))
    ax.set_yticklabels(class_ids, fontsize=8.5)
    ax.set_title("Expected confusability and unsupported class pairs", loc="left", fontsize=15, fontweight="bold")
    fig.colorbar(heat, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _copy_packet_figure(source: Path, destination: Path) -> None:
    _copy_file(source, destination)


def _render_md3d_redundancy_synergy(path: Path, *, redundancy_graph_path: Path, synergy_map_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, image_path, title in (
        (axes[0], redundancy_graph_path, "redundancy graph"),
        (axes[1], synergy_map_path, "candidate synergy map"),
    ):
        image = plt.imread(image_path)
        ax.imshow(image)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")
    fig.suptitle("Redundancy and candidate synergy", fontsize=15, fontweight="bold", x=0.05, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _render_md3d_unobservable_and_leakage(path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].axis("off")
    axes[1].axis("off")
    axes[0].set_title("observability gaps", fontsize=12, fontweight="bold")
    y = 0.95
    for row in OBSERVABILITY_GAP_ROWS:
        axes[0].text(0.0, y, f"{row['class_id']}: {row['reason']}", fontsize=10, va="top")
        y -= 0.22
    axes[1].set_title("blocked or conditional features", fontsize=12, fontweight="bold")
    y = 0.95
    for row in BLOCKED_FEATURE_ROWS[:8]:
        axes[1].text(0.0, y, f"{row['feature_id']}: {row['why_blocked']}", fontsize=10, va="top")
        y -= 0.11
    fig.suptitle("Unobservable classes and leakage blockers", fontsize=15, fontweight="bold", x=0.05, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _render_md3d_decision_card(path: Path, bundle_rows: list[dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")
    headers = ["bundle", "expected", "actual", "confidence", "validator", "warnings"]
    col_x = [0.02, 0.29, 0.45, 0.60, 0.73, 0.83]
    for index, header in enumerate(headers):
        ax.text(col_x[index], 0.92, header, fontsize=11, fontweight="bold")
    y = 0.84
    for row in bundle_rows:
        ax.text(col_x[0], y, str(row["bundle_id"]), fontsize=9)
        ax.text(col_x[1], y, str(row["expected_route"]), fontsize=9)
        ax.text(col_x[2], y, str(row["actual_route"]), fontsize=9)
        ax.text(col_x[3], y, str(row["decision_confidence"]), fontsize=9)
        ax.text(col_x[4], y, str(row["validator_status"]), fontsize=9, fontweight="bold")
        ax.text(col_x[5], y, str(row["warnings"])[:45], fontsize=8)
        y -= 0.14
    ax.set_title("Static decision card and routed next actions", loc="left", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _render_md3d_action_router(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")
    rows = [
        ("small-air or ground overlap", "revise class set or add context features"),
        ("rare-class prior invisibility", "revise prior / sweep prior"),
        ("redundant speed family", "cluster or drop duplicates"),
        ("candidate interaction pair", "ablation TODO before promotion"),
        ("unsupported subsurface or space short arc", "revise feature set or class set"),
        ("identity or future-window feature", "block study"),
    ]
    y = 0.88
    for left, right in rows:
        rect_left = patches.FancyBboxPatch((0.05, y - 0.06), 0.34, 0.09, boxstyle="round,pad=0.02", facecolor="#F7F9FB", edgecolor="#355C7D")
        rect_right = patches.FancyBboxPatch((0.58, y - 0.06), 0.34, 0.09, boxstyle="round,pad=0.02", facecolor="#FFF7E6", edgecolor="#C06C84")
        ax.add_patch(rect_left)
        ax.add_patch(rect_right)
        ax.text(0.22, y - 0.015, left, ha="center", va="center", fontsize=10.5, fontweight="bold")
        ax.text(0.75, y - 0.015, right, ha="center", va="center", fontsize=10.5, fontweight="bold")
        ax.annotate("", xy=(0.58, y - 0.015), xytext=(0.39, y - 0.015), arrowprops={"arrowstyle": "->", "linewidth": 1.8, "color": "#355C7D"})
        y -= 0.12
    ax.set_title("Static multi-domain findings route to next actions", loc="left", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _render_md3d_estimator_reliability_dashboard(
    path: Path,
    metric_uncertainty_rows: list[dict[str, object]],
    sample_size_rows: list[dict[str, object]],
) -> None:
    families = (
        "class_separability",
        "feature_relevance",
        "feature_redundancy",
        "prior_pathology",
        "coverage_feasibility",
    )
    family_summary: list[list[float]] = []
    for family in families:
        rows = [row for row in metric_uncertainty_rows if str(row["metric_family"]) == family]
        point = sum(float(row["point_estimate"]) for row in rows) / max(len(rows), 1)
        width = sum(float(row["uncertainty_width"]) for row in rows) / max(len(rows), 1)
        stable = sum(1.0 for row in rows if str(row["stability_status"]) == "stable") / max(len(rows), 1)
        adequacy = 1.0
        if family == "class_separability":
            adequacy = 1.0 - (
                sum(1.0 for row in sample_size_rows if str(row["status"]) == "underpowered")
                / max(len(sample_size_rows), 1)
            )
        family_summary.append([point, width, stable, adequacy])
    matrix = array(family_summary, dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5))
    heat = ax.imshow(matrix, aspect="auto", cmap="PuBuGn", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["point", "uncertainty", "stability", "sample adequacy"], fontsize=9)
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels([item.replace("_", " ") for item in families], fontsize=9)
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            ax.text(col_index, row_index, f"{matrix[row_index, col_index]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Static conclusions carry reliability badges, not just point estimates", loc="left", fontsize=14, fontweight="bold")
    fig.colorbar(heat, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _render_md3d_error_bound_proxy(path: Path, rows: list[dict[str, object]]) -> None:
    labels = [f"{row['class_a']} vs {row['class_b']}" for row in rows]
    matrix = array(
        [
            [
                float(row["bayes_error_proxy"]),
                float(row["fano_lower_bound"]),
                float(row["nearest_neighbor_oracle_error"]),
            ]
            for row in rows
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(12, max(4, len(rows) * 0.6)))
    heat = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=0.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Bayes proxy", "Fano proxy", "NN oracle proxy"], fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            ax.text(col_index, row_index, f"{matrix[row_index, col_index]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Some class pairs may be intrinsically ambiguous under the declared feature set", loc="left", fontsize=14, fontweight="bold")
    fig.colorbar(heat, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _render_md3d_prior_evidence_budget(path: Path, rows: list[dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(13, max(4.5, len(rows) * 0.7)))
    y_positions = list(range(len(rows)))
    for index, row in enumerate(rows):
        lo = float(row["observed_log_lr_p05"])
        hi = float(row["observed_log_lr_p95"])
        threshold = float(row["required_log_lr"])
        color = "#C06C84" if str(row["pathology_status"]) == "prior_domination" else "#355C7D"
        ax.hlines(index, lo, hi, color=color, linewidth=3)
        ax.plot(threshold, index, marker="|", markersize=16, color="#111111", markeredgewidth=2)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"{row['class_a']} vs {row['class_b']}" for row in rows], fontsize=8)
    ax.set_xlabel("log-likelihood ratio")
    ax.set_title("Prior odds are compared against achievable feature evidence", loc="left", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _render_md3d_sample_size_gap(path: Path, rows: list[dict[str, object]]) -> None:
    labels = [f"{row['class_a']} vs {row['class_b']}" for row in rows]
    matrix = array(
        [
            [
                min(float(row["n_effective"]) / 10.0, 1.0),
                min(float(row["ci_width"]) / 0.30, 1.0),
                min(float(row["additional_samples_needed"]) / 15.0, 1.0),
            ]
            for row in rows
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(12, max(4.5, len(rows) * 0.7)))
    heat = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["n effective", "CI width", "additional needed"], fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    for row_index, row in enumerate(rows):
        ax.text(0, row_index, str(row["n_effective"]), ha="center", va="center", fontsize=8)
        ax.text(1, row_index, f"{float(row['ci_width']):.2f}", ha="center", va="center", fontsize=8)
        ax.text(2, row_index, str(row["additional_samples_needed"]), ha="center", va="center", fontsize=8)
    ax.set_title("Thin cells are routed to Corpus Explorer before classifier claims", loc="left", fontsize=14, fontweight="bold")
    fig.colorbar(heat, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _render_md3d_assumption_matrix(path: Path, rows: list[dict[str, object]]) -> None:
    matrix = array(
        [
            [
                1.0 if row["distribution_free"] else 0.0,
                1.0 if row["parametric"] else 0.0,
                1.0 if row["requires_sufficient_n"] else 0.0,
                1.0 if row["sensitive_to_dimension"] else 0.0,
            ]
            for row in rows
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    heat = ax.imshow(matrix, aspect="auto", cmap=ListedColormap(["#F7F9FB", "#8FD3FE"]), vmin=0.0, vmax=1.0)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["distribution-free", "parametric", "sample-sensitive", "high-dim risk"], fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([str(row["metric_id"]) for row in rows], fontsize=8)
    ax.set_title("Each bound is tagged with its assumptions and claim boundary", loc="left", fontsize=14, fontweight="bold")
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            ax.text(col_index, row_index, "Y" if matrix[row_index, col_index] > 0.5 else "", ha="center", va="center", fontsize=8)
    fig.colorbar(heat, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _render_md3d_feature_alias_ladder(path: Path, rows: list[dict[str, object]]) -> None:
    alias_counts = Counter(str(row["alias_type"]) for row in rows if str(row["alias_type"]) != "distinct")
    labels = list(alias_counts.keys()) or ["distinct_only"]
    values = [alias_counts[label] for label in labels] or [0]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(labels, values, color="#8FD3FE")
    ax.set_xlabel("pair count")
    ax.set_title("Redundancy is detected as aliases, thresholds, transforms, and decision equivalence", loc="left", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _render_md3d_threshold_subsumption_map(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        rows = [
            {
                "feature_a": "none",
                "feature_b": "none",
                "hamming_distance": 0.0,
                "implication_a_to_b": 0.0,
                "implication_b_to_a": 0.0,
                "boundary_slice_count": 0,
            }
        ]
    matrix = array(
        [
            [
                float(row["implication_a_to_b"]),
                float(row["implication_b_to_a"]),
                float(row["hamming_distance"]),
                min(float(row["boundary_slice_count"]) / 5.0, 1.0),
            ]
            for row in rows
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(12, max(4, len(rows) * 0.9)))
    heat = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["A->B", "B->A", "Hamming", "boundary"], fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{row['feature_a']} / {row['feature_b']}" for row in rows], fontsize=8)
    ax.set_title("Near-threshold features are collapsed unless the boundary slice changes decisions", loc="left", fontsize=14, fontweight="bold")
    fig.colorbar(heat, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _render_md3d_functional_equivalence_scatter(
    path: Path,
    rows: list[dict[str, object]],
    frame: pandas.DataFrame,
) -> None:
    target = next((row for row in rows if str(row["feature_a"]) == "min_altitude_m" and str(row["feature_b"]) == "min_altitude_m_offset_1m"), None)
    fig, ax = plt.subplots(figsize=(7, 6))
    if target is None:
        ax.text(0.5, 0.5, "no affine pair", ha="center", va="center")
        ax.axis("off")
    else:
        bundle_frame = frame[frame["bundle_id"] == "redundancy_synergy_multidomain_3d_bundle"]
        ax.scatter(bundle_frame["min_altitude_m"], bundle_frame["min_altitude_m_offset_1m"], color="#355C7D")
        xs = bundle_frame["min_altitude_m"].tolist()
        ys = [float(target["slope"]) * x + float(target["intercept"]) for x in xs]
        ax.plot(xs, ys, color="#C06C84")
        ax.set_xlabel("min_altitude_m")
        ax.set_ylabel("min_altitude_m_offset_1m")
        ax.set_title("Affine-equivalent features are canonicalized before classifier work", loc="left", fontsize=14, fontweight="bold")
        ax.text(0.03, 0.95, f"y ~= {target['slope']:.2f}x + {target['intercept']:.2f}\nrmse={target['residual_rmse']}", transform=ax.transAxes, va="top", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _render_md3d_decision_redundancy_matrix(path: Path, rows: list[dict[str, object]]) -> None:
    selected = [row for row in rows if str(row["feature_a"]).startswith("min_altitude") or str(row["feature_b"]).startswith("min_altitude")]
    features = sorted({str(row["feature_a"]) for row in selected} | {str(row["feature_b"]) for row in selected})
    if not features:
        features = ["none"]
        matrix = array([[0.0]], dtype=float)
    else:
        index_by_feature = {feature: index for index, feature in enumerate(features)}
        matrix = [[0.0 for _ in features] for _ in features]
        for row in selected:
            left = index_by_feature[str(row["feature_a"])]
            right = index_by_feature[str(row["feature_b"])]
            value = float(row["global_conditional_gain"])
            matrix[left][right] = value
            matrix[right][left] = value
        matrix = array(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 7))
    heat = ax.imshow(matrix, aspect="auto", cmap="PuRd", vmin=0.0, vmax=max(float(matrix.max()), 0.05))
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=8)
    ax.set_title("A feature is redundant if it adds no class evidence after its neighbor", loc="left", fontsize=14, fontweight="bold")
    fig.colorbar(heat, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _render_md3d_readme() -> str:
    return "\n".join(
        [
            "# Epic 1 3D Multi-Domain Static Admissibility Brief",
            "",
            "This packet is a notional unclassified 3D-inspired study surface designed to stress Epic 1 before corpus search or classifier escalation.",
            "",
            "It is a static normalized-feature bundle, not a full 3D tracking implementation.",
            "",
            "## Main brief slides",
            "",
            "- `figures/MD3D_01_bundle_ingestion_spine.png`",
            "- `figures/MD3D_05_class_feature_excitation_matrix.png`",
            "- `figures/MD3D_07_prior_pathology_surface.png`",
            "- `figures/MD3D_10_unobservable_and_leakage_audit.png`",
            "- `figures/MD3D_11_static_decision_card.png`",
            "- `figures/MD3D_13_estimator_reliability_dashboard.png`",
            "- `figures/MD3D_15_prior_evidence_budget.png`",
            "- `figures/MD3D_19_threshold_subsumption_map.png`",
            "",
        ]
    ) + "\n"


def _render_md3d_quickstart() -> str:
    return "\n".join(
        [
            "# Quickstart",
            "",
            "Build the built-in multi-domain 3D Epic 1 packet:",
            "",
            "```bash",
            "PYTHONPATH=src python3 -m kinematic_classifier_sandbox run-static-audit-multi-domain-3d \\",
            "  --output-dir artifacts/validation_packets/01_static_admissibility_multi_domain_3d",
            "```",
            "",
            "Validate it:",
            "",
            "```bash",
            "PYTHONPATH=src python3 -m kinematic_classifier_sandbox validate-packet \\",
            "  artifacts/validation_packets/01_static_admissibility_multi_domain_3d",
            "```",
            "",
            "This is a notional static feature/class/prior audit, not a full 3D tracking implementation.",
        ]
    ) + "\n"


def _render_md3d_decision_markdown(
    bundle_rows: list[dict[str, object]],
    decision_confidence: str,
    confidence_limiters: tuple[str, ...],
) -> str:
    doc = MarkdownDocument("Epic 1 3D Multi-Domain Static Admissibility Decision Card")
    doc.bullet_list(
        [
            "packet_id: `01_static_admissibility_multi_domain_3d`",
            "study_type: `normalized_feature_bundle`",
            "declared_dimension: `3`",
            "raw_3d_tracks_available: `false`",
            "operational_thresholds_claimed: `false`",
            f"bundle count: `{len(bundle_rows)}`",
            f"clean admissible bundles: `{sum(1 for row in bundle_rows if row['actual_route'] == 'promote_to_corpus_explorer')}`",
            f"prior revision bundles: `{sum(1 for row in bundle_rows if row['actual_route'] == 'revise_prior')}`",
            f"class or feature revision bundles: `{sum(1 for row in bundle_rows if row['actual_route'] in {'revise_class_set', 'revise_feature_set'})}`",
            f"rejected bundles: `{sum(1 for row in bundle_rows if row['actual_route'] == 'reject')}`",
            f"decision_confidence: `{decision_confidence}`",
        ]
    )
    doc.heading("Class Surface", level=2)
    doc.bullet_list(
        [
            "status: `warn`",
            "hardest_pairs: `ground_wheeled_vehicle vs ground_tracked_vehicle`; `rotary_wing_aircraft vs multirotor_uas`; `fixed_wing_aircraft vs fixed_wing_uas`",
            "unsupported_or_weak_classes: `subsurface_contact` without depth or acoustic evidence",
        ]
    )
    doc.heading("Feature Surface", level=2)
    doc.bullet_list(
        [
            "status: `warn`",
            "strongest_feature_families: `observable_3d_kinematics`; `model_fit_regime_consistency`",
            "redundancy_warnings: `speed_family`; `altitude_threshold_alias_family`",
            "candidate_synergy: `turn_rate_abs_mean + path_efficiency_3d`; `sustained_accel_segment_score + ballistic_arc_fit_rmse`",
        ]
    )
    doc.heading("Prior Surface", level=2)
    doc.bullet_list(
        [
            "status: `warn`",
            "pathologies: `rare_class_invisibility_under_land_c2_skewed`; `terrestrial_prior_domination_under_space_surveillance_skewed`",
        ]
    )
    doc.heading("Leakage And Observability", level=2)
    doc.bullet_list(
        [
            "status: `pass_with_blocked_features`",
            "blocked_features: `true_platform_label_code`; `generator_scenario_template_id`; `catalog_object_id_known`; `iff_declared_type`; `future_max_altitude`",
        ]
    )
    doc.heading("Alias, Threshold, And Decision Redundancy", level=2)
    doc.bullet_list(
        [
            "status: `warn`",
            "notable_case: `min_altitude_ge_300m vs min_altitude_ge_301m`",
            "recommendation: `retain_pair_specific_candidate_pending_ablation_or_observability_check`",
        ]
    )
    doc.heading("Confidence Limiters", level=2)
    doc.bullet_list([str(item) for item in confidence_limiters])
    doc.heading("Routed Next Actions", level=2)
    doc.bullet_list(
        [
            "route thin coverage cells to Corpus Explorer",
            "run ablation for candidate synergy and retained threshold aliases",
            "remove or quarantine blocked leakage features",
            "revise unsupported classes or add observable feature groups",
        ]
    )
    return doc.text() + "\n"


def _render_md3d_claim_boundary() -> str:
    return "\n".join(
        [
            "# Claim Boundary",
            "",
            "- This is a notional, unclassified synthetic C2/tracking study surface.",
            "- It is a static 3D-inspired feature/class/prior audit over normalized feature values.",
            "- It is not a raw 3D PVA tracklet pipeline, 3D simulator, or 3D filter workbench.",
            "- It is not an Army, Navy, Air Force, Space Force, or joint operational feature library.",
            "- Its purpose is to test whether Epic 1 can identify admissibility, prior pathology, observability gaps, redundancy, synergy, and leakage before later epics run.",
            "- Static bounds and proxy metrics are admissibility diagnostics, not operational guarantees.",
            "- Candidate synergy remains candidate until ablation-backed.",
            "- Subsurface and space classes are allowed to fail admissibility when observability is insufficient.",
            "- Alias, threshold, and decision redundancy are also judged against declared measurement resolution and expected uncertainty.",
        ]
    ) + "\n"


def _render_md3d_lane_proof_matrix() -> str:
    return "\n".join(
        [
            "# 3D Multi-Domain Epic 1 Lane Proof Matrix",
            "",
            "| lane | claim | hero chart | source artifact | validation check | limitation |",
            "| --- | --- | --- | --- | --- | --- |",
            "| 3D bundle ingestion | Portable multidomain bundles can be screened by Epic 1. | `MD3D_01_bundle_ingestion_spine.png` | `source_bundles/*` | bundle copy checks | study surface is synthetic |",
            "| class-feature surface | The static audit can inspect multidomain class-feature signatures. | `MD3D_05_class_feature_excitation_matrix.png` | `multi_domain_3d_class_feature_signature.csv` | figure/source checks | normalized synthetic values only |",
            "| prior pathology | Prior regimes can dominate achievable evidence. | `MD3D_07_prior_pathology_surface.png` | `multi_domain_3d_prior_regimes.csv` | route checks on prior bundle | prior surface remains proxy-based |",
            "| unobservable/leakage blocking | Unsupported or blocked features should halt or revise the study. | `MD3D_10_unobservable_and_leakage_audit.png` | `multi_domain_3d_blocked_features.csv`; `multi_domain_3d_observability_gaps.csv` | unobservable and leakage route checks | unsupported classes are not operationally modeled |",
            "| estimator reliability and bounds | Static diagnostics carry uncertainty, assumption, and evidence-budget metadata. | `MD3D_13_estimator_reliability_dashboard.png`; `MD3D_15_prior_evidence_budget.png` | `static_metric_uncertainty.csv`; `prior_evidence_budget.csv`; `metric_assumption_registry.csv` | bound validity and sample-size checks | prototype proxy bounds, not operational guarantees |",
            "| alias and threshold redundancy | Redundancy includes semantic aliases, affine transforms, threshold subsumption, observability floors, and decision redundancy. | `MD3D_19_threshold_subsumption_map.png`; `MD3D_21_decision_redundancy_matrix.png` | `feature_alias_candidates.csv`; `feature_threshold_subsumption.csv`; `feature_decision_redundancy.csv` | threshold and decision-redundancy checks | prototype alias heuristics over the declared schema and samples |",
        ]
    ) + "\n"


def _render_md3d_automated_brief(
    bundle_rows: list[dict[str, object]],
    decision_confidence: str,
    confidence_limiters: tuple[str, ...],
) -> str:
    lines = [
        "# Epic 1 3D Multi-Domain Static Admissibility Brief",
        "",
        "This brief asks a narrow question:",
        "",
        "Can the static audit decide whether a proposed 3D feature/class/prior bundle is admissible before corpus search or classifier escalation?",
        "",
        "The bundle is intentionally a normalized feature matrix, not a full 3D tracking implementation.",
        "",
        "## Bundle families",
        "",
    ]
    lines.extend(
        f"- `{row['bundle_id']}` -> expected `{row['expected_route']}` / actual `{row['actual_route']}`"
        for row in bundle_rows
    )
    lines.extend(
        [
            "",
            "## Estimator Reliability and Bounds",
            "",
            f"- decision_confidence: `{decision_confidence}`",
            "- bounds scope: static admissibility diagnostics over the declared feature matrix",
            "- non-claim: not an operational classifier performance guarantee",
            "",
        ]
    )
    lines.extend(f"- confidence_limiter: {item}" for item in confidence_limiters)
    lines.extend(
        [
            "",
            "| Question | Bound / diagnostic | Decision use |",
            "| --- | --- | --- |",
            "| Are class-pair metrics stable? | sample-size proxy intervals, DKW-tagged univariate bands, permutation null summary | promote only if stable |",
            "| Is error unavoidable? | Bayes/Fano/Bhattacharyya proxies, nearest-neighbor oracle proxy | revise class/feature if lower-bound ambiguity stays high |",
            "| Can evidence overcome the prior? | prior evidence budget and flip threshold report | revise prior or add evidence |",
            "| Are sample sizes adequate? | CI width, n per class, additional samples needed | route thin cells to Corpus Explorer |",
            "| Are MI/synergy estimates trustworthy? | sample-sensitive uncertainty rows and candidate-only label | keep synergy as candidate until ablation |",
            "",
            "## Alias, Threshold, and Decision Redundancy",
            "",
            "- The packet also looks for exact duplicates, affine aliases, near-threshold aliases, semantic aliases, and decision redundancy.",
            "- Threshold distinctions are compared against declared measurement resolution and expected uncertainty before the study keeps both features.",
            "- A pair such as `min_altitude_ge_300m` vs `min_altitude_ge_301m` is collapsed when the boundary slice is tiny and the threshold gap falls below the observability floor.",
            "",
            "## Closing decision",
            "",
            "The packet ends with a static decision card and routed next actions across clean, prior-pathological, redundancy-heavy, unobservable, and leakage-blocked bundles.",
            "",
            "## Roadmap",
            "",
            "- Current: static feature/class/prior audit from normalized 3D-inspired features",
            "- Next: compute these features from toy 3D PVA tracklets",
            "- Later: run 3D corpus and classifier/filter ladders on raw track families",
            "",
        ]
    )
    return "\n".join(lines)


def _render_md3d_latex() -> str:
    return "\n".join(
        [
            "\\subsection{3D Multi-Domain Static Admissibility Exemplar}",
            "\\label{subsec:epic1-multidomain-3d}",
            "The multi-domain 3D exemplar is a notional, unclassified study bundle designed to exercise the static audit beyond the one-dimensional witness suite.",
            "\\subsection{Estimator Reliability and Static Bounds}",
            "\\label{subsec:static-estimator-bounds}",
            "The static audit attaches uncertainty and assumption metadata to its metrics. These bounds are not operational performance guarantees; they are admissibility diagnostics over the declared feature matrix, class schema, prior regime, and estimator assumptions.",
            "\\subsection{Alias, Threshold, and Decision Redundancy}",
            "Feature redundancy is not limited to high correlation. In the prototype packet, the audit combines schema-level semantic checks, affine-equivalence tests, near-threshold subsumption tests, observability-aware threshold-gap checks, and decision-conditional information gain proxies.",
            "",
            "\\begin{figure}[ht]",
            "  \\centering",
            "  \\includegraphics[width=\\linewidth]{figures/MD3D_05_class_feature_excitation_matrix.png}",
            "  \\caption{Notional 3D multi-domain class-feature excitation matrix. The matrix is used to test whether the static admissibility stage can identify separable classes, unsupported classes, redundant features, and feature groups that require external context.}",
            "  \\label{fig:md3d-class-feature-excitation}",
            "\\end{figure}",
            "",
            "\\begin{figure}[ht]",
            "  \\centering",
            "  \\includegraphics[width=\\linewidth]{figures/MD3D_13_estimator_reliability_dashboard.png}",
            "  \\caption{Estimator reliability dashboard for the static audit. The dashboard tags point estimates with uncertainty width, sample adequacy, and assumption-sensitive confidence.}",
            "  \\label{fig:md3d-estimator-reliability}",
            "\\end{figure}",
            "",
        ]
    ) + "\n"


def _md3d_hero_rows() -> list[dict[str, str]]:
    return [
        {
            "chart_id": "MD3D_01_bundle_ingestion_spine",
            "role": "main",
            "path": "figures/MD3D_01_bundle_ingestion_spine.png",
            "evidence_tier": "artifact-backed",
            "source_artifact": "source_bundles/",
            "claim": "Portable multidomain bundles can feed Epic 1.",
            "claim_boundary": "ingestion and provenance surface, not a classifier result",
        },
        {
            "chart_id": "MD3D_02_class_surface_map",
            "role": "main",
            "path": "figures/MD3D_02_class_surface_map.png",
            "evidence_tier": "artifact-backed",
            "source_artifact": "source_artifacts/multi_domain_3d_class_schema.csv",
            "claim": "The exemplar spans land, maritime, air, high-dynamic, and space archetypes.",
            "claim_boundary": "notional class taxonomy only",
        },
        {
            "chart_id": "MD3D_03_prior_regime_matrix",
            "role": "main",
            "path": "figures/MD3D_03_prior_regime_matrix.png",
            "evidence_tier": "artifact-backed",
            "source_artifact": "source_artifacts/multi_domain_3d_prior_regimes.csv",
            "claim": "The same class surface can become healthy or pathological under different priors.",
            "claim_boundary": "declared notional prior regimes, not operational prevalence estimates",
        },
        {
            "chart_id": "MD3D_04_feature_taxonomy_observability",
            "role": "main",
            "path": "figures/MD3D_04_feature_taxonomy_observability.png",
            "evidence_tier": "artifact-backed",
            "source_artifact": "source_artifacts/multi_domain_3d_feature_schema.csv",
            "claim": "The feature taxonomy mixes observable, context-dependent, and blocked features.",
            "claim_boundary": "schema-level taxonomy, not empirical performance",
        },
        {
            "chart_id": "MD3D_05_class_feature_excitation_matrix",
            "role": "main",
            "path": "figures/MD3D_05_class_feature_excitation_matrix.png",
            "evidence_tier": "artifact-backed",
            "source_artifact": "source_artifacts/multi_domain_3d_class_feature_signature.csv",
            "claim": "Class-feature excitation patterns reveal separability and unsupported classes.",
            "claim_boundary": "normalized synthetic feature levels, not operational thresholds",
        },
        {
            "chart_id": "MD3D_06_class_confusability_matrix",
            "role": "appendix",
            "path": "figures/MD3D_06_class_confusability_matrix.png",
            "evidence_tier": "artifact-backed",
            "source_artifact": "source_artifacts/multi_domain_3d_expected_confusions.csv",
            "claim": "Expected confusion pairs are declared before classifier escalation.",
            "claim_boundary": "expected confusion design surface, not empirical confusion from a classifier",
        },
        {
            "chart_id": "MD3D_07_prior_pathology_surface",
            "role": "main",
            "path": "figures/MD3D_07_prior_pathology_surface.png",
            "evidence_tier": "run-backed",
            "source_artifact": "source_runs/prior_pathology_multidomain_3d_bundle/prior_pathology_report.csv",
            "claim": "Skewed priors can dominate achievable static evidence.",
            "claim_boundary": "bundle-specific proxy prior pathology surface",
        },
        {
            "chart_id": "MD3D_08_prior_flip_thresholds",
            "role": "appendix",
            "path": "figures/MD3D_08_prior_flip_thresholds.png",
            "evidence_tier": "run-backed",
            "source_artifact": "source_runs/prior_pathology_multidomain_3d_bundle/prior_flip_thresholds.csv",
            "claim": "Prior odds are compared against achievable evidence ranges.",
            "claim_boundary": "bundle-specific threshold intervals",
        },
        {
            "chart_id": "MD3D_09_redundancy_synergy_graph",
            "role": "main",
            "path": "figures/MD3D_09_redundancy_synergy_graph.png",
            "evidence_tier": "run-backed",
            "source_artifact": "source_runs/redundancy_synergy_multidomain_3d_bundle/feature_redundancy_matrix.csv;source_runs/redundancy_synergy_multidomain_3d_bundle/feature_synergy_candidates.csv",
            "claim": "Redundant features and candidate synergy can coexist in one bundle.",
            "claim_boundary": "candidate synergy only; downstream ablation still required",
        },
        {
            "chart_id": "MD3D_10_unobservable_and_leakage_audit",
            "role": "main",
            "path": "figures/MD3D_10_unobservable_and_leakage_audit.png",
            "evidence_tier": "artifact-backed",
            "source_artifact": "source_artifacts/multi_domain_3d_blocked_features.csv;source_artifacts/multi_domain_3d_observability_gaps.csv",
            "claim": "Unsupported classes and blocked features should halt or revise the study before later epics.",
            "claim_boundary": "schema and observability audit, not empirical classifier failure",
        },
        {
            "chart_id": "MD3D_11_static_decision_card",
            "role": "main",
            "path": "figures/MD3D_11_static_decision_card.png",
            "evidence_tier": "run-backed",
            "source_artifact": "source_artifacts/multidomain_bundle_route_matrix.csv",
            "claim": "The five multidomain bundles cover promotion, revision, and blocking routes with explicit confidence labels.",
            "claim_boundary": "bundle-family summary over a notional synthetic study surface",
        },
        {
            "chart_id": "MD3D_12_action_router",
            "role": "appendix",
            "path": "figures/MD3D_12_action_router.png",
            "evidence_tier": "artifact-backed",
            "source_artifact": "source_artifacts/multidomain_bundle_diagnostics.csv",
            "claim": "Static findings route to concrete next actions.",
            "claim_boundary": "routing policy map rather than a learned model",
        },
        {
            "chart_id": "MD3D_13_estimator_reliability_dashboard",
            "role": "main",
            "path": "figures/MD3D_13_estimator_reliability_dashboard.png",
            "evidence_tier": "proxy",
            "source_artifact": "source_artifacts/static_metric_uncertainty.csv;source_artifacts/sample_size_adequacy_report.csv",
            "claim": "Static conclusions carry reliability badges, uncertainty widths, and sample-adequacy tags.",
            "claim_boundary": "prototype uncertainty summaries over the declared feature matrix, not operational guarantees",
        },
        {
            "chart_id": "MD3D_14_pairwise_error_bound_proxy",
            "role": "appendix",
            "path": "figures/MD3D_14_pairwise_error_bound_proxy.png",
            "evidence_tier": "proxy",
            "source_artifact": "source_artifacts/pairwise_error_bound_proxy.csv",
            "claim": "Some class pairs may remain intrinsically ambiguous under the declared feature set.",
            "claim_boundary": "Bayes/Fano/Bhattacharyya outputs remain proxy-level ambiguity diagnostics",
        },
        {
            "chart_id": "MD3D_15_prior_evidence_budget",
            "role": "main",
            "path": "figures/MD3D_15_prior_evidence_budget.png",
            "evidence_tier": "proxy",
            "source_artifact": "source_artifacts/prior_evidence_budget.csv",
            "claim": "Prior odds are compared against achievable feature evidence budgets.",
            "claim_boundary": "posterior-odds budget proxy over declared priors and estimated likelihood-ratio ranges",
        },
        {
            "chart_id": "MD3D_16_sample_size_gap_heatmap",
            "role": "appendix",
            "path": "figures/MD3D_16_sample_size_gap_heatmap.png",
            "evidence_tier": "proxy",
            "source_artifact": "source_artifacts/sample_size_adequacy_report.csv",
            "claim": "Thin cells are routed to Corpus Explorer before classifier claims.",
            "claim_boundary": "sample-size adequacy proxy over the current synthetic bundle counts",
        },
        {
            "chart_id": "MD3D_17_bound_assumption_matrix",
            "role": "appendix",
            "path": "figures/MD3D_17_bound_assumption_matrix.png",
            "evidence_tier": "artifact-backed",
            "source_artifact": "source_artifacts/metric_assumption_registry.csv;source_artifacts/bound_validity_manifest.yaml",
            "claim": "Each bound is tagged with assumptions, evidence tier, and claim boundary.",
            "claim_boundary": "assumption registry only; does not itself validate the metric",
        },
        {
            "chart_id": "MD3D_18_feature_alias_ladder",
            "role": "appendix",
            "path": "figures/MD3D_18_feature_alias_ladder.png",
            "evidence_tier": "proxy",
            "source_artifact": "source_artifacts/feature_alias_candidates.csv",
            "claim": "Redundancy includes aliases, threshold variants, transforms, and semantic near-duplicates.",
            "claim_boundary": "prototype alias taxonomy over schema metadata and declared samples",
        },
        {
            "chart_id": "MD3D_19_threshold_subsumption_map",
            "role": "main",
            "path": "figures/MD3D_19_threshold_subsumption_map.png",
            "evidence_tier": "proxy",
            "source_artifact": "source_artifacts/feature_threshold_subsumption.csv",
            "claim": "Near-threshold features are collapsed unless the boundary slice changes decisions.",
            "claim_boundary": "threshold-subsumption and observability proxy over declared threshold features",
        },
        {
            "chart_id": "MD3D_20_functional_equivalence_scatter",
            "role": "appendix",
            "path": "figures/MD3D_20_functional_equivalence_scatter.png",
            "evidence_tier": "proxy",
            "source_artifact": "source_artifacts/feature_functional_equivalence.csv;source_artifacts/multi_domain_3d_synthetic_samples.csv",
            "claim": "Affine-equivalent features are canonicalized before classifier work.",
            "claim_boundary": "functional-equivalence proxy over the declared synthetic feature bundle",
        },
        {
            "chart_id": "MD3D_21_decision_redundancy_matrix",
            "role": "main",
            "path": "figures/MD3D_21_decision_redundancy_matrix.png",
            "evidence_tier": "proxy",
            "source_artifact": "source_artifacts/feature_decision_redundancy.csv",
            "claim": "A feature is redundant if it adds no class evidence after its neighbor.",
            "claim_boundary": "conditional-gain proxy; pair-specific utility still requires ablation",
        },
    ]
