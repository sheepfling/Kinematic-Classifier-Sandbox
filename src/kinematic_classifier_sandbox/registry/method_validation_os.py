from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from ..markdown_builder import MarkdownDocument
from ..utils.io import write_csv
from ..utils.plotting import plt

LaneId = Literal[
    "transparent_kinematic_classifiers",
    "modern_time_series_classifiers",
    "segmentation_regime_models",
    "state_space_filters",
    "neural_sequence_models",
    "learned_hybrid_filters",
    "uncertainty_calibration",
    "exploration_generators",
    "tracking_2d_plus",
]
StatusId = Literal[
    "researched",
    "implemented",
    "trace_validated",
    "oracle_validated",
    "witness_supported",
    "study_justified",
    "generalized",
]
FailureStatusId = Literal[
    "missing",
    "blocked",
    "insufficient_evidence",
    "invalid_assumption",
    "fails_oracle",
    "fails_robustness",
    "not_complexity_justified",
]


@dataclass(frozen=True, slots=True)
class MethodSpec:
    method_id: str
    display_name: str
    lane_id: LaneId
    role_in_project: str
    intended_failure_modes: tuple[str, ...]
    competing_baselines: tuple[str, ...]
    required_witnesses: tuple[str, ...]
    current_status: StatusId
    current_failure_status: FailureStatusId | str
    statistical_confidence: str
    model_confidence: str
    implementation_confidence: str
    decision_confidence: str
    notes: str


@dataclass(frozen=True, slots=True)
class WitnessSpec:
    witness_id: str
    designed_to_justify: tuple[str, ...]
    competing_baselines: tuple[str, ...]
    oracle_available: bool
    negative_control: bool
    failure_mode: str
    current_coverage_status: str
    notes: str


@dataclass(frozen=True, slots=True)
class MethodValidationOSResult:
    method_rows: tuple[MethodSpec, ...]
    witness_rows: tuple[WitnessSpec, ...]
    promotion_status_rows: tuple[dict[str, object], ...]
    witness_coverage_rows: tuple[dict[str, object], ...]
    lane_summary_rows: tuple[dict[str, object], ...]
    summary: dict[str, object]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class MethodValidationOSArtifacts:
    run_dir: Path
    report_path: Path
    summary_path: Path
    method_specs_path: Path
    witness_specs_path: Path
    promotion_status_matrix_path: Path
    witness_coverage_matrix_path: Path
    lane_summary_path: Path
    promotion_status_plot_path: Path
    witness_coverage_plot_path: Path


LANE_DESCRIPTIONS: dict[LaneId, str] = {
    "transparent_kinematic_classifiers": "Interpretable baselines and failure diagnostics.",
    "modern_time_series_classifiers": "Strong classification baselines and accuracy ceilings.",
    "segmentation_regime_models": "Unknown switch, duration, and maneuver-onset reasoning.",
    "state_space_filters": "Physics-aware posterior, state, and uncertainty estimation.",
    "neural_sequence_models": "Neural sequence baselines kept outside the core proof ladder.",
    "learned_hybrid_filters": "Future learned-model and differentiable filtering lane.",
    "uncertainty_calibration": "Prediction-set, abstention, and calibration wrappers over evidence providers.",
    "exploration_generators": "Trajectory and witness search backends under a shared exploration contract.",
    "tracking_2d_plus": "Future operational multi-target and clutter lane.",
}


def default_method_specs() -> tuple[MethodSpec, ...]:
    return (
        MethodSpec(
            method_id="pointwise",
            display_name="Pointwise",
            lane_id="transparent_kinematic_classifiers",
            role_in_project="Simplest interpretable posterior baseline.",
            intended_failure_modes=("no_temporal_context",),
            competing_baselines=(),
            required_witnesses=("pointwise_overlap_prior_flip",),
            current_status="implemented",
            current_failure_status="",
            statistical_confidence="medium",
            model_confidence="high",
            implementation_confidence="high",
            decision_confidence="high",
            notes="Canonical local evidence baseline.",
        ),
        MethodSpec(
            method_id="windowed",
            display_name="Windowed / Robust Windowed",
            lane_id="transparent_kinematic_classifiers",
            role_in_project="Local temporal summaries without a full state model.",
            intended_failure_modes=("pointwise_noise", "local_outliers"),
            competing_baselines=("pointwise",),
            required_witnesses=("windowed_outlier_extrema",),
            current_status="implemented",
            current_failure_status="",
            statistical_confidence="medium",
            model_confidence="high",
            implementation_confidence="high",
            decision_confidence="medium",
            notes="Robust windows are implemented; broader witness expansion remains open.",
        ),
        MethodSpec(
            method_id="shapelet",
            display_name="Shapelet / Motif",
            lane_id="transparent_kinematic_classifiers",
            role_in_project="Short discriminative kinematic motifs.",
            intended_failure_modes=("localized_maneuver_signature",),
            competing_baselines=("windowed",),
            required_witnesses=("shapelet_maneuver_motif",),
            current_status="researched",
            current_failure_status="missing",
            statistical_confidence="blocked",
            model_confidence="medium",
            implementation_confidence="blocked",
            decision_confidence="blocked",
            notes="Research lane defined; implementation missing.",
        ),
        MethodSpec(
            method_id="minirocket_family",
            display_name="MiniRocket / MultiRocket / HYDRA",
            lane_id="modern_time_series_classifiers",
            role_in_project="Strong modern classification baselines.",
            intended_failure_modes=("handcrafted_feature_ceiling",),
            competing_baselines=("windowed", "shapelet"),
            required_witnesses=("rocket_baseline_frontier",),
            current_status="researched",
            current_failure_status="missing",
            statistical_confidence="blocked",
            model_confidence="medium",
            implementation_confidence="blocked",
            decision_confidence="blocked",
            notes="Important benchmark lane, not yet wrapped in repo artifacts.",
        ),
        MethodSpec(
            method_id="hive_cote",
            display_name="HIVE-COTE 2.0",
            lane_id="modern_time_series_classifiers",
            role_in_project="Accuracy-ceiling ensemble baseline for modern time-series classification.",
            intended_failure_modes=("single_representation_ceiling",),
            competing_baselines=("minirocket_family", "shapelet"),
            required_witnesses=("tsc_archive_baseline_frontier",),
            current_status="researched",
            current_failure_status="missing",
            statistical_confidence="blocked",
            model_confidence="medium",
            implementation_confidence="blocked",
            decision_confidence="blocked",
            notes="Tracked to show the repo does not ignore strong non-physics ensemble baselines.",
        ),
        MethodSpec(
            method_id="gradient_boosted_features",
            display_name="Gradient Boosting on Engineered Features",
            lane_id="modern_time_series_classifiers",
            role_in_project="Feature-engineering bridge baseline before heavy TSC ensembles or neural models.",
            intended_failure_modes=("linear_feature_headroom",),
            competing_baselines=("windowed", "shapelet"),
            required_witnesses=("feature_headroom_frontier",),
            current_status="researched",
            current_failure_status="missing",
            statistical_confidence="blocked",
            model_confidence="medium",
            implementation_confidence="blocked",
            decision_confidence="blocked",
            notes="Low-risk benchmark addition that stays compatible with current feature surfaces.",
        ),
        MethodSpec(
            method_id="hmm_transition",
            display_name="HMM / Transition Matrix",
            lane_id="segmentation_regime_models",
            role_in_project="Temporal persistence and regime transitions.",
            intended_failure_modes=("posterior_flicker", "impossible_transitions"),
            competing_baselines=("pointwise", "windowed"),
            required_witnesses=("transition_flicker_persistence",),
            current_status="trace_validated",
            current_failure_status="",
            statistical_confidence="medium",
            model_confidence="high",
            implementation_confidence="high",
            decision_confidence="high",
            notes="Shared trace packet and switching witness are implemented.",
        ),
        MethodSpec(
            method_id="hsmm",
            display_name="HSMM",
            lane_id="segmentation_regime_models",
            role_in_project="Explicit dwell-time and duration modeling.",
            intended_failure_modes=("non_geometric_durations",),
            competing_baselines=("hmm_transition",),
            required_witnesses=("duration_limited_maneuver",),
            current_status="researched",
            current_failure_status="missing",
            statistical_confidence="blocked",
            model_confidence="medium",
            implementation_confidence="blocked",
            decision_confidence="blocked",
            notes="Needed before stronger duration-sensitive claims.",
        ),
        MethodSpec(
            method_id="bocpd",
            display_name="BOCPD",
            lane_id="segmentation_regime_models",
            role_in_project="Online unknown maneuver onset reasoning.",
            intended_failure_modes=("unknown_change_point",),
            competing_baselines=("hmm_transition", "hsmm"),
            required_witnesses=("unknown_maneuver_onset",),
            current_status="researched",
            current_failure_status="missing",
            statistical_confidence="blocked",
            model_confidence="medium",
            implementation_confidence="blocked",
            decision_confidence="blocked",
            notes="Research note exists conceptually; artifact lane not built.",
        ),
        MethodSpec(
            method_id="kalman_bank",
            display_name="Kalman Bank",
            lane_id="state_space_filters",
            role_in_project="Dynamics-aware evidence with innovation diagnostics.",
            intended_failure_modes=("matched_endpoint_dynamics",),
            competing_baselines=("windowed", "hmm_transition"),
            required_witnesses=("kalman_endpoint_match",),
            current_status="trace_validated",
            current_failure_status="",
            statistical_confidence="medium",
            model_confidence="high",
            implementation_confidence="high",
            decision_confidence="high",
            notes="Shared state and innovation trace packet is implemented.",
        ),
        MethodSpec(
            method_id="ukf",
            display_name="UKF / EKF",
            lane_id="state_space_filters",
            role_in_project="Nonlinear Gaussian-approximate state estimation.",
            intended_failure_modes=("nonlinear_unimodal_sensor",),
            competing_baselines=("kalman_bank",),
            required_witnesses=("nonlinear_unimodal_sensor",),
            current_status="researched",
            current_failure_status="missing",
            statistical_confidence="blocked",
            model_confidence="medium",
            implementation_confidence="blocked",
            decision_confidence="blocked",
            notes="High-priority blocker before wider PF claims.",
        ),
        MethodSpec(
            method_id="student_t_kalman",
            display_name="Student-t / Robust Kalman",
            lane_id="state_space_filters",
            role_in_project="Heavy-tail robust filtering before PF escalation.",
            intended_failure_modes=("heavy_tail_outliers",),
            competing_baselines=("kalman_bank",),
            required_witnesses=("heavy_tail_outlier_tracking",),
            current_status="researched",
            current_failure_status="missing",
            statistical_confidence="blocked",
            model_confidence="medium",
            implementation_confidence="blocked",
            decision_confidence="blocked",
            notes="High-priority blocker before PF escalation.",
        ),
        MethodSpec(
            method_id="gaussian_sum_filter",
            display_name="Gaussian Sum Filter",
            lane_id="state_space_filters",
            role_in_project="Small-mixture alternative before particle methods.",
            intended_failure_modes=("multimodal_but_manageable_posterior",),
            competing_baselines=("ukf", "student_t_kalman"),
            required_witnesses=("abs_range_multimodal_1d",),
            current_status="witness_supported",
            current_failure_status="insufficient_evidence",
            statistical_confidence="low",
            model_confidence="high",
            implementation_confidence="medium",
            decision_confidence="low",
            notes="Oracle-backed witness is now implemented, but robustness and PF comparison sweeps are still missing.",
        ),
        MethodSpec(
            method_id="imm",
            display_name="IMM",
            lane_id="state_space_filters",
            role_in_project="Markov-switching dynamic models with mixed state estimates.",
            intended_failure_modes=("markov_switching_acceleration",),
            competing_baselines=("hmm_transition", "kalman_bank"),
            required_witnesses=("markov_switching_acceleration",),
            current_status="witness_supported",
            current_failure_status="fails_robustness",
            statistical_confidence="medium",
            model_confidence="high",
            implementation_confidence="high",
            decision_confidence="medium",
            notes="Switching witness and trace packet exist; robustness gate still open.",
        ),
        MethodSpec(
            method_id="switching_kalman_slds",
            display_name="Switching Kalman / SLDS",
            lane_id="state_space_filters",
            role_in_project="Separate label switching from latent dynamic-mode switching.",
            intended_failure_modes=("latent_dynamic_modes_not_class_labels",),
            competing_baselines=("imm", "hmm_transition"),
            required_witnesses=("switch_cv_ca_regime_split",),
            current_status="researched",
            current_failure_status="missing",
            statistical_confidence="blocked",
            model_confidence="medium",
            implementation_confidence="blocked",
            decision_confidence="blocked",
            notes="Useful when IMM-style mixing is needed but the repo must distinguish class transitions from latent modes.",
        ),
        MethodSpec(
            method_id="particle_filter",
            display_name="Particle Filter",
            lane_id="state_space_filters",
            role_in_project="Sequential Monte Carlo for nonlinear, non-Gaussian, multimodal posteriors.",
            intended_failure_modes=("multimodal_posterior_collapse", "non_gaussian_state_evidence"),
            competing_baselines=("ukf", "gaussian_sum_filter", "student_t_kalman"),
            required_witnesses=("abs_range_multimodal_1d", "linear_gaussian_negative_control"),
            current_status="study_justified",
            current_failure_status="",
            statistical_confidence="medium",
            model_confidence="high",
            implementation_confidence="high",
            decision_confidence="medium",
            notes="Currently justified only for the abs-range multimodal oracle family, pending direct robustness comparison against GSF.",
        ),
        MethodSpec(
            method_id="rbpf",
            display_name="RBPF",
            lane_id="state_space_filters",
            role_in_project="Sample hard latent structure and marginalize tractable continuous state.",
            intended_failure_modes=("latent_event_timing_with_conditional_linear_state",),
            competing_baselines=("particle_filter", "imm", "bocpd"),
            required_witnesses=("latent_maneuver_onset_duration",),
            current_status="witness_supported",
            current_failure_status="not_complexity_justified",
            statistical_confidence="medium",
            model_confidence="high",
            implementation_confidence="high",
            decision_confidence="low",
            notes="Witness exists, but compute-normalized frontier is still split against PF.",
        ),
        MethodSpec(
            method_id="tcn",
            display_name="TCN",
            lane_id="neural_sequence_models",
            role_in_project="First neural sequence baseline before heavier research backbones.",
            intended_failure_modes=("handcrafted_temporal_feature_ceiling",),
            competing_baselines=("minirocket_family", "gradient_boosted_features"),
            required_witnesses=("neural_sequence_vs_physics_frontier",),
            current_status="researched",
            current_failure_status="missing",
            statistical_confidence="blocked",
            model_confidence="medium",
            implementation_confidence="blocked",
            decision_confidence="blocked",
            notes="Should answer whether a simple modern sequence learner beats the current physics ladder on matched corpora.",
        ),
        MethodSpec(
            method_id="inceptiontime",
            display_name="InceptionTime",
            lane_id="neural_sequence_models",
            role_in_project="Scalable time-series CNN ensemble baseline.",
            intended_failure_modes=("cnn_sequence_capacity_gap",),
            competing_baselines=("tcn", "minirocket_family"),
            required_witnesses=("neural_sequence_vs_physics_frontier",),
            current_status="researched",
            current_failure_status="missing",
            statistical_confidence="blocked",
            model_confidence="medium",
            implementation_confidence="blocked",
            decision_confidence="blocked",
            notes="Worth tracking as a stronger neural baseline without turning the repo into a deep-learning-first project.",
        ),
        MethodSpec(
            method_id="ts2vec",
            display_name="TS2Vec",
            lane_id="learned_hybrid_filters",
            role_in_project="Learned representation baseline when labels are limited.",
            intended_failure_modes=("handcrafted_feature_underfit",),
            competing_baselines=("minirocket_family",),
            required_witnesses=("embedding_baseline_frontier",),
            current_status="researched",
            current_failure_status="missing",
            statistical_confidence="blocked",
            model_confidence="medium",
            implementation_confidence="blocked",
            decision_confidence="blocked",
            notes="Future learned baseline lane.",
        ),
        MethodSpec(
            method_id="kalmannet",
            display_name="KalmanNet",
            lane_id="learned_hybrid_filters",
            role_in_project="Neural correction over partial model knowledge.",
            intended_failure_modes=("model_mismatch_with_state_structure",),
            competing_baselines=("ukf", "particle_filter"),
            required_witnesses=("learned_model_mismatch",),
            current_status="researched",
            current_failure_status="missing",
            statistical_confidence="blocked",
            model_confidence="medium",
            implementation_confidence="blocked",
            decision_confidence="blocked",
            notes="Deferred until classical mismatch ladder is credible.",
        ),
        MethodSpec(
            method_id="differentiable_pf",
            display_name="Differentiable PF",
            lane_id="learned_hybrid_filters",
            role_in_project="Learned motion or likelihood inside PF structure.",
            intended_failure_modes=("hand_coded_pf_model_bottleneck",),
            competing_baselines=("particle_filter", "kalmannet"),
            required_witnesses=("learned_model_mismatch",),
            current_status="researched",
            current_failure_status="missing",
            statistical_confidence="blocked",
            model_confidence="medium",
            implementation_confidence="blocked",
            decision_confidence="blocked",
            notes="Deferred until PF/RBPF proof machinery is more complete.",
        ),
        MethodSpec(
            method_id="temperature_scaling",
            display_name="Temperature / Isotonic Calibration",
            lane_id="uncertainty_calibration",
            role_in_project="Calibrate scores and posteriors before decision-card promotion claims.",
            intended_failure_modes=("miscalibrated_confidence",),
            competing_baselines=("raw_posteriors",),
            required_witnesses=("confidence_calibration_shift",),
            current_status="researched",
            current_failure_status="missing",
            statistical_confidence="blocked",
            model_confidence="high",
            implementation_confidence="blocked",
            decision_confidence="blocked",
            notes="High-value wrapper lane because it improves decision-card quality without changing the evidence provider.",
        ),
        MethodSpec(
            method_id="conformal_wrapper",
            display_name="Conformal / Sequential Conformal",
            lane_id="uncertainty_calibration",
            role_in_project="Prediction-set and abstention lane for coverage-aware decisions.",
            intended_failure_modes=("overconfident_single_label_decisions",),
            competing_baselines=("temperature_scaling",),
            required_witnesses=("coverage_control_under_shift",),
            current_status="researched",
            current_failure_status="missing",
            statistical_confidence="blocked",
            model_confidence="medium",
            implementation_confidence="blocked",
            decision_confidence="blocked",
            notes="Tracked as the uncertainty lane for classify, defer, or abstain decisions.",
        ),
        MethodSpec(
            method_id="cmaes",
            display_name="CMA-ES",
            lane_id="exploration_generators",
            role_in_project="Derivative-free black-box optimizer for witness and corpus objectives.",
            intended_failure_modes=("cem_distribution_stagnation",),
            competing_baselines=("blackbox_optimizer", "heuristic_search"),
            required_witnesses=("continuous_generator_frontier",),
            current_status="researched",
            current_failure_status="missing",
            statistical_confidence="blocked",
            model_confidence="high",
            implementation_confidence="blocked",
            decision_confidence="blocked",
            notes="Near-term generator addition for continuous trajectory parameters and posterior-target objectives.",
        ),
        MethodSpec(
            method_id="map_elites",
            display_name="MAP-Elites",
            lane_id="exploration_generators",
            role_in_project="Diverse archive search for coverage-oriented corpus expansion.",
            intended_failure_modes=("single_optimum_search_misses_diversity",),
            competing_baselines=("heuristic_search", "cmaes"),
            required_witnesses=("coverage_archive_diversity_frontier",),
            current_status="implemented",
            current_failure_status="",
            statistical_confidence="medium",
            model_confidence="high",
            implementation_confidence="high",
            decision_confidence="medium",
            notes="Current quality-diversity lane for coverage and archive expansion.",
        ),
        MethodSpec(
            method_id="sac_td3",
            display_name="SAC / TD3",
            lane_id="exploration_generators",
            role_in_project="Off-policy sequential-control generators for longer control-history witnesses.",
            intended_failure_modes=("ppo_sample_inefficiency",),
            competing_baselines=("rl_policy", "cmaes"),
            required_witnesses=("sequential_control_generator_frontier",),
            current_status="researched",
            current_failure_status="missing",
            statistical_confidence="blocked",
            model_confidence="medium",
            implementation_confidence="blocked",
            decision_confidence="blocked",
            notes="Tracked as future continuous-control alternatives to PPO-style policy search.",
        ),
        MethodSpec(
            method_id="jpda_mht_rfs",
            display_name="PDA / JPDA / MHT / RFS",
            lane_id="tracking_2d_plus",
            role_in_project="Future 2D+ association, clutter, and multi-target tracking lane.",
            intended_failure_modes=("ambiguous_detections", "clutter", "multi_target_cardinality"),
            competing_baselines=("nearest_neighbor_tracking",),
            required_witnesses=("2d_clutter_association",),
            current_status="researched",
            current_failure_status="missing",
            statistical_confidence="blocked",
            model_confidence="medium",
            implementation_confidence="blocked",
            decision_confidence="blocked",
            notes="Roadmap lane only; not part of current 1D core claims.",
        ),
    )


def default_witness_specs() -> tuple[WitnessSpec, ...]:
    return (
        WitnessSpec("linear_gaussian_negative_control", ("kalman_bank", "ukf", "gaussian_sum_filter", "particle_filter", "rbpf"), ("kalman_bank", "particle_filter"), True, True, "reject_unnecessary_particles", "planned_not_implemented", "Needed to prove the workbench rejects complexity."),
        WitnessSpec("pointwise_overlap_prior_flip", ("pointwise", "hmm_transition"), ("pointwise",), False, False, "prior_sensitivity_under_overlap", "implemented_for_pointwise_family", "Current pointwise prior tooling is close but not yet normalized under this name."),
        WitnessSpec("windowed_outlier_extrema", ("windowed",), ("pointwise",), False, False, "local_outlier_noise", "implemented", "Windowed and robust-windowed lane exists."),
        WitnessSpec("shapelet_maneuver_motif", ("shapelet",), ("windowed", "minirocket_family"), False, False, "short_discriminative_motifs", "missing", "Needed for motif-specific baseline claims."),
        WitnessSpec("tsc_archive_baseline_frontier", ("minirocket_family", "hive_cote"), ("shapelet", "gradient_boosted_features"), False, False, "modern_tsc_baseline_gap", "missing", "Shared witness packet for modern TSC baselines on the same trajectory corpus."),
        WitnessSpec("feature_headroom_frontier", ("gradient_boosted_features",), ("windowed", "shapelet"), False, False, "nonlinear_feature_headroom", "missing", "Needed before claiming engineered-feature learners close the gap to stronger TSC methods."),
        WitnessSpec("transition_flicker_persistence", ("hmm_transition",), ("pointwise", "windowed"), False, False, "posterior_flicker_and_persistence", "implemented_under_transition_switching", "Existing transition switching packet covers most of this logic."),
        WitnessSpec("duration_limited_maneuver", ("hsmm",), ("hmm_transition", "bocpd"), False, False, "non_geometric_dwell_times", "missing", "Explicit duration witness is not yet built."),
        WitnessSpec("unknown_maneuver_onset", ("bocpd",), ("hmm_transition", "hsmm"), False, False, "unknown_change_point", "missing", "Dedicated BOCPD witness is not yet built."),
        WitnessSpec("kalman_endpoint_match", ("kalman_bank",), ("windowed", "hmm_transition"), False, False, "matched_endpoint_dynamics", "implemented", "Current Kalman bank witness exists."),
        WitnessSpec("nonlinear_unimodal_sensor", ("ukf",), ("kalman_bank", "particle_filter"), True, False, "nonlinear_but_unimodal_measurement", "missing", "Needed before broader PF claims."),
        WitnessSpec("heavy_tail_outlier_tracking", ("student_t_kalman",), ("kalman_bank", "particle_filter"), False, False, "heavy_tail_outliers", "missing", "Needed before broader PF claims."),
        WitnessSpec("abs_range_multimodal_1d", ("gaussian_sum_filter", "particle_filter"), ("ukf", "gaussian_sum_filter", "particle_filter"), True, False, "multimodal_posterior_collapse", "implemented_for_pf_and_gsf_oracle", "PF and GSF oracle witnesses are implemented; comparative robustness remains open."),
        WitnessSpec("markov_switching_acceleration", ("imm",), ("kalman_bank", "hmm_transition"), False, False, "markov_switching_state_dynamics", "implemented", "IMM witness exists with trace packet and switching panel."),
        WitnessSpec("switch_cv_ca_regime_split", ("switching_kalman_slds",), ("imm", "hmm_transition"), False, False, "latent_mode_switch_not_label_switch", "missing", "Separates class transitions from latent dynamical regime switching."),
        WitnessSpec("latent_maneuver_onset_duration", ("rbpf",), ("particle_filter", "imm", "bocpd"), False, False, "latent_event_timing_with_conditional_state", "implemented_for_rbpf", "RBPF witness exists; frontier remains split."),
        WitnessSpec("neural_sequence_vs_physics_frontier", ("tcn", "inceptiontime"), ("minirocket_family", "kalman_bank", "imm"), False, False, "physics_vs_learned_sequence_gap", "missing", "Shared benchmark packet to compare neural sequence baselines against physics-aware methods."),
        WitnessSpec("learned_model_mismatch", ("kalmannet", "differentiable_pf"), ("ukf", "particle_filter", "student_t_kalman"), False, False, "partially_known_dynamics_mismatch", "missing", "Deferred learned-hybrid witness."),
        WitnessSpec("confidence_calibration_shift", ("temperature_scaling",), ("raw_posteriors",), False, False, "miscalibrated_posteriors_under_shift", "missing", "Needed before calibrated-confidence claims move beyond prose."),
        WitnessSpec("coverage_control_under_shift", ("conformal_wrapper",), ("temperature_scaling",), False, False, "prediction_set_coverage_under_temporal_shift", "missing", "Needed before abstention and coverage claims are promoted."),
        WitnessSpec("continuous_generator_frontier", ("cmaes",), ("blackbox_optimizer", "heuristic_search"), False, False, "continuous_search_efficiency", "missing", "Benchmark frontier for CEM-style and CMA-ES continuous generators."),
        WitnessSpec("coverage_archive_diversity_frontier", ("map_elites",), ("heuristic_search", "cmaes"), False, False, "coverage_diversity_archive_quality", "implemented_under_qd_archive", "Quality-diversity archive lane already exists and needs only broader evaluation coverage."),
        WitnessSpec("sequential_control_generator_frontier", ("sac_td3",), ("rl_policy", "cmaes"), False, False, "sequential_control_search_efficiency", "missing", "Needed before off-policy generator claims are promoted."),
        WitnessSpec("2d_range_bearing_geometry", ("ukf", "particle_filter"), ("ekf", "ukf"), True, False, "nonlinear_geometric_measurement_2d", "missing", "Future 2D nonlinear lane."),
        WitnessSpec("2d_clutter_association", ("jpda_mht_rfs",), ("nearest_neighbor_tracking", "pda"), False, False, "association_under_clutter", "missing", "Future 2D+ multi-target lane."),
    )


def _status_value(status: StatusId) -> int:
    ordering: tuple[StatusId, ...] = (
        "researched",
        "implemented",
        "trace_validated",
        "oracle_validated",
        "witness_supported",
        "study_justified",
        "generalized",
    )
    return ordering.index(status)


def analyze_method_validation_os() -> MethodValidationOSResult:
    method_rows = default_method_specs()
    witness_rows = default_witness_specs()
    witness_lookup = {witness.witness_id: witness for witness in witness_rows}
    promotion_status_rows: list[dict[str, object]] = []
    for method in method_rows:
        row = {
            "method_id": method.method_id,
            "display_name": method.display_name,
            "lane_id": method.lane_id,
            "current_status": method.current_status,
            "current_failure_status": method.current_failure_status,
            "statistical_confidence": method.statistical_confidence,
            "model_confidence": method.model_confidence,
            "implementation_confidence": method.implementation_confidence,
            "decision_confidence": method.decision_confidence,
        }
        for status in (
            "researched",
            "implemented",
            "trace_validated",
            "oracle_validated",
            "witness_supported",
            "study_justified",
            "generalized",
        ):
            row[status] = "yes" if _status_value(method.current_status) >= _status_value(status) else "no"
        promotion_status_rows.append(row)

    witness_coverage_rows: list[dict[str, object]] = []
    for witness in witness_rows:
        supported_methods = set(witness.designed_to_justify)
        for method in method_rows:
            witness_coverage_rows.append(
                {
                    "witness_id": witness.witness_id,
                    "method_id": method.method_id,
                    "designed_to_justify": "yes" if method.method_id in supported_methods else "no",
                    "method_current_status": method.current_status,
                    "witness_current_coverage_status": witness.current_coverage_status,
                    "oracle_available": "yes" if witness.oracle_available else "no",
                    "negative_control": "yes" if witness.negative_control else "no",
                }
            )

    lane_summary_rows: list[dict[str, object]] = []
    for lane_id, description in LANE_DESCRIPTIONS.items():
        lane_methods = [method for method in method_rows if method.lane_id == lane_id]
        lane_summary_rows.append(
            {
                "lane_id": lane_id,
                "description": description,
                "method_count": len(lane_methods),
                "researched_or_better": sum(_status_value(method.current_status) >= _status_value("researched") for method in lane_methods),
                "implemented_or_better": sum(_status_value(method.current_status) >= _status_value("implemented") for method in lane_methods),
                "witness_supported_or_better": sum(_status_value(method.current_status) >= _status_value("witness_supported") for method in lane_methods),
                "study_justified_or_better": sum(_status_value(method.current_status) >= _status_value("study_justified") for method in lane_methods),
            }
        )

    summary = {
        "method_count": len(method_rows),
        "witness_count": len(witness_rows),
        "lane_count": len(LANE_DESCRIPTIONS),
        "implemented_method_count": sum(_status_value(method.current_status) >= _status_value("implemented") for method in method_rows),
        "witness_supported_method_count": sum(_status_value(method.current_status) >= _status_value("witness_supported") for method in method_rows),
        "study_justified_method_count": sum(_status_value(method.current_status) >= _status_value("study_justified") for method in method_rows),
        "negative_control_count": sum(1 for witness in witness_rows if witness.negative_control),
        "oracle_available_witness_count": sum(1 for witness in witness_rows if witness.oracle_available),
    }

    report = MarkdownDocument("Method Validation Operating System V1")
    report.paragraph(
        "This artifact turns the repo into a method-validation operating system rather than a loose algorithm collection. Each method lane is tracked against a shared status ladder, failure ladder, and witness registry."
    )
    report.heading("Lanes", level=2)
    report.table(
        ["Lane", "Description", "Methods", "Witness-supported+", "Study-justified+"],
        [
            (
                row["lane_id"],
                row["description"],
                row["method_count"],
                row["witness_supported_or_better"],
                row["study_justified_or_better"],
            )
            for row in lane_summary_rows
        ],
    )
    report.heading("Status Model", level=2)
    report.table(
        ["Status", "Meaning"],
        [
            ("researched", "Research note and intended failure mode are defined."),
            ("implemented", "Code exists inside the shared method-validation surface."),
            ("trace_validated", "Required traces and diagnostics are emitted and checked."),
            ("oracle_validated", "Oracle or negative-control checks exist for the method family."),
            ("witness_supported", "A named witness improves and the packet explains why."),
            ("study_justified", "Robustness and complexity gates pass for that witness family."),
            ("generalized", "Reserved for broader evidence across witnesses and corpora."),
        ],
    )
    report.heading("Current Read", level=2)
    report.bullet_list(
        [
            "The transparent, transition, Kalman-bank, and advanced-filter lanes now share an explicit status ladder.",
            "PF is currently study-justified only for the abs-range multimodal oracle family.",
            "IMM and RBPF remain witness-supported rather than study-justified.",
            "UKF, Student-t Kalman, HSMM, BOCPD, and modern time-series wrappers remain the main missing blockers before broader advanced-filter claims.",
        ]
    )
    return MethodValidationOSResult(
        method_rows=method_rows,
        witness_rows=witness_rows,
        promotion_status_rows=tuple(promotion_status_rows),
        witness_coverage_rows=tuple(witness_coverage_rows),
        lane_summary_rows=tuple(lane_summary_rows),
        summary=summary,
        report_markdown=report.text(),
    )


def _write_promotion_status_plot(path: Path, rows: tuple[dict[str, object], ...]) -> None:
    statuses = (
        "researched",
        "implemented",
        "trace_validated",
        "oracle_validated",
        "witness_supported",
        "study_justified",
        "generalized",
    )
    methods = [str(row["display_name"]) for row in rows]
    matrix = [[1.0 if str(row[status]) == "yes" else 0.0 for status in statuses] for row in rows]
    fig, ax = plt.subplots(figsize=(10, max(4, len(methods) * 0.35)), dpi=150)
    image = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_title("Algorithm Promotion Status Matrix")
    ax.set_xlabel("status")
    ax.set_ylabel("method")
    ax.set_xticks(range(len(statuses)), statuses, rotation=30, ha="right")
    ax.set_yticks(range(len(methods)), methods)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="status reached")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_witness_coverage_plot(path: Path, methods: tuple[MethodSpec, ...], witnesses: tuple[WitnessSpec, ...]) -> None:
    matrix = [
        [1.0 if method.method_id in set(witness.designed_to_justify) else 0.0 for method in methods]
        for witness in witnesses
    ]
    fig, ax = plt.subplots(figsize=(10, max(4, len(witnesses) * 0.35)), dpi=150)
    image = ax.imshow(matrix, aspect="auto", cmap="Greens", vmin=0.0, vmax=1.0)
    ax.set_title("Witness To Method Coverage Matrix")
    ax.set_xlabel("method")
    ax.set_ylabel("witness")
    ax.set_xticks(range(len(methods)), [method.display_name for method in methods], rotation=35, ha="right")
    ax.set_yticks(range(len(witnesses)), [witness.witness_id for witness in witnesses])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="designed to justify")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_method_validation_os_artifacts(output_dir: str | Path) -> MethodValidationOSArtifacts:
    result = analyze_method_validation_os()
    run_dir = Path(output_dir) / "method_validation_os_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "method_validation_os_report.md"
    summary_path = run_dir / "summary.json"
    method_specs_path = run_dir / "method_specs.json"
    witness_specs_path = run_dir / "witness_specs.json"
    promotion_status_matrix_path = run_dir / "algorithm_promotion_status_matrix.csv"
    witness_coverage_matrix_path = run_dir / "witness_to_method_coverage_matrix.csv"
    lane_summary_path = run_dir / "lane_summary.csv"
    promotion_status_plot_path = run_dir / "algorithm_promotion_status_matrix.png"
    witness_coverage_plot_path = run_dir / "witness_to_method_coverage_matrix.png"

    report_path.write_text(result.report_markdown, encoding="utf-8")
    summary_path.write_text(json.dumps(result.summary, indent=2), encoding="utf-8")
    method_specs_path.write_text(json.dumps([asdict(row) for row in result.method_rows], indent=2), encoding="utf-8")
    witness_specs_path.write_text(json.dumps([asdict(row) for row in result.witness_rows], indent=2), encoding="utf-8")
    write_csv(promotion_status_matrix_path, list(result.promotion_status_rows), list(result.promotion_status_rows[0]))
    write_csv(witness_coverage_matrix_path, list(result.witness_coverage_rows), list(result.witness_coverage_rows[0]))
    write_csv(lane_summary_path, list(result.lane_summary_rows), list(result.lane_summary_rows[0]))
    _write_promotion_status_plot(promotion_status_plot_path, result.promotion_status_rows)
    _write_witness_coverage_plot(witness_coverage_plot_path, result.method_rows, result.witness_rows)
    return MethodValidationOSArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        summary_path=summary_path,
        method_specs_path=method_specs_path,
        witness_specs_path=witness_specs_path,
        promotion_status_matrix_path=promotion_status_matrix_path,
        witness_coverage_matrix_path=witness_coverage_matrix_path,
        lane_summary_path=lane_summary_path,
        promotion_status_plot_path=promotion_status_plot_path,
        witness_coverage_plot_path=witness_coverage_plot_path,
    )
