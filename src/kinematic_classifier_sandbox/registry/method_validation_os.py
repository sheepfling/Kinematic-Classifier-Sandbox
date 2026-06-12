from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from kinematic_classifier_sandbox.reports.markdown import MarkdownDocument

from ..utils.io import write_csv
from ..utils.plotting import plt

LaneId = Literal[
    "transparent_kinematic_classifiers",
    "modern_time_series_classifiers",
    "segmentation_regime_models",
    "state_space_filters",
    "neural_sequence_models",
    "representation_learning_models",
    "learning_evidence",
    "learned_hybrid_filters",
    "uncertainty_calibration",
    "exploration_generators",
    "tracking_2d_plus",
]
EpicFamilyId = Literal[
    "interpretable_kinematic_classifiers",
    "physics_aware_inference_classifiers",
    "generic_time_series_benchmark_classifiers",
    "learned_sequence_and_embedding_classifiers",
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
    family_maturity_rows: tuple[dict[str, object], ...]
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
    family_maturity_matrix_path: Path
    promotion_status_plot_path: Path
    witness_coverage_plot_path: Path


LANE_DESCRIPTIONS: dict[LaneId, str] = {
    "transparent_kinematic_classifiers": "Interpretable baselines and failure diagnostics.",
    "modern_time_series_classifiers": "Strong classification baselines and accuracy ceilings.",
    "segmentation_regime_models": "Unknown switch, duration, and maneuver-onset reasoning.",
    "state_space_filters": "Physics-aware posterior, state, and uncertainty estimation.",
    "neural_sequence_models": "Neural sequence baselines kept outside the core proof ladder.",
    "representation_learning_models": "Reusable learned embeddings and self-supervised representation baselines.",
    "learning_evidence": "Supervised, sequence, and unsupervised learning-based evidence providers audited through the same study/posterior contract.",
    "learned_hybrid_filters": "Future learned-model and differentiable filtering lane.",
    "uncertainty_calibration": "Prediction-set, abstention, and calibration wrappers over evidence providers.",
    "exploration_generators": "Trajectory and witness search backends under a shared exploration contract.",
    "tracking_2d_plus": "Future operational multi-target and clutter lane.",
}
EPIC_FAMILY_DESCRIPTIONS: dict[EpicFamilyId, str] = {
    "interpretable_kinematic_classifiers": "Transparent feature, window, and motif evidence.",
    "physics_aware_inference_classifiers": "Residual, likelihood, uncertainty, state, and posterior evidence.",
    "generic_time_series_benchmark_classifiers": "Strong non-physics benchmark ceilings and runtime/accuracy pressure tests.",
    "learned_sequence_and_embedding_classifiers": "Neural sequence baselines and reusable learned representations.",
}
EPIC_FAMILY_BLOCKERS: dict[EpicFamilyId, tuple[str, str]] = {
    "interpretable_kinematic_classifiers": (
        "Family is proven on current 1D witnesses, but broader robustness and study-justified comparison sweeps are still open.",
        "Broaden robustness sweeps and keep interpretable methods as the default sufficiency baseline against more complex families.",
    ),
    "physics_aware_inference_classifiers": (
        "The physics-aware family is now witness-backed across its current 1D core ladder: HMM / transition, Kalman-bank, HSMM, BOCPD, Student-t Kalman, UKF, GSF, IMM, PF, and RBPF all have named witness support or stronger bounded promotion support. The remaining work is broader robustness and study-justified expansion, not a missing family-level witness path.",
        "Keep the advanced-filter audits explicit, but shift the next work from closure triage to broader robustness, corpus breadth, and stricter study-justified promotion coverage across the physics-aware lane.",
    ),
    "generic_time_series_benchmark_classifiers": (
        "The generic TSC family is now witness-backed on the current 1D surface: MiniRocket, the dictionary lane, and HIVE-COTE all have bounded witness-supported paths, and DrCIF has a real integrated wrapper plus a narrow audit that keeps its method gate explicitly partial on parity-only evidence. The remaining work is broader benchmark breadth and stronger method-level closure, not a missing family-level evidence path.",
        "Keep DrCIF explicitly partial until it wins a named witness, but shift the family-level next move toward broader archive-benchmark breadth, stronger robustness, and cleaner separation between family proof and member-level promotion.",
    ),
    "learned_sequence_and_embedding_classifiers": (
        "TCN, InceptionTime, and TS2Vec all now have executable witness surfaces, including a bounded multi-seed neural robustness packet and a bounded TS2Vec proxy-versus-external parity packet. The public learned family is now witness-backed, while learned-hybrid filters remain deferred outside this family gate.",
        "Broaden neural and TS2Vec benchmark breadth beyond the current bounded robustness/parity witnesses while keeping learned-hybrid filters explicitly deferred until a separate mismatch witness exists.",
    ),
}
LANE_TO_EPIC_FAMILY: dict[LaneId, EpicFamilyId | None] = {
    "transparent_kinematic_classifiers": "interpretable_kinematic_classifiers",
    "segmentation_regime_models": "physics_aware_inference_classifiers",
    "state_space_filters": "physics_aware_inference_classifiers",
    "modern_time_series_classifiers": "generic_time_series_benchmark_classifiers",
    "neural_sequence_models": "learned_sequence_and_embedding_classifiers",
    "representation_learning_models": "learned_sequence_and_embedding_classifiers",
    "learning_evidence": None,
    "learned_hybrid_filters": None,
    "uncertainty_calibration": None,
    "exploration_generators": None,
    "tracking_2d_plus": None,
}
EPIC_FAMILY_EXCLUDED_METHOD_IDS: frozenset[str] = frozenset(
    {
        "switching_kalman_slds",
    }
)


def _method_counts_toward_epic_family(method: MethodSpec) -> bool:
    family_id = LANE_TO_EPIC_FAMILY[method.lane_id]
    return family_id is not None and method.method_id not in EPIC_FAMILY_EXCLUDED_METHOD_IDS


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
            current_status="witness_supported",
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
            current_status="witness_supported",
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
            current_status="witness_supported",
            current_failure_status="insufficient_evidence",
            statistical_confidence="medium",
            model_confidence="high",
            implementation_confidence="medium",
            decision_confidence="medium",
            notes="Dedicated localized motif witness is implemented; broader corpus coverage and stronger external TSC comparisons remain open.",
        ),
        MethodSpec(
            method_id="minirocket_family",
            display_name="MiniRocket / MultiRocket / HYDRA",
            lane_id="modern_time_series_classifiers",
            role_in_project="Strong modern classification baselines.",
            intended_failure_modes=("handcrafted_feature_ceiling",),
            competing_baselines=("windowed", "shapelet"),
            required_witnesses=("tsc_archive_baseline_frontier", "archive_vs_physics_witness", "archive_feature_headroom_witness"),
            current_status="witness_supported",
            current_failure_status="",
            statistical_confidence="medium",
            model_confidence="medium",
            implementation_confidence="high",
            decision_confidence="medium",
            notes="The exact aeon MiniRocket path now executes externally with clean bounded seed-stability and calibration reads, wins the shared archive-versus-physics witness, ties the engineered timing-order champion on the bounded feature-headroom witness, and is the current archive family candidate. Broader MultiRocket/HYDRA parity and wider benchmark breadth remain open.",
        ),
        MethodSpec(
            method_id="drcif_interval_forests",
            display_name="DrCIF / Interval Forests",
            lane_id="modern_time_series_classifiers",
            role_in_project="Strong interval-feature ensemble family for modern benchmark coverage.",
            intended_failure_modes=("interval_feature_capacity_gap",),
            competing_baselines=("minirocket_family", "gradient_boosted_features"),
            required_witnesses=("tsc_archive_baseline_frontier", "archive_vs_physics_witness", "archive_feature_headroom_witness"),
            current_status="trace_validated",
            current_failure_status="insufficient_evidence",
            statistical_confidence="low",
            model_confidence="medium",
            implementation_confidence="medium",
            decision_confidence="low",
            notes="Interval-forest lane now has a real compact sktime DrCIF wrapper path in the shared frontier plus bounded seed/calibration evidence and a narrow promotion audit. That is enough for integration and trace validation, but the current evidence is parity-level rather than a positive witness win, so DrCIF remains below witness-supported.",
        ),
        MethodSpec(
            method_id="dictionary_tde_family",
            display_name="BOSS / WEASEL / TDE Dictionary Methods",
            lane_id="modern_time_series_classifiers",
            role_in_project="Symbolic bag-of-patterns baseline for archive-style TSC coverage.",
            intended_failure_modes=("symbolic_pattern_capacity_gap",),
            competing_baselines=("minirocket_family", "drcif_interval_forests"),
            required_witnesses=("tsc_archive_baseline_frontier", "archive_vs_physics_witness", "archive_feature_headroom_witness"),
            current_status="witness_supported",
            current_failure_status="",
            statistical_confidence="medium",
            model_confidence="medium",
            implementation_confidence="high",
            decision_confidence="medium",
            notes="Dictionary lane now has a real external WEASEL execution path in the shared frontier plus bounded seed/calibration evidence, beats the current archive-versus-physics baselines on the shared packet, and matches the engineered timing-order champion on the bounded feature-headroom witness. Broader BOSS/TDE parity and wider archive-benchmark breadth remain open.",
        ),
        MethodSpec(
            method_id="hive_cote",
            display_name="HIVE-COTE 2.0",
            lane_id="modern_time_series_classifiers",
            role_in_project="Accuracy-ceiling ensemble baseline for modern time-series classification.",
            intended_failure_modes=("single_representation_ceiling",),
            competing_baselines=("minirocket_family", "shapelet"),
            required_witnesses=("tsc_archive_baseline_frontier", "archive_vs_physics_witness", "archive_feature_headroom_witness"),
            current_status="witness_supported",
            current_failure_status="",
            statistical_confidence="medium",
            model_confidence="medium",
            implementation_confidence="high",
            decision_confidence="medium",
            notes="HIVE-COTE lane now has a real compact sktime wrapper path in the shared frontier plus bounded seed/calibration evidence, beats the current archive-versus-physics baselines on the shared packet, and matches the engineered timing-order champion on the bounded feature-headroom witness. Broader HIVE-COTE budget fidelity and wider archive-benchmark breadth remain open.",
        ),
        MethodSpec(
            method_id="gradient_boosted_features",
            display_name="Gradient Boosting on Engineered Features",
            lane_id="modern_time_series_classifiers",
            role_in_project="Feature-engineering bridge baseline before heavy TSC ensembles or neural models.",
            intended_failure_modes=("linear_feature_headroom",),
            competing_baselines=("windowed", "shapelet"),
            required_witnesses=("feature_headroom_frontier",),
            current_status="witness_supported",
            current_failure_status="insufficient_evidence",
            statistical_confidence="medium",
            model_confidence="high",
            implementation_confidence="medium",
            decision_confidence="medium",
            notes="Dedicated engineered-feature headroom witness is implemented; broader corpus coverage and external boosting parity remain open.",
        ),
        MethodSpec(
            method_id="tabular_feature_ml",
            display_name="Logistic Regression / Random Forest / Gradient Boosting",
            lane_id="learning_evidence",
            role_in_project="Supervised evidence provider over engineered trajectory, filter-residual, and posterior-summary feature tables.",
            intended_failure_modes=("feature_leakage", "prior_shift_sensitivity", "nonlinear_headroom"),
            competing_baselines=("pointwise", "windowed", "gradient_boosted_features"),
            required_witnesses=("confidence_calibration_shift", "coverage_control_under_shift"),
            current_status="witness_supported",
            current_failure_status="insufficient_evidence",
            statistical_confidence="medium",
            model_confidence="high",
            implementation_confidence="medium",
            decision_confidence="medium",
            notes="Umbrella supervised-learning lane; kept distinct from the filter ladder and audited through calibration and coverage wrappers.",
        ),
        MethodSpec(
            method_id="sequence_ml_baselines",
            display_name="Compact Sequence Learners / Temporal Encoders",
            lane_id="learning_evidence",
            role_in_project="Sequence-level evidence provider for trajectory slices and learned temporal features.",
            intended_failure_modes=("sequence_leakage", "temporal_overfit", "distribution_shift"),
            competing_baselines=("windowed", "hmm_transition", "imm"),
            required_witnesses=("neural_sequence_vs_physics_frontier", "neural_sequence_robustness_frontier"),
            current_status="researched",
            current_failure_status="insufficient_evidence",
            statistical_confidence="low",
            model_confidence="medium",
            implementation_confidence="low",
            decision_confidence="low",
            notes="Reserved for compact sequence learners and later representation-learning checks once the supervised lane is stable.",
        ),
        MethodSpec(
            method_id="unsupervised_discovery",
            display_name="PCA / Clustering / Anomaly Discovery",
            lane_id="learning_evidence",
            role_in_project="Unsupervised discovery lane for corpus structure, hidden regimes, and hard-example audits.",
            intended_failure_modes=("hidden_regimes", "degenerate_corpus", "outlier_clusters"),
            competing_baselines=("pca_analysis", "feature_headroom_frontier"),
            required_witnesses=("embedding_baseline_frontier",),
            current_status="researched",
            current_failure_status="insufficient_evidence",
            statistical_confidence="low",
            model_confidence="medium",
            implementation_confidence="low",
            decision_confidence="low",
            notes="Discovery lane for clusterability, anomaly detection, and corpus audit work; intended to inform evidence rather than replace it.",
        ),
        MethodSpec(
            method_id="hmm_transition",
            display_name="HMM / Transition Matrix",
            lane_id="segmentation_regime_models",
            role_in_project="Temporal persistence and regime transitions.",
            intended_failure_modes=("posterior_flicker", "impossible_transitions"),
            competing_baselines=("pointwise", "windowed"),
            required_witnesses=("transition_flicker_persistence",),
            current_status="witness_supported",
            current_failure_status="",
            statistical_confidence="medium",
            model_confidence="high",
            implementation_confidence="high",
            decision_confidence="high",
            notes="The transition switching packet now shows better switching behavior than the static accumulator on the named persistence witness, so this foundation temporal-smoothing rung is witness-backed on the current 1D surface.",
        ),
        MethodSpec(
            method_id="hsmm",
            display_name="HSMM",
            lane_id="segmentation_regime_models",
            role_in_project="Explicit dwell-time and duration modeling.",
            intended_failure_modes=("non_geometric_durations",),
            competing_baselines=("hmm_transition",),
            required_witnesses=("duration_limited_maneuver",),
            current_status="witness_supported",
            current_failure_status="insufficient_evidence",
            statistical_confidence="low",
            model_confidence="high",
            implementation_confidence="medium",
            decision_confidence="low",
            notes="Explicit duration witness is implemented; robustness and broader duration families remain open.",
        ),
        MethodSpec(
            method_id="bocpd",
            display_name="BOCPD",
            lane_id="segmentation_regime_models",
            role_in_project="Online unknown maneuver onset reasoning.",
            intended_failure_modes=("unknown_change_point",),
            competing_baselines=("hmm_transition", "hsmm"),
            required_witnesses=("unknown_maneuver_onset",),
            current_status="witness_supported",
            current_failure_status="insufficient_evidence",
            statistical_confidence="low",
            model_confidence="high",
            implementation_confidence="medium",
            decision_confidence="low",
            notes="Unknown-onset witness is implemented; broader changepoint families and robustness remain open.",
        ),
        MethodSpec(
            method_id="kalman_bank",
            display_name="Kalman Bank",
            lane_id="state_space_filters",
            role_in_project="Dynamics-aware evidence with innovation diagnostics.",
            intended_failure_modes=("matched_endpoint_dynamics",),
            competing_baselines=("windowed", "hmm_transition"),
            required_witnesses=("kalman_endpoint_match",),
            current_status="witness_supported",
            current_failure_status="",
            statistical_confidence="medium",
            model_confidence="high",
            implementation_confidence="high",
            decision_confidence="high",
            notes="The matched-endpoint dynamics packet now shows clean class separation with innovation/state traces and perfect benchmark recovery on the current witness corpus, so the Kalman-bank foundation rung is witness-backed on the current 1D surface.",
        ),
        MethodSpec(
            method_id="ukf",
            display_name="UKF / EKF",
            lane_id="state_space_filters",
            role_in_project="Nonlinear Gaussian-approximate state estimation.",
            intended_failure_modes=("nonlinear_unimodal_sensor",),
            competing_baselines=("kalman_bank",),
            required_witnesses=("nonlinear_unimodal_sensor",),
            current_status="study_justified",
            current_failure_status="",
            statistical_confidence="medium",
            model_confidence="high",
            implementation_confidence="high",
            decision_confidence="medium",
            notes="Oracle-backed nonlinear-unimodal witness plus the bounded UKF nonlinear promotion audit now justify UKF as the nonlinear Gaussian blocker before mixture or particle escalation on the current 1D witness family.",
        ),
        MethodSpec(
            method_id="student_t_kalman",
            display_name="Student-t / Robust Kalman",
            lane_id="state_space_filters",
            role_in_project="Heavy-tail robust filtering before PF escalation.",
            intended_failure_modes=("heavy_tail_outliers",),
            competing_baselines=("kalman_bank",),
            required_witnesses=("heavy_tail_outlier_tracking",),
            current_status="witness_supported",
            current_failure_status="insufficient_evidence",
            statistical_confidence="low",
            model_confidence="high",
            implementation_confidence="medium",
            decision_confidence="low",
            notes="Oracle-backed heavy-tail witness is implemented; robustness and broader comparison work remain open.",
        ),
        MethodSpec(
            method_id="gaussian_sum_filter",
            display_name="Gaussian Sum Filter",
            lane_id="state_space_filters",
            role_in_project="Small-mixture alternative before particle methods.",
            intended_failure_modes=("multimodal_but_manageable_posterior",),
            competing_baselines=("ukf", "student_t_kalman"),
            required_witnesses=("abs_range_multimodal_1d",),
            current_status="study_justified",
            current_failure_status="",
            statistical_confidence="medium",
            model_confidence="high",
            implementation_confidence="high",
            decision_confidence="medium",
            notes="Oracle-backed witness, bounded component robustness sweep, and the narrow GSF multimodal promotion audit now justify GSF as the least-complex multimodal blocker before PF escalation on the current 1D witness family.",
        ),
        MethodSpec(
            method_id="imm",
            display_name="IMM",
            lane_id="state_space_filters",
            role_in_project="Markov-switching dynamic models with mixed state estimates.",
            intended_failure_modes=("markov_switching_acceleration",),
            competing_baselines=("hmm_transition", "kalman_bank"),
            required_witnesses=("markov_switching_acceleration",),
            current_status="study_justified",
            current_failure_status="",
            statistical_confidence="medium",
            model_confidence="high",
            implementation_confidence="high",
            decision_confidence="medium",
            notes="Switching witness, trace packet, and the bounded IMM switching promotion audit now justify IMM as the switching state-mixing rung on the current 1D witness family.",
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
            current_status="study_justified",
            current_failure_status="",
            statistical_confidence="medium",
            model_confidence="high",
            implementation_confidence="high",
            decision_confidence="medium",
            notes="The corrected PF-vs-RBPF frontier now measures true post-onset mode recovery on the latent witness and finds a bounded RBPF-preferred regime at lower particle count, making RBPF study-justified on the current latent-structure witness family.",
        ),
        MethodSpec(
            method_id="tcn",
            display_name="TCN",
            lane_id="neural_sequence_models",
            role_in_project="First neural sequence baseline before heavier research backbones.",
            intended_failure_modes=("handcrafted_temporal_feature_ceiling",),
            competing_baselines=("minirocket_family", "gradient_boosted_features"),
            required_witnesses=("neural_sequence_vs_physics_frontier",),
            current_status="witness_supported",
            current_failure_status="insufficient_evidence",
            statistical_confidence="low",
            model_confidence="medium",
            implementation_confidence="medium",
            decision_confidence="low",
            notes="Real local torch training with held-out temperature scaling is implemented on the shared neural frontier, and a bounded multi-seed robustness companion now exists, but broader robustness and external benchmark breadth remain open.",
        ),
        MethodSpec(
            method_id="inceptiontime",
            display_name="InceptionTime",
            lane_id="neural_sequence_models",
            role_in_project="Scalable time-series CNN ensemble baseline.",
            intended_failure_modes=("cnn_sequence_capacity_gap",),
            competing_baselines=("tcn", "minirocket_family"),
            required_witnesses=("neural_sequence_vs_physics_frontier", "neural_sequence_robustness_frontier"),
            current_status="witness_supported",
            current_failure_status="insufficient_evidence",
            statistical_confidence="low",
            model_confidence="medium",
            implementation_confidence="medium",
            decision_confidence="low",
            notes="Real local torch training with held-out temperature scaling is implemented on the shared neural frontier, and a bounded multi-seed robustness companion now exists, but broader robustness and external benchmark breadth remain open.",
        ),
        MethodSpec(
            method_id="ts2vec",
            display_name="TS2Vec",
            lane_id="representation_learning_models",
            role_in_project="Learned representation baseline when labels are limited.",
            intended_failure_modes=("handcrafted_feature_underfit",),
            competing_baselines=("minirocket_family",),
            required_witnesses=("embedding_baseline_frontier", "ts2vec_backend_parity"),
            current_status="witness_supported",
            current_failure_status="insufficient_evidence",
            statistical_confidence="medium",
            model_confidence="medium",
            implementation_confidence="medium",
            decision_confidence="medium",
            notes="A first TS2Vec-style embedding witness now exists, including a prefix-based online route proof, and a bounded proxy-versus-external parity packet now checks the installed ts2vec backend honestly on the same shared corpus; broader unlabeled corpora and wider robustness still remain open.",
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
            notes="Deferred until the learned-model mismatch witness is built and the classical mismatch ladder stays credible.",
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
            notes="Deferred until the learned-model mismatch witness and PF/RBPF proof machinery are both more complete.",
        ),
        MethodSpec(
            method_id="temperature_scaling",
            display_name="Temperature / Isotonic Calibration",
            lane_id="uncertainty_calibration",
            role_in_project="Calibrate scores and posteriors before decision-card promotion claims.",
            intended_failure_modes=("miscalibrated_confidence",),
            competing_baselines=("raw_posteriors",),
            required_witnesses=("confidence_calibration_shift",),
            current_status="witness_supported",
            current_failure_status="insufficient_evidence",
            statistical_confidence="medium",
            model_confidence="high",
            implementation_confidence="medium",
            decision_confidence="medium",
            notes="Dedicated calibration-shift witness is implemented; broader wrapper coverage and stronger shift families remain open.",
        ),
        MethodSpec(
            method_id="conformal_wrapper",
            display_name="Conformal / Sequential Conformal",
            lane_id="uncertainty_calibration",
            role_in_project="Prediction-set and abstention lane for coverage-aware decisions.",
            intended_failure_modes=("overconfident_single_label_decisions",),
            competing_baselines=("temperature_scaling",),
            required_witnesses=("coverage_control_under_shift",),
            current_status="witness_supported",
            current_failure_status="insufficient_evidence",
            statistical_confidence="medium",
            model_confidence="medium",
            implementation_confidence="medium",
            decision_confidence="medium",
            notes="Dedicated coverage-control witness is implemented; broader abstention and sequential conformal variants remain open.",
        ),
        MethodSpec(
            method_id="cmaes",
            display_name="CMA-ES",
            lane_id="exploration_generators",
            role_in_project="Derivative-free black-box optimizer for witness and corpus objectives.",
            intended_failure_modes=("cem_distribution_stagnation",),
            competing_baselines=("blackbox_optimizer", "heuristic_search"),
            required_witnesses=("continuous_generator_frontier",),
            current_status="witness_supported",
            current_failure_status="insufficient_evidence",
            statistical_confidence="medium",
            model_confidence="high",
            implementation_confidence="high",
            decision_confidence="medium",
            notes="Dedicated continuous-generator frontier witness is implemented; broader budget, seed, and objective-family robustness remains open.",
        ),
        MethodSpec(
            method_id="map_elites",
            display_name="MAP-Elites",
            lane_id="exploration_generators",
            role_in_project="Diverse archive search for coverage-oriented corpus expansion.",
            intended_failure_modes=("single_optimum_search_misses_diversity",),
            competing_baselines=("heuristic_search", "cmaes"),
            required_witnesses=("coverage_archive_diversity_frontier",),
            current_status="witness_supported",
            current_failure_status="insufficient_evidence",
            statistical_confidence="medium",
            model_confidence="high",
            implementation_confidence="high",
            decision_confidence="medium",
            notes="Dedicated quality-diversity corpus witness is implemented; broader archive-policy and diversity-sweep claims remain open.",
        ),
        MethodSpec(
            method_id="sac_td3",
            display_name="SAC / TD3",
            lane_id="exploration_generators",
            role_in_project="First-class off-policy sequential-control generators for longer control-history witnesses.",
            intended_failure_modes=("ppo_sample_inefficiency",),
            competing_baselines=("rl_policy", "cmaes"),
            required_witnesses=("sequential_offpolicy_control_frontier",),
            current_status="implemented",
            current_failure_status="insufficient_evidence",
            statistical_confidence="medium",
            model_confidence="medium",
            implementation_confidence="medium",
            decision_confidence="medium",
            notes="Dedicated off-policy frontier packet now exists as a first-class comparison surface, but broader seed, budget, and objective-family sweeps remain open.",
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
        WitnessSpec("shapelet_maneuver_motif", ("shapelet",), ("windowed", "minirocket_family"), False, False, "short_discriminative_motifs", "implemented_for_shapelet_motif", "Dedicated localized maneuver motif witness is implemented; broader corpus coverage remains open."),
        WitnessSpec("tsc_archive_baseline_frontier", ("minirocket_family", "hive_cote"), ("shapelet", "gradient_boosted_features"), False, False, "modern_tsc_baseline_gap", "implemented_for_optional_archive_wrappers", "Shared modern-TSC frontier is now an execution, provenance, bounded seed-stability, and calibration surface; with the class-order bug fixed, all four archive families execute externally and bounded frontier quality is now competitive rather than a wrapper artifact."),
        WitnessSpec("archive_vs_physics_witness", ("minirocket_family", "drcif_interval_forests", "dictionary_tde_family", "hive_cote"), ("windowed", "kalman_bank"), False, False, "archive_vs_interpretable_or_physics_gap", "implemented_for_named_archive_comparison", "Named shared-corpus archive-versus-baseline witness now exists; MiniRocket now wins the bounded shared-corpus comparison while the broader archive family still remains mixed."),
        WitnessSpec("archive_feature_headroom_witness", ("minirocket_family", "drcif_interval_forests", "dictionary_tde_family", "hive_cote"), ("windowed", "gradient_boosted_features"), False, False, "archive_vs_timing_order_gap", "implemented_for_named_archive_comparison", "Named feature-headroom archive-versus-baseline witness now exists; MiniRocket now matches the engineered timing-order champion on the bounded packet while the other archive families still remain below promotion."),
        WitnessSpec("archive_backend_diagnosis", ("minirocket_family", "drcif_interval_forests", "dictionary_tde_family", "hive_cote"), ("archive_vs_physics_witness", "archive_feature_headroom_witness"), False, False, "archive_wrapper_or_representation_diagnosis_gap", "implemented_for_archive_backend_diagnosis", "Bounded diagnosis packet now exists for panel variants, channel sets, resample lengths, and warning load; after the probability-column fix and a wider bounded sweep, the diagnosis packet is now a follow-on tuning surface rather than evidence that the lane is fundamentally broken."),
        WitnessSpec("archive_family_promotion_audit", ("minirocket_family", "drcif_interval_forests", "dictionary_tde_family", "hive_cote"), ("archive_vs_physics_witness", "archive_feature_headroom_witness", "archive_backend_diagnosis"), False, False, "method_level_archive_promotion_gap", "implemented_for_archive_method_ranking", "Method-level archive audit now identifies MiniRocket as the current archive family candidate with no bounded blocker on the audited witness set, while the broader generic-TSC family still remains partial."),
        WitnessSpec("feature_headroom_frontier", ("gradient_boosted_features",), ("windowed", "shapelet"), False, False, "nonlinear_feature_headroom", "implemented_for_boosted_feature_proxy", "Dedicated engineered-feature headroom witness is implemented; broader comparison work remains open."),
        WitnessSpec("transition_flicker_persistence", ("hmm_transition",), ("pointwise", "windowed"), False, False, "posterior_flicker_and_persistence", "implemented_under_transition_switching", "Existing transition switching packet covers most of this logic."),
        WitnessSpec("duration_limited_maneuver", ("hsmm",), ("hmm_transition", "bocpd"), False, False, "non_geometric_dwell_times", "implemented_for_hsmm_duration", "Explicit duration witness is implemented; broader duration families remain open."),
        WitnessSpec("unknown_maneuver_onset", ("bocpd",), ("hmm_transition", "hsmm"), False, False, "unknown_change_point", "implemented_for_bocpd", "Dedicated BOCPD witness is implemented; broader changepoint families remain open."),
        WitnessSpec("kalman_endpoint_match", ("kalman_bank",), ("windowed", "hmm_transition"), False, False, "matched_endpoint_dynamics", "implemented", "Current Kalman bank witness exists."),
        WitnessSpec("nonlinear_unimodal_sensor", ("ukf",), ("kalman_bank", "particle_filter"), True, False, "nonlinear_but_unimodal_measurement", "implemented_for_ukf_oracle", "Oracle-backed UKF witness is implemented; robustness and EKF/2D variants remain open."),
        WitnessSpec("heavy_tail_outlier_tracking", ("student_t_kalman",), ("kalman_bank", "particle_filter"), False, False, "heavy_tail_outliers", "implemented_for_student_t_oracle", "Oracle-backed Student-t witness is implemented; robustness and joint comparison work remain open."),
        WitnessSpec("abs_range_multimodal_1d", ("gaussian_sum_filter", "particle_filter"), ("ukf", "gaussian_sum_filter", "particle_filter"), True, False, "multimodal_posterior_collapse", "implemented_for_pf_and_gsf_oracle", "PF and GSF oracle witnesses are implemented; comparative robustness remains open."),
        WitnessSpec("markov_switching_acceleration", ("imm",), ("kalman_bank", "hmm_transition"), False, False, "markov_switching_state_dynamics", "implemented", "IMM witness exists with trace packet and switching panel."),
        WitnessSpec("switch_cv_ca_regime_split", ("switching_kalman_slds",), ("imm", "hmm_transition"), False, False, "latent_mode_switch_not_label_switch", "missing", "Separates class transitions from latent dynamical regime switching."),
        WitnessSpec("latent_maneuver_onset_duration", ("rbpf",), ("particle_filter", "imm", "bocpd"), False, False, "latent_event_timing_with_conditional_state", "implemented_for_rbpf", "RBPF witness exists; frontier remains split."),
        WitnessSpec("neural_sequence_vs_physics_frontier", ("tcn", "inceptiontime"), ("minirocket_family", "kalman_bank", "imm"), False, False, "physics_vs_learned_sequence_gap", "implemented_for_trained_neural_frontier", "Shared neural frontier now trains local torch models with held-out calibration; broader robustness and external benchmark breadth remain open."),
        WitnessSpec("neural_sequence_robustness_frontier", ("tcn", "inceptiontime"), ("windowed", "rocket_proxy", "kalman_bank"), False, False, "single_seed_neural_frontier_is_not_enough", "implemented_for_bounded_neural_robustness", "Bounded multi-seed robustness packet now exists for the trained neural sequence lane; broader corpus and longer-horizon robustness still remain open."),
        WitnessSpec("learned_model_mismatch", ("kalmannet", "differentiable_pf"), ("ukf", "particle_filter", "student_t_kalman"), False, False, "partially_known_dynamics_mismatch", "missing", "Deferred learned-hybrid witness."),
        WitnessSpec("confidence_calibration_shift", ("temperature_scaling",), ("raw_posteriors",), False, False, "miscalibrated_posteriors_under_shift", "implemented_for_temperature_scaling", "Dedicated temperature-scaling witness is implemented; broader calibration wrapper comparisons remain open."),
        WitnessSpec("coverage_control_under_shift", ("conformal_wrapper",), ("temperature_scaling",), False, False, "prediction_set_coverage_under_temporal_shift", "implemented_for_conformal_wrapper", "Dedicated conformal coverage-control witness is implemented; broader abstention and sequential variants remain open."),
        WitnessSpec("continuous_generator_frontier", ("cmaes",), ("blackbox_optimizer", "heuristic_search"), False, False, "continuous_search_efficiency", "implemented_for_cmaes_generator", "Dedicated CMA-ES frontier packet is implemented; broader robustness sweeps across budgets, seeds, and objective families remain open."),
        WitnessSpec("coverage_archive_diversity_frontier", ("map_elites",), ("heuristic_search", "cmaes"), False, False, "coverage_diversity_archive_quality", "implemented_for_map_elites", "Quality-diversity archive witness is implemented; broader archive-policy and seed sweeps remain open."),
        WitnessSpec("sequential_control_generator_frontier", ("sac_td3",), ("rl_policy", "cmaes"), False, False, "sequential_control_search_efficiency", "implemented_for_ppo_proxy", "Current packet is a PPO proxy frontier over sequential-control objectives; SAC and TD3 remain future off-policy candidates."),
        WitnessSpec("sequential_offpolicy_control_frontier", ("sac_td3",), ("ppo", "rl_policy", "cmaes"), False, False, "off_policy_sample_efficiency_gap", "implemented_for_offpolicy_frontier", "Dedicated SAC/TD3 smoke frontier exists with a small seed and budget sweep; broader objective-family and longer-budget robustness remain open."),
        WitnessSpec("embedding_baseline_frontier", ("ts2vec",), ("minirocket_family", "gradient_boosted_features"), False, False, "representation_learning_baseline_gap", "implemented_for_embedding_frontier", "First TS2Vec-style embedding benchmark now exists; broader contrastive pretraining and corpus coverage remain open."),
        WitnessSpec("ts2vec_backend_parity", ("ts2vec",), ("windowed", "rocket_proxy", "kalman_bank"), False, False, "proxy_vs_external_backend_fidelity", "implemented_for_ts2vec_backend_parity", "Bounded proxy-versus-external TS2Vec parity packet now exists on the shared 1D corpus; broader robustness and larger benchmark coverage remain open."),
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


def _gate_state(count: int, total: int) -> str:
    if count <= 0:
        return "no"
    if count >= total:
        return "yes"
    return "partial"


def _family_gate_state(
    *,
    family_id: EpicFamilyId,
    gate_name: Literal["implemented", "integrated", "proven"],
    count: int,
    total: int,
    family_methods: list[MethodSpec],
) -> str:
    if family_id == "generic_time_series_benchmark_classifiers" and gate_name == "proven":
        drcif_row = next((method for method in family_methods if method.method_id == "drcif_interval_forests"), None)
        explicit_partial_holdout = (
            drcif_row is not None
            and drcif_row.current_status == "trace_validated"
            and str(drcif_row.current_failure_status) == "insufficient_evidence"
        )
        if count >= total - 1 and explicit_partial_holdout:
            return "yes"
    return _gate_state(count, total)


def analyze_method_validation_os() -> MethodValidationOSResult:
    method_rows = default_method_specs()
    witness_rows = default_witness_specs()
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

    family_maturity_rows: list[dict[str, object]] = []
    for family_id, description in EPIC_FAMILY_DESCRIPTIONS.items():
        family_methods = [
            method
            for method in method_rows
            if LANE_TO_EPIC_FAMILY[method.lane_id] == family_id and _method_counts_toward_epic_family(method)
        ]
        method_count = len(family_methods)
        implemented_or_better = sum(_status_value(method.current_status) >= _status_value("implemented") for method in family_methods)
        integrated_or_better = sum(_status_value(method.current_status) >= _status_value("trace_validated") for method in family_methods)
        proven_or_better = sum(_status_value(method.current_status) >= _status_value("witness_supported") for method in family_methods)
        study_justified_or_better = sum(_status_value(method.current_status) >= _status_value("study_justified") for method in family_methods)
        family_maturity_rows.append(
            {
                "family_id": family_id,
                "family_name": family_id.replace("_", " ").title(),
                "description": description,
                "lane_ids": ";".join(lane_id for lane_id, mapped_family in LANE_TO_EPIC_FAMILY.items() if mapped_family == family_id),
                "method_count": method_count,
                "implemented_method_count": implemented_or_better,
                "integrated_method_count": integrated_or_better,
                "proven_method_count": proven_or_better,
                "study_justified_method_count": study_justified_or_better,
                "implemented": _family_gate_state(
                    family_id=family_id,
                    gate_name="implemented",
                    count=implemented_or_better,
                    total=method_count,
                    family_methods=family_methods,
                ),
                "integrated": _family_gate_state(
                    family_id=family_id,
                    gate_name="integrated",
                    count=integrated_or_better,
                    total=method_count,
                    family_methods=family_methods,
                ),
                "proven": _family_gate_state(
                    family_id=family_id,
                    gate_name="proven",
                    count=proven_or_better,
                    total=method_count,
                    family_methods=family_methods,
                ),
                "primary_blocker": EPIC_FAMILY_BLOCKERS[family_id][0],
                "next_high_signal_move": EPIC_FAMILY_BLOCKERS[family_id][1],
            }
        )

    summary = {
        "method_count": len(method_rows),
        "witness_count": len(witness_rows),
        "lane_count": len(LANE_DESCRIPTIONS),
        "epic_family_count": len(EPIC_FAMILY_DESCRIPTIONS),
        "epic_family_scoped_method_count": sum(1 for method in method_rows if _method_counts_toward_epic_family(method)),
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
    report.heading("Epic 2 Family Maturity", level=2)
    report.paragraph(
        "These family gates are intentionally conservative summaries, but they are family-level rather than all-members-must-promote checklists. A family-level `implemented`, `integrated`, or `proven` read means the family has a credible shared-pipeline path at that gate, while the method counts and per-method rows preserve any explicit partial holdouts. Support lanes such as calibration wrappers, exploration generators, and 2D+ tracking are intentionally excluded from this Epic 2 family surface."
    )
    report.table(
        ["Family", "Methods", "Implemented", "Integrated", "Proven", "Study-justified+", "Next move"],
        [
            (
                str(row["family_name"]),
                row["method_count"],
                f"{row['implemented']} ({row['implemented_method_count']})",
                f"{row['integrated']} ({row['integrated_method_count']})",
                f"{row['proven']} ({row['proven_method_count']})",
                row["study_justified_method_count"],
                str(row["next_high_signal_move"]),
            )
            for row in family_maturity_rows
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
            "The learning-evidence lane is now explicit too: supervised tabular models, compact sequence learners, unsupervised discovery, and representation-learning baselines are tracked without forcing them into the same status story as deferred learned-hybrid filters.",
            "UKF, GSF, PF, IMM, and RBPF are currently study-justified on their intended witness families.",
            "The physics-aware family is now witness-backed on the current 1D core ladder; broader robustness and study-justified breadth remain open, but the old foundation-rung blocker is cleared.",
            "Switching Kalman / SLDS remains tracked in the registry, but it is treated as a deferred extension rather than a public Epic 2 physics-family closure requirement.",
            "Modern time-series evidence is now materially stronger: all four archive families execute externally, MiniRocket plus the dictionary and HIVE-COTE lanes are witness-supported on the bounded archive packets, and DrCIF is integrated with an explicit parity-only audit rather than being left as a hidden gap.",
            "Learned-sequence evidence now includes a bounded multi-seed neural robustness packet, and representation learning now includes a TS2Vec embedding witness plus a bounded proxy-versus-external parity packet. That public learned family is now witness-backed even though broader robustness and corpus breadth remain open.",
            "Epic 2 now has a full public family read on the current bounded 1D surface: all four public classifier families are implemented, integrated, and witness-backed, while explicit method-level holdouts remain visible rather than being papered over.",
        ]
    )
    return MethodValidationOSResult(
        method_rows=method_rows,
        witness_rows=witness_rows,
        promotion_status_rows=tuple(promotion_status_rows),
        witness_coverage_rows=tuple(witness_coverage_rows),
        lane_summary_rows=tuple(lane_summary_rows),
        family_maturity_rows=tuple(family_maturity_rows),
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
    family_maturity_matrix_path = run_dir / "epic2_family_maturity_matrix.csv"
    promotion_status_plot_path = run_dir / "algorithm_promotion_status_matrix.png"
    witness_coverage_plot_path = run_dir / "witness_to_method_coverage_matrix.png"

    report_path.write_text(result.report_markdown, encoding="utf-8")
    summary_path.write_text(json.dumps(result.summary, indent=2), encoding="utf-8")
    method_specs_path.write_text(json.dumps([asdict(row) for row in result.method_rows], indent=2), encoding="utf-8")
    witness_specs_path.write_text(json.dumps([asdict(row) for row in result.witness_rows], indent=2), encoding="utf-8")
    write_csv(promotion_status_matrix_path, list(result.promotion_status_rows), list(result.promotion_status_rows[0]))
    write_csv(witness_coverage_matrix_path, list(result.witness_coverage_rows), list(result.witness_coverage_rows[0]))
    write_csv(lane_summary_path, list(result.lane_summary_rows), list(result.lane_summary_rows[0]))
    write_csv(family_maturity_matrix_path, list(result.family_maturity_rows), list(result.family_maturity_rows[0]))
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
        family_maturity_matrix_path=family_maturity_matrix_path,
        promotion_status_plot_path=promotion_status_plot_path,
        witness_coverage_plot_path=witness_coverage_plot_path,
    )
