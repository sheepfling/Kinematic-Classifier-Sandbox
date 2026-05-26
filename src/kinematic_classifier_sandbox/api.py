"""Canonical front-door API for the methodology workbench.

This module exposes the curated mainline surface only.
"""

from __future__ import annotations

from .analysis.common_dataset_comparison import (
    analyze_common_dataset_comparison,
    write_common_dataset_comparison_artifacts,
)
from .artifacts import render_method_survey_markdown, render_posterior_math_markdown, render_posterior_numeric_walkthrough_markdown, render_posterior_numeric_walkthrough_png_bytes
from .corpus.exploration.backend_adapter_proof import analyze_backend_adapter_proof, write_backend_adapter_proof_artifacts
from .corpus.exploration.candidate_generation import generate_candidates_from_objective_file
from .corpus.exploration.capability_aware_search import analyze_capability_aware_search
from .corpus.search_baseline import analyze_corpus_search_baseline
from .corpus.synthesis_comparison import analyze_corpus_synthesis_comparison
from .corpus.adaptive_stress import analyze_adaptive_stress_corpus, write_adaptive_stress_corpus_artifacts
from .analysis.feature_analysis import (
    BaseFeatureComputationContext,
    FeatureComputationContext,
    OneDimensionalFeatureComputationContext,
    analyze_feature_datasets,
    load_feature_registry,
    load_feature_set_manifest,
    resolve_feature_names,
    write_feature_analysis_artifacts,
)
from .analysis.dimensional_lift_audit import analyze_dimensional_lift_audit, write_dimensional_lift_audit_artifacts
from .corpus.exploration.environment_aware_corpus import analyze_environment_aware_corpus
from .analysis.generated_corpus_features import analyze_generated_corpus_features
from .corpus.exploration.generic_corpus_exploration import analyze_generic_corpus_exploration_weight_sweep
from .analysis.inspection_bundle import recommend_feature_set, recommend_hardest_class_pair, write_abstract_inspection_artifacts
from .analysis.pca_analysis import analyze_feature_pca, write_pca_analysis_artifacts
from .analysis.pca_dimensionality_audit import analyze_pca_dimensionality, write_pca_dimensionality_audit_artifacts
from .analysis.short_horizon_identifiability import (
    analyze_short_horizon_identifiability,
    write_short_horizon_identifiability_artifacts,
)
from .registry.catalog import METHOD_CATALOG, method_families
from .contracts import (
    ClassifierOutputArtifact,
    Milestone0SampleArtifacts,
    TrajectoryArtifact,
    validate_classifier_output_artifact,
    validate_milestone0_sample_run_artifacts,
    validate_trajectory_artifact,
    write_milestone0_sample_run_artifacts,
)
from .common_experiment.config import list_common_studies, load_common_experiment_config, resolve_common_study_adapter
from .common_experiment.runner import analyze_common_experiment, analyze_common_trajectory_corpus
from .common_experiment.artifact_io import write_common_experiment_artifacts
from .corpus.adequacy_audit import analyze_corpus_adequacy, write_corpus_adequacy_artifacts
from .corpus.autodevelopment import analyze_corpus_autodevelopment, write_corpus_autodevelopment_artifacts
from .corpus.classifier_scoring import analyze_corpus_classifier_scoring, write_corpus_classifier_scoring_artifacts
from .corpus.coverage_report import analyze_coverage_report, write_coverage_report_artifacts
from .corpus.gym import CorpusGymAction, CorpusGymEnvironment, CorpusGymTarget, analyze_corpus_gym_contract, write_corpus_gym_artifacts
from .corpus.objectives import analyze_corpus_objectives, default_corpus_objectives, load_corpus_objectives_from_yaml, validate_corpus_objective, write_corpus_objective_artifacts
from .corpus.exploration.objective_corpus_gym_runner import execute_objective_candidates_via_corpus_gym
from .corpus.exploration.objective_driven_qd_archive import analyze_objective_driven_qd_archive
from .corpus.policy_sweep import write_corpus_policy_tuning_artifacts
from .corpus.quality_diversity import analyze_quality_diversity_corpus, write_quality_diversity_corpus_artifacts
from .corpus.selected_generated_corpus import analyze_selected_generated_corpus, write_selected_generated_corpus_artifacts
from .corpus.rl_backend_decision import analyze_rl_backend_decision
from .corpus.trajectory_backend_contract import (
    analyze_trajectory_backend_contract,
    default_backend_contract_definitions,
    validate_backend_contract_definition,
)
from .external_backend_examples import analyze_external_backend_examples
from .corpus.trajectory_backend_contract_rendering import write_trajectory_backend_contract_artifacts
from .external_backend_examples_rendering import write_external_backend_examples_artifacts
from .corpus.exploration.candidate_generation import analyze_candidate_generation, write_candidate_generation_artifacts
from .corpus.exploration.generic_corpus_exploration import analyze_generic_corpus_exploration, write_generic_corpus_exploration_artifacts
from .inference.advanced_state_inference import (
    analyze_advanced_filter_contract,
    analyze_advanced_state_inference,
    write_advanced_filter_contract_artifacts,
    write_advanced_state_inference_artifacts,
)
from .inference.irregular_window_comparison import analyze_irregular_window_comparison, render_irregular_window_report, write_irregular_window_artifacts
from .inference.kalman_filter_bank import KalmanTrajectory, default_kalman_model_specs, render_kalman_bank_png_bytes, render_kalman_bank_report
from .inference.kalman_observable_comparison import analyze_kalman_observable_comparison
from .inference.kalman_variant_comparison import analyze_kalman_variant_comparison
from .inference.monte_carlo_benchmark import render_monte_carlo_accuracy_png_bytes
from .inference.pointwise_baseline import GaussianPointwiseClassifier
from .inference.prior_sensitivity_analysis import (
    analyze_cross_method_prior_comparison,
    analyze_pointwise_prior_sensitivity,
    analyze_prior_sensitivity,
    analyze_windowed_prior_sensitivity,
    write_cross_method_prior_comparison_artifacts,
    write_prior_sensitivity_artifacts,
)
from .inference.sequential_bayes_accumulator import SequentialBayesAccumulator, default_accumulator_class_specs, render_accumulator_png_bytes, render_accumulator_report, run_accumulator, run_accumulator_benchmark
from .inference.transition_matrix_accumulator import (
    run_transition_benchmark,
    write_transition_benchmark_artifacts,
)
from .validation.shared_evaluation import CallableSharedClassifierAdapter
from .inference.velocity_aided_kalman_comparison import (
    analyze_velocity_aided_kalman_comparison,
    render_velocity_aided_kalman_comparison_report,
    write_velocity_aided_kalman_comparison_artifacts,
)
from .inference.windowed_baseline import default_windowed_class_specs, extract_windowed_feature_rows, generate_windowed_trajectories, render_windowed_benchmark_png_bytes, run_windowed_benchmark, write_windowed_benchmark_artifacts
from .methodology.classification_evidence import (
    EvidenceStep,
    analyze_generic_classification_evidence_proof,
    posterior_history_from_evidence_stream,
    write_generic_classification_evidence_proof_artifacts,
)
from .methodology.feature_taxonomy import (
    analyze_generic_feature_taxonomy,
    write_generic_feature_taxonomy_artifacts,
)
from .methodology.filtering_contract import analyze_generic_filtering_contract, write_generic_filtering_contract_artifacts
from .methodology.inference_contract import analyze_generic_inference_contract, write_generic_inference_contract_artifacts
from .rung_sufficiency.analysis import analyze_rung_sufficiency, load_ladder_witness_suite_config, write_ladder_witness_suite_artifacts
from .study_candidate_generation import analyze_study_candidate_generation, write_study_candidate_generation_artifacts
from .study_candidate_protocol import analyze_study_candidate_protocol, write_study_candidate_protocol_artifacts
from .trajectory_generator_rendering import write_trajectory_generator_artifacts
from .trajectory_generator import generate_perturbation_sweep_scenarios
from .validation.advanced_filter_decision import (
    analyze_advanced_filter_decision,
    render_advanced_filter_decision_numeric_walkthrough_markdown,
    render_advanced_filter_decision_report,
    write_advanced_filter_decision_artifacts,
)
from .validation.technique_comparison import (
    analyze_technique_comparison,
    write_technique_comparison_artifacts,
)
from .validation.class_validity import analyze_class_validity, write_class_validity_artifacts
from .validation.validation_ladder import analyze_validation_ladder, write_validation_ladder_artifacts

__all__ = [
    "METHOD_CATALOG",
    "analyze_advanced_filter_decision",
    "analyze_advanced_filter_contract",
    "render_advanced_filter_decision_numeric_walkthrough_markdown",
    "render_advanced_filter_decision_report",
    "analyze_advanced_state_inference",
    "analyze_adaptive_stress_corpus",
    "analyze_backend_adapter_proof",
    "analyze_candidate_generation",
    "analyze_capability_aware_search",
    "analyze_common_experiment",
    "analyze_common_trajectory_corpus",
    "analyze_class_validity",
    "analyze_common_dataset_comparison",
    "analyze_corpus_adequacy",
    "analyze_corpus_autodevelopment",
    "analyze_corpus_classifier_scoring",
    "analyze_corpus_gym_contract",
    "analyze_corpus_objectives",
    "analyze_corpus_search_baseline",
    "analyze_corpus_synthesis_comparison",
    "analyze_external_backend_examples",
    "analyze_coverage_report",
    "analyze_cross_method_prior_comparison",
    "analyze_dimensional_lift_audit",
    "analyze_environment_aware_corpus",
    "analyze_generated_corpus_features",
    "analyze_generic_corpus_exploration_weight_sweep",
    "analyze_feature_datasets",
    "analyze_pca_dimensionality",
    "analyze_feature_pca",
    "analyze_generic_classification_evidence_proof",
    "analyze_generic_corpus_exploration",
    "analyze_generic_feature_taxonomy",
    "analyze_generic_filtering_contract",
    "analyze_generic_inference_contract",
    "analyze_pointwise_prior_sensitivity",
    "analyze_prior_sensitivity",
    "analyze_quality_diversity_corpus",
    "analyze_rung_sufficiency",
    "analyze_selected_generated_corpus",
    "analyze_short_horizon_identifiability",
    "analyze_study_candidate_generation",
    "analyze_study_candidate_protocol",
    "analyze_technique_comparison",
    "analyze_validation_ladder",
    "analyze_trajectory_backend_contract",
    "analyze_windowed_prior_sensitivity",
    "analyze_irregular_window_comparison",
    "render_irregular_window_report",
    "analyze_kalman_observable_comparison",
    "analyze_kalman_variant_comparison",
    "analyze_objective_driven_qd_archive",
    "analyze_feature_pca",
    "analyze_rl_backend_decision",
    "CorpusGymAction",
    "CorpusGymEnvironment",
    "CorpusGymTarget",
    "default_corpus_objectives",
    "default_kalman_model_specs",
    "default_accumulator_class_specs",
    "default_windowed_class_specs",
    "EvidenceStep",
    "BaseFeatureComputationContext",
    "GaussianPointwiseClassifier",
    "ClassifierOutputArtifact",
    "FeatureComputationContext",
    "KalmanTrajectory",
    "SequentialBayesAccumulator",
    "CallableSharedClassifierAdapter",
    "load_feature_registry",
    "load_feature_set_manifest",
    "load_corpus_objectives_from_yaml",
    "load_common_experiment_config",
    "load_ladder_witness_suite_config",
    "method_families",
    "OneDimensionalFeatureComputationContext",
    "render_method_survey_markdown",
    "render_posterior_math_markdown",
    "render_posterior_numeric_walkthrough_markdown",
    "render_posterior_numeric_walkthrough_png_bytes",
    "posterior_history_from_evidence_stream",
    "render_monte_carlo_accuracy_png_bytes",
    "execute_objective_candidates_via_corpus_gym",
    "render_kalman_bank_report",
    "render_accumulator_png_bytes",
    "render_accumulator_report",
    "render_kalman_bank_png_bytes",
    "render_velocity_aided_kalman_comparison_report",
    "resolve_feature_names",
    "run_accumulator",
    "run_accumulator_benchmark",
    "run_transition_benchmark",
    "analyze_velocity_aided_kalman_comparison",
    "run_windowed_benchmark",
    "render_windowed_benchmark_png_bytes",
    "extract_windowed_feature_rows",
    "generate_windowed_trajectories",
    "generate_candidates_from_objective_file",
    "generate_perturbation_sweep_scenarios",
    "recommend_feature_set",
    "recommend_hardest_class_pair",
    "write_abstract_inspection_artifacts",
    "write_backend_adapter_proof_artifacts",
    "write_dimensional_lift_audit_artifacts",
    "write_pca_analysis_artifacts",
    "write_irregular_window_artifacts",
    "Milestone0SampleArtifacts",
    "validate_corpus_objective",
    "validate_classifier_output_artifact",
    "validate_milestone0_sample_run_artifacts",
    "validate_backend_contract_definition",
    "validate_trajectory_artifact",
    "resolve_common_study_adapter",
    "write_advanced_filter_decision_artifacts",
    "write_advanced_filter_contract_artifacts",
    "write_advanced_state_inference_artifacts",
    "write_adaptive_stress_corpus_artifacts",
    "write_candidate_generation_artifacts",
    "write_class_validity_artifacts",
    "write_common_experiment_artifacts",
    "write_common_dataset_comparison_artifacts",
    "write_corpus_adequacy_artifacts",
    "write_corpus_autodevelopment_artifacts",
    "write_corpus_classifier_scoring_artifacts",
    "write_corpus_gym_artifacts",
    "write_corpus_objective_artifacts",
    "write_corpus_policy_tuning_artifacts",
    "write_coverage_report_artifacts",
    "write_cross_method_prior_comparison_artifacts",
    "write_feature_analysis_artifacts",
    "write_pca_dimensionality_audit_artifacts",
    "write_generic_classification_evidence_proof_artifacts",
    "write_generic_corpus_exploration_artifacts",
    "write_generic_feature_taxonomy_artifacts",
    "write_generic_filtering_contract_artifacts",
    "write_generic_inference_contract_artifacts",
    "write_ladder_witness_suite_artifacts",
    "write_prior_sensitivity_artifacts",
    "write_quality_diversity_corpus_artifacts",
    "write_selected_generated_corpus_artifacts",
    "write_study_candidate_generation_artifacts",
    "write_study_candidate_protocol_artifacts",
    "write_short_horizon_identifiability_artifacts",
    "write_milestone0_sample_run_artifacts",
    "write_trajectory_backend_contract_artifacts",
    "write_trajectory_generator_artifacts",
    "write_validation_ladder_artifacts",
    "write_technique_comparison_artifacts",
    "write_transition_benchmark_artifacts",
    "write_velocity_aided_kalman_comparison_artifacts",
    "write_windowed_benchmark_artifacts",
    "write_external_backend_examples_artifacts",
    "list_common_studies",
    "default_backend_contract_definitions",
    "TrajectoryArtifact",
]
