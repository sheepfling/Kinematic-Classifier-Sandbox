"""Canonical front-door API for the methodology workbench.

This module is intentionally smaller than ``kinematic_classifier_sandbox.__init__``.
It groups the most important study, corpus, evaluation, ladder, and story entry
points behind a stable import surface for new readers and downstream callers.
"""

from __future__ import annotations

from .advanced_filter_decision import analyze_advanced_filter_decision, write_advanced_filter_decision_artifacts
from .advanced_state_inference import analyze_advanced_state_inference, write_advanced_state_inference_artifacts
from .candidate_generation import analyze_candidate_generation, write_candidate_generation_artifacts
from .catalog import METHOD_CATALOG, method_families
from .class_validity import analyze_class_validity, write_class_validity_artifacts
from .common_dataset_comparison import analyze_common_dataset_comparison, render_common_dataset_comparison_report, write_common_dataset_comparison_artifacts
from .corpus_adequacy_audit import analyze_corpus_adequacy, write_corpus_adequacy_artifacts
from .corpus_autodevelopment import analyze_corpus_autodevelopment, render_corpus_autodevelopment_numeric_walkthrough_markdown, write_corpus_autodevelopment_artifacts
from .corpus_classifier_scoring import analyze_corpus_classifier_scoring, write_corpus_classifier_scoring_artifacts
from .corpus_gym import analyze_corpus_gym_contract, write_corpus_gym_artifacts
from .corpus_gym import CorpusGymAction, CorpusGymEnvironment, CorpusGymTarget, default_corpus_gym_targets, render_corpus_gym_numeric_walkthrough_markdown
from .corpus_objectives import analyze_corpus_objectives, default_corpus_objectives, load_corpus_objectives_from_yaml, validate_corpus_objective, write_corpus_objective_artifacts
from .corpus_policy_sweep import write_corpus_policy_tuning_artifacts
from .coverage_report import analyze_coverage_report, write_coverage_report_artifacts
from .feature_analysis import analyze_feature_datasets, write_feature_analysis_artifacts
from .functional_surface_catalog import analyze_functional_surface_catalog, write_functional_surface_catalog_artifacts
from .generic_corpus_exploration import analyze_generic_corpus_exploration, render_generic_corpus_exploration_numeric_walkthrough_markdown, write_generic_corpus_exploration_artifacts
from .generic_classification_evidence_proof import EvidenceStep, analyze_generic_classification_evidence_proof, posterior_history_from_evidence_stream, write_generic_classification_evidence_proof_artifacts
from .generic_inference_contract import analyze_generic_inference_contract, write_generic_inference_contract_artifacts
from .generic_filtering_contract import analyze_generic_filtering_contract, write_generic_filtering_contract_artifacts
from .methodology_compendium import analyze_methodology_compendium, write_methodology_compendium_artifacts
from .methodology_latex import analyze_methodology_latex, write_methodology_latex_artifacts
from .prior_sensitivity_analysis import (
    analyze_cross_method_prior_comparison,
    analyze_pointwise_prior_sensitivity,
    analyze_prior_sensitivity,
    analyze_windowed_prior_sensitivity,
    render_cross_method_prior_comparison_png_bytes,
    render_cross_method_prior_comparison_report,
    render_cross_method_prior_comparison_svg,
    render_prior_sensitivity_decision_png_bytes,
    render_prior_sensitivity_decision_svg,
    render_prior_sensitivity_decomposition_png_bytes,
    render_prior_sensitivity_decomposition_svg,
    render_prior_sensitivity_fragility_png_bytes,
    render_prior_sensitivity_fragility_svg,
    render_prior_sensitivity_flip_png_bytes,
    render_prior_sensitivity_flip_svg,
    render_prior_sensitivity_heatmap_png_bytes,
    render_prior_sensitivity_heatmap_svg,
    render_prior_sensitivity_pairwise_flip_png_bytes,
    render_prior_sensitivity_pairwise_flip_svg,
    render_prior_sensitivity_posterior_png_bytes,
    render_prior_sensitivity_posterior_svg,
    render_prior_sensitivity_report,
    write_cross_method_prior_comparison_artifacts,
    write_prior_sensitivity_artifacts,
)
from .repo_story import write_repo_story_artifacts
from .rung_sufficiency import analyze_rung_sufficiency, write_ladder_witness_suite_artifacts
from .selected_generated_corpus import analyze_selected_generated_corpus, write_selected_generated_corpus_artifacts
from .quality_diversity_corpus import analyze_quality_diversity_corpus, write_quality_diversity_corpus_artifacts
from .sequential_bayes_accumulator import run_accumulator, run_accumulator_benchmark
from .study_candidate_generation import analyze_study_candidate_generation, write_study_candidate_generation_artifacts
from .study_candidate_protocol import analyze_study_candidate_protocol, write_study_candidate_protocol_artifacts
from .trajectory_generator import write_trajectory_generator_artifacts
from .transition_matrix_accumulator import run_transition_benchmark
from .validation_ladder import analyze_validation_ladder, write_validation_ladder_artifacts
from .windowed_baseline import run_windowed_benchmark, write_windowed_benchmark_artifacts

__all__ = [
    "METHOD_CATALOG",
    "analyze_advanced_filter_decision",
    "analyze_advanced_state_inference",
    "analyze_candidate_generation",
    "analyze_class_validity",
    "analyze_common_dataset_comparison",
    "analyze_corpus_autodevelopment",
    "analyze_corpus_classifier_scoring",
    "analyze_corpus_adequacy",
    "analyze_corpus_gym_contract",
    "analyze_corpus_objectives",
    "analyze_coverage_report",
    "analyze_feature_datasets",
    "analyze_functional_surface_catalog",
    "analyze_generic_corpus_exploration",
    "analyze_generic_classification_evidence_proof",
    "analyze_generic_inference_contract",
    "analyze_generic_filtering_contract",
    "analyze_cross_method_prior_comparison",
    "analyze_pointwise_prior_sensitivity",
    "analyze_windowed_prior_sensitivity",
    "analyze_methodology_compendium",
    "analyze_methodology_latex",
    "analyze_prior_sensitivity",
    "analyze_rung_sufficiency",
    "analyze_quality_diversity_corpus",
    "analyze_selected_generated_corpus",
    "analyze_study_candidate_generation",
    "analyze_study_candidate_protocol",
    "analyze_validation_ladder",
    "method_families",
    "CorpusGymAction",
    "CorpusGymEnvironment",
    "CorpusGymTarget",
    "default_corpus_gym_targets",
    "default_corpus_objectives",
    "EvidenceStep",
    "load_corpus_objectives_from_yaml",
    "render_corpus_gym_numeric_walkthrough_markdown",
    "render_corpus_autodevelopment_numeric_walkthrough_markdown",
    "render_common_dataset_comparison_report",
    "render_cross_method_prior_comparison_png_bytes",
    "render_cross_method_prior_comparison_report",
    "render_cross_method_prior_comparison_svg",
    "render_generic_corpus_exploration_numeric_walkthrough_markdown",
    "posterior_history_from_evidence_stream",
    "render_prior_sensitivity_decision_png_bytes",
    "render_prior_sensitivity_decision_svg",
    "render_prior_sensitivity_decomposition_png_bytes",
    "render_prior_sensitivity_decomposition_svg",
    "render_prior_sensitivity_fragility_png_bytes",
    "render_prior_sensitivity_fragility_svg",
    "render_prior_sensitivity_flip_png_bytes",
    "render_prior_sensitivity_flip_svg",
    "render_prior_sensitivity_heatmap_png_bytes",
    "render_prior_sensitivity_heatmap_svg",
    "render_prior_sensitivity_pairwise_flip_png_bytes",
    "render_prior_sensitivity_pairwise_flip_svg",
    "render_prior_sensitivity_posterior_png_bytes",
    "render_prior_sensitivity_posterior_svg",
    "render_prior_sensitivity_report",
    "run_accumulator",
    "run_accumulator_benchmark",
    "run_transition_benchmark",
    "run_windowed_benchmark",
    "validate_corpus_objective",
    "write_advanced_filter_decision_artifacts",
    "write_advanced_state_inference_artifacts",
    "write_common_dataset_comparison_artifacts",
    "write_corpus_autodevelopment_artifacts",
    "write_candidate_generation_artifacts",
    "write_class_validity_artifacts",
    "write_corpus_adequacy_artifacts",
    "write_corpus_gym_artifacts",
    "write_corpus_classifier_scoring_artifacts",
    "write_corpus_objective_artifacts",
    "write_corpus_policy_tuning_artifacts",
    "write_coverage_report_artifacts",
    "write_feature_analysis_artifacts",
    "write_functional_surface_catalog_artifacts",
    "write_generic_corpus_exploration_artifacts",
    "write_generic_classification_evidence_proof_artifacts",
    "write_generic_inference_contract_artifacts",
    "write_generic_filtering_contract_artifacts",
    "write_cross_method_prior_comparison_artifacts",
    "write_ladder_witness_suite_artifacts",
    "write_methodology_compendium_artifacts",
    "write_methodology_latex_artifacts",
    "write_prior_sensitivity_artifacts",
    "write_quality_diversity_corpus_artifacts",
    "write_repo_story_artifacts",
    "write_selected_generated_corpus_artifacts",
    "write_validation_ladder_artifacts",
    "write_study_candidate_generation_artifacts",
    "write_study_candidate_protocol_artifacts",
    "write_trajectory_generator_artifacts",
    "write_windowed_benchmark_artifacts",
]
