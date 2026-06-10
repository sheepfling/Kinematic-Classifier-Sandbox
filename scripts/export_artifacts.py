from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
existing_pythonpath = os.environ.get("PYTHONPATH")
os.environ["PYTHONPATH"] = (
    str(SRC) if not existing_pythonpath else f"{SRC}{os.pathsep}{existing_pythonpath}"
)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _remove_svg_outputs(root: Path) -> None:
    for path in root.rglob("*.svg"):
        path.unlink(missing_ok=True)

from kinematic_classifier_sandbox.analysis.inspection_bundle import (
    write_abstract_inspection_artifacts,
)
from kinematic_classifier_sandbox.analysis.dimensional_lift_audit import (
    write_dimensional_lift_audit_artifacts,
)
from kinematic_classifier_sandbox.common_experiment.artifact_io import (
    write_common_experiment_artifacts,
)
from kinematic_classifier_sandbox.contracts_rendering import write_milestone0_sample_run_artifacts
from kinematic_classifier_sandbox.corpus.gym import write_corpus_gym_artifacts
from kinematic_classifier_sandbox.corpus.objectives import write_corpus_objective_artifacts
from kinematic_classifier_sandbox.corpus.policy_sweep import write_corpus_policy_tuning_artifacts
from kinematic_classifier_sandbox.inference.advanced_state_inference import write_advanced_filter_contract_artifacts, write_advanced_state_inference_artifacts
from kinematic_classifier_sandbox.inference.pointwise_baseline import (
    run_pointwise_benchmark,
    write_pointwise_benchmark_artifacts,
)
from kinematic_classifier_sandbox.inference.prior_sensitivity_analysis import analyze_pointwise_prior_sensitivity
from kinematic_classifier_sandbox.inference.windowed_baseline import (
    run_windowed_benchmark,
    write_windowed_benchmark_artifacts,
)
from kinematic_classifier_sandbox.trajectory_generator_rendering import write_trajectory_generator_artifacts
from kinematic_classifier_sandbox.validation.advanced_filter_decision import write_advanced_filter_decision_artifacts
from kinematic_classifier_sandbox.validation.class_validity import write_class_validity_artifacts
from kinematic_classifier_sandbox.corpus.trajectory_backend_contract_rendering import write_trajectory_backend_contract_artifacts
from kinematic_classifier_sandbox.validation.technique_comparison_artifact_io import write_technique_comparison_artifacts
from kinematic_classifier_sandbox.validation.validation_ladder import write_validation_ladder_artifacts
from kinematic_classifier_sandbox.analysis.common_dataset_comparison import (
    write_common_dataset_comparison_artifacts,
)
from kinematic_classifier_sandbox.analysis.dimensional_lift_audit import (
    write_dimensional_lift_audit_artifacts,
)
from kinematic_classifier_sandbox.analysis.feature_analysis import write_feature_analysis_artifacts
from kinematic_classifier_sandbox.analysis.generated_corpus_features import (
    write_generated_corpus_feature_artifacts,
)
from kinematic_classifier_sandbox.analysis.pca_analysis import write_pca_analysis_artifacts
from kinematic_classifier_sandbox.analysis.pca_dimensionality_audit import (
    write_pca_dimensionality_audit_artifacts,
)
from kinematic_classifier_sandbox.analysis.short_horizon_identifiability_artifact_io import (
    write_short_horizon_identifiability_artifacts,
)
from kinematic_classifier_sandbox.artifacts import (
    write_method_survey_artifact,
    write_posterior_math_artifacts,
    write_posterior_numeric_walkthrough_artifacts,
    write_probability_primitives_artifacts,
)
from kinematic_classifier_sandbox.corpus.adaptive_stress import (
    write_adaptive_stress_corpus_artifacts,
)
from kinematic_classifier_sandbox.corpus.autodevelopment import (
    write_corpus_autodevelopment_artifacts,
)
from kinematic_classifier_sandbox.corpus.adequacy_audit import (
    write_corpus_adequacy_artifacts,
)
from kinematic_classifier_sandbox.corpus.classifier_scoring import (
    write_corpus_classifier_scoring_artifacts,
)
from kinematic_classifier_sandbox.corpus.exploration.backend_adapter_proof import (
    write_backend_adapter_proof_artifacts,
)
from kinematic_classifier_sandbox.corpus.exploration.candidate_generation import (
    write_candidate_generation_artifacts,
)
from kinematic_classifier_sandbox.corpus.exploration.capability_aware_search import (
    write_capability_aware_search_artifacts,
)
from kinematic_classifier_sandbox.corpus.exploration.environment_aware_corpus import (
    write_environment_aware_corpus_artifacts,
)
from kinematic_classifier_sandbox.corpus.exploration.generic_corpus_exploration import (
    write_generic_corpus_exploration_artifacts,
    write_generic_corpus_exploration_weight_sweep_artifacts,
)
from kinematic_classifier_sandbox.corpus.exploration.objective_driven_qd_archive import (
    write_objective_driven_qd_archive_artifacts,
)
from kinematic_classifier_sandbox.corpus.gym import write_corpus_gym_artifacts
from kinematic_classifier_sandbox.corpus.objectives import write_corpus_objective_artifacts
from kinematic_classifier_sandbox.corpus.policy_sweep import write_corpus_policy_tuning_artifacts
from kinematic_classifier_sandbox.corpus.quality_diversity import (
    write_quality_diversity_corpus_artifacts,
)
from kinematic_classifier_sandbox.corpus.rl_backend_decision import (
    write_rl_backend_decision_artifacts,
)
from kinematic_classifier_sandbox.corpus.search_baseline import (
    write_corpus_search_baseline_artifacts,
)
from kinematic_classifier_sandbox.corpus.synthesis_comparison import (
    write_corpus_synthesis_comparison_artifacts,
)
from kinematic_classifier_sandbox.corpus.trajectory_backend_contract_rendering import (
    write_trajectory_backend_contract_artifacts,
)
from kinematic_classifier_sandbox.corpus.selected_generated_corpus import (
    write_selected_generated_corpus_artifacts,
)
from kinematic_classifier_sandbox.inference.advanced_state_inference import (
    write_advanced_filter_contract_artifacts,
    write_advanced_state_inference_artifacts,
)
from kinematic_classifier_sandbox.inference.irregular_window_comparison import (
    write_irregular_window_artifacts,
)
from kinematic_classifier_sandbox.inference.kalman_filter_bank import write_kalman_bank_artifacts
from kinematic_classifier_sandbox.inference.kalman_observable_comparison import (
    write_kalman_observable_comparison_artifacts,
)
from kinematic_classifier_sandbox.inference.kalman_variant_comparison import (
    write_kalman_variant_comparison_artifacts,
)
from kinematic_classifier_sandbox.inference.monte_carlo_benchmark import (
    run_accumulator_monte_carlo_benchmark,
    write_monte_carlo_artifacts,
)
from kinematic_classifier_sandbox.inference.pointwise_baseline import (
    write_pointwise_benchmark_artifacts,
)
from kinematic_classifier_sandbox.inference.prior_sensitivity.artifact_io import (
    write_prior_sensitivity_artifacts,
)
from kinematic_classifier_sandbox.inference.prior_sensitivity_analysis import (
    analyze_cross_method_prior_comparison,
    analyze_prior_sensitivity,
    analyze_windowed_prior_sensitivity,
    write_cross_method_prior_comparison_artifacts,
)
from kinematic_classifier_sandbox.inference.sequential_bayes_accumulator import (
    run_accumulator_benchmark,
    write_accumulator_artifacts,
)
from kinematic_classifier_sandbox.inference.transition_matrix_accumulator import (
    write_transition_benchmark_artifacts,
)
from kinematic_classifier_sandbox.inference.velocity_aided_kalman.artifact_io import (
    write_velocity_aided_kalman_comparison_artifacts,
)
from kinematic_classifier_sandbox.inference.windowed_baseline import (
    write_windowed_benchmark_artifacts,
)
from kinematic_classifier_sandbox.methodology.classification_evidence import (
    write_generic_classification_evidence_proof_artifacts,
)
from kinematic_classifier_sandbox.methodology.feature_taxonomy import (
    write_generic_feature_taxonomy_artifacts,
)
from kinematic_classifier_sandbox.methodology.filtering_contract import (
    write_generic_filtering_contract_artifacts,
)
from kinematic_classifier_sandbox.methodology.inference_contract import (
    write_generic_inference_contract_artifacts,
)
from kinematic_classifier_sandbox.methodology.latex import (
    write_methodology_latex_artifacts,
    write_methodology_section_symbol_audit_artifacts,
)
from kinematic_classifier_sandbox.rung_sufficiency.analysis import (
    write_ladder_witness_suite_artifacts,
    write_rung_sufficiency_artifacts,
)
from kinematic_classifier_sandbox.showcase.builder import build_showcase_artifacts
from kinematic_classifier_sandbox.story.repo_story import write_repo_story_artifacts
from kinematic_classifier_sandbox.strict_equation_audit import write_strict_equation_audit_artifacts
from kinematic_classifier_sandbox.validation.advanced_filter_decision import (
    write_advanced_filter_decision_artifacts,
)
from kinematic_classifier_sandbox.validation.class_validity import (
    write_class_validity_artifacts,
)
from kinematic_classifier_sandbox.validation.technique_comparison_artifact_io import (
    write_technique_comparison_artifacts,
)
from kinematic_classifier_sandbox.validation.validation_ladder_artifact_io import (
    write_validation_ladder_artifacts,
)
from kinematic_classifier_sandbox.corpus.coverage_report import write_coverage_report_artifacts
from kinematic_classifier_sandbox.corpus.exploration.external_backend_examples_rendering import (
    write_external_backend_examples_artifacts,
)
from kinematic_classifier_sandbox.corpus.quality_diversity import (
    write_quality_diversity_corpus_artifacts,
)
from kinematic_classifier_sandbox.corpus.selected_generated_corpus import (
    write_selected_generated_corpus_artifacts,
)
from kinematic_classifier_sandbox.formal_math_registry import write_formal_math_registry_artifacts
from kinematic_classifier_sandbox.formal_math_visual_registry import (
    write_formal_math_visual_registry_artifacts,
)
from kinematic_classifier_sandbox.functional_surface_catalog import (
    write_functional_surface_catalog_artifacts,
)
from kinematic_classifier_sandbox.methodology.classification_evidence import (
    write_generic_classification_evidence_proof_artifacts,
)
from kinematic_classifier_sandbox.methodology.filtering_contract import (
    write_generic_filtering_contract_artifacts,
)
from kinematic_classifier_sandbox.methodology.inference_contract import (
    write_generic_inference_contract_artifacts,
)
from kinematic_classifier_sandbox.methodology.latex import (
    write_methodology_latex_artifacts,
    write_methodology_section_symbol_audit_artifacts,
)
from kinematic_classifier_sandbox.rung_sufficiency.analysis import (
    write_ladder_witness_suite_artifacts,
)
from kinematic_classifier_sandbox.showcase.builder import build_showcase_artifacts
from kinematic_classifier_sandbox.story.repo_story import write_repo_story_artifacts
from kinematic_classifier_sandbox.strict_equation_audit import write_strict_equation_audit_artifacts
from kinematic_classifier_sandbox.study_candidate_generation import (
    write_study_candidate_generation_artifacts,
)
from kinematic_classifier_sandbox.study_candidate_protocol import (
    write_study_candidate_protocol_artifacts,
)
from kinematic_classifier_sandbox.witnesses.identity_1d.core import (
    run_identity_benchmark,
    write_identity_benchmark_artifacts,
    write_identity_feature_confusion_artifacts,
)
from kinematic_classifier_sandbox.witnesses.identity_1d.posterior_explainer import (
    write_identity_posterior_comparison_artifacts,
    write_identity_posterior_explainer_artifacts,
    write_identity_posterior_failure_artifacts,
    write_identity_posterior_margin_trace_artifacts,
)
from kinematic_classifier_sandbox.witnesses.toy_1d.bayesian_walkthroughs import (
    write_bayesian_walkthrough_artifacts,
)
from kinematic_classifier_sandbox.witnesses.toy_1d.core import (
    render_toy_benchmark_markdown,
    run_toy_benchmark,
    write_toy_benchmark_plot_artifacts,
    write_toy_benchmark_trace_csv,
    write_toy_feature_confusion_artifacts,
)
from kinematic_classifier_sandbox.witnesses.toy_1d.posterior_explainer import (
    write_posterior_comparison_artifacts,
    write_posterior_explainer_artifacts,
    write_posterior_failure_artifacts,
    write_posterior_margin_trace_artifacts,
)


def main() -> int:
    survey_path = write_method_survey_artifact(ROOT / "artifacts")
    milestone0_artifacts = write_milestone0_sample_run_artifacts(ROOT / "artifacts")
    pointwise_result = run_pointwise_benchmark()
    pointwise_artifacts = write_pointwise_benchmark_artifacts(ROOT / "artifacts", result=pointwise_result)
    windowed_result = run_windowed_benchmark()
    windowed_artifacts = write_windowed_benchmark_artifacts(ROOT / "artifacts", result=windowed_result)
    accumulator_result = run_accumulator_benchmark()
    accumulator_artifacts = write_accumulator_artifacts(ROOT / "artifacts", result=accumulator_result)
    monte_carlo_result = run_accumulator_monte_carlo_benchmark(seed=7, trajectories_per_class=12)
    monte_carlo_artifacts = write_monte_carlo_artifacts(ROOT / "artifacts", result=monte_carlo_result)
    kalman_bank_artifacts = write_kalman_bank_artifacts(ROOT / "artifacts")
    kalman_variant_artifacts = write_kalman_variant_comparison_artifacts(ROOT / "artifacts")
    kalman_observable_artifacts = write_kalman_observable_comparison_artifacts(ROOT / "artifacts")
    short_horizon_identifiability_artifacts = write_short_horizon_identifiability_artifacts(ROOT / "artifacts")
    irregular_window_artifacts = write_irregular_window_artifacts(ROOT / "artifacts")
    transition_benchmark_artifacts = write_transition_benchmark_artifacts(ROOT / "artifacts")
    advanced_filter_decision_artifacts = write_advanced_filter_decision_artifacts(ROOT / "artifacts")
    write_advanced_filter_contract_artifacts(ROOT / "artifacts")
    write_advanced_state_inference_artifacts(ROOT / "artifacts")
    velocity_aided_kalman_artifacts = write_velocity_aided_kalman_comparison_artifacts(ROOT / "artifacts")
    prior_sensitivity_result = analyze_prior_sensitivity(seed=7, trajectories_per_class=3)
    prior_sensitivity_artifacts = write_prior_sensitivity_artifacts(ROOT / "artifacts", result=prior_sensitivity_result)
    pointwise_prior_result = analyze_pointwise_prior_sensitivity(seed=7)
    pointwise_prior_artifacts = write_prior_sensitivity_artifacts(ROOT / "artifacts", result=pointwise_prior_result)
    windowed_raw_prior_result = analyze_windowed_prior_sensitivity(seed=7, feature_mode="raw")
    windowed_raw_prior_artifacts = write_prior_sensitivity_artifacts(ROOT / "artifacts", result=windowed_raw_prior_result)
    windowed_robust_prior_result = analyze_windowed_prior_sensitivity(seed=7, feature_mode="robust")
    windowed_robust_prior_artifacts = write_prior_sensitivity_artifacts(ROOT / "artifacts", result=windowed_robust_prior_result)
    cross_method_prior_result = analyze_cross_method_prior_comparison(seed=7)
    cross_method_prior_artifacts = write_cross_method_prior_comparison_artifacts(ROOT / "artifacts", result=cross_method_prior_result)
    technique_comparison_artifacts = write_technique_comparison_artifacts(ROOT / "artifacts")
    common_dataset_comparison_artifacts = write_common_dataset_comparison_artifacts(ROOT / "artifacts")
    common_experiment_artifacts = write_common_experiment_artifacts(ROOT / "artifacts")
    boundary_common_experiment_artifacts = write_common_experiment_artifacts(
        ROOT / "artifacts",
        config_path=ROOT / "experiments" / "common_1d_boundary_study" / "common_experiment_config.yaml",
    )
    generic_inference_contract_artifacts = write_generic_inference_contract_artifacts(ROOT / "artifacts")
    generic_feature_taxonomy_artifacts = write_generic_feature_taxonomy_artifacts(ROOT / "artifacts")
    generic_classification_evidence_proof_artifacts = write_generic_classification_evidence_proof_artifacts(ROOT / "artifacts")
    generic_filtering_contract_artifacts = write_generic_filtering_contract_artifacts(ROOT / "artifacts")
    dimensional_lift_audit_artifacts = write_dimensional_lift_audit_artifacts(ROOT / "artifacts")
    study_candidate_protocol_artifacts = write_study_candidate_protocol_artifacts(ROOT / "artifacts")
    corpus_autodevelopment_artifacts = write_corpus_autodevelopment_artifacts(ROOT / "artifacts")
    corpus_gym_artifacts = write_corpus_gym_artifacts(ROOT / "artifacts")
    trajectory_backend_contract_artifacts = write_trajectory_backend_contract_artifacts(ROOT / "artifacts")
    backend_adapter_proof_artifacts = write_backend_adapter_proof_artifacts(ROOT / "artifacts")
    write_external_backend_examples_artifacts(ROOT / "artifacts")
    environment_aware_corpus_artifacts = write_environment_aware_corpus_artifacts(ROOT / "artifacts")
    capability_aware_search_artifacts = write_capability_aware_search_artifacts(ROOT / "artifacts")
    generic_corpus_exploration_artifacts = write_generic_corpus_exploration_artifacts(ROOT / "artifacts")
    generic_corpus_exploration_weight_sweep_artifacts = write_generic_corpus_exploration_weight_sweep_artifacts(
        ROOT / "artifacts",
        config_path=ROOT / "experiments" / "generic_corpus_exploration_weight_sweep" / "generic_corpus_exploration_weight_sweep.yaml",
    )
    corpus_objective_artifacts = write_corpus_objective_artifacts(ROOT / "artifacts")
    candidate_generation_artifacts = write_candidate_generation_artifacts(ROOT / "artifacts")
    class_validity_artifacts = write_class_validity_artifacts(ROOT / "artifacts")
    generated_corpus_feature_artifacts = write_generated_corpus_feature_artifacts(ROOT / "artifacts")
    corpus_classifier_scoring_artifacts = write_corpus_classifier_scoring_artifacts(ROOT / "artifacts")
    objective_driven_qd_archive_artifacts = write_objective_driven_qd_archive_artifacts(ROOT / "artifacts")
    selected_generated_corpus_artifacts = write_selected_generated_corpus_artifacts(ROOT / "artifacts")
    corpus_search_baseline_artifacts = write_corpus_search_baseline_artifacts(ROOT / "artifacts")
    quality_diversity_corpus_artifacts = write_quality_diversity_corpus_artifacts(ROOT / "artifacts")
    adaptive_stress_corpus_artifacts = write_adaptive_stress_corpus_artifacts(ROOT / "artifacts")
    rl_backend_decision_artifacts = write_rl_backend_decision_artifacts(ROOT / "artifacts")
    corpus_synthesis_comparison_artifacts = write_corpus_synthesis_comparison_artifacts(ROOT / "artifacts")
    study_candidate_generation_artifacts = write_study_candidate_generation_artifacts(ROOT / "artifacts")
    validation_ladder_artifacts = write_validation_ladder_artifacts(ROOT / "artifacts")
    write_rung_sufficiency_artifacts(ROOT / "artifacts")
    ladder_witness_suite_artifacts = write_ladder_witness_suite_artifacts(
        ROOT / "artifacts",
        config_path=ROOT / "experiments" / "ladder_witness_suite" / "ladder_witness_suite.yaml",
    )
    bayesian_walkthrough_artifacts = write_bayesian_walkthrough_artifacts(ROOT / "artifacts")
    trajectory_generator_artifacts = write_trajectory_generator_artifacts(ROOT / "artifacts")
    abstract_inspection_artifacts = write_abstract_inspection_artifacts(
        ROOT / "artifacts",
        seed=7,
        trajectories_per_class=5,
        n_components=3,
    )
    feature_analysis_artifacts = write_feature_analysis_artifacts(ROOT / "artifacts", seed=7, trajectories_per_class=5)
    corpus_adequacy_artifacts = write_corpus_adequacy_artifacts(ROOT / "artifacts", seed=7, trajectories_per_class=5)
    coverage_report_artifacts = write_coverage_report_artifacts(ROOT / "artifacts", seed=7, trajectories_per_class=5)
    pca_analysis_artifacts = write_pca_analysis_artifacts(ROOT / "artifacts", seed=7, trajectories_per_class=5, n_components=3)
    methodology_latex_artifacts = write_methodology_latex_artifacts(ROOT / "artifacts")
    methodology_section_symbol_audit_artifacts = write_methodology_section_symbol_audit_artifacts(ROOT / "artifacts")
    functional_surface_catalog_artifacts = write_functional_surface_catalog_artifacts(ROOT / "artifacts")
    formal_math_registry_artifacts = write_formal_math_registry_artifacts(ROOT / "artifacts")
    formal_math_visual_registry_artifacts = write_formal_math_visual_registry_artifacts(ROOT / "artifacts")
    strict_equation_audit_artifacts = write_strict_equation_audit_artifacts(ROOT / "artifacts")
    pca_dimensionality_artifacts = write_pca_dimensionality_audit_artifacts(
        ROOT / "artifacts",
        seed=7,
        trajectories_per_class=5,
        max_components=5,
    )
    showcase_artifacts = build_showcase_artifacts(ROOT / "artifacts", refresh=False, create_zip=False)
    repo_story_artifacts = write_repo_story_artifacts(ROOT / "artifacts", docs_root=ROOT / "docs", write_showcase=True)
    posterior_math_artifacts = write_posterior_math_artifacts(
        ROOT / "artifacts"
    )
    posterior_math_markdown_path = posterior_math_artifacts.markdown_path
    posterior_math_png_path = posterior_math_artifacts.png_path
    probability_primitives_artifacts = write_probability_primitives_artifacts(
        ROOT / "artifacts"
    )
    probability_primitives_markdown_path = probability_primitives_artifacts.markdown_path
    probability_primitives_png_path = probability_primitives_artifacts.png_path
    posterior_numeric_artifacts = write_posterior_numeric_walkthrough_artifacts(
        ROOT / "artifacts"
    )
    posterior_numeric_markdown_path = posterior_numeric_artifacts.markdown_path
    posterior_numeric_png_path = posterior_numeric_artifacts.png_path
    identity_result = run_identity_benchmark()
    identity_artifacts = write_identity_benchmark_artifacts(
        ROOT / "artifacts",
        result=identity_result,
    )
    identity_markdown_path = identity_artifacts.summary_path
    identity_png_path = identity_artifacts.plot_path
    identity_csv_path = identity_artifacts.trace_path
    identity_feature_confusion_png_path = write_identity_feature_confusion_artifacts(
        ROOT / "artifacts",
        result=identity_result,
    )
    identity_posterior_markdown_path, identity_posterior_png_path = write_identity_posterior_explainer_artifacts(
        ROOT / "artifacts",
        result=identity_result,
    )
    identity_posterior_failure_markdown_path, identity_posterior_failure_png_path = write_identity_posterior_failure_artifacts(
        ROOT / "artifacts",
        result=identity_result,
    )
    identity_posterior_comparison_markdown_path, identity_posterior_comparison_png_path = write_identity_posterior_comparison_artifacts(
        ROOT / "artifacts",
        result=identity_result,
    )
    identity_posterior_margin_markdown_path, identity_posterior_margin_png_path = write_identity_posterior_margin_trace_artifacts(
        ROOT / "artifacts",
        result=identity_result,
    )
    result = run_toy_benchmark()
    benchmark_path = ROOT / "artifacts" / "toy_1d_benchmark_summary.md"
    benchmark_plot_png_path = write_toy_benchmark_plot_artifacts(
        ROOT / "artifacts",
        result=result,
    )
    toy_feature_confusion_png_path = write_toy_feature_confusion_artifacts(
        ROOT / "artifacts",
        result=result,
    )
    benchmark_trace_csv = write_toy_benchmark_trace_csv(result, ROOT / "artifacts")
    posterior_artifacts = write_posterior_explainer_artifacts(
        ROOT / "artifacts",
        result=result,
    )
    posterior_markdown_path = posterior_artifacts.markdown_path
    posterior_png_path = posterior_artifacts.png_path
    posterior_failure_artifacts = write_posterior_failure_artifacts(
        ROOT / "artifacts",
        result=result,
    )
    posterior_failure_markdown_path = posterior_failure_artifacts.markdown_path
    posterior_failure_png_path = posterior_failure_artifacts.png_path
    posterior_comparison_artifacts = write_posterior_comparison_artifacts(
        ROOT / "artifacts",
        result=result,
    )
    posterior_comparison_markdown_path = posterior_comparison_artifacts.markdown_path
    posterior_comparison_png_path = posterior_comparison_artifacts.png_path
    posterior_margin_artifacts = write_posterior_margin_trace_artifacts(
        ROOT / "artifacts",
        result=result,
    )
    posterior_margin_markdown_path = posterior_margin_artifacts.markdown_path
    posterior_margin_png_path = posterior_margin_artifacts.png_path
    benchmark_path.write_text(render_toy_benchmark_markdown(result), encoding="utf-8")
    _remove_svg_outputs(ROOT / "artifacts")
    print(survey_path)
    print(milestone0_artifacts.run_dir)
    print(milestone0_artifacts.report_path)
    print(pointwise_artifacts.run_dir)
    print(pointwise_artifacts.report_path)
    print(windowed_artifacts.run_dir)
    print(windowed_artifacts.report_path)
    print(accumulator_artifacts.run_dir)
    print(accumulator_artifacts.report_path)
    print(monte_carlo_artifacts.run_dir)
    print(monte_carlo_artifacts.report_path)
    print(kalman_bank_artifacts.run_dir)
    print(kalman_bank_artifacts.report_path)
    print(kalman_variant_artifacts.run_dir)
    print(kalman_variant_artifacts.report_path)
    print(kalman_observable_artifacts.run_dir)
    print(kalman_observable_artifacts.report_path)
    print(short_horizon_identifiability_artifacts.run_dir)
    print(short_horizon_identifiability_artifacts.report_path)
    print(irregular_window_artifacts.run_dir)
    print(irregular_window_artifacts.report_path)
    print(transition_benchmark_artifacts.run_dir)
    print(transition_benchmark_artifacts.report_path)
    print(advanced_filter_decision_artifacts.run_dir)
    print(advanced_filter_decision_artifacts.report_path)
    print(velocity_aided_kalman_artifacts.run_dir)
    print(velocity_aided_kalman_artifacts.report_path)
    print(prior_sensitivity_artifacts.run_dir)
    print(prior_sensitivity_artifacts.report_path)
    print(pointwise_prior_artifacts.run_dir)
    print(pointwise_prior_artifacts.report_path)
    print(windowed_raw_prior_artifacts.run_dir)
    print(windowed_raw_prior_artifacts.report_path)
    print(windowed_robust_prior_artifacts.run_dir)
    print(windowed_robust_prior_artifacts.report_path)
    print(cross_method_prior_artifacts.run_dir)
    print(cross_method_prior_artifacts.report_path)
    print(technique_comparison_artifacts.run_dir)
    print(technique_comparison_artifacts.report_path)
    print(common_dataset_comparison_artifacts.run_dir)
    print(common_dataset_comparison_artifacts.report_path)
    print(common_experiment_artifacts.run_dir)
    print(common_experiment_artifacts.report_path)
    print(boundary_common_experiment_artifacts.run_dir)
    print(boundary_common_experiment_artifacts.report_path)
    print(generic_inference_contract_artifacts.run_dir)
    print(generic_inference_contract_artifacts.report_path)
    print(generic_feature_taxonomy_artifacts.run_dir)
    print(generic_feature_taxonomy_artifacts.transfer_report_path)
    print(generic_classification_evidence_proof_artifacts.run_dir)
    print(generic_classification_evidence_proof_artifacts.classification_principles_report_path)
    print(generic_filtering_contract_artifacts.run_dir)
    print(generic_filtering_contract_artifacts.filtering_principles_report_path)
    print(dimensional_lift_audit_artifacts.run_dir)
    print(dimensional_lift_audit_artifacts.audit_report_path)
    print(study_candidate_protocol_artifacts.run_dir)
    print(study_candidate_protocol_artifacts.protocol_path)
    print(study_candidate_protocol_artifacts.validation_ladder_schema_path)
    print(corpus_autodevelopment_artifacts.run_dir)
    print(corpus_autodevelopment_artifacts.report_path)
    print(corpus_gym_artifacts.run_dir)
    print(corpus_gym_artifacts.report_path)
    print(trajectory_backend_contract_artifacts.run_dir)
    print(trajectory_backend_contract_artifacts.report_path)
    print(backend_adapter_proof_artifacts.run_dir)
    print(backend_adapter_proof_artifacts.backend_output_equivalence_report_path)
    print(environment_aware_corpus_artifacts.run_dir)
    print(environment_aware_corpus_artifacts.report_path)
    print(capability_aware_search_artifacts.run_dir)
    print(capability_aware_search_artifacts.report_path)
    print(generic_corpus_exploration_artifacts.run_dir)
    print(generic_corpus_exploration_artifacts.report_path)
    print(generic_corpus_exploration_weight_sweep_artifacts.run_dir)
    print(generic_corpus_exploration_weight_sweep_artifacts.config_path)
    print(generic_corpus_exploration_weight_sweep_artifacts.report_path)
    print(generic_corpus_exploration_weight_sweep_artifacts.summary_path)
    print(generic_corpus_exploration_weight_sweep_artifacts.rows_path)
    print(generic_corpus_exploration_weight_sweep_artifacts.overlap_matrix_path)
    print(generic_corpus_exploration_weight_sweep_artifacts.weight_matrix_path)
    print(generic_corpus_exploration_weight_sweep_artifacts.tradeoff_png_path)
    print(generic_corpus_exploration_weight_sweep_artifacts.selected_set_png_path)
    print(generic_corpus_exploration_weight_sweep_artifacts.baseline_manifest_path)
    print(ladder_witness_suite_artifacts.run_dir)
    print(ladder_witness_suite_artifacts.config_path)
    print(ladder_witness_suite_artifacts.schema_path)
    print(ladder_witness_suite_artifacts.manifest_path)
    print(ladder_witness_suite_artifacts.claim_matrix_path)
    print(ladder_witness_suite_artifacts.index_path)
    print(corpus_objective_artifacts.run_dir)
    print(corpus_objective_artifacts.report_path)
    print(candidate_generation_artifacts.run_dir)
    print(candidate_generation_artifacts.report_path)
    print(class_validity_artifacts.run_dir)
    print(class_validity_artifacts.report_path)
    print(generated_corpus_feature_artifacts.run_dir)
    print(generated_corpus_feature_artifacts.report_path)
    print(corpus_classifier_scoring_artifacts.run_dir)
    print(corpus_classifier_scoring_artifacts.report_path)
    print(objective_driven_qd_archive_artifacts.run_dir)
    print(objective_driven_qd_archive_artifacts.report_path)
    print(selected_generated_corpus_artifacts.run_dir)
    print(selected_generated_corpus_artifacts.report_path)
    print(corpus_search_baseline_artifacts.run_dir)
    print(corpus_search_baseline_artifacts.report_path)
    print(quality_diversity_corpus_artifacts.run_dir)
    print(quality_diversity_corpus_artifacts.report_path)
    print(adaptive_stress_corpus_artifacts.run_dir)
    print(adaptive_stress_corpus_artifacts.report_path)
    print(rl_backend_decision_artifacts.run_dir)
    print(rl_backend_decision_artifacts.report_path)
    print(corpus_synthesis_comparison_artifacts.run_dir)
    print(corpus_synthesis_comparison_artifacts.report_path)
    print(study_candidate_generation_artifacts.run_dir)
    print(study_candidate_generation_artifacts.decision_report_path)
    print(validation_ladder_artifacts.run_dir)
    print(validation_ladder_artifacts.report_path)
    print(bayesian_walkthrough_artifacts.run_dir)
    print(bayesian_walkthrough_artifacts.report_path)
    print(methodology_latex_artifacts.run_dir)
    print(methodology_latex_artifacts.artifact_tex_path)
    if methodology_latex_artifacts.pdf_path is not None:
        print(methodology_latex_artifacts.pdf_path)
    print(methodology_section_symbol_audit_artifacts.run_dir)
    print(methodology_section_symbol_audit_artifacts.report_path)
    print(functional_surface_catalog_artifacts.run_dir)
    print(functional_surface_catalog_artifacts.report_path)
    print(formal_math_registry_artifacts.run_dir)
    print(formal_math_registry_artifacts.report_path)
    print(formal_math_visual_registry_artifacts.run_dir)
    print(formal_math_visual_registry_artifacts.report_path)
    print(formal_math_visual_registry_artifacts.provenance_path)
    print(formal_math_visual_registry_artifacts.runbook_path)
    print(strict_equation_audit_artifacts.run_dir)
    print(strict_equation_audit_artifacts.report_path)
    print(trajectory_generator_artifacts.run_dir)
    print(trajectory_generator_artifacts.report_path)
    print(abstract_inspection_artifacts.run_dir)
    print(abstract_inspection_artifacts.index_path)
    print(feature_analysis_artifacts.run_dir)
    print(feature_analysis_artifacts.report_path)
    print(corpus_adequacy_artifacts.run_dir)
    print(corpus_adequacy_artifacts.report_path)
    print(pca_dimensionality_artifacts.run_dir)
    print(pca_dimensionality_artifacts.report_path)
    print(coverage_report_artifacts.run_dir)
    print(coverage_report_artifacts.report_path)
    print(pca_analysis_artifacts.run_dir)
    print(pca_analysis_artifacts.report_path)
    print(showcase_artifacts.showcase_dir)
    print(showcase_artifacts.index_path)
    print(showcase_artifacts.team_packet_dir)
    print(showcase_artifacts.validation_path)
    print(repo_story_artifacts.run_dir)
    print(repo_story_artifacts.claim_matrix_path)
    print(repo_story_artifacts.artifact_manifest_path)
    print(posterior_math_markdown_path)
    print(posterior_math_png_path)
    print(probability_primitives_markdown_path)
    print(probability_primitives_png_path)
    print(posterior_numeric_markdown_path)
    print(posterior_numeric_png_path)
    print(identity_markdown_path)
    print(identity_png_path)
    print(identity_csv_path)
    print(identity_feature_confusion_png_path)
    print(identity_posterior_markdown_path)
    print(identity_posterior_png_path)
    print(identity_posterior_failure_markdown_path)
    print(identity_posterior_failure_png_path)
    print(identity_posterior_comparison_markdown_path)
    print(identity_posterior_comparison_png_path)
    print(identity_posterior_margin_markdown_path)
    print(identity_posterior_margin_png_path)
    print(benchmark_path)
    print(benchmark_plot_png_path)
    print(toy_feature_confusion_png_path)
    print(benchmark_trace_csv)
    print(posterior_markdown_path)
    print(posterior_png_path)
    print(posterior_failure_markdown_path)
    print(posterior_failure_png_path)
    print(posterior_comparison_markdown_path)
    print(posterior_comparison_png_path)
    print(posterior_margin_markdown_path)
    print(posterior_margin_png_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
