from __future__ import annotations

import argparse
from pathlib import Path
from _bootstrap import bootstrap_repo

ROOT = bootstrap_repo(configure_runtime=True)




def _remove_svg_outputs(root: Path) -> None:
    for path in root.rglob("*.svg"):
        path.unlink(missing_ok=True)

from kinematic_classifier_sandbox.analysis.inspection_bundle import (
    write_abstract_inspection_artifacts,
)
from kinematic_classifier_sandbox.advanced_filters.evaluation import (
    write_advanced_filter_comparison_artifacts,
    write_particle_filter_witness_artifacts,
    write_rbpf_witness_artifacts,
)
from kinematic_classifier_sandbox.advanced_filters.oracle_pf_1d import (
    write_pf_abs_range_multimodal_witness_artifacts,
)
from kinematic_classifier_sandbox.advanced_filters.oracle_gsf_1d import (
    write_gsf_abs_range_multimodal_witness_artifacts,
)
from kinematic_classifier_sandbox.advanced_filters.oracle_ukf_1d import (
    write_ukf_nonlinear_unimodal_witness_artifacts,
)
from kinematic_classifier_sandbox.advanced_filters.oracle_student_t_1d import (
    write_student_t_heavy_tail_witness_artifacts,
)
from kinematic_classifier_sandbox.advanced_filters.hsmm_duration_witness import (
    write_hsmm_duration_limited_witness_artifacts,
)
from kinematic_classifier_sandbox.advanced_filters.bocpd_onset_witness import (
    write_bocpd_unknown_onset_witness_artifacts,
)
from kinematic_classifier_sandbox.advanced_filters.ou_witness import (
    write_ornstein_uhlenbeck_witness_artifacts,
)
from kinematic_classifier_sandbox.advanced_filters.runner import write_imm_artifacts
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
from kinematic_classifier_sandbox.analysis.common_dataset_comparison_artifact_io import (
    write_common_dataset_comparison_artifacts,
)
from kinematic_classifier_sandbox.analysis.cmaes_generator_witness import (
    write_continuous_generator_frontier_artifacts,
)
from kinematic_classifier_sandbox.analysis.shapelet_motif_witness import (
    write_shapelet_maneuver_motif_witness_artifacts,
)
from kinematic_classifier_sandbox.analysis.gradient_boosted_feature_witness import (
    write_feature_headroom_frontier_artifacts,
)
from kinematic_classifier_sandbox.analysis.neural_sequence_frontier import (
    write_neural_sequence_vs_physics_frontier_artifacts,
)
from kinematic_classifier_sandbox.analysis.sequential_control_generator_frontier import (
    write_sequential_control_generator_frontier_artifacts,
)
from kinematic_classifier_sandbox.analysis.sequential_offpolicy_control_frontier import (
    write_sequential_offpolicy_control_frontier_artifacts,
)
from kinematic_classifier_sandbox.analysis.tsc_archive_frontier import (
    write_tsc_archive_baseline_frontier_artifacts,
)
from kinematic_classifier_sandbox.analysis.sequential_offpolicy_control_frontier import (
    write_sequential_offpolicy_control_frontier_artifacts,
)
from kinematic_classifier_sandbox.analysis.dimensional_lift_audit_artifact_io import (
    write_dimensional_lift_audit_artifacts,
)
from kinematic_classifier_sandbox.analysis.feature_analysis import (
    analyze_feature_datasets,
)
from kinematic_classifier_sandbox.analysis.feature_analysis_artifact_io import write_feature_analysis_artifacts
from kinematic_classifier_sandbox.analysis.generated_corpus_features import (
    write_generated_corpus_feature_artifacts,
)
from kinematic_classifier_sandbox.analysis.embedding_baseline_frontier import (
    write_embedding_baseline_frontier_artifacts,
)
from kinematic_classifier_sandbox.analysis.pca_analysis_artifact_io import write_pca_analysis_artifacts
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
from kinematic_classifier_sandbox.corpus.adequacy_artifact_io import write_corpus_adequacy_artifacts
from kinematic_classifier_sandbox.corpus.adequacy_audit import analyze_corpus_adequacy
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
from kinematic_classifier_sandbox.corpus.selected_generated_corpus_artifact_io import (
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
from kinematic_classifier_sandbox.methodology.context import (
    build_methodology_execution_context,
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
    analyze_methodology_latex,
    write_methodology_latex_artifacts,
    write_methodology_section_symbol_audit_artifacts,
)
from kinematic_classifier_sandbox.rung_sufficiency.analysis import (
    write_ladder_witness_suite_artifacts,
    write_rung_sufficiency_artifacts,
)
from kinematic_classifier_sandbox.showcase.builder import build_showcase_artifacts
from kinematic_classifier_sandbox.story.repo_story import write_repo_story_artifacts
from kinematic_classifier_sandbox.tracing.filter_trace_validation_packet import (
    write_filter_trace_validation_artifacts,
)
from kinematic_classifier_sandbox.registry.strict_equation_audit import write_strict_equation_audit_artifacts
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
from kinematic_classifier_sandbox.corpus.coverage_artifact_io import write_coverage_report_artifacts
from kinematic_classifier_sandbox.corpus.coverage_report import analyze_coverage_report
from kinematic_classifier_sandbox.corpus.exploration.external_backend_examples_rendering import (
    write_external_backend_examples_artifacts,
)
from kinematic_classifier_sandbox.corpus.quality_diversity import (
    write_quality_diversity_corpus_artifacts,
)
from kinematic_classifier_sandbox.registry.formal_math_registry import write_formal_math_registry_artifacts
from kinematic_classifier_sandbox.registry.formal_math_visual_registry import (
    write_formal_math_visual_registry_artifacts,
)
from kinematic_classifier_sandbox.registry.functional_surface_catalog import (
    write_functional_surface_catalog_artifacts,
)
from kinematic_classifier_sandbox.registry.method_validation_os import (
    write_method_validation_os_artifacts,
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
from kinematic_classifier_sandbox.registry.strict_equation_audit import write_strict_equation_audit_artifacts
from kinematic_classifier_sandbox.study_candidate_generation import (
    write_study_candidate_generation_artifacts,
)
from kinematic_classifier_sandbox.study_candidate_protocol import (
    write_study_candidate_protocol_artifacts,
)
from kinematic_classifier_sandbox.utils.analysis_cache import (
    describe_analysis_cache,
    describe_analysis_cache_stats,
    reset_analysis_cache_stats,
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python3 scripts/export_artifacts.py")
    parser.add_argument(
        "--scope",
        choices=("full", "front-door"),
        default="full",
        help="Artifact scope to regenerate.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use cheaper artifact settings where supported. Intended for `--scope front-door`.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Output directory for `--scope front-door`. Full scope requires the default repo `artifacts/` output.",
    )
    return parser


def _print_cache_status(label: str) -> None:
    summary = describe_analysis_cache()
    stats = describe_analysis_cache_stats()
    print(
        f"analysis_cache[{label}]: root={summary['root']} namespaces={summary['namespace_count']} "
        f"entries={summary['entry_count']} bytes={summary['bytes']} hits={stats['hit']} "
        f"misses={stats['miss']} corrupt={stats['corrupt']} disabled={stats['disabled']}"
    )
    for row in stats["namespaces"]:
        print(
            f"analysis_cache[{label}].{row['namespace']}: hits={row['hit']} misses={row['miss']} "
            f"corrupt={row['corrupt']} disabled={row['disabled']}"
        )


def _run_front_door_export(output_root: Path, *, fast: bool) -> int:
    reset_analysis_cache_stats()
    methodology_context = build_methodology_execution_context(
        seed=7,
        trajectories_per_case=6,
        use_cache=True,
    )
    methodology_latex_result = analyze_methodology_latex(
        methodology_context=methodology_context,
        use_cache=True,
    )
    artifact_mode = "fast" if fast else "full"
    feature_analysis_result = analyze_feature_datasets(
        seed=7,
        trajectories_per_class=5,
    )
    corpus_adequacy_result = analyze_corpus_adequacy(
        seed=7,
        trajectories_per_class=5,
        feature_analysis_result=feature_analysis_result,
    )
    coverage_report_result = analyze_coverage_report(
        seed=7,
        trajectories_per_class=5,
        corpus_adequacy_result=corpus_adequacy_result,
    )
    selected_generated_corpus_artifacts = write_selected_generated_corpus_artifacts(output_root)
    study_candidate_generation_artifacts = write_study_candidate_generation_artifacts(
        output_root,
        result=methodology_context.study_generation_result,
    )
    validation_ladder_artifacts = write_validation_ladder_artifacts(
        output_root,
        result=methodology_context.validation_result,
    )
    feature_analysis_artifacts = write_feature_analysis_artifacts(
        output_root,
        seed=7,
        trajectories_per_class=5,
        result=feature_analysis_result,
    )
    corpus_adequacy_artifacts = write_corpus_adequacy_artifacts(
        output_root,
        seed=7,
        trajectories_per_class=5,
        result=corpus_adequacy_result,
    )
    coverage_report_artifacts = write_coverage_report_artifacts(
        output_root,
        seed=7,
        trajectories_per_class=5,
        result=coverage_report_result,
    )
    methodology_latex_artifacts = write_methodology_latex_artifacts(
        output_root,
        result=methodology_latex_result,
        methodology_context=methodology_context,
        artifact_mode=artifact_mode,
    )
    methodology_section_symbol_audit_artifacts = write_methodology_section_symbol_audit_artifacts(
        output_root,
        methodology_tex=methodology_latex_result.methodology_tex,
        build_pdf=not fast,
    )
    showcase_artifacts = None
    if not fast:
        showcase_artifacts = build_showcase_artifacts(
            output_root,
            refresh=False,
            create_zip=False,
            methodology_context=methodology_context,
            artifact_mode=artifact_mode,
        )
    repo_story_artifacts = write_repo_story_artifacts(
        output_root,
        docs_root=ROOT / "docs",
        write_showcase=not fast,
    )
    _remove_svg_outputs(output_root)
    print(selected_generated_corpus_artifacts.run_dir)
    print(selected_generated_corpus_artifacts.report_path)
    print(study_candidate_generation_artifacts.run_dir)
    print(study_candidate_generation_artifacts.decision_report_path)
    print(validation_ladder_artifacts.run_dir)
    print(validation_ladder_artifacts.report_path)
    print(feature_analysis_artifacts.run_dir)
    print(feature_analysis_artifacts.report_path)
    print(corpus_adequacy_artifacts.run_dir)
    print(corpus_adequacy_artifacts.report_path)
    print(coverage_report_artifacts.run_dir)
    print(coverage_report_artifacts.report_path)
    print(methodology_latex_artifacts.run_dir)
    print(methodology_latex_artifacts.artifact_tex_path)
    if methodology_latex_artifacts.pdf_path is not None:
        print(methodology_latex_artifacts.pdf_path)
    print(methodology_section_symbol_audit_artifacts.run_dir)
    print(methodology_section_symbol_audit_artifacts.report_path)
    if showcase_artifacts is not None:
        print(showcase_artifacts.showcase_dir)
        print(showcase_artifacts.index_path)
        print(showcase_artifacts.team_packet_dir)
        print(showcase_artifacts.validation_path)
    print(repo_story_artifacts.run_dir)
    print(repo_story_artifacts.claim_matrix_path)
    print(repo_story_artifacts.artifact_manifest_path)
    _print_cache_status("front-door")
    return 0


def main() -> int:
    args = _build_parser().parse_args()
    if args.scope == "front-door":
        output_root = Path(args.output_dir)
        if not output_root.is_absolute():
            output_root = ROOT / output_root
        return _run_front_door_export(output_root, fast=args.fast)
    if args.output_dir != "artifacts":
        raise SystemExit("--output-dir is only supported with --scope front-door")

    methodology_context = build_methodology_execution_context(
        seed=7,
        trajectories_per_case=6,
        use_cache=True,
    )
    methodology_latex_result = analyze_methodology_latex(
        methodology_context=methodology_context,
        use_cache=True,
    )
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
    imm_artifacts = write_imm_artifacts(ROOT / "artifacts")
    particle_filter_artifacts = write_particle_filter_witness_artifacts(ROOT / "artifacts")
    pf_oracle_artifacts = write_pf_abs_range_multimodal_witness_artifacts(ROOT / "artifacts")
    gsf_oracle_artifacts = write_gsf_abs_range_multimodal_witness_artifacts(ROOT / "artifacts")
    ukf_oracle_artifacts = write_ukf_nonlinear_unimodal_witness_artifacts(ROOT / "artifacts")
    student_t_oracle_artifacts = write_student_t_heavy_tail_witness_artifacts(ROOT / "artifacts")
    hsmm_duration_artifacts = write_hsmm_duration_limited_witness_artifacts(ROOT / "artifacts")
    bocpd_onset_artifacts = write_bocpd_unknown_onset_witness_artifacts(ROOT / "artifacts")
    rbpf_artifacts = write_rbpf_witness_artifacts(ROOT / "artifacts")
    ou_witness_artifacts = write_ornstein_uhlenbeck_witness_artifacts(ROOT / "artifacts")
    advanced_filter_comparison_artifacts = write_advanced_filter_comparison_artifacts(ROOT / "artifacts")
    filter_trace_validation_artifacts = write_filter_trace_validation_artifacts(ROOT / "artifacts")
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
    shapelet_motif_artifacts = write_shapelet_maneuver_motif_witness_artifacts(ROOT / "artifacts")
    feature_headroom_artifacts = write_feature_headroom_frontier_artifacts(ROOT / "artifacts")
    embedding_baseline_artifacts = write_embedding_baseline_frontier_artifacts(ROOT / "artifacts")
    neural_sequence_frontier_artifacts = write_neural_sequence_vs_physics_frontier_artifacts(ROOT / "artifacts")
    tsc_archive_frontier_artifacts = write_tsc_archive_baseline_frontier_artifacts(ROOT / "artifacts")
    cmaes_generator_frontier_artifacts = write_continuous_generator_frontier_artifacts(ROOT / "artifacts")
    sequential_control_frontier_artifacts = write_sequential_control_generator_frontier_artifacts(ROOT / "artifacts")
    sequential_offpolicy_frontier_artifacts = write_sequential_offpolicy_control_frontier_artifacts(ROOT / "artifacts")
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
    study_candidate_generation_artifacts = write_study_candidate_generation_artifacts(
        ROOT / "artifacts",
        result=methodology_context.study_generation_result,
    )
    validation_ladder_artifacts = write_validation_ladder_artifacts(
        ROOT / "artifacts",
        result=methodology_context.validation_result,
    )
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
    methodology_latex_artifacts = write_methodology_latex_artifacts(
        ROOT / "artifacts",
        result=methodology_latex_result,
        methodology_context=methodology_context,
    )
    methodology_section_symbol_audit_artifacts = write_methodology_section_symbol_audit_artifacts(
        ROOT / "artifacts",
        methodology_tex=methodology_latex_result.methodology_tex,
    )
    functional_surface_catalog_artifacts = write_functional_surface_catalog_artifacts(ROOT / "artifacts")
    method_validation_os_artifacts = write_method_validation_os_artifacts(ROOT / "artifacts")
    formal_math_registry_artifacts = write_formal_math_registry_artifacts(ROOT / "artifacts")
    formal_math_visual_registry_artifacts = write_formal_math_visual_registry_artifacts(ROOT / "artifacts")
    strict_equation_audit_artifacts = write_strict_equation_audit_artifacts(ROOT / "artifacts")
    pca_dimensionality_artifacts = write_pca_dimensionality_audit_artifacts(
        ROOT / "artifacts",
        seed=7,
        trajectories_per_class=5,
        max_components=5,
    )
    showcase_artifacts = build_showcase_artifacts(
        ROOT / "artifacts",
        refresh=False,
        create_zip=False,
        methodology_context=methodology_context,
    )
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
    print(imm_artifacts.run_dir)
    print(imm_artifacts.report_path)
    print(particle_filter_artifacts.run_dir)
    print(particle_filter_artifacts.report_path)
    print(rbpf_artifacts.run_dir)
    print(rbpf_artifacts.report_path)
    print(ou_witness_artifacts.run_dir)
    print(ou_witness_artifacts.report_path)
    print(advanced_filter_comparison_artifacts.run_dir)
    print(advanced_filter_comparison_artifacts.report_path)
    print(filter_trace_validation_artifacts.run_dir)
    print(filter_trace_validation_artifacts.report_path)
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
    print(embedding_baseline_artifacts.run_dir)
    print(embedding_baseline_artifacts.report_path)
    print(sequential_control_frontier_artifacts.run_dir)
    print(sequential_control_frontier_artifacts.report_path)
    print(sequential_offpolicy_frontier_artifacts.run_dir)
    print(sequential_offpolicy_frontier_artifacts.report_path)
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
    print(method_validation_os_artifacts.run_dir)
    print(method_validation_os_artifacts.report_path)
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
