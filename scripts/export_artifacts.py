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

from kinematic_classifier_sandbox import (
    run_accumulator_monte_carlo_benchmark,
    run_accumulator_benchmark,
    run_identity_benchmark,
    run_pointwise_benchmark,
    run_windowed_benchmark,
    analyze_pointwise_prior_sensitivity,
    analyze_prior_sensitivity,
    analyze_windowed_prior_sensitivity,
    analyze_cross_method_prior_comparison,
    write_coverage_report_artifacts,
    write_corpus_adequacy_artifacts,
    write_feature_analysis_artifacts,
    write_kalman_bank_artifacts,
    write_kalman_variant_comparison_artifacts,
    write_kalman_observable_comparison_artifacts,
    write_short_horizon_identifiability_artifacts,
    write_irregular_window_artifacts,
    write_transition_benchmark_artifacts,
    write_advanced_filter_decision_artifacts,
    write_velocity_aided_kalman_comparison_artifacts,
    write_pca_analysis_artifacts,
    write_cross_method_prior_comparison_artifacts,
    write_prior_sensitivity_artifacts,
    write_technique_comparison_artifacts,
    write_common_dataset_comparison_artifacts,
    write_common_experiment_artifacts,
    write_trajectory_generator_artifacts,
    write_abstract_inspection_artifacts,
    write_accumulator_artifacts,
    write_monte_carlo_artifacts,
    write_identity_feature_confusion_artifacts,
    write_identity_posterior_comparison_artifacts,
    write_identity_posterior_explainer_artifacts,
    write_identity_posterior_failure_artifacts,
    write_identity_posterior_margin_trace_artifacts,
    write_posterior_comparison_artifacts,
    render_posterior_explainer_markdown,
    render_toy_benchmark_markdown,
    run_toy_benchmark,
    write_identity_benchmark_artifacts,
    write_method_survey_artifact,
    write_posterior_numeric_walkthrough_artifacts,
    write_posterior_math_artifacts,
    write_probability_primitives_artifacts,
    write_posterior_explainer_artifacts,
    write_posterior_failure_artifacts,
    write_posterior_margin_trace_artifacts,
    write_pointwise_benchmark_artifacts,
    write_windowed_benchmark_artifacts,
    write_toy_benchmark_plot_artifacts,
    write_toy_feature_confusion_artifacts,
    write_toy_benchmark_trace_csv,
    write_milestone0_sample_run_artifacts,
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
    posterior_math_markdown_path, posterior_math_png_path = write_posterior_math_artifacts(
        ROOT / "artifacts"
    )
    probability_primitives_markdown_path, probability_primitives_png_path = write_probability_primitives_artifacts(
        ROOT / "artifacts"
    )
    posterior_numeric_markdown_path, posterior_numeric_png_path = write_posterior_numeric_walkthrough_artifacts(
        ROOT / "artifacts"
    )
    identity_result = run_identity_benchmark()
    identity_markdown_path, identity_png_path, identity_csv_path = write_identity_benchmark_artifacts(
        ROOT / "artifacts",
        result=identity_result,
    )
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
    posterior_markdown_path, posterior_png_path = write_posterior_explainer_artifacts(
        ROOT / "artifacts",
        result=result,
    )
    posterior_failure_markdown_path, posterior_failure_png_path = write_posterior_failure_artifacts(
        ROOT / "artifacts",
        result=result,
    )
    posterior_comparison_markdown_path, posterior_comparison_png_path = write_posterior_comparison_artifacts(
        ROOT / "artifacts",
        result=result,
    )
    posterior_margin_markdown_path, posterior_margin_png_path = write_posterior_margin_trace_artifacts(
        ROOT / "artifacts",
        result=result,
    )
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
    print(trajectory_generator_artifacts.run_dir)
    print(trajectory_generator_artifacts.report_path)
    print(abstract_inspection_artifacts.run_dir)
    print(abstract_inspection_artifacts.index_path)
    print(feature_analysis_artifacts.run_dir)
    print(feature_analysis_artifacts.report_path)
    print(corpus_adequacy_artifacts.run_dir)
    print(corpus_adequacy_artifacts.report_path)
    print(coverage_report_artifacts.run_dir)
    print(coverage_report_artifacts.report_path)
    print(pca_analysis_artifacts.run_dir)
    print(pca_analysis_artifacts.report_path)
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
