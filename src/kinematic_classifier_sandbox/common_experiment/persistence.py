from __future__ import annotations

import json
import shutil
from pathlib import Path

from kinematic_classifier_sandbox.utils.io import write_csv
from kinematic_classifier_sandbox.utils.method_evaluation_summary import (
    METHOD_EVALUATION_SUMMARY_FIELDS,
    PosteriorMetricSample,
    build_method_evaluation_summary_row,
    compute_multiclass_posterior_metrics,
)

from ..analysis.feature_analysis import load_feature_set_manifest
from ..scenarios import list_scenario_ids
from ..trajectory_generator import default_trajectory_class_definitions
from ..validation.shared_evaluation import sensor_regime_summary_rows
from .contracts import CommonExperimentArtifacts, CommonExperimentResult
from .plot_pack import write_common_experiment_plot_pack
from .reporting import render_common_experiment_report


def write_common_experiment_artifacts(
        output_dir: str | Path,
        *,
        analysis: CommonExperimentResult,
) -> CommonExperimentArtifacts:
    run_dir = Path(output_dir) / analysis.config.output_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)

    output_filenames = analysis.config.output_filenames
    config_path = run_dir / "config.yaml"
    dataset_manifest_path = run_dir / "dataset_manifest.json"
    class_definitions_path = run_dir / "class_definitions.json"
    feature_manifest_path = run_dir / "feature_manifest.json"
    feature_sets_path = run_dir / "feature_sets.json"
    class_pair_manifest_path = run_dir / "class_pair_manifest.json"
    classifier_manifest_path = run_dir / "classifier_manifest.json"
    sensor_regimes_path = run_dir / "sensor_regimes.json"
    predictions_path = run_dir / output_filenames.get("predictions_path", "unified_predictions.csv")
    posterior_history_path = run_dir / output_filenames.get("posterior_history_path", "unified_posterior_history.csv")
    likelihood_history_path = run_dir / output_filenames.get("likelihood_history_path",
                                                             "unified_likelihood_history.csv")
    method_evaluation_summary_path = run_dir / output_filenames.get(
        "method_evaluation_summary_path",
        "method_evaluation_summary.csv",
    )
    feature_matrix_path = run_dir / output_filenames.get("feature_matrix_path", "unified_feature_matrix.csv")
    metrics_by_classifier_path = run_dir / output_filenames.get("metrics_by_classifier_path",
                                                                "metrics_by_classifier.csv")
    metrics_by_sensor_regime_path = run_dir / output_filenames.get("metrics_by_sensor_regime_path",
                                                                   "metrics_by_sensor_regime.csv")
    metrics_by_classifier_and_feature_set_path = run_dir / output_filenames.get(
        "metrics_by_classifier_and_feature_set_path", "metrics_by_classifier_and_feature_set.csv")
    metrics_by_class_pair_path = run_dir / output_filenames.get("metrics_by_class_pair_path",
                                                                "metrics_by_class_pair.csv")
    prior_sensitivity_by_class_pair_path = run_dir / output_filenames.get("prior_sensitivity_by_class_pair_path",
                                                                          "prior_sensitivity_by_class_pair.csv")
    feature_set_comparison_path = run_dir / output_filenames.get("feature_set_comparison_path",
                                                                 "feature_set_comparison.csv")
    irregular_window_comparison_path = run_dir / output_filenames.get("irregular_window_comparison_path",
                                                                      "irregular_window_comparison.csv")
    class_pair_duration_study_path = run_dir / output_filenames.get("class_pair_duration_study_path",
                                                                    "class_pair_duration_study.csv")
    class_pair_scenario_study_path = run_dir / output_filenames.get("class_pair_scenario_study_path",
                                                                    "class_pair_scenario_study.csv")
    covariate_leakage_audit_path = run_dir / output_filenames.get("covariate_leakage_audit_path",
                                                                  "covariate_leakage_audit.csv")
    feature_excitation_matrix_path = run_dir / output_filenames.get("feature_excitation_matrix_path",
                                                                    "feature_excitation_matrix.csv")
    identifiability_matrix_path = run_dir / output_filenames.get("identifiability_matrix_path",
                                                                 "identifiability_matrix.csv")
    oracle_classifier_results_path = run_dir / output_filenames.get("oracle_classifier_results_path",
                                                                    "oracle_classifier_results.csv")
    report_path = run_dir / output_filenames.get("report_path", "common_experiment_report.md")
    canonical_report_path = run_dir / "report.md"

    shutil.copyfile(analysis.config.config_path, config_path)
    shutil.copyfile(analysis.config.feature_sets_path, feature_sets_path)
    shutil.copyfile(analysis.config.class_pair_manifest_path, class_pair_manifest_path)
    shutil.copyfile(analysis.config.classifier_manifest_path, classifier_manifest_path)

    feature_manifest = load_feature_set_manifest(analysis.config.feature_sets_path)
    feature_manifest_path.write_text(json.dumps(feature_manifest, indent=2), encoding="utf-8")
    class_definitions = [
        {
            "name": definition.name,
            "kind": definition.kind,
            "description": definition.description,
            "nominal_steps": list(definition.nominal_steps),
            "dt_range": list(definition.dt_range),
            "measurement_std_range": list(definition.measurement_std_range),
        }
        for definition in default_trajectory_class_definitions()
    ]
    class_definitions_path.write_text(json.dumps({"classes": class_definitions}, indent=2), encoding="utf-8")
    dataset_manifest_path.write_text(
        json.dumps(
            {
                "experiment_name": analysis.summary.experiment_name,
                "executable_class_pairs": list(analysis.summary.executable_class_pairs),
                "trajectories_per_case": analysis.summary.trajectories_per_case,
                "num_pair_trajectories": analysis.summary.num_pair_trajectories,
                "num_pair_predictions": analysis.summary.num_pair_predictions,
                "scenario_ids": list(list_scenario_ids()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    sensor_regimes_path.write_text(json.dumps(sensor_regime_summary_rows(analysis.comparison.runs), indent=2),
                                   encoding="utf-8")

    write_csv(predictions_path, list(analysis.pair_prediction_rows), [
        "run_id", "classifier_id", "feature_set_id", "sensor_regime_id", "measurement_dim", "coordinate_frame",
        "class_pair_id", "class_a", "class_b", "trajectory_id", "scenario_id", "scenario_family", "dataset_tier",
        "time", "true_class", "predicted_class", "confidence", "posterior_class_a", "posterior_class_b",
    ])
    write_csv(posterior_history_path, list(analysis.posterior_history_rows), [
        "run_id", "classifier_id", "feature_set_id", "sensor_regime_id", "class_pair_id", "class_a", "class_b",
        "trajectory_id", "scenario_id", "scenario_family", "dataset_tier", "time", "true_class",
        "posterior_class_a", "posterior_class_b",
    ])
    write_csv(likelihood_history_path, list(analysis.likelihood_history_rows), [
        "run_id", "classifier_id", "feature_set_id", "sensor_regime_id", "class_pair_id", "trajectory_id",
        "scenario_id", "scenario_family", "dataset_tier", "time", "score_type", "class_a", "class_b",
        "log_likelihood_class_a", "log_likelihood_class_b",
    ])
    write_csv(
        method_evaluation_summary_path,
        _build_common_method_evaluation_rows(analysis),
        list(METHOD_EVALUATION_SUMMARY_FIELDS),
    )
    write_csv(feature_matrix_path, list(analysis.feature_rows), [
        "trajectory_id", "class_pair_id", "scenario_id", "scenario_family", "dataset_tier", "true_class",
        "feature_set_id", "duration", "position_range", "speed_range", "acceleration_range", "acceleration_variance",
        "curvature_proxy", "velocity_sign_changes", "acceleration_sign_changes", "monotonicity",
        "linear_fit_residual", "quadratic_fit_residual", "outlier_score",
    ])
    write_csv(metrics_by_classifier_path, list(analysis.metrics_by_classifier_rows),
              ["classifier_id", "overall_accuracy", "num_predictions"])
    write_csv(metrics_by_sensor_regime_path, list(analysis.metrics_by_sensor_regime_rows), [
        "sensor_regime_id", "same_sensor_fairness_bucket", "overall_accuracy", "mean_confidence", "num_predictions",
        "num_classifiers", "measurement_dims", "coordinate_frames",
    ])
    write_csv(metrics_by_classifier_and_feature_set_path, list(analysis.metrics_by_classifier_and_feature_set_rows),
              ["classifier_id", "feature_set_id", "overall_accuracy"])
    write_csv(metrics_by_class_pair_path, list(analysis.metrics_by_class_pair_rows),
              ["classifier_id", "class_pair", "overall_accuracy", "status"])
    write_csv(prior_sensitivity_by_class_pair_path, list(analysis.prior_sensitivity_rows),
              ["classifier_id", "class_pair_id", "prior_id", "accuracy"])
    write_csv(feature_set_comparison_path, list(analysis.feature_set_comparison_rows), [
        "feature_set_id", "history_behavior", "num_features", "overall_accuracy", "min_pair_accuracy",
        "max_pair_accuracy", "mean_confidence",
    ])
    write_csv(irregular_window_comparison_path, list(analysis.irregular_window_rows), [
        "class_pair_id", "feature_set_id", "history_behavior", "window_definition", "window_sample_count",
        "window_duration", "num_predictions", "overall_accuracy", "mean_confidence", "mean_selected_sample_count",
        "mean_selected_duration", "cross_window_prediction_disagreement_rate", "mean_cross_window_feature_delta",
    ])
    write_csv(class_pair_duration_study_path, list(analysis.class_pair_duration_rows), [
        "classifier_id", "class_pair_id", "time", "num_prefixes", "prefix_accuracy", "mean_confidence",
        "posterior_margin",
    ])
    write_csv(class_pair_scenario_study_path, list(analysis.class_pair_scenario_rows), [
        "classifier_id", "class_pair_id", "scenario_id", "scenario_family", "overall_accuracy", "mean_confidence",
        "num_predictions",
    ])
    write_csv(covariate_leakage_audit_path, list(analysis.covariate_rows), [
        "class_pair_id", "dataset_tier", "scenario_family", "true_class", "num_trajectories", "mean_duration",
        "mean_sample_count", "mean_dt", "std_dt", "max_dt", "sampling_irregularity", "measurement_std",
        "outlier_fraction", "max_covariate_delta_name", "max_covariate_delta_ratio", "status",
    ])
    feature_excitation_rows = []
    for row in analysis.feature_excitation_rows:
        flattened_row = {
            "class_pair_id": row["class_pair_id"],
            "dataset_tier": row["dataset_tier"],
            "scenario_family": row["scenario_family"],
            "feature_set_id": row["feature_set_id"],
            "num_rows": row["num_rows"],
        }
        feature_means = row["feature_means"]
        feature_stds = row["feature_stds"]
        for feature_name in ("position_range", "speed_range", "acceleration_range", "acceleration_variance",
                             "curvature_proxy", "velocity_sign_changes", "acceleration_sign_changes", "monotonicity",
                             "linear_fit_residual", "quadratic_fit_residual", "outlier_score"):
            flattened_row[f"{feature_name}_mean_abs"] = feature_means[feature_name]
            flattened_row[f"{feature_name}_std"] = feature_stds[feature_name]
        feature_excitation_rows.append(flattened_row)
    write_csv(feature_excitation_matrix_path, feature_excitation_rows, [
        "class_pair_id", "dataset_tier", "scenario_family", "feature_set_id", "num_rows", "position_range_mean_abs",
        "position_range_std", "speed_range_mean_abs", "speed_range_std", "acceleration_range_mean_abs",
        "acceleration_range_std", "acceleration_variance_mean_abs", "acceleration_variance_std",
        "curvature_proxy_mean_abs", "curvature_proxy_std", "velocity_sign_changes_mean_abs",
        "velocity_sign_changes_std", "acceleration_sign_changes_mean_abs", "acceleration_sign_changes_std",
        "monotonicity_mean_abs", "monotonicity_std", "linear_fit_residual_mean_abs", "linear_fit_residual_std",
        "quadratic_fit_residual_mean_abs", "quadratic_fit_residual_std", "outlier_score_mean_abs", "outlier_score_std",
    ])
    write_csv(identifiability_matrix_path, list(analysis.identifiability_rows), [
        "class_pair_id", "feature_set_id", "history_behavior", "class_a", "class_b", "num_examples", "num_features",
        "mean_absolute_feature_distance", "mean_standardized_feature_distance", "overlap_estimate",
        "confusability_score", "identifiability_status",
    ])
    write_csv(oracle_classifier_results_path, list(analysis.oracle_rows), [
        "class_pair_id", "feature_set_id", "oracle_accuracy", "mean_confidence", "mean_posterior_margin",
        "num_examples", "history_behavior", "best_feature_set_for_pair", "best_oracle_accuracy_for_pair",
        "is_best_feature_set",
    ])
    report_text = render_common_experiment_report(analysis)
    report_path.write_text(report_text, encoding="utf-8")
    canonical_report_path.write_text(report_text, encoding="utf-8")
    plots_dir = write_common_experiment_plot_pack(run_dir, result=analysis)

    return CommonExperimentArtifacts(
        run_dir=run_dir,
        config_path=config_path,
        dataset_manifest_path=dataset_manifest_path,
        class_definitions_path=class_definitions_path,
        feature_manifest_path=feature_manifest_path,
        feature_sets_path=feature_sets_path,
        class_pair_manifest_path=class_pair_manifest_path,
        classifier_manifest_path=classifier_manifest_path,
        sensor_regimes_path=sensor_regimes_path,
        predictions_path=predictions_path,
        posterior_history_path=posterior_history_path,
        likelihood_history_path=likelihood_history_path,
        method_evaluation_summary_path=method_evaluation_summary_path,
        feature_matrix_path=feature_matrix_path,
        metrics_by_classifier_path=metrics_by_classifier_path,
        metrics_by_sensor_regime_path=metrics_by_sensor_regime_path,
        metrics_by_classifier_and_feature_set_path=metrics_by_classifier_and_feature_set_path,
        metrics_by_class_pair_path=metrics_by_class_pair_path,
        prior_sensitivity_by_class_pair_path=prior_sensitivity_by_class_pair_path,
        feature_set_comparison_path=feature_set_comparison_path,
        irregular_window_comparison_path=irregular_window_comparison_path,
        class_pair_duration_study_path=class_pair_duration_study_path,
        class_pair_scenario_study_path=class_pair_scenario_study_path,
        covariate_leakage_audit_path=covariate_leakage_audit_path,
        feature_excitation_matrix_path=feature_excitation_matrix_path,
        identifiability_matrix_path=identifiability_matrix_path,
        oracle_classifier_results_path=oracle_classifier_results_path,
        report_path=report_path,
        canonical_report_path=canonical_report_path,
        plots_dir=plots_dir,
    )


def _build_common_method_evaluation_rows(
    analysis: CommonExperimentResult,
) -> list[dict[str, object]]:
    grouped_rows: dict[str, list[dict[str, object]]] = {}
    for row in analysis.pair_prediction_rows:
        grouped_rows.setdefault(str(row["classifier_id"]), []).append(
            {
                "true_class": str(row["true_class"]),
                "predicted_class": str(row["predicted_class"]),
                "confidence": float(row["confidence"]),
                "posterior_by_label": {
                    str(row["class_a"]): float(row["posterior_class_a"]),
                    str(row["class_b"]): float(row["posterior_class_b"]),
                },
            }
        )
    rows: list[dict[str, object]] = []
    for classifier_id, prediction_rows in sorted(grouped_rows.items()):
        samples = [
            PosteriorMetricSample(
                true_label=prediction_row["true_class"],
                predicted_label=prediction_row["predicted_class"],
                confidence=prediction_row["confidence"],
                posterior_by_label=prediction_row["posterior_by_label"],
            )
            for prediction_row in prediction_rows
        ]
        metrics = compute_multiclass_posterior_metrics(samples)
        study_surface, evaluation_surface = _common_method_surfaces(classifier_id)
        rows.append(
            build_method_evaluation_summary_row(
                method_id=classifier_id,
                study_surface=study_surface,
                evaluation_surface=evaluation_surface,
                metrics=metrics,
            )
        )
    return rows


def _common_method_surfaces(classifier_id: str) -> tuple[str, str]:
    mapping = {
        "pointwise": ("common_1d_classifier_study", "local_overlap"),
        "windowed_robust_extrema": ("common_1d_classifier_study", "window_outlier"),
        "bayes_accumulator": ("common_1d_classifier_study", "weak_repeated_evidence"),
        "kalman_bank": ("common_1d_classifier_study", "matched_endpoint_dynamics"),
    }
    return mapping.get(classifier_id, ("common_1d_classifier_study", "shared_benchmark"))
