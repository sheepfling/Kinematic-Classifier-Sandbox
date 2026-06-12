from __future__ import annotations

from kinematic_classifier_sandbox.reports.markdown import MarkdownDocument

from .contracts import CommonExperimentResult


def render_common_experiment_report(result: CommonExperimentResult) -> str:
    executed_pairs = ", ".join(result.summary.executable_class_pairs)
    
    doc = MarkdownDocument("Common Experiment Harness")
    doc.paragraph(
        "Milestone 10 and Milestone 11 executable common-study subset for the manifest-driven 1D classifier study."
    )
    
    doc.heading("Summary", level=2)
    doc.bullet_list(
        [
            f"Experiment: `{result.summary.experiment_name}`",
            f"Study adapter: `{result.summary.study_adapter_id}`",
            f"Executable class pairs: `{executed_pairs}`",
            f"Pair trajectories: `{result.summary.num_pair_trajectories}`",
            f"Pair predictions: `{result.summary.num_pair_predictions}`",
            f"Shared-comparison methods: `{len(result.comparison.rows)}`",
        ]
    )
    
    doc.heading("Metrics By Classifier", level=2)
    doc.table(
        ["classifier_id", "overall_accuracy", "num_predictions"],
        [
            (row["classifier_id"], f"{float(row['overall_accuracy']):.3f}", int(row["num_predictions"]))
            for row in result.metrics_by_classifier_rows
        ]
    )

    doc.heading("Metrics By Sensor Regime", level=2)
    doc.table(
        ["sensor_regime_id", "same_sensor_fairness_bucket", "overall_accuracy", "mean_confidence", "num_predictions", "num_classifiers"],
        [
            (
                row["sensor_regime_id"],
                row["same_sensor_fairness_bucket"],
                f"{float(row['overall_accuracy']):.3f}",
                f"{float(row['mean_confidence']):.3f}",
                int(row["num_predictions"]),
                int(row["num_classifiers"]),
            )
            for row in result.metrics_by_sensor_regime_rows
        ]
    )

    doc.heading("Metrics By Class Pair", level=2)
    doc.table(
        ["classifier_id", "class_pair", "overall_accuracy", "status"],
        [
            (row["classifier_id"], row["class_pair"], f"{float(row['overall_accuracy']):.3f}", row["status"])
            for row in result.metrics_by_class_pair_rows
        ]
    )

    doc.heading("Feature-Set Study", level=2)
    doc.table(
        ["feature_set_id", "history_behavior", "num_features", "overall_accuracy", "min_pair_accuracy"],
        [
            (
                row["feature_set_id"],
                row["history_behavior"],
                int(row["num_features"]),
                f"{float(row['overall_accuracy']):.3f}",
                f"{float(row['min_pair_accuracy']):.3f}",
            )
            for row in result.feature_set_comparison_rows
        ]
    )

    doc.heading("Irregular Window Comparison", level=2)
    if result.irregular_window_rows:
        doc.table(
            ["class_pair_id", "feature_set_id", "window_definition", "overall_accuracy", "mean_selected_duration", "cross_window_prediction_disagreement_rate"],
            [
                (
                    row["class_pair_id"],
                    row["feature_set_id"],
                    row["window_definition"],
                    f"{float(row['overall_accuracy']):.3f}",
                    f"{float(row['mean_selected_duration']):.2f}",
                    f"{float(row['cross_window_prediction_disagreement_rate']):.3f}",
                )
                for row in result.irregular_window_rows[:18]
            ]
        )
    else:
        doc.table(
            ["class_pair_id", "feature_set_id", "window_definition", "overall_accuracy", "mean_selected_duration", "cross_window_prediction_disagreement_rate"],
            [("n/a", "n/a", "n/a", "0.000", "0.00", "0.000")]
        )

    doc.heading("Class-Pair Duration Study", level=2)
    doc.table(
        ["classifier_id", "class_pair_id", "time", "prefix_accuracy", "mean_confidence"],
        [
            (
                row["classifier_id"],
                row["class_pair_id"],
                f"{float(row['time']):.2f}",
                f"{float(row['prefix_accuracy']):.3f}",
                f"{float(row['mean_confidence']):.3f}",
            )
            for row in result.class_pair_duration_rows[:12]
        ]
    )

    doc.heading("Class-Pair Scenario Study", level=2)
    doc.table(
        ["classifier_id", "class_pair_id", "scenario_id", "scenario_family", "overall_accuracy"],
        [
            (
                row["classifier_id"],
                row["class_pair_id"],
                row["scenario_id"],
                row["scenario_family"],
                f"{float(row['overall_accuracy']):.3f}",
            )
            for row in result.class_pair_scenario_rows[:12]
        ]
    )

    doc.heading("Covariate Leakage Audit", level=2)
    doc.table(
        ["class_pair_id", "dataset_tier", "scenario_family", "true_class", "mean_duration", "measurement_std", "max_covariate_delta_ratio", "status"],
        [
            (
                row["class_pair_id"],
                row["dataset_tier"],
                row["scenario_family"],
                row["true_class"],
                f"{float(row['mean_duration']):.2f}",
                f"{float(row['measurement_std']):.2f}",
                f"{float(row['max_covariate_delta_ratio']):.2f}",
                row["status"],
            )
            for row in result.covariate_rows[:12]
        ]
    )

    doc.heading("Feature Excitation Matrix", level=2)
    doc.table(
        ["class_pair_id", "dataset_tier", "scenario_family", "feature_set_id", "num_rows", "position_range_mean_abs", "curvature_proxy_mean_abs"],
        [
            (
                row["class_pair_id"],
                row["dataset_tier"],
                row["scenario_family"],
                row["feature_set_id"],
                int(row["num_rows"]),
                f"{float(row['feature_means']['position_range']):.2f}",
                f"{float(row['feature_means']['curvature_proxy']):.2f}",
            )
            for row in result.feature_excitation_rows[:12]
        ]
    )

    doc.heading("Identifiability Matrix", level=2)
    doc.table(
        ["class_pair_id", "feature_set_id", "mean_standardized_feature_distance", "overlap_estimate", "identifiability_status"],
        [
            (
                row["class_pair_id"],
                row["feature_set_id"],
                f"{float(row['mean_standardized_feature_distance']):.3f}",
                f"{float(row['overlap_estimate']):.3f}",
                row["identifiability_status"],
            )
            for row in result.identifiability_rows[:18]
        ]
    )

    doc.heading("Oracle Separability Baseline", level=2)
    doc.table(
        ["class_pair_id", "feature_set_id", "oracle_accuracy", "mean_confidence", "mean_posterior_margin", "is_best_feature_set"],
        [
            (
                row["class_pair_id"],
                row["feature_set_id"],
                f"{float(row['oracle_accuracy']):.3f}",
                f"{float(row['mean_confidence']):.3f}",
                f"{float(row['mean_posterior_margin']):.3f}",
                str(row["is_best_feature_set"]),
            )
            for row in result.oracle_rows[:18]
        ]
    )

    doc.heading("Notes", level=2)
    doc.bullet_list(
        [
            "This artifact keeps the M10 contract honest by emitting one unified run folder from the experiment manifests.",
            "The executable subset now includes a hard shape pair: `maneuver_vs_bounded_acceleration`.",
            "`unified_likelihood_history.csv` currently stores standardized log-likelihood proxies so every classifier family can share one artifact surface.",
            "The shared `comparison` block remains the common binary study used elsewhere in the repo; the pairwise outputs here are the manifest-aligned executable subset.",
            "The M11 additions promote feature-bundle comparison and pairwise duration/scenario slices to explicit artifacts instead of leaving them implicit in the raw prediction table.",
            "The M13 additions audit class-linked covariates by tier and preserve dataset-tier context in the feature excitation matrix.",
            "The common run folder now emits `identifiability_matrix.csv`, `report.md`, and a minimal `plots/` pack so the experiment artifact surface matches the roadmap contract more closely.",
            "The M14 oracle rows are now feature-only leave-one-out separability baselines, independent of the production classifier ladder.",
            "The M15 irregular-window study compares fixed sample-count windows against elapsed-time windows on irregularly sampled trajectories.",
        ]
    )

    return doc.text()
