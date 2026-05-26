from __future__ import annotations

from pathlib import Path

from ..utils.math import _matvec
from ..utils.math import mean as _mean
from ..utils.math import std as _std
from ..utils.plotting import write_plot
from ..utils.plotting import plt
from .contracts import CommonExperimentResult


def write_common_experiment_plot_pack(
    run_dir: Path,
    *,
    result: CommonExperimentResult,
) -> Path:
    plots_dir = run_dir / "plots"
    overview_dir = plots_dir / "overview"
    trajectory_dir = plots_dir / "single_trajectory_examples"
    posterior_dir = plots_dir / "posteriors"
    likelihood_dir = plots_dir / "likelihoods"
    confusion_dir = plots_dir / "confusion_matrices"
    monte_carlo_dir = plots_dir / "monte_carlo"
    feature_dir = plots_dir / "feature_space"
    prior_dir = plots_dir / "priors"
    pca_dir = plots_dir / "pca"
    class_pair_dir = plots_dir / "class_pair_reports"
    for directory in (
        overview_dir,
        trajectory_dir,
        posterior_dir,
        likelihood_dir,
        confusion_dir,
        monte_carlo_dir,
        feature_dir,
        prior_dir,
        pca_dir,
        class_pair_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    scenario_counts: dict[str, int] = {}
    for row in result.pair_prediction_rows:
        scenario_counts.setdefault(str(row["scenario_id"]), 0)
        scenario_counts[str(row["scenario_id"])] += 1
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    scenario_names = list(sorted(scenario_counts))
    ax.bar(scenario_names, [scenario_counts[name] for name in scenario_names], color="#4C78A8")
    ax.set_title("Dataset Balance by Scenario")
    ax.set_ylabel("Predictions")
    ax.tick_params(axis="x", rotation=25)
    write_plot(fig, overview_dir / "dataset_balance.png")

    if result.pair_prediction_rows:
        selected_run_id = str(result.pair_prediction_rows[0]["run_id"])
        selected_posteriors = [row for row in result.posterior_history_rows if str(row["run_id"]) == selected_run_id]
        selected_likelihoods = [row for row in result.likelihood_history_rows if str(row["run_id"]) == selected_run_id]
        selected_prediction = next(row for row in result.pair_prediction_rows if str(row["run_id"]) == selected_run_id)
        selected_trajectory_id = str(selected_prediction["trajectory_id"])
        selected_features = [row for row in result.feature_rows if str(row["trajectory_id"]) == selected_trajectory_id]

        fig, ax = plt.subplots(figsize=(9.5, 4.6))
        ax.plot(
            [float(row["time"]) for row in selected_posteriors],
            [float(row["posterior_class_a"]) for row in selected_posteriors],
            label=str(selected_prediction["class_a"]),
            linewidth=2.0,
        )
        ax.plot(
            [float(row["time"]) for row in selected_posteriors],
            [float(row["posterior_class_b"]) for row in selected_posteriors],
            label=str(selected_prediction["class_b"]),
            linewidth=2.0,
        )
        ax.set_title(f"Posterior Example: {selected_run_id}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Posterior")
        ax.legend()
        write_plot(fig, posterior_dir / "posterior_example.png")

        fig, ax = plt.subplots(figsize=(9.5, 4.6))
        ax.plot(
            [float(row["time"]) for row in selected_likelihoods],
            [float(row["log_likelihood_class_a"]) for row in selected_likelihoods],
            label=str(selected_prediction["class_a"]),
            linewidth=2.0,
        )
        ax.plot(
            [float(row["time"]) for row in selected_likelihoods],
            [float(row["log_likelihood_class_b"]) for row in selected_likelihoods],
            label=str(selected_prediction["class_b"]),
            linewidth=2.0,
        )
        ax.set_title(f"Likelihood Example: {selected_run_id}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Log-likelihood proxy")
        ax.legend()
        write_plot(fig, likelihood_dir / "likelihood_example.png")

        if selected_features:
            feature_row = selected_features[0]
            feature_names = [
                "position_range",
                "speed_range",
                "acceleration_range",
                "curvature_proxy",
                "linear_fit_residual",
                "outlier_score",
            ]
            fig, ax = plt.subplots(figsize=(10.0, 4.8))
            ax.bar(feature_names, [float(feature_row[name]) for name in feature_names], color="#72B7B2")
            ax.set_title(f"Single-Trajectory Feature Snapshot: {selected_trajectory_id}")
            ax.tick_params(axis="x", rotation=25)
            write_plot(fig, trajectory_dir / "feature_snapshot.png")

    class_pairs = sorted({str(row["class_pair"]) for row in result.metrics_by_class_pair_rows})
    classifier_order = sorted({str(row["classifier_id"]) for row in result.metrics_by_class_pair_rows})
    heatmap = []
    for classifier_id in classifier_order:
        row_values = []
        for class_pair in class_pairs:
            matched = next(
                (
                    float(row["overall_accuracy"])
                    for row in result.metrics_by_class_pair_rows
                    if str(row["classifier_id"]) == classifier_id and str(row["class_pair"]) == class_pair
                ),
                0.0,
            )
            row_values.append(matched)
        heatmap.append(row_values)
    fig, ax = plt.subplots(figsize=(max(7.5, 1.5 * len(class_pairs)), max(4.0, 0.5 * len(classifier_order) + 2.0)))
    image = ax.imshow(heatmap, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(class_pairs)))
    ax.set_xticklabels(class_pairs, rotation=25, ha="right")
    ax.set_yticks(range(len(classifier_order)))
    ax.set_yticklabels(classifier_order)
    ax.set_title("Classifier Accuracy by Class Pair")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    write_plot(fig, confusion_dir / "classifier_pair_accuracy_heatmap.png")

    grouped_duration: dict[str, list[dict[str, object]]] = {}
    for row in result.class_pair_duration_rows:
        grouped_duration.setdefault(str(row["classifier_id"]), []).append(row)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for classifier_id, rows in sorted(grouped_duration.items()):
        ordered = sorted(rows, key=lambda item: float(item["time"]))
        ax.plot(
            [float(row["time"]) for row in ordered],
            [float(row["prefix_accuracy"]) for row in ordered],
            label=classifier_id,
            linewidth=1.6,
        )
    ax.set_title("Prefix Accuracy vs Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Prefix accuracy")
    ax.legend(fontsize=7, ncol=2)
    write_plot(fig, monte_carlo_dir / "prefix_accuracy_curve.png")

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ordered_ident = sorted(
        result.identifiability_rows,
        key=lambda row: (str(row["class_pair_id"]), float(row["mean_standardized_feature_distance"])),
    )
    labels = [f"{row['class_pair_id']}:{row['feature_set_id']}" for row in ordered_ident]
    ax.barh(labels, [float(row["mean_standardized_feature_distance"]) for row in ordered_ident], color="#F58518")
    ax.set_title("Identifiability by Class Pair and Feature Set")
    ax.set_xlabel("Mean standardized feature distance")
    write_plot(fig, feature_dir / "identifiability_summary.png")

    prior_grouped: dict[str, list[dict[str, object]]] = {}
    prior_order = {"uniform": 0, "mild_bias": 1, "strong_bias": 2}
    for row in result.prior_sensitivity_rows:
        prior_grouped.setdefault(f"{row['classifier_id']}:{row['class_pair_id']}", []).append(row)
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    for label, rows in sorted(prior_grouped.items()):
        ordered = sorted(rows, key=lambda item: prior_order.get(str(item["prior_id"]), 99))
        ax.plot(
            [str(row["prior_id"]) for row in ordered],
            [float(row["accuracy"]) for row in ordered],
            marker="o",
            linewidth=1.5,
            label=label,
        )
    ax.set_title("Prior Sensitivity by Classifier and Pair")
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=6, ncol=2)
    write_plot(fig, prior_dir / "prior_sensitivity.png")

    pca_feature_names = [
        "position_range",
        "speed_range",
        "acceleration_range",
        "curvature_proxy",
        "linear_fit_residual",
        "quadratic_fit_residual",
    ]
    centered_rows = []
    metadata = []
    for row in result.feature_rows:
        centered_rows.append([float(row[name]) for name in pca_feature_names])
        metadata.append(str(row["true_class"]))
    means = [_mean([values[index] for values in centered_rows]) for index in range(len(pca_feature_names))]
    stds = [max(_std([values[index] for values in centered_rows]), 1e-9) for index in range(len(pca_feature_names))]
    standardized = [
        [(values[index] - means[index]) / stds[index] for index in range(len(pca_feature_names))]
        for values in centered_rows
    ]
    covariance = [[0.0 for _ in pca_feature_names] for _ in pca_feature_names]
    for values in standardized:
        for row_index in range(len(pca_feature_names)):
            for col_index in range(len(pca_feature_names)):
                covariance[row_index][col_index] += values[row_index] * values[col_index]
    denom = max(len(standardized) - 1, 1)
    for row_index in range(len(pca_feature_names)):
        for col_index in range(len(pca_feature_names)):
            covariance[row_index][col_index] /= denom

    def _norm(vector: list[float]) -> float:
        return sum(value * value for value in vector) ** 0.5

    def _normalize(vector: list[float]) -> list[float]:
        norm = max(_norm(vector), 1e-12)
        return [value / norm for value in vector]

    def _dot(left: list[float], right: list[float]) -> float:
        return sum(left[index] * right[index] for index in range(len(left)))

    def _power_iteration(matrix: list[list[float]]) -> tuple[float, list[float]]:
        vector = _normalize([1.0 + index for index in range(len(matrix))])
        previous = 0.0
        for _ in range(200):
            vector = _normalize(_matvec(matrix, vector))
            value = _dot(vector, _matvec(matrix, vector))
            if abs(value - previous) <= 1e-9:
                return value, vector
            previous = value
        return previous, vector

    eig1, vec1 = _power_iteration(covariance)
    deflated = [
        [covariance[row][col] - eig1 * vec1[row] * vec1[col] for col in range(len(covariance))]
        for row in range(len(covariance))
    ]
    _, vec2 = _power_iteration(deflated)
    coords = [(_dot(values, vec1), _dot(values, vec2)) for values in standardized]
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    class_names = sorted(set(metadata))
    palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]
    for index, class_name in enumerate(class_names):
        xs = [coord[0] for coord, label in zip(coords, metadata) if label == class_name]
        ys = [coord[1] for coord, label in zip(coords, metadata) if label == class_name]
        ax.scatter(xs, ys, s=28, alpha=0.75, label=class_name, color=palette[index % len(palette)])
    ax.set_title("Feature PCA Snapshot")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=7)
    write_plot(fig, pca_dir / "feature_pca_snapshot.png")

    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    ordered_pairs = sorted({str(row["class_pair_id"]) for row in result.class_pair_scenario_rows})
    best_by_pair = []
    for pair_id in ordered_pairs:
        rows = [row for row in result.metrics_by_class_pair_rows if str(row["class_pair"]) == pair_id]
        best_by_pair.append(max((float(row["overall_accuracy"]) for row in rows), default=0.0))
    ax.bar(ordered_pairs, best_by_pair, color="#54A24B")
    ax.set_title("Best Accuracy by Class Pair")
    ax.set_ylabel("Accuracy")
    ax.tick_params(axis="x", rotation=25)
    write_plot(fig, class_pair_dir / "best_accuracy_by_pair.png")

    return plots_dir
