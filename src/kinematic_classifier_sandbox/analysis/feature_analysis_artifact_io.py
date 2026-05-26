from __future__ import annotations

import io
import json
from dataclasses import asdict
from pathlib import Path

from kinematic_classifier_sandbox.utils.io import write_csv

from ..utils.plotting import _figure_to_png
from ..utils.plotting import plt
from .feature_analysis_contracts import FeatureAnalysisArtifacts
from .feature_analysis_reporting import render_feature_analysis_report


def _render_heatmap(
    matrix: list[list[float]],
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    cmap: str = "Blues",
):
    fig, ax = plt.subplots(figsize=(8.5, 6.8))
    image = ax.imshow(matrix, cmap=cmap)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            ax.text(col_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def _render_feature_scatter(result):
    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    x_feature = result.summary.top_features[0]
    y_feature = result.summary.top_features[1] if len(result.summary.top_features) > 1 else result.summary.top_features[0]
    class_names = sorted({row.true_class for row in result.feature_rows})
    colors = {
        name: color
        for name, color in zip(
            class_names,
            ("#2563eb", "#dc2626", "#16a34a", "#7c3aed", "#d97706", "#0f766e", "#db2777"),
        )
    }
    top_confusing = set(result.summary.top_confusing_pairs[0]) if result.summary.top_confusing_pairs else set()
    for class_name in class_names:
        class_rows = [row for row in result.feature_rows if row.true_class == class_name]
        xs = [row.feature_values.get(x_feature, 0.0) for row in class_rows]
        ys = [row.feature_values.get(y_feature, 0.0) for row in class_rows]
        alpha = 0.95 if class_name in top_confusing else 0.45
        size = 52 if class_name in top_confusing else 28
        label = f"{class_name} (top confusing pair)" if class_name in top_confusing else class_name
        ax.scatter(xs, ys, s=size, alpha=alpha, color=colors[class_name], edgecolors="white", linewidths=0.4, label=label)
    ax.set_title("Feature Space Map for Confusable Classes", loc="left", fontweight="bold")
    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, fontsize=8, ncols=2)
    fig.tight_layout()
    return fig


def _render_confusability_heatmap(class_names: list[str], confusability_matrix: list[list[float]]):
    return _render_heatmap(
        confusability_matrix,
        class_names,
        class_names,
        "Class Confusability Map (1 - Pairwise AUC)",
        cmap="Reds",
    )


def _render_feature_ranking_summary(result):
    ordered_rows = sorted(
        result.feature_separation_rows,
        key=lambda item: float(item["avg_pairwise_auc"]),
        reverse=True,
    )[:10]
    feature_names = [str(row["feature"]) for row in ordered_rows][::-1]
    avg_auc = [float(row["avg_pairwise_auc"]) for row in ordered_rows][::-1]
    min_auc = [float(row["min_pairwise_auc"]) for row in ordered_rows][::-1]
    max_auc = [float(row["max_pairwise_auc"]) for row in ordered_rows][::-1]
    mean_d = [float(row["mean_abs_cohens_d"]) for row in ordered_rows][::-1]

    fig, ax = plt.subplots(figsize=(9.0, max(4.8, 0.46 * len(feature_names) + 1.8)))
    positions = list(range(len(feature_names)))
    ax.barh(positions, avg_auc, color="#2563eb", alpha=0.86, label="avg pairwise AUC")
    ax.scatter(min_auc, positions, color="#dc2626", s=34, label="min pairwise AUC", zorder=3)
    ax.scatter(max_auc, positions, color="#16a34a", s=34, label="max pairwise AUC", zorder=3)
    for index, value in enumerate(avg_auc):
        ax.text(min(value + 0.01, 0.995), index, f"{value:.2f}", va="center", fontsize=8)
    ax.set_title("Top Feature Ranking Summary", loc="left", fontweight="bold")
    ax.set_xlabel("pairwise AUC")
    ax.set_ylabel("feature")
    ax.set_xlim(0.45, 1.02)
    ax.set_yticks(positions)
    ax.set_yticklabels(feature_names)
    ax.grid(True, axis="x", alpha=0.2)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    twin = ax.twiny()
    twin.plot(mean_d, positions, color="#7c3aed", linewidth=1.5, marker="o", markersize=3.5, label="mean |Cohen's d|")
    twin.set_xlim(0.0, max(mean_d + [1.0]) * 1.08)
    twin.set_xlabel("mean |Cohen's d|")
    fig.tight_layout()
    return fig


def write_feature_analysis_artifacts(
    output_dir: str | Path,
    *,
    seed: int = 7,
    trajectories_per_class: int = 5,
    feature_set: str | None = None,
    feature_names: tuple[str, ...] | list[str] | None = None,
) -> FeatureAnalysisArtifacts:
    from .feature_analysis import (
        FEATURE_ROW_METADATA_FIELDNAMES,
        analyze_feature_datasets,
    )

    result = analyze_feature_datasets(
        seed=seed,
        trajectories_per_class=trajectories_per_class,
        feature_set=feature_set,
        feature_names=feature_names,
    )
    output_root = Path(output_dir)
    run_dir_name = (
        "feature_analysis_v1"
        if result.summary.feature_set_name == "all_engineered"
        else f"feature_analysis_{result.summary.feature_set_name}_v1"
    )
    run_dir = output_root / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_dir / "feature_analysis_report.md"
    feature_matrix_path = run_dir / "feature_matrix.csv"
    feature_summary_path = run_dir / "feature_summary_by_class.csv"
    feature_excitation_path = run_dir / "feature_excitation_matrix.csv"
    feature_excitation_summary_path = run_dir / "feature_excitation_summary.json"
    feature_separation_scores_path = run_dir / "feature_separation_scores.csv"
    identifiability_matrix_path = run_dir / "identifiability_matrix.csv"
    pairwise_distance_matrix_path = run_dir / "pairwise_distance_matrix.csv"
    pairwise_overlap_matrix_path = run_dir / "pairwise_overlap_matrix.csv"
    pairwise_auc_matrix_path = run_dir / "pairwise_auc_matrix.csv"
    plot_excitation_png_path = run_dir / "feature_excitation_heatmap.png"
    plot_distance_png_path = run_dir / "pairwise_distance_heatmap.png"
    plot_overlap_png_path = run_dir / "pairwise_overlap_heatmap.png"
    plot_scatter_png_path = run_dir / "feature_space_confusion_map.png"
    plot_confusability_png_path = run_dir / "class_confusability_heatmap.png"
    plot_ranking_png_path = run_dir / "feature_ranking_summary.png"

    report_path.write_text(render_feature_analysis_report(result), encoding="utf-8")
    write_csv(
        feature_matrix_path,
        [row.as_flat_dict(result.summary.feature_names) for row in result.feature_rows],
        [*FEATURE_ROW_METADATA_FIELDNAMES, *result.summary.feature_names],
    )
    write_csv(feature_summary_path, [dict(row) for row in result.summary_rows], ["true_class", "feature", "mean", "std", "median", "iqr", "min", "max", "p05", "p95", "missing_rate"])
    write_csv(
        feature_excitation_path,
        [dict(row) for row in result.excitation_rows],
        [*FEATURE_ROW_METADATA_FIELDNAMES, *result.summary.feature_names, *[f"{name}_level" for name in result.summary.feature_names]],
    )
    feature_excitation_summary_path.write_text(json.dumps(asdict(result.summary), indent=2, sort_keys=True), encoding="utf-8")
    write_csv(feature_separation_scores_path, [dict(row) for row in result.feature_separation_rows], ["feature", "mean_abs_cohens_d", "avg_pairwise_auc", "max_pairwise_auc", "min_pairwise_auc"])
    write_csv(identifiability_matrix_path, [dict(row) for row in result.pairwise_rows], ["class_a", "class_b", "mean_feature_distance", "standardized_mean_difference", "mahalanobis_distance", "bhattacharyya_distance", "js_divergence", "wasserstein_distance", "overlap_estimate", "pairwise_classifier_accuracy", "average_log_likelihood_ratio", "pairwise_auc"])

    class_names = sorted({row.true_class for row in result.feature_rows})
    distance_matrix = [[0.0 for _ in class_names] for _ in class_names]
    overlap_matrix = [[0.0 for _ in class_names] for _ in class_names]
    auc_matrix = [[0.5 for _ in class_names] for _ in class_names]
    confusability_matrix = [[0.0 for _ in class_names] for _ in class_names]
    row_lookup = {(row["class_a"], row["class_b"]): row for row in result.pairwise_rows}
    row_lookup.update({(row["class_b"], row["class_a"]): row for row in result.pairwise_rows})
    for i, class_a in enumerate(class_names):
        for j, class_b in enumerate(class_names):
            if class_a == class_b:
                auc_matrix[i][j] = 1.0
                continue
            row = row_lookup[(class_a, class_b)]
            distance_matrix[i][j] = float(row["mahalanobis_distance"])
            overlap_matrix[i][j] = float(row["overlap_estimate"])
            auc_matrix[i][j] = float(row["pairwise_auc"]) if row["class_a"] == class_a else 1.0 - float(row["pairwise_auc"])
            confusability_matrix[i][j] = 1.0 - auc_matrix[i][j]
    write_csv(
        pairwise_distance_matrix_path,
        [{"class": class_names[index], **{class_names[col_index]: distance_matrix[index][col_index] for col_index in range(len(class_names))}} for index in range(len(class_names))],
        ["class", *class_names],
    )
    write_csv(
        pairwise_overlap_matrix_path,
        [{"class": class_names[index], **{class_names[col_index]: overlap_matrix[index][col_index] for col_index in range(len(class_names))}} for index in range(len(class_names))],
        ["class", *class_names],
    )
    write_csv(
        pairwise_auc_matrix_path,
        [{"class": class_names[index], **{class_names[col_index]: auc_matrix[index][col_index] for col_index in range(len(class_names))}} for index in range(len(class_names))],
        ["class", *class_names],
    )

    excitation_matrix = [[float(result.summary.excitation_totals[feature][level]) for feature in result.summary.feature_names] for level in ("not_excited", "weak", "moderate", "strong")]
    plot_excitation_png_path.write_bytes(
        _figure_to_png(
            _render_heatmap(
                excitation_matrix,
                ["not_excited", "weak", "moderate", "strong"],
                list(result.summary.feature_names),
                "Feature Excitation Totals",
                cmap="viridis",
            )
        )
    )
    plot_distance_png_path.write_bytes(
        _figure_to_png(
            _render_heatmap(distance_matrix, class_names, class_names, "Pairwise Mahalanobis Distance", cmap="Blues")
        )
    )
    plot_overlap_png_path.write_bytes(
        _figure_to_png(
            _render_heatmap(overlap_matrix, class_names, class_names, "Pairwise Overlap Estimate", cmap="Oranges")
        )
    )
    plot_scatter_png_path.write_bytes(_figure_to_png(_render_feature_scatter(result)))
    plot_confusability_png_path.write_bytes(
        _figure_to_png(_render_confusability_heatmap(class_names, confusability_matrix))
    )
    plot_ranking_png_path.write_bytes(_figure_to_png(_render_feature_ranking_summary(result)))

    return FeatureAnalysisArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        feature_matrix_path=feature_matrix_path,
        feature_summary_path=feature_summary_path,
        feature_excitation_path=feature_excitation_path,
        feature_excitation_summary_path=feature_excitation_summary_path,
        feature_separation_scores_path=feature_separation_scores_path,
        identifiability_matrix_path=identifiability_matrix_path,
        pairwise_distance_matrix_path=pairwise_distance_matrix_path,
        pairwise_overlap_matrix_path=pairwise_overlap_matrix_path,
        pairwise_auc_matrix_path=pairwise_auc_matrix_path,
        plot_excitation_png_path=plot_excitation_png_path,
        plot_distance_png_path=plot_distance_png_path,
        plot_overlap_png_path=plot_overlap_png_path,
        plot_scatter_png_path=plot_scatter_png_path,
        plot_confusability_png_path=plot_confusability_png_path,
        plot_ranking_png_path=plot_ranking_png_path,
    )


__all__ = ["write_feature_analysis_artifacts"]
