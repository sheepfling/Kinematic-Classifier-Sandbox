from __future__ import annotations

from collections.abc import Mapping
from statistics import mean
from typing import Any

from .utils.plotting import plt


def _render_static_vs_statistical_score(result: Any):
    static_lookup = {str(row["study_id"]): float(row["static_score"]) for row in result.static_score_rows}
    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    for row in result.monte_carlo_score_rows:
        if row["monte_carlo_score"] is None:
            continue
        decision = str(row["decision"])
        color = {"promote": "#16a34a", "revise": "#d97706", "reject": "#dc2626", "defer": "#6b7280"}[decision]
        ax.scatter(static_lookup[str(row["study_id"])], float(row["monte_carlo_score"]), color=color, s=42, alpha=0.8)
    ax.set_xlabel("Static score")
    ax.set_ylabel("Monte Carlo score")
    ax.set_title("Static vs Statistical Candidate Score", loc="left", fontweight="bold")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def _render_candidate_promotion_matrix(result: Any):
    classifiers = sorted({str(row["classifier_id"]) for row in result.monte_carlo_score_rows})
    feature_sets = sorted({str(row["feature_set_id"]) for row in result.monte_carlo_score_rows})
    score_map = {"promote": 3, "revise": 2, "defer": 1, "reject": 0}
    matrix = []
    for classifier_id in classifiers:
        row_values = []
        for feature_set_id in feature_sets:
            candidates = [
                score_map[str(row["decision"])]
                for row in result.monte_carlo_score_rows
                if str(row["classifier_id"]) == classifier_id and str(row["feature_set_id"]) == feature_set_id
            ]
            row_values.append(max(candidates) if candidates else 0)
        matrix.append(row_values)
    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    image = ax.imshow(matrix, cmap="YlGn", aspect="auto", vmin=0, vmax=3)
    ax.set_xticks(range(len(feature_sets)))
    ax.set_xticklabels(feature_sets, rotation=30, ha="right")
    ax.set_yticks(range(len(classifiers)))
    ax.set_yticklabels(classifiers)
    ax.set_title("Candidate Promotion Matrix", loc="left", fontweight="bold")
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            label = {0: "R", 1: "D", 2: "V", 3: "P"}[value]
            ax.text(col_index, row_index, label, ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def _render_classifier_feature_class_heatmap(result: Any):
    promoted = result.promoted_rows or result.monte_carlo_score_rows[:20]
    class_pairs = sorted({str(row["class_pair_id"]) for row in promoted})
    classifier_feature_keys = sorted({f"{row['classifier_id']}:{row['feature_set_id']}" for row in promoted})
    matrix = []
    for key in classifier_feature_keys:
        classifier_id, feature_set_id = key.split(":", 1)
        row_values = []
        for class_pair_id in class_pairs:
            matches = [
                float(row["accuracy"])
                for row in promoted
                if str(row["classifier_id"]) == classifier_id
                and str(row["feature_set_id"]) == feature_set_id
                and str(row["class_pair_id"]) == class_pair_id
                and row["accuracy"] is not None
            ]
            row_values.append(mean(matches) if matches else 0.0)
        matrix.append(row_values)
    fig, ax = plt.subplots(figsize=(10.4, 6.2))
    image = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(class_pairs)))
    ax.set_xticklabels(class_pairs, rotation=30, ha="right")
    ax.set_yticks(range(len(classifier_feature_keys)))
    ax.set_yticklabels(classifier_feature_keys)
    ax.set_title("Classifier + Feature + Class-Pair Performance", loc="left", fontweight="bold")
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            ax.text(col_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig
