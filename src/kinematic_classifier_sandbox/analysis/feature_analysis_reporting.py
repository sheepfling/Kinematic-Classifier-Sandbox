from __future__ import annotations

from kinematic_classifier_sandbox.reports.markdown import MarkdownDocument


def render_feature_analysis_report(result) -> str:
    report = MarkdownDocument("Feature Excitation and Identifiability")
    report.paragraph(
        "This report summarizes feature excitation coverage and pairwise class separability from the synthetic trajectory generator."
    )
    report.heading("Summary", level=2)
    report.bullet_list(
        [
            f"Trajectories analyzed: {result.summary.total_trajectories}",
            f"Feature set: {result.summary.feature_set_name}",
            f"Active features: {', '.join(result.summary.feature_names)}",
            f"Top features: {', '.join(result.summary.top_features)}",
            f"Top separating pairs: {', '.join(f'{a} vs {b}' for a, b in result.summary.top_separating_pairs)}",
            f"Top confusing pairs: {', '.join(f'{a} vs {b}' for a, b in result.summary.top_confusing_pairs)}",
            f"Caveat status: {result.summary.caveat_status}",
            f"Caveat warnings: {result.summary.caveat_warning_count}",
        ]
    )

    report.heading("Feature Excitation", level=2)
    report.table(
        ["feature", "not_excited", "weak", "moderate", "strong"],
        [
            (
                feature_name,
                counts["not_excited"],
                counts["weak"],
                counts["moderate"],
                counts["strong"],
            )
            for feature_name in result.summary.feature_names
            if (counts := result.summary.excitation_totals[feature_name])
        ],
    )

    report.heading("Pairwise Separability", level=2)
    report.table(
        [
            "class_a",
            "class_b",
            "pairwise_auc",
            "overlap",
            "mahalanobis",
            "bhattacharyya",
            "js_divergence",
            "wasserstein",
        ],
        [
            (
                row["class_a"],
                row["class_b"],
                f"{row['pairwise_auc']:.3f}",
                f"{row['overlap_estimate']:.3f}",
                f"{row['mahalanobis_distance']:.3f}",
                f"{row['bhattacharyya_distance']:.3f}",
                f"{row['js_divergence']:.3f}",
                f"{row['wasserstein_distance']:.3f}",
            )
            for row in sorted(result.pairwise_rows, key=lambda item: (item["class_a"], item["class_b"]))
        ],
    )

    report.heading("Feature Ranking", level=2)
    report.table(
        [
            "feature",
            "mean_abs_cohens_d",
            "avg_pairwise_auc",
            "max_pairwise_auc",
            "min_pairwise_auc",
        ],
        [
            (
                row["feature"],
                f"{row['mean_abs_cohens_d']:.3f}",
                f"{row['avg_pairwise_auc']:.3f}",
                f"{row['max_pairwise_auc']:.3f}",
                f"{row['min_pairwise_auc']:.3f}",
            )
            for row in sorted(result.feature_separation_rows, key=lambda item: item["avg_pairwise_auc"], reverse=True)
        ],
    )

    report.heading("Evidence Caveats", level=2)
    report.table(
        ["feature", "history_behavior", "caveat_types", "status"],
        [
            (
                row["feature"],
                row["history_behavior"],
                row["caveat_types"],
                row["status"],
            )
            for row in result.caveat_rows
        ],
    )
    report.bullet_list(
        [
            "History-bearing features are flagged so cumulative or windowed evidence is not treated as interchangeable with memoryless evidence.",
            "Correlated bundles are surfaced explicitly when multiple dependency tags suggest overlap or double-counting risk.",
            "This is a reporting layer only; it governs interpretation without redesigning the underlying classifiers.",
        ]
    )

    report.heading("Validation Notes", level=2)
    report.bullet_list(
        [
            "Excitation matrices show which synthetic scenarios actually stress each feature.",
            "Pairwise metrics distinguish clearly separated classes from intentionally confusable ones.",
            "The same extracted features can be reused by downstream classifier experiments.",
        ]
    )
    return report.text()


__all__ = ["render_feature_analysis_report"]
