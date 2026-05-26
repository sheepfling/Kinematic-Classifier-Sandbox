from __future__ import annotations

from ..markdown_builder import MarkdownDocument
from .common_dataset_comparison_contracts import CommonComparisonResult


def render_common_dataset_comparison_report(result: CommonComparisonResult) -> str:
    report = MarkdownDocument("Common-Dataset Technique Comparison")
    report.paragraph(
        "This artifact evaluates the current technique families on the same shared binary dynamics corpus: "
        "constant velocity versus constant acceleration with easy, irregular-`dt`, matched-endpoint irregular, "
        "short-horizon boundary, short-horizon noisy, and outlier-corrupted scenarios."
    )
    report.heading("Method Metrics", level=2)
    report.table(
        [
            "method",
            "overall",
            "easy",
            "irregular",
            "endpoint_match",
            "short",
            "short_noisy",
            "outlier",
            "prior_flip_fraction",
        ],
        [
            (
                row.method_name,
                f"{row.overall_accuracy:.3f}",
                f"{row.easy_accuracy:.3f}",
                f"{row.irregular_accuracy:.3f}",
                f"{row.endpoint_match_accuracy:.3f}",
                f"{row.short_accuracy:.3f}",
                f"{row.noisy_accuracy:.3f}",
                f"{row.outlier_accuracy:.3f}",
                f"{row.prior_flip_fraction:.3f}",
            )
            for row in result.rows
        ],
    )
    report.heading("Interpretation", level=2)
    report.bullet_list(
        [
            "This is the first apples-to-apples technique comparison on one shared corpus.",
            "Pointwise should act as the weak lower bound because it only uses the last measurement.",
            "The matched-endpoint irregular case removes most endpoint information, so methods need the time history rather than just the last sample.",
            "Short-horizon cases are boundary cases: there is not much elapsed time for acceleration to separate from constant velocity.",
            "The outlier case is there to expose the difference between raw feature accumulation and more robust temporal/model-based methods.",
            "`kalman_bank` remains a position-only sensing regime, even when it uses derived pseudo-observations.",
            "`kalman_bank_velocity_aided` is a separate sensor regime with an actual extra velocity stream.",
        ]
    )
    return report.text()


__all__ = ["render_common_dataset_comparison_report"]
