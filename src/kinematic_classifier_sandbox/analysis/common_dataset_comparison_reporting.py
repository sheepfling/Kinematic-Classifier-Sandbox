from __future__ import annotations

from ..markdown_builder import MarkdownDocument
from .common_dataset_comparison_contracts import CommonComparisonResult


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def render_common_dataset_comparison_report(result: CommonComparisonResult) -> str:
    report = MarkdownDocument("Common-Dataset Technique Comparison")
    report.paragraph(
        "This artifact evaluates the shared classifier family against the same binary dynamics corpus while remaining capability-aware. "
        "Methods that belong to advanced nonlinear, switching, or stochastic-mean-reversion families stay visible in the table, but they are marked witness-only instead of being forced onto a mismatched shared corpus."
    )
    report.heading("Method Metrics", level=2)
    report.table(
        [
            "method",
            "status",
            "primary_family",
            "overall",
            "easy",
            "irregular",
            "endpoint_match",
            "short",
            "short_noisy",
            "outlier",
            "prior_flip_fraction",
            "witness_artifact",
        ],
        [
            (
                row.method_name,
                row.applicability_status,
                row.primary_evaluation_family,
                _fmt(row.overall_accuracy),
                _fmt(row.easy_accuracy),
                _fmt(row.irregular_accuracy),
                _fmt(row.endpoint_match_accuracy),
                _fmt(row.short_accuracy),
                _fmt(row.noisy_accuracy),
                _fmt(row.outlier_accuracy),
                _fmt(row.prior_flip_fraction),
                row.witness_artifact or "n/a",
            )
            for row in result.rows
        ],
    )
    report.heading("Interpretation", level=2)
    report.bullet_list(
        [
            "The shared binary corpus remains the apples-to-apples scorecard for pointwise, windowed, accumulator, and Kalman-family methods.",
            "PF, RBPF, and the OU witness remain part of the generic classifier story, but their evidence is attached through witness artifacts and applicability notes.",
            "This keeps one comparison vocabulary without pretending every method belongs on every benchmark.",
        ]
    )
    return report.text()


__all__ = ["render_common_dataset_comparison_report"]
