from __future__ import annotations

from ...markdown_builder import MarkdownDocument
from .contracts import VelocityAidedComparisonResult


def render_velocity_aided_kalman_comparison_report(result: VelocityAidedComparisonResult) -> str:
    report = MarkdownDocument("Velocity-Aided Kalman Comparison")
    report.paragraph("This artifact compares the same robust/adaptive Kalman family under two sensor stacks on the shared corpus.")
    report.table(
        ["measurement_mode", "overall", "endpoint_match", "short", "short_noisy", "outlier"],
        [
            (
                row.measurement_mode,
                f"{row.overall_accuracy:.3f}",
                f"{row.endpoint_match_accuracy:.3f}",
                f"{row.short_accuracy:.3f}",
                f"{row.short_noisy_accuracy:.3f}",
                f"{row.outlier_accuracy:.3f}",
            )
            for row in result.rows
        ],
    )
    report.heading("Interpretation", level=2)
    report.bullet_list(
        [
            "`position_only` is the baseline position-measurement Kalman bank.",
            "`position_plus_direct_velocity` adds an actual velocity sensor stream rather than a pseudo-observation derived from the same positions.",
            "This isolates the value of genuinely stronger sensing from cleverer reuse of the same data.",
        ]
    )
    return report.text()
