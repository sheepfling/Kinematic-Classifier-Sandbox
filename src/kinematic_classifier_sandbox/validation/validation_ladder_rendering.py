from __future__ import annotations

from ..markdown_builder import MarkdownDocument
from .validation_ladder_contracts import ValidationLadderArtifacts, ValidationLadderResult


def render_validation_ladder_report(result: ValidationLadderResult) -> str:
    decision_counts: dict[str, int] = {}
    for row in result.decision_rows:
        decision_counts[str(row["final_decision"])] = decision_counts.get(str(row["final_decision"]), 0) + 1

    report = MarkdownDocument()
    report.heading("Validation Ladder", level=1)
    report.paragraph(
        "This artifact collapses the protocol, selected corpus, candidate generator, and common-study evidence "
        "into one ten-level ladder per canonical study."
    )
    report.heading("Decision Counts", level=2)
    report.bullet_list(
        [
            f"Promote: `{decision_counts.get('promote', 0)}`",
            f"Revise: `{decision_counts.get('revise', 0)}`",
            f"Reject: `{decision_counts.get('reject', 0)}`",
            f"Defer: `{decision_counts.get('defer', 0)}`",
        ]
    )
    report.heading("Top Decisions", level=2)
    report.bullet_list(
        [
            (
                f"`{row['study_id']}` -> `{row['final_decision']}` "
                f"(acc={float(row['classifier_accuracy']):.3f}, static={float(row['static_score']):.3f}, prior={float(row['prior_sensitivity_score']):.3f})"
            )
            for row in sorted(
                result.decision_rows,
                key=lambda row: (
                    {"promote": 0, "revise": 1, "defer": 2, "reject": 3}[str(row["final_decision"])],
                    -float(row["classifier_accuracy"]),
                    -float(row["static_score"]),
                ),
            )[:12]
        ]
    )
    report.heading("Notes", level=2)
    report.bullet_list(
        [
            "The canonical ladder is built on the `uniform` prior study rows so prior sensitivity stays a dedicated level instead of tripling the study list.",
            "Corpus adequacy is pair-aware and uses the selected M19 corpus rather than a manual judgment.",
            "Dimensional transfer is assessed explicitly but does not automatically block 1D promotion when the rest of the evidence is strong.",
        ]
    )
    return report.text()

__all__ = [
    "ValidationLadderArtifacts",
    "render_validation_ladder_report",
]
