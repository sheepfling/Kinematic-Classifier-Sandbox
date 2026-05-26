from __future__ import annotations

from ..markdown_builder import MarkdownDocument
from .contracts import IMMBenchmarkResult


def render_imm_report(result: IMMBenchmarkResult) -> str:
    metrics = result.metrics
    doc = MarkdownDocument("IMM Filter V1 Report")
    doc.paragraph(
        "This is the IMM switching-witness report for the evaluation-first advanced-filter rung. PF and RBPF are evaluated in their own nonlinear and latent-mode witness reports, then summarized with IMM in `artifacts/advanced_filter_comparison_v1`."
    )

    doc.heading("Evaluation Target", level=2)
    doc.bullet_list(
        [
            "Failure case: static class/mode assumption under switching dynamics.",
            "Baselines: static mode likelihood and transition-matrix accumulator.",
            "Advanced method: IMM with state mixing and one Kalman filter per mode.",
        ]
    )

    doc.heading("Metrics", level=2)
    doc.bullet_list(
        [
            f"Mode accuracy: `{metrics['mode_accuracy']}`",
            f"Post-switch accuracy: `{metrics['post_switch_accuracy']}`",
            f"Switch detection delay median: `{metrics['switch_detection_delay_median']}`",
            f"Mode NLL: `{metrics['mode_nll']}`",
            f"Mean entropy: `{metrics['mean_entropy']}`",
            f"State position RMSE: `{metrics['state_position_rmse']}`",
            f"State velocity RMSE: `{metrics['state_velocity_rmse']}`",
            f"Decision: `{metrics['promotion_decision']}`",
        ]
    )

    doc.heading("Method Comparison", level=2)
    doc.table(
        ["Method", "Mode accuracy", "Post-switch accuracy", "Decision"],
        [
            (row["method_id"], row["mode_accuracy"], row["post_switch_accuracy"], row["promotion_decision"])
            for row in result.method_comparison
        ],
    )

    doc.heading("Required Interpretation Order", level=2)
    doc.ordered_list(
        [
            "Confirm switching witness corpus coverage.",
            "Inspect mode evidence and posterior history.",
            "Compare post-switch accuracy and switch delay.",
            "Check state RMSE and entropy.",
            "Assign promote/revise/reject/defer.",
        ]
    )

    doc.paragraph(
        "PF and RBPF decisions are intentionally left to `artifacts/advanced_filter_comparison_v1`, where their targeted witness metrics are compared beside IMM."
    )

    return doc.text()
