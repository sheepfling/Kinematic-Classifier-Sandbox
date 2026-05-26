from __future__ import annotations

from ..markdown_builder import MarkdownDocument
from ..utils.plotting import plt
from .technique_comparison_contracts import TechniqueComparisonResult, TechniqueComparisonRow


def render_technique_comparison_report(result: TechniqueComparisonResult) -> str:
    report = MarkdownDocument("Technique Comparison Study")
    report.paragraph(
        "This package compares the current baseline techniques using existing benchmark suites, prior-sensitivity diagnostics, and scenario-family stress cases. "
        "The numbers are useful for debugging and roadmap decisions, but they are not yet a single common-dataset bakeoff."
    )
    report.heading("Summary Metrics", level=2)
    report.table(
        ["method", "sensor_regime", "overall_accuracy", "prior_flip_fraction", "median_flip_threshold"],
        [
            (
                row.method_name,
                row.sensor_regime_id,
                f"{row.overall_accuracy:.3f}",
                f"{row.prior_flip_fraction:.3f}",
                "n/a" if row.median_flip_threshold is None else f"{row.median_flip_threshold:.2f}",
            )
            for row in result.rows
        ],
    )
    report.heading("Scenario-Family Accuracy", level=2)
    report.table(
        ["method", "easy", "boundary", "outlier", "transition", "long_history", "irregular_dt", "acceleration"],
        [
            (
                row.method_name,
                "n/a" if row.easy_accuracy is None else f"{row.easy_accuracy:.3f}",
                "n/a" if row.boundary_accuracy is None else f"{row.boundary_accuracy:.3f}",
                "n/a" if row.outlier_accuracy is None else f"{row.outlier_accuracy:.3f}",
                "n/a" if row.transition_accuracy is None else f"{row.transition_accuracy:.3f}",
                "n/a" if row.long_history_accuracy is None else f"{row.long_history_accuracy:.3f}",
                "n/a" if row.irregular_dt_accuracy is None else f"{row.irregular_dt_accuracy:.3f}",
                "n/a" if row.acceleration_accuracy is None else f"{row.acceleration_accuracy:.3f}",
            )
            for row in result.rows
        ],
    )
    report.heading("Capability Flags", level=2)
    report.table(
        ["method", "temporal_history", "model_based", "irregular_dt_native", "outlier_aware", "stronger_sensor_stream"],
        [
            (
                row.method_name,
                f"{row.uses_temporal_history:.1f}",
                f"{row.model_based:.1f}",
                f"{row.irregular_dt_native:.1f}",
                f"{row.outlier_aware:.1f}",
                f"{row.stronger_sensor_stream:.1f}",
            )
            for row in result.rows
        ],
    )
    report.heading("Interpretation", level=2)
    report.bullet_list(
        [
            "Use this report to see where a method is strong, where it is brittle, and which failure mode is due to priors versus model mismatch.",
            "The windowed methods are strong on outlier and long-history comparisons but do not supply model-based irregular-dt prediction.",
            "The Kalman bank is the first explicitly model-based classifier in the ladder and is the right baseline before considering IMM.",
            "`kalman_bank_velocity_aided` should be read as a separate sensor regime, not as a fair same-sensor improvement over the position-only techniques.",
        ]
    )
    return report.text()


def _metric_value(row: TechniqueComparisonRow, field: str) -> float | None:
    return getattr(row, field)


def render_technique_metric_heatmap(result: TechniqueComparisonResult):
    metric_fields = (
        "overall_accuracy",
        "prior_flip_fraction",
        "easy_accuracy",
        "boundary_accuracy",
        "outlier_accuracy",
        "transition_accuracy",
        "long_history_accuracy",
        "irregular_dt_accuracy",
        "acceleration_accuracy",
    )
    matrix = []
    for row in result.rows:
        matrix.append([float("nan") if _metric_value(row, field) is None else float(_metric_value(row, field)) for field in metric_fields])
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    colormap = plt.get_cmap("YlGnBu").copy()
    colormap.set_bad(color="#e5e7eb")
    image = ax.imshow(matrix, aspect="auto", cmap=colormap, vmin=0.0, vmax=1.0)
    ax.set_title("Technique Metric Heatmap", loc="left", fontsize=13, fontweight="bold")
    ax.set_xticks(range(len(metric_fields)))
    ax.set_xticklabels([field.removesuffix("_accuracy") for field in metric_fields], rotation=35, ha="right")
    ax.set_yticks(range(len(result.rows)))
    ax.set_yticklabels([row.method_name for row in result.rows])
    for row_index, row in enumerate(result.rows):
        for col_index, field in enumerate(metric_fields):
            value = _metric_value(row, field)
            ax.text(col_index, row_index, "n/a" if value is None else f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="metric value")
    fig.tight_layout()
    return fig


def render_accuracy_fragility_scatter(result: TechniqueComparisonResult):
    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    colors = {
        "pointwise": "#2563eb",
        "windowed_raw": "#dc2626",
        "windowed_robust": "#16a34a",
        "accumulator": "#d97706",
        "kalman_bank": "#7c3aed",
        "kalman_bank_velocity_aided": "#0f766e",
    }
    for row in result.rows:
        ax.scatter(row.prior_flip_fraction, row.overall_accuracy, s=90, color=colors[row.method_name], label=row.method_name)
        ax.text(row.prior_flip_fraction + 0.01, row.overall_accuracy + 0.005, row.method_name, fontsize=8)
    ax.set_title("Accuracy vs Prior Fragility", loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("fraction flipped by small prior perturbation")
    ax.set_ylabel("overall accuracy")
    ax.set_xlim(-0.01, max(row.prior_flip_fraction for row in result.rows) + 0.10)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def render_technique_capability_heatmap(result: TechniqueComparisonResult):
    capability_fields = ("uses_temporal_history", "model_based", "irregular_dt_native", "outlier_aware", "stronger_sensor_stream")
    matrix = [[float(getattr(row, field)) for field in capability_fields] for row in result.rows]
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    image = ax.imshow(matrix, aspect="auto", cmap="Greens", vmin=0.0, vmax=1.0)
    ax.set_title("Technique Capability Matrix", loc="left", fontsize=13, fontweight="bold")
    ax.set_xticks(range(len(capability_fields)))
    ax.set_xticklabels(["temporal", "model", "irregular_dt", "outlier", "sensor+"], rotation=25, ha="right")
    ax.set_yticks(range(len(result.rows)))
    ax.set_yticklabels([row.method_name for row in result.rows])
    for row_index, row in enumerate(result.rows):
        for col_index, field in enumerate(capability_fields):
            ax.text(col_index, row_index, f"{getattr(row, field):.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="capability score")
    fig.tight_layout()
    return fig


__all__ = [
    "render_accuracy_fragility_scatter",
    "render_technique_capability_heatmap",
    "render_technique_comparison_report",
    "render_technique_metric_heatmap",
]
