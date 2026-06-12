from __future__ import annotations

from kinematic_classifier_sandbox.reports.markdown import MarkdownDocument

from ..utils.plotting import plt
from .technique_comparison_contracts import TechniqueComparisonResult, TechniqueComparisonRow


def _fmt(value: float | None, *, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def render_technique_comparison_report(result: TechniqueComparisonResult) -> str:
    report = MarkdownDocument("Technique Comparison Study")
    report.paragraph(
        "This package compares the current baseline techniques using benchmark suites, prior-sensitivity diagnostics, and capability-aware scenario families. "
        "Advanced methods remain members of the shared classifier registry, but broad reports now mark whether a scenario is supported, not applicable, or witness-only."
    )
    report.heading("Summary Metrics", level=2)
    report.table(
        ["method", "sensor_regime", "status", "primary_family", "overall_accuracy", "prior_flip_fraction", "witness_artifact"],
        [
            (
                row.method_name,
                row.sensor_regime_id,
                row.applicability_status,
                row.primary_evaluation_family,
                _fmt(row.overall_accuracy),
                _fmt(row.prior_flip_fraction),
                row.witness_artifact or "n/a",
            )
            for row in result.rows
        ],
    )
    report.heading("Scenario Applicability", level=2)
    report.table(
        ["method", "scenario_family", "status", "metric_name", "metric_value", "note"],
        [
            (
                row.method_name,
                row.scenario_family,
                row.applicability_status,
                row.metric_name or "n/a",
                _fmt(row.metric_value),
                row.note,
            )
            for row in result.scenario_support_rows
        ],
    )
    report.heading("Capability Flags", level=2)
    report.table(
        ["method", "local_feature", "recursive", "model_based", "switching_aware", "nonlinear_nongaussian", "sampled_latent", "stochastic_mean_reversion"],
        [
            (
                spec.method_name,
                "1.0" if spec.capabilities.local_feature else "0.0",
                "1.0" if spec.capabilities.recursive else "0.0",
                "1.0" if spec.capabilities.model_based else "0.0",
                "1.0" if spec.capabilities.switching_aware else "0.0",
                "1.0" if spec.capabilities.nonlinear_nongaussian else "0.0",
                "1.0" if spec.capabilities.sampled_latent else "0.0",
                "1.0" if spec.capabilities.stochastic_mean_reversion else "0.0",
            )
            for spec in result.method_specs
        ],
    )
    report.heading("Interpretation", level=2)
    report.bullet_list(
        [
            "Universal shared-corpus scores remain useful for the local-feature through Kalman-family methods.",
            "PF and RBPF stay on the generic classifier ladder, but their primary evidence comes from advanced witnesses rather than the binary shared corpus.",
            "The OU study is carried in the same vocabulary as PF and RBPF and documents stochastic mean-reversion support without claiming a separate canonical rung.",
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
    fig, ax = plt.subplots(figsize=(12.2, 5.6))
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
    colors = ("#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0f766e", "#db2777", "#0891b2", "#65a30d")
    plotted = [row for row in result.rows if row.overall_accuracy is not None and row.prior_flip_fraction is not None]
    for index, row in enumerate(plotted):
        ax.scatter(row.prior_flip_fraction, row.overall_accuracy, s=90, color=colors[index % len(colors)], label=row.method_name)
        ax.text(row.prior_flip_fraction + 0.01, row.overall_accuracy + 0.005, row.method_name, fontsize=8)
    ax.set_title("Accuracy vs Prior Fragility", loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("fraction flipped by small prior perturbation")
    ax.set_ylabel("overall accuracy")
    ax.set_xlim(-0.01, max((row.prior_flip_fraction for row in plotted), default=0.0) + 0.10)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def render_technique_capability_heatmap(result: TechniqueComparisonResult):
    capability_fields = (
        "local_feature",
        "recursive",
        "model_based",
        "switching_aware",
        "nonlinear_nongaussian",
        "sampled_latent",
        "stochastic_mean_reversion",
    )
    matrix = [[1.0 if getattr(spec.capabilities, field) else 0.0 for field in capability_fields] for spec in result.method_specs]
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    image = ax.imshow(matrix, aspect="auto", cmap="Greens", vmin=0.0, vmax=1.0)
    ax.set_title("Technique Capability Matrix", loc="left", fontsize=13, fontweight="bold")
    ax.set_xticks(range(len(capability_fields)))
    ax.set_xticklabels(["local", "recursive", "model", "switching", "nonlinear", "sampled", "mean_revert"], rotation=25, ha="right")
    ax.set_yticks(range(len(result.method_specs)))
    ax.set_yticklabels([spec.method_name for spec in result.method_specs])
    for row_index, spec in enumerate(result.method_specs):
        for col_index, field in enumerate(capability_fields):
            ax.text(col_index, row_index, "1.0" if getattr(spec.capabilities, field) else "0.0", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="capability score")
    fig.tight_layout()
    return fig


__all__ = [
    "render_accuracy_fragility_scatter",
    "render_technique_capability_heatmap",
    "render_technique_comparison_report",
    "render_technique_metric_heatmap",
]
