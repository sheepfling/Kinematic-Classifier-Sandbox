from __future__ import annotations

from pathlib import Path

from ..utils.io import write_csv
from ..utils.plotting import figure_to_png_bytes
from .technique_comparison_contracts import TechniqueComparisonArtifacts, TechniqueComparisonResult
from .technique_comparison_reporting import (
    render_accuracy_fragility_scatter,
    render_technique_capability_heatmap,
    render_technique_comparison_report,
    render_technique_metric_heatmap,
)
from .technique_comparison_runner import analyze_technique_comparison


def write_technique_comparison_artifacts(
    output_dir: str | Path,
    *,
    seed: int = 7,
    result: TechniqueComparisonResult | None = None,
) -> TechniqueComparisonArtifacts:
    comparison = result or analyze_technique_comparison(seed=seed)
    output_root = Path(output_dir)
    run_dir = output_root / "technique_comparison_v1"
    run_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_dir / "technique_comparison_report.md"
    summary_csv_path = run_dir / "technique_summary.csv"
    scenario_csv_path = run_dir / "technique_scenario_metrics.csv"
    capability_csv_path = run_dir / "technique_capabilities.csv"
    metric_heatmap_png_path = run_dir / "technique_metric_heatmap.png"
    scatter_png_path = run_dir / "accuracy_vs_prior_fragility.png"
    capability_png_path = run_dir / "technique_capability_heatmap.png"

    report_path.write_text(render_technique_comparison_report(comparison), encoding="utf-8")
    write_csv(
        summary_csv_path,
        [
            {
                "method_name": row.method_name,
                "sensor_regime_id": row.sensor_regime_id,
                "overall_accuracy": row.overall_accuracy,
                "prior_flip_fraction": row.prior_flip_fraction,
                "median_flip_threshold": row.median_flip_threshold,
            }
            for row in comparison.rows
        ],
        ["method_name", "sensor_regime_id", "overall_accuracy", "prior_flip_fraction", "median_flip_threshold"],
    )
    write_csv(
        scenario_csv_path,
        [
            {
                "method_name": row.method_name,
                "easy_accuracy": row.easy_accuracy,
                "boundary_accuracy": row.boundary_accuracy,
                "outlier_accuracy": row.outlier_accuracy,
                "transition_accuracy": row.transition_accuracy,
                "long_history_accuracy": row.long_history_accuracy,
                "irregular_dt_accuracy": row.irregular_dt_accuracy,
                "acceleration_accuracy": row.acceleration_accuracy,
            }
            for row in comparison.rows
        ],
        [
            "method_name",
            "easy_accuracy",
            "boundary_accuracy",
            "outlier_accuracy",
            "transition_accuracy",
            "long_history_accuracy",
            "irregular_dt_accuracy",
            "acceleration_accuracy",
        ],
    )
    write_csv(
        capability_csv_path,
        [
            {
                "method_name": row.method_name,
                "uses_temporal_history": row.uses_temporal_history,
                "model_based": row.model_based,
                "irregular_dt_native": row.irregular_dt_native,
                "outlier_aware": row.outlier_aware,
            }
            for row in comparison.rows
        ],
        ["method_name", "uses_temporal_history", "model_based", "irregular_dt_native", "outlier_aware"],
    )
    metric_heatmap_png_path.write_bytes(figure_to_png_bytes(render_technique_metric_heatmap(comparison)))
    scatter_png_path.write_bytes(figure_to_png_bytes(render_accuracy_fragility_scatter(comparison)))
    capability_png_path.write_bytes(figure_to_png_bytes(render_technique_capability_heatmap(comparison)))

    return TechniqueComparisonArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        summary_csv_path=summary_csv_path,
        scenario_csv_path=scenario_csv_path,
        capability_csv_path=capability_csv_path,
        metric_heatmap_png_path=metric_heatmap_png_path,
        scatter_png_path=scatter_png_path,
        capability_png_path=capability_png_path,
    )


__all__ = ["write_technique_comparison_artifacts"]
