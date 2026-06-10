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
                "applicability_status": row.applicability_status,
                "primary_evaluation_family": row.primary_evaluation_family,
                "overall_accuracy": row.overall_accuracy,
                "prior_flip_fraction": row.prior_flip_fraction,
                "median_flip_threshold": row.median_flip_threshold,
                "witness_artifact": row.witness_artifact,
            }
            for row in comparison.rows
        ],
        ["method_name", "sensor_regime_id", "applicability_status", "primary_evaluation_family", "overall_accuracy", "prior_flip_fraction", "median_flip_threshold", "witness_artifact"],
    )
    write_csv(
        scenario_csv_path,
        [
            {
                "method_name": row.method_name,
                "scenario_family": row.scenario_family,
                "applicability_status": row.applicability_status,
                "metric_name": row.metric_name,
                "metric_value": row.metric_value,
                "note": row.note,
            }
            for row in comparison.scenario_support_rows
        ],
        ["method_name", "scenario_family", "applicability_status", "metric_name", "metric_value", "note"],
    )
    write_csv(
        capability_csv_path,
        [
            {
                "method_name": spec.method_name,
                "sensor_regime_id": spec.sensor_regime_id,
                "primary_evaluation_family": spec.primary_evaluation_family,
                "supported_scenario_families": " ".join(spec.supported_scenario_families),
                "local_feature": spec.capabilities.local_feature,
                "recursive": spec.capabilities.recursive,
                "model_based": spec.capabilities.model_based,
                "switching_aware": spec.capabilities.switching_aware,
                "nonlinear_nongaussian": spec.capabilities.nonlinear_nongaussian,
                "sampled_latent": spec.capabilities.sampled_latent,
                "stochastic_mean_reversion": spec.capabilities.stochastic_mean_reversion,
                "witness_artifact": spec.witness_artifact,
            }
            for spec in comparison.method_specs
        ],
        ["method_name", "sensor_regime_id", "primary_evaluation_family", "supported_scenario_families", "local_feature", "recursive", "model_based", "switching_aware", "nonlinear_nongaussian", "sampled_latent", "stochastic_mean_reversion", "witness_artifact"],
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
