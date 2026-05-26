from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from kinematic_classifier_sandbox.utils.io import write_csv

from .contracts import (
    CrossMethodPriorComparisonArtifacts,
    CrossMethodPriorComparisonResult,
    PriorSensitivityArtifacts,
    PriorSensitivityResult,
)
from .reporting import (
    render_cross_method_prior_comparison_png_bytes,
    render_cross_method_prior_comparison_report,
    render_prior_sensitivity_decision_png_bytes,
    render_prior_sensitivity_decomposition_png_bytes,
    render_prior_sensitivity_flip_png_bytes,
    render_prior_sensitivity_fragility_png_bytes,
    render_prior_sensitivity_heatmap_png_bytes,
    render_prior_sensitivity_pairwise_flip_png_bytes,
    render_prior_sensitivity_posterior_png_bytes,
    render_prior_sensitivity_report,
)
from .runner import (
    analyze_cross_method_prior_comparison,
    analyze_prior_sensitivity,
)


def write_cross_method_prior_comparison_artifacts(
    output_dir: str | Path,
    *,
    seed: int = 7,
    result: CrossMethodPriorComparisonResult | None = None,
) -> CrossMethodPriorComparisonArtifacts:
    analysis = result or analyze_cross_method_prior_comparison(seed=seed)
    output_root = Path(output_dir)
    run_dir = output_root / "prior_sensitivity_cross_method_v1"
    run_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_dir / "cross_method_prior_comparison_report.md"
    comparison_csv_path = run_dir / "cross_method_prior_comparison.csv"
    status_csv_path = run_dir / "cross_method_prior_comparison_status.csv"
    plot_png_path = run_dir / "cross_method_prior_fragility_heatmap.png"

    report_path.write_text(render_cross_method_prior_comparison_report(analysis), encoding="utf-8")
    write_csv(
        comparison_csv_path,
        [
            {
                "method_name": row["method_name"],
                **{scenario_name: row[scenario_name] for scenario_name in analysis.scenario_names},
                "fraction_flipped_by_small_prior_perturbation": row["fraction_flipped_by_small_prior_perturbation"],
            }
            for row in analysis.rows
        ],
        ["method_name", *analysis.scenario_names, "fraction_flipped_by_small_prior_perturbation"],
    )
    write_csv(
        status_csv_path,
        [{"method_name": row["method_name"], **{scenario_name: row[f"{scenario_name}_status"] for scenario_name in analysis.scenario_names}} for row in analysis.rows],
        ["method_name", *analysis.scenario_names],
    )
    plot_png_path.write_bytes(render_cross_method_prior_comparison_png_bytes(analysis))
    return CrossMethodPriorComparisonArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        comparison_csv_path=comparison_csv_path,
        status_csv_path=status_csv_path,
        plot_png_path=plot_png_path,
    )


def write_prior_sensitivity_artifacts(
    output_dir: str | Path,
    *,
    seed: int = 7,
    forgetting_factor: float = 1.0,
    confidence_threshold: float = 0.75,
    trajectories_per_class: int = 3,
    result: PriorSensitivityResult | None = None,
) -> PriorSensitivityArtifacts:
    analysis = result or analyze_prior_sensitivity(
        seed=seed,
        forgetting_factor=forgetting_factor,
        confidence_threshold=confidence_threshold,
        trajectories_per_class=trajectories_per_class,
    )
    output_root = Path(output_dir)
    run_dir_name = "prior_sensitivity_v1" if analysis.method_name == "accumulator" else f"prior_sensitivity_{analysis.method_name}_v1"
    run_dir = output_root / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_dir / "prior_sensitivity_report.md"
    sweep_path = run_dir / "prior_sensitivity.csv"
    flip_thresholds_path = run_dir / "prior_flip_thresholds.csv"
    metrics_path = run_dir / "prior_dominance_metrics.json"
    config_path = run_dir / "prior_sensitivity_config.yaml"
    plot_posterior_png_path = run_dir / "posterior_vs_prior.png"
    plot_flip_png_path = run_dir / "decision_flip_thresholds.png"
    plot_heatmap_png_path = run_dir / "prior_dominance_heatmap.png"
    plot_decision_png_path = run_dir / "prior_decision_map.png"
    plot_decomposition_png_path = run_dir / "log_odds_decomposition.png"
    plot_pairwise_flip_png_path = run_dir / "pairwise_flip_threshold_heatmap.png"
    plot_fragility_png_path = run_dir / "trajectory_prior_fragility_overview.png"

    report_path.write_text(render_prior_sensitivity_report(analysis), encoding="utf-8")
    sweep_path.write_text("", encoding="utf-8")
    flip_thresholds_path.write_text("", encoding="utf-8")
    metrics_path.write_text(json.dumps(analysis.prior_dominance_metrics, indent=2, sort_keys=True), encoding="utf-8")
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  name: prior_sensitivity_v1",
                f"  seed: {seed}",
                "classifier:",
                "  type: sequential_bayes_accumulator",
                f"  forgetting_factor: {forgetting_factor}",
                f"  confidence_threshold: {confidence_threshold}",
                "evaluation:",
                "  prior_grid: [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    plot_posterior_png_path.write_bytes(render_prior_sensitivity_posterior_png_bytes(analysis))
    plot_flip_png_path.write_bytes(render_prior_sensitivity_flip_png_bytes(analysis))
    plot_heatmap_png_path.write_bytes(render_prior_sensitivity_heatmap_png_bytes(analysis))
    plot_decision_png_path.write_bytes(render_prior_sensitivity_decision_png_bytes(analysis))
    plot_decomposition_png_path.write_bytes(render_prior_sensitivity_decomposition_png_bytes(analysis))
    plot_pairwise_flip_png_path.write_bytes(render_prior_sensitivity_pairwise_flip_png_bytes(analysis))
    plot_fragility_png_path.write_bytes(render_prior_sensitivity_fragility_png_bytes(analysis))

    sweep_rows = [asdict(row) for row in analysis.sweep_rows]
    flip_rows = [asdict(row) for row in analysis.flip_thresholds]
    write_csv(
        sweep_path,
        sweep_rows,
        [
            "trajectory_id",
            "scenario_name",
            "true_class",
            "prior_a",
            "prior_b",
            "log_prior_odds",
            "final_class",
            "final_confidence",
            "abstained",
            "posterior_a",
            "posterior_b",
            "final_log_posterior_odds",
            "cumulative_log_likelihood_ratio",
        ],
    )
    write_csv(
        flip_thresholds_path,
        flip_rows,
        [
            "trajectory_id",
            "scenario_name",
            "true_class",
            "uniform_prior_class",
            "uniform_prior_confidence",
            "min_prior_a_for_a",
            "max_prior_a_for_b",
            "smallest_prior_shift_to_flip",
            "smallest_log_prior_shift_to_flip",
        ],
    )
    return PriorSensitivityArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        sweep_path=sweep_path,
        flip_thresholds_path=flip_thresholds_path,
        metrics_path=metrics_path,
        config_path=config_path,
        plot_posterior_png_path=plot_posterior_png_path,
        plot_flip_png_path=plot_flip_png_path,
        plot_heatmap_png_path=plot_heatmap_png_path,
        plot_decision_png_path=plot_decision_png_path,
        plot_decomposition_png_path=plot_decomposition_png_path,
        plot_pairwise_flip_png_path=plot_pairwise_flip_png_path,
        plot_fragility_png_path=plot_fragility_png_path,
    )
