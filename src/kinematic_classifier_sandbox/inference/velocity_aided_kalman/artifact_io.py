from __future__ import annotations

from pathlib import Path

from kinematic_classifier_sandbox.utils.io import write_csv

from ...runtime_paths import prepare_matplotlib
from .contracts import VelocityAidedComparisonArtifacts, VelocityAidedComparisonResult
from .reporting import render_velocity_aided_kalman_comparison_report
from .runner import analyze_velocity_aided_kalman_comparison


def _render_heatmap(result: VelocityAidedComparisonResult):
    plt = prepare_matplotlib()
    fields = ("overall_accuracy", "endpoint_match_accuracy", "short_accuracy", "short_noisy_accuracy", "outlier_accuracy")
    matrix = [[getattr(row, field) for field in fields] for row in result.rows]
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    image = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_title("Velocity-Aided Kalman Comparison", loc="left", fontsize=13, fontweight="bold")
    ax.set_xticks(range(len(fields)))
    ax.set_xticklabels(["overall", "endpoint", "short", "short_noisy", "outlier"], rotation=20, ha="right")
    ax.set_yticks(range(len(result.rows)))
    ax.set_yticklabels([row.measurement_mode for row in result.rows])
    for row_index, row in enumerate(result.rows):
        for col_index, field in enumerate(fields):
            ax.text(col_index, row_index, f"{getattr(row, field):.2f}", ha="center", va="center", fontsize=9, color="#0f172a")
    fig.colorbar(image, ax=ax, fraction=0.05, pad=0.04)
    fig.tight_layout()
    return fig


def _render_diagnostics(result: VelocityAidedComparisonResult):
    plt = prepare_matplotlib()
    scenario_names = ("short_noisy", "endpoint_match", "outlier")
    fig, axes = plt.subplots(len(scenario_names), 1, figsize=(9.6, 8.8), sharex=False)
    color_map = {"position_only": "#2563eb", "position_plus_direct_velocity": "#dc2626"}
    for axis, scenario_name in zip(axes, scenario_names):
        scenario_traces = [trace for trace in result.traces if trace.scenario_name == scenario_name]
        for trace in scenario_traces:
            axis.plot(trace.times, trace.true_class_posterior, label=trace.measurement_mode, linewidth=2.0, color=color_map[trace.measurement_mode])
        example = scenario_traces[0]
        axis.scatter(example.times, example.measurements, color="#111827", s=14, alpha=0.55, label="position z")
        if any(any(abs(value) > 1e-9 for value in trace.velocity_measurements) for trace in scenario_traces):
            axis2 = axis.twinx()
            velocity_trace = next(trace for trace in scenario_traces if trace.measurement_mode == "position_plus_direct_velocity")
            axis2.plot(velocity_trace.times, velocity_trace.velocity_measurements, color="#16a34a", linestyle="--", linewidth=1.6, alpha=0.8, label="velocity z")
            axis2.set_ylabel("velocity")
        axis.set_ylim(-0.05, 1.05)
        axis.set_ylabel("true-class posterior")
        axis.set_title(f"{scenario_name} ({example.true_class})", loc="left", fontsize=12, fontweight="bold")
        axis.grid(alpha=0.25, linewidth=0.6)
    axes[-1].set_xlabel("time")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    return fig


def write_velocity_aided_kalman_comparison_artifacts(
    output_root: str | Path,
    *,
    result: VelocityAidedComparisonResult | None = None,
) -> VelocityAidedComparisonArtifacts:
    base_path = Path(output_root)
    run_dir = base_path / "velocity_aided_kalman_comparison_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    comparison = result or analyze_velocity_aided_kalman_comparison()

    report_path = run_dir / "velocity_aided_kalman_comparison_report.md"
    summary_csv_path = run_dir / "velocity_aided_kalman_summary.csv"
    trace_csv_path = run_dir / "velocity_aided_kalman_trace_summary.csv"
    heatmap_png_path = run_dir / "velocity_aided_kalman_heatmap.png"
    diagnostics_png_path = run_dir / "velocity_aided_kalman_diagnostics.png"

    report_path.write_text(render_velocity_aided_kalman_comparison_report(comparison), encoding="utf-8")
    write_csv(
        summary_csv_path,
        [
            {
                "measurement_mode": row.measurement_mode,
                "overall_accuracy": row.overall_accuracy,
                "endpoint_match_accuracy": row.endpoint_match_accuracy,
                "short_accuracy": row.short_accuracy,
                "short_noisy_accuracy": row.short_noisy_accuracy,
                "outlier_accuracy": row.outlier_accuracy,
            }
            for row in comparison.rows
        ],
        ["measurement_mode", "overall_accuracy", "endpoint_match_accuracy", "short_accuracy", "short_noisy_accuracy", "outlier_accuracy"],
    )
    write_csv(
        trace_csv_path,
        [
            {
                "measurement_mode": trace.measurement_mode,
                "trajectory_id": trace.trajectory_id,
                "scenario_name": trace.scenario_name,
                "true_class": trace.true_class,
                "final_predicted_class": trace.final_predicted_class,
                "final_confidence": trace.final_confidence,
                "times": " ".join(f"{value:.3f}" for value in trace.times),
                "measurements": " ".join(f"{value:.3f}" for value in trace.measurements),
                "velocity_measurements": " ".join(f"{value:.3f}" for value in trace.velocity_measurements),
                "true_class_posterior": " ".join(f"{value:.3f}" for value in trace.true_class_posterior),
            }
            for trace in comparison.traces
        ],
        ["measurement_mode", "trajectory_id", "scenario_name", "true_class", "final_predicted_class", "final_confidence", "times", "measurements", "velocity_measurements", "true_class_posterior"],
    )

    heatmap_figure = _render_heatmap(comparison)
    heatmap_figure.savefig(heatmap_png_path, format="png", dpi=160, bbox_inches="tight")
    heatmap_figure.clf()

    diagnostics_figure = _render_diagnostics(comparison)
    diagnostics_figure.savefig(diagnostics_png_path, format="png", dpi=160, bbox_inches="tight")
    diagnostics_figure.clf()

    plt = prepare_matplotlib()
    plt.close(heatmap_figure)
    plt.close(diagnostics_figure)

    return VelocityAidedComparisonArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        summary_csv_path=summary_csv_path,
        trace_csv_path=trace_csv_path,
        heatmap_png_path=heatmap_png_path,
        diagnostics_png_path=diagnostics_png_path,
    )
