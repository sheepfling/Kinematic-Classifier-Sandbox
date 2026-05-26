from __future__ import annotations

import io
import json
from pathlib import Path

from kinematic_classifier_sandbox.utils.io import write_csv

from ...runtime_paths import prepare_matplotlib
from .contracts import TransitionBenchmarkArtifacts, TransitionBenchmarkResult
from .reporting import render_transition_benchmark_report, render_transition_numeric_walkthrough_markdown
from .runner import run_transition_benchmark


def _build_figure(result: TransitionBenchmarkResult):
    plt = prepare_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.0))
    selected_names = ("stationary_then_moving", "constant_velocity_then_braking", "constant_velocity_then_maneuver")
    colors = {"stationary": "#6b7280", "constant_velocity": "#2563eb", "braking": "#dc2626", "maneuver": "#7c3aed"}
    for axis, scenario_name in zip(axes.flat[:3], selected_names):
        transition_run = next(run for run in result.transition_runs if run.scenario_name == scenario_name)
        kalman_run = next(run for run in result.kalman_runs if run.scenario_name == scenario_name)
        for mode in colors:
            axis.plot([step.time for step in transition_run.steps], [step.posterior_weights[mode] for step in transition_run.steps], color=colors[mode], linewidth=2.2, label=f"{mode} transition")
            axis.plot([step.time for step in kalman_run.steps], [step.posterior_weights[mode] for step in kalman_run.steps], color=colors[mode], linewidth=1.2, alpha=0.35, linestyle="--")
        axis.set_title(scenario_name, loc="left", fontsize=12, fontweight="bold")
        axis.set_ylim(0.0, 1.0)
        axis.grid(True, alpha=0.25)
        axis.set_xlabel("time")
        axis.set_ylabel("posterior")
    summary_ax = axes.flat[3]
    labels = ["static", "transition", "kalman"]
    values = [result.summary.static_post_switch_accuracy, result.summary.transition_post_switch_accuracy, result.summary.kalman_post_switch_accuracy]
    summary_ax.bar(labels, values, color=["#9ca3af", "#059669", "#2563eb"], width=0.55)
    summary_ax.set_ylim(0.0, 1.0)
    summary_ax.set_title("Post-switch accuracy", loc="left", fontsize=12, fontweight="bold")
    summary_ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def write_transition_benchmark_artifacts(
    output_dir: str | Path,
    *,
    result: TransitionBenchmarkResult | None = None,
) -> TransitionBenchmarkArtifacts:
    analysis = result or run_transition_benchmark()
    run_dir = Path(output_dir) / "transition_matrix_accumulator_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "transition_matrix_accumulator_report.md"
    numeric_walkthrough_path = run_dir / "transition_matrix_numeric_walkthrough.md"
    posterior_history_path = run_dir / "transition_matrix_posterior_history.csv"
    scenario_summary_path = run_dir / "transition_matrix_scenario_summary.csv"
    config_path = run_dir / "transition_matrix_config.yaml"
    dataset_manifest_path = run_dir / "transition_matrix_dataset_manifest.json"
    plot_png_path = run_dir / "transition_matrix_diagnostics.png"

    report_path.write_text(render_transition_benchmark_report(analysis), encoding="utf-8")
    numeric_walkthrough_path.write_text(render_transition_numeric_walkthrough_markdown(analysis), encoding="utf-8")
    config_path.write_text("\n".join(["experiment:", "  name: transition_matrix_accumulator", "dataset:", "  source: switching_scenarios_v1", "classifier:", "  baseline: static_mode_accumulator", "  candidate: transition_matrix_accumulator", ""]), encoding="utf-8")
    dataset_manifest_path.write_text(json.dumps({"scenario_count": analysis.summary.num_scenarios, "scenario_names": sorted({scenario.scenario_name for scenario in analysis.scenarios}), "replicas": len(analysis.scenarios) // 3 if analysis.scenarios else 0}, indent=2), encoding="utf-8")

    posterior_rows: list[dict[str, object]] = []
    scenario_rows: list[dict[str, object]] = []
    for static_run, transition_run, kalman_run in zip(analysis.static_runs, analysis.transition_runs, analysis.kalman_runs):
        for run in (static_run, transition_run, kalman_run):
            for step in run.steps:
                posterior_rows.append(
                    {
                        "trajectory_id": run.trajectory_id,
                        "scenario_name": run.scenario_name,
                        "mode": run.mode,
                        "step": step.step,
                        "time": step.time,
                        "measurement": step.measurement,
                        "estimated_speed": step.estimated_speed,
                        "estimated_accel": step.estimated_accel,
                        "true_mode": step.true_mode,
                        "predicted_mode": step.predicted_mode,
                        "confidence": step.confidence,
                        **{f"posterior_{name}": value for name, value in step.posterior_weights.items()},
                    }
                )
        scenario_rows.append(
            {
                "scenario_name": static_run.scenario_name,
                "trajectory_id": static_run.trajectory_id,
                "static_accuracy": static_run.accuracy,
                "transition_accuracy": transition_run.accuracy,
                "kalman_accuracy": kalman_run.accuracy,
                "static_post_switch_accuracy": static_run.post_switch_accuracy,
                "transition_post_switch_accuracy": transition_run.post_switch_accuracy,
                "kalman_post_switch_accuracy": kalman_run.post_switch_accuracy,
            }
        )

    mode_names = list(analysis.static_runs[0].steps[0].posterior_weights) if analysis.static_runs else []
    write_csv(posterior_history_path, posterior_rows, ["trajectory_id", "scenario_name", "mode", "step", "time", "measurement", "estimated_speed", "estimated_accel", "true_mode", "predicted_mode", "confidence", *[f"posterior_{name}" for name in mode_names]])
    write_csv(scenario_summary_path, scenario_rows, ["scenario_name", "trajectory_id", "static_accuracy", "transition_accuracy", "kalman_accuracy", "static_post_switch_accuracy", "transition_post_switch_accuracy", "kalman_post_switch_accuracy"])
    plt = prepare_matplotlib()
    fig = _build_figure(analysis)
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        plot_png_path.write_bytes(buffer.getvalue())
    finally:
        plt.close(fig)
    return TransitionBenchmarkArtifacts(run_dir=run_dir, report_path=report_path, numeric_walkthrough_path=numeric_walkthrough_path, posterior_history_path=posterior_history_path, scenario_summary_path=scenario_summary_path, config_path=config_path, dataset_manifest_path=dataset_manifest_path, plot_png_path=plot_png_path)
