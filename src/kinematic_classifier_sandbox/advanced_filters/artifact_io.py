from __future__ import annotations

from pathlib import Path

from kinematic_classifier_sandbox.utils.io import write_csv

from ..runtime_paths import prepare_matplotlib
from .contracts import IMMArtifacts, IMMBenchmarkResult
from .reporting import render_imm_report


def write_imm_artifacts(output_dir: str | Path, *, result: IMMBenchmarkResult | None = None) -> IMMArtifacts:
    if result is None:
        from .runner import run_imm_switching_benchmark

        result = run_imm_switching_benchmark()
    run_dir = Path(output_dir) / "imm_filter_v1"
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    mode_probability_history_path = run_dir / "mode_probability_history.csv"
    mixing_probability_history_path = run_dir / "mixing_probability_history.csv"
    mode_likelihood_history_path = run_dir / "mode_likelihood_history.csv"
    state_estimate_history_path = run_dir / "state_estimate_history.csv"
    posterior_history_path = run_dir / "posterior_history.csv"
    switching_detection_metrics_path = run_dir / "switching_detection_metrics.csv"
    method_comparison_path = run_dir / "advanced_filter_method_comparison.csv"
    decision_matrix_path = run_dir / "advanced_filter_decision_matrix.csv"
    config_path = run_dir / "imm_config.yaml"
    report_path = run_dir / "imm_report.md"
    mode_probability_plot_path = plot_dir / "imm_mode_probability_timeline.png"
    state_plot_path = plot_dir / "imm_state_vs_measurement.png"

    probability_rows: list[dict[str, object]] = []
    likelihood_rows: list[dict[str, object]] = []
    mixing_rows: list[dict[str, object]] = []
    state_rows: list[dict[str, object]] = []
    for run in result.runs:
        for index, step in enumerate(run.steps):
            for mode_id, probability in step.posterior_by_label.items():
                probability_rows.append(
                    {
                        "trajectory_id": run.trajectory_id,
                        "scenario_name": run.scenario_name,
                        "time": step.time,
                        "true_mode": run.true_modes[index],
                        "mode_id": mode_id,
                        "probability": probability,
                        "predicted_mode": step.predicted_label,
                        "confidence": step.confidence,
                    }
                )
            for mode_id, value in step.log_evidence_by_label.items():
                likelihood_rows.append(
                    {
                        "trajectory_id": run.trajectory_id,
                        "scenario_name": run.scenario_name,
                        "time": step.time,
                        "mode_id": mode_id,
                        "log_likelihood": value,
                    }
                )
            mixing_rows.append(
                {
                    "trajectory_id": run.trajectory_id,
                    "scenario_name": run.scenario_name,
                    "time": step.time,
                    "mixing_prob_sum_error_max": step.diagnostics["mixing_prob_sum_error_max"],
                }
            )
            state_rows.append(
                {
                    "trajectory_id": run.trajectory_id,
                    "scenario_name": run.scenario_name,
                    "time": step.time,
                    "measurement": run.measurements[index],
                    "state_position": run.state_means[index][0],
                    "state_velocity": run.state_means[index][1],
                    "state_acceleration": run.state_means[index][2],
                    "true_mode": run.true_modes[index],
                    "predicted_mode": step.predicted_label,
                }
            )
    write_csv(
        mode_probability_history_path,
        probability_rows,
        ["trajectory_id", "scenario_name", "time", "true_mode", "mode_id", "probability", "predicted_mode", "confidence"],
    )
    write_csv(
        posterior_history_path,
        probability_rows,
        ["trajectory_id", "scenario_name", "time", "true_mode", "mode_id", "probability", "predicted_mode", "confidence"],
    )
    write_csv(mode_likelihood_history_path, likelihood_rows, ["trajectory_id", "scenario_name", "time", "mode_id", "log_likelihood"])
    write_csv(mixing_probability_history_path, mixing_rows, ["trajectory_id", "scenario_name", "time", "mixing_prob_sum_error_max"])
    write_csv(
        state_estimate_history_path,
        state_rows,
        ["trajectory_id", "scenario_name", "time", "measurement", "state_position", "state_velocity", "state_acceleration", "true_mode", "predicted_mode"],
    )
    metric_rows = [result.metrics]
    write_csv(switching_detection_metrics_path, metric_rows, list(metric_rows[0]))
    write_csv(method_comparison_path, list(result.method_comparison), list(result.method_comparison[0]))
    decision_rows = [
        {
            "method": "IMM",
            "failure_case": "switching modes",
            "baseline_failed": "yes",
            "method_improved": result.metrics["promotion_decision"] == "promote",
            "cost_acceptable": "yes",
            "decision": result.metrics["promotion_decision"],
        },
        {
            "method": "PF",
            "failure_case": "nonlinear drag / outlier noise",
            "baseline_failed": "see advanced_filter_comparison_v1",
            "method_improved": "see advanced_filter_comparison_v1",
            "cost_acceptable": "see advanced_filter_comparison_v1",
            "decision": "see advanced_filter_comparison_v1",
        },
        {
            "method": "RBPF",
            "failure_case": "latent maneuver onset",
            "baseline_failed": "see advanced_filter_comparison_v1",
            "method_improved": "see advanced_filter_comparison_v1",
            "cost_acceptable": "see advanced_filter_comparison_v1",
            "decision": "see advanced_filter_comparison_v1",
        },
    ]
    write_csv(
        decision_matrix_path,
        decision_rows,
        ["method", "failure_case", "baseline_failed", "method_improved", "cost_acceptable", "decision"],
    )
    config_path.write_text(render_imm_config_text(), encoding="utf-8")
    report_path.write_text(render_imm_report(result), encoding="utf-8")
    _render_imm_plots(result, mode_probability_plot_path, state_plot_path)
    return IMMArtifacts(
        run_dir,
        config_path,
        report_path,
        mode_probability_history_path,
        mixing_probability_history_path,
        mode_likelihood_history_path,
        state_estimate_history_path,
        posterior_history_path,
        switching_detection_metrics_path,
        method_comparison_path,
        decision_matrix_path,
        plot_dir,
        mode_probability_plot_path,
        state_plot_path,
    )


def render_imm_config_text() -> str:
    return """experiment:
  id: imm_switching_v1
  seed: 17
  output_dir: artifacts/imm_filter_v1
dataset:
  witness_families:
    - stationary_then_moving
    - constant_velocity_then_braking
    - constant_velocity_then_maneuver
    - acceleration_then_coast
  replicas: 12
modes:
  ids:
    - stationary
    - constant_velocity
    - constant_acceleration
    - braking
    - maneuver
evaluation:
  detection_threshold: 0.6
  confidence_thresholds: [0.5, 0.7, 0.9]
"""


def _render_imm_plots(result: IMMBenchmarkResult, mode_probability_path: Path, state_path: Path) -> None:
    plt = prepare_matplotlib()
    run = result.runs[0]
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    labels = list(run.steps[0].posterior_by_label)
    for label in labels:
        ax.plot(run.times, [step.posterior_by_label[label] for step in run.steps], label=label)
    ax.set_title("IMM mode probabilities")
    ax.set_xlabel("time")
    ax.set_ylabel("posterior")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(mode_probability_path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.plot(run.times, run.measurements, marker="o", label="measurement")
    ax.plot(run.times, [state[0] for state in run.state_means], marker="x", label="IMM position mean")
    ax.set_title("IMM state estimate")
    ax.set_xlabel("time")
    ax.set_ylabel("position")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(state_path)
    plt.close(fig)
