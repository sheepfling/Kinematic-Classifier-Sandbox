from __future__ import annotations

from math import exp, log
from pathlib import Path

from kinematic_classifier_sandbox.corpus.trajectory_exploration.comparison_surface import write_comparison_summary_csv
from kinematic_classifier_sandbox.utils.io import write_csv
from kinematic_classifier_sandbox.utils.plotting import plt

from ..inference.transition_matrix_accumulator import (
    _run_mode_accumulator,
    default_switching_mode_specs,
    default_transition_matrix,
)
from ..render.intermediate_plots import (
    render_likelihood_strip,
    render_posterior_timeline,
    render_prior_likelihood_posterior_waterfall,
)
from ..render.step_cards import write_step_card
from ..tracing.filter_trace import FilterStepTrace, write_filter_step_trace_csv
from ..tracing.trace_validation import validate_filter_step_trace_set
from ..utils.method_evaluation_summary import (
    METHOD_EVALUATION_SUMMARY_FIELDS,
    PosteriorMetricSample,
    build_method_evaluation_summary_row,
    compute_multiclass_posterior_metrics,
)
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
    summary_path = run_dir / "summary.csv"
    method_evaluation_summary_path = run_dir / "method_evaluation_summary.csv"
    decision_matrix_path = run_dir / "advanced_filter_decision_matrix.csv"
    config_path = run_dir / "imm_config.yaml"
    report_path = run_dir / "imm_report.md"
    mode_probability_plot_path = plot_dir / "imm_mode_probability_timeline.png"
    state_plot_path = plot_dir / "imm_state_vs_measurement.png"
    trace_dir = run_dir / "traces"
    intermediate_plot_dir = run_dir / "plots" / "intermediate"
    step_card_dir = run_dir / "step_cards"
    filter_step_trace_path = trace_dir / "filter_step_trace.csv"
    per_method_diagnostics_path = trace_dir / "per_method_diagnostics.csv"
    posterior_timeline_plot_path = intermediate_plot_dir / "posterior_timeline_with_regimes.png"
    likelihood_strip_plot_path = intermediate_plot_dir / "innovation_likelihood_strip.png"
    waterfall_plot_path = intermediate_plot_dir / "prior_likelihood_posterior_waterfall.png"
    mixing_heatmap_plot_path = intermediate_plot_dir / "mixing_probability_heatmap.png"
    mode_conditioned_state_plot_path = intermediate_plot_dir / "mode_conditioned_state_traces.png"
    switch_recovery_plot_path = intermediate_plot_dir / "switch_recovery_panel.png"

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
            for dest_mode in run.mode_ids:
                source_weights = run.mixing_probabilities[index].get(dest_mode, ())
                for source_index, source_mode in enumerate(run.mode_ids):
                    mixing_rows.append(
                        {
                            "trajectory_id": run.trajectory_id,
                            "scenario_name": run.scenario_name,
                            "time": step.time,
                            "dest_mode": dest_mode,
                            "source_mode": source_mode,
                            "mixing_probability": (
                                source_weights[source_index] if source_index < len(source_weights) else ""
                            ),
                            "mixing_prob_sum_error_max": step.diagnostics["mixing_prob_sum_error_max"],
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
    write_csv(
        mixing_probability_history_path,
        mixing_rows,
        [
            "trajectory_id",
            "scenario_name",
            "time",
            "dest_mode",
            "source_mode",
            "mixing_probability",
            "mixing_prob_sum_error_max",
        ],
    )
    write_csv(
        state_estimate_history_path,
        state_rows,
        ["trajectory_id", "scenario_name", "time", "measurement", "state_position", "state_velocity", "state_acceleration", "true_mode", "predicted_mode"],
    )
    traces = _build_imm_filter_step_traces(result)
    validate_filter_step_trace_set(traces)
    write_filter_step_trace_csv(filter_step_trace_path, traces)
    diagnostic_rows = [
        {
            "run_id": trace.run_id,
            "trajectory_id": trace.trajectory_id,
            "method_id": trace.method_id,
            "time_index": trace.time_index,
            "time": trace.time,
            "class_or_model": trace.class_or_model,
            "posterior_entropy": trace.posterior_entropy,
            "normalized_innovation_squared": trace.normalized_innovation_squared,
            "log_likelihood": trace.log_likelihood,
        }
        for trace in traces
    ]
    write_csv(
        per_method_diagnostics_path,
        diagnostic_rows,
        ["run_id", "trajectory_id", "method_id", "time_index", "time", "class_or_model", "posterior_entropy", "normalized_innovation_squared", "log_likelihood"],
    )
    metric_rows = [result.metrics]
    write_csv(switching_detection_metrics_path, metric_rows, list(metric_rows[0]))
    write_csv(method_comparison_path, list(result.method_comparison), list(result.method_comparison[0]))
    write_comparison_summary_csv(summary_path, list(result.method_comparison), filename="summary.csv")
    write_csv(
        method_evaluation_summary_path,
        _build_imm_method_evaluation_rows(result),
        list(METHOD_EVALUATION_SUMMARY_FIELDS),
    )
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
    representative_run = result.runs[0]
    transition_run = _run_mode_accumulator(
        result.scenarios[0],
        default_switching_mode_specs(),
        mode="transition",
        transition_matrix=default_transition_matrix(),
    )
    representative_traces = tuple(trace for trace in traces if trace.trajectory_id == result.runs[0].trajectory_id)
    render_posterior_timeline(posterior_timeline_plot_path, representative_traces)
    render_likelihood_strip(likelihood_strip_plot_path, representative_traces)
    switch_index = _first_switch_index(result.runs[0].true_modes)
    render_prior_likelihood_posterior_waterfall(waterfall_plot_path, representative_traces, time_index=switch_index)
    _render_imm_mixing_heatmap(representative_run, mixing_heatmap_plot_path, time_index=switch_index)
    _render_mode_conditioned_state_plot(representative_run, mode_conditioned_state_plot_path)
    _render_switch_recovery_panel(
        representative_run,
        transition_run,
        result,
        switch_recovery_plot_path,
        switch_index=switch_index,
    )
    step_card_paths = _write_imm_step_cards(step_card_dir, representative_traces, switch_index=switch_index)
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
        summary_path,
        method_evaluation_summary_path,
        decision_matrix_path,
        plot_dir,
        mode_probability_plot_path,
        state_plot_path,
        trace_dir,
        filter_step_trace_path,
        per_method_diagnostics_path,
        intermediate_plot_dir,
        posterior_timeline_plot_path,
        likelihood_strip_plot_path,
        waterfall_plot_path,
        mixing_heatmap_plot_path,
        mode_conditioned_state_plot_path,
        switch_recovery_plot_path,
        step_card_dir,
        step_card_paths,
    )


def _build_imm_method_evaluation_rows(result: IMMBenchmarkResult) -> list[dict[str, object]]:
    samples = [
        PosteriorMetricSample(
            true_label=run.true_modes[index],
            predicted_label=step.predicted_label,
            confidence=float(step.confidence),
            posterior_by_label=step.posterior_by_label,
        )
        for run in result.runs
        for index, step in enumerate(run.steps)
    ]
    metrics = compute_multiclass_posterior_metrics(samples)
    method_row = next(row for row in result.method_comparison if row["method_id"] == "imm_v1")
    return [
        build_method_evaluation_summary_row(
            method_id="imm_v1",
            study_surface="imm_switching_v1",
            evaluation_surface="state_mixing_switch",
            metrics=metrics,
            post_switch_accuracy=float(method_row["post_switch_accuracy"]),
            switch_detection_delay=float(method_row["switch_detection_delay_median"]),
            runtime_seconds=float(method_row["runtime_seconds"]),
            promotion_decision=str(method_row["promotion_decision"]),
        )
    ]


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


def _render_imm_mixing_heatmap(run, path: Path, *, time_index: int) -> None:
    matrix_rows = [
        list(run.mixing_probabilities[min(time_index, len(run.mixing_probabilities) - 1)].get(dest_mode, (0.0,) * len(run.mode_ids)))
        for dest_mode in run.mode_ids
    ]
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    image = ax.imshow(matrix_rows, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title(f"IMM mixing probabilities at t={run.times[min(time_index, len(run.times) - 1)]:.2f}")
    ax.set_xlabel("source mode")
    ax.set_ylabel("destination mode")
    ax.set_xticks(range(len(run.mode_ids)), run.mode_ids, rotation=30, ha="right")
    ax.set_yticks(range(len(run.mode_ids)), run.mode_ids)
    for row_index, row in enumerate(matrix_rows):
        for col_index, value in enumerate(row):
            ax.text(col_index, row_index, f"{value:.2f}", ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="mixing probability")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _render_mode_conditioned_state_plot(run, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
    for mode_id in run.mode_ids:
        ax.plot(
            run.times,
            [state_by_mode[mode_id][0] for state_by_mode in run.mode_state_means],
            label=f"{mode_id} position",
        )
    switch_index = _first_switch_index(run.true_modes)
    if switch_index < len(run.times):
        ax.axvline(run.times[switch_index], color="black", linestyle="--", linewidth=1.0, label="switch")
    ax.plot(run.times, run.measurements, color="0.5", linewidth=1.0, alpha=0.8, label="measurement")
    ax.set_title("Mode-conditioned position traces")
    ax.set_xlabel("time")
    ax.set_ylabel("position")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _render_switch_recovery_panel(
    run,
    transition_run,
    result: IMMBenchmarkResult,
    path: Path,
    *,
    switch_index: int,
) -> None:
    switch_index = min(switch_index, len(run.times) - 1, len(transition_run.steps) - 1)
    switched_mode = run.true_modes[switch_index]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150)

    axes[0].plot(
        run.times,
        [step.posterior_by_label.get(switched_mode, 0.0) for step in run.steps],
        marker="o",
        label="IMM posterior",
    )
    axes[0].plot(
        [step.time for step in transition_run.steps],
        [step.posterior_weights.get(switched_mode, 0.0) for step in transition_run.steps],
        marker="x",
        label="Transition posterior",
    )
    axes[0].axvline(run.times[switch_index], color="black", linestyle="--", linewidth=1.0, label="switch")
    axes[0].set_title(f"Recovery for switched mode: {switched_mode}")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("posterior")
    axes[0].legend(fontsize=8)

    comparison_rows = {str(row["method_id"]): row for row in result.method_comparison}
    methods = ["static_mode_likelihood", "transition_matrix_accumulator", "imm_v1"]
    values = [float(comparison_rows[method]["post_switch_accuracy"]) for method in methods]
    axes[1].bar(["static", "transition", "IMM"], values, color=["0.75", "0.45", "#1f77b4"])
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Aggregate post-switch accuracy")
    axes[1].set_ylabel("accuracy")
    for index, value in enumerate(values):
        axes[1].text(index, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _build_imm_filter_step_traces(result: IMMBenchmarkResult) -> tuple[FilterStepTrace, ...]:
    traces: list[FilterStepTrace] = []
    for run in result.runs:
        previous_time: float | None = None
        for time_index, step in enumerate(run.steps):
            dt = float(step.diagnostics.get("dt", 1.0 if previous_time is None else max(step.time - previous_time, 1.0e-9)))
            previous_time = step.time
            predicted_probabilities = _predicted_probabilities_from_step(step.posterior_by_label, step.log_evidence_by_label)
            state_mean = tuple(float(value) for value in run.state_means[time_index])
            state_covariance_diag = _covariance_diag(run.state_covariances[time_index])
            posterior_entropy = float(step.diagnostics.get("mode_entropy", 0.0))
            for mode_id in step.posterior_by_label:
                innovation = step.diagnostics.get(f"innovation_{mode_id}")
                innovation_tuple = (float(innovation),) if innovation is not None else None
                normalized_innovation_squared = None
                if innovation is not None:
                    normalized_innovation_squared = float(innovation) * float(innovation)
                traces.append(
                    FilterStepTrace(
                        run_id="imm_filter_v1",
                        study_id="imm_switching_v1",
                        trajectory_id=run.trajectory_id,
                        method_id="imm_v1",
                        rung="IMM",
                        time_index=time_index,
                        time=step.time,
                        dt=dt,
                        class_or_model=mode_id,
                        true_class=None,
                        true_mode=run.true_modes[time_index],
                        prior_probability=predicted_probabilities.get(mode_id),
                        predicted_probability=predicted_probabilities.get(mode_id),
                        log_transition_probability=None,
                        measurement=(float(run.measurements[time_index]),),
                        predicted_measurement=None,
                        innovation=innovation_tuple,
                        innovation_covariance_diag=None,
                        normalized_innovation_squared=normalized_innovation_squared,
                        log_likelihood=step.log_evidence_by_label.get(mode_id),
                        incremental_log_evidence=step.log_evidence_by_label.get(mode_id),
                        posterior_probability=step.posterior_by_label.get(mode_id),
                        posterior_entropy=posterior_entropy,
                        predicted_state_mean=None,
                        predicted_state_covariance_diag=None,
                        updated_state_mean=state_mean,
                        updated_state_covariance_diag=state_covariance_diag,
                        effective_sample_size=None,
                        is_resampled=None,
                    )
                )
    return tuple(traces)


def _predicted_probabilities_from_step(
    posterior_by_label: dict[str, float],
    log_evidence_by_label: dict[str, float],
) -> dict[str, float]:
    log_raw: dict[str, float] = {}
    for label, posterior in posterior_by_label.items():
        log_raw[label] = log(max(float(posterior), 1.0e-300)) - float(log_evidence_by_label.get(label, 0.0))
    if not log_raw:
        return {}
    pivot = max(log_raw.values())
    raw = {label: exp(value - pivot) for label, value in log_raw.items()}
    total = sum(raw.values())
    if total <= 1.0e-300:
        return {label: 1.0 / max(len(raw), 1) for label in raw}
    return {label: value / total for label, value in raw.items()}


def _covariance_diag(flat_covariance: tuple[float, ...]) -> tuple[float, ...]:
    width = int(round(len(flat_covariance) ** 0.5))
    if width * width != len(flat_covariance):
        return ()
    return tuple(float(flat_covariance[index * width + index]) for index in range(width))


def _first_switch_index(true_modes: tuple[str, ...]) -> int:
    if not true_modes:
        return 0
    first = true_modes[0]
    return next((index for index, mode in enumerate(true_modes) if mode != first), 0)


def _write_imm_step_cards(
    step_card_dir: Path,
    representative_traces: tuple[FilterStepTrace, ...],
    *,
    switch_index: int,
) -> tuple[Path, ...]:
    if not representative_traces:
        return ()
    final_index = max(trace.time_index for trace in representative_traces)
    selected = (0, switch_index, final_index)
    paths: list[Path] = []
    for label, time_index in zip(("t_000", "t_switch", "t_final"), selected, strict=True):
        rows = tuple(trace for trace in representative_traces if trace.time_index == time_index)
        if rows:
            paths.append(write_step_card(step_card_dir / f"{label}.md", rows))
    return tuple(paths)
