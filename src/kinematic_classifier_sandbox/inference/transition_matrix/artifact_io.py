from __future__ import annotations

import io
import json
from math import log
from pathlib import Path

from kinematic_classifier_sandbox.utils.io import write_csv
from ...runtime_paths import prepare_matplotlib
from ...render.intermediate_plots import (
    render_likelihood_strip,
    render_posterior_timeline,
    render_prior_likelihood_posterior_waterfall,
)
from ...render.step_cards import write_step_card
from ...tracing.filter_trace import FilterStepTrace, write_filter_step_trace_csv
from ...tracing.trace_validation import validate_filter_step_trace_set
from ...utils.plotting import plt
from ...utils.method_evaluation_summary import (
    METHOD_EVALUATION_SUMMARY_FIELDS,
    PosteriorMetricSample,
    build_method_evaluation_summary_row,
    compute_multiclass_posterior_metrics,
)

from .contracts import TransitionBenchmarkArtifacts, TransitionBenchmarkResult
from .reporting import (
    render_transition_benchmark_report,
    render_transition_numeric_walkthrough_markdown,
)


def _build_figure(result: TransitionBenchmarkResult):
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
    if result is None:
        from .runner import run_transition_benchmark

        analysis = run_transition_benchmark()
    else:
        analysis = result
    run_dir = Path(output_dir) / "transition_matrix_accumulator_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "transition_matrix_accumulator_report.md"
    numeric_walkthrough_path = run_dir / "transition_matrix_numeric_walkthrough.md"
    posterior_history_path = run_dir / "transition_matrix_posterior_history.csv"
    scenario_summary_path = run_dir / "transition_matrix_scenario_summary.csv"
    method_evaluation_summary_path = run_dir / "method_evaluation_summary.csv"
    config_path = run_dir / "transition_matrix_config.yaml"
    dataset_manifest_path = run_dir / "transition_matrix_dataset_manifest.json"
    plot_png_path = run_dir / "transition_matrix_diagnostics.png"
    trace_dir = run_dir / "traces"
    intermediate_plot_dir = run_dir / "plots" / "intermediate"
    step_card_dir = run_dir / "step_cards"
    filter_step_trace_path = trace_dir / "filter_step_trace.csv"
    per_method_diagnostics_path = trace_dir / "per_method_diagnostics.csv"
    posterior_timeline_plot_path = intermediate_plot_dir / "posterior_timeline_with_regimes.png"
    likelihood_strip_plot_path = intermediate_plot_dir / "innovation_likelihood_strip.png"
    waterfall_plot_path = intermediate_plot_dir / "prior_likelihood_posterior_waterfall.png"
    static_vs_transition_plot_path = intermediate_plot_dir / "static_vs_transition_flicker.png"

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
    method_evaluation_rows = _build_transition_method_evaluation_rows(analysis)
    write_csv(method_evaluation_summary_path, method_evaluation_rows, list(METHOD_EVALUATION_SUMMARY_FIELDS))
    traces = _build_transition_filter_step_traces(analysis)
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
            "prior_probability": trace.prior_probability,
            "predicted_probability": trace.predicted_probability,
            "log_likelihood": trace.log_likelihood,
            "posterior_probability": trace.posterior_probability,
            "posterior_entropy": trace.posterior_entropy,
        }
        for trace in traces
    ]
    write_csv(
        per_method_diagnostics_path,
        diagnostic_rows,
        [
            "run_id",
            "trajectory_id",
            "method_id",
            "time_index",
            "time",
            "class_or_model",
            "prior_probability",
            "predicted_probability",
            "log_likelihood",
            "posterior_probability",
            "posterior_entropy",
        ],
    )
    fig = _build_figure(analysis)
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        plot_png_path.write_bytes(buffer.getvalue())
    finally:
        plt.close(fig)
    representative = tuple(
        trace
        for trace in traces
        if trace.trajectory_id == analysis.transition_runs[0].trajectory_id
        and trace.method_id == "transition_matrix_accumulator"
    )
    render_posterior_timeline(posterior_timeline_plot_path, representative)
    render_likelihood_strip(likelihood_strip_plot_path, representative)
    switch_index = _first_switch_index(analysis.transition_runs[0].steps)
    render_prior_likelihood_posterior_waterfall(waterfall_plot_path, representative, time_index=switch_index)
    _render_static_vs_transition_plot(static_vs_transition_plot_path, analysis)
    step_card_paths = _write_transition_step_cards(step_card_dir, representative, switch_index=switch_index)
    return TransitionBenchmarkArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        numeric_walkthrough_path=numeric_walkthrough_path,
        posterior_history_path=posterior_history_path,
        scenario_summary_path=scenario_summary_path,
        method_evaluation_summary_path=method_evaluation_summary_path,
        config_path=config_path,
        dataset_manifest_path=dataset_manifest_path,
        plot_png_path=plot_png_path,
        trace_dir=trace_dir,
        filter_step_trace_path=filter_step_trace_path,
        per_method_diagnostics_path=per_method_diagnostics_path,
        intermediate_plot_dir=intermediate_plot_dir,
        posterior_timeline_plot_path=posterior_timeline_plot_path,
        likelihood_strip_plot_path=likelihood_strip_plot_path,
        waterfall_plot_path=waterfall_plot_path,
        static_vs_transition_plot_path=static_vs_transition_plot_path,
        step_card_dir=step_card_dir,
        step_card_paths=step_card_paths,
    )


def _build_transition_method_evaluation_rows(
    result: TransitionBenchmarkResult,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    run_groups = (
        ("static_mode_accumulator", "transition_label_switching_v1", "label_switching", result.static_runs, "baseline"),
        ("transition_matrix_accumulator", "transition_label_switching_v1", "label_switching", result.transition_runs, "promote"),
        ("kalman_mode_bank", "transition_label_switching_v1", "matched_endpoint_dynamics", result.kalman_runs, "competitive"),
    )
    for method_id, study_surface, evaluation_surface, runs, decision in run_groups:
        samples = [
            PosteriorMetricSample(
                true_label=step.true_mode,
                predicted_label=step.predicted_mode,
                confidence=float(step.confidence),
                posterior_by_label=step.posterior_weights,
            )
            for run in runs
            for step in run.steps
        ]
        metrics = compute_multiclass_posterior_metrics(samples)
        rows.append(
            build_method_evaluation_summary_row(
                method_id=method_id,
                study_surface=study_surface,
                evaluation_surface=evaluation_surface,
                metrics=metrics,
                post_switch_accuracy=_mean_post_switch_accuracy(runs),
                promotion_decision=decision,
            )
        )
    return rows


def _mean_post_switch_accuracy(runs) -> float:
    if not runs:
        return 0.0
    return sum(float(run.post_switch_accuracy) for run in runs) / len(runs)


def _build_transition_filter_step_traces(result: TransitionBenchmarkResult) -> tuple[FilterStepTrace, ...]:
    traces: list[FilterStepTrace] = []
    for run in (*result.static_runs, *result.transition_runs):
        for index, step in enumerate(run.steps):
            previous_posterior = step.prior_weights if index == 0 else run.steps[index - 1].posterior_weights
            posterior_entropy = -sum(
                float(value) * log(max(float(value), 1.0e-300))
                for value in step.posterior_weights.values()
            )
            for mode_id, posterior_probability in step.posterior_weights.items():
                traces.append(
                    FilterStepTrace(
                        run_id="transition_matrix_accumulator_v1",
                        study_id="transition_switching",
                        trajectory_id=run.trajectory_id,
                        method_id="transition_matrix_accumulator" if run.mode == "transition_matrix" else "static_mode_accumulator",
                        rung="Transition Matrix" if run.mode == "transition_matrix" else "Sequential Bayes",
                        time_index=step.step,
                        time=step.time,
                        dt=_step_dt(run.steps, index),
                        class_or_model=mode_id,
                        true_class=None,
                        true_mode=step.true_mode,
                        prior_probability=previous_posterior.get(mode_id),
                        predicted_probability=step.prior_weights.get(mode_id),
                        log_transition_probability=None,
                        measurement=(float(step.measurement),),
                        predicted_measurement=None,
                        innovation=(float(step.estimated_speed), float(step.estimated_accel)),
                        innovation_covariance_diag=None,
                        normalized_innovation_squared=None,
                        log_likelihood=None if step.emission_log_scores is None else step.emission_log_scores.get(mode_id),
                        incremental_log_evidence=None if step.emission_log_scores is None else step.emission_log_scores.get(mode_id),
                        posterior_probability=posterior_probability,
                        posterior_entropy=posterior_entropy,
                        predicted_state_mean=(float(step.estimated_speed), float(step.estimated_accel)),
                        predicted_state_covariance_diag=None,
                        updated_state_mean=(float(step.estimated_speed), float(step.estimated_accel)),
                        updated_state_covariance_diag=None,
                        effective_sample_size=None,
                        is_resampled=None,
                    )
                )
    return tuple(traces)


def _step_dt(steps, index: int) -> float:
    if index == 0:
        return max(float(steps[1].time - steps[0].time), 1.0e-9) if len(steps) > 1 else 1.0
    return max(float(steps[index].time - steps[index - 1].time), 1.0e-9)


def _first_switch_index(steps) -> int:
    if not steps:
        return 0
    first_mode = steps[0].true_mode
    return next((index for index, step in enumerate(steps) if step.true_mode != first_mode), 0)


def _render_static_vs_transition_plot(path: Path, result: TransitionBenchmarkResult) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    transition_run = result.transition_runs[0]
    static_run = next(run for run in result.static_runs if run.trajectory_id == transition_run.trajectory_id)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.plot([step.time for step in static_run.steps], [step.confidence for step in static_run.steps], label="static confidence", color="#6b7280")
    ax.plot([step.time for step in transition_run.steps], [step.confidence for step in transition_run.steps], label="transition confidence", color="#059669")
    ax.set_title("Static vs Transition Confidence", loc="left")
    ax.set_xlabel("time")
    ax.set_ylabel("winner posterior")
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _write_transition_step_cards(step_card_dir: Path, traces: tuple[FilterStepTrace, ...], *, switch_index: int) -> tuple[Path, ...]:
    if not traces:
        return ()
    final_index = max(trace.time_index for trace in traces)
    selected = (0, switch_index, final_index)
    paths: list[Path] = []
    for label, time_index in zip(("t_000", "t_switch", "t_final"), selected, strict=True):
        rows = tuple(trace for trace in traces if trace.time_index == time_index)
        if rows:
            paths.append(write_step_card(step_card_dir / f"{label}.md", rows))
    return tuple(paths)
