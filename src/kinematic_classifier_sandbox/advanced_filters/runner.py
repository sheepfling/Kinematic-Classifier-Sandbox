from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import os
from pathlib import Path
import time as wall_time

import numpy as np

from .contracts import AdvancedFilterStep, validate_advanced_filter_step
from .imm_filter import IMMFilter
from .linear_gaussian import LinearGaussianModeSpec
from ..trajectory_generator import generate_switching_scenarios
from ..transition_matrix_accumulator import (
    SwitchingScenario,
    default_switching_mode_specs,
    default_transition_matrix,
    _run_mode_accumulator,
)


def _prepare_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp/kinematic-classifier-sandbox-mpl")))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


@dataclass(frozen=True, slots=True)
class IMMSwitchingRun:
    trajectory_id: str
    scenario_name: str
    true_modes: tuple[str, ...]
    times: tuple[float, ...]
    measurements: tuple[float, ...]
    steps: tuple[AdvancedFilterStep, ...]
    state_means: tuple[tuple[float, float, float], ...]
    state_covariances: tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class IMMBenchmarkResult:
    scenarios: tuple[SwitchingScenario, ...]
    runs: tuple[IMMSwitchingRun, ...]
    metrics: dict[str, float | int | str]
    method_comparison: tuple[dict[str, float | int | str], ...]


@dataclass(frozen=True, slots=True)
class IMMArtifacts:
    run_dir: Path
    config_path: Path
    report_path: Path
    mode_probability_history_path: Path
    mixing_probability_history_path: Path
    mode_likelihood_history_path: Path
    state_estimate_history_path: Path
    posterior_history_path: Path
    switching_detection_metrics_path: Path
    method_comparison_path: Path
    decision_matrix_path: Path
    plot_dir: Path
    mode_probability_plot_path: Path
    state_plot_path: Path


def default_imm_mode_specs() -> list[LinearGaussianModeSpec]:
    return [
        LinearGaussianModeSpec("stationary", process_noise_scale=0.002, measurement_noise=0.04, acceleration_bias=0.0),
        LinearGaussianModeSpec("constant_velocity", process_noise_scale=0.010, measurement_noise=0.04, acceleration_bias=0.0),
        LinearGaussianModeSpec("constant_acceleration", process_noise_scale=0.050, measurement_noise=0.04, acceleration_bias=0.12),
        LinearGaussianModeSpec("braking", process_noise_scale=0.050, measurement_noise=0.04, acceleration_bias=-0.20),
        LinearGaussianModeSpec("maneuver", process_noise_scale=0.250, measurement_noise=0.04, acceleration_bias=0.0),
    ]


def default_imm_transition_matrix() -> np.ndarray:
    return np.array(
        [
            [0.92, 0.06, 0.01, 0.00, 0.01],
            [0.01, 0.88, 0.03, 0.05, 0.03],
            [0.00, 0.08, 0.84, 0.03, 0.05],
            [0.08, 0.10, 0.02, 0.78, 0.02],
            [0.01, 0.12, 0.08, 0.04, 0.75],
        ],
        dtype=np.float64,
    )


def generate_imm_switching_witnesses(*, seed: int = 17, replicas: int = 12) -> tuple[SwitchingScenario, ...]:
    scenarios: list[SwitchingScenario] = []
    for replica in range(replicas):
        for artifact in generate_switching_scenarios(seed=seed + replica * 41):
            params = artifact.generator_parameters
            segment_modes = list(params["segment_modes"])
            switch_time = float(params["switch_time"])
            true_modes = tuple(segment_modes[0] if time < switch_time else segment_modes[1] for time in artifact.times)
            scenarios.append(
                SwitchingScenario(
                    trajectory_id=f"{artifact.trajectory_id}_{replica}",
                    scenario_name=artifact.scenario_id,
                    seed=artifact.seed,
                    times=artifact.times,
                    measurements=artifact.measurements,
                    true_mode_by_step=true_modes,
                )
            )
        scenarios.append(_make_acceleration_then_coast(seed=seed + replica * 41 + 997, replica=replica))
    return tuple(scenarios)


def _make_acceleration_then_coast(*, seed: int, replica: int) -> SwitchingScenario:
    rng = np.random.default_rng(seed)
    times = tuple(0.4 * index for index in range(12))
    positions = [0.0]
    velocities = [0.2]
    for index in range(1, len(times)):
        dt = times[index] - times[index - 1]
        accel = 0.55 if times[index - 1] < 2.0 else 0.0
        velocities.append(velocities[-1] + accel * dt)
        positions.append(positions[-1] + velocities[-2] * dt + 0.5 * accel * dt * dt)
    measurements = tuple(float(position + rng.normal(0.0, 0.05)) for position in positions)
    true_modes = tuple("constant_acceleration" if time < 2.0 else "constant_velocity" for time in times)
    return SwitchingScenario(
        trajectory_id=f"acceleration_then_coast_{replica}",
        scenario_name="acceleration_then_coast",
        seed=seed,
        times=times,
        measurements=measurements,
        true_mode_by_step=true_modes,
    )


def run_imm_switching_benchmark(*, seed: int = 17, replicas: int = 12) -> IMMBenchmarkResult:
    scenarios = generate_imm_switching_witnesses(seed=seed, replicas=replicas)
    runs: list[IMMSwitchingRun] = []
    start = wall_time.perf_counter()
    for scenario in scenarios:
        imm = IMMFilter(default_imm_mode_specs(), default_imm_transition_matrix())
        imm.reset(scenario.trajectory_id, np.array([scenario.measurements[0]], dtype=np.float64))
        steps: list[AdvancedFilterStep] = []
        state_means: list[tuple[float, float, float]] = []
        state_covariances: list[tuple[float, ...]] = []
        for time_value, measurement in zip(scenario.times, scenario.measurements, strict=True):
            step = imm.update(time_value, np.array([measurement], dtype=np.float64))
            validate_advanced_filter_step(step)
            summary = imm.state_summary()
            steps.append(step)
            state_means.append(tuple(float(value) for value in summary.state_mean))
            assert summary.state_covariance is not None
            state_covariances.append(tuple(float(value) for value in summary.state_covariance.ravel()))
        runs.append(
            IMMSwitchingRun(
                trajectory_id=scenario.trajectory_id,
                scenario_name=scenario.scenario_name,
                true_modes=scenario.true_mode_by_step,
                times=scenario.times,
                measurements=scenario.measurements,
                steps=tuple(steps),
                state_means=tuple(state_means),
                state_covariances=tuple(state_covariances),
            )
        )
    runtime_seconds = wall_time.perf_counter() - start
    imm_metrics = _compute_imm_metrics(runs)
    imm_metrics["runtime_seconds"] = runtime_seconds
    method_comparison = _build_method_comparison(scenarios, runs, imm_metrics)
    baseline_post_switch = max(float(row["post_switch_accuracy"]) for row in method_comparison if row["method_id"] != "imm_v1")
    imm_metrics["promotion_decision"] = (
        "promote"
        if float(imm_metrics["post_switch_accuracy"]) > baseline_post_switch + 0.05
        else "revise"
    )
    method_comparison = _build_method_comparison(scenarios, runs, imm_metrics)
    return IMMBenchmarkResult(scenarios=scenarios, runs=tuple(runs), metrics=imm_metrics, method_comparison=tuple(method_comparison))


def _compute_imm_metrics(runs: list[IMMSwitchingRun]) -> dict[str, float | int | str]:
    total = 0
    correct = 0
    post_total = 0
    post_correct = 0
    nll_values: list[float] = []
    entropy_values: list[float] = []
    delays: list[float] = []
    position_sq: list[float] = []
    velocity_sq: list[float] = []
    for run in runs:
        switch_index = next((idx for idx, mode in enumerate(run.true_modes) if mode != run.true_modes[0]), len(run.true_modes))
        detected_time: float | None = None
        new_mode = run.true_modes[switch_index] if switch_index < len(run.true_modes) else run.true_modes[-1]
        switch_time = run.times[switch_index] if switch_index < len(run.times) else run.times[-1]
        for index, (step, true_mode) in enumerate(zip(run.steps, run.true_modes, strict=True)):
            total += 1
            correct += int(step.predicted_label == true_mode)
            nll_values.append(-np.log(max(step.posterior_by_label.get(true_mode, 1.0e-300), 1.0e-300)))
            posterior = np.array(list(step.posterior_by_label.values()), dtype=np.float64)
            entropy_values.append(float(-np.sum(posterior * np.log(np.maximum(posterior, 1.0e-300)))))
            state_mean = run.state_means[index]
            position_sq.append((state_mean[0] - run.measurements[index]) ** 2)
            if index > 0:
                dt = max(run.times[index] - run.times[index - 1], 1.0e-9)
                measured_velocity = (run.measurements[index] - run.measurements[index - 1]) / dt
                velocity_sq.append((state_mean[1] - measured_velocity) ** 2)
            if index >= switch_index:
                post_total += 1
                post_correct += int(step.predicted_label == true_mode)
                if detected_time is None and (
                    step.posterior_by_label.get(new_mode, 0.0) >= 0.6
                    or step.predicted_label == new_mode
                ):
                    detected_time = step.time
        if switch_index < len(run.true_modes) and detected_time is not None:
            delays.append(max(0.0, detected_time - switch_time))
    return {
        "method_id": "imm_v1",
        "trajectory_count": len(runs),
        "mode_accuracy": correct / max(total, 1),
        "post_switch_accuracy": post_correct / max(post_total, 1),
        "switch_detection_delay_median": float(np.median(delays)) if delays else float("nan"),
        "mode_nll": float(np.mean(nll_values)) if nll_values else float("nan"),
        "mean_entropy": float(np.mean(entropy_values)) if entropy_values else float("nan"),
        "state_position_rmse": float(np.sqrt(np.mean(position_sq))) if position_sq else float("nan"),
        "state_velocity_rmse": float(np.sqrt(np.mean(velocity_sq))) if velocity_sq else float("nan"),
        "detected_switch_count": len(delays),
        "promotion_decision": "promote" if post_correct / max(post_total, 1) >= 0.60 else "revise",
    }


def _build_method_comparison(
    scenarios: tuple[SwitchingScenario, ...],
    runs: list[IMMSwitchingRun],
    imm_metrics: dict[str, float | int | str],
) -> list[dict[str, float | int | str]]:
    specs = default_switching_mode_specs()
    transition = default_transition_matrix()
    static_runs = [_run_mode_accumulator(scenario, specs, mode="static") for scenario in scenarios]
    transition_runs = [_run_mode_accumulator(scenario, specs, mode="transition", transition_matrix=transition) for scenario in scenarios]

    def summarize(method_id: str, run_rows) -> dict[str, float | int | str]:
        return {
            "method_id": method_id,
            "corpus_objective_id": "imm_switching_v1",
            "scenario_family": "switching_1d",
            "mode_accuracy": float(np.mean([run.accuracy for run in run_rows])),
            "post_switch_accuracy": float(np.mean([run.post_switch_accuracy for run in run_rows])),
            "switch_detection_delay_median": "",
            "state_position_rmse": "",
            "state_velocity_rmse": "",
            "nll": "",
            "mean_entropy": "",
            "runtime_seconds": "",
            "promotion_decision": "baseline",
        }

    return [
        summarize("static_mode_likelihood", static_runs),
        summarize("transition_matrix_accumulator", transition_runs),
        {
            "method_id": "imm_v1",
            "corpus_objective_id": "imm_switching_v1",
            "scenario_family": "switching_1d",
            "mode_accuracy": imm_metrics["mode_accuracy"],
            "post_switch_accuracy": imm_metrics["post_switch_accuracy"],
            "switch_detection_delay_median": imm_metrics["switch_detection_delay_median"],
            "state_position_rmse": imm_metrics["state_position_rmse"],
            "state_velocity_rmse": imm_metrics["state_velocity_rmse"],
            "nll": imm_metrics["mode_nll"],
            "mean_entropy": imm_metrics["mean_entropy"],
            "runtime_seconds": imm_metrics["runtime_seconds"],
            "promotion_decision": imm_metrics["promotion_decision"],
        },
    ]


def write_imm_artifacts(output_dir: str | Path, *, result: IMMBenchmarkResult | None = None) -> IMMArtifacts:
    if result is None:
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
                probability_rows.append({"trajectory_id": run.trajectory_id, "scenario_name": run.scenario_name, "time": step.time, "true_mode": run.true_modes[index], "mode_id": mode_id, "probability": probability, "predicted_mode": step.predicted_label, "confidence": step.confidence})
            for mode_id, value in step.log_evidence_by_label.items():
                likelihood_rows.append({"trajectory_id": run.trajectory_id, "scenario_name": run.scenario_name, "time": step.time, "mode_id": mode_id, "log_likelihood": value})
            mixing_rows.append({"trajectory_id": run.trajectory_id, "scenario_name": run.scenario_name, "time": step.time, "mixing_prob_sum_error_max": step.diagnostics["mixing_prob_sum_error_max"]})
            state_rows.append({"trajectory_id": run.trajectory_id, "scenario_name": run.scenario_name, "time": step.time, "measurement": run.measurements[index], "state_position": run.state_means[index][0], "state_velocity": run.state_means[index][1], "state_acceleration": run.state_means[index][2], "true_mode": run.true_modes[index], "predicted_mode": step.predicted_label})
    _write_csv(mode_probability_history_path, probability_rows, ["trajectory_id", "scenario_name", "time", "true_mode", "mode_id", "probability", "predicted_mode", "confidence"])
    _write_csv(posterior_history_path, probability_rows, ["trajectory_id", "scenario_name", "time", "true_mode", "mode_id", "probability", "predicted_mode", "confidence"])
    _write_csv(mode_likelihood_history_path, likelihood_rows, ["trajectory_id", "scenario_name", "time", "mode_id", "log_likelihood"])
    _write_csv(mixing_probability_history_path, mixing_rows, ["trajectory_id", "scenario_name", "time", "mixing_prob_sum_error_max"])
    _write_csv(state_estimate_history_path, state_rows, ["trajectory_id", "scenario_name", "time", "measurement", "state_position", "state_velocity", "state_acceleration", "true_mode", "predicted_mode"])
    metric_rows = [result.metrics]
    _write_csv(switching_detection_metrics_path, metric_rows, list(metric_rows[0]))
    _write_csv(method_comparison_path, list(result.method_comparison), list(result.method_comparison[0]))
    decision_rows = [
        {
            "method": "IMM",
            "failure_case": "switching modes",
            "baseline_failed": "yes",
            "method_improved": result.metrics["promotion_decision"] == "promote",
            "cost_acceptable": "yes",
            "decision": result.metrics["promotion_decision"],
        },
        {"method": "PF", "failure_case": "nonlinear drag / outlier noise", "baseline_failed": "see advanced_filter_comparison_v1", "method_improved": "see advanced_filter_comparison_v1", "cost_acceptable": "see advanced_filter_comparison_v1", "decision": "see advanced_filter_comparison_v1"},
        {"method": "RBPF", "failure_case": "latent maneuver onset", "baseline_failed": "see advanced_filter_comparison_v1", "method_improved": "see advanced_filter_comparison_v1", "cost_acceptable": "see advanced_filter_comparison_v1", "decision": "see advanced_filter_comparison_v1"},
    ]
    _write_csv(decision_matrix_path, decision_rows, ["method", "failure_case", "baseline_failed", "method_improved", "cost_acceptable", "decision"])
    config_path.write_text(_render_config_text(), encoding="utf-8")
    report_path.write_text(render_imm_report(result), encoding="utf-8")
    _render_imm_plots(result, mode_probability_plot_path, state_plot_path)
    return IMMArtifacts(run_dir, config_path, report_path, mode_probability_history_path, mixing_probability_history_path, mode_likelihood_history_path, state_estimate_history_path, posterior_history_path, switching_detection_metrics_path, method_comparison_path, decision_matrix_path, plot_dir, mode_probability_plot_path, state_plot_path)


def render_imm_report(result: IMMBenchmarkResult) -> str:
    metrics = result.metrics
    comparison_rows = "\n".join(
        f"| {row['method_id']} | {row['mode_accuracy']} | {row['post_switch_accuracy']} | {row['promotion_decision']} |"
        for row in result.method_comparison
    )
    return f"""# IMM Filter V1 Report

This is the IMM switching-witness report for the evaluation-first advanced-filter rung. PF and RBPF are evaluated in their own nonlinear and latent-mode witness reports, then summarized with IMM in `artifacts/advanced_filter_comparison_v1`.

## Evaluation Target

- Failure case: static class/mode assumption under switching dynamics.
- Baselines: static mode likelihood and transition-matrix accumulator.
- Advanced method: IMM with state mixing and one Kalman filter per mode.

## Metrics

- Mode accuracy: `{metrics['mode_accuracy']}`
- Post-switch accuracy: `{metrics['post_switch_accuracy']}`
- Switch detection delay median: `{metrics['switch_detection_delay_median']}`
- Mode NLL: `{metrics['mode_nll']}`
- Mean entropy: `{metrics['mean_entropy']}`
- State position RMSE: `{metrics['state_position_rmse']}`
- State velocity RMSE: `{metrics['state_velocity_rmse']}`
- Decision: `{metrics['promotion_decision']}`

## Method Comparison

| Method | Mode accuracy | Post-switch accuracy | Decision |
| --- | ---: | ---: | --- |
{comparison_rows}

## Required Interpretation Order

1. Confirm switching witness corpus coverage.
2. Inspect mode evidence and posterior history.
3. Compare post-switch accuracy and switch delay.
4. Check state RMSE and entropy.
5. Assign promote/revise/reject/defer.

PF and RBPF decisions are intentionally left to `artifacts/advanced_filter_comparison_v1`, where their targeted witness metrics are compared beside IMM.
"""


def run_advanced_filter_comparison(output_dir: str | Path) -> IMMArtifacts:
    return write_imm_artifacts(output_dir)


def _render_config_text() -> str:
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


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render_imm_plots(result: IMMBenchmarkResult, mode_probability_path: Path, state_path: Path) -> None:
    plt = _prepare_matplotlib()
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
