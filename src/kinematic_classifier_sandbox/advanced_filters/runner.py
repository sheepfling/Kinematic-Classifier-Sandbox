from __future__ import annotations

import time as wall_time
from pathlib import Path

import numpy.random as random
from numpy import array, float64, log, maximum, mean, median, ndarray, sqrt
from numpy import sum as nsum

from ..inference.transition_matrix_accumulator import (
    SwitchingScenario,
    _run_mode_accumulator,
    default_switching_mode_specs,
    default_transition_matrix,
)
from .artifact_io import write_imm_artifacts
from .contracts import (
    AdvancedFilterStep,
    IMMArtifacts,
    IMMBenchmarkResult,
    IMMSwitchingRun,
)
from .evaluation import AdvancedFilterComparisonArtifacts, write_advanced_filter_comparison_artifacts
from .imm_filter import IMMFilter
from .linear_gaussian import LinearGaussianModeSpec
from .protocols import validate_advanced_filter_step
from .surface import AdvancedFilterSurface
from ..witnesses.trajectory_scenarios import generate_switching_scenarios


def default_imm_mode_specs() -> list[LinearGaussianModeSpec]:
    return [
        LinearGaussianModeSpec("stationary", process_noise_scale=0.002, measurement_noise=0.04, acceleration_bias=0.0),
        LinearGaussianModeSpec("constant_velocity", process_noise_scale=0.010, measurement_noise=0.04, acceleration_bias=0.0),
        LinearGaussianModeSpec("constant_acceleration", process_noise_scale=0.050, measurement_noise=0.04, acceleration_bias=0.12),
        LinearGaussianModeSpec("braking", process_noise_scale=0.050, measurement_noise=0.04, acceleration_bias=-0.20),
        LinearGaussianModeSpec("maneuver", process_noise_scale=0.250, measurement_noise=0.04, acceleration_bias=0.0),
    ]


def default_imm_transition_matrix() -> ndarray:
    return array(
        [
            [0.92, 0.06, 0.01, 0.00, 0.01],
            [0.01, 0.88, 0.03, 0.05, 0.03],
            [0.00, 0.08, 0.84, 0.03, 0.05],
            [0.08, 0.10, 0.02, 0.78, 0.02],
            [0.01, 0.12, 0.08, 0.04, 0.75],
        ],
        dtype=float64,
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
    rng = random.default_rng(seed)
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
        imm.reset(scenario.trajectory_id, array([scenario.measurements[0]], dtype=float64))
        steps: list[AdvancedFilterStep] = []
        state_means: list[tuple[float, float, float]] = []
        state_covariances: list[tuple[float, ...]] = []
        mode_state_means: list[dict[str, tuple[float, float, float]]] = []
        mode_state_covariances: list[dict[str, tuple[float, ...]]] = []
        mixing_probabilities: list[dict[str, tuple[float, ...]]] = []
        for time_value, measurement in zip(scenario.times, scenario.measurements, strict=True):
            step = imm.update(time_value, array([measurement], dtype=float64))
            validate_advanced_filter_step(step)
            summary = imm.state_summary()
            steps.append(step)
            state_means.append(tuple(float(value) for value in summary.state_mean))
            assert summary.state_covariance is not None
            state_covariances.append(tuple(float(value) for value in summary.state_covariance.ravel()))
            assert imm.state is not None
            mode_state_means.append(
                {
                    mode_id: tuple(float(value) for value in imm.state.mode_states[mode_id].mean)
                    for mode_id in imm.mode_ids
                }
            )
            mode_state_covariances.append(
                {
                    mode_id: tuple(float(value) for value in imm.state.mode_states[mode_id].covariance.ravel())
                    for mode_id in imm.mode_ids
                }
            )
            mixing_probabilities.append(
                {
                    mode_id: tuple(float(value) for value in imm.state.latest_mixing_probabilities.get(mode_id, ()))
                    for mode_id in imm.mode_ids
                }
            )
        runs.append(
            IMMSwitchingRun(
                trajectory_id=scenario.trajectory_id,
                scenario_name=scenario.scenario_name,
                mode_ids=tuple(imm.mode_ids),
                true_modes=scenario.true_mode_by_step,
                times=scenario.times,
                measurements=scenario.measurements,
                steps=tuple(steps),
                state_means=tuple(state_means),
                state_covariances=tuple(state_covariances),
                mode_state_means=tuple(mode_state_means),
                mode_state_covariances=tuple(mode_state_covariances),
                mixing_probabilities=tuple(mixing_probabilities),
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
            nll_values.append(-log(max(step.posterior_by_label.get(true_mode, 1.0e-300), 1.0e-300)))
            posterior = array(list(step.posterior_by_label.values()), dtype=float64)
            entropy_values.append(float(-nsum(posterior * log(maximum(posterior, 1.0e-300)))))
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
        "switch_detection_delay_median": float(median(delays)) if delays else float("nan"),
        "mode_nll": float(mean(nll_values)) if nll_values else float("nan"),
        "mean_entropy": float(mean(entropy_values)) if entropy_values else float("nan"),
        "state_position_rmse": float(sqrt(mean(position_sq))) if position_sq else float("nan"),
        "state_velocity_rmse": float(sqrt(mean(velocity_sq))) if velocity_sq else float("nan"),
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
            "mode_accuracy": float(mean([run.accuracy for run in run_rows])),
            "post_switch_accuracy": float(mean([run.post_switch_accuracy for run in run_rows])),
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

def imm_witness_surface() -> AdvancedFilterSurface[IMMBenchmarkResult, IMMArtifacts]:
    return AdvancedFilterSurface(
        study_id="imm_filter_v1",
        run=run_imm_switching_benchmark,
        write_artifacts=write_imm_artifacts,
        describe_artifacts=lambda artifacts: (
            str(artifacts.run_dir),
            str(artifacts.report_path),
            str(artifacts.method_comparison_path),
        ),
        metadata={
            "study_kind": "1d_witness",
            "problem_family": "imm_1d",
        },
    )

def run_advanced_filter_comparison(output_dir: str | Path) -> AdvancedFilterComparisonArtifacts:
    return write_advanced_filter_comparison_artifacts(output_dir)


def advanced_filter_comparison_surface() -> AdvancedFilterSurface[None, AdvancedFilterComparisonArtifacts]:
    return AdvancedFilterSurface(
        study_id="advanced_filter_comparison_v1",
        run=lambda: None,
        write_artifacts=write_advanced_filter_comparison_artifacts,
        describe_artifacts=lambda artifacts: (
            str(artifacts.run_dir),
            str(artifacts.method_comparison_path),
            str(artifacts.gate_matrix_path),
            str(artifacts.decision_matrix_path),
            str(artifacts.report_path),
        ),
        metadata={
            "study_kind": "comparison",
            "problem_family": "advanced_filters_1d",
        },
    )
