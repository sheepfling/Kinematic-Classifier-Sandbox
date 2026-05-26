from __future__ import annotations

import io
import json
import random
from dataclasses import dataclass
from math import log
from pathlib import Path

from ...utils.io import write_csv

from ...markdown_builder import MarkdownDocument
from ...utils.math import (
    _add_matrices,
    _identity,
    _innovation_log_likelihood,
    _least_squares_slope,
    _local_quadratic_acceleration,
    _logsumexp,
    _matmul,
    _matvec,
    _transpose,
    kalman_transition_and_noise,
    kalman_update_scalar,
)
from ...utils.plotting import plt
from ...utils.math import (
    normalize_log_scores as _normalize_log_scores,
)


def _unique_class_names(model_specs: tuple[KalmanModelSpec, ...]) -> tuple[str, ...]:
    ordered: list[str] = []
    for spec in model_specs:
        if spec.class_name not in ordered:
            ordered.append(spec.class_name)
    return tuple(ordered)


def _effective_measurement_sigma(
    predicted_covariance: list[list[float]],
    measurement: float,
    predicted_mean: list[float],
    base_measurement_sigma: float,
) -> float:
    innovation = measurement - predicted_mean[0]
    baseline_variance = predicted_covariance[0][0] + base_measurement_sigma * base_measurement_sigma
    normalized_innovation_squared = (innovation * innovation) / max(baseline_variance, 1e-9)
    gate = 9.0
    if normalized_innovation_squared <= gate:
        return base_measurement_sigma
    inflation = normalized_innovation_squared / gate
    return base_measurement_sigma * (inflation ** 0.5)


def _next_process_scale(
    *,
    previous_scale: float,
    innovation: float,
    innovation_variance: float,
) -> float:
    normalized_innovation_squared = (innovation * innovation) / max(innovation_variance, 1e-9)
    trigger_nis = 2.0
    if normalized_innovation_squared <= trigger_nis:
        desired_scale = 1.0
    else:
        desired_scale = min(2.5, 1.0 + 0.20 * (normalized_innovation_squared - trigger_nis))
    smoothed_scale = 0.90 * previous_scale + 0.10 * desired_scale
    return max(1.0, min(2.5, smoothed_scale))


@dataclass(frozen=True, slots=True)
class KalmanModelSpec:
    name: str
    class_name: str
    state_dim: int
    process_sigma: float
    measurement_sigma: float
    initial_covariance_scale: float
    prior_weight: float


@dataclass(frozen=True, slots=True)
class KalmanTrajectory:
    trajectory_id: str
    true_class: str
    scenario_name: str
    seed: int
    times: tuple[float, ...]
    measurements: tuple[float, ...]
    true_position: tuple[float, ...]
    true_velocity: tuple[float, ...]
    true_acceleration: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class KalmanPosteriorStep:
    time: float
    measurement: float
    posterior_weights: dict[str, float]
    log_likelihood_terms: dict[str, float]
    innovations: dict[str, float]
    innovation_variances: dict[str, float]
    predicted_class: str
    confidence: float


@dataclass(frozen=True, slots=True)
class KalmanFilterState:
    mean: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class KalmanClassificationRun:
    trajectory_id: str
    true_class: str
    scenario_name: str
    steps: tuple[KalmanPosteriorStep, ...]
    final_weights: dict[str, float]
    final_predicted_class: str
    final_confidence: float
    final_states: dict[str, KalmanFilterState]


@dataclass(frozen=True, slots=True)
class KalmanBenchmarkSummary:
    total_trajectories: int
    final_accuracy: float
    per_class_accuracy: dict[str, float]
    confusion_counts: dict[str, dict[str, int]]


@dataclass(frozen=True, slots=True)
class KalmanBenchmarkResult:
    model_specs: tuple[KalmanModelSpec, ...]
    trajectories: tuple[KalmanTrajectory, ...]
    runs: tuple[KalmanClassificationRun, ...]
    summary: KalmanBenchmarkSummary


@dataclass(frozen=True, slots=True)
class KalmanBenchmarkArtifacts:
    run_dir: Path
    report_path: Path
    innovation_history_path: Path
    state_estimate_history_path: Path
    posterior_history_path: Path
    confusion_matrix_path: Path
    config_path: Path
    dataset_manifest_path: Path
    model_definitions_path: Path
    plot_png_path: Path


def default_kalman_model_specs() -> tuple[KalmanModelSpec, ...]:
    return (
        KalmanModelSpec(
            name="stationary",
            class_name="stationary",
            state_dim=1,
            process_sigma=0.04,
            measurement_sigma=0.20,
            initial_covariance_scale=4.0,
            prior_weight=1.0 / 3.0,
        ),
        KalmanModelSpec(
            name="constant_velocity",
            class_name="constant_velocity",
            state_dim=2,
            process_sigma=0.14,
            measurement_sigma=0.20,
            initial_covariance_scale=5.0,
            prior_weight=1.0 / 3.0,
        ),
        KalmanModelSpec(
            name="constant_acceleration",
            class_name="constant_acceleration",
            state_dim=3,
            process_sigma=0.24,
            measurement_sigma=0.20,
            initial_covariance_scale=6.0,
            prior_weight=1.0 / 3.0,
        ),
    )


def _transition_and_noise(model: KalmanModelSpec, dt: float, process_scale: float = 1.0) -> tuple[list[list[float]], list[list[float]]]:
    return kalman_transition_and_noise(model.state_dim, model.process_sigma, dt, process_scale)


def _predict(
    mean: list[float],
    covariance: list[list[float]],
    model: KalmanModelSpec,
    dt: float,
    process_scale: float = 1.0,
) -> tuple[list[float], list[list[float]]]:
    transition, process_noise = _transition_and_noise(model, dt, process_scale)
    predicted_mean = _matvec(transition, mean)
    predicted_covariance = _add_matrices(_matmul(_matmul(transition, covariance), _transpose(transition)), process_noise)
    return predicted_mean, predicted_covariance


def _update(
    predicted_mean: list[float],
    predicted_covariance: list[list[float]],
    measurement: float,
    measurement_sigma: float,
) -> tuple[list[float], list[list[float]], float, float]:
    return _update_scalar_measurement(
        predicted_mean,
        predicted_covariance,
        measurement,
        measurement_sigma * measurement_sigma,
        [1.0] + [0.0] * (len(predicted_mean) - 1),
    )


def _update_scalar_measurement(
    predicted_mean: list[float],
    predicted_covariance: list[list[float]],
    measurement: float,
    measurement_variance: float,
    h_vector: list[float],
) -> tuple[list[float], list[list[float]], float, float]:
    return kalman_update_scalar(predicted_mean, predicted_covariance, measurement, measurement_variance, h_vector)


def _pad_state(mean: list[float], covariance: list[list[float]]) -> KalmanFilterState:
    padded_mean = [0.0, 0.0, 0.0]
    padded_covariance = [[0.0, 0.0, 0.0] for _ in range(3)]
    for row in range(len(mean)):
        padded_mean[row] = mean[row]
        for col in range(len(mean)):
            padded_covariance[row][col] = covariance[row][col]
    return KalmanFilterState(
        mean=tuple(padded_mean),
        covariance=tuple(tuple(row) for row in padded_covariance),
    )


def _make_kalman_trajectory(
    *,
    trajectory_id: str,
    true_class: str,
    scenario_name: str,
    seed: int,
    times: tuple[float, ...],
    position0: float,
    velocity0: float,
    acceleration: float,
    measurement_sigma: float,
) -> KalmanTrajectory:
    rng = random.Random(seed)
    positions = []
    velocities = []
    accelerations = []
    for time in times:
        positions.append(position0 + velocity0 * time + 0.5 * acceleration * time * time)
        velocities.append(velocity0 + acceleration * time)
        accelerations.append(acceleration)
    measurements = tuple(position + rng.gauss(0.0, measurement_sigma) for position in positions)
    return KalmanTrajectory(
        trajectory_id=trajectory_id,
        true_class=true_class,
        scenario_name=scenario_name,
        seed=seed,
        times=times,
        measurements=measurements,
        true_position=tuple(positions),
        true_velocity=tuple(velocities),
        true_acceleration=tuple(accelerations),
    )


def generate_kalman_bank_trajectories(
    *,
    seed: int = 7,
    trajectories_per_class: int = 4,
    measurement_sigma: float = 0.20,
) -> tuple[KalmanTrajectory, ...]:
    rng = random.Random(seed)
    trajectories: list[KalmanTrajectory] = []
    for index in range(trajectories_per_class):
        start = rng.uniform(-2.0, 2.0)
        stationary_times = tuple(float(step) for step in range(10))
        trajectories.append(
            _make_kalman_trajectory(
                trajectory_id=f"stationary_regular_{index}",
                true_class="stationary",
                scenario_name="stationary_regular",
                seed=seed + 10 + index,
                times=stationary_times,
                position0=start,
                velocity0=0.0,
                acceleration=0.0,
                measurement_sigma=measurement_sigma,
            )
        )
        velocity = rng.uniform(0.8, 1.6)
        cv_times = tuple(float(step) for step in range(10))
        trajectories.append(
            _make_kalman_trajectory(
                trajectory_id=f"constant_velocity_regular_{index}",
                true_class="constant_velocity",
                scenario_name="constant_velocity_regular",
                seed=seed + 100 + index,
                times=cv_times,
                position0=start,
                velocity0=velocity,
                acceleration=0.0,
                measurement_sigma=measurement_sigma,
            )
        )
        irregular_times = (0.0, 0.7, 1.6, 2.8, 4.1, 5.0, 6.6, 7.4, 8.9, 10.0)
        trajectories.append(
            _make_kalman_trajectory(
                trajectory_id=f"constant_velocity_irregular_{index}",
                true_class="constant_velocity",
                scenario_name="constant_velocity_irregular",
                seed=seed + 200 + index,
                times=irregular_times,
                position0=start,
                velocity0=velocity,
                acceleration=0.0,
                measurement_sigma=measurement_sigma,
            )
        )
        acceleration = rng.uniform(0.18, 0.40)
        ca_times = tuple(float(step) for step in range(10))
        trajectories.append(
            _make_kalman_trajectory(
                trajectory_id=f"constant_acceleration_regular_{index}",
                true_class="constant_acceleration",
                scenario_name="constant_acceleration_regular",
                seed=seed + 300 + index,
                times=ca_times,
                position0=start,
                velocity0=rng.uniform(0.1, 0.5),
                acceleration=acceleration,
                measurement_sigma=measurement_sigma,
            )
        )
    return tuple(trajectories)


def run_kalman_filter_bank(
    trajectory: KalmanTrajectory,
    model_specs: tuple[KalmanModelSpec, ...],
    *,
    prior: dict[str, float] | None = None,
    robust_measurement_update: bool = True,
    adaptive_process_noise: bool = True,
    derived_velocity_observation: bool = False,
    derived_acceleration_observation: bool = False,
    velocity_measurements: tuple[float, ...] | None = None,
    velocity_measurement_sigma: float | None = None,
) -> KalmanClassificationRun:
    class_names = _unique_class_names(model_specs)
    total_prior = sum(spec.prior_weight for spec in model_specs)
    if prior is None:
        model_posterior = {spec.name: spec.prior_weight / total_prior for spec in model_specs}
    else:
        model_posterior = {}
        for class_name in class_names:
            members = [spec for spec in model_specs if spec.class_name == class_name]
            class_prior = max(prior.get(class_name, 0.0), 0.0)
            share = class_prior / max(len(members), 1)
            for spec in members:
                model_posterior[spec.name] = share
        normalization = sum(model_posterior.values())
        if normalization <= 1e-12:
            model_posterior = {spec.name: spec.prior_weight / total_prior for spec in model_specs}
        else:
            model_posterior = {name: value / normalization for name, value in model_posterior.items()}
    states = {}
    process_scales = {}
    for spec in model_specs:
        mean = [trajectory.measurements[0]] + [0.0] * (spec.state_dim - 1)
        covariance = _identity(spec.state_dim)
        for row in range(spec.state_dim):
            covariance[row][row] *= spec.initial_covariance_scale
        states[spec.name] = (mean, covariance)
        process_scales[spec.name] = 1.0

    steps: list[KalmanPosteriorStep] = []
    previous_time = trajectory.times[0]
    measurement_history_times: list[float] = []
    measurement_history_values: list[float] = []
    if velocity_measurements is not None and len(velocity_measurements) != len(trajectory.measurements):
        raise ValueError("velocity_measurements must align with trajectory.measurements")

    for step_index, (time, measurement) in enumerate(zip(trajectory.times, trajectory.measurements)):
        dt = 0.0 if step_index == 0 else time - previous_time
        previous_time = time
        measurement_history_times.append(time)
        measurement_history_values.append(measurement)
        model_log_scores = {}
        model_log_likelihoods = {}
        model_innovations = {}
        model_innovation_variances = {}
        updated_states = {}
        updated_process_scales = {}
        for spec in model_specs:
            mean, covariance = states[spec.name]
            predicted_mean, predicted_covariance = _predict(
                mean,
                covariance,
                spec,
                dt,
                process_scales[spec.name],
            )
            effective_measurement_sigma = (
                _effective_measurement_sigma(
                    predicted_covariance,
                    measurement,
                    predicted_mean,
                    spec.measurement_sigma,
                )
                if robust_measurement_update
                else spec.measurement_sigma
            )
            updated_mean, updated_covariance, innovation, innovation_variance = _update(
                predicted_mean,
                predicted_covariance,
                measurement,
                effective_measurement_sigma,
            )
            pseudo_log_likelihood = 0.0
            if derived_velocity_observation and spec.state_dim >= 2 and len(measurement_history_values) >= 3:
                window_times = measurement_history_times[-3:]
                window_values = measurement_history_values[-3:]
                velocity_observation = _least_squares_slope(window_times, window_values)
                slope_denominator = sum((time - (sum(window_times) / max(len(window_times), 1))) ** 2 for time in window_times)
                velocity_variance = max(
                    4.0 * (spec.measurement_sigma * spec.measurement_sigma) / max(slope_denominator, 1e-6),
                    0.10,
                )
                updated_mean, updated_covariance, velocity_innovation, velocity_innovation_variance = _update_scalar_measurement(
                    updated_mean,
                    updated_covariance,
                    velocity_observation,
                    velocity_variance,
                    [0.0, 1.0] + [0.0] * (len(updated_mean) - 2),
                )
                # Temper the pseudo-observation contribution because it is derived
                # from the same position measurements already used above.
                pseudo_log_likelihood += 0.5 * _innovation_log_likelihood(
                    velocity_innovation,
                    velocity_innovation_variance,
                )
            if derived_acceleration_observation and len(measurement_history_values) >= 3:
                acceleration_observation, variance_scale = _local_quadratic_acceleration(
                    measurement_history_times,
                    measurement_history_values,
                )
                acceleration_variance = max(
                    (spec.measurement_sigma * spec.measurement_sigma) * variance_scale,
                    0.25,
                )
                if spec.state_dim >= 3:
                    updated_mean, updated_covariance, acceleration_innovation, acceleration_innovation_variance = _update_scalar_measurement(
                        updated_mean,
                        updated_covariance,
                        acceleration_observation,
                        acceleration_variance,
                        [0.0, 0.0, 1.0],
                    )
                else:
                    expected_acceleration = 0.0
                    acceleration_innovation = acceleration_observation - expected_acceleration
                    acceleration_innovation_variance = acceleration_variance + max(spec.process_sigma * spec.process_sigma, 0.05)
                pseudo_log_likelihood += 0.35 * _innovation_log_likelihood(
                    acceleration_innovation,
                    acceleration_innovation_variance,
                )
            if velocity_measurements is not None and velocity_measurement_sigma is not None:
                velocity_measurement = velocity_measurements[step_index]
                velocity_variance = velocity_measurement_sigma * velocity_measurement_sigma
                if spec.state_dim >= 2:
                    updated_mean, updated_covariance, velocity_sensor_innovation, velocity_sensor_innovation_variance = _update_scalar_measurement(
                        updated_mean,
                        updated_covariance,
                        velocity_measurement,
                        velocity_variance,
                        [0.0, 1.0] + [0.0] * (len(updated_mean) - 2),
                    )
                else:
                    velocity_sensor_innovation = velocity_measurement
                    velocity_sensor_innovation_variance = velocity_variance + max(spec.process_sigma * spec.process_sigma, 0.05)
                pseudo_log_likelihood += _innovation_log_likelihood(
                    velocity_sensor_innovation,
                    velocity_sensor_innovation_variance,
                )
            log_likelihood = _innovation_log_likelihood(innovation, innovation_variance) + pseudo_log_likelihood
            model_log_scores[spec.name] = log(max(model_posterior[spec.name], 1e-12)) + log_likelihood
            model_log_likelihoods[spec.name] = log_likelihood
            model_innovations[spec.name] = innovation
            model_innovation_variances[spec.name] = innovation_variance
            updated_states[spec.name] = (updated_mean, updated_covariance)
            updated_process_scales[spec.name] = (
                _next_process_scale(
                    previous_scale=process_scales[spec.name],
                    innovation=innovation,
                    innovation_variance=innovation_variance,
                )
                if adaptive_process_noise
                else 1.0
            )
        model_posterior = _normalize_log_scores(model_log_scores)
        class_log_scores = {
            class_name: _logsumexp([model_log_scores[spec.name] for spec in model_specs if spec.class_name == class_name])
            for class_name in class_names
        }
        posterior = _normalize_log_scores(class_log_scores)
        log_likelihood_terms = {}
        innovations = {}
        innovation_variances = {}
        for class_name in class_names:
            members = [spec for spec in model_specs if spec.class_name == class_name]
            member_weights = [model_posterior[spec.name] for spec in members]
            weight_total = sum(member_weights)
            normalized_weights = [weight / max(weight_total, 1e-12) for weight in member_weights]
            log_likelihood_terms[class_name] = _logsumexp([model_log_likelihoods[spec.name] for spec in members])
            innovations[class_name] = sum(
                normalized_weights[index] * model_innovations[spec.name]
                for index, spec in enumerate(members)
            )
            innovation_variances[class_name] = sum(
                normalized_weights[index] * model_innovation_variances[spec.name]
                for index, spec in enumerate(members)
            )
        predicted_class = max(posterior, key=posterior.get)
        steps.append(
            KalmanPosteriorStep(
                time=time,
                measurement=measurement,
                posterior_weights=posterior,
                log_likelihood_terms=log_likelihood_terms,
                innovations=innovations,
                innovation_variances=innovation_variances,
                predicted_class=predicted_class,
                confidence=posterior[predicted_class],
            )
        )
        states = updated_states
        process_scales = updated_process_scales

    final_states = {name: _pad_state(mean, covariance) for name, (mean, covariance) in states.items()}
    final_class = max(posterior, key=posterior.get)
    return KalmanClassificationRun(
        trajectory_id=trajectory.trajectory_id,
        true_class=trajectory.true_class,
        scenario_name=trajectory.scenario_name,
        steps=tuple(steps),
        final_weights=posterior,
        final_predicted_class=final_class,
        final_confidence=posterior[final_class],
        final_states=final_states,
    )


def _summarize_runs(runs: tuple[KalmanClassificationRun, ...], class_names: tuple[str, ...]) -> KalmanBenchmarkSummary:
    confusion = {name: {predicted: 0 for predicted in class_names} for name in class_names}
    correct = {name: 0 for name in class_names}
    totals = {name: 0 for name in class_names}
    for run in runs:
        confusion[run.true_class][run.final_predicted_class] += 1
        totals[run.true_class] += 1
        if run.final_predicted_class == run.true_class:
            correct[run.true_class] += 1
    return KalmanBenchmarkSummary(
        total_trajectories=len(runs),
        final_accuracy=sum(correct.values()) / max(len(runs), 1),
        per_class_accuracy={name: correct[name] / totals[name] if totals[name] else 0.0 for name in class_names},
        confusion_counts=confusion,
    )


def run_kalman_bank_benchmark(
    *,
    seed: int = 7,
    trajectories_per_class: int = 4,
    model_specs: tuple[KalmanModelSpec, ...] | None = None,
) -> KalmanBenchmarkResult:
    specs = model_specs or default_kalman_model_specs()
    trajectories = generate_kalman_bank_trajectories(seed=seed, trajectories_per_class=trajectories_per_class)
    runs = tuple(run_kalman_filter_bank(trajectory, specs) for trajectory in trajectories)
    summary = _summarize_runs(runs, _unique_class_names(specs))
    return KalmanBenchmarkResult(
        model_specs=specs,
        trajectories=trajectories,
        runs=runs,
        summary=summary,
    )


def render_kalman_bank_report(result: KalmanBenchmarkResult) -> str:
    report = MarkdownDocument("Kalman Filter Bank")
    report.paragraph(
        "This benchmark runs one linear-Gaussian motion model per class, scores scalar position innovations, and updates "
        "class posterior weights recursively over time. The measurement update uses innovation-based variance inflation "
        "so isolated surprising measurements do not dominate the state or class posterior as aggressively as a plain "
        "Kalman update. Process noise is also adapted over time from recent innovation energy so the filter can "
        "temporarily loosen its motion assumptions on noisy segments."
    )
    report.heading("Summary", level=2)
    report.bullet_list(
        [
            f"Trajectories: {result.summary.total_trajectories}",
            f"Final accuracy: {result.summary.final_accuracy:.3f}",
        ]
    )
    report.heading("Per-Class Accuracy", level=2)
    report.bullet_list([f"{report.inline_code(class_name)}: {value:.3f}" for class_name, value in result.summary.per_class_accuracy.items()])
    report.heading("Acceptance Notes", level=2)
    report.bullet_list(
        [
            "Constant-velocity tracks should favor the constant-velocity model over the stationary model.",
            "Constant-acceleration tracks should move toward the constant-acceleration model after enough measurements.",
            "Irregular sampling should still produce stable class decisions because each transition uses the observed `dt`.",
            "Large normalized innovations should be downweighted by adaptive measurement variance inflation instead of being trusted at face value.",
            "Repeated elevated innovation energy should increase effective process noise for subsequent prediction steps.",
        ]
    )
    return report.text()


def _build_kalman_bank_figure(result: KalmanBenchmarkResult):
    selected = (
        next(run for run in result.runs if run.scenario_name == "constant_velocity_regular"),
        next(run for run in result.runs if run.scenario_name == "constant_velocity_irregular"),
        next(run for run in result.runs if run.scenario_name == "constant_acceleration_regular"),
    )
    trajectory_lookup = {trajectory.trajectory_id: trajectory for trajectory in result.trajectories}
    colors = {spec.name: color for spec, color in zip(result.model_specs, ("#2563eb", "#16a34a", "#d97706"))}
    fig, axes = plt.subplots(3, 2, figsize=(13.0, 10.0), sharex=False)
    for row_axes, run in zip(axes, selected):
        measurement_ax, posterior_ax = row_axes
        trajectory = trajectory_lookup[run.trajectory_id]
        steps = list(range(len(run.steps)))
        measurement_ax.plot(trajectory.times, trajectory.measurements, color="#111827", linewidth=2.0, marker="o", label="measurement")
        measurement_ax.plot(trajectory.times, trajectory.true_position, color="#9ca3af", linewidth=1.6, linestyle="--", label="true position")
        measurement_ax.set_title(run.scenario_name, loc="left", fontsize=12, fontweight="bold")
        measurement_ax.grid(True, alpha=0.25)
        measurement_ax.legend(frameon=False, fontsize=8)
        for spec in result.model_specs:
            posterior_ax.plot(steps, [step.posterior_weights[spec.name] for step in run.steps], color=colors[spec.name], linewidth=2.1, label=spec.name)
        posterior_ax.set_ylim(0.0, 1.0)
        posterior_ax.grid(True, alpha=0.25)
        posterior_ax.legend(frameon=False, fontsize=8)
        posterior_ax.set_ylabel("posterior")
        measurement_ax.set_ylabel("position")
        posterior_ax.set_xlabel("step")
    fig.suptitle("Kalman Filter Bank Diagnostics", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def render_kalman_bank_svg(result: KalmanBenchmarkResult) -> str:
    fig = _build_kalman_bank_figure(result)
    buffer = io.StringIO()
    try:
        fig.savefig(buffer, format="svg", bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def render_kalman_bank_png_bytes(result: KalmanBenchmarkResult) -> bytes:
    fig = _build_kalman_bank_figure(result)
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def write_kalman_bank_artifacts(
    output_dir: str | Path,
    *,
    seed: int = 7,
    result: KalmanBenchmarkResult | None = None,
) -> KalmanBenchmarkArtifacts:
    benchmark = result or run_kalman_bank_benchmark(seed=seed)
    output_root = Path(output_dir)
    run_dir = output_root / "kalman_filter_bank"
    run_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_dir / "kalman_bank_report.md"
    innovation_history_path = run_dir / "innovation_history.csv"
    state_estimate_history_path = run_dir / "state_estimate_history.csv"
    posterior_history_path = run_dir / "posterior_history.csv"
    confusion_matrix_path = run_dir / "confusion_final.csv"
    config_path = run_dir / "kalman_bank_config.yaml"
    dataset_manifest_path = run_dir / "dataset_manifest.json"
    model_definitions_path = run_dir / "kalman_model_definitions.json"
    plot_png_path = run_dir / "kalman_bank_diagnostics.png"

    report_path.write_text(render_kalman_bank_report(benchmark), encoding="utf-8")
    plot_png_path.write_bytes(render_kalman_bank_png_bytes(benchmark))

    innovation_rows: list[dict[str, object]] = []
    posterior_rows: list[dict[str, object]] = []
    state_rows: list[dict[str, object]] = []
    class_names = list(_unique_class_names(benchmark.model_specs))
    for run in benchmark.runs:
        for step_index, step in enumerate(run.steps):
            posterior_rows.append(
                {
                    "trajectory_id": run.trajectory_id,
                    "scenario_name": run.scenario_name,
                    "step": step_index,
                    "time": step.time,
                    "measurement": step.measurement,
                    "true_class": run.true_class,
                    "predicted_class": step.predicted_class,
                    "confidence": step.confidence,
                    **{f"posterior_{name}": step.posterior_weights[name] for name in class_names},
                    **{f"log_likelihood_{name}": step.log_likelihood_terms[name] for name in class_names},
                }
            )
            innovation_rows.append(
                {
                    "trajectory_id": run.trajectory_id,
                    "scenario_name": run.scenario_name,
                    "step": step_index,
                    "time": step.time,
                    **{f"innovation_{name}": step.innovations[name] for name in class_names},
                    **{f"innovation_variance_{name}": step.innovation_variances[name] for name in class_names},
                }
            )
        for model_name, state in run.final_states.items():
            state_rows.append(
                {
                    "trajectory_id": run.trajectory_id,
                    "scenario_name": run.scenario_name,
                    "true_class": run.true_class,
                    "model_name": model_name,
                    "position": state.mean[0],
                    "velocity": state.mean[1],
                    "acceleration": state.mean[2],
                    "position_variance": state.covariance[0][0],
                    "velocity_variance": state.covariance[1][1],
                    "acceleration_variance": state.covariance[2][2],
                }
            )

    write_csv(
        innovation_history_path,
        innovation_rows,
        ["trajectory_id", "scenario_name", "step", "time", *[f"innovation_{name}" for name in class_names], *[f"innovation_variance_{name}" for name in class_names]],
    )
    write_csv(
        posterior_history_path,
        posterior_rows,
        ["trajectory_id", "scenario_name", "step", "time", "measurement", "true_class", "predicted_class", "confidence", *[f"posterior_{name}" for name in class_names], *[f"log_likelihood_{name}" for name in class_names]],
    )
    write_csv(
        state_estimate_history_path,
        state_rows,
        ["trajectory_id", "scenario_name", "true_class", "model_name", "position", "velocity", "acceleration", "position_variance", "velocity_variance", "acceleration_variance"],
    )
    write_csv(
        confusion_matrix_path,
        [{"true_class": true_name, **predicted_counts} for true_name, predicted_counts in benchmark.summary.confusion_counts.items()],
        ["true_class", *class_names],
    )

    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  name: kalman_filter_bank",
                f"  seed: {seed}",
                "classifier:",
                "  type: kalman_filter_bank",
                "  models: [stationary, constant_velocity, constant_acceleration]",
                "  robust_measurement_update:",
                "    enabled: true",
                "    innovation_gate_nis: 9.0",
                "  adaptive_process_noise:",
                "    enabled: true",
                "    trigger_nis: 2.0",
                "    smoothing: 0.10",
                "    scale_range: [1.0, 2.5]",
                "dataset:",
                "  scenarios: [stationary_regular, constant_velocity_regular, constant_velocity_irregular, constant_acceleration_regular]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    dataset_manifest_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "trajectory_count": benchmark.summary.total_trajectories,
                "scenario_names": [trajectory.scenario_name for trajectory in benchmark.trajectories],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    model_definitions_path.write_text(
        json.dumps(
            {
                spec.name: {
                    "class_name": spec.class_name,
                    "state_dim": spec.state_dim,
                    "process_sigma": spec.process_sigma,
                    "measurement_sigma": spec.measurement_sigma,
                    "initial_covariance_scale": spec.initial_covariance_scale,
                }
                for spec in benchmark.model_specs
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return KalmanBenchmarkArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        innovation_history_path=innovation_history_path,
        state_estimate_history_path=state_estimate_history_path,
        posterior_history_path=posterior_history_path,
        confusion_matrix_path=confusion_matrix_path,
        config_path=config_path,
        dataset_manifest_path=dataset_manifest_path,
        model_definitions_path=model_definitions_path,
        plot_png_path=plot_png_path,
    )
