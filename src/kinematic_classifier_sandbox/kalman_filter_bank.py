from __future__ import annotations

from dataclasses import dataclass, asdict
from math import exp, log, pi
import csv
import io
import json
import os
from pathlib import Path
import random


def _prepare_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp/kinematic-classifier-sandbox-mpl")))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _logsumexp(values: list[float]) -> float:
    pivot = max(values)
    return pivot + log(sum(exp(value - pivot) for value in values))


def _normalize_log_scores(log_scores: dict[str, float]) -> dict[str, float]:
    log_norm = _logsumexp(list(log_scores.values()))
    return {name: exp(value - log_norm) for name, value in log_scores.items()}


def _identity(size: int) -> list[list[float]]:
    return [[1.0 if row == col else 0.0 for col in range(size)] for row in range(size)]


def _transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [list(column) for column in zip(*matrix)]


def _matmul(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    rows = len(left)
    cols = len(right[0])
    inner = len(right)
    return [
        [sum(left[row][index] * right[index][col] for index in range(inner)) for col in range(cols)]
        for row in range(rows)
    ]


def _matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(row[index] * vector[index] for index in range(len(vector))) for row in matrix]


def _add_matrices(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [[left[row][col] + right[row][col] for col in range(len(left[row]))] for row in range(len(left))]


def _subtract_matrices(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [[left[row][col] - right[row][col] for col in range(len(left[row]))] for row in range(len(left))]


def _outer(left: list[float], right: list[float]) -> list[list[float]]:
    return [[left[row] * right[col] for col in range(len(right))] for row in range(len(left))]


def _innovation_log_likelihood(innovation: float, variance: float) -> float:
    safe_variance = max(variance, 1e-9)
    return -0.5 * (log(2.0 * pi * safe_variance) + (innovation * innovation) / safe_variance)


@dataclass(frozen=True, slots=True)
class KalmanModelSpec:
    name: str
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
    plot_svg_path: Path
    plot_png_path: Path


def default_kalman_model_specs() -> tuple[KalmanModelSpec, ...]:
    return (
        KalmanModelSpec(
            name="stationary",
            state_dim=1,
            process_sigma=0.04,
            measurement_sigma=0.20,
            initial_covariance_scale=4.0,
            prior_weight=1.0 / 3.0,
        ),
        KalmanModelSpec(
            name="constant_velocity",
            state_dim=2,
            process_sigma=0.14,
            measurement_sigma=0.20,
            initial_covariance_scale=5.0,
            prior_weight=1.0 / 3.0,
        ),
        KalmanModelSpec(
            name="constant_acceleration",
            state_dim=3,
            process_sigma=0.24,
            measurement_sigma=0.20,
            initial_covariance_scale=6.0,
            prior_weight=1.0 / 3.0,
        ),
    )


def _transition_and_noise(model: KalmanModelSpec, dt: float) -> tuple[list[list[float]], list[list[float]]]:
    safe_dt = max(dt, 1e-6)
    q = model.process_sigma * model.process_sigma
    if model.state_dim == 1:
        return [[1.0]], [[q * safe_dt]]
    if model.state_dim == 2:
        f = [[1.0, safe_dt], [0.0, 1.0]]
        q_matrix = [
            [q * (safe_dt ** 4) / 4.0, q * (safe_dt ** 3) / 2.0],
            [q * (safe_dt ** 3) / 2.0, q * (safe_dt ** 2)],
        ]
        return f, q_matrix
    f = [
        [1.0, safe_dt, 0.5 * safe_dt * safe_dt],
        [0.0, 1.0, safe_dt],
        [0.0, 0.0, 1.0],
    ]
    q_matrix = [
        [q * (safe_dt ** 5) / 20.0, q * (safe_dt ** 4) / 8.0, q * (safe_dt ** 3) / 6.0],
        [q * (safe_dt ** 4) / 8.0, q * (safe_dt ** 3) / 3.0, q * (safe_dt ** 2) / 2.0],
        [q * (safe_dt ** 3) / 6.0, q * (safe_dt ** 2) / 2.0, q * safe_dt],
    ]
    return f, q_matrix


def _predict(mean: list[float], covariance: list[list[float]], model: KalmanModelSpec, dt: float) -> tuple[list[float], list[list[float]]]:
    transition, process_noise = _transition_and_noise(model, dt)
    predicted_mean = _matvec(transition, mean)
    predicted_covariance = _add_matrices(_matmul(_matmul(transition, covariance), _transpose(transition)), process_noise)
    return predicted_mean, predicted_covariance


def _update(
    predicted_mean: list[float],
    predicted_covariance: list[list[float]],
    measurement: float,
    measurement_sigma: float,
) -> tuple[list[float], list[list[float]], float, float]:
    innovation = measurement - predicted_mean[0]
    innovation_variance = predicted_covariance[0][0] + measurement_sigma * measurement_sigma
    gain = [predicted_covariance[row][0] / innovation_variance for row in range(len(predicted_mean))]
    updated_mean = [predicted_mean[index] + gain[index] * innovation for index in range(len(predicted_mean))]
    h = [[1.0] + [0.0] * (len(predicted_mean) - 1)]
    i = _identity(len(predicted_mean))
    kh = _outer(gain, h[0])
    updated_covariance = _matmul(_subtract_matrices(i, kh), predicted_covariance)
    return updated_mean, updated_covariance, innovation, innovation_variance


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
) -> KalmanClassificationRun:
    total_prior = sum(spec.prior_weight for spec in model_specs)
    posterior = prior or {spec.name: spec.prior_weight / total_prior for spec in model_specs}
    states = {}
    for spec in model_specs:
        mean = [trajectory.measurements[0]] + [0.0] * (spec.state_dim - 1)
        covariance = _identity(spec.state_dim)
        for row in range(spec.state_dim):
            covariance[row][row] *= spec.initial_covariance_scale
        states[spec.name] = (mean, covariance)

    steps: list[KalmanPosteriorStep] = []
    previous_time = trajectory.times[0]
    for step_index, (time, measurement) in enumerate(zip(trajectory.times, trajectory.measurements)):
        dt = 0.0 if step_index == 0 else time - previous_time
        previous_time = time
        log_scores = {}
        log_likelihood_terms = {}
        innovations = {}
        innovation_variances = {}
        updated_states = {}
        for spec in model_specs:
            mean, covariance = states[spec.name]
            predicted_mean, predicted_covariance = _predict(mean, covariance, spec, dt)
            updated_mean, updated_covariance, innovation, innovation_variance = _update(
                predicted_mean,
                predicted_covariance,
                measurement,
                spec.measurement_sigma,
            )
            log_likelihood = _innovation_log_likelihood(innovation, innovation_variance)
            log_scores[spec.name] = log(max(posterior[spec.name], 1e-12)) + log_likelihood
            log_likelihood_terms[spec.name] = log_likelihood
            innovations[spec.name] = innovation
            innovation_variances[spec.name] = innovation_variance
            updated_states[spec.name] = (updated_mean, updated_covariance)
        posterior = _normalize_log_scores(log_scores)
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
    summary = _summarize_runs(runs, tuple(spec.name for spec in specs))
    return KalmanBenchmarkResult(
        model_specs=specs,
        trajectories=trajectories,
        runs=runs,
        summary=summary,
    )


def render_kalman_bank_report(result: KalmanBenchmarkResult) -> str:
    lines = [
        "# Kalman Filter Bank",
        "",
        "This benchmark runs one linear-Gaussian motion model per class, scores scalar position innovations, and updates class posterior weights recursively over time.",
        "",
        "## Summary",
        "",
        f"- Trajectories: {result.summary.total_trajectories}",
        f"- Final accuracy: {result.summary.final_accuracy:.3f}",
        "",
        "## Per-Class Accuracy",
        "",
    ]
    for class_name, value in result.summary.per_class_accuracy.items():
        lines.append(f"- `{class_name}`: {value:.3f}")
    lines.extend(
        [
            "",
            "## Acceptance Notes",
            "",
            "- Constant-velocity tracks should favor the constant-velocity model over the stationary model.",
            "- Constant-acceleration tracks should move toward the constant-acceleration model after enough measurements.",
            "- Irregular sampling should still produce stable class decisions because each transition uses the observed `dt`.",
        ]
    )
    return "\n".join(lines)


def _build_kalman_bank_figure(result: KalmanBenchmarkResult):
    plt = _prepare_matplotlib()
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
    plt = _prepare_matplotlib()
    fig = _build_kalman_bank_figure(result)
    buffer = io.StringIO()
    try:
        fig.savefig(buffer, format="svg", bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def render_kalman_bank_png_bytes(result: KalmanBenchmarkResult) -> bytes:
    plt = _prepare_matplotlib()
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
    plot_svg_path = run_dir / "kalman_bank_diagnostics.svg"
    plot_png_path = run_dir / "kalman_bank_diagnostics.png"

    report_path.write_text(render_kalman_bank_report(benchmark), encoding="utf-8")
    plot_svg_path.write_text(render_kalman_bank_svg(benchmark), encoding="utf-8")
    plot_png_path.write_bytes(render_kalman_bank_png_bytes(benchmark))

    innovation_rows: list[dict[str, object]] = []
    posterior_rows: list[dict[str, object]] = []
    state_rows: list[dict[str, object]] = []
    class_names = [spec.name for spec in benchmark.model_specs]
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

    _write_csv(
        innovation_history_path,
        innovation_rows,
        ["trajectory_id", "scenario_name", "step", "time", *[f"innovation_{name}" for name in class_names], *[f"innovation_variance_{name}" for name in class_names]],
    )
    _write_csv(
        posterior_history_path,
        posterior_rows,
        ["trajectory_id", "scenario_name", "step", "time", "measurement", "true_class", "predicted_class", "confidence", *[f"posterior_{name}" for name in class_names], *[f"log_likelihood_{name}" for name in class_names]],
    )
    _write_csv(
        state_estimate_history_path,
        state_rows,
        ["trajectory_id", "scenario_name", "true_class", "model_name", "position", "velocity", "acceleration", "position_variance", "velocity_variance", "acceleration_variance"],
    )
    _write_csv(
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
        plot_svg_path=plot_svg_path,
        plot_png_path=plot_png_path,
    )
