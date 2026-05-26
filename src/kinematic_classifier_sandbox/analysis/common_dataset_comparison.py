from __future__ import annotations
from kinematic_classifier_sandbox.utils.io import write_csv

from ..runtime_paths import prepare_matplotlib
from ..markdown_builder import MarkdownDocument
from ..utils.math import (
    gaussian_logpdf as _gaussian_logpdf,
    logsumexp as _logsumexp,
    normalize_log_scores as _normalize_log_scores,
    normalize_prior as _lib_normalize_prior,
    _median3 as _median3,
)
from dataclasses import dataclass
import csv
from math import exp, log, pi
import io
import json
import os
from pathlib import Path
import random

from ..inference.kalman_filter_bank import KalmanModelSpec, KalmanTrajectory as SharedKalmanTrajectory, run_kalman_filter_bank
from ..validation.shared_evaluation import (
    CallableSharedClassifierAdapter,
    SharedClassifierRun,
    evaluate_shared_classifier_registry,
    sensor_regime_summary_rows,
)
from ..utils.plotting import _figure_to_png
from ..scenarios import (
    SCENARIO_MEASUREMENT_SIGMA,
    SCENARIO_TIMES,
    get_scenario_dynamics as _scenario_dynamics,
)



def _normalize_prior(prior: dict[str, float] | None) -> dict[str, float]:
    return _lib_normalize_prior(prior, CLASS_NAMES)


@dataclass(frozen=True, slots=True)
class SharedDynamicsTrajectory:
    trajectory_id: str
    true_class: str
    scenario_name: str
    seed: int
    times: tuple[float, ...]
    measurements: tuple[float, ...]
    true_position: tuple[float, ...]
    true_velocity: tuple[float, ...]
    true_acceleration: tuple[float, ...]
    measurement_dim: int = 1
    coordinate_frame: str = "scalar_line"


CommonMethodRun = SharedClassifierRun


@dataclass(frozen=True, slots=True)
class CommonComparisonRow:
    method_name: str
    overall_accuracy: float
    easy_accuracy: float
    irregular_accuracy: float
    endpoint_match_accuracy: float
    short_accuracy: float
    noisy_accuracy: float
    outlier_accuracy: float
    prior_flip_fraction: float


@dataclass(frozen=True, slots=True)
class CommonComparisonResult:
    trajectories: tuple[SharedDynamicsTrajectory, ...]
    runs: tuple[CommonMethodRun, ...]
    rows: tuple[CommonComparisonRow, ...]


@dataclass(frozen=True, slots=True)
class CommonComparisonArtifacts:
    run_dir: Path
    report_path: Path
    trajectory_path: Path
    run_summary_path: Path
    method_summary_path: Path
    sensor_regimes_path: Path
    sensor_regime_metrics_path: Path
    heatmap_png_path: Path
    confusion_png_path: Path
    plots_dir: Path
    overview_balance_png_path: Path
    overview_covariates_png_path: Path
    scenario_profile_png_path: Path
    prior_sensitivity_png_path: Path
    trajectory_examples_png_path: Path
    final_confusion_png_path: Path
CLASS_NAMES = ("constant_velocity", "constant_acceleration")


def _make_shared_trajectory(
    *,
    trajectory_id: str,
    true_class: str,
    scenario_name: str,
    seed: int,
) -> SharedDynamicsTrajectory:
    rng = random.Random(seed)
    times = SCENARIO_TIMES[scenario_name]
    position0 = 0.0
    velocity0, acceleration = _scenario_dynamics(scenario_name, true_class)
    true_position = tuple(position0 + velocity0 * time + 0.5 * acceleration * time * time for time in times)
    true_velocity = tuple(velocity0 + acceleration * time for time in times)
    true_acceleration = tuple(acceleration for _ in times)
    measurement_sigma = SCENARIO_MEASUREMENT_SIGMA[scenario_name]
    measurements = tuple(value + rng.gauss(0.0, measurement_sigma) for value in true_position)
    if scenario_name == "outlier":
        outlier_index = len(measurements) // 2
        rebound_index = min(outlier_index + 1, len(measurements) - 1)
        glitch = 2.2
        measurements = tuple(
            value
            + (-glitch if index == outlier_index else 0.0)
            + (glitch if index == rebound_index else 0.0)
            for index, value in enumerate(measurements)
        )
    return SharedDynamicsTrajectory(
        trajectory_id=trajectory_id,
        true_class=true_class,
        scenario_name=scenario_name,
        seed=seed,
        times=times,
        measurements=measurements,
        true_position=true_position,
        true_velocity=true_velocity,
        true_acceleration=true_acceleration,
    )


def generate_shared_dynamics_dataset(*, seed: int = 7, trajectories_per_case: int = 8) -> tuple[SharedDynamicsTrajectory, ...]:
    trajectories: list[SharedDynamicsTrajectory] = []
    scenario_names = ("easy", "irregular", "endpoint_match", "short", "short_noisy", "outlier")
    for scenario_index, scenario_name in enumerate(scenario_names):
        for class_index, class_name in enumerate(CLASS_NAMES):
            for example_index in range(trajectories_per_case):
                trajectories.append(
                    _make_shared_trajectory(
                        trajectory_id=f"{scenario_name}_{class_name}_{example_index}",
                        true_class=class_name,
                        scenario_name=scenario_name,
                        seed=seed + scenario_index * 100 + class_index * 20 + example_index,
                    )
                )
    return tuple(trajectories)


def _class_expected_position(class_name: str, time: float, scenario_name: str) -> float:
    velocity0, acceleration = _scenario_dynamics(scenario_name, class_name)
    return velocity0 * time + 0.5 * acceleration * time * time


def _pointwise_predict(
    trajectory: SharedDynamicsTrajectory,
    *,
    prior: dict[str, float] | None = None,
) -> CommonMethodRun:
    last_measurement = trajectory.measurements[-1]
    last_time = trajectory.times[-1]
    prior_weights = _normalize_prior(prior)
    log_scores = {
        class_name: log(max(prior_weights[class_name], 1e-12))
        + _gaussian_logpdf(last_measurement, _class_expected_position(class_name, last_time, trajectory.scenario_name), 0.35 ** 2)
        for class_name in CLASS_NAMES
    }
    weights = _normalize_log_scores(log_scores)
    predicted = max(weights, key=weights.get)
    return CommonMethodRun(
        method_name="pointwise",
        sensor_regime_id="position_only",
        trajectory_id=trajectory.trajectory_id,
        true_class=trajectory.true_class,
        scenario_name=trajectory.scenario_name,
        final_predicted_class=predicted,
        final_confidence=weights[predicted],
        final_weights=weights,
        measurement_dim=trajectory.measurement_dim,
        coordinate_frame=trajectory.coordinate_frame,
    )


def _accumulator_predict(
    trajectory: SharedDynamicsTrajectory,
    *,
    prior: dict[str, float] | None = None,
) -> CommonMethodRun:
    posterior = _normalize_prior(prior)
    for time, measurement in zip(trajectory.times, trajectory.measurements):
        log_scores = {
            class_name: log(max(posterior[class_name], 1e-12))
            + _gaussian_logpdf(measurement, _class_expected_position(class_name, time, trajectory.scenario_name), 0.25 ** 2)
            for class_name in CLASS_NAMES
        }
        posterior = _normalize_log_scores(log_scores)
    predicted = max(posterior, key=posterior.get)
    return CommonMethodRun(
        method_name="accumulator",
        sensor_regime_id="position_only",
        trajectory_id=trajectory.trajectory_id,
        true_class=trajectory.true_class,
        scenario_name=trajectory.scenario_name,
        final_predicted_class=predicted,
        final_confidence=posterior[predicted],
        final_weights=posterior,
        measurement_dim=trajectory.measurement_dim,
        coordinate_frame=trajectory.coordinate_frame,
    )


def _window_features(trajectory: SharedDynamicsTrajectory, *, robust: bool) -> dict[str, float]:
    times = list(trajectory.times)
    values = list(trajectory.measurements)
    if robust and len(values) >= 3:
        values = [_median3(values, index) for index in range(len(values))]
    n = len(values)
    mean_t = sum(times) / n
    mean_y = sum(values) / n
    denominator = sum((time - mean_t) ** 2 for time in times)
    slope = 0.0 if denominator <= 1e-9 else sum((time - mean_t) * (value - mean_y) for time, value in zip(times, values)) / denominator
    quadratic_proxy = 0.0
    if len(times) >= 3:
        midpoint = len(times) // 2
        quadratic_proxy = values[-1] - 2.0 * values[midpoint] + values[0]
    return {"slope": slope, "quadratic_proxy": quadratic_proxy}


def _windowed_predict(
    trajectory: SharedDynamicsTrajectory,
    *,
    robust: bool,
    prior: dict[str, float] | None = None,
) -> CommonMethodRun:
    features = _window_features(trajectory, robust=robust)
    prior_weights = _normalize_prior(prior)
    if robust:
        specs = {
            "constant_velocity": {"slope": 0.8, "quadratic_proxy": 0.0, "sigma_slope": 0.12, "sigma_quad": 2.80},
            "constant_acceleration": {"slope": 2.2, "quadratic_proxy": 6.5, "sigma_slope": 0.18, "sigma_quad": 3.60},
        }
        method_name = "windowed_robust"
    else:
        specs = {
            "constant_velocity": {"slope": 0.8, "quadratic_proxy": 0.0, "sigma_slope": 0.22, "sigma_quad": 0.90},
            "constant_acceleration": {"slope": 2.2, "quadratic_proxy": 6.5, "sigma_slope": 0.40, "sigma_quad": 1.20},
        }
        method_name = "windowed_raw"
    log_scores = {}
    for class_name in CLASS_NAMES:
        spec = specs[class_name]
        log_scores[class_name] = (
            log(max(prior_weights[class_name], 1e-12))
            +
            _gaussian_logpdf(features["slope"], spec["slope"], spec["sigma_slope"] ** 2)
            + _gaussian_logpdf(features["quadratic_proxy"], spec["quadratic_proxy"], spec["sigma_quad"] ** 2)
        )
    weights = _normalize_log_scores(log_scores)
    predicted = max(weights, key=weights.get)
    return CommonMethodRun(
        method_name=method_name,
        sensor_regime_id="position_only",
        trajectory_id=trajectory.trajectory_id,
        true_class=trajectory.true_class,
        scenario_name=trajectory.scenario_name,
        final_predicted_class=predicted,
        final_confidence=weights[predicted],
        final_weights=weights,
        measurement_dim=trajectory.measurement_dim,
        coordinate_frame=trajectory.coordinate_frame,
    )


def _kalman_predict(
    trajectory: SharedDynamicsTrajectory,
    *,
    prior: dict[str, float] | None = None,
) -> CommonMethodRun:
    prior_weights = _normalize_prior(prior)
    model_specs = _shared_kalman_model_specs(prior_weights)
    kalman_trajectory = _shared_kalman_trajectory(trajectory)
    run = run_kalman_filter_bank(
        kalman_trajectory,
        model_specs,
        derived_velocity_observation=True,
        derived_acceleration_observation=True,
    )
    return CommonMethodRun(
        method_name="kalman_bank",
        sensor_regime_id="position_only",
        trajectory_id=trajectory.trajectory_id,
        true_class=trajectory.true_class,
        scenario_name=trajectory.scenario_name,
        final_predicted_class=run.final_predicted_class,
        final_confidence=run.final_confidence,
        final_weights=run.final_weights,
        measurement_dim=trajectory.measurement_dim,
        coordinate_frame=trajectory.coordinate_frame,
    )


def _shared_kalman_model_specs(prior_weights: dict[str, float]) -> tuple[KalmanModelSpec, ...]:
    return (
        KalmanModelSpec(
            name="constant_velocity_quiet",
            class_name="constant_velocity",
            state_dim=2,
            process_sigma=0.14,
            measurement_sigma=0.20,
            initial_covariance_scale=5.0,
            prior_weight=0.75 * prior_weights["constant_velocity"],
        ),
        KalmanModelSpec(
            name="constant_velocity_rough",
            class_name="constant_velocity",
            state_dim=2,
            process_sigma=0.22,
            measurement_sigma=0.20,
            initial_covariance_scale=5.5,
            prior_weight=0.25 * prior_weights["constant_velocity"],
        ),
        KalmanModelSpec(
            name="constant_acceleration_quiet",
            class_name="constant_acceleration",
            state_dim=3,
            process_sigma=0.24,
            measurement_sigma=0.20,
            initial_covariance_scale=6.0,
            prior_weight=0.75 * prior_weights["constant_acceleration"],
        ),
        KalmanModelSpec(
            name="constant_acceleration_rough",
            class_name="constant_acceleration",
            state_dim=3,
            process_sigma=0.34,
            measurement_sigma=0.20,
            initial_covariance_scale=6.5,
            prior_weight=0.25 * prior_weights["constant_acceleration"],
        ),
    )


def _shared_kalman_trajectory(trajectory: SharedDynamicsTrajectory):
    return SharedKalmanTrajectory(
        trajectory_id=trajectory.trajectory_id,
        true_class=trajectory.true_class,
        scenario_name=trajectory.scenario_name,
        seed=trajectory.seed,
        times=trajectory.times,
        measurements=trajectory.measurements,
        true_position=trajectory.true_position,
        true_velocity=trajectory.true_velocity,
        true_acceleration=trajectory.true_acceleration,
    )


def _synthesized_velocity_measurements(
    trajectory: SharedDynamicsTrajectory,
    *,
    sigma: float = 0.12,
) -> tuple[float, ...]:
    rng = random.Random(trajectory.seed + 9001)
    return tuple(value + rng.gauss(0.0, sigma) for value in trajectory.true_velocity)


def _kalman_velocity_aided_predict(
    trajectory: SharedDynamicsTrajectory,
    *,
    prior: dict[str, float] | None = None,
) -> CommonMethodRun:
    prior_weights = _normalize_prior(prior)
    model_specs = _shared_kalman_model_specs(prior_weights)
    kalman_trajectory = _shared_kalman_trajectory(trajectory)
    velocity_sigma = 0.12
    run = run_kalman_filter_bank(
        kalman_trajectory,
        model_specs,
        velocity_measurements=_synthesized_velocity_measurements(trajectory, sigma=velocity_sigma),
        velocity_measurement_sigma=velocity_sigma,
    )
    return CommonMethodRun(
        method_name="kalman_bank_velocity_aided",
        sensor_regime_id="position_plus_direct_velocity",
        trajectory_id=trajectory.trajectory_id,
        true_class=trajectory.true_class,
        scenario_name=trajectory.scenario_name,
        final_predicted_class=run.final_predicted_class,
        final_confidence=run.final_confidence,
        final_weights=run.final_weights,
        measurement_dim=trajectory.measurement_dim,
        coordinate_frame=trajectory.coordinate_frame,
    )


def _predict_with_method(
    method_name: str,
    trajectory: SharedDynamicsTrajectory,
    *,
    prior: dict[str, float] | None = None,
) -> CommonMethodRun:
    if method_name == "pointwise":
        return _pointwise_predict(trajectory, prior=prior)
    if method_name == "windowed_raw":
        return _windowed_predict(trajectory, robust=False, prior=prior)
    if method_name == "windowed_robust":
        return _windowed_predict(trajectory, robust=True, prior=prior)
    if method_name == "accumulator":
        return _accumulator_predict(trajectory, prior=prior)
    if method_name == "kalman_bank":
        return _kalman_predict(trajectory, prior=prior)
    if method_name == "kalman_bank_velocity_aided":
        return _kalman_velocity_aided_predict(trajectory, prior=prior)
    raise ValueError(f"Unsupported method: {method_name}")


def default_shared_classifier_adapters() -> tuple[CallableSharedClassifierAdapter, ...]:
    return (
        CallableSharedClassifierAdapter(
            method_name="pointwise",
            sensor_regime_id="position_only",
            predict_fn=lambda trajectory, prior=None: _pointwise_predict(trajectory, prior=prior),
        ),
        CallableSharedClassifierAdapter(
            method_name="windowed_raw",
            sensor_regime_id="position_only",
            predict_fn=lambda trajectory, prior=None: _windowed_predict(trajectory, robust=False, prior=prior),
        ),
        CallableSharedClassifierAdapter(
            method_name="windowed_robust",
            sensor_regime_id="position_only",
            predict_fn=lambda trajectory, prior=None: _windowed_predict(trajectory, robust=True, prior=prior),
        ),
        CallableSharedClassifierAdapter(
            method_name="accumulator",
            sensor_regime_id="position_only",
            predict_fn=lambda trajectory, prior=None: _accumulator_predict(trajectory, prior=prior),
        ),
        CallableSharedClassifierAdapter(
            method_name="kalman_bank",
            sensor_regime_id="position_only",
            predict_fn=lambda trajectory, prior=None: _kalman_predict(trajectory, prior=prior),
        ),
        CallableSharedClassifierAdapter(
            method_name="kalman_bank_velocity_aided",
            sensor_regime_id="position_plus_direct_velocity",
            predict_fn=lambda trajectory, prior=None: _kalman_velocity_aided_predict(trajectory, prior=prior),
        ),
    )


def _shared_prior_flip_fraction(method_name: str, trajectories: tuple[SharedDynamicsTrajectory, ...]) -> float:
    base_prior = {"constant_velocity": 0.5, "constant_acceleration": 0.5}
    priors_to_compare = (
        {"constant_velocity": 0.25, "constant_acceleration": 0.75},
        {"constant_velocity": 0.75, "constant_acceleration": 0.25},
    )
    flip_count = 0
    for trajectory in trajectories:
        baseline = _predict_with_method(method_name, trajectory, prior=base_prior).final_predicted_class
        alternatives = [
            _predict_with_method(method_name, trajectory, prior=prior).final_predicted_class
            for prior in priors_to_compare
        ]
        if any(predicted != baseline for predicted in alternatives):
            flip_count += 1
    return flip_count / len(trajectories)


def analyze_common_dataset_comparison(*, seed: int = 7, trajectories_per_case: int = 8) -> CommonComparisonResult:
    trajectories = generate_shared_dynamics_dataset(seed=seed, trajectories_per_case=trajectories_per_case)
    classifiers = default_shared_classifier_adapters()
    method_names = tuple(classifier.method_name for classifier in classifiers)
    runs = evaluate_shared_classifier_registry(trajectories, classifiers)

    rows: list[CommonComparisonRow] = []
    for method_name in method_names:
        method_runs = [run for run in runs if run.method_name == method_name]
        overall_accuracy = sum(1.0 if run.final_predicted_class == run.true_class else 0.0 for run in method_runs) / len(method_runs)

        def _scenario_accuracy(scenario_name: str) -> float:
            selected = [run for run in method_runs if run.scenario_name == scenario_name]
            return sum(1.0 if run.final_predicted_class == run.true_class else 0.0 for run in selected) / len(selected)

        rows.append(
            CommonComparisonRow(
                method_name=method_name,
                overall_accuracy=overall_accuracy,
                easy_accuracy=_scenario_accuracy("easy"),
                irregular_accuracy=_scenario_accuracy("irregular"),
                endpoint_match_accuracy=_scenario_accuracy("endpoint_match"),
                short_accuracy=_scenario_accuracy("short"),
                noisy_accuracy=_scenario_accuracy("short_noisy"),
                outlier_accuracy=_scenario_accuracy("outlier"),
                prior_flip_fraction=_shared_prior_flip_fraction(method_name, trajectories),
            )
        )
    return CommonComparisonResult(trajectories=trajectories, runs=tuple(runs), rows=tuple(rows))


def render_common_dataset_comparison_report(result: CommonComparisonResult) -> str:
    report = MarkdownDocument("Common-Dataset Technique Comparison")
    report.paragraph(
        "This artifact evaluates the current technique families on the same shared binary dynamics corpus: "
        "constant velocity versus constant acceleration with easy, irregular-`dt`, matched-endpoint irregular, "
        "short-horizon boundary, short-horizon noisy, and outlier-corrupted scenarios."
    )
    report.heading("Method Metrics", level=2)
    report.table(
        [
            "method",
            "overall",
            "easy",
            "irregular",
            "endpoint_match",
            "short",
            "short_noisy",
            "outlier",
            "prior_flip_fraction",
        ],
        [
            (
                row.method_name,
                f"{row.overall_accuracy:.3f}",
                f"{row.easy_accuracy:.3f}",
                f"{row.irregular_accuracy:.3f}",
                f"{row.endpoint_match_accuracy:.3f}",
                f"{row.short_accuracy:.3f}",
                f"{row.noisy_accuracy:.3f}",
                f"{row.outlier_accuracy:.3f}",
                f"{row.prior_flip_fraction:.3f}",
            )
            for row in result.rows
        ],
    )
    report.heading("Interpretation", level=2)
    report.bullet_list(
        [
            "This is the first apples-to-apples technique comparison on one shared corpus.",
            "Pointwise should act as the weak lower bound because it only uses the last measurement.",
            "The matched-endpoint irregular case removes most endpoint information, so methods need the time history rather than just the last sample.",
            "Short-horizon cases are boundary cases: there is not much elapsed time for acceleration to separate from constant velocity.",
            "The outlier case is there to expose the difference between raw feature accumulation and more robust temporal/model-based methods.",
            "`kalman_bank` remains a position-only sensing regime, even when it uses derived pseudo-observations.",
            "`kalman_bank_velocity_aided` is a separate sensor regime with an actual extra velocity stream.",
        ]
    )
    return report.text()


def _sensor_regime_metadata() -> tuple[dict[str, object], ...]:
    return (
        {
            "sensor_regime_id": "position_only",
            "description": "Position measurements only; derived pseudo-observations may be allowed but no independent auxiliary sensor is present.",
            "same_sensor_fairness_bucket": "position_only",
            "supported_measurement_dims": [1],
            "supported_coordinate_frames": ["scalar_line"],
        },
        {
            "sensor_regime_id": "position_plus_direct_velocity",
            "description": "Position measurements plus an independent direct velocity sensor stream.",
            "same_sensor_fairness_bucket": "position_plus_direct_velocity",
            "supported_measurement_dims": [1],
            "supported_coordinate_frames": ["scalar_line"],
        },
    )


def _sensor_regime_summary_rows(result: CommonComparisonResult) -> list[dict[str, object]]:
    return sensor_regime_summary_rows(result.runs)


def _render_common_metric_heatmap(result: CommonComparisonResult):
    plt = prepare_matplotlib()
    fields = (
        "overall_accuracy",
        "easy_accuracy",
        "irregular_accuracy",
        "endpoint_match_accuracy",
        "short_accuracy",
        "noisy_accuracy",
        "outlier_accuracy",
        "prior_flip_fraction",
    )
    matrix = [[getattr(row, field) for field in fields] for row in result.rows]
    fig, ax = plt.subplots(figsize=(10.6, 4.8))
    image = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_title("Common-Dataset Method Metrics", loc="left", fontsize=13, fontweight="bold")
    ax.set_xticks(range(len(fields)))
    ax.set_xticklabels(["overall", "easy", "irregular", "endpoint", "short", "short_noisy", "outlier", "prior_flip"], rotation=25, ha="right")
    ax.set_yticks(range(len(result.rows)))
    ax.set_yticklabels([row.method_name for row in result.rows])
    for row_index, row in enumerate(result.rows):
        for col_index, field in enumerate(fields):
            ax.text(col_index, row_index, f"{getattr(row, field):.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="metric value")
    fig.tight_layout()
    return fig


def _render_common_confusion_bars(result: CommonComparisonResult):
    plt = prepare_matplotlib()
    fig, ax = plt.subplots(figsize=(10.6, 5.0))
    method_names = [row.method_name for row in result.rows]
    overall = [row.overall_accuracy for row in result.rows]
    irregular = [row.irregular_accuracy for row in result.rows]
    endpoint = [row.endpoint_match_accuracy for row in result.rows]
    short = [row.short_accuracy for row in result.rows]
    noisy = [row.noisy_accuracy for row in result.rows]
    x = list(range(len(method_names)))
    ax.bar([value - 0.36 for value in x], overall, width=0.14, label="overall", color="#2563eb")
    ax.bar([value - 0.18 for value in x], irregular, width=0.14, label="irregular", color="#16a34a")
    ax.bar(x, endpoint, width=0.14, label="endpoint_match", color="#7c3aed")
    ax.bar([value + 0.18 for value in x], short, width=0.14, label="short", color="#dc2626")
    ax.bar([value + 0.36 for value in x], noisy, width=0.14, label="short_noisy", color="#d97706")
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title("Shared-Corpus Accuracy by Method", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _render_dataset_balance(result: CommonComparisonResult):
    plt = prepare_matplotlib()
    scenario_names = sorted({trajectory.scenario_name for trajectory in result.trajectories})
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    class_counts = {
        class_name: [sum(1 for trajectory in result.trajectories if trajectory.scenario_name == scenario_name and trajectory.true_class == class_name) for scenario_name in scenario_names]
        for class_name in CLASS_NAMES
    }
    x = list(range(len(scenario_names)))
    width = 0.34
    ax.bar([value - width / 2 for value in x], class_counts["constant_velocity"], width=width, label="constant_velocity", color="#2563eb")
    ax.bar([value + width / 2 for value in x], class_counts["constant_acceleration"], width=width, label="constant_acceleration", color="#dc2626")
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, rotation=20, ha="right")
    ax.set_ylabel("trajectory count")
    ax.set_title("Dataset Class Balance by Scenario", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _render_covariate_audit(result: CommonComparisonResult):
    plt = prepare_matplotlib()
    scenario_names = sorted({trajectory.scenario_name for trajectory in result.trajectories})
    metrics = ("duration", "sample_count", "mean_dt", "measurement_rmse")
    matrix: list[list[float]] = []
    for scenario_name in scenario_names:
        selected = [trajectory for trajectory in result.trajectories if trajectory.scenario_name == scenario_name]
        durations = [trajectory.times[-1] - trajectory.times[0] for trajectory in selected]
        sample_counts = [float(len(trajectory.times)) for trajectory in selected]
        mean_dts = [
            sum(trajectory.times[index] - trajectory.times[index - 1] for index in range(1, len(trajectory.times))) / max(len(trajectory.times) - 1, 1)
            for trajectory in selected
        ]
        rmses = [
            (
                sum((measurement - truth) ** 2 for measurement, truth in zip(trajectory.measurements, trajectory.true_position)) / max(len(trajectory.measurements), 1)
            ) ** 0.5
            for trajectory in selected
        ]
        matrix.append(
            [
                sum(durations) / len(durations),
                sum(sample_counts) / len(sample_counts),
                sum(mean_dts) / len(mean_dts),
                sum(rmses) / len(rmses),
            ]
        )

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    image = ax.imshow(matrix, aspect="auto", cmap="YlOrBr")
    ax.set_title("Scenario Covariate Audit", loc="left", fontsize=13, fontweight="bold")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=20, ha="right")
    ax.set_yticks(range(len(scenario_names)))
    ax.set_yticklabels(scenario_names)
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            ax.text(col_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="mean value")
    fig.tight_layout()
    return fig


def _render_scenario_profile(result: CommonComparisonResult):
    plt = prepare_matplotlib()
    scenario_order = ["easy", "irregular", "endpoint_match", "short", "short_noisy", "outlier"]
    palette = {
        "pointwise": "#2563eb",
        "windowed_raw": "#dc2626",
        "windowed_robust": "#d97706",
        "accumulator": "#16a34a",
        "kalman_bank": "#7c3aed",
        "kalman_bank_velocity_aided": "#0f766e",
    }
    labels = {
        "easy": "easy",
        "irregular": "irregular",
        "endpoint_match": "endpoint",
        "short": "short",
        "short_noisy": "short+noise",
        "outlier": "outlier",
    }
    fig, ax = plt.subplots(figsize=(10.4, 5.2))
    x = list(range(len(scenario_order)))
    for row in result.rows:
        ys = [
            row.easy_accuracy,
            row.irregular_accuracy,
            row.endpoint_match_accuracy,
            row.short_accuracy,
            row.noisy_accuracy,
            row.outlier_accuracy,
        ]
        ax.plot(
            x,
            ys,
            marker="o",
            linewidth=2.2,
            markersize=6.0,
            color=palette[row.method_name],
            label=row.method_name,
        )
        ax.text(x[-1] + 0.08, ys[-1], row.method_name, color=palette[row.method_name], fontsize=8, va="center")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[name] for name in scenario_order], rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title("Scenario Accuracy Profile by Method", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="lower left")
    fig.tight_layout()
    return fig


def _render_prior_flip_tradeoff(result: CommonComparisonResult):
    plt = prepare_matplotlib()
    palette = {
        "pointwise": "#2563eb",
        "windowed_raw": "#dc2626",
        "windowed_robust": "#d97706",
        "accumulator": "#16a34a",
        "kalman_bank": "#7c3aed",
        "kalman_bank_velocity_aided": "#0f766e",
    }
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for row in result.rows:
        ax.scatter(
            row.prior_flip_fraction,
            row.overall_accuracy,
            s=120,
            color=palette[row.method_name],
            alpha=0.9,
        )
        ax.text(
            row.prior_flip_fraction + 0.01,
            row.overall_accuracy,
            row.method_name,
            fontsize=8.5,
            color=palette[row.method_name],
            va="center",
        )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("prior flip fraction")
    ax.set_ylabel("overall accuracy")
    ax.set_title("Accuracy vs Prior Sensitivity", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def _render_trajectory_examples(result: CommonComparisonResult):
    plt = prepare_matplotlib()
    scenario_names = sorted({trajectory.scenario_name for trajectory in result.trajectories})
    fig, axes = plt.subplots(len(scenario_names), 1, figsize=(10.5, 2.5 * len(scenario_names)), sharex=False)
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]
    palette = {
        "constant_velocity": "#2563eb",
        "constant_acceleration": "#dc2626",
    }

    for ax, scenario_name in zip(axes_list, scenario_names):
        selected = [trajectory for trajectory in result.trajectories if trajectory.scenario_name == scenario_name]
        representatives = {}
        for class_name in CLASS_NAMES:
            representatives[class_name] = next(trajectory for trajectory in selected if trajectory.true_class == class_name)
        for class_name, trajectory in representatives.items():
            ax.plot(trajectory.times, trajectory.true_position, color=palette[class_name], linewidth=2.2, label=f"{class_name} true")
            ax.scatter(trajectory.times, trajectory.measurements, color=palette[class_name], s=18, alpha=0.75, marker="o")
        ax.set_title(scenario_name, loc="left", fontsize=11, fontweight="bold")
        ax.set_ylabel("position")
        ax.grid(True, alpha=0.25)
    if axes_list:
        axes_list[0].legend(frameon=False, ncol=2, loc="upper left")
        axes_list[-1].set_xlabel("time")
    fig.suptitle("Representative Trajectories by Scenario", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def _render_method_confusion_heatmaps(result: CommonComparisonResult):
    plt = prepare_matplotlib()
    method_names = [row.method_name for row in result.rows]
    fig, axes = plt.subplots(1, len(method_names), figsize=(3.2 * len(method_names), 4.2))
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]
    class_names = list(CLASS_NAMES)
    for ax, method_name in zip(axes_list, method_names):
        method_runs = [run for run in result.runs if run.method_name == method_name]
        matrix = [
            [
                sum(1 for run in method_runs if run.true_class == true_class and run.final_predicted_class == predicted_class)
                for predicted_class in class_names
            ]
            for true_class in class_names
        ]
        image = ax.imshow(matrix, aspect="auto", cmap="Blues")
        ax.set_title(method_name, fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=25, ha="right", fontsize=8)
        ax.set_yticks(range(len(class_names)))
        ax.set_yticklabels(class_names, fontsize=8)
        for row_index, row in enumerate(matrix):
            for col_index, value in enumerate(row):
                ax.text(col_index, row_index, str(value), ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Final Confusion by Method", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def _figure_to_svg(fig) -> str:
    plt = prepare_matplotlib()
    buffer = io.StringIO()
    try:
        fig.savefig(buffer, format="svg", bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def write_common_dataset_comparison_artifacts(
    output_dir: str | Path,
    *,
    seed: int = 7,
    result: CommonComparisonResult | None = None,
) -> CommonComparisonArtifacts:
    comparison = result or analyze_common_dataset_comparison(seed=seed)
    output_root = Path(output_dir)
    run_dir = output_root / "common_dataset_comparison_v1"
    run_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_dir / "common_dataset_comparison_report.md"
    trajectory_path = run_dir / "shared_trajectories.csv"
    run_summary_path = run_dir / "method_run_summary.csv"
    method_summary_path = run_dir / "method_summary.csv"
    sensor_regimes_path = run_dir / "sensor_regimes.json"
    sensor_regime_metrics_path = run_dir / "metrics_by_sensor_regime.csv"
    heatmap_png_path = run_dir / "common_dataset_metric_heatmap.png"
    confusion_png_path = run_dir / "shared_accuracy_bars.png"
    plots_dir = run_dir / "plots"
    overview_dir = plots_dir / "overview"
    single_trajectory_dir = plots_dir / "single_trajectory"
    confusion_dir = plots_dir / "confusion"
    diagnostics_dir = plots_dir / "diagnostics"
    overview_dir.mkdir(parents=True, exist_ok=True)
    single_trajectory_dir.mkdir(parents=True, exist_ok=True)
    confusion_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    overview_balance_png_path = overview_dir / "dataset_class_balance.png"
    overview_covariates_png_path = overview_dir / "covariate_leakage_audit.png"
    scenario_profile_png_path = diagnostics_dir / "scenario_accuracy_profile.png"
    prior_sensitivity_png_path = diagnostics_dir / "accuracy_vs_prior_sensitivity.png"
    trajectory_examples_png_path = single_trajectory_dir / "trajectory_examples_by_scenario.png"
    final_confusion_png_path = confusion_dir / "final_confusion_by_method.png"

    report_path.write_text(render_common_dataset_comparison_report(comparison), encoding="utf-8")
    write_csv(
        trajectory_path,
        [
            {
                "trajectory_id": trajectory.trajectory_id,
                "true_class": trajectory.true_class,
                "scenario_name": trajectory.scenario_name,
                "seed": trajectory.seed,
                "measurement_dim": trajectory.measurement_dim,
                "coordinate_frame": trajectory.coordinate_frame,
                "times": " ".join(f"{value:.3f}" for value in trajectory.times),
                "measurements": " ".join(f"{value:.3f}" for value in trajectory.measurements),
            }
            for trajectory in comparison.trajectories
        ],
        ["trajectory_id", "true_class", "scenario_name", "seed", "measurement_dim", "coordinate_frame", "times", "measurements"],
    )
    write_csv(
        run_summary_path,
        [
            {
                "method_name": run.method_name,
                "sensor_regime_id": run.sensor_regime_id,
                "measurement_dim": run.measurement_dim,
                "coordinate_frame": run.coordinate_frame,
                "trajectory_id": run.trajectory_id,
                "true_class": run.true_class,
                "scenario_name": run.scenario_name,
                "final_predicted_class": run.final_predicted_class,
                "final_confidence": run.final_confidence,
                **{f"posterior_{class_name}": run.final_weights[class_name] for class_name in CLASS_NAMES},
            }
            for run in comparison.runs
        ],
        ["method_name", "sensor_regime_id", "measurement_dim", "coordinate_frame", "trajectory_id", "true_class", "scenario_name", "final_predicted_class", "final_confidence", "posterior_constant_velocity", "posterior_constant_acceleration"],
    )
    write_csv(
        method_summary_path,
        [
            {
                "method_name": row.method_name,
                "overall_accuracy": row.overall_accuracy,
                "easy_accuracy": row.easy_accuracy,
                "irregular_accuracy": row.irregular_accuracy,
                "endpoint_match_accuracy": row.endpoint_match_accuracy,
                "short_accuracy": row.short_accuracy,
                "noisy_accuracy": row.noisy_accuracy,
                "outlier_accuracy": row.outlier_accuracy,
                "prior_flip_fraction": row.prior_flip_fraction,
            }
            for row in comparison.rows
        ],
        [
            "method_name",
            "overall_accuracy",
            "easy_accuracy",
            "irregular_accuracy",
            "endpoint_match_accuracy",
            "short_accuracy",
            "noisy_accuracy",
            "outlier_accuracy",
            "prior_flip_fraction",
        ],
    )
    sensor_regimes_path.write_text(json.dumps(_sensor_regime_metadata(), indent=2), encoding="utf-8")
    write_csv(
        sensor_regime_metrics_path,
        _sensor_regime_summary_rows(comparison),
        ["sensor_regime_id", "num_predictions", "mean_accuracy", "mean_confidence", "measurement_dims", "coordinate_frames", "methods"],
    )
    heatmap_png_path.write_bytes(_figure_to_png(_render_common_metric_heatmap(comparison)))
    confusion_png_path.write_bytes(_figure_to_png(_render_common_confusion_bars(comparison)))
    overview_balance_png_path.write_bytes(_figure_to_png(_render_dataset_balance(comparison)))
    overview_covariates_png_path.write_bytes(_figure_to_png(_render_covariate_audit(comparison)))
    scenario_profile_png_path.write_bytes(_figure_to_png(_render_scenario_profile(comparison)))
    prior_sensitivity_png_path.write_bytes(_figure_to_png(_render_prior_flip_tradeoff(comparison)))
    trajectory_examples_png_path.write_bytes(_figure_to_png(_render_trajectory_examples(comparison)))
    final_confusion_png_path.write_bytes(_figure_to_png(_render_method_confusion_heatmaps(comparison)))
    return CommonComparisonArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        trajectory_path=trajectory_path,
        run_summary_path=run_summary_path,
        method_summary_path=method_summary_path,
        sensor_regimes_path=sensor_regimes_path,
        sensor_regime_metrics_path=sensor_regime_metrics_path,
        heatmap_png_path=heatmap_png_path,
        confusion_png_path=confusion_png_path,
        plots_dir=plots_dir,
        overview_balance_png_path=overview_balance_png_path,
        overview_covariates_png_path=overview_covariates_png_path,
        scenario_profile_png_path=scenario_profile_png_path,
        prior_sensitivity_png_path=prior_sensitivity_png_path,
        trajectory_examples_png_path=trajectory_examples_png_path,
        final_confusion_png_path=final_confusion_png_path,
    )
