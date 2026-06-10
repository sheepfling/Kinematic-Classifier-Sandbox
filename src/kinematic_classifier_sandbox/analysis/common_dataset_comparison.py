from __future__ import annotations

import io
import random
from math import log

from ..inference.kalman_filter_bank import KalmanModelSpec, run_kalman_filter_bank
from ..inference.kalman_filter_bank import KalmanTrajectory as SharedKalmanTrajectory
from ..advanced_filters.shared_classifier_methods import (
    as_shared_classifier_run,
    run_shared_particle_filter_classifier,
    run_shared_rbpf_classifier,
)
from ..scenarios import (
    SCENARIO_MEASUREMENT_SIGMA,
    SCENARIO_TIMES,
)
from ..scenarios import (
    get_scenario_dynamics as _scenario_dynamics,
)
from ..utils.math import (
    _median3 as _median3,
)
from ..utils.math import (
    gaussian_logpdf as _gaussian_logpdf,
)
from ..utils.math import (
    normalize_log_scores as _normalize_log_scores,
)
from ..utils.math import (
    normalize_prior as _lib_normalize_prior,
)
from ..utils.plotting import plt
from ..validation.shared_evaluation import (
    CallableSharedClassifierAdapter,
    evaluate_shared_classifier_registry,
)
from .common_dataset_comparison_artifact_io import write_common_dataset_comparison_artifacts
from .common_dataset_comparison_contracts import (
    CommonComparisonArtifacts,
    CommonComparisonResult,
    CommonComparisonRow,
    CommonMethodRun,
    SharedDynamicsTrajectory,
)
from .common_dataset_comparison_reporting import render_common_dataset_comparison_report


def _normalize_prior(prior: dict[str, float] | None) -> dict[str, float]:
    return _lib_normalize_prior(prior, CLASS_NAMES)


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


def _particle_filter_predict(
    trajectory: SharedDynamicsTrajectory,
    *,
    prior: dict[str, float] | None = None,
) -> CommonMethodRun:
    return as_shared_classifier_run(
        trajectory,
        run_shared_particle_filter_classifier(trajectory, prior=prior),
    )


def _rbpf_predict(
    trajectory: SharedDynamicsTrajectory,
    *,
    prior: dict[str, float] | None = None,
) -> CommonMethodRun:
    return as_shared_classifier_run(
        trajectory,
        run_shared_rbpf_classifier(trajectory, prior=prior),
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
    if method_name == "particle_filter_bank":
        return _particle_filter_predict(trajectory, prior=prior)
    if method_name == "rbpf":
        return _rbpf_predict(trajectory, prior=prior)
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
        CallableSharedClassifierAdapter(
            method_name="particle_filter_bank",
            sensor_regime_id="position_only",
            predict_fn=lambda trajectory, prior=None: _particle_filter_predict(trajectory, prior=prior),
        ),
        CallableSharedClassifierAdapter(
            method_name="rbpf",
            sensor_regime_id="position_only",
            predict_fn=lambda trajectory, prior=None: _rbpf_predict(trajectory, prior=prior),
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


def _render_common_metric_heatmap(result: CommonComparisonResult):
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
