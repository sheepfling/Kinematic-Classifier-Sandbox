from __future__ import annotations

import random
from math import log

from ..advanced_filters.shared_classifier_methods import (
    as_shared_classifier_run,
    run_shared_particle_filter_classifier,
    run_shared_rbpf_classifier,
)
from ..inference.kalman_filter_bank import KalmanModelSpec, run_kalman_filter_bank
from ..inference.kalman_filter_bank import KalmanTrajectory as SharedKalmanTrajectory
from ..scenarios import SCENARIO_MEASUREMENT_SIGMA, SCENARIO_TIMES
from ..scenarios import get_scenario_dynamics as _scenario_dynamics
from ..utils.math import _median3 as _median3
from ..utils.math import gaussian_logpdf as _gaussian_logpdf
from ..utils.math import normalize_log_scores as _normalize_log_scores
from ..utils.math import normalize_prior as _lib_normalize_prior
from ..validation.shared_evaluation import (
    CallableSharedClassifierAdapter,
    evaluate_shared_classifier_registry,
)
from ..validation.shared_evaluation_contracts import (
    SharedClassifierMethodSpec,
    SharedMethodCapabilities,
)
from .common_dataset_comparison_contracts import (
    CommonComparisonArtifacts,
    CommonComparisonResult,
    CommonComparisonRow,
    CommonMethodRun,
    SharedDynamicsTrajectory,
)
from .common_dataset_comparison_reporting import render_common_dataset_comparison_report


CLASS_NAMES = ("constant_velocity", "constant_acceleration")


def _normalize_prior(prior: dict[str, float] | None) -> dict[str, float]:
    return _lib_normalize_prior(prior, CLASS_NAMES)


def _shared_method_specs() -> tuple[SharedClassifierMethodSpec, ...]:
    return (
        SharedClassifierMethodSpec(
            method_name="pointwise",
            sensor_regime_id="position_only",
            primary_evaluation_family="shared_binary_dynamics",
            supported_scenario_families=("shared_binary_dynamics",),
            witness_artifact=None,
            capabilities=SharedMethodCapabilities(True, False, False, False, False, False, False),
        ),
        SharedClassifierMethodSpec(
            method_name="windowed_raw",
            sensor_regime_id="position_only",
            primary_evaluation_family="shared_binary_dynamics",
            supported_scenario_families=("shared_binary_dynamics",),
            witness_artifact=None,
            capabilities=SharedMethodCapabilities(False, True, False, False, False, False, False),
        ),
        SharedClassifierMethodSpec(
            method_name="windowed_robust",
            sensor_regime_id="position_only",
            primary_evaluation_family="shared_binary_dynamics",
            supported_scenario_families=("shared_binary_dynamics",),
            witness_artifact=None,
            capabilities=SharedMethodCapabilities(False, True, False, False, False, False, False),
        ),
        SharedClassifierMethodSpec(
            method_name="accumulator",
            sensor_regime_id="position_only",
            primary_evaluation_family="shared_binary_dynamics",
            supported_scenario_families=("shared_binary_dynamics",),
            witness_artifact=None,
            capabilities=SharedMethodCapabilities(False, True, False, False, False, False, False),
        ),
        SharedClassifierMethodSpec(
            method_name="kalman_bank",
            sensor_regime_id="position_only",
            primary_evaluation_family="shared_binary_dynamics",
            supported_scenario_families=("shared_binary_dynamics",),
            witness_artifact=None,
            capabilities=SharedMethodCapabilities(False, True, True, False, False, False, False),
        ),
        SharedClassifierMethodSpec(
            method_name="kalman_bank_velocity_aided",
            sensor_regime_id="position_plus_direct_velocity",
            primary_evaluation_family="shared_binary_dynamics",
            supported_scenario_families=("shared_binary_dynamics",),
            witness_artifact=None,
            capabilities=SharedMethodCapabilities(False, True, True, False, False, False, False),
        ),
        SharedClassifierMethodSpec(
            method_name="particle_filter_bank",
            sensor_regime_id="position_only",
            primary_evaluation_family="abs_range_multimodal",
            supported_scenario_families=("abs_range_multimodal", "ou_mean_reversion"),
            witness_artifact="artifacts/pf_abs_range_multimodal_oracle_v1/metrics_against_oracle.csv",
            capabilities=SharedMethodCapabilities(False, True, True, False, True, True, True),
        ),
        SharedClassifierMethodSpec(
            method_name="rbpf",
            sensor_regime_id="position_only",
            primary_evaluation_family="latent_maneuver_onset",
            supported_scenario_families=("latent_maneuver_onset",),
            witness_artifact="artifacts/advanced_filter_comparison_v1/pf_vs_rbpf_frontier_summary.csv",
            capabilities=SharedMethodCapabilities(False, True, True, True, True, True, False),
        ),
    )


def _make_shared_trajectory(*, trajectory_id: str, true_class: str, scenario_name: str, seed: int) -> SharedDynamicsTrajectory:
    rng = random.Random(seed)
    times = SCENARIO_TIMES[scenario_name]
    velocity0, acceleration = _scenario_dynamics(scenario_name, true_class)
    true_position = tuple(velocity0 * time + 0.5 * acceleration * time * time for time in times)
    true_velocity = tuple(velocity0 + acceleration * time for time in times)
    true_acceleration = tuple(acceleration for _ in times)
    measurement_sigma = SCENARIO_MEASUREMENT_SIGMA[scenario_name]
    measurements = tuple(value + rng.gauss(0.0, measurement_sigma) for value in true_position)
    if scenario_name == "outlier":
        outlier_index = len(measurements) // 2
        rebound_index = min(outlier_index + 1, len(measurements) - 1)
        glitch = 2.2
        measurements = tuple(
            value + (-glitch if index == outlier_index else 0.0) + (glitch if index == rebound_index else 0.0)
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


def _pointwise_predict(trajectory: SharedDynamicsTrajectory, *, prior: dict[str, float] | None = None) -> CommonMethodRun:
    last_measurement = trajectory.measurements[-1]
    last_time = trajectory.times[-1]
    prior_weights = _normalize_prior(prior)
    log_scores = {
        class_name: log(max(prior_weights[class_name], 1e-12))
        + _gaussian_logpdf(last_measurement, _class_expected_position(class_name, last_time, trajectory.scenario_name), 0.35**2)
        for class_name in CLASS_NAMES
    }
    weights = _normalize_log_scores(log_scores)
    predicted = max(weights, key=weights.get)
    return CommonMethodRun("pointwise", "position_only", trajectory.trajectory_id, trajectory.true_class, trajectory.scenario_name, predicted, weights[predicted], weights, trajectory.measurement_dim, trajectory.coordinate_frame)


def _accumulator_predict(trajectory: SharedDynamicsTrajectory, *, prior: dict[str, float] | None = None) -> CommonMethodRun:
    posterior = _normalize_prior(prior)
    for time, measurement in zip(trajectory.times, trajectory.measurements, strict=True):
        log_scores = {
            class_name: log(max(posterior[class_name], 1e-12))
            + _gaussian_logpdf(measurement, _class_expected_position(class_name, time, trajectory.scenario_name), 0.25**2)
            for class_name in CLASS_NAMES
        }
        posterior = _normalize_log_scores(log_scores)
    predicted = max(posterior, key=posterior.get)
    return CommonMethodRun("accumulator", "position_only", trajectory.trajectory_id, trajectory.true_class, trajectory.scenario_name, predicted, posterior[predicted], posterior, trajectory.measurement_dim, trajectory.coordinate_frame)


def _window_features(trajectory: SharedDynamicsTrajectory, *, robust: bool) -> dict[str, float]:
    times = list(trajectory.times)
    values = list(trajectory.measurements)
    if robust and len(values) >= 3:
        values = [_median3(values, index) for index in range(len(values))]
    mean_t = sum(times) / len(times)
    mean_y = sum(values) / len(values)
    denominator = sum((time - mean_t) ** 2 for time in times)
    slope = 0.0 if denominator <= 1e-9 else sum((time - mean_t) * (value - mean_y) for time, value in zip(times, values, strict=True)) / denominator
    midpoint = len(times) // 2
    quadratic_proxy = values[-1] - 2.0 * values[midpoint] + values[0] if len(times) >= 3 else 0.0
    return {"slope": slope, "quadratic_proxy": quadratic_proxy}


def _windowed_predict(trajectory: SharedDynamicsTrajectory, *, robust: bool, prior: dict[str, float] | None = None) -> CommonMethodRun:
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
    log_scores = {
        class_name: log(max(prior_weights[class_name], 1e-12))
        + _gaussian_logpdf(features["slope"], specs[class_name]["slope"], specs[class_name]["sigma_slope"] ** 2)
        + _gaussian_logpdf(features["quadratic_proxy"], specs[class_name]["quadratic_proxy"], specs[class_name]["sigma_quad"] ** 2)
        for class_name in CLASS_NAMES
    }
    weights = _normalize_log_scores(log_scores)
    predicted = max(weights, key=weights.get)
    return CommonMethodRun(method_name, "position_only", trajectory.trajectory_id, trajectory.true_class, trajectory.scenario_name, predicted, weights[predicted], weights, trajectory.measurement_dim, trajectory.coordinate_frame)


def _shared_kalman_model_specs(prior_weights: dict[str, float]) -> tuple[KalmanModelSpec, ...]:
    return (
        KalmanModelSpec("constant_velocity_quiet", "constant_velocity", 2, 0.14, 0.20, 5.0, 0.75 * prior_weights["constant_velocity"]),
        KalmanModelSpec("constant_velocity_rough", "constant_velocity", 2, 0.22, 0.20, 5.5, 0.25 * prior_weights["constant_velocity"]),
        KalmanModelSpec("constant_acceleration_quiet", "constant_acceleration", 3, 0.24, 0.20, 6.0, 0.75 * prior_weights["constant_acceleration"]),
        KalmanModelSpec("constant_acceleration_rough", "constant_acceleration", 3, 0.34, 0.20, 6.5, 0.25 * prior_weights["constant_acceleration"]),
    )


def _shared_kalman_trajectory(trajectory: SharedDynamicsTrajectory) -> SharedKalmanTrajectory:
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


def _synthesized_velocity_measurements(trajectory: SharedDynamicsTrajectory, *, sigma: float = 0.12) -> tuple[float, ...]:
    rng = random.Random(trajectory.seed + 9001)
    return tuple(value + rng.gauss(0.0, sigma) for value in trajectory.true_velocity)


def _kalman_predict(trajectory: SharedDynamicsTrajectory, *, prior: dict[str, float] | None = None) -> CommonMethodRun:
    run = run_kalman_filter_bank(_shared_kalman_trajectory(trajectory), _shared_kalman_model_specs(_normalize_prior(prior)), derived_velocity_observation=True, derived_acceleration_observation=True)
    return CommonMethodRun("kalman_bank", "position_only", trajectory.trajectory_id, trajectory.true_class, trajectory.scenario_name, run.final_predicted_class, run.final_confidence, run.final_weights, trajectory.measurement_dim, trajectory.coordinate_frame)


def _kalman_velocity_aided_predict(trajectory: SharedDynamicsTrajectory, *, prior: dict[str, float] | None = None) -> CommonMethodRun:
    velocity_sigma = 0.12
    run = run_kalman_filter_bank(
        _shared_kalman_trajectory(trajectory),
        _shared_kalman_model_specs(_normalize_prior(prior)),
        velocity_measurements=_synthesized_velocity_measurements(trajectory, sigma=velocity_sigma),
        velocity_measurement_sigma=velocity_sigma,
    )
    return CommonMethodRun("kalman_bank_velocity_aided", "position_plus_direct_velocity", trajectory.trajectory_id, trajectory.true_class, trajectory.scenario_name, run.final_predicted_class, run.final_confidence, run.final_weights, trajectory.measurement_dim, trajectory.coordinate_frame)


def _particle_filter_predict(trajectory: SharedDynamicsTrajectory, *, prior: dict[str, float] | None = None) -> CommonMethodRun:
    return as_shared_classifier_run(trajectory, run_shared_particle_filter_classifier(trajectory, prior=prior))


def _rbpf_predict(trajectory: SharedDynamicsTrajectory, *, prior: dict[str, float] | None = None) -> CommonMethodRun:
    return as_shared_classifier_run(trajectory, run_shared_rbpf_classifier(trajectory, prior=prior))


def _predict_with_method(method_name: str, trajectory: SharedDynamicsTrajectory, *, prior: dict[str, float] | None = None) -> CommonMethodRun:
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
    specs = {spec.method_name: spec for spec in _shared_method_specs()}
    return (
        CallableSharedClassifierAdapter(specs["pointwise"], lambda trajectory, prior=None: _pointwise_predict(trajectory, prior=prior)),
        CallableSharedClassifierAdapter(specs["windowed_raw"], lambda trajectory, prior=None: _windowed_predict(trajectory, robust=False, prior=prior)),
        CallableSharedClassifierAdapter(specs["windowed_robust"], lambda trajectory, prior=None: _windowed_predict(trajectory, robust=True, prior=prior)),
        CallableSharedClassifierAdapter(specs["accumulator"], lambda trajectory, prior=None: _accumulator_predict(trajectory, prior=prior)),
        CallableSharedClassifierAdapter(specs["kalman_bank"], lambda trajectory, prior=None: _kalman_predict(trajectory, prior=prior)),
        CallableSharedClassifierAdapter(specs["kalman_bank_velocity_aided"], lambda trajectory, prior=None: _kalman_velocity_aided_predict(trajectory, prior=prior)),
        CallableSharedClassifierAdapter(specs["particle_filter_bank"], lambda trajectory, prior=None: _particle_filter_predict(trajectory, prior=prior)),
        CallableSharedClassifierAdapter(specs["rbpf"], lambda trajectory, prior=None: _rbpf_predict(trajectory, prior=prior)),
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
        alternatives = [_predict_with_method(method_name, trajectory, prior=prior).final_predicted_class for prior in priors_to_compare]
        if any(predicted != baseline for predicted in alternatives):
            flip_count += 1
    return flip_count / len(trajectories)


def analyze_common_dataset_comparison(*, seed: int = 7, trajectories_per_case: int = 8) -> CommonComparisonResult:
    trajectories = generate_shared_dynamics_dataset(seed=seed, trajectories_per_case=trajectories_per_case)
    classifiers = default_shared_classifier_adapters()
    method_specs = tuple(classifier.method_spec for classifier in classifiers)
    shared_classifiers = tuple(classifier for classifier in classifiers if "shared_binary_dynamics" in classifier.method_spec.supported_scenario_families)
    runs = evaluate_shared_classifier_registry(trajectories, shared_classifiers)

    rows: list[CommonComparisonRow] = []
    for classifier in shared_classifiers:
        method_name = classifier.method_name
        method_runs = [run for run in runs if run.method_name == method_name]
        overall_accuracy = sum(1.0 if run.final_predicted_class == run.true_class else 0.0 for run in method_runs) / len(method_runs)

        def _scenario_accuracy(scenario_name: str) -> float:
            selected = [run for run in method_runs if run.scenario_name == scenario_name]
            return sum(1.0 if run.final_predicted_class == run.true_class else 0.0 for run in selected) / len(selected)

        rows.append(
            CommonComparisonRow(
                method_name=method_name,
                sensor_regime_id=classifier.sensor_regime_id,
                applicability_status="supported",
                primary_evaluation_family="shared_binary_dynamics",
                witness_artifact=None,
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
    for spec in method_specs:
        if "shared_binary_dynamics" in spec.supported_scenario_families:
            continue
        rows.append(
            CommonComparisonRow(
                method_name=spec.method_name,
                sensor_regime_id=spec.sensor_regime_id,
                applicability_status="witness_only",
                primary_evaluation_family=spec.primary_evaluation_family,
                witness_artifact=spec.witness_artifact,
                overall_accuracy=None,
                easy_accuracy=None,
                irregular_accuracy=None,
                endpoint_match_accuracy=None,
                short_accuracy=None,
                noisy_accuracy=None,
                outlier_accuracy=None,
                prior_flip_fraction=None,
            )
        )
    rows.append(
        CommonComparisonRow(
            method_name="ornstein_uhlenbeck_pf_v1",
            sensor_regime_id="position_only",
            applicability_status="witness_only",
            primary_evaluation_family="ou_mean_reversion",
            witness_artifact="artifacts/ornstein_uhlenbeck_witness_v1/ou_method_comparison.csv",
            overall_accuracy=None,
            easy_accuracy=None,
            irregular_accuracy=None,
            endpoint_match_accuracy=None,
            short_accuracy=None,
            noisy_accuracy=None,
            outlier_accuracy=None,
            prior_flip_fraction=None,
        )
    )
    return CommonComparisonResult(trajectories=trajectories, runs=tuple(runs), rows=tuple(rows), method_specs=method_specs)


__all__ = [
    "CommonComparisonArtifacts",
    "CommonComparisonResult",
    "CommonComparisonRow",
    "CommonMethodRun",
    "SCENARIO_MEASUREMENT_SIGMA",
    "SCENARIO_TIMES",
    "SharedDynamicsTrajectory",
    "analyze_common_dataset_comparison",
    "default_shared_classifier_adapters",
    "generate_shared_dynamics_dataset",
    "render_common_dataset_comparison_report",
]
