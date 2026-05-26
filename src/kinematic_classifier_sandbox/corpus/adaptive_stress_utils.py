from __future__ import annotations

import random
from dataclasses import dataclass, replace
from math import exp, log
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import yaml

from kinematic_classifier_sandbox.utils.io import write_csv

from ..analysis.common_dataset_comparison import (
    SharedDynamicsTrajectory,
    _accumulator_predict,
    _kalman_predict,
    _pointwise_predict,
    _windowed_predict,
)
from ..analysis.feature_analysis import _one_dimensional_feature_context_from_trajectory
from ..inference.irregular_window_comparison import (
    WindowRegimeTrajectory,
    _duration_window,
    _sample_count_window,
    generate_window_regime_trajectories,
)
from ..inference.irregular_window_comparison import (
    _gaussian_logpdf as _window_gaussian_logpdf,
)
from ..inference.prior_sensitivity_types import PriorSweepPredictions
from ..inference.irregular_window_comparison import (
    _normalize as _window_normalize,
)
from ..inference.transition_matrix_accumulator import (
    _run_mode_accumulator,
    default_switching_mode_specs,
    default_transition_matrix,
    generate_transition_switching_scenarios,
)
from ..trajectory_generator import GeneratedTrajectoryDataset
from ..utils.math import (
    _clamp,
    _entropy,
    _least_squares_slope,
    _median3,
    _quadratic_fit,
    _union_fieldnames,
)
from ..utils.plotting import plt
from ..utils.plotting import _figure_to_png
from .gym import CorpusGymAction, CorpusGymEnvironment, CorpusGymTarget


class ReferenceWindowStats(NamedTuple):
    sample_count: dict[str, dict[str, float]]
    duration: dict[str, dict[str, float]]


def _local_window_features(shared: SharedDynamicsTrajectory, *, robust: bool) -> dict[str, float]:
    times = list(shared.times)
    values = list(shared.measurements)
    if robust and len(values) >= 3:
        values = [_median3(values, index) for index in range(len(values))]
    slope = _least_squares_slope(times, values)
    _, _, quadratic = _quadratic_fit(times, values)
    return {
        "slope": slope,
        "quadratic_proxy": 2.0 * quadratic,
        "position_range": max(values) - min(values),
    }


def _observable_pair_posterior(shared: SharedDynamicsTrajectory, *, prior_cv: float = 0.5) -> tuple[dict[str, float], dict[str, float]]:
    times = list(shared.times)
    values = list(shared.measurements)
    _, _, quadratic = _quadratic_fit(times, values)
    acceleration_observation = 2.0 * quadratic
    duration = max(times[-1] - times[0], 1e-6)
    measurement_sigma = float(getattr(shared, "measurement_std", 0.0) or 0.0)
    sigma = max(0.14, measurement_sigma * 3.0 + 0.40 / duration)
    log_scores = {
        "constant_velocity": log(max(prior_cv, 1e-12)) + _window_gaussian_logpdf(acceleration_observation, 0.0, sigma),
        "constant_acceleration": log(max(1.0 - prior_cv, 1e-12)) + _window_gaussian_logpdf(acceleration_observation, 0.28, sigma),
    }
    weights = _window_normalize(log_scores)
    return weights, {"acceleration_observation": acceleration_observation, "observable_sigma": sigma}


def _stress_targets() -> tuple[CorpusGymTarget, ...]:
    return (
        CorpusGymTarget(
            target_id="stress_wrong_classification",
            target_type="target_class_pair",
            description="Boundary CV/CA trajectories that trigger pointwise misclassification.",
            class_name="constant_velocity",
            class_pair=("constant_velocity", "constant_acceleration"),
            target_tier="stress_v1",
        ),
        CorpusGymTarget(
            target_id="stress_high_entropy",
            target_type="target_class_pair",
            description="Boundary CV/CA trajectories that preserve high posterior entropy.",
            class_name="constant_acceleration",
            class_pair=("constant_velocity", "constant_acceleration"),
            target_tier="stress_v1",
        ),
        CorpusGymTarget(
            target_id="stress_prior_flip",
            target_type="target_prior_sensitivity",
            description="CV/CA trajectories with small prior-flip thresholds.",
            class_name="constant_velocity",
            class_pair=("constant_velocity", "constant_acceleration"),
            target_prior_sensitivity="high",
            target_tier="boundary_v1",
        ),
        CorpusGymTarget(
            target_id="stress_raw_extrema_failure",
            target_type="target_failure_mode",
            description="Outlier-corrupted CV/CA trajectories where raw extrema fail and robust extrema recover.",
            class_name="constant_velocity",
            class_pair=("constant_velocity", "constant_acceleration"),
            target_failure_mode="raw_extrema_failure",
            target_tier="adversarial_v1",
        ),
        CorpusGymTarget(
            target_id="stress_irregular_window_failure",
            target_type="target_failure_mode",
            description="Irregularly sampled CV/CA trajectories where duration-aware windows beat sample-count windows.",
            class_name="constant_acceleration",
            class_pair=("constant_velocity", "constant_acceleration"),
            target_failure_mode="irregular_window_failure",
            target_tier="boundary_v1",
        ),
        CorpusGymTarget(
            target_id="stress_kalman_mismatch",
            target_type="target_failure_mode",
            description="Short noisy CV/CA trajectories where Kalman mismatches while another position-only method recovers.",
            class_name="constant_acceleration",
            class_pair=("constant_velocity", "constant_acceleration"),
            target_failure_mode="kalman_mismatch",
            target_tier="stress_v1",
        ),
    )


def _random_action(rng: random.Random, target: CorpusGymTarget, *, seed: int) -> CorpusGymAction:
    return CorpusGymAction(
        seed=seed,
        tier_name=target.target_tier or "realistic_v1",
        duration_scale=rng.uniform(0.75, 1.25),
        measurement_scale=rng.uniform(0.80, 1.35),
        irregularity_scale=rng.uniform(0.75, 1.35),
        outlier_scale=rng.uniform(0.75, 1.35),
        step_scale=rng.uniform(0.80, 1.20),
    )


def _guided_action(rng: random.Random, failure_mode: str, target: CorpusGymTarget, *, seed: int) -> CorpusGymAction:
    if failure_mode == "wrong_classification":
        return CorpusGymAction(seed=seed, tier_name="stress_v1", duration_scale=rng.uniform(0.70, 0.95), measurement_scale=rng.uniform(1.15, 1.45), irregularity_scale=rng.uniform(0.95, 1.30), outlier_scale=rng.uniform(0.90, 1.10), step_scale=rng.uniform(0.75, 0.95))
    if failure_mode == "high_entropy":
        return CorpusGymAction(seed=seed, tier_name="stress_v1", duration_scale=rng.uniform(0.68, 0.92), measurement_scale=rng.uniform(1.20, 1.60), irregularity_scale=rng.uniform(0.95, 1.30), outlier_scale=rng.uniform(0.90, 1.15), step_scale=rng.uniform(0.72, 0.92))
    if failure_mode == "prior_flip":
        return CorpusGymAction(seed=seed, tier_name="boundary_v1", duration_scale=rng.uniform(0.72, 0.95), measurement_scale=rng.uniform(1.15, 1.45), irregularity_scale=rng.uniform(0.95, 1.20), outlier_scale=rng.uniform(0.90, 1.10), step_scale=rng.uniform(0.74, 0.94))
    if failure_mode == "raw_extrema_failure":
        return CorpusGymAction(seed=seed, tier_name="adversarial_v1", duration_scale=rng.uniform(0.85, 1.10), measurement_scale=rng.uniform(1.00, 1.25), irregularity_scale=rng.uniform(1.05, 1.40), outlier_scale=rng.uniform(1.55, 2.10), step_scale=rng.uniform(0.90, 1.05))
    if failure_mode == "irregular_window_failure":
        return CorpusGymAction(seed=seed, tier_name="boundary_v1", duration_scale=rng.uniform(0.95, 1.20), measurement_scale=rng.uniform(0.95, 1.20), irregularity_scale=rng.uniform(1.35, 1.90), outlier_scale=rng.uniform(0.90, 1.10), step_scale=rng.uniform(0.90, 1.10))
    if failure_mode == "kalman_mismatch":
        return CorpusGymAction(seed=seed, tier_name="stress_v1", duration_scale=rng.uniform(0.70, 0.95), measurement_scale=rng.uniform(1.20, 1.55), irregularity_scale=rng.uniform(0.95, 1.30), outlier_scale=rng.uniform(1.05, 1.35), step_scale=rng.uniform(0.72, 0.95))
    return _random_action(rng, target, seed=seed)


def _infer_shared_scenario_name(trajectory, context) -> str:
    if context.outlier_score >= 2.5:
        return "outlier"
    if len(trajectory.times) <= 5 and float(trajectory.measurement_std or 0.0) >= 0.18:
        return "short_noisy"
    if len(trajectory.times) <= 5:
        return "short"
    if context.sampling_irregularity >= 0.18:
        return "irregular"
    return "easy"


def _to_shared_trajectory(trajectory) -> SharedDynamicsTrajectory:
    dataset = GeneratedTrajectoryDataset(
        tier=str(trajectory.generator_parameters.get("tier", "realistic_v1")),
        seed=trajectory.seed,
        class_definitions=(),
        tier_definition=None,  # type: ignore[arg-type]
        trajectories=(trajectory,),
    )
    context = _one_dimensional_feature_context_from_trajectory(dataset, trajectory)
    return SharedDynamicsTrajectory(
        trajectory_id=trajectory.trajectory_id,
        true_class=trajectory.true_class,
        scenario_name=_infer_shared_scenario_name(trajectory, context),
        seed=trajectory.seed,
        times=trajectory.times,
        measurements=trajectory.measurements,
        true_position=trajectory.true_position or trajectory.measurements,
        true_velocity=trajectory.true_velocity or tuple(0.0 for _ in trajectory.times),
        true_acceleration=trajectory.true_acceleration or tuple(0.0 for _ in trajectory.times),
    )


def _reference_window_stats() -> ReferenceWindowStats:
    reference = generate_window_regime_trajectories(seed=7, replicas=12)
    sample_rows = [_sample_count_window(trajectory, 5) for trajectory in reference if trajectory.sampling_regime == "regular"]
    duration_rows = [_duration_window(trajectory, 5.0) for trajectory in reference if trajectory.sampling_regime == "regular"]
    stats: dict[str, dict[str, dict[str, float]]] = {"sample_count": {}, "duration": {}}
    for mode, rows in (("sample_count", sample_rows), ("duration", duration_rows)):
        for class_name in ("constant_velocity", "constant_acceleration"):
            selected = [row for row in rows if row.true_class == class_name]
            stats[mode][class_name] = {
                "slope_mean": sum(row.slope for row in selected) / len(selected),
                "curvature_mean": sum(row.curvature_proxy for row in selected) / len(selected),
                "range_mean": sum(row.position_range for row in selected) / len(selected),
                "slope_sigma": max((sum((row.slope - (sum(r.slope for r in selected) / len(selected))) ** 2 for row in selected) / max(len(selected) - 1, 1)) ** 0.5, 0.05),
                "curvature_sigma": max((sum((row.curvature_proxy - (sum(r.curvature_proxy for r in selected) / len(selected))) ** 2 for row in selected) / max(len(selected) - 1, 1)) ** 0.5, 0.05),
                "range_sigma": max((sum((row.position_range - (sum(r.position_range for r in selected) / len(selected))) ** 2 for row in selected) / max(len(selected) - 1, 1)) ** 0.5, 0.05),
            }
    return ReferenceWindowStats(sample_count=stats["sample_count"], duration=stats["duration"])


def _classify_window_row(row, stats: dict[str, dict[str, float]]) -> tuple[str, float]:
    log_scores: dict[str, float] = {}
    for class_name, spec in stats.items():
        log_scores[class_name] = (
            log(0.5)
            + _window_gaussian_logpdf(row.slope, spec["slope_mean"], spec["slope_sigma"])
            + _window_gaussian_logpdf(row.curvature_proxy, spec["curvature_mean"], spec["curvature_sigma"])
            + _window_gaussian_logpdf(row.position_range, spec["range_mean"], spec["range_sigma"])
        )
    weights = _window_normalize(log_scores)
    predicted = max(weights, key=weights.get)
    return predicted, weights[predicted]


def _prior_sweep_predictions(shared: SharedDynamicsTrajectory) -> PriorSweepPredictions:
    rows = []
    for prior_cv in (0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90):
        run = _accumulator_predict(
            shared,
            prior={"constant_velocity": prior_cv, "constant_acceleration": 1.0 - prior_cv},
        )
        rows.append((prior_cv, run.final_predicted_class, run.final_confidence))
    return PriorSweepPredictions(tuple(rows))


def _accumulator_trace(shared: SharedDynamicsTrajectory, prior: dict[str, float] | None = None) -> tuple[dict[str, object], ...]:
    posterior = {"constant_velocity": 0.5, "constant_acceleration": 0.5}
    if prior is not None:
        total = sum(prior.values())
        posterior = {name: value / max(total, 1e-12) for name, value in prior.items()}
    rows = []
    for time, measurement in zip(shared.times, shared.measurements):
        log_scores = {}
        for class_name in ("constant_velocity", "constant_acceleration"):
            expected = 0.8 * time if class_name == "constant_velocity" else 0.8 * time + 0.14 * time * time
            log_scores[class_name] = log(max(posterior[class_name], 1e-12)) - 0.5 * ((measurement - expected) / 0.25) ** 2
        pivot = max(log_scores.values())
        normalizer = pivot + log(sum(exp(value - pivot) for value in log_scores.values()))
        posterior = {name: exp(value - normalizer) for name, value in log_scores.items()}
        rows.append(
            {
                "time": time,
                "true_class_probability": posterior.get(shared.true_class, 0.0),
                "constant_velocity_probability": posterior["constant_velocity"],
                "constant_acceleration_probability": posterior["constant_acceleration"],
            }
        )
    return tuple(rows)


def _prediction_bundle(shared: SharedDynamicsTrajectory, *, prior: dict[str, float] | None = None) -> dict[str, object]:
    return {
        "pointwise": _pointwise_predict(shared, prior=prior),
        "accumulator": _accumulator_predict(shared, prior=prior),
        "windowed_raw": _windowed_predict(shared, robust=False, prior=prior),
        "windowed_robust": _windowed_predict(shared, robust=True, prior=prior),
        "kalman": _kalman_predict(shared, prior=prior),
    }


def _wrong_classification_score(shared: SharedDynamicsTrajectory) -> tuple[float, dict[str, object], dict[str, object]]:
    bundle = _prediction_bundle(shared)
    pointwise = bundle["pointwise"]
    accumulator = bundle["accumulator"]
    score = 0.0
    if pointwise.final_predicted_class != shared.true_class:
        score += 1.0
    score += max(0.0, 1.0 - float(pointwise.final_confidence))
    score += max(0.0, float(accumulator.final_confidence) - float(pointwise.final_confidence)) * 0.25
    details = {
        "true_class": shared.true_class,
        "pointwise_prediction": pointwise.final_predicted_class,
        "pointwise_confidence": pointwise.final_confidence,
        "accumulator_prediction": accumulator.final_predicted_class,
        "accumulator_confidence": accumulator.final_confidence,
    }
    payload = {
        "failure_mode": "wrong_classification",
        "trajectory_id": shared.trajectory_id,
        "posterior_trace": _accumulator_trace(shared),
    }
    return score, details, payload


def _high_entropy_score(shared: SharedDynamicsTrajectory) -> tuple[float, dict[str, object], dict[str, object]]:
    bundle = _prediction_bundle(shared)
    pointwise = bundle["pointwise"]
    accumulator = bundle["accumulator"]
    raw_entropy = _entropy(pointwise.final_weights, normalize_by_n=True)
    accumulator_entropy = _entropy(accumulator.final_weights, normalize_by_n=True)
    score = 0.5 * raw_entropy + 0.5 * accumulator_entropy
    details = {
        "pointwise_entropy": raw_entropy,
        "accumulator_entropy": accumulator_entropy,
        "pointwise_prediction": pointwise.final_predicted_class,
        "accumulator_prediction": accumulator.final_predicted_class,
    }
    payload = {
        "failure_mode": "high_entropy",
        "trajectory_id": shared.trajectory_id,
        "posterior_trace": _accumulator_trace(shared),
    }
    return score, details, payload


def _prior_flip_score(shared: SharedDynamicsTrajectory) -> tuple[float, dict[str, object], dict[str, object]]:
    sweep = _prior_sweep_predictions(shared)
    rows = sweep.rows
    reference = rows[2][1] if len(rows) >= 3 else rows[0][1]
    flipped = [row for row in rows if row[1] != reference]
    if flipped:
        smallest_shift = min(abs(row[0] - 0.5) for row in flipped)
        score = 1.0 - min(1.0, smallest_shift * 2.0)
    else:
        smallest_shift = 1.0
        score = 0.0
    details = {
        "reference_prediction": reference,
        "smallest_shift_to_flip": smallest_shift,
        "flip_count": len(flipped),
    }
    payload = {
        "failure_mode": "prior_flip",
        "trajectory_id": shared.trajectory_id,
        "sweep": sweep,
    }
    return score, details, payload


def _raw_extrema_failure_score(shared: SharedDynamicsTrajectory) -> tuple[float, dict[str, object], dict[str, object]]:
    raw = _windowed_predict(shared, robust=False)
    robust = _windowed_predict(shared, robust=True)
    score = 0.0
    if raw.final_predicted_class != shared.true_class:
        score += 1.0
    if robust.final_predicted_class == shared.true_class:
        score += 1.0
    score += max(0.0, float(robust.final_confidence) - float(raw.final_confidence))
    details = {
        "true_class": shared.true_class,
        "raw_prediction": raw.final_predicted_class,
        "raw_confidence": raw.final_confidence,
        "robust_prediction": robust.final_predicted_class,
        "robust_confidence": robust.final_confidence,
    }
    payload = {
        "failure_mode": "raw_extrema_failure",
        "times": shared.times,
        "measurements": shared.measurements,
        "raw_prediction": raw.final_predicted_class,
        "robust_prediction": robust.final_predicted_class,
    }
    return score, details, payload


def _irregular_window_failure_score(
    shared: SharedDynamicsTrajectory,
    sample_stats: dict[str, dict[str, float]],
    duration_stats: dict[str, dict[str, float]],
) -> tuple[float, dict[str, object], dict[str, object]]:
    features = SimpleNamespace(**_local_window_features(shared, robust=False))
    sample_pred, sample_conf = _classify_window_row(features, sample_stats)
    duration_pred, duration_conf = _classify_window_row(features, duration_stats)
    score = max(0.0, duration_conf - sample_conf)
    if duration_pred == shared.true_class and sample_pred != shared.true_class:
        score += 1.0
    details = {
        "true_class": shared.true_class,
        "sample_prediction": sample_pred,
        "sample_confidence": sample_conf,
        "duration_prediction": duration_pred,
        "duration_confidence": duration_conf,
    }
    payload = {
        "failure_mode": "irregular_window_failure",
        "times": shared.times,
        "measurements": shared.measurements,
    }
    return score, details, payload


def _kalman_mismatch_score(shared: SharedDynamicsTrajectory) -> tuple[float, dict[str, object], dict[str, object]]:
    kalman = _kalman_predict(shared)
    pointwise = _pointwise_predict(shared)
    score = 0.0
    if kalman.final_predicted_class != shared.true_class:
        score += 1.0
    score += max(0.0, float(pointwise.final_confidence) - float(kalman.final_confidence))
    details = {
        "true_class": shared.true_class,
        "kalman_prediction": kalman.final_predicted_class,
        "kalman_confidence": kalman.final_confidence,
        "pointwise_prediction": pointwise.final_predicted_class,
        "pointwise_confidence": pointwise.final_confidence,
    }
    payload = {
        "failure_mode": "kalman_mismatch",
        "times": shared.times,
        "measurements": shared.measurements,
    }
    return score, details, payload


def _transition_delay_candidates(
    *,
    seed: int,
    random_candidates: int,
    guided_candidates: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    scenarios = generate_transition_switching_scenarios(seed=seed, replicas=max(1, random_candidates + guided_candidates))
    specs = default_switching_mode_specs()
    transition_matrix = default_transition_matrix()
    rows: list[dict[str, object]] = []
    posterior_payloads: list[dict[str, object]] = []
    feature_payloads: list[dict[str, object]] = []
    for index, scenario in enumerate(scenarios[: max(1, random_candidates + guided_candidates)]):
        static_run = _run_mode_accumulator(scenario, specs, mode="static")
        transition_run = _run_mode_accumulator(scenario, specs, mode="transition", transition_matrix=transition_matrix)
        improvement = float(transition_run.post_switch_accuracy - static_run.post_switch_accuracy)
        score = max(0.0, improvement) + max(0.0, float(transition_run.accuracy) - float(static_run.accuracy)) * 0.5
        rows.append(
            {
                "failure_mode": "transition_delay",
                "search_method": "guided" if improvement >= 0.0 else "random",
                "candidate_id": scenario.trajectory_id,
                "scenario_id": scenario.scenario_name,
                "stress_score": score,
                "details": {
                    "static_post_switch_accuracy": static_run.post_switch_accuracy,
                    "transition_post_switch_accuracy": transition_run.post_switch_accuracy,
                    "accuracy_improvement": improvement,
                },
            }
        )
        if index == 0:
            posterior_payloads.append(
                {
                    "failure_mode": "transition_delay",
                    "trajectory_id": scenario.trajectory_id,
                    "posterior_trace": [
                        {
                            "time": step.time,
                            "constant_velocity_probability": step.posterior_weights.get("constant_velocity", 0.0),
                            "constant_acceleration_probability": step.posterior_weights.get("braking", 0.0)
                            + step.posterior_weights.get("maneuver", 0.0),
                        }
                        for step in transition_run.steps
                    ],
                }
            )
            feature_payloads.append(
                {
                    "failure_mode": "transition_delay",
                    "times": scenario.times,
                    "measurements": scenario.measurements,
                }
            )
    return rows, posterior_payloads, feature_payloads


def _wrong_classification_score(shared: SharedDynamicsTrajectory) -> tuple[float, dict[str, float], dict[str, object]]:
    pointwise = _pointwise_predict(shared)
    score = pointwise.final_confidence if pointwise.final_predicted_class != shared.true_class else 0.0
    payload = {
        "failure_mode": "wrong_classification",
        "posterior_trace": _accumulator_trace(shared),
    }
    return score, {"pointwise_confidence": pointwise.final_confidence}, payload


def _high_entropy_score(shared: SharedDynamicsTrajectory) -> tuple[float, dict[str, float], dict[str, object]]:
    posterior, details = _observable_pair_posterior(shared)
    entropy = _entropy(list(posterior.values()))
    payload = {
        "failure_mode": "high_entropy",
        "posterior_trace": _accumulator_trace(shared),
    }
    return entropy, {"entropy": entropy, **details}, payload


def _prior_flip_score(shared: SharedDynamicsTrajectory) -> tuple[float, dict[str, float], dict[str, object]]:
    sweep = _prior_sweep_predictions(shared)
    predicted_classes = {row[1] for row in sweep.rows}
    score = 1.0 if len(predicted_classes) > 1 else 0.0
    payload = {
        "failure_mode": "prior_flip",
        "sweep": sweep,
    }
    return score, {"num_distinct_predictions": float(len(predicted_classes))}, payload


def _raw_extrema_failure_score(shared: SharedDynamicsTrajectory) -> tuple[float, dict[str, float], dict[str, object]]:
    raw = _local_window_features(shared, robust=False)
    robust = _local_window_features(shared, robust=True)
    inflation = abs(raw["position_range"] - robust["position_range"])
    payload = {
        "failure_mode": "raw_extrema_failure",
        "times": list(shared.times),
        "measurements": list(shared.measurements),
    }
    return inflation, {"range_inflation": inflation}, payload


def _irregular_window_failure_score(
    shared: SharedDynamicsTrajectory,
    sample_stats: dict[str, dict[str, float]],
    duration_stats: dict[str, dict[str, float]],
) -> tuple[float, dict[str, float], dict[str, object]]:
    irregular = WindowRegimeTrajectory(
        trajectory_id=shared.trajectory_id,
        true_class=shared.true_class,
        sampling_regime="irregular",
        seed=int(getattr(shared, "seed", 0)),
        times=tuple(shared.times),
        measurements=tuple(shared.measurements),
        true_positions=tuple(shared.true_position),
    )
    sample_row = _sample_count_window(irregular, 5)
    duration_row = _duration_window(irregular, 5.0)
    sample_predicted, sample_conf = _classify_window_row(sample_row, sample_stats)
    duration_predicted, duration_conf = _classify_window_row(duration_row, duration_stats)
    score = max(0.0, duration_conf - sample_conf) if duration_predicted == shared.true_class else 0.0
    payload = {
        "failure_mode": "irregular_window_failure",
        "times": list(shared.times),
        "measurements": list(shared.measurements),
    }
    return score, {"sample_confidence": sample_conf, "duration_confidence": duration_conf}, payload


def _kalman_mismatch_score(shared: SharedDynamicsTrajectory) -> tuple[float, dict[str, float], dict[str, object]]:
    kalman = _kalman_predict(shared)
    pointwise = _pointwise_predict(shared)
    score = pointwise.final_confidence if kalman.final_predicted_class != shared.true_class and pointwise.final_predicted_class == shared.true_class else 0.0
    payload = {
        "failure_mode": "kalman_mismatch",
        "times": list(shared.times),
        "measurements": list(shared.measurements),
    }
    return score, {"kalman_confidence": kalman.final_confidence, "pointwise_confidence": pointwise.final_confidence}, payload


def _static_candidate_row(
    *,
    failure_mode: str,
    search_method: str,
    target: CorpusGymTarget,
    episode,
    score: float,
    details: dict[str, float],
) -> dict[str, object]:
    row = {
        "failure_mode": failure_mode,
        "search_method": search_method,
        "candidate_id": episode.trajectory.trajectory_id,
        "true_class": episode.trajectory.true_class,
        "tier_name": target.target_tier or "",
        "duration": float(episode.diagnostics.get("duration", 0.0) or 0.0),
        "acceleration_range": float(episode.diagnostics.get("acceleration_range", 0.0) or 0.0),
        "sampling_irregularity": float(episode.diagnostics.get("sampling_irregularity", 0.0) or 0.0),
        "outlier_score": float(episode.diagnostics.get("outlier_score", 0.0) or 0.0),
        "class_validity": float(episode.reward.class_validity),
        "feature_excitation": float(episode.reward.feature_excitation),
        "boundary_closeness": float(episode.reward.boundary_closeness),
        "classifier_stress": float(episode.reward.classifier_stress),
        "prior_sensitivity": float(episode.reward.prior_sensitivity),
        "leakage_penalty": float(episode.reward.leakage_penalty),
        "physical_invalidity_penalty": float(episode.reward.physical_invalidity_penalty),
        "total_utility": float(episode.reward.total_utility),
        "stress_score": float(score),
    }
    row.update(details)
    return row


def _transition_delay_candidates(
    *,
    seed: int,
    random_candidates: int,
    guided_candidates: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    del seed, random_candidates, guided_candidates
    return [], [], []
