from __future__ import annotations

from dataclasses import dataclass, replace
import csv
import io
import json
from math import exp, log
import os
from pathlib import Path
import random

import yaml

from .common_dataset_comparison import (
    SharedDynamicsTrajectory,
    _accumulator_predict,
    _kalman_predict,
    _pointwise_predict,
    _windowed_predict,
)
from .corpus_gym import CorpusGymAction, CorpusGymEnvironment, CorpusGymTarget
from .feature_analysis import _one_dimensional_feature_context_from_trajectory
from .irregular_window_comparison import (
    REGULAR_TIMES,
    WindowRegimeTrajectory,
    _duration_window,
    _gaussian_logpdf as _window_gaussian_logpdf,
    _normalize as _window_normalize,
    _sample_count_window,
    _true_positions,
    generate_window_regime_trajectories,
)
from .trajectory_generator import GeneratedTrajectoryDataset
from .transition_matrix_accumulator import (
    default_switching_mode_specs,
    default_transition_matrix,
    generate_transition_switching_scenarios,
    _run_mode_accumulator,
)


def _prepare_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp/kinematic-classifier-sandbox-mpl")))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _figure_to_png(fig) -> bytes:
    plt = _prepare_matplotlib()
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _union_fieldnames(rows: tuple[dict[str, object], ...] | list[dict[str, object]]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                ordered.append(key)
    return ordered


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _entropy(weights: dict[str, float]) -> float:
    total = 0.0
    for value in weights.values():
        if value > 1e-12:
            total -= value * log(value)
    return total / log(max(len(weights), 2))


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _least_squares_slope(times: list[float], values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_t = _mean(times)
    mean_y = _mean(values)
    denominator = sum((time - mean_t) ** 2 for time in times)
    if denominator <= 1e-9:
        return 0.0
    numerator = sum((time - mean_t) * (value - mean_y) for time, value in zip(times, values))
    return numerator / denominator


def _quadratic_fit(times: list[float], values: list[float]) -> tuple[float, float, float]:
    if len(times) < 3:
        return 0.0, _least_squares_slope(times, values), 0.0
    shifted = [time - times[0] for time in times]
    s1 = len(shifted)
    s_t = sum(shifted)
    s_t2 = sum(time * time for time in shifted)
    s_t3 = sum(time * time * time for time in shifted)
    s_t4 = sum(time * time * time * time for time in shifted)
    s_y = sum(values)
    s_ty = sum(time * value for time, value in zip(shifted, values))
    s_t2y = sum(time * time * value for time, value in zip(shifted, values))
    augmented = [
        [float(s1), s_t, s_t2, s_y],
        [s_t, s_t2, s_t3, s_ty],
        [s_t2, s_t3, s_t4, s_t2y],
    ]
    for pivot_index in range(3):
        pivot_row = max(range(pivot_index, 3), key=lambda row: abs(augmented[row][pivot_index]))
        if abs(augmented[pivot_row][pivot_index]) < 1e-12:
            return 0.0, _least_squares_slope(times, values), 0.0
        augmented[pivot_index], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_index]
        pivot = augmented[pivot_index][pivot_index]
        for col in range(pivot_index, 4):
            augmented[pivot_index][col] /= pivot
        for row in range(3):
            if row == pivot_index:
                continue
            factor = augmented[row][pivot_index]
            for col in range(pivot_index, 4):
                augmented[row][col] -= factor * augmented[pivot_index][col]
    return augmented[0][3], augmented[1][3], augmented[2][3]


def _median3(values: list[float], index: int) -> float:
    start = max(0, index - 1)
    stop = min(len(values), index + 2)
    window = sorted(values[start:stop])
    return window[len(window) // 2]


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


def _reference_window_stats() -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
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
    return stats["sample_count"], stats["duration"]


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


def _prior_sweep_predictions(shared: SharedDynamicsTrajectory) -> tuple[tuple[float, str, float], ...]:
    rows = []
    for prior_cv in (0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90):
        run = _accumulator_predict(
            shared,
            prior={"constant_velocity": prior_cv, "constant_acceleration": 1.0 - prior_cv},
        )
        rows.append((prior_cv, run.final_predicted_class, run.final_confidence))
    return tuple(rows)


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


@dataclass(frozen=True, slots=True)
class AdaptiveStressCorpusResult:
    config: dict[str, object]
    stress_case_rows: tuple[dict[str, object], ...]
    stress_score_rows: tuple[dict[str, object], ...]
    report_markdown: str
    posterior_trace_payloads: tuple[dict[str, object], ...]
    feature_trace_payloads: tuple[dict[str, object], ...]
    prior_flip_payloads: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class AdaptiveStressCorpusArtifacts:
    run_dir: Path
    config_path: Path
    stress_cases_path: Path
    stress_scores_path: Path
    report_path: Path
    posterior_timelines_path: Path
    feature_traces_path: Path
    prior_flip_examples_path: Path


def _static_candidate_row(
    *,
    failure_mode: str,
    search_method: str,
    target: CorpusGymTarget,
    episode,
    score: float,
    details: dict[str, object],
) -> dict[str, object]:
    context_dataset = GeneratedTrajectoryDataset(
        tier=str(episode.trajectory.generator_parameters.get("tier", target.target_tier or "")),
        seed=episode.trajectory.seed,
        class_definitions=(),
        tier_definition=None,  # type: ignore[arg-type]
        trajectories=(episode.trajectory,),
    )
    context = _one_dimensional_feature_context_from_trajectory(context_dataset, episode.trajectory)
    return {
        "failure_mode": failure_mode,
        "candidate_id": f"{failure_mode}_{search_method}_{episode.action.seed}",
        "search_method": search_method,
        "target_id": target.target_id,
        "trajectory_id": episode.trajectory.trajectory_id,
        "true_class": episode.trajectory.true_class,
        "seed": episode.action.seed,
        "tier_name": episode.action.tier_name,
        "duration_scale": episode.action.duration_scale,
        "measurement_scale": episode.action.measurement_scale,
        "irregularity_scale": episode.action.irregularity_scale,
        "outlier_scale": episode.action.outlier_scale,
        "step_scale": episode.action.step_scale,
        "class_validity": episode.reward.class_validity,
        "boundary_closeness": episode.reward.boundary_closeness,
        "classifier_stress": episode.reward.classifier_stress,
        "prior_sensitivity": episode.reward.prior_sensitivity,
        "leakage_penalty": episode.reward.leakage_penalty,
        "physical_invalidity_penalty": episode.reward.physical_invalidity_penalty,
        "total_utility": episode.reward.total_utility,
        "duration": context.duration,
        "acceleration_range": context.acceleration_range,
        "sampling_irregularity": context.sampling_irregularity,
        "outlier_score": context.outlier_score,
        "stress_score": score,
        **details,
    }


def _wrong_classification_score(shared: SharedDynamicsTrajectory) -> tuple[float, dict[str, object], dict[str, object]]:
    run = _pointwise_predict(shared)
    entropy = _entropy(run.final_weights)
    wrong = 1.0 if run.final_predicted_class != shared.true_class else 0.0
    score = _clamp(0.75 * wrong + 0.25 * entropy, 0.0, 1.0)
    trace = {
        "failure_mode": "wrong_classification",
        "trajectory_id": shared.trajectory_id,
        "times": shared.times,
        "measurements": shared.measurements,
        "posterior_trace": _accumulator_trace(shared),
    }
    return score, {
        "method_name": run.method_name,
        "predicted_class": run.final_predicted_class,
        "confidence": run.final_confidence,
        "posterior_entropy": entropy,
        "wrong_classification": wrong,
    }, trace


def _high_entropy_score(shared: SharedDynamicsTrajectory) -> tuple[float, dict[str, object], dict[str, object]]:
    run = _accumulator_predict(shared)
    observable_weights, diagnostics = _observable_pair_posterior(shared, prior_cv=0.5)
    classifier_entropy = _entropy(run.final_weights)
    observable_entropy = _entropy(observable_weights)
    disagreement = 1.0 if max(run.final_weights, key=run.final_weights.get) != max(observable_weights, key=observable_weights.get) else 0.0
    score = _clamp(0.50 * observable_entropy + 0.25 * classifier_entropy + 0.15 * disagreement + 0.10 * (1.0 - run.final_confidence), 0.0, 1.0)
    trace = {
        "failure_mode": "high_entropy",
        "trajectory_id": shared.trajectory_id,
        "times": shared.times,
        "measurements": shared.measurements,
        "posterior_trace": _accumulator_trace(shared),
    }
    return score, {
        "method_name": run.method_name,
        "predicted_class": run.final_predicted_class,
        "confidence": run.final_confidence,
        "posterior_entropy": classifier_entropy,
        "observable_entropy": observable_entropy,
        "observable_prediction": max(observable_weights, key=observable_weights.get),
        "observable_confidence": max(observable_weights.values()),
        **diagnostics,
    }, trace


def _prior_flip_score(shared: SharedDynamicsTrajectory) -> tuple[float, dict[str, object], dict[str, object]]:
    sweep = []
    for prior_cv in (0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90):
        weights, _ = _observable_pair_posterior(shared, prior_cv=prior_cv)
        predicted = max(weights, key=weights.get)
        sweep.append((prior_cv, predicted, max(weights.values())))
    sweep = tuple(sweep)
    baseline_prediction = next(predicted for prior, predicted, _ in sweep if abs(prior - 0.5) < 1e-9)
    smallest_shift = 1.0
    for prior_cv, predicted, _ in sweep:
        if predicted != baseline_prediction:
            smallest_shift = min(smallest_shift, abs(prior_cv - 0.5))
    flipped = 1.0 if smallest_shift < 1.0 else 0.0
    observable_weights, diagnostics = _observable_pair_posterior(shared, prior_cv=0.5)
    entropy_bonus = _entropy(observable_weights)
    score = _clamp((0.80 * (1.0 - smallest_shift / 0.4) + 0.20 * entropy_bonus) if flipped else 0.20 * entropy_bonus, 0.0, 1.0)
    payload = {
        "failure_mode": "prior_flip",
        "trajectory_id": shared.trajectory_id,
        "sweep": sweep,
    }
    return score, {
        "baseline_prediction": baseline_prediction,
        "smallest_prior_shift": smallest_shift if flipped else None,
        "prior_flip_detected": flipped,
        "confidence_at_uniform_prior": next(conf for prior, _, conf in sweep if abs(prior - 0.5) < 1e-9),
        "observable_entropy": entropy_bonus,
        **diagnostics,
    }, payload


def _raw_extrema_failure_score(shared: SharedDynamicsTrajectory) -> tuple[float, dict[str, object], dict[str, object]]:
    raw = _windowed_predict(shared, robust=False)
    robust = _windowed_predict(shared, robust=True)
    raw_features = _local_window_features(shared, robust=False)
    robust_features = _local_window_features(shared, robust=True)
    raw_wrong = 1.0 if raw.final_predicted_class != shared.true_class else 0.0
    robust_correct = 1.0 if robust.final_predicted_class == shared.true_class else 0.0
    range_inflation = _clamp((raw_features["position_range"] - robust_features["position_range"]) / 1.2, 0.0, 1.0)
    curvature_inflation = _clamp(abs(raw_features["quadratic_proxy"] - robust_features["quadratic_proxy"]) / 1.2, 0.0, 1.0)
    score = _clamp(
        0.40 * raw_wrong
        + 0.20 * robust_correct
        + 0.20 * range_inflation
        + 0.10 * curvature_inflation
        + 0.10 * (1.0 if raw.final_predicted_class != robust.final_predicted_class else 0.0),
        0.0,
        1.0,
    )
    payload = {
        "failure_mode": "raw_extrema_failure",
        "trajectory_id": shared.trajectory_id,
        "times": shared.times,
        "measurements": shared.measurements,
        "true_class": shared.true_class,
    }
    return score, {
        "raw_prediction": raw.final_predicted_class,
        "robust_prediction": robust.final_predicted_class,
        "raw_confidence": raw.final_confidence,
        "robust_confidence": robust.final_confidence,
        "raw_wrong": raw_wrong,
        "robust_correct": robust_correct,
        "raw_range": raw_features["position_range"],
        "robust_range": robust_features["position_range"],
        "raw_curvature_proxy": raw_features["quadratic_proxy"],
        "robust_curvature_proxy": robust_features["quadratic_proxy"],
        "range_inflation": range_inflation,
        "curvature_inflation": curvature_inflation,
    }, payload


def _irregular_window_failure_score(shared: SharedDynamicsTrajectory, sample_stats, duration_stats) -> tuple[float, dict[str, object], dict[str, object]]:
    truth = tuple(value for value in (shared.true_position or shared.measurements))
    trajectory = WindowRegimeTrajectory(
        trajectory_id=shared.trajectory_id,
        true_class=shared.true_class,
        sampling_regime="irregular",
        seed=shared.seed,
        times=shared.times,
        measurements=shared.measurements,
        true_positions=truth,
    )
    sample_row = _sample_count_window(trajectory, 5)
    duration_row = _duration_window(trajectory, 5.0)
    sample_prediction, sample_confidence = _classify_window_row(sample_row, sample_stats)
    duration_prediction, duration_confidence = _classify_window_row(duration_row, duration_stats)
    sample_wrong = 1.0 if sample_prediction != shared.true_class else 0.0
    duration_correct = 1.0 if duration_prediction == shared.true_class else 0.0
    score = _clamp(0.60 * sample_wrong + 0.30 * duration_correct + 0.10 * (1.0 if sample_prediction != duration_prediction else 0.0), 0.0, 1.0)
    payload = {
        "failure_mode": "irregular_window_failure",
        "trajectory_id": shared.trajectory_id,
        "times": shared.times,
        "measurements": shared.measurements,
        "sample_count_window_duration": sample_row.duration,
        "duration_window_duration": duration_row.duration,
    }
    return score, {
        "sample_count_prediction": sample_prediction,
        "duration_prediction": duration_prediction,
        "sample_count_confidence": sample_confidence,
        "duration_confidence": duration_confidence,
        "sample_count_wrong": sample_wrong,
        "duration_correct": duration_correct,
    }, payload


def _kalman_mismatch_score(shared: SharedDynamicsTrajectory) -> tuple[float, dict[str, object], dict[str, object]]:
    kalman = _kalman_predict(shared)
    accumulator = _accumulator_predict(shared)
    robust = _windowed_predict(shared, robust=True)
    helpers_correct = max(
        1.0 if accumulator.final_predicted_class == shared.true_class else 0.0,
        1.0 if robust.final_predicted_class == shared.true_class else 0.0,
    )
    kalman_wrong = 1.0 if kalman.final_predicted_class != shared.true_class else 0.0
    score = _clamp(0.65 * kalman_wrong + 0.25 * helpers_correct + 0.10 * (1.0 - kalman.final_confidence), 0.0, 1.0)
    payload = {
        "failure_mode": "kalman_mismatch",
        "trajectory_id": shared.trajectory_id,
        "times": shared.times,
        "measurements": shared.measurements,
        "true_class": shared.true_class,
    }
    return score, {
        "kalman_prediction": kalman.final_predicted_class,
        "accumulator_prediction": accumulator.final_predicted_class,
        "robust_window_prediction": robust.final_predicted_class,
        "kalman_confidence": kalman.final_confidence,
        "accumulator_confidence": accumulator.final_confidence,
        "robust_window_confidence": robust.final_confidence,
        "kalman_wrong": kalman_wrong,
        "helper_correct": helpers_correct,
    }, payload


def _transition_delay_candidates(*, seed: int, random_candidates: int, guided_candidates: int) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    specs = default_switching_mode_specs()
    transition_matrix = default_transition_matrix()
    all_rows: list[dict[str, object]] = []
    posterior_payloads: list[dict[str, object]] = []
    feature_payloads: list[dict[str, object]] = []
    scenario_pool = generate_transition_switching_scenarios(seed=seed, replicas=max(random_candidates + guided_candidates, 10))
    for index, scenario in enumerate(scenario_pool[: random_candidates + guided_candidates]):
        transition_run = _run_mode_accumulator(scenario, specs, mode="transition", transition_matrix=transition_matrix)
        static_run = _run_mode_accumulator(scenario, specs, mode="static", transition_matrix=None)
        switch_index = next((i for i in range(1, len(scenario.true_mode_by_step)) if scenario.true_mode_by_step[i] != scenario.true_mode_by_step[i - 1]), len(scenario.true_mode_by_step) - 1)
        target_mode = scenario.true_mode_by_step[switch_index]
        transition_recovery_time = scenario.times[-1] - scenario.times[switch_index]
        for step in transition_run.steps[switch_index:]:
            if step.predicted_mode == target_mode:
                transition_recovery_time = step.time - scenario.times[switch_index]
                break
        max_horizon = max(scenario.times[-1] - scenario.times[switch_index], 1e-9)
        delay_score = _clamp(transition_recovery_time / max_horizon, 0.0, 1.0)
        row = {
            "failure_mode": "transition_delay",
            "candidate_id": f"transition_delay_{index}",
            "search_method": "random" if index < random_candidates else "guided",
            "target_id": "stress_transition_delay",
            "trajectory_id": scenario.trajectory_id,
            "true_class": "->".join((scenario.true_mode_by_step[0], target_mode)),
            "seed": scenario.seed,
            "tier_name": "switching_scenarios_v1",
            "duration_scale": "",
            "measurement_scale": "",
            "irregularity_scale": "",
            "outlier_scale": "",
            "step_scale": "",
            "class_validity": 1.0,
            "boundary_closeness": 0.0,
            "classifier_stress": 0.0,
            "prior_sensitivity": 0.0,
            "leakage_penalty": 0.0,
            "physical_invalidity_penalty": 0.0,
            "total_utility": 0.0,
            "duration": scenario.times[-1] - scenario.times[0],
            "acceleration_range": 0.0,
            "sampling_irregularity": 0.0,
            "outlier_score": 0.0,
            "stress_score": delay_score,
            "predicted_class": transition_run.final_predicted_mode,
            "confidence": max(transition_run.final_weights.values()),
            "switch_delay": transition_recovery_time,
            "transition_post_switch_accuracy": transition_run.post_switch_accuracy,
            "static_post_switch_accuracy": static_run.post_switch_accuracy,
        }
        all_rows.append(row)
        posterior_payloads.append(
            {
                "failure_mode": "transition_delay",
                "trajectory_id": scenario.trajectory_id,
                "times": tuple(step.time for step in transition_run.steps),
                "posterior_trace": tuple(
                    {
                        "time": step.time,
                        "true_class_probability": step.posterior_weights.get(target_mode, 0.0),
                        "constant_velocity_probability": step.posterior_weights.get("constant_velocity", 0.0),
                        "constant_acceleration_probability": step.posterior_weights.get("braking", 0.0),
                    }
                    for step in transition_run.steps
                ),
            }
        )
        feature_payloads.append(
            {
                "failure_mode": "transition_delay",
                "trajectory_id": scenario.trajectory_id,
                "times": scenario.times,
                "measurements": scenario.measurements,
                "true_class": "transition",
            }
        )
    selected = sorted(all_rows, key=lambda row: float(row["stress_score"]), reverse=True)[:2]
    return all_rows, posterior_payloads, feature_payloads + selected  # type: ignore[operator]


def analyze_adaptive_stress_corpus(
    *,
    seed: int = 7,
    random_candidates_per_mode: int = 8,
    guided_candidates_per_mode: int = 14,
) -> AdaptiveStressCorpusResult:
    rng = random.Random(seed)
    environment = CorpusGymEnvironment()
    sample_stats, duration_stats = _reference_window_stats()

    score_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    posterior_payloads: list[dict[str, object]] = []
    feature_payloads: list[dict[str, object]] = []
    prior_flip_payloads: list[dict[str, object]] = []

    failure_modes = {
        "wrong_classification": _wrong_classification_score,
        "high_entropy": _high_entropy_score,
        "prior_flip": _prior_flip_score,
        "raw_extrema_failure": _raw_extrema_failure_score,
        "irregular_window_failure": lambda shared: _irregular_window_failure_score(shared, sample_stats, duration_stats),
        "kalman_mismatch": _kalman_mismatch_score,
    }
    targets = _stress_targets()

    for target in targets:
        mode_name = target.target_id.removeprefix("stress_")
        evaluator = failure_modes[mode_name]
        mode_rows: list[dict[str, object]] = []
        for candidate_index in range(random_candidates_per_mode + guided_candidates_per_mode):
            target_for_candidate = target
            if target.class_pair is not None:
                target_for_candidate = replace(
                    target,
                    class_name=target.class_pair[candidate_index % len(target.class_pair)],
                )
            search_method = "random" if candidate_index < random_candidates_per_mode else "guided"
            action_seed = seed * 100_000 + len(score_rows) * 17 + candidate_index
            action = (
                _random_action(rng, target_for_candidate, seed=action_seed)
                if search_method == "random"
                else _guided_action(rng, mode_name, target_for_candidate, seed=action_seed)
            )
            environment.reset(target_for_candidate)
            episode = environment.simulate(action)
            if episode.reward.class_validity < 0.45:
                continue
            if episode.trajectory.true_class not in ("constant_velocity", "constant_acceleration"):
                continue
            shared = _to_shared_trajectory(episode.trajectory)
            stress_score, details, payload = evaluator(shared)
            row = _static_candidate_row(
                failure_mode=mode_name,
                search_method=search_method,
                target=target,
                episode=episode,
                score=stress_score,
                details=details,
            )
            mode_rows.append(row)
            score_rows.append(row)
            if mode_name in ("wrong_classification", "high_entropy", "transition_delay"):
                posterior_payloads.append(payload)
            if mode_name in ("raw_extrema_failure", "irregular_window_failure", "kalman_mismatch"):
                feature_payloads.append(payload)
            if mode_name == "prior_flip":
                prior_flip_payloads.append(payload)
        selected_rows.extend(sorted(mode_rows, key=lambda row: float(row["stress_score"]), reverse=True)[:2])

    transition_rows, transition_posteriors, transition_features = _transition_delay_candidates(
        seed=seed + 101,
        random_candidates=random_candidates_per_mode,
        guided_candidates=guided_candidates_per_mode,
    )
    score_rows.extend(transition_rows)
    selected_rows.extend(sorted(transition_rows, key=lambda row: float(row["stress_score"]), reverse=True)[:2])
    posterior_payloads.extend(transition_posteriors[:2])

    selected_rows.sort(key=lambda row: (str(row["failure_mode"]), -float(row["stress_score"])))
    report_lines = [
        "# Adaptive Stress Corpus",
        "",
        "This artifact runs the first failure-targeted corpus search layer on top of CorpusGym and the existing classifier/filter diagnostics.",
        "",
        "## Summary",
        "",
    ]
    for failure_mode in (
        "wrong_classification",
        "high_entropy",
        "prior_flip",
        "raw_extrema_failure",
        "irregular_window_failure",
        "kalman_mismatch",
        "transition_delay",
    ):
        rows = [row for row in score_rows if row["failure_mode"] == failure_mode]
        random_rows = [row for row in rows if row["search_method"] == "random"]
        guided_rows = [row for row in rows if row["search_method"] == "guided"]
        mean_random = sum(float(row["stress_score"]) for row in random_rows) / max(len(random_rows), 1)
        best_guided = max((float(row["stress_score"]) for row in guided_rows), default=0.0)
        status = "resolved" if best_guided > mean_random else "not_yet_resolved"
        report_lines.append(f"- `{failure_mode}`: random mean `{mean_random:.3f}`, best guided `{best_guided:.3f}`, status `{status}`")
    report_lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Wrong-classification, entropy, prior-flip, raw-extrema, irregular-window, and Kalman-mismatch cases are scored on generated CV/CA trajectories using existing repo comparators.",
            "- Transition-delay cases are searched over switching scenarios using the transition-matrix accumulator itself, because the current CorpusGym environment is still single-trajectory rather than switching-sequence native.",
            "- Guided search currently means failure-mode-shaped parameter sampling plus rejection and selection, not RL.",
            "- `high_entropy`, `prior_flip`, and `raw_extrema_failure` now use richer observables than the original shared-classifier hooks: local quadratic acceleration evidence for ambiguity and prior flips, plus explicit raw-versus-robust feature inflation for extrema stress.",
        ]
    )
    config = {
        "search_id": "adaptive_stress_corpus_v1",
        "seed": seed,
        "random_candidates_per_mode": random_candidates_per_mode,
        "guided_candidates_per_mode": guided_candidates_per_mode,
        "failure_modes": [
            "wrong_classification",
            "high_entropy",
            "prior_flip",
            "raw_extrema_failure",
            "irregular_window_failure",
            "kalman_mismatch",
            "transition_delay",
        ],
    }
    return AdaptiveStressCorpusResult(
        config=config,
        stress_case_rows=tuple(selected_rows),
        stress_score_rows=tuple(score_rows),
        report_markdown="\n".join(report_lines),
        posterior_trace_payloads=tuple(posterior_payloads),
        feature_trace_payloads=tuple(feature_payloads),
        prior_flip_payloads=tuple(prior_flip_payloads),
    )


def _plot_posterior_timelines(result: AdaptiveStressCorpusResult):
    plt = _prepare_matplotlib()
    selected_modes = ("wrong_classification", "high_entropy", "transition_delay")
    payloads = []
    for mode in selected_modes:
        payload = next((item for item in result.posterior_trace_payloads if item["failure_mode"] == mode), None)
        if payload is not None:
            payloads.append(payload)
    fig, axes = plt.subplots(1, max(1, len(payloads)), figsize=(5.0 * max(1, len(payloads)), 4.0))
    if hasattr(axes, "ravel"):
        axes = list(axes.ravel())
    elif not isinstance(axes, (list, tuple)):
        axes = [axes]
    for ax, payload in zip(axes, payloads):
        trace = payload["posterior_trace"]
        ax.plot([row["time"] for row in trace], [row["constant_velocity_probability"] for row in trace], label="P(CV)", color="#2563eb")
        ax.plot([row["time"] for row in trace], [row["constant_acceleration_probability"] for row in trace], label="P(CA/new mode)", color="#dc2626")
        ax.set_title(str(payload["failure_mode"]).replace("_", " "), loc="left", fontsize=11, fontweight="bold")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Stress Case Posterior Timelines", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def _plot_feature_traces(result: AdaptiveStressCorpusResult):
    plt = _prepare_matplotlib()
    selected_modes = ("raw_extrema_failure", "irregular_window_failure", "kalman_mismatch")
    payloads = []
    for mode in selected_modes:
        payload = next((item for item in result.feature_trace_payloads if item["failure_mode"] == mode), None)
        if payload is not None:
            payloads.append(payload)
    fig, axes = plt.subplots(1, max(1, len(payloads)), figsize=(5.0 * max(1, len(payloads)), 4.0))
    if hasattr(axes, "ravel"):
        axes = list(axes.ravel())
    elif not isinstance(axes, (list, tuple)):
        axes = [axes]
    for ax, payload in zip(axes, payloads):
        ax.plot(payload["times"], payload["measurements"], marker="o", linewidth=1.5, color="#111827")
        ax.set_title(str(payload["failure_mode"]).replace("_", " "), loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("time")
        ax.set_ylabel("measurement")
        ax.grid(True, alpha=0.25)
    fig.suptitle("Stress Case Feature Traces", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def _plot_prior_flip_examples(result: AdaptiveStressCorpusResult):
    plt = _prepare_matplotlib()
    payload = next(iter(result.prior_flip_payloads), None)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    if payload is not None:
        priors = [row[0] for row in payload["sweep"]]
        confidences = [row[2] for row in payload["sweep"]]
        predicted = [row[1] for row in payload["sweep"]]
        colors = ["#2563eb" if name == "constant_velocity" else "#dc2626" for name in predicted]
        ax.scatter(priors, confidences, c=colors, s=50)
        ax.plot(priors, confidences, color="#6b7280", alpha=0.5)
    ax.set_title("Prior Flip Examples", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("prior P(CV)")
    ax.set_ylabel("final confidence")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def write_adaptive_stress_corpus_artifacts(
    output_dir: str | Path,
    *,
    result: AdaptiveStressCorpusResult | None = None,
) -> AdaptiveStressCorpusArtifacts:
    analysis = result or analyze_adaptive_stress_corpus()
    run_dir = Path(output_dir) / "adaptive_stress_corpus"
    plots_dir = run_dir / "plots"
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "stress_search_config.yaml"
    stress_cases_path = run_dir / "stress_cases.csv"
    stress_scores_path = run_dir / "stress_case_scores.csv"
    report_path = run_dir / "stress_case_report.md"
    posterior_timelines_path = plots_dir / "stress_case_posterior_timelines.png"
    feature_traces_path = plots_dir / "stress_case_feature_traces.png"
    prior_flip_examples_path = plots_dir / "prior_flip_examples.png"

    config_path.write_text(yaml.safe_dump(analysis.config, sort_keys=False), encoding="utf-8")
    if analysis.stress_case_rows:
        _write_csv(stress_cases_path, list(analysis.stress_case_rows), _union_fieldnames(analysis.stress_case_rows))
    else:
        _write_csv(stress_cases_path, [], ["failure_mode", "candidate_id"])
    if analysis.stress_score_rows:
        _write_csv(stress_scores_path, list(analysis.stress_score_rows), _union_fieldnames(analysis.stress_score_rows))
    else:
        _write_csv(stress_scores_path, [], ["failure_mode", "candidate_id"])
    report_path.write_text(analysis.report_markdown, encoding="utf-8")
    posterior_timelines_path.write_bytes(_figure_to_png(_plot_posterior_timelines(analysis)))
    feature_traces_path.write_bytes(_figure_to_png(_plot_feature_traces(analysis)))
    prior_flip_examples_path.write_bytes(_figure_to_png(_plot_prior_flip_examples(analysis)))

    return AdaptiveStressCorpusArtifacts(
        run_dir=run_dir,
        config_path=config_path,
        stress_cases_path=stress_cases_path,
        stress_scores_path=stress_scores_path,
        report_path=report_path,
        posterior_timelines_path=posterior_timelines_path,
        feature_traces_path=feature_traces_path,
        prior_flip_examples_path=prior_flip_examples_path,
    )
