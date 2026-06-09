from __future__ import annotations

import random
from dataclasses import replace
from math import log
from typing import NamedTuple

from ..analysis.common_dataset_comparison import SharedDynamicsTrajectory
from ..analysis.feature_analysis import _one_dimensional_feature_context_from_trajectory
from ..inference.irregular_window_comparison import (
    WindowRegimeTrajectory,
    _duration_window,
    _sample_count_window,
    generate_window_regime_trajectories,
)
from ..inference.irregular_window_comparison import _gaussian_logpdf as _window_gaussian_logpdf
from ..inference.irregular_window_comparison import _normalize as _window_normalize
from ..trajectory_generator import GeneratedTrajectoryDataset
from ..utils.math import _least_squares_slope, _median3, _quadratic_fit
from .gym import CorpusGymAction, CorpusGymTarget


class ReferenceWindowStats(NamedTuple):
    sample_count: dict[str, dict[str, float]]
    duration: dict[str, dict[str, float]]


class WindowClassification(NamedTuple):
    predicted_class: str
    confidence: float


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


def _classify_window_row(row, stats: dict[str, dict[str, float]]) -> WindowClassification:
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
    return WindowClassification(predicted_class=predicted, confidence=weights[predicted])
