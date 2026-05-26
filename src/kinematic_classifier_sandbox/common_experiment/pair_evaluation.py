from __future__ import annotations

import statistics
from typing import Callable, NamedTuple
from typing import NamedTuple

from kinematic_classifier_sandbox.utils.math import (
    _quadratic_fit,
)
from kinematic_classifier_sandbox.utils.math import (
    gaussian_logpdf as _gaussian_logpdf,
)
from kinematic_classifier_sandbox.utils.math import (
    linear_fit as _linear_fit,
)
from kinematic_classifier_sandbox.utils.math import (
    mean as _mean,
)
from kinematic_classifier_sandbox.utils.math import (
    median3 as _median3,
)
from kinematic_classifier_sandbox.utils.math import (
    normalize_log_scores as _normalize_scores,
)
from kinematic_classifier_sandbox.utils.math import (
    safe_log as _safe_log,
)

from ..analysis.feature_analysis import load_feature_set_manifest, resolve_feature_names
from ..corpus.coverage_report import load_classifier_manifest
from .adapters import ExecutablePairSpec, ExecutableTrajectory, build_reference_trajectory
from .config import CommonExperimentConfig
from .contracts import (
    FamilyScoringContext,
    FeatureTableRow,
    LikelihoodHistoryRow,
    PairPredictionRow,
    PosteriorHistoryRow,
)
from .scoring import score_classifier_family


class TruncatedTrajectory(NamedTuple):
    times: tuple[float, ...]
    trajectory: ExecutableTrajectory


class WindowFeatureScores(NamedTuple):
    scores: dict[str, float]
    observed: dict[str, float]
    selected_count: int
    selected_duration: float


def trajectory_features(
    trajectory: ExecutableTrajectory,
    *,
    robust: bool,
) -> dict[str, float]:
    times = list(trajectory.times)
    measurements = list(trajectory.measurements)
    if robust and len(measurements) >= 3:
        measurements = [_median3(measurements, index) for index in range(len(measurements))]
    duration = times[-1] - times[0] if len(times) >= 2 else 0.0
    position_range = max(measurements) - min(measurements) if measurements else 0.0
    velocity = [
        (measurements[index] - measurements[index - 1]) / max(times[index] - times[index - 1], 1e-9)
        for index in range(1, len(times))
    ]
    acceleration = [
        (velocity[index] - velocity[index - 1]) / max(times[index + 1] - times[index], 1e-9)
        for index in range(1, len(velocity))
    ]
    speed_range = (max(velocity) - min(velocity)) if velocity else 0.0
    acceleration_range = (max(acceleration) - min(acceleration)) if acceleration else 0.0
    acceleration_variance = statistics.pvariance(acceleration) if len(acceleration) >= 2 else 0.0
    _, slope = _linear_fit(times, measurements)
    intercept_q, slope_q, curvature = _quadratic_fit(times, measurements)
    linear_residual = (
        sum((value - (intercept_q + slope * time)) ** 2 for time, value in zip(times, measurements)) / max(len(times), 1)
    ) ** 0.5
    quadratic_residual = (
        sum((value - (intercept_q + slope_q * (time - times[0]) + curvature * (time - times[0]) ** 2)) ** 2 for time, value in zip(times, measurements))
        / max(len(times), 1)
    ) ** 0.5
    monotonicity = sum(1 for left, right in zip(measurements, measurements[1:]) if right >= left) / max(len(measurements) - 1, 1)
    median_measurement = statistics.median(measurements) if measurements else 0.0
    outlier_score = max(abs(value - median_measurement) for value in measurements) if measurements else 0.0
    return {
        "duration": duration,
        "position_range": position_range,
        "speed_range": speed_range,
        "acceleration_range": acceleration_range,
        "acceleration_variance": acceleration_variance,
        "curvature_proxy": 2.0 * curvature,
        "velocity_sign_changes": float(_count_sign_changes(velocity)),
        "acceleration_sign_changes": float(_count_sign_changes(acceleration)),
        "monotonicity": monotonicity,
        "linear_fit_residual": linear_residual,
        "quadratic_fit_residual": quadratic_residual,
        "outlier_score": outlier_score,
    }


def reference_trajectory(
    pair_spec: ExecutablePairSpec,
    class_name: str,
    scenario_id: str,
    times: tuple[float, ...],
) -> ExecutableTrajectory:
    return build_reference_trajectory(pair_spec, class_name, scenario_id, times)


def feature_sigma(feature_name: str) -> float:
    return {
        "duration": 0.50,
        "position_range": 0.50,
        "speed_range": 0.35,
        "acceleration_range": 0.35,
        "acceleration_variance": 0.06,
        "curvature_proxy": 0.22,
        "velocity_sign_changes": 0.90,
        "acceleration_sign_changes": 0.90,
        "monotonicity": 0.12,
        "linear_fit_residual": 0.25,
        "quadratic_fit_residual": 0.25,
        "outlier_score": 0.55,
    }.get(feature_name, 0.30)


def pair_priors(class_a: str, class_b: str, prior_id: str) -> dict[str, float]:
    if prior_id == "uniform":
        return {class_a: 0.5, class_b: 0.5}
    if prior_id == "mild_bias":
        return {class_a: 0.65, class_b: 0.35}
    if prior_id == "strong_bias":
        return {class_a: 0.85, class_b: 0.15}
    raise KeyError(prior_id)


def truncated_trajectory(
    trajectory: ExecutableTrajectory,
    prefix_length: int,
) -> TruncatedTrajectory:
    times = trajectory.times[:prefix_length]
    truncated = ExecutableTrajectory(
        trajectory_id=trajectory.trajectory_id,
        class_pair_id=trajectory.class_pair_id,
        class_a=trajectory.class_a,
        class_b=trajectory.class_b,
        true_class=trajectory.true_class,
        scenario_id=trajectory.scenario_id,
        seed=trajectory.seed,
        times=times,
        measurements=trajectory.measurements[:prefix_length],
        true_position=trajectory.true_position[:prefix_length],
        true_velocity=trajectory.true_velocity[:prefix_length],
        true_acceleration=trajectory.true_acceleration[:prefix_length],
    )
    return TruncatedTrajectory(times=times, trajectory=truncated)


def classifier_scores_for_prefix(
    classifier_entry: dict[str, object],
    pair_spec: ExecutablePairSpec,
    trajectory: ExecutableTrajectory,
    prefix_length: int,
    prior_weights: dict[str, float],
    feature_manifest: dict[str, dict[str, object]],
) -> dict[str, float]:
    truncated_bundle = truncated_trajectory(trajectory, prefix_length)
    times = truncated_bundle.times
    truncated = truncated_bundle.trajectory
    context = FamilyScoringContext(
        pair_spec=pair_spec,
        trajectory=trajectory,
        truncated=truncated,
        times=times,
        prior_weights=prior_weights,
        feature_manifest=feature_manifest,
        reference_builder=reference_trajectory,
        feature_extractor=trajectory_features,
        feature_sigma=feature_sigma,
        gaussian_logpdf=_gaussian_logpdf,
        safe_log=_safe_log,
    )
    return score_classifier_family(classifier_entry, context)


def feature_set_scores_for_prefix(
    *,
    feature_set_id: str,
    feature_entry: dict[str, object],
    pair_spec: ExecutablePairSpec,
    trajectory: ExecutableTrajectory,
    prefix_length: int,
    prior_weights: dict[str, float],
) -> dict[str, float]:
    times = trajectory.times[:prefix_length]
    truncated = ExecutableTrajectory(
        trajectory_id=trajectory.trajectory_id,
        class_pair_id=trajectory.class_pair_id,
        class_a=trajectory.class_a,
        class_b=trajectory.class_b,
        true_class=trajectory.true_class,
        scenario_id=trajectory.scenario_id,
        seed=trajectory.seed,
        times=times,
        measurements=trajectory.measurements[:prefix_length],
        true_position=trajectory.true_position[:prefix_length],
        true_velocity=trajectory.true_velocity[:prefix_length],
        true_acceleration=trajectory.true_acceleration[:prefix_length],
    )
    features = feature_entry.get("features")
    if not isinstance(features, list) or not features:
        raise KeyError(feature_set_id)
    robust = feature_set_id == "robust_extrema"
    observed = trajectory_features(truncated, robust=robust)
    scores: dict[str, float] = {}
    for class_name in (pair_spec.class_a, pair_spec.class_b):
        reference = reference_trajectory(pair_spec, class_name, trajectory.scenario_id, times)
        reference_features = trajectory_features(reference, robust=robust)
        score = _safe_log(prior_weights[class_name])
        for feature_name in features:
            name = str(feature_name)
            score += _gaussian_logpdf(observed[name], reference_features[name], feature_sigma(name))
        scores[class_name] = score
    return scores


def slice_trailing_window(
    trajectory: ExecutableTrajectory,
    *,
    window_definition: str,
    window_sample_count: int,
    window_duration: float,
) -> ExecutableTrajectory:
    if window_definition == "sample_count":
        start = max(0, len(trajectory.times) - window_sample_count)
        selected_indices = list(range(start, len(trajectory.times)))
    elif window_definition == "elapsed_time":
        threshold = trajectory.times[-1] - window_duration
        selected_indices = [index for index, time in enumerate(trajectory.times) if time >= threshold]
        if not selected_indices:
            selected_indices = [len(trajectory.times) - 1]
    else:
        raise KeyError(window_definition)

    return ExecutableTrajectory(
        trajectory_id=trajectory.trajectory_id,
        class_pair_id=trajectory.class_pair_id,
        class_a=trajectory.class_a,
        class_b=trajectory.class_b,
        true_class=trajectory.true_class,
        scenario_id=trajectory.scenario_id,
        seed=trajectory.seed,
        times=tuple(trajectory.times[index] for index in selected_indices),
        measurements=tuple(trajectory.measurements[index] for index in selected_indices),
        true_position=tuple(trajectory.true_position[index] for index in selected_indices),
        true_velocity=tuple(trajectory.true_velocity[index] for index in selected_indices),
        true_acceleration=tuple(trajectory.true_acceleration[index] for index in selected_indices),
        measurement_dim=trajectory.measurement_dim,
        coordinate_frame=trajectory.coordinate_frame,
    )


def feature_set_scores_for_window(
    *,
    feature_set_id: str,
    feature_manifest: dict[str, dict[str, object]],
    pair_spec: ExecutablePairSpec,
    trajectory: ExecutableTrajectory,
    window_definition: str,
    window_sample_count: int,
    window_duration: float,
    prior_weights: dict[str, float],
) -> WindowFeatureScores:
    truncated = slice_trailing_window(
        trajectory,
        window_definition=window_definition,
        window_sample_count=window_sample_count,
        window_duration=window_duration,
    )
    features = resolve_feature_names(feature_set=feature_set_id, manifest=feature_manifest)
    robust = feature_set_id == "robust_extrema"
    observed = trajectory_features(truncated, robust=robust)
    scores: dict[str, float] = {}
    for class_name in (pair_spec.class_a, pair_spec.class_b):
        reference = reference_trajectory(pair_spec, class_name, trajectory.scenario_id, truncated.times)
        reference_features = trajectory_features(reference, robust=robust)
        score = _safe_log(prior_weights[class_name])
        for feature_name in features:
            score += _gaussian_logpdf(observed[feature_name], reference_features[feature_name], feature_sigma(feature_name))
        scores[class_name] = score
    selected_duration = truncated.times[-1] - truncated.times[0] if len(truncated.times) >= 2 else 0.0
    return WindowFeatureScores(
        scores=scores,
        observed=observed,
        selected_count=len(truncated.times),
        selected_duration=selected_duration,
    )


# Compatibility aliases for legacy private import sites.
_classifier_scores_for_prefix = classifier_scores_for_prefix
_feature_set_scores_for_prefix = feature_set_scores_for_prefix
_feature_sigma = feature_sigma
_pair_priors = pair_priors
_reference_trajectory = reference_trajectory
_trajectory_features = trajectory_features


def evaluate_executable_pairs(
    *,
    config: CommonExperimentConfig,
    pair_specs: tuple[ExecutablePairSpec, ...],
    trajectories: tuple[ExecutableTrajectory, ...],
    scenario_family_fn: Callable[[str], str],
    scenario_tier_fn: Callable[[str], str],
) -> tuple[
    tuple[PairPredictionRow, ...],
    tuple[PosteriorHistoryRow, ...],
    tuple[LikelihoodHistoryRow, ...],
    tuple[FeatureTableRow, ...],
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
]:
    classifier_manifest = load_classifier_manifest(config.classifier_manifest_path)
    feature_manifest = load_feature_set_manifest(config.feature_sets_path)
    pair_lookup = {spec.pair_id: spec for spec in pair_specs}
    feature_rows: list[FeatureTableRow] = []
    prediction_rows: list[PairPredictionRow] = []
    posterior_rows: list[PosteriorHistoryRow] = []
    likelihood_rows: list[LikelihoodHistoryRow] = []
    metrics_by_pair: list[dict[str, object]] = []
    prior_rows: list[dict[str, object]] = []
    metrics_by_feature_set: list[dict[str, object]] = []

    trajectory_feature_cache: dict[tuple[str, bool], dict[str, float]] = {}
    for trajectory in trajectories:
        for feature_set_id in feature_manifest:
            robust = feature_set_id == "robust_extrema"
            key = (trajectory.trajectory_id, robust)
            if key not in trajectory_feature_cache:
                trajectory_feature_cache[key] = trajectory_features(trajectory, robust=robust)
            feature_values = trajectory_feature_cache[key]
            feature_rows.append(
                FeatureTableRow(
                    trajectory_id=trajectory.trajectory_id,
                    class_pair_id=trajectory.class_pair_id,
                    scenario_id=trajectory.scenario_id,
                    scenario_family=scenario_family_fn(trajectory.scenario_id),
                    dataset_tier=scenario_tier_fn(trajectory.scenario_id),
                    true_class=trajectory.true_class,
                    feature_set_id=feature_set_id,
                    feature_values=feature_values,
                )
            )

    grouped_predictions: dict[tuple[str, str], list[PairPredictionRow]] = {}
    for classifier_entry in classifier_manifest:
        classifier_id = str(classifier_entry["id"])
        sensor_regime_id = "position_only"
        feature_set_id = str(
            classifier_entry.get(
                "feature_set_id",
                classifier_entry.get("requires_feature_set", "instantaneous"),
            )
        )
        for trajectory in trajectories:
            pair_spec = pair_lookup[trajectory.class_pair_id]
            prior = pair_priors(pair_spec.class_a, pair_spec.class_b, "uniform")
            final_scores = classifier_scores_for_prefix(
                classifier_entry,
                pair_spec,
                trajectory,
                len(trajectory.times),
                prior,
                feature_manifest,
            )
            final_weights = _normalize_scores(final_scores)
            predicted_class = max(final_weights, key=final_weights.get)
            run_id = f"{classifier_id}:{trajectory.trajectory_id}"
            prediction_row = PairPredictionRow(
                run_id=run_id,
                classifier_id=classifier_id,
                feature_set_id=feature_set_id,
                sensor_regime_id=sensor_regime_id,
                measurement_dim=trajectory.measurement_dim,
                coordinate_frame=trajectory.coordinate_frame,
                class_pair_id=trajectory.class_pair_id,
                class_a=pair_spec.class_a,
                class_b=pair_spec.class_b,
                trajectory_id=trajectory.trajectory_id,
                scenario_id=trajectory.scenario_id,
                scenario_family=scenario_family_fn(trajectory.scenario_id),
                dataset_tier=scenario_tier_fn(trajectory.scenario_id),
                time=trajectory.times[-1],
                true_class=trajectory.true_class,
                predicted_class=predicted_class,
                confidence=final_weights[predicted_class],
                posterior_class_a=final_weights[pair_spec.class_a],
                posterior_class_b=final_weights[pair_spec.class_b],
            )
            prediction_rows.append(prediction_row)
            grouped_predictions.setdefault((classifier_id, trajectory.class_pair_id), []).append(prediction_row)

            for prefix_length in range(1, len(trajectory.times) + 1):
                scores = classifier_scores_for_prefix(
                    classifier_entry,
                    pair_spec,
                    trajectory,
                    prefix_length,
                    prior,
                    feature_manifest,
                )
                weights = _normalize_scores(scores)
                posterior_rows.append(
                    PosteriorHistoryRow(
                        run_id=run_id,
                        classifier_id=classifier_id,
                        feature_set_id=feature_set_id,
                        sensor_regime_id=sensor_regime_id,
                        class_pair_id=trajectory.class_pair_id,
                        class_a=pair_spec.class_a,
                        class_b=pair_spec.class_b,
                        trajectory_id=trajectory.trajectory_id,
                        scenario_id=trajectory.scenario_id,
                        scenario_family=scenario_family_fn(trajectory.scenario_id),
                        dataset_tier=scenario_tier_fn(trajectory.scenario_id),
                        time=trajectory.times[prefix_length - 1],
                        true_class=trajectory.true_class,
                        posterior_class_a=weights[pair_spec.class_a],
                        posterior_class_b=weights[pair_spec.class_b],
                    )
                )
                likelihood_rows.append(
                    LikelihoodHistoryRow(
                        run_id=run_id,
                        classifier_id=classifier_id,
                        feature_set_id=feature_set_id,
                        sensor_regime_id=sensor_regime_id,
                        class_pair_id=trajectory.class_pair_id,
                        trajectory_id=trajectory.trajectory_id,
                        scenario_id=trajectory.scenario_id,
                        scenario_family=scenario_family_fn(trajectory.scenario_id),
                        dataset_tier=scenario_tier_fn(trajectory.scenario_id),
                        time=trajectory.times[prefix_length - 1],
                        score_type="log_likelihood_proxy",
                        class_a=pair_spec.class_a,
                        class_b=pair_spec.class_b,
                        log_likelihood_class_a=scores[pair_spec.class_a],
                        log_likelihood_class_b=scores[pair_spec.class_b],
                    )
                )

        for pair_spec in pair_specs:
            pair_trajectories = [trajectory for trajectory in trajectories if trajectory.class_pair_id == pair_spec.pair_id]
            for prior_id in ("uniform", "mild_bias", "strong_bias"):
                prior = pair_priors(pair_spec.class_a, pair_spec.class_b, prior_id)
                accuracy_hits = 0
                for trajectory in pair_trajectories:
                    weights = _normalize_scores(
                        classifier_scores_for_prefix(
                            classifier_entry,
                            pair_spec,
                            trajectory,
                            len(trajectory.times),
                            prior,
                            feature_manifest,
                        )
                    )
                    predicted = max(weights, key=weights.get)
                    accuracy_hits += 1 if predicted == trajectory.true_class else 0
                prior_rows.append(
                    {
                        "classifier_id": classifier_id,
                        "class_pair_id": pair_spec.pair_id,
                        "prior_id": prior_id,
                        "accuracy": accuracy_hits / max(len(pair_trajectories), 1),
                    }
                )

    for (classifier_id, pair_id), rows in sorted(grouped_predictions.items()):
        accuracy = sum(1 for row in rows if row.predicted_class == row.true_class) / max(len(rows), 1)
        metrics_by_pair.append(
            {
                "classifier_id": classifier_id,
                "class_pair": pair_id,
                "overall_accuracy": accuracy,
                "status": "executed",
            }
        )

    feature_set_accuracy: dict[tuple[str, str], list[float]] = {}
    for row in prediction_rows:
        key = (row.classifier_id, row.feature_set_id)
        feature_set_accuracy.setdefault(key, []).append(1.0 if row.predicted_class == row.true_class else 0.0)
    for (classifier_id, feature_set_id), values in sorted(feature_set_accuracy.items()):
        metrics_by_feature_set.append(
            {
                "classifier_id": classifier_id,
                "feature_set_id": feature_set_id,
                "overall_accuracy": _mean(values),
            }
        )

    return (
        tuple(prediction_rows),
        tuple(posterior_rows),
        tuple(likelihood_rows),
        tuple(feature_rows),
        tuple(metrics_by_pair),
        tuple(prior_rows),
        tuple(metrics_by_feature_set),
    )


def _count_sign_changes(values: list[float], *, tolerance: float = 1e-9) -> int:
    filtered = [0 if abs(value) <= tolerance else (1 if value > 0.0 else -1) for value in values]
    filtered = [value for value in filtered if value != 0]
    if len(filtered) < 2:
        return 0
    return sum(1 for left, right in zip(filtered, filtered[1:]) if left != right)
