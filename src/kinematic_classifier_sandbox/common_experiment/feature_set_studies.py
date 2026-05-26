from __future__ import annotations

from typing import Callable

from ..analysis.feature_analysis import load_feature_set_manifest, resolve_feature_names
from ..utils.math import mean as _mean
from ..utils.math import normalize_log_scores as _normalize_scores
from .contracts import CommonExperimentConfig, ExecutablePairSpec, ExecutableTrajectory


def build_feature_set_comparison_rows(
    *,
    config: CommonExperimentConfig,
    pair_specs: tuple[ExecutablePairSpec, ...],
    trajectories: tuple[ExecutableTrajectory, ...],
    pair_priors: Callable[[str, str, str], dict[str, float]],
    feature_set_scores_for_prefix: Callable[..., dict[str, float]],
) -> tuple[dict[str, object], ...]:
    feature_manifest = load_feature_set_manifest(config.feature_sets_path)
    pair_lookup = {spec.pair_id: spec for spec in pair_specs}
    rows: list[dict[str, object]] = []
    for feature_set_id, entry in sorted(feature_manifest.items()):
        features = entry.get("features")
        if not isinstance(features, list) or not features:
            continue
        hits: list[float] = []
        pair_hits: dict[str, list[float]] = {}
        confidence_values: list[float] = []
        for trajectory in trajectories:
            pair_spec = pair_lookup[trajectory.class_pair_id]
            prior = pair_priors(pair_spec.class_a, pair_spec.class_b, "uniform")
            scores = feature_set_scores_for_prefix(
                feature_set_id=feature_set_id,
                feature_entry=entry,
                pair_spec=pair_spec,
                trajectory=trajectory,
                prefix_length=len(trajectory.times),
                prior_weights=prior,
            )
            weights = _normalize_scores(scores)
            predicted = max(weights, key=weights.get)
            hit = 1.0 if predicted == trajectory.true_class else 0.0
            hits.append(hit)
            pair_hits.setdefault(trajectory.class_pair_id, []).append(hit)
            confidence_values.append(max(weights.values()))
        pair_accuracies = [_mean(values) for values in pair_hits.values()]
        rows.append(
            {
                "feature_set_id": feature_set_id,
                "history_behavior": str(entry.get("history_behavior", "unknown")),
                "num_features": len(features),
                "overall_accuracy": _mean(hits),
                "min_pair_accuracy": min(pair_accuracies) if pair_accuracies else 0.0,
                "max_pair_accuracy": max(pair_accuracies) if pair_accuracies else 0.0,
                "mean_confidence": _mean(confidence_values),
            }
        )
    return tuple(rows)


def build_irregular_window_comparison_rows(
    *,
    config: CommonExperimentConfig,
    pair_specs: tuple[ExecutablePairSpec, ...],
    trajectories: tuple[ExecutableTrajectory, ...],
    pair_priors: Callable[[str, str, str], dict[str, float]],
    feature_set_scores_for_window: Callable[..., tuple[dict[str, float], dict[str, float], int, float]],
) -> tuple[dict[str, object], ...]:
    feature_manifest = load_feature_set_manifest(config.feature_sets_path)
    pair_lookup = {spec.pair_id: spec for spec in pair_specs}
    irregular_trajectories = [trajectory for trajectory in trajectories if trajectory.scenario_id == "irregular"]
    if not irregular_trajectories:
        return ()

    window_sample_count = 4
    window_duration = 3.0
    per_def_rows: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    paired_results: dict[tuple[str, str, str], dict[str, object]] = {}

    for feature_set_id, entry in sorted(feature_manifest.items()):
        try:
            resolve_feature_names(feature_set=feature_set_id, manifest=feature_manifest)
        except (KeyError, ValueError):
            continue
        for trajectory in irregular_trajectories:
            pair_spec = pair_lookup[trajectory.class_pair_id]
            prior = pair_priors(pair_spec.class_a, pair_spec.class_b, "uniform")
            trajectory_results: dict[str, dict[str, object]] = {}
            for window_definition in ("sample_count", "elapsed_time"):
                window_scores = feature_set_scores_for_window(
                    feature_set_id=feature_set_id,
                    feature_manifest=feature_manifest,
                    pair_spec=pair_spec,
                    trajectory=trajectory,
                    window_definition=window_definition,
                    window_sample_count=window_sample_count,
                    window_duration=window_duration,
                    prior_weights=prior,
                )
                scores = window_scores.scores
                observed = window_scores.observed
                selected_count = window_scores.selected_count
                selected_duration = window_scores.selected_duration
                weights = _normalize_scores(scores)
                predicted_class = max(weights, key=weights.get)
                trajectory_results[window_definition] = {
                    "predicted_class": predicted_class,
                    "confidence": weights[predicted_class],
                    "selected_sample_count": selected_count,
                    "selected_duration": selected_duration,
                    "features": observed,
                }
                per_def_rows.setdefault((trajectory.class_pair_id, feature_set_id, window_definition), []).append(
                    {
                        "hit": 1.0 if predicted_class == trajectory.true_class else 0.0,
                        "confidence": weights[predicted_class],
                        "selected_sample_count": float(selected_count),
                        "selected_duration": selected_duration,
                    }
                )
            paired_results[(trajectory.class_pair_id, feature_set_id, trajectory.trajectory_id)] = trajectory_results

    rows: list[dict[str, object]] = []
    for (class_pair_id, feature_set_id, window_definition), selected in sorted(per_def_rows.items()):
        paired = [
            result
            for (pair_id, feature_id, _), result in paired_results.items()
            if pair_id == class_pair_id and feature_id == feature_set_id
        ]
        disagreement_flags = []
        feature_delta_values = []
        for result in paired:
            sample_result = result["sample_count"]
            elapsed_result = result["elapsed_time"]
            disagreement_flags.append(
                1.0 if sample_result["predicted_class"] != elapsed_result["predicted_class"] else 0.0
            )
            shared_feature_names = set(sample_result["features"]) & set(elapsed_result["features"])
            deltas = [
                abs(float(sample_result["features"][name]) - float(elapsed_result["features"][name]))
                for name in shared_feature_names
            ]
            feature_delta_values.append(_mean(deltas) if deltas else 0.0)
        rows.append(
            {
                "class_pair_id": class_pair_id,
                "feature_set_id": feature_set_id,
                "history_behavior": str(feature_manifest[feature_set_id].get("history_behavior", "unknown")),
                "window_definition": window_definition,
                "window_sample_count": window_sample_count,
                "window_duration": window_duration,
                "num_predictions": len(selected),
                "overall_accuracy": _mean([float(row["hit"]) for row in selected]),
                "mean_confidence": _mean([float(row["confidence"]) for row in selected]),
                "mean_selected_sample_count": _mean([float(row["selected_sample_count"]) for row in selected]),
                "mean_selected_duration": _mean([float(row["selected_duration"]) for row in selected]),
                "cross_window_prediction_disagreement_rate": _mean(disagreement_flags),
                "mean_cross_window_feature_delta": _mean(feature_delta_values),
            }
        )
    return tuple(rows)
