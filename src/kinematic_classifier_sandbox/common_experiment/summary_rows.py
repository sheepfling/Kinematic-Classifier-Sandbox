from __future__ import annotations

from dataclasses import asdict
from math import exp
from typing import Callable, cast

from ..analysis.feature_analysis import resolve_feature_names
from ..scenarios import get_scenario_family, get_scenario_measurement_sigma
from ..utils.math import (
    gaussian_logpdf as _gaussian_logpdf,
)
from ..utils.math import (
    mean as _mean,
)
from ..utils.math import (
    normalize_log_scores as _normalize_scores,
)
from ..utils.math import (
    safe_log as _safe_log,
)
from ..utils.math import (
    std as _std,
)
from .contracts import (
    ClassPairDurationRow,
    ClassPairScenarioRow,
    CovariateRow,
    ExecutableTrajectory,
    FeatureExcitationRow,
    PairPredictionRow,
)
from .summary_rows_types import (
    ClassPairDurationSummaryRow,
    ClassPairScenarioSummaryRow,
    CovariateAuditRow,
    FeatureExcitationSummaryRow,
    IdentifiabilitySummaryRow,
    MetricsByClassifierRow,
    MetricsBySensorRegimeRow,
    OracleSummaryRow,
)


def _feature_sigma(feature_name: str) -> float:
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


def trajectory_covariates(trajectory: ExecutableTrajectory) -> dict[str, float]:
    dt_values = [trajectory.times[index] - trajectory.times[index - 1] for index in range(1, len(trajectory.times))]
    mean_dt = _mean(dt_values) if dt_values else 0.0
    std_dt = (
        (
            sum((value - mean_dt) ** 2 for value in dt_values)
            / max(len(dt_values) - 1, 1)
        )
        ** 0.5
        if len(dt_values) >= 2
        else 0.0
    )
    outlier_count = 0
    try:
        sigma = get_scenario_measurement_sigma(trajectory.scenario_id)
    except KeyError:
        sigma = 0.0
    if sigma > 0.0:
        outlier_count = sum(
            1
            for measurement, truth in zip(trajectory.measurements, trajectory.true_position)
            if abs(measurement - truth) > 3.0 * sigma
        )
    return {
        "duration": float(trajectory.times[-1] - trajectory.times[0]) if trajectory.times else 0.0,
        "sample_count": float(len(trajectory.times)),
        "mean_dt": mean_dt,
        "std_dt": std_dt,
        "max_dt": max(dt_values) if dt_values else 0.0,
        "sampling_irregularity": std_dt / max(mean_dt, 1e-6) if len(dt_values) >= 2 else 0.0,
        "measurement_std": sigma,
        "outlier_fraction": outlier_count / max(len(trajectory.times), 1),
    }


def metrics_by_classifier(prediction_rows: tuple[PairPredictionRow, ...]) -> tuple[MetricsByClassifierRow, ...]:
    grouped: dict[str, list[PairPredictionRow]] = {}
    for row in prediction_rows:
        grouped.setdefault(row.classifier_id, []).append(row)
    rows: list[MetricsByClassifierRow] = []
    for classifier_id, classifier_rows in sorted(grouped.items()):
        accuracy = sum(1 for row in classifier_rows if row.predicted_class == row.true_class) / max(len(classifier_rows), 1)
        rows.append(
            {
                "classifier_id": classifier_id,
                "overall_accuracy": accuracy,
                "num_predictions": len(classifier_rows),
            }
        )
    return tuple(rows)


def metrics_by_sensor_regime(prediction_rows: tuple[PairPredictionRow, ...]) -> tuple[MetricsBySensorRegimeRow, ...]:
    grouped: dict[str, list[PairPredictionRow]] = {}
    for row in prediction_rows:
        grouped.setdefault(row.sensor_regime_id, []).append(row)
    rows: list[MetricsBySensorRegimeRow] = []
    for sensor_regime_id, regime_rows in sorted(grouped.items()):
        hits = [1.0 if row.predicted_class == row.true_class else 0.0 for row in regime_rows]
        confidences = [row.confidence for row in regime_rows]
        classifier_ids = {row.classifier_id for row in regime_rows}
        measurement_dims = sorted({row.measurement_dim for row in regime_rows})
        coordinate_frames = sorted({row.coordinate_frame for row in regime_rows})
        rows.append(
            {
                "sensor_regime_id": sensor_regime_id,
                "same_sensor_fairness_bucket": sensor_regime_id,
                "overall_accuracy": _mean(hits),
                "mean_confidence": _mean(confidences),
                "num_predictions": len(regime_rows),
                "num_classifiers": len(classifier_ids),
                "measurement_dims": ",".join(str(value) for value in measurement_dims),
                "coordinate_frames": ",".join(coordinate_frames),
            }
        )
    return tuple(rows)


def covariate_rows(
    trajectories: tuple[ExecutableTrajectory, ...],
    *,
    scenario_tier_fn: Callable[[str], str],
    scenario_family_fn: Callable[[str], str],
) -> tuple[CovariateAuditRow, ...]:
    grouped: dict[tuple[str, str, str, str], list[ExecutableTrajectory]] = {}
    pair_tier_values: dict[tuple[str, str], dict[str, list[float]]] = {}
    for trajectory in trajectories:
        dataset_tier = scenario_tier_fn(trajectory.scenario_id)
        scenario_family = scenario_family_fn(trajectory.scenario_id)
        grouped.setdefault(
            (trajectory.class_pair_id, dataset_tier, scenario_family, trajectory.true_class),
            [],
        ).append(trajectory)
        pair_tier_key = (trajectory.class_pair_id, dataset_tier)
        pair_tier_values.setdefault(pair_tier_key, {})
        covariates = trajectory_covariates(trajectory)
        for name, value in covariates.items():
            pair_tier_values[pair_tier_key].setdefault(name, []).append(value)

    rows: list[CovariateRow] = []
    audited_covariates = (
        "duration",
        "sample_count",
        "measurement_std",
        "outlier_fraction",
        "sampling_irregularity",
    )
    for (pair_id, dataset_tier, scenario_family, true_class), selected in sorted(grouped.items()):
        covariate_dicts = [trajectory_covariates(trajectory) for trajectory in selected]
        mean_values = {
            name: _mean([row[name] for row in covariate_dicts])
            for name in covariate_dicts[0]
        }
        baseline = {
            name: _mean(pair_tier_values[(pair_id, dataset_tier)][name])
            for name in covariate_dicts[0]
        }
        delta_ratios = {
            name: abs(mean_values[name] - baseline[name]) / max(abs(baseline[name]), 1e-6)
            for name in audited_covariates
        }
        max_delta_name = max(delta_ratios, key=delta_ratios.get)
        max_delta_ratio = delta_ratios[max_delta_name]
        status = "pass" if max_delta_ratio <= 0.20 else ("warn" if max_delta_ratio <= 0.40 else "fail")
        rows.append(
            CovariateRow(
                class_pair_id=pair_id,
                dataset_tier=dataset_tier,
                scenario_family=scenario_family,
                true_class=true_class,
                num_trajectories=len(selected),
                mean_duration=mean_values["duration"],
                mean_sample_count=mean_values["sample_count"],
                mean_dt=mean_values["mean_dt"],
                std_dt=mean_values["std_dt"],
                max_dt=mean_values["max_dt"],
                sampling_irregularity=mean_values["sampling_irregularity"],
                measurement_std=mean_values["measurement_std"],
                outlier_fraction=mean_values["outlier_fraction"],
                max_covariate_delta_name=max_delta_name,
                max_covariate_delta_ratio=max_delta_ratio,
                status=status,
            )
        )
    return cast(tuple[CovariateAuditRow, ...], tuple(asdict(row) for row in rows))


def feature_excitation_rows(feature_rows: tuple[dict[str, object], ...]) -> tuple[FeatureExcitationSummaryRow, ...]:
    feature_names = (
        "position_range",
        "speed_range",
        "acceleration_range",
        "acceleration_variance",
        "curvature_proxy",
        "velocity_sign_changes",
        "acceleration_sign_changes",
        "monotonicity",
        "linear_fit_residual",
        "quadratic_fit_residual",
        "outlier_score",
    )
    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = {}
    for row in feature_rows:
        grouped.setdefault(
            (
                str(row["class_pair_id"]),
                str(row["dataset_tier"]),
                str(row["scenario_family"]),
                str(row["feature_set_id"]),
            ),
            [],
        ).append(row)
    rows: list[FeatureExcitationRow] = []
    for (pair_id, dataset_tier, scenario_family, feature_set_id), selected in sorted(grouped.items()):
        feature_means: dict[str, float] = {}
        feature_stds: dict[str, float] = {}
        for feature_name in feature_names:
            values = [float(item[feature_name]) for item in selected]
            feature_means[feature_name] = _mean([abs(value) for value in values])
            feature_stds[feature_name] = _std(values)
        rows.append(
            FeatureExcitationRow(
                class_pair_id=pair_id,
                dataset_tier=dataset_tier,
                scenario_family=scenario_family,
                feature_set_id=feature_set_id,
                num_rows=len(selected),
                feature_means=feature_means,
                feature_stds=feature_stds,
            )
        )
    return cast(tuple[FeatureExcitationSummaryRow, ...], tuple(asdict(row) for row in rows))


def class_pair_duration_rows(
    posterior_rows: tuple[dict[str, object], ...],
) -> tuple[ClassPairDurationSummaryRow, ...]:
    grouped: dict[tuple[str, str, float], list[dict[str, object]]] = {}
    for row in posterior_rows:
        grouped.setdefault(
            (
                str(row["classifier_id"]),
                str(row["class_pair_id"]),
                float(row["time"]),
            ),
            [],
        ).append(row)
    rows: list[ClassPairDurationSummaryRow] = []
    for (classifier_id, class_pair_id, time_value), selected in sorted(grouped.items()):
        hits = 0
        confidence_sum = 0.0
        margin_sum = 0.0
        for row in selected:
            posterior_a = float(row["posterior_class_a"])
            posterior_b = float(row["posterior_class_b"])
            predicted = str(row["class_a"]) if posterior_a >= posterior_b else str(row["class_b"])
            hits += 1 if predicted == str(row["true_class"]) else 0
            confidence_sum += max(posterior_a, posterior_b)
            margin_sum += abs(posterior_a - posterior_b)
        rows.append(
            ClassPairDurationRow(
                classifier_id=classifier_id,
                class_pair_id=class_pair_id,
                time=time_value,
                num_prefixes=len(selected),
                prefix_accuracy=hits / max(len(selected), 1),
                mean_confidence=confidence_sum / max(len(selected), 1),
                posterior_margin=margin_sum / max(len(selected), 1),
            )
        )
    return cast(tuple[ClassPairDurationSummaryRow, ...], tuple(asdict(row) for row in rows))


def class_pair_scenario_rows(
    prediction_rows: tuple[dict[str, object], ...],
) -> tuple[ClassPairScenarioSummaryRow, ...]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in prediction_rows:
        grouped.setdefault(
            (
                str(row["classifier_id"]),
                str(row["class_pair_id"]),
                str(row["scenario_id"]),
            ),
            [],
        ).append(row)
    rows: list[ClassPairScenarioSummaryRow] = []
    for (classifier_id, class_pair_id, scenario_id), selected in sorted(grouped.items()):
        rows.append(
            ClassPairScenarioRow(
                classifier_id=classifier_id,
                class_pair_id=class_pair_id,
                scenario_id=scenario_id,
                scenario_family=(
                    get_scenario_family(scenario_id)
                    if scenario_id in {"easy", "irregular", "endpoint_match", "short", "short_noisy", "outlier"}
                    else "other"
                ),
                overall_accuracy=_mean([1.0 if row["predicted_class"] == row["true_class"] else 0.0 for row in selected]),
                mean_confidence=_mean([float(row["confidence"]) for row in selected]),
                num_predictions=len(selected),
            )
        )
    return cast(tuple[ClassPairScenarioSummaryRow, ...], tuple(asdict(row) for row in rows))


def identifiability_rows(
    feature_rows: tuple[dict[str, object], ...],
    *,
    feature_manifest: dict[str, dict[str, object]],
) -> tuple[IdentifiabilitySummaryRow, ...]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in feature_rows:
        grouped.setdefault((str(row["class_pair_id"]), str(row["feature_set_id"])), []).append(row)

    rows: list[IdentifiabilitySummaryRow] = []
    for (pair_id, feature_set_id), selected in sorted(grouped.items()):
        feature_names = resolve_feature_names(feature_set=feature_set_id, manifest=feature_manifest)
        class_names = sorted({str(row["true_class"]) for row in selected})
        if len(class_names) != 2:
            continue
        class_a_rows = [row for row in selected if str(row["true_class"]) == class_names[0]]
        class_b_rows = [row for row in selected if str(row["true_class"]) == class_names[1]]
        if not class_a_rows or not class_b_rows:
            continue

        standardized_distances: list[float] = []
        absolute_distances: list[float] = []
        overlap_estimates: list[float] = []
        for feature_name in feature_names:
            values_a = [float(row[feature_name]) for row in class_a_rows]
            values_b = [float(row[feature_name]) for row in class_b_rows]
            mean_a = _mean(values_a)
            mean_b = _mean(values_b)
            sigma_a = max(_std(values_a), 1e-6)
            sigma_b = max(_std(values_b), 1e-6)
            pooled_sigma = max((sigma_a + sigma_b) / 2.0, 1e-6)
            standardized_gap = abs(mean_a - mean_b) / pooled_sigma
            standardized_distances.append(standardized_gap)
            absolute_distances.append(abs(mean_a - mean_b))
            overlap_estimates.append(exp(-0.5 * standardized_gap))

        mean_standardized_distance = _mean(standardized_distances)
        mean_overlap_estimate = _mean(overlap_estimates)
        rows.append(
            {
                "class_pair_id": pair_id,
                "feature_set_id": feature_set_id,
                "history_behavior": str(feature_manifest[feature_set_id].get("history_behavior", "unknown")),
                "class_a": class_names[0],
                "class_b": class_names[1],
                "num_examples": len(selected),
                "num_features": len(feature_names),
                "mean_absolute_feature_distance": _mean(absolute_distances),
                "mean_standardized_feature_distance": mean_standardized_distance,
                "overlap_estimate": mean_overlap_estimate,
                "confusability_score": 1.0 / (1.0 + mean_standardized_distance),
                "identifiability_status": (
                    "separable"
                    if mean_standardized_distance >= 1.5
                    else ("borderline" if mean_standardized_distance >= 0.8 else "confusable")
                ),
            }
        )
    return tuple(rows)


def oracle_rows(
    feature_rows: tuple[dict[str, object], ...],
    *,
    feature_manifest: dict[str, dict[str, object]],
) -> tuple[OracleSummaryRow, ...]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in feature_rows:
        grouped.setdefault((str(row["class_pair_id"]), str(row["feature_set_id"])), []).append(row)

    pair_rows: dict[str, list[dict[str, object]]] = {}
    for (pair_id, feature_set_id), selected in sorted(grouped.items()):
        feature_names = resolve_feature_names(feature_set=feature_set_id, manifest=feature_manifest)
        hits: list[float] = []
        confidences: list[float] = []
        margins: list[float] = []

        for held_out in selected:
            train_rows = [row for row in selected if str(row["trajectory_id"]) != str(held_out["trajectory_id"])]
            class_names = sorted({str(row["true_class"]) for row in train_rows})
            if len(class_names) != 2:
                continue

            log_scores: dict[str, float] = {}
            for class_name in class_names:
                class_rows = [row for row in train_rows if str(row["true_class"]) == class_name]
                score = _safe_log(1.0 / len(class_names))
                for feature_name in feature_names:
                    values = [float(row[feature_name]) for row in class_rows]
                    mean_value = _mean(values)
                    sigma = max(_std(values), 0.5 * _feature_sigma(feature_name))
                    score += _gaussian_logpdf(float(held_out[feature_name]), mean_value, sigma)
                log_scores[class_name] = score

            weights = _normalize_scores(log_scores)
            ranked = sorted(weights.items(), key=lambda item: item[1], reverse=True)
            predicted_class = ranked[0][0]
            hits.append(1.0 if predicted_class == str(held_out["true_class"]) else 0.0)
            confidences.append(ranked[0][1])
            margins.append(ranked[0][1] - ranked[1][1])

        feature_row = {
            "class_pair_id": pair_id,
            "feature_set_id": feature_set_id,
            "oracle_accuracy": _mean(hits),
            "mean_confidence": _mean(confidences),
            "mean_posterior_margin": _mean(margins),
            "num_examples": len(selected),
            "history_behavior": str(feature_manifest[feature_set_id].get("history_behavior", "unknown")),
            "best_feature_set_for_pair": "",
            "best_oracle_accuracy_for_pair": 0.0,
            "is_best_feature_set": False,
        }
        pair_rows.setdefault(pair_id, []).append(feature_row)

    rows: list[OracleSummaryRow] = []
    for pair_id, selected in sorted(pair_rows.items()):
        best_row = max(selected, key=lambda row: (float(row["oracle_accuracy"]), float(row["mean_posterior_margin"])))
        for row in selected:
            rows.append(
                {
                    **row,
                    "best_feature_set_for_pair": str(best_row["feature_set_id"]),
                    "best_oracle_accuracy_for_pair": float(best_row["oracle_accuracy"]),
                    "is_best_feature_set": str(row["feature_set_id"]) == str(best_row["feature_set_id"]),
                }
            )
    return tuple(rows)
