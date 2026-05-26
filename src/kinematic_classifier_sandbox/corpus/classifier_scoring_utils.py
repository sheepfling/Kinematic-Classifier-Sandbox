from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kinematic_classifier_sandbox.utils.math import _entropy
from kinematic_classifier_sandbox.utils.math import mean as _mean

from ..analysis.generated_corpus_features import GeneratedCorpusRecord
from ..inference.kalman_filter_bank import KalmanModelSpec, KalmanTrajectory, run_kalman_filter_bank
from ..inference.pointwise_baseline import (
    PointwiseClassSpec,
    PointwiseTrajectory,
    run_pointwise_classifier,
)
from ..inference.sequential_bayes_accumulator import (
    AccumulatorClassSpec,
    AccumulatorTrajectory,
    run_accumulator,
)
from ..inference.windowed_baseline import (
    WindowedClassSpec,
    WindowedFeatureClassifier,
    WindowedTrajectory,
    extract_windowed_feature_rows,
)


def _measurements(record: GeneratedCorpusRecord) -> tuple[float, ...]:
    return tuple(float(value) for value in record.execution.trajectory_run.observations.get("position", ()))


def _mean_sigma(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    avg = _mean(values)
    variance = sum((value - avg) ** 2 for value in values) / max(len(values), 1)
    sigma = max(variance ** 0.5, 0.05)
    return avg, sigma


def _class_measurement_stats(records: tuple[GeneratedCorpusRecord, ...]) -> dict[str, tuple[float, float]]:
    values_by_class: dict[str, list[float]] = {}
    for record in records:
        values_by_class.setdefault(record.assigned_class, []).extend(_measurements(record))
    return {class_name: _mean_sigma(values) for class_name, values in values_by_class.items()}


def _pointwise_specs(records: tuple[GeneratedCorpusRecord, ...]) -> tuple[PointwiseClassSpec, ...]:
    stats = _class_measurement_stats(records)
    return tuple(
        PointwiseClassSpec(name=class_name, mean=avg, sigma=sigma, prior_weight=1.0 / max(len(stats), 1))
        for class_name, (avg, sigma) in sorted(stats.items())
    )


def _accumulator_specs(records: tuple[GeneratedCorpusRecord, ...]) -> tuple[AccumulatorClassSpec, ...]:
    stats = _class_measurement_stats(records)
    return tuple(
        AccumulatorClassSpec(name=class_name, mean=avg, sigma=max(sigma, 0.08), prior_weight=1.0 / max(len(stats), 1))
        for class_name, (avg, sigma) in sorted(stats.items())
    )


def _windowed_specs(
    trajectories: tuple[WindowedTrajectory, ...],
    *,
    feature_mode: str,
) -> tuple[WindowedClassSpec, ...]:
    features = ("running_min", "running_max", "running_range", "slope") if feature_mode == "raw" else (
        "robust_min",
        "robust_max",
        "trimmed_range",
        "slope",
    )
    grouped_rows: dict[str, list[Any]] = {}
    for trajectory in trajectories:
        grouped_rows.setdefault(trajectory.true_class, []).extend(extract_windowed_feature_rows(trajectory))
    specs: list[WindowedClassSpec] = []
    for class_name, rows in sorted(grouped_rows.items()):
        means = {feature: _mean([float(getattr(row, feature)) for row in rows]) for feature in features}
        sigmas = {}
        for feature in features:
            values = [float(getattr(row, feature)) for row in rows]
            _, sigma = _mean_sigma(values)
            sigmas[feature] = max(sigma, 0.05)
        specs.append(
            WindowedClassSpec(
                name=class_name,
                prior_weight=1.0 / max(len(grouped_rows), 1),
                feature_means=means,
                feature_sigmas=sigmas,
            )
        )
    return tuple(specs)


def _kalman_specs(records: tuple[GeneratedCorpusRecord, ...]) -> tuple[KalmanModelSpec, ...]:
    measurement_sigma = max(_mean([record.candidate.measurement_std for record in records]), 0.05) if records else 0.20
    templates = {
        "stationary": (1, 0.04),
        "constant_velocity": (2, 0.14),
        "constant_acceleration": (3, 0.24),
        "braking": (3, 0.30),
        "maneuver": (3, 0.38),
    }
    specs: list[KalmanModelSpec] = []
    classes = sorted({record.assigned_class for record in records})
    for class_name in classes:
        state_dim, process_sigma = templates.get(class_name, (3, 0.32))
        specs.append(
            KalmanModelSpec(
                name=class_name,
                class_name=class_name,
                state_dim=state_dim,
                process_sigma=process_sigma,
                measurement_sigma=measurement_sigma,
                initial_covariance_scale=5.0 + state_dim,
                prior_weight=1.0 / max(len(classes), 1),
            )
        )
    return tuple(specs)


def _pointwise_trajectory(record: GeneratedCorpusRecord) -> PointwiseTrajectory:
    run = record.execution.trajectory_run
    return PointwiseTrajectory(
        trajectory_id=run.run_id,
        true_class=record.assigned_class,
        scenario_name=record.candidate.scenario_family,
        seed=run.seed,
        times=tuple(float(value) for value in run.times),
        measurements=_measurements(record),
    )


def _accumulator_trajectory(record: GeneratedCorpusRecord) -> AccumulatorTrajectory:
    run = record.execution.trajectory_run
    return AccumulatorTrajectory(
        trajectory_id=run.run_id,
        true_class=record.assigned_class,
        scenario_name=record.candidate.scenario_family,
        seed=run.seed,
        times=tuple(float(value) for value in run.times),
        measurements=_measurements(record),
    )


def _windowed_trajectory(record: GeneratedCorpusRecord) -> WindowedTrajectory:
    run = record.execution.trajectory_run
    return WindowedTrajectory(
        trajectory_id=run.run_id,
        true_class=record.assigned_class,
        scenario_name=record.candidate.scenario_family,
        seed=run.seed,
        times=tuple(float(value) for value in run.times),
        measurements=_measurements(record),
    )


def _kalman_trajectory(record: GeneratedCorpusRecord) -> KalmanTrajectory:
    run = record.execution.trajectory_run
    truth = run.truth_state
    return KalmanTrajectory(
        trajectory_id=run.run_id,
        true_class=record.assigned_class,
        scenario_name=record.candidate.scenario_family,
        seed=run.seed,
        times=tuple(float(value) for value in run.times),
        measurements=_measurements(record),
        true_position=tuple(float(value) for value in truth.get("position", ())),
        true_velocity=tuple(float(value) for value in truth.get("velocity", ())),
        true_acceleration=tuple(float(value) for value in truth.get("acceleration", ())),
    )


def _margin(weights: dict[str, float]) -> float:
    ordered = sorted(weights.values(), reverse=True)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    return ordered[0] - ordered[1]


def _time_to_confidence(times: tuple[float, ...], confidences: list[float], threshold: float = 0.75) -> float | None:
    for time, confidence in zip(times, confidences):
        if confidence >= threshold:
            return float(time)
    return None


def _prior_sweep(
    method_name: str,
    record: GeneratedCorpusRecord,
    class_names: tuple[str, ...],
    rerun: Any,
) -> tuple[dict[str, Any], ...]:
    if len(class_names) < 2:
        return ()
    primary = record.assigned_class
    alternate = next((name for name in class_names if name != primary), class_names[0])
    rows: list[dict[str, Any]] = []
    flip_threshold: float | None = None
    for primary_prior in (0.1, 0.25, 0.5, 0.75, 0.9):
        remaining = max(1.0 - primary_prior, 0.0)
        if len(class_names) == 2:
            prior = {primary: primary_prior, alternate: remaining}
        else:
            share = remaining / max(len(class_names) - 1, 1)
            prior = {name: share for name in class_names}
            prior[primary] = primary_prior
        run = rerun(prior)
        final_predicted_class = str(run.final_predicted_class)
        if flip_threshold is None and final_predicted_class != primary:
            flip_threshold = primary_prior
        rows.append(
            {
                "method_name": method_name,
                "trajectory_id": record.execution.trajectory_run.run_id,
                "true_class": record.assigned_class,
                "target_primary_class": primary,
                "alternate_class": alternate,
                "primary_prior": primary_prior,
                "predicted_class": final_predicted_class,
                "final_confidence": max(float(value) for value in run.final_weights.values()),
                "prior_flip_threshold": flip_threshold if flip_threshold is not None else 1.0,
            }
        )
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class _Run:
    final_weights: dict[str, float]
    final_predicted_class: str


def _rerun_windowed(
    trajectory: WindowedTrajectory,
    specs: tuple[WindowedClassSpec, ...],
    feature_mode: str,
    prior: dict[str, float],
) -> _Run:
    classifier = WindowedFeatureClassifier(specs, feature_mode=feature_mode, prior=prior)
    classifier.reset(prior)
    for feature_row in extract_windowed_feature_rows(trajectory):
        classifier.update(feature_row)
    return _Run(final_weights=classifier.posterior(), final_predicted_class=classifier.predict())
