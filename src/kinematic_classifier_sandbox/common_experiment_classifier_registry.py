from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .common_dataset_comparison import SCENARIO_MEASUREMENT_SIGMA
from .common_1d_study_adapter import ExecutablePairSpec, ExecutableTrajectory


FeatureExtractor = Callable[[ExecutableTrajectory, bool], dict[str, float]]
FeatureSigma = Callable[[str], float]
ReferenceBuilder = Callable[[ExecutablePairSpec, str, str, tuple[float, ...]], ExecutableTrajectory]
GaussianLogPdf = Callable[[float, float, float], float]
SafeLog = Callable[[float], float]


@dataclass(frozen=True, slots=True)
class FamilyScoringContext:
    pair_spec: ExecutablePairSpec
    trajectory: ExecutableTrajectory
    truncated: ExecutableTrajectory
    times: tuple[float, ...]
    prior_weights: dict[str, float]
    feature_manifest: dict[str, dict[str, object]]
    reference_builder: ReferenceBuilder
    feature_extractor: FeatureExtractor
    feature_sigma: FeatureSigma
    gaussian_logpdf: GaussianLogPdf
    safe_log: SafeLog


def _pointwise_family_scores(context: FamilyScoringContext) -> dict[str, float]:
    last_time = context.times[-1]
    last_value = context.truncated.measurements[-1]
    scores = {}
    for class_name in (context.pair_spec.class_a, context.pair_spec.class_b):
        reference = context.reference_builder(context.pair_spec, class_name, context.trajectory.scenario_id, context.times)
        last_velocity = 0.0
        if len(context.times) >= 2:
            dt = context.times[-1] - context.times[-2]
            last_velocity = (context.truncated.measurements[-1] - context.truncated.measurements[-2]) / max(dt, 1e-9)
        ref_velocity = reference.true_velocity[-1]
        scores[class_name] = (
            context.safe_log(context.prior_weights[class_name])
            + context.gaussian_logpdf(last_value, reference.true_position[-1], SCENARIO_MEASUREMENT_SIGMA[context.trajectory.scenario_id] * 1.4)
            + 0.35 * context.gaussian_logpdf(last_velocity, ref_velocity, 0.45)
            + 0.05 * context.gaussian_logpdf(last_time, last_time, 1.0)
        )
    return scores


def _sequential_bayes_family_scores(context: FamilyScoringContext) -> dict[str, float]:
    scores = {}
    for class_name in (context.pair_spec.class_a, context.pair_spec.class_b):
        reference = context.reference_builder(context.pair_spec, class_name, context.trajectory.scenario_id, context.truncated.times)
        value = context.safe_log(context.prior_weights[class_name])
        for measurement, expected in zip(context.truncated.measurements, reference.true_position):
            value += context.gaussian_logpdf(measurement, expected, SCENARIO_MEASUREMENT_SIGMA[context.trajectory.scenario_id] * 1.15)
        scores[class_name] = value
    return scores


def _windowed_family_scores(context: FamilyScoringContext, *, family: str, feature_set_id: str) -> dict[str, float]:
    features = context.feature_manifest[feature_set_id]["features"]
    robust = feature_set_id == "robust_extrema"
    observed = context.feature_extractor(context.truncated, robust=robust)
    scores = {}
    for class_name in (context.pair_spec.class_a, context.pair_spec.class_b):
        reference = context.reference_builder(context.pair_spec, class_name, context.trajectory.scenario_id, context.truncated.times)
        reference_features = context.feature_extractor(reference, robust=robust)
        score = context.safe_log(context.prior_weights[class_name])
        for feature_name in features:
            name = str(feature_name)
            sigma = context.feature_sigma(name)
            if family == "state_space" and name in {"linear_fit_residual", "quadratic_fit_residual"}:
                sigma *= 0.75
            score += context.gaussian_logpdf(observed[name], reference_features[name], sigma)
        if family == "state_space":
            for measurement, expected in zip(context.truncated.measurements, reference.true_position):
                score += 0.25 * context.gaussian_logpdf(measurement, expected, SCENARIO_MEASUREMENT_SIGMA[context.trajectory.scenario_id] * 1.35)
        scores[class_name] = score
    return scores


def score_classifier_family(
    classifier_entry: dict[str, object],
    context: FamilyScoringContext,
) -> dict[str, float]:
    family = str(classifier_entry["family"])
    feature_set_id = str(
        classifier_entry.get(
            "feature_set_id",
            classifier_entry.get("requires_feature_set", "instantaneous"),
        )
    )
    if family == "pointwise":
        return _pointwise_family_scores(context)
    if family == "sequential_bayes":
        return _sequential_bayes_family_scores(context)
    if family in {"windowed", "state_space"}:
        if feature_set_id not in context.feature_manifest:
            raise KeyError(feature_set_id)
        return _windowed_family_scores(context, family=family, feature_set_id=feature_set_id)
    raise KeyError(f"unsupported classifier family: {family}")
