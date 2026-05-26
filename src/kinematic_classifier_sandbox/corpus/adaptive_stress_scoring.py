from __future__ import annotations

from math import exp, log
from types import SimpleNamespace
from typing import NamedTuple

from ..analysis.common_dataset_comparison import SharedDynamicsTrajectory, _accumulator_predict, _kalman_predict, _pointwise_predict, _windowed_predict
from ..inference.irregular_window_comparison import (
    WindowRegimeTrajectory,
    _duration_window,
    _sample_count_window,
    generate_window_regime_trajectories,
)
from ..inference.irregular_window_comparison import _gaussian_logpdf as _window_gaussian_logpdf
from ..inference.irregular_window_comparison import _normalize as _window_normalize
from ..inference.prior_sensitivity_types import PriorSweepPredictions
from ..inference.transition_matrix_accumulator import _run_mode_accumulator, default_switching_mode_specs, default_transition_matrix, generate_transition_switching_scenarios
from ..utils.math import _entropy
from .adaptive_stress_utils import _classify_window_row, _local_window_features, _observable_pair_posterior
from .gym import CorpusGymTarget


class ReferenceWindowStats(NamedTuple):
    sample_count: dict[str, dict[str, float]]
    duration: dict[str, dict[str, float]]


class WindowClassification(NamedTuple):
    predicted_class: str
    confidence: float


class TransitionDelayCandidates(NamedTuple):
    rows: list[dict[str, object]]
    posterior_payloads: list[dict[str, object]]
    feature_payloads: list[dict[str, object]]


def _prior_sweep_predictions(shared: SharedDynamicsTrajectory) -> PriorSweepPredictions:
    rows = []
    for prior_cv in (0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90):
        run = _accumulator_predict(
            shared,
            prior={"constant_velocity": prior_cv, "constant_acceleration": 1.0 - prior_cv},
        )
        rows.append((prior_cv, run.final_predicted_class, run.final_confidence))
    return PriorSweepPredictions(tuple(rows))


def _wrong_classification_score(shared: SharedDynamicsTrajectory) -> tuple[float, dict[str, object], dict[str, object]]:
    pointwise = _pointwise_predict(shared)
    score = pointwise.final_confidence if pointwise.final_predicted_class != shared.true_class else 0.0
    payload = {
        "failure_mode": "wrong_classification",
        "posterior_trace": _accumulator_trace(shared),
    }
    return score, {"pointwise_confidence": pointwise.final_confidence}, payload


def _high_entropy_score(shared: SharedDynamicsTrajectory) -> tuple[float, dict[str, object], dict[str, object]]:
    posterior, details = _observable_pair_posterior(shared)
    entropy = _entropy(list(posterior.values()))
    payload = {
        "failure_mode": "high_entropy",
        "posterior_trace": _accumulator_trace(shared),
    }
    return entropy, {"entropy": entropy, **details}, payload


def _prior_flip_score(shared: SharedDynamicsTrajectory) -> tuple[float, dict[str, object], dict[str, object]]:
    sweep = _prior_sweep_predictions(shared)
    rows = sweep.rows
    predicted_classes = {row[1] for row in rows}
    reference_row = rows[2] if len(rows) >= 3 else rows[0]
    if len(predicted_classes) > 1:
        score = 1.0
        smallest_shift = min(abs(row[0] - 0.5) for row in rows if row[1] != reference_row[1])
    else:
        reference_confidence = reference_row[2]
        score = max(0.0, 1.0 - abs(reference_confidence - 0.5) * 2.0)
        smallest_shift = 0.5
    payload = {
        "failure_mode": "prior_flip",
        "sweep": rows,
    }
    return score, {"num_distinct_predictions": float(len(predicted_classes)), "smallest_shift_to_flip": smallest_shift}, payload


def _raw_extrema_failure_score(shared: SharedDynamicsTrajectory) -> tuple[float, dict[str, object], dict[str, object]]:
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
    sample_classification = _classify_window_row(sample_row, sample_stats)
    duration_classification = _classify_window_row(duration_row, duration_stats)
    sample_conf = sample_classification.confidence
    duration_predicted = duration_classification.predicted_class
    duration_conf = duration_classification.confidence
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


def _transition_delay_candidates(
    *,
    seed: int,
    random_candidates: int,
    guided_candidates: int,
) -> TransitionDelayCandidates:
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
    return TransitionDelayCandidates(
        rows=rows,
        posterior_payloads=posterior_payloads,
        feature_payloads=feature_payloads,
    )


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
