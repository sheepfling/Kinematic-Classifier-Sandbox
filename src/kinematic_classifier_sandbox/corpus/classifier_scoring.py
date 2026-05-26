from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kinematic_classifier_sandbox.utils.math import _entropy

from ..analysis.generated_corpus_features import select_generated_corpus_records
from .classifier_scoring_artifact_io import write_corpus_classifier_scoring_artifacts
from .classifier_scoring_reporting import render_corpus_classifier_scoring_report
from .classifier_scoring_types import CorpusClassifierScoringArtifacts, CorpusClassifierScoringResult
from .classifier_scoring_utils import (
    _accumulator_specs,
    _accumulator_trajectory,
    _class_measurement_stats,
    _kalman_specs,
    _kalman_trajectory,
    _margin,
    _measurements,
    _mean_sigma,
    _pointwise_specs,
    _pointwise_trajectory,
    _prior_sweep,
    _rerun_windowed,
    _time_to_confidence,
    _windowed_specs,
    _windowed_trajectory,
)
from ..inference.kalman_filter_bank import run_kalman_filter_bank
from ..inference.pointwise_baseline import run_pointwise_classifier
from ..inference.sequential_bayes_accumulator import run_accumulator
from ..inference.windowed_baseline import WindowedFeatureClassifier, extract_windowed_feature_rows

def analyze_corpus_classifier_scoring() -> CorpusClassifierScoringResult:
    records = select_generated_corpus_records()
    pointwise_specs = _pointwise_specs(records)
    accumulator_specs = _accumulator_specs(records)
    kalman_specs = _kalman_specs(records)
    windowed_trajectories = tuple(_windowed_trajectory(record) for record in records)
    windowed_raw_specs = _windowed_specs(windowed_trajectories, feature_mode="raw")
    windowed_robust_specs = _windowed_specs(windowed_trajectories, feature_mode="robust")

    candidate_score_rows: list[dict[str, Any]] = []
    posterior_rows: list[dict[str, Any]] = []
    prior_rows: list[dict[str, Any]] = []
    per_trajectory_predictions: dict[str, dict[str, str]] = {}

    def append_method(
        *,
        method_name: str,
        record: GeneratedCorpusRecord,
        final_weights: dict[str, float],
        final_predicted_class: str,
        step_rows: list[dict[str, Any]],
        confidences: list[float],
        times: tuple[float, ...],
        heuristic_reference: float,
    ) -> None:
        entropy = _entropy(final_weights)
        margin = _margin(final_weights)
        confidence = max(final_weights.values()) if final_weights else 0.0
        ttc = _time_to_confidence(times, confidences)
        candidate_score_rows.append(
            {
                "method_name": method_name,
                "trajectory_id": record.execution.trajectory_run.run_id,
                "candidate_id": record.candidate.candidate_id,
                "backend_id": record.execution.backend_id,
                "true_class": record.assigned_class,
                "predicted_class": final_predicted_class,
                "correct": final_predicted_class == record.assigned_class,
                "confidence": confidence,
                "posterior_entropy": entropy,
                "top_two_margin": margin,
                "confident_error": final_predicted_class != record.assigned_class and confidence >= 0.75,
                "time_to_confidence": "" if ttc is None else ttc,
                "measured_classifier_stress": (1.0 - margin) * 0.5 + entropy * 0.5 + (0.35 if final_predicted_class != record.assigned_class else 0.0),
                "heuristic_stress_reference": heuristic_reference,
                "difficulty_tier": record.candidate.difficulty_tier,
                "scenario_family": record.candidate.scenario_family,
            }
        )
        posterior_rows.extend(step_rows)
        per_trajectory_predictions.setdefault(record.execution.trajectory_run.run_id, {})[method_name] = final_predicted_class

    class_names = tuple(spec.name for spec in pointwise_specs)
    for record in records:
        pointwise_trajectory = _pointwise_trajectory(record)
        pointwise_run = run_pointwise_classifier(pointwise_trajectory, pointwise_specs)
        append_method(
            method_name="pointwise",
            record=record,
            final_weights=pointwise_run.final_weights,
            final_predicted_class=pointwise_run.final_predicted_class,
            step_rows=[
                {
                    "method_name": "pointwise",
                    "trajectory_id": pointwise_run.trajectory_id,
                    "time": step.time,
                    "true_class": pointwise_run.true_class,
                    "predicted_class": step.predicted_class,
                    "confidence": step.confidence,
                    **{f"posterior_{name}": step.posterior_weights.get(name, 0.0) for name in class_names},
                }
                for step in pointwise_run.steps
            ],
            confidences=[step.confidence for step in pointwise_run.steps],
            times=pointwise_trajectory.times,
            heuristic_reference=0.45,
        )
        prior_rows.extend(
            _prior_sweep(
                "pointwise",
                record,
                class_names,
                lambda prior: run_pointwise_classifier(pointwise_trajectory, pointwise_specs, prior=prior),
            )
        )

        accumulator_trajectory = _accumulator_trajectory(record)
        accumulator_run = run_accumulator(accumulator_trajectory, accumulator_specs, forgetting_factor=0.97, confidence_threshold=0.72)
        append_method(
            method_name="sequential_bayes",
            record=record,
            final_weights=accumulator_run.final_weights,
            final_predicted_class=accumulator_run.final_predicted_class,
            step_rows=[
                {
                    "method_name": "sequential_bayes",
                    "trajectory_id": accumulator_run.trajectory_id,
                    "time": step.time,
                    "true_class": accumulator_run.true_class,
                    "predicted_class": step.predicted_class,
                    "confidence": step.confidence,
                    **{f"posterior_{name}": step.posterior_weights.get(name, 0.0) for name in class_names},
                }
                for step in accumulator_run.steps
            ],
            confidences=[step.confidence for step in accumulator_run.steps],
            times=accumulator_trajectory.times,
            heuristic_reference=0.45,
        )
        prior_rows.extend(
            _prior_sweep(
                "sequential_bayes",
                record,
                class_names,
                lambda prior: run_accumulator(accumulator_trajectory, accumulator_specs, forgetting_factor=0.97, confidence_threshold=0.72, prior=prior),
            )
        )

        windowed_trajectory = _windowed_trajectory(record)
        for method_name, feature_mode, specs, heuristic_reference in (
            ("windowed_raw", "raw", windowed_raw_specs, 0.45),
            ("windowed_robust", "robust", windowed_robust_specs, 0.45),
        ):
            classifier = WindowedFeatureClassifier(specs, feature_mode=feature_mode)
            classifier.reset()
            feature_rows = extract_windowed_feature_rows(windowed_trajectory)
            for feature_row in feature_rows:
                classifier.update(feature_row)
            steps = classifier.history()
            final_weights = classifier.posterior()
            final_predicted_class = classifier.predict()
            append_method(
                method_name=method_name,
                record=record,
                final_weights=final_weights,
                final_predicted_class=final_predicted_class,
                step_rows=[
                    {
                        "method_name": method_name,
                        "trajectory_id": windowed_trajectory.trajectory_id,
                        "time": step.time,
                        "true_class": windowed_trajectory.true_class,
                        "predicted_class": step.predicted_class,
                        "confidence": step.confidence,
                        **{f"posterior_{name}": step.posterior_weights.get(name, 0.0) for name in class_names},
                    }
                    for step in steps
                ],
                confidences=[step.confidence for step in steps],
                times=windowed_trajectory.times,
                heuristic_reference=0.45 if feature_mode == "raw" else 0.60,
            )
            prior_rows.extend(
                _prior_sweep(
                    method_name,
                    record,
                    class_names,
                    lambda prior, wm=windowed_trajectory, sp=specs, fm=feature_mode: _rerun_windowed(wm, sp, fm, prior),
                )
            )

        kalman_trajectory = _kalman_trajectory(record)
        kalman_run = run_kalman_filter_bank(kalman_trajectory, kalman_specs)
        append_method(
            method_name="kalman_bank",
            record=record,
            final_weights=kalman_run.final_weights,
            final_predicted_class=kalman_run.final_predicted_class,
            step_rows=[
                {
                    "method_name": "kalman_bank",
                    "trajectory_id": kalman_run.trajectory_id,
                    "time": step.time,
                    "true_class": kalman_run.true_class,
                    "predicted_class": step.predicted_class,
                    "confidence": step.confidence,
                    **{f"posterior_{name}": step.posterior_weights.get(name, 0.0) for name in class_names},
                }
                for step in kalman_run.steps
            ],
            confidences=[step.confidence for step in kalman_run.steps],
            times=kalman_trajectory.times,
            heuristic_reference=0.70 if record.candidate.scenario_family == "file_backend_case" else 0.45,
        )
        prior_rows.extend(
            _prior_sweep(
                "kalman_bank",
                record,
                class_names,
                lambda prior: run_kalman_filter_bank(kalman_trajectory, kalman_specs, prior=prior),
            )
        )

    disagreement_rows = []
    for trajectory_id, predictions in sorted(per_trajectory_predictions.items()):
        unique_predictions = sorted(set(predictions.values()))
        disagreement_rows.append(
            {
                "trajectory_id": trajectory_id,
                "method_count": len(predictions),
                "unique_prediction_count": len(unique_predictions),
                "predicted_classes": ",".join(unique_predictions),
                "has_disagreement": len(unique_predictions) > 1,
            }
        )

    report_markdown = render_corpus_classifier_scoring_report(
        record_count=len(records),
        candidate_score_rows=tuple(candidate_score_rows),
        posterior_rows=tuple(posterior_rows),
        disagreement_rows=tuple(disagreement_rows),
    )
    return CorpusClassifierScoringResult(
        candidate_score_rows=tuple(candidate_score_rows),
        posterior_rows=tuple(posterior_rows),
        prior_sensitivity_rows=tuple(prior_rows),
        disagreement_rows=tuple(disagreement_rows),
        report_markdown=report_markdown,
    )


def _rerun_windowed(
    trajectory: WindowedTrajectory,
    specs: tuple[WindowedClassSpec, ...],
    feature_mode: str,
    prior: dict[str, float],
) -> Any:
    classifier = WindowedFeatureClassifier(specs, feature_mode=feature_mode, prior=prior)
    classifier.reset(prior)
    for feature_row in extract_windowed_feature_rows(trajectory):
        classifier.update(feature_row)

    @dataclass(frozen=True, slots=True)
    class _Run:
        final_weights: dict[str, float]
        final_predicted_class: str

    return _Run(final_weights=classifier.posterior(), final_predicted_class=classifier.predict())
