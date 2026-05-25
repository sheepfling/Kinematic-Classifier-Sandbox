from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from math import log
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .generated_corpus_features import GeneratedCorpusRecord, select_generated_corpus_records
from .kalman_filter_bank import KalmanModelSpec, KalmanTrajectory, run_kalman_filter_bank
from .pointwise_baseline import PointwiseClassSpec, PointwiseTrajectory, run_pointwise_classifier
from .sequential_bayes_accumulator import AccumulatorClassSpec, AccumulatorTrajectory, run_accumulator
from .windowed_baseline import (
    WindowedClassSpec,
    WindowedFeatureClassifier,
    WindowedTrajectory,
    extract_windowed_feature_rows,
)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@dataclass(frozen=True, slots=True)
class CorpusClassifierScoringResult:
    candidate_score_rows: tuple[dict[str, Any], ...]
    posterior_rows: tuple[dict[str, Any], ...]
    prior_sensitivity_rows: tuple[dict[str, Any], ...]
    disagreement_rows: tuple[dict[str, Any], ...]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class CorpusClassifierScoringArtifacts:
    run_dir: Path
    candidate_scores_path: Path
    posterior_history_path: Path
    prior_sensitivity_path: Path
    disagreement_path: Path
    report_path: Path
    posterior_plot_path: Path
    disagreement_plot_path: Path
    stress_plot_path: Path


def _measurements(record: GeneratedCorpusRecord) -> tuple[float, ...]:
    return tuple(float(value) for value in record.execution.trajectory_run.observations.get("position", ()))


def _mean_sigma(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    avg = mean(values)
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
        means = {feature: mean(getattr(row, feature) for row in rows) for feature in features}
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
    measurement_sigma = max(mean(record.candidate.measurement_std for record in records), 0.05) if records else 0.20
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


def _entropy(weights: dict[str, float]) -> float:
    return -sum(value * log(max(value, 1e-12)) for value in weights.values())


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

    report_markdown = "\n".join(
        [
            "# Corpus Classifier Scoring",
            "",
            "## Summary",
            f"- scored trajectories: `{len(records)}`",
            f"- classifier result rows: `{len(candidate_score_rows)}`",
            f"- posterior history rows: `{len(posterior_rows)}`",
            f"- disagreement cases: `{sum(1 for row in disagreement_rows if row['has_disagreement'])}`",
            "",
            "## Methods",
            "- `pointwise`",
            "- `sequential_bayes`",
            "- `windowed_raw`",
            "- `windowed_robust`",
            "- `kalman_bank`",
            "",
            "## Notes",
            "- Classifier stress is now measured from real posterior outputs, margins, and errors rather than assigned from scenario family heuristics.",
            "- The class-model parameters are fit from the generated corpus slice so this milestone can score objective-driven trajectories without pretending the benchmark defaults apply unchanged.",
        ]
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


def _render_posterior_plot(rows: tuple[dict[str, Any], ...]) -> bytes:
    selected = [row for row in rows if row["method_name"] == "sequential_bayes"][:24]
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    ax.plot([row["time"] for row in selected], [row["confidence"] for row in selected], marker="o", linewidth=1.0)
    ax.set_title("Posterior Confidence Trace Preview")
    ax.set_xlabel("Time")
    ax.set_ylabel("Confidence")
    fig.tight_layout()
    from io import BytesIO

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    return buffer.getvalue()


def _render_disagreement_plot(rows: tuple[dict[str, Any], ...]) -> bytes:
    labels = [str(row["trajectory_id"]).split("_", 1)[-1] for row in rows[:12]]
    values = [int(row["unique_prediction_count"]) for row in rows[:12]]
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    ax.bar(labels, values, color="#b56b4d")
    ax.set_title("Method Disagreement By Trajectory")
    ax.set_ylabel("Unique Final Predictions")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    from io import BytesIO

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    return buffer.getvalue()


def _render_stress_plot(rows: tuple[dict[str, Any], ...]) -> bytes:
    methods = sorted({str(row["method_name"]) for row in rows})
    values = [mean(float(row["measured_classifier_stress"]) for row in rows if row["method_name"] == method) for method in methods]
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.bar(methods, values, color="#5c7ea5")
    ax.set_title("Measured Classifier Stress By Method")
    ax.set_ylabel("Mean Stress")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    from io import BytesIO

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    return buffer.getvalue()


def write_corpus_classifier_scoring_artifacts(
    base_dir: str | Path,
    *,
    result: CorpusClassifierScoringResult | None = None,
) -> CorpusClassifierScoringArtifacts:
    run_dir = Path(base_dir) / "corpus_classifier_scoring"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = result or analyze_corpus_classifier_scoring()

    candidate_scores_path = run_dir / "classifier_candidate_scores.csv"
    posterior_history_path = run_dir / "posterior_history.csv"
    prior_sensitivity_path = run_dir / "prior_sensitivity_scores.csv"
    disagreement_path = run_dir / "method_disagreement_scores.csv"
    report_path = run_dir / "classifier_scoring_report.md"
    posterior_plot_path = run_dir / "posterior_confidence_preview.png"
    disagreement_plot_path = run_dir / "method_disagreement_preview.png"
    stress_plot_path = run_dir / "classifier_stress_by_method.png"

    _write_csv(candidate_scores_path, list(payload.candidate_score_rows), list(payload.candidate_score_rows[0].keys()) if payload.candidate_score_rows else [])
    _write_csv(posterior_history_path, list(payload.posterior_rows), list(payload.posterior_rows[0].keys()) if payload.posterior_rows else [])
    _write_csv(prior_sensitivity_path, list(payload.prior_sensitivity_rows), list(payload.prior_sensitivity_rows[0].keys()) if payload.prior_sensitivity_rows else [])
    _write_csv(disagreement_path, list(payload.disagreement_rows), list(payload.disagreement_rows[0].keys()) if payload.disagreement_rows else [])
    report_path.write_text(payload.report_markdown, encoding="utf-8")
    posterior_plot_path.write_bytes(_render_posterior_plot(payload.posterior_rows))
    disagreement_plot_path.write_bytes(_render_disagreement_plot(payload.disagreement_rows))
    stress_plot_path.write_bytes(_render_stress_plot(payload.candidate_score_rows))

    return CorpusClassifierScoringArtifacts(
        run_dir=run_dir,
        candidate_scores_path=candidate_scores_path,
        posterior_history_path=posterior_history_path,
        prior_sensitivity_path=prior_sensitivity_path,
        disagreement_path=disagreement_path,
        report_path=report_path,
        posterior_plot_path=posterior_plot_path,
        disagreement_plot_path=disagreement_plot_path,
        stress_plot_path=stress_plot_path,
    )
