from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kinematic_classifier_sandbox.utils.io import write_csv
from kinematic_classifier_sandbox.utils.math import _entropy
from kinematic_classifier_sandbox.utils.math import _mean
from ..runtime_paths import prepare_matplotlib
from ..utils.plotting import figure_to_png_bytes
from ..utils.plotting import plt

from ..analysis.generated_corpus_features import select_generated_corpus_records
from .classifier_scoring_rendering import (
    _render_disagreement_plot,
    _render_posterior_plot,
    _render_stress_plot,
)
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

    return figure_to_png_bytes(fig, dpi=180)


def _render_disagreement_plot(rows: tuple[dict[str, Any], ...]) -> bytes:
    labels = [str(row["trajectory_id"]).split("_", 1)[-1] for row in rows[:12]]
    values = [int(row["unique_prediction_count"]) for row in rows[:12]]
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    ax.bar(labels, values, color="#b56b4d")
    ax.set_title("Method Disagreement By Trajectory")
    ax.set_ylabel("Unique Final Predictions")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()

    return figure_to_png_bytes(fig, dpi=180)


def _render_stress_plot(rows: tuple[dict[str, Any], ...]) -> bytes:
    methods = sorted({str(row["method_name"]) for row in rows})
    values = [_mean([float(row["measured_classifier_stress"]) for row in rows if row["method_name"] == method]) for method in methods]
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.bar(methods, values, color="#5c7ea5")
    ax.set_title("Measured Classifier Stress By Method")
    ax.set_ylabel("Mean Stress")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()

    return figure_to_png_bytes(fig, dpi=180)


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

    write_csv(candidate_scores_path, list(payload.candidate_score_rows), list(payload.candidate_score_rows[0].keys()) if payload.candidate_score_rows else [])
    write_csv(posterior_history_path, list(payload.posterior_rows), list(payload.posterior_rows[0].keys()) if payload.posterior_rows else [])
    write_csv(prior_sensitivity_path, list(payload.prior_sensitivity_rows), list(payload.prior_sensitivity_rows[0].keys()) if payload.prior_sensitivity_rows else [])
    write_csv(disagreement_path, list(payload.disagreement_rows), list(payload.disagreement_rows[0].keys()) if payload.disagreement_rows else [])
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
