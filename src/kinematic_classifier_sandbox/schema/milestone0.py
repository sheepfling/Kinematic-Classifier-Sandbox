from __future__ import annotations

import json
from dataclasses import dataclass
from math import log
from pathlib import Path
from typing import Any

from kinematic_classifier_sandbox.reports.markdown import MarkdownDocument
from kinematic_classifier_sandbox.utils.io import write_csv
from kinematic_classifier_sandbox.utils.math import (
    _gaussian_logpdf,
    _normalize_posterior,
    _running_mean,
    _running_range,
    _running_slope,
)

from .artifacts import TrajectoryArtifact


@dataclass(frozen=True, slots=True)
class Milestone0SampleArtifacts:
    run_dir: Path
    config_path: Path
    dataset_manifest_path: Path
    class_definitions_path: Path
    feature_manifest_path: Path
    trajectory_path: Path
    feature_matrix_path: Path
    posterior_history_path: Path
    likelihood_history_path: Path
    metrics_path: Path
    report_path: Path


def _simple_sample_trajectory() -> TrajectoryArtifact:
    return TrajectoryArtifact(
        trajectory_id="traj_a",
        true_class="A",
        scenario_id="sample_two_class",
        seed=7,
        times=(0.0, 1.0, 2.0, 3.0),
        measurements=(0.10, -0.20, 0.05, 0.15),
        measurement_std=0.25,
        true_position=(0.0, 0.0, 0.0, 0.0),
        true_velocity=(0.0, 0.0, 0.0, 0.0),
        true_acceleration=(0.0, 0.0, 0.0, 0.0),
        generator_parameters={"class_means": {"A": 0.0, "B": 5.0}, "measurement_std": 0.25},
    )


def _simple_sample_trajectory_b() -> TrajectoryArtifact:
    return TrajectoryArtifact(
        trajectory_id="traj_b",
        true_class="B",
        scenario_id="sample_two_class",
        seed=8,
        times=(0.0, 1.0, 2.0, 3.0),
        measurements=(5.05, 4.90, 5.20, 4.85),
        measurement_std=0.25,
        true_position=(5.0, 5.0, 5.0, 5.0),
        true_velocity=(0.0, 0.0, 0.0, 0.0),
        true_acceleration=(0.0, 0.0, 0.0, 0.0),
        generator_parameters={"class_means": {"A": 0.0, "B": 5.0}, "measurement_std": 0.25},
    )


def _sample_feature_rows(trajectory: Any) -> tuple[dict[str, Any], ...]:
    running_mean = _running_mean(trajectory.measurements)
    running_range = _running_range(trajectory.measurements)
    running_slope = _running_slope(trajectory.times, trajectory.measurements)
    rows: list[dict[str, Any]] = []
    for index, time in enumerate(trajectory.times):
        rows.append(
            {
                "trajectory_id": trajectory.trajectory_id,
                "time": time,
                "window_start": trajectory.times[0],
                "window_end": time,
                "running_mean": running_mean[index],
                "running_range": running_range[index],
                "running_slope": running_slope[index],
            }
        )
    return tuple(rows)


def _predict_sample_classifier(trajectory: Any, class_means: dict[str, float], class_sigma: float) -> tuple[dict[str, Any], ...]:
    prior = {name: 1.0 / len(class_means) for name in class_means}
    rows: list[dict[str, Any]] = []
    sigma_sq = class_sigma * class_sigma
    for time, measurement in zip(trajectory.times, trajectory.measurements):
        log_scores = {
            name: log(prior[name]) + _gaussian_logpdf(measurement, mean, sigma_sq)
            for name, mean in class_means.items()
        }
        posterior = _normalize_posterior(log_scores)
        predicted_class = max(posterior, key=posterior.get)
        rows.append(
            {
                "trajectory_id": trajectory.trajectory_id,
                "time": time,
                "true_class": trajectory.true_class,
                "predicted_class": predicted_class,
                "confidence": posterior[predicted_class],
                **{f"posterior_{name}": posterior[name] for name in class_means},
                **{f"log_likelihood_{name}": log_scores[name] for name in class_means},
            }
        )
        prior = posterior
    return tuple(rows)


def _infer_metrics(posterior_rows: tuple[dict[str, Any], ...], class_names: tuple[str, ...]) -> dict[str, Any]:
    if not posterior_rows:
        return {"trajectory_count": 0, "row_count": 0, "final_accuracy": 0.0, "step_accuracy": 0.0, "confusion_counts": {}}
    trajectory_ids = {row["trajectory_id"] for row in posterior_rows}
    final_rows: dict[str, dict[str, Any]] = {}
    correct_steps = 0
    for row in posterior_rows:
        if row["true_class"] == row["predicted_class"]:
            correct_steps += 1
        final_rows[row["trajectory_id"]] = row
    confusion: dict[str, dict[str, int]] = {name: {other: 0 for other in class_names} for name in class_names}
    correct_final = 0
    for row in final_rows.values():
        confusion[row["true_class"]][row["predicted_class"]] += 1
        if row["true_class"] == row["predicted_class"]:
            correct_final += 1
    return {
        "trajectory_count": len(trajectory_ids),
        "row_count": len(posterior_rows),
        "final_accuracy": correct_final / max(len(final_rows), 1),
        "step_accuracy": correct_steps / len(posterior_rows),
        "confusion_counts": confusion,
    }


def _render_run_config_yaml(run_name: str, seed: int, class_names: tuple[str, ...], trajectory_count: int) -> str:
    class_list = ", ".join(class_names)
    return "\n".join(
        [
            "experiment:",
            f"  name: {run_name}",
            f"  seed: {seed}",
            "dataset:",
            "  generator: sample_two_class_gaussian",
            f"  trajectory_count: {trajectory_count}",
            f"  class_names: [{class_list}]",
            "features:",
            "  enabled_feature_groups: [running_mean, running_range, running_slope]",
            "  window_duration: null",
            "classifier:",
            "  type: pointwise_gaussian",
            "  prior: uniform",
            "evaluation:",
            "  metrics: [final_accuracy, step_accuracy, confusion_counts]",
            "visualization:",
            "  enabled_plots: [posterior_history, feature_matrix]",
            "",
        ]
    )


def _render_sample_report(run_name: str, trajectories: tuple[Any, ...], metrics: dict[str, Any]) -> str:
    report = MarkdownDocument(run_name)
    report.heading("Experiment Summary", level=2)
    report.bullet_list(
        [
            f"Trajectories: {len(trajectories)}",
            f"Final accuracy: {metrics['final_accuracy']:.3f}",
            f"Step accuracy: {metrics['step_accuracy']:.3f}",
        ]
    )
    report.heading("Artifact Contract", level=2)
    report.bullet_list(
        [
            "Trajectory artifacts contain raw measurements and optional truth.",
            "Feature artifacts use explicit time windows and documented units.",
            "Classifier outputs carry posterior probabilities for every class.",
        ]
    )
    report.heading("Validation", level=2)
    report.bullet_list(
        [
            "Times are strictly increasing.",
            "Posterior rows sum to one.",
            "Predicted class equals posterior argmax.",
        ]
    )
    return report.text()


def write_milestone0_sample_run_artifacts(output_dir: str | Path, *, run_name: str = "milestone0_contract_demo") -> Milestone0SampleArtifacts:
    output_root = Path(output_dir)
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    class_means = {"A": 0.0, "B": 5.0}
    class_sigma = 0.75
    trajectories = (_simple_sample_trajectory(), _simple_sample_trajectory_b())
    trajectory_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    posterior_rows: list[dict[str, Any]] = []
    likelihood_rows: list[dict[str, Any]] = []

    for trajectory in trajectories:
        trajectory_rows.append(
            {
                "trajectory_id": trajectory.trajectory_id,
                "true_class": trajectory.true_class,
                "scenario_id": trajectory.scenario_id,
                "seed": trajectory.seed,
                "times": json.dumps(list(trajectory.times)),
                "measurements": json.dumps(list(trajectory.measurements)),
                "measurement_std": trajectory.measurement_std,
                "generator_parameters": json.dumps(trajectory.generator_parameters, sort_keys=True),
            }
        )
        feature_rows.extend(_sample_feature_rows(trajectory))
        trajectory_posterior_rows = _predict_sample_classifier(trajectory, class_means, class_sigma)
        posterior_rows.extend(trajectory_posterior_rows)
        likelihood_rows.extend(
            {
                "trajectory_id": row["trajectory_id"],
                "time": row["time"],
                **{f"log_likelihood_{name}": row[f"log_likelihood_{name}"] for name in class_means},
            }
            for row in trajectory_posterior_rows
        )

    posterior_rows = list(posterior_rows)
    metrics = _infer_metrics(tuple(posterior_rows), tuple(class_means))
    metrics["class_names"] = list(class_means)
    metrics["trajectory_ids"] = [trajectory.trajectory_id for trajectory in trajectories]

    config_path = run_dir / "config.yaml"
    dataset_manifest_path = run_dir / "dataset_manifest.json"
    class_definitions_path = run_dir / "class_definitions.json"
    feature_manifest_path = run_dir / "feature_manifest.json"
    trajectory_path = run_dir / "sample_trajectory.csv"
    feature_matrix_path = run_dir / "sample_feature_matrix.csv"
    posterior_history_path = run_dir / "sample_posterior_history.csv"
    likelihood_history_path = run_dir / "sample_likelihood_history.csv"
    metrics_path = run_dir / "metrics.json"
    report_path = run_dir / "report.md"

    config_path.write_text(_render_run_config_yaml(run_name, 7, tuple(class_means), len(trajectories)), encoding="utf-8")
    dataset_manifest_path.write_text(
        json.dumps(
            {
                "run_name": run_name,
                "seed": 7,
                "trajectory_count": len(trajectories),
                "scenario_id": trajectories[0].scenario_id,
                "class_counts": {name: sum(1 for trajectory in trajectories if trajectory.true_class == name) for name in class_means},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    class_definitions_path.write_text(
        json.dumps(
            {
                "class_names": list(class_means),
                "class_means": class_means,
                "class_sigma": class_sigma,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    feature_manifest_path.write_text(
        json.dumps(
            {
                "feature_names": ["running_mean", "running_range", "running_slope"],
                "feature_units": {"running_mean": "unit", "running_range": "unit", "running_slope": "unit / time"},
                "history_behavior": {
                    "running_mean": "cumulative",
                    "running_range": "cumulative",
                    "running_slope": "history-aware",
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    write_csv(
        trajectory_path,
        trajectory_rows,
        [
            "trajectory_id",
            "true_class",
            "scenario_id",
            "seed",
            "times",
            "measurements",
            "measurement_std",
            "generator_parameters",
        ],
    )
    write_csv(
        feature_matrix_path,
        feature_rows,
        ["trajectory_id", "time", "window_start", "window_end", "running_mean", "running_range", "running_slope"],
    )
    write_csv(
        posterior_history_path,
        posterior_rows,
        [
            "trajectory_id",
            "time",
            "true_class",
            "predicted_class",
            "confidence",
            "posterior_A",
            "posterior_B",
            "log_likelihood_A",
            "log_likelihood_B",
        ],
    )
    write_csv(
        likelihood_history_path,
        likelihood_rows,
        ["trajectory_id", "time", "log_likelihood_A", "log_likelihood_B"],
    )
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(_render_sample_report(run_name, trajectories, metrics), encoding="utf-8")

    return Milestone0SampleArtifacts(
        run_dir=run_dir,
        config_path=config_path,
        dataset_manifest_path=dataset_manifest_path,
        class_definitions_path=class_definitions_path,
        feature_manifest_path=feature_manifest_path,
        trajectory_path=trajectory_path,
        feature_matrix_path=feature_matrix_path,
        posterior_history_path=posterior_history_path,
        likelihood_history_path=likelihood_history_path,
        metrics_path=metrics_path,
        report_path=report_path,
    )


__all__ = [
    "Milestone0SampleArtifacts",
    "write_milestone0_sample_run_artifacts",
]
