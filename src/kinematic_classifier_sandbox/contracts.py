from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, log, pi, sqrt
import csv
import json
from pathlib import Path
from typing import Any


def _is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def _logsumexp(values: list[float]) -> float:
    if not values:
        return float("-inf")
    pivot = max(values)
    return pivot + log(sum(exp(value - pivot) for value in values))


def _gaussian_logpdf(value: float, mean: float, variance: float) -> float:
    safe_variance = max(variance, 1e-9)
    return -0.5 * (log(2.0 * pi * safe_variance) + ((value - mean) ** 2) / safe_variance)


def _normalize_posterior(log_scores: dict[str, float]) -> dict[str, float]:
    log_norm = _logsumexp(list(log_scores.values()))
    return {name: exp(score - log_norm) for name, score in log_scores.items()}


def _running_mean(values: tuple[float, ...]) -> tuple[float, ...]:
    running: list[float] = []
    total = 0.0
    for index, value in enumerate(values, start=1):
        total += value
        running.append(total / index)
    return tuple(running)


def _running_range(values: tuple[float, ...]) -> tuple[float, ...]:
    running: list[float] = []
    current_min = values[0]
    current_max = values[0]
    for value in values:
        current_min = min(current_min, value)
        current_max = max(current_max, value)
        running.append(current_max - current_min)
    return tuple(running)


def _running_slope(times: tuple[float, ...], values: tuple[float, ...]) -> tuple[float, ...]:
    slopes: list[float] = []
    for index, time in enumerate(times):
        if index == 0:
            slopes.append(0.0)
            continue
        dt = max(times[index] - times[0], 1e-9)
        slopes.append((values[index] - values[0]) / dt)
    return tuple(slopes)


@dataclass(frozen=True, slots=True)
class TrajectoryArtifact:
    trajectory_id: str
    true_class: str
    scenario_id: str
    seed: int
    times: tuple[float, ...]
    measurements: tuple[float, ...]
    measurement_dim: int = 1
    measurement_axes: tuple[str, ...] = ("position",)
    coordinate_frame: str = "scalar_line"
    measurement_std: float | None = None
    true_position: tuple[float, ...] | None = None
    true_velocity: tuple[float, ...] | None = None
    true_acceleration: tuple[float, ...] | None = None
    state_dim: int = 1
    state_axes: tuple[str, ...] = ("position",)
    truth_series: dict[str, tuple[float, ...]] = field(default_factory=dict)
    generator_parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FeatureArtifact:
    trajectory_id: str
    feature_names: tuple[str, ...]
    feature_units: dict[str, str]
    rows: tuple[dict[str, Any], ...]
    history_notes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ClassifierOutputArtifact:
    trajectory_id: str
    class_names: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]
    classifier_id: str | None = None
    sensor_regime_id: str | None = None
    feature_set_id: str | None = None
    run_id: str | None = None


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


def validate_trajectory_artifact(artifact: TrajectoryArtifact) -> list[str]:
    errors: list[str] = []
    if not artifact.trajectory_id:
        errors.append("trajectory_id is required")
    if not artifact.true_class:
        errors.append("true_class is required")
    if not artifact.scenario_id:
        errors.append("scenario_id is required")
    if artifact.seed is None:
        errors.append("seed is required")
    if len(artifact.times) != len(artifact.measurements):
        errors.append("times and measurements must have the same length")
    if len(artifact.times) == 0:
        errors.append("times and measurements must not be empty")
    if artifact.measurement_dim < 1:
        errors.append("measurement_dim must be at least 1")
    if artifact.state_dim < 1:
        errors.append("state_dim must be at least 1")
    if len(artifact.measurement_axes) != artifact.measurement_dim:
        errors.append("measurement_axes must match measurement_dim")
    if len(artifact.state_axes) != artifact.state_dim:
        errors.append("state_axes must match state_dim")
    for index, value in enumerate(artifact.times):
        if not _is_finite(value):
            errors.append(f"time[{index}] is not finite")
        if index > 0 and artifact.times[index] <= artifact.times[index - 1]:
            errors.append("times must be strictly increasing")
            break
    for index, value in enumerate(artifact.measurements):
        if not _is_finite(value):
            errors.append(f"measurement[{index}] is not finite")
    optional_series = {
        "true_position": artifact.true_position,
        "true_velocity": artifact.true_velocity,
        "true_acceleration": artifact.true_acceleration,
    }
    for name, series in optional_series.items():
        if series is not None and len(series) != len(artifact.times):
            errors.append(f"{name} must match the trajectory length when provided")
    for name, series in artifact.truth_series.items():
        if len(series) != len(artifact.times):
            errors.append(f"truth_series[{name}] must match the trajectory length")
            continue
        for index, value in enumerate(series):
            if not _is_finite(value):
                errors.append(f"truth_series[{name}][{index}] is not finite")
                break
    return errors


def validate_feature_artifact(artifact: FeatureArtifact) -> list[str]:
    errors: list[str] = []
    if not artifact.trajectory_id:
        errors.append("trajectory_id is required")
    if not artifact.feature_names:
        errors.append("feature_names are required")
    missing_units = [name for name in artifact.feature_names if name not in artifact.feature_units]
    if missing_units:
        errors.append(f"missing feature units for: {', '.join(missing_units)}")
    for row_index, row in enumerate(artifact.rows):
        for required in ("trajectory_id", "time", "window_start", "window_end"):
            if required not in row:
                errors.append(f"row {row_index} is missing {required}")
        for name in artifact.feature_names:
            value = row.get(name)
            if value is None or not _is_finite(float(value)):
                errors.append(f"row {row_index} feature {name} is not finite")
    return errors


def validate_classifier_output_artifact(artifact: ClassifierOutputArtifact) -> list[str]:
    errors: list[str] = []
    if not artifact.trajectory_id:
        errors.append("trajectory_id is required")
    if not artifact.class_names:
        errors.append("class_names are required")
    if artifact.sensor_regime_id is not None and not artifact.sensor_regime_id:
        errors.append("sensor_regime_id must not be empty when provided")
    if artifact.classifier_id is not None and not artifact.classifier_id:
        errors.append("classifier_id must not be empty when provided")
    posterior_columns = [f"posterior_{name}" for name in artifact.class_names]
    likelihood_columns = [f"log_likelihood_{name}" for name in artifact.class_names]
    for row_index, row in enumerate(artifact.rows):
        if row.get("trajectory_id") != artifact.trajectory_id:
            errors.append(f"row {row_index} has mismatched trajectory_id")
        if "time" not in row:
            errors.append(f"row {row_index} is missing time")
        if "true_class" not in row or "predicted_class" not in row:
            errors.append(f"row {row_index} is missing class labels")
        if "confidence" not in row:
            errors.append(f"row {row_index} is missing confidence")
        if "predicted_class" in row and "confidence" in row:
            posteriors = {name: float(row.get(name, 0.0)) for name in posterior_columns}
            if abs(sum(posteriors.values()) - 1.0) > 1e-6:
                errors.append(f"row {row_index} posterior probabilities do not sum to 1")
            for name, value in posteriors.items():
                if value < 0.0 or value > 1.0:
                    errors.append(f"row {row_index} posterior {name} is out of range")
            predicted = max(posteriors, key=posteriors.get).removeprefix("posterior_")
            if row.get("predicted_class") != predicted:
                errors.append(f"row {row_index} predicted_class does not match argmax posterior")
            if abs(float(row.get("confidence", 0.0)) - max(posteriors.values())) > 1e-6:
                errors.append(f"row {row_index} confidence does not match max posterior")
        if any(column not in row for column in posterior_columns):
            errors.append(f"row {row_index} is missing posterior columns")
        if any(column not in row for column in likelihood_columns):
            errors.append(f"row {row_index} is missing likelihood columns")
    return errors


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def _sample_feature_rows(trajectory: TrajectoryArtifact) -> tuple[dict[str, Any], ...]:
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


def _predict_sample_classifier(trajectory: TrajectoryArtifact, class_means: dict[str, float], class_sigma: float) -> tuple[dict[str, Any], ...]:
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


def _render_sample_report(run_name: str, trajectories: tuple[TrajectoryArtifact, ...], metrics: dict[str, Any]) -> str:
    lines = [
        f"# {run_name}",
        "",
        "## Experiment Summary",
        "",
        f"- Trajectories: {len(trajectories)}",
        f"- Final accuracy: {metrics['final_accuracy']:.3f}",
        f"- Step accuracy: {metrics['step_accuracy']:.3f}",
        "",
        "## Artifact Contract",
        "",
        "- Trajectory artifacts contain raw measurements and optional truth.",
        "- Feature artifacts use explicit time windows and documented units.",
        "- Classifier outputs carry posterior probabilities for every class.",
        "",
        "## Validation",
        "",
        "- Times are strictly increasing.",
        "- Posterior rows sum to one.",
        "- Predicted class equals posterior argmax.",
        "",
    ]
    return "\n".join(lines)


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

    _write_csv(
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
    _write_csv(
        feature_matrix_path,
        feature_rows,
        ["trajectory_id", "time", "window_start", "window_end", "running_mean", "running_range", "running_slope"],
    )
    _write_csv(
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
    _write_csv(
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


def validate_milestone0_sample_run_artifacts(artifacts: Milestone0SampleArtifacts) -> list[str]:
    errors: list[str] = []
    for path in (
        artifacts.config_path,
        artifacts.dataset_manifest_path,
        artifacts.class_definitions_path,
        artifacts.feature_manifest_path,
        artifacts.trajectory_path,
        artifacts.feature_matrix_path,
        artifacts.posterior_history_path,
        artifacts.likelihood_history_path,
        artifacts.metrics_path,
        artifacts.report_path,
    ):
        if not path.exists():
            errors.append(f"missing artifact: {path.name}")
    if artifacts.report_path.exists():
        report = artifacts.report_path.read_text(encoding="utf-8")
        for heading in ("## Experiment Summary", "## Artifact Contract", "## Validation"):
            if heading not in report:
                errors.append(f"report is missing heading: {heading}")
    if artifacts.metrics_path.exists():
        metrics = json.loads(artifacts.metrics_path.read_text(encoding="utf-8"))
        if metrics.get("trajectory_count") != 2:
            errors.append("trajectory_count should be 2")
        if abs(float(metrics.get("final_accuracy", 0.0)) - 1.0) > 1e-6:
            errors.append("final_accuracy should be 1.0")
    return errors
