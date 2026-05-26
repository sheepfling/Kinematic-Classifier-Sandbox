from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

from .utils.math import _is_finite


@dataclass(frozen=True, slots=True)
class TrajectoryArtifact:
    trajectory_id: str
    true_class: str
    scenario_id: str
    seed: int
    times: tuple[float, ...]
    measurements: tuple[Any, ...]
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
        if artifact.measurement_dim == 1:
            if not _is_finite(float(value)):
                errors.append(f"measurement[{index}] is not finite")
        else:
            if not isinstance(value, (tuple, list)):
                errors.append(f"measurement[{index}] must be a sequence for measurement_dim > 1")
                continue
            if len(value) != artifact.measurement_dim:
                errors.append(f"measurement[{index}] must match measurement_dim")
                continue
            for component_index, component in enumerate(value):
                if not _is_finite(float(component)):
                    errors.append(f"measurement[{index}][{component_index}] is not finite")
                    break
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



from .contracts_rendering import Milestone0SampleArtifacts, write_milestone0_sample_run_artifacts


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
