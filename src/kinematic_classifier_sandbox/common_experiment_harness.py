from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import io
import json
from math import exp, log, pi
import os
from pathlib import Path
import shutil
import statistics
from typing import Callable

from .common_dataset_comparison import (
    CommonComparisonResult,
    SCENARIO_MEASUREMENT_SIGMA,
    SCENARIO_TIMES,
    analyze_common_dataset_comparison,
)
from .common_1d_study_adapter import (
    ExecutablePairSpec,
    ExecutableTrajectory,
    build_reference_trajectory,
    build_pair_specs,
    generate_boundary_pair_dataset,
    generate_pair_dataset,
)
from .common_experiment_classifier_registry import (
    FamilyScoringContext,
    score_classifier_family,
)
from .coverage_report import load_classifier_manifest
from .feature_analysis import load_feature_set_manifest
from .feature_analysis import resolve_feature_names
from .shared_evaluation import sensor_regime_summary_rows
from .trajectory_generator import default_trajectory_class_definitions


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT / "experiments" / "common_1d_classifier_study"
CONFIG_PATH = EXPERIMENT_DIR / "common_experiment_config.yaml"
FEATURE_SET_PATH = EXPERIMENT_DIR / "feature_sets.json"
CLASS_PAIR_PATH = EXPERIMENT_DIR / "class_pair_manifest.json"
CLASSIFIER_MANIFEST_PATH = EXPERIMENT_DIR / "classifier_manifest.json"
BOUNDARY_EXPERIMENT_DIR = ROOT / "experiments" / "common_1d_boundary_study"

SCENARIO_FAMILY_MAP = {
    "easy": "nominal",
    "irregular": "irregular_sampling",
    "endpoint_match": "boundary",
    "short": "short_horizon",
    "short_noisy": "noise_stress",
    "outlier": "outlier_stress",
}

SCENARIO_TIER_MAP = {
    "easy": "easy_v1",
    "irregular": "realistic_v1",
    "endpoint_match": "boundary_v1",
    "short": "boundary_v1",
    "short_noisy": "stress_v1",
    "outlier": "adversarial_v1",
}


def _prepare_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp/kinematic-classifier-sandbox-mpl")))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _figure_to_png(fig) -> bytes:
    plt = _prepare_matplotlib()
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)

def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _safe_log(value: float) -> float:
    return log(max(value, 1e-12))


def _gaussian_logpdf(value: float, mean: float, sigma: float) -> float:
    variance = max(sigma * sigma, 1e-12)
    return -0.5 * (log(2.0 * pi * variance) + ((value - mean) ** 2) / variance)


def _logsumexp(values: list[float]) -> float:
    pivot = max(values)
    return pivot + log(sum(exp(value - pivot) for value in values))


def _normalize_scores(log_scores: dict[str, float]) -> dict[str, float]:
    normalizer = _logsumexp(list(log_scores.values()))
    return {name: exp(score - normalizer) for name, score in log_scores.items()}


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_value = _mean(values)
    return (sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def _median3(values: list[float], index: int) -> float:
    start = max(0, index - 1)
    stop = min(len(values), index + 2)
    window = sorted(values[start:stop])
    return window[len(window) // 2]


def _linear_fit(times: list[float], values: list[float]) -> tuple[float, float]:
    if len(times) != len(values) or not times:
        return 0.0, 0.0
    mean_t = _mean(times)
    mean_y = _mean(values)
    denominator = sum((time - mean_t) ** 2 for time in times)
    if abs(denominator) <= 1e-12:
        return mean_y, 0.0
    slope = sum((time - mean_t) * (value - mean_y) for time, value in zip(times, values)) / denominator
    intercept = mean_y - slope * mean_t
    return intercept, slope


def _quadratic_fit(times: list[float], values: list[float]) -> tuple[float, float, float]:
    if len(times) < 3:
        intercept, slope = _linear_fit(times, values)
        return intercept, slope, 0.0
    t0 = times[0]
    shifted = [time - t0 for time in times]
    s1 = len(shifted)
    s_t = sum(shifted)
    s_t2 = sum(time * time for time in shifted)
    s_t3 = sum(time * time * time for time in shifted)
    s_t4 = sum(time * time * time * time for time in shifted)
    s_y = sum(values)
    s_ty = sum(time * value for time, value in zip(shifted, values))
    s_t2y = sum(time * time * value for time, value in zip(shifted, values))
    matrix = [
        [float(s1), s_t, s_t2],
        [s_t, s_t2, s_t3],
        [s_t2, s_t3, s_t4],
    ]
    vector = [s_y, s_ty, s_t2y]

    augmented = [row[:] + [vector[index]] for index, row in enumerate(matrix)]
    size = 3
    for pivot_index in range(size):
        pivot_row = max(range(pivot_index, size), key=lambda row: abs(augmented[row][pivot_index]))
        if abs(augmented[pivot_row][pivot_index]) < 1e-12:
            intercept, slope = _linear_fit(times, values)
            return intercept, slope, 0.0
        augmented[pivot_index], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_index]
        pivot = augmented[pivot_index][pivot_index]
        for col_index in range(pivot_index, size + 1):
            augmented[pivot_index][col_index] /= pivot
        for row_index in range(size):
            if row_index == pivot_index:
                continue
            factor = augmented[row_index][pivot_index]
            for col_index in range(pivot_index, size + 1):
                augmented[row_index][col_index] -= factor * augmented[pivot_index][col_index]
    return augmented[0][3], augmented[1][3], augmented[2][3]


def _count_sign_changes(values: list[float], *, tolerance: float = 1e-9) -> int:
    filtered = [0 if abs(value) <= tolerance else (1 if value > 0.0 else -1) for value in values]
    filtered = [value for value in filtered if value != 0]
    if len(filtered) < 2:
        return 0
    return sum(1 for left, right in zip(filtered, filtered[1:]) if left != right)


def _scenario_family(scenario_id: str) -> str:
    return SCENARIO_FAMILY_MAP.get(scenario_id, "other")


def _scenario_tier(scenario_id: str) -> str:
    return SCENARIO_TIER_MAP.get(scenario_id, "other_v1")


def _trajectory_covariates(trajectory: ExecutableTrajectory) -> dict[str, float]:
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
    sigma = SCENARIO_MEASUREMENT_SIGMA.get(trajectory.scenario_id, 0.0)
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


@dataclass(frozen=True, slots=True)
class CommonExperimentSummary:
    experiment_name: str
    study_adapter_id: str
    executable_class_pairs: tuple[str, ...]
    trajectories_per_case: int
    num_pair_trajectories: int
    num_pair_predictions: int


@dataclass(frozen=True, slots=True)
class CommonExperimentConfig:
    experiment_name: str
    study_adapter_id: str
    config_path: Path
    output_dir_name: str
    dataset_generator_id: str
    declared_class_pairs: tuple[tuple[str, str], ...]
    output_filenames: dict[str, str]
    feature_sets_path: Path
    class_pair_manifest_path: Path
    classifier_manifest_path: Path


@dataclass(frozen=True, slots=True)
class CommonStudyAdapter:
    study_id: str
    description: str
    pair_spec_builder: Callable[[CommonExperimentConfig], tuple[ExecutablePairSpec, ...]]
    trajectory_generator: Callable[[tuple[ExecutablePairSpec, ...], int, int], tuple[ExecutableTrajectory, ...]]


@dataclass(frozen=True, slots=True)
class CommonExperimentResult:
    config: CommonExperimentConfig
    summary: CommonExperimentSummary
    comparison: CommonComparisonResult
    pair_prediction_rows: tuple[dict[str, object], ...]
    posterior_history_rows: tuple[dict[str, object], ...]
    likelihood_history_rows: tuple[dict[str, object], ...]
    feature_rows: tuple[dict[str, object], ...]
    metrics_by_classifier_rows: tuple[dict[str, object], ...]
    metrics_by_sensor_regime_rows: tuple[dict[str, object], ...]
    metrics_by_classifier_and_feature_set_rows: tuple[dict[str, object], ...]
    metrics_by_class_pair_rows: tuple[dict[str, object], ...]
    prior_sensitivity_rows: tuple[dict[str, object], ...]
    feature_set_comparison_rows: tuple[dict[str, object], ...]
    irregular_window_rows: tuple[dict[str, object], ...]
    class_pair_duration_rows: tuple[dict[str, object], ...]
    class_pair_scenario_rows: tuple[dict[str, object], ...]
    covariate_rows: tuple[dict[str, object], ...]
    feature_excitation_rows: tuple[dict[str, object], ...]
    identifiability_rows: tuple[dict[str, object], ...]
    oracle_rows: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class CommonExperimentArtifacts:
    run_dir: Path
    config_path: Path
    dataset_manifest_path: Path
    class_definitions_path: Path
    feature_manifest_path: Path
    feature_sets_path: Path
    class_pair_manifest_path: Path
    classifier_manifest_path: Path
    sensor_regimes_path: Path
    predictions_path: Path
    posterior_history_path: Path
    likelihood_history_path: Path
    feature_matrix_path: Path
    metrics_by_classifier_path: Path
    metrics_by_sensor_regime_path: Path
    metrics_by_classifier_and_feature_set_path: Path
    metrics_by_class_pair_path: Path
    prior_sensitivity_by_class_pair_path: Path
    feature_set_comparison_path: Path
    irregular_window_comparison_path: Path
    class_pair_duration_study_path: Path
    class_pair_scenario_study_path: Path
    covariate_leakage_audit_path: Path
    feature_excitation_matrix_path: Path
    identifiability_matrix_path: Path
    oracle_classifier_results_path: Path
    report_path: Path
    canonical_report_path: Path
    plots_dir: Path


def load_common_experiment_config(config_path: str | Path | None = None) -> CommonExperimentConfig:
    path = Path(config_path) if config_path is not None else CONFIG_PATH
    lines = path.read_text(encoding="utf-8").splitlines()
    experiment_name = "common_1d_classifier_study"
    output_dir_name = "common_1d_classifier_study"
    study_adapter_id = "common_1d_classifier_study"
    dataset_generator_id = "trajectory_generator_v1"
    declared_class_pairs: list[tuple[str, str]] = []
    output_filenames: dict[str, str] = {}
    feature_sets_path = FEATURE_SET_PATH
    class_pair_manifest_path = CLASS_PAIR_PATH
    classifier_manifest_path = CLASSIFIER_MANIFEST_PATH
    section = ""
    subsection = ""
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not raw_line.startswith(" "):
            section = stripped.rstrip(":")
            subsection = ""
            continue
        if raw_line.startswith("  ") and stripped.endswith(":") and not raw_line.startswith("    "):
            subsection = stripped.rstrip(":")
            continue
        if "name:" in stripped and section == "experiment":
            experiment_name = stripped.split(":", 1)[1].strip()
        elif "study_adapter_id:" in stripped and section == "experiment":
            study_adapter_id = stripped.split(":", 1)[1].strip()
        elif "output_dir:" in stripped and section == "experiment":
            output_dir_name = Path(stripped.split(":", 1)[1].strip()).name
        elif "generator:" in stripped and section == "dataset":
            dataset_generator_id = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("- [") and section == "dataset" and subsection == "class_pairs":
            inside = stripped[3:].strip()
            inside = inside.lstrip("[").rstrip("]")
            left, right = [item.strip() for item in inside.split(",", 1)]
            declared_class_pairs.append((left, right))
        elif "manifest_path:" in stripped and section == "feature_sets":
            feature_sets_path = ROOT / stripped.split(":", 1)[1].strip()
        elif "manifest_path:" in stripped and section == "class_pairs":
            class_pair_manifest_path = ROOT / stripped.split(":", 1)[1].strip()
        elif "manifest_path:" in stripped and section == "classifiers":
            classifier_manifest_path = ROOT / stripped.split(":", 1)[1].strip()
        elif ":" in stripped and section == "outputs":
            key, value = stripped.split(":", 1)
            output_filenames[key.strip()] = value.strip()
    return CommonExperimentConfig(
        experiment_name=experiment_name,
        study_adapter_id=study_adapter_id,
        config_path=path,
        output_dir_name=output_dir_name,
        dataset_generator_id=dataset_generator_id,
        declared_class_pairs=tuple(declared_class_pairs),
        output_filenames=output_filenames,
        feature_sets_path=feature_sets_path,
        class_pair_manifest_path=class_pair_manifest_path,
        classifier_manifest_path=classifier_manifest_path,
    )


def list_common_studies() -> tuple[CommonStudyAdapter, ...]:
    return (
        CommonStudyAdapter(
            study_id="common_1d_classifier_study",
            description="Manifest-driven 1D common experiment study.",
            pair_spec_builder=_parse_executable_pair_specs,
            trajectory_generator=generate_pair_dataset,
        ),
        CommonStudyAdapter(
            study_id="common_1d_boundary_study",
            description="Boundary-focused 1D common experiment study with harder scenarios only.",
            pair_spec_builder=_parse_executable_pair_specs,
            trajectory_generator=generate_boundary_pair_dataset,
        ),
    )


def resolve_common_study_adapter(study_id: str | CommonExperimentConfig) -> CommonStudyAdapter:
    resolved = study_id.study_adapter_id if isinstance(study_id, CommonExperimentConfig) else study_id
    for adapter in list_common_studies():
        if adapter.study_id == resolved:
            return adapter
    raise KeyError(f"unknown common study: {resolved}")


def _parse_executable_pair_specs(config: CommonExperimentConfig) -> tuple[ExecutablePairSpec, ...]:
    return build_pair_specs(
        declared_class_pairs=config.declared_class_pairs,
        class_pair_manifest_path=config.class_pair_manifest_path,
    )


def _trajectory_features(
    trajectory: ExecutableTrajectory,
    *,
    robust: bool,
) -> dict[str, float]:
    times = list(trajectory.times)
    measurements = list(trajectory.measurements)
    if robust and len(measurements) >= 3:
        measurements = [_median3(measurements, index) for index in range(len(measurements))]
    duration = times[-1] - times[0] if len(times) >= 2 else 0.0
    position_range = max(measurements) - min(measurements) if measurements else 0.0
    dt_values = [times[index] - times[index - 1] for index in range(1, len(times))]
    velocity = [
        (measurements[index] - measurements[index - 1]) / max(times[index] - times[index - 1], 1e-9)
        for index in range(1, len(times))
    ]
    acceleration = [
        (velocity[index] - velocity[index - 1]) / max(times[index + 1] - times[index], 1e-9)
        for index in range(1, len(velocity))
    ]
    speed_range = (max(velocity) - min(velocity)) if velocity else 0.0
    acceleration_range = (max(acceleration) - min(acceleration)) if acceleration else 0.0
    acceleration_variance = statistics.pvariance(acceleration) if len(acceleration) >= 2 else 0.0
    _, slope = _linear_fit(times, measurements)
    intercept_q, slope_q, curvature = _quadratic_fit(times, measurements)
    linear_residual = (
        sum((value - (intercept_q + slope * time)) ** 2 for time, value in zip(times, measurements)) / max(len(times), 1)
    ) ** 0.5
    quadratic_residual = (
        sum((value - (intercept_q + slope_q * (time - times[0]) + curvature * (time - times[0]) ** 2)) ** 2 for time, value in zip(times, measurements))
        / max(len(times), 1)
    ) ** 0.5
    monotonicity = sum(1 for left, right in zip(measurements, measurements[1:]) if right >= left) / max(len(measurements) - 1, 1)
    median_measurement = statistics.median(measurements) if measurements else 0.0
    outlier_score = max(abs(value - median_measurement) for value in measurements) if measurements else 0.0
    return {
        "duration": duration,
        "position_range": position_range,
        "speed_range": speed_range,
        "acceleration_range": acceleration_range,
        "acceleration_variance": acceleration_variance,
        "curvature_proxy": 2.0 * curvature,
        "velocity_sign_changes": float(_count_sign_changes(velocity)),
        "acceleration_sign_changes": float(_count_sign_changes(acceleration)),
        "monotonicity": monotonicity,
        "linear_fit_residual": linear_residual,
        "quadratic_fit_residual": quadratic_residual,
        "outlier_score": outlier_score,
    }


def _reference_trajectory(
    pair_spec: ExecutablePairSpec,
    class_name: str,
    scenario_id: str,
    times: tuple[float, ...],
) -> ExecutableTrajectory:
    return build_reference_trajectory(pair_spec, class_name, scenario_id, times)


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


def _pair_priors(class_a: str, class_b: str, prior_id: str) -> dict[str, float]:
    if prior_id == "uniform":
        return {class_a: 0.5, class_b: 0.5}
    if prior_id == "mild_bias":
        return {class_a: 0.65, class_b: 0.35}
    if prior_id == "strong_bias":
        return {class_a: 0.85, class_b: 0.15}
    raise KeyError(prior_id)


def _truncated_trajectory(
    trajectory: ExecutableTrajectory,
    prefix_length: int,
) -> tuple[tuple[float, ...], ExecutableTrajectory]:
    times = trajectory.times[:prefix_length]
    truncated = ExecutableTrajectory(
        trajectory_id=trajectory.trajectory_id,
        class_pair_id=trajectory.class_pair_id,
        class_a=trajectory.class_a,
        class_b=trajectory.class_b,
        true_class=trajectory.true_class,
        scenario_id=trajectory.scenario_id,
        seed=trajectory.seed,
        times=times,
        measurements=trajectory.measurements[:prefix_length],
        true_position=trajectory.true_position[:prefix_length],
        true_velocity=trajectory.true_velocity[:prefix_length],
        true_acceleration=trajectory.true_acceleration[:prefix_length],
    )
    return times, truncated


def _classifier_scores_for_prefix(
    classifier_entry: dict[str, object],
    pair_spec: ExecutablePairSpec,
    trajectory: ExecutableTrajectory,
    prefix_length: int,
    prior_weights: dict[str, float],
    feature_manifest: dict[str, dict[str, object]],
) -> dict[str, float]:
    times, truncated = _truncated_trajectory(trajectory, prefix_length)
    context = FamilyScoringContext(
        pair_spec=pair_spec,
        trajectory=trajectory,
        truncated=truncated,
        times=times,
        prior_weights=prior_weights,
        feature_manifest=feature_manifest,
        reference_builder=_reference_trajectory,
        feature_extractor=_trajectory_features,
        feature_sigma=_feature_sigma,
        gaussian_logpdf=_gaussian_logpdf,
        safe_log=_safe_log,
    )
    return score_classifier_family(classifier_entry, context)


def _feature_set_scores_for_prefix(
    *,
    feature_set_id: str,
    feature_entry: dict[str, object],
    pair_spec: ExecutablePairSpec,
    trajectory: ExecutableTrajectory,
    prefix_length: int,
    prior_weights: dict[str, float],
) -> dict[str, float]:
    times = trajectory.times[:prefix_length]
    truncated = ExecutableTrajectory(
        trajectory_id=trajectory.trajectory_id,
        class_pair_id=trajectory.class_pair_id,
        class_a=trajectory.class_a,
        class_b=trajectory.class_b,
        true_class=trajectory.true_class,
        scenario_id=trajectory.scenario_id,
        seed=trajectory.seed,
        times=times,
        measurements=trajectory.measurements[:prefix_length],
        true_position=trajectory.true_position[:prefix_length],
        true_velocity=trajectory.true_velocity[:prefix_length],
        true_acceleration=trajectory.true_acceleration[:prefix_length],
    )
    features = feature_entry.get("features")
    if not isinstance(features, list) or not features:
        raise KeyError(feature_set_id)
    robust = feature_set_id == "robust_extrema"
    observed = _trajectory_features(truncated, robust=robust)
    scores: dict[str, float] = {}
    for class_name in (pair_spec.class_a, pair_spec.class_b):
        reference = _reference_trajectory(pair_spec, class_name, trajectory.scenario_id, times)
        reference_features = _trajectory_features(reference, robust=robust)
        score = _safe_log(prior_weights[class_name])
        for feature_name in features:
            name = str(feature_name)
            score += _gaussian_logpdf(observed[name], reference_features[name], _feature_sigma(name))
        scores[class_name] = score
    return scores


def _slice_trailing_window(
    trajectory: ExecutableTrajectory,
    *,
    window_definition: str,
    window_sample_count: int,
    window_duration: float,
) -> ExecutableTrajectory:
    if window_definition == "sample_count":
        start = max(0, len(trajectory.times) - window_sample_count)
        selected_indices = list(range(start, len(trajectory.times)))
    elif window_definition == "elapsed_time":
        threshold = trajectory.times[-1] - window_duration
        selected_indices = [index for index, time in enumerate(trajectory.times) if time >= threshold]
        if not selected_indices:
            selected_indices = [len(trajectory.times) - 1]
    else:
        raise KeyError(window_definition)

    return ExecutableTrajectory(
        trajectory_id=trajectory.trajectory_id,
        class_pair_id=trajectory.class_pair_id,
        class_a=trajectory.class_a,
        class_b=trajectory.class_b,
        true_class=trajectory.true_class,
        scenario_id=trajectory.scenario_id,
        seed=trajectory.seed,
        times=tuple(trajectory.times[index] for index in selected_indices),
        measurements=tuple(trajectory.measurements[index] for index in selected_indices),
        true_position=tuple(trajectory.true_position[index] for index in selected_indices),
        true_velocity=tuple(trajectory.true_velocity[index] for index in selected_indices),
        true_acceleration=tuple(trajectory.true_acceleration[index] for index in selected_indices),
        measurement_dim=trajectory.measurement_dim,
        coordinate_frame=trajectory.coordinate_frame,
    )


def _feature_set_scores_for_window(
    *,
    feature_set_id: str,
    feature_manifest: dict[str, dict[str, object]],
    pair_spec: ExecutablePairSpec,
    trajectory: ExecutableTrajectory,
    window_definition: str,
    window_sample_count: int,
    window_duration: float,
    prior_weights: dict[str, float],
) -> tuple[dict[str, float], dict[str, float], int, float]:
    truncated = _slice_trailing_window(
        trajectory,
        window_definition=window_definition,
        window_sample_count=window_sample_count,
        window_duration=window_duration,
    )
    features = resolve_feature_names(feature_set=feature_set_id, manifest=feature_manifest)
    robust = feature_set_id == "robust_extrema"
    observed = _trajectory_features(truncated, robust=robust)
    scores: dict[str, float] = {}
    for class_name in (pair_spec.class_a, pair_spec.class_b):
        reference = _reference_trajectory(pair_spec, class_name, trajectory.scenario_id, truncated.times)
        reference_features = _trajectory_features(reference, robust=robust)
        score = _safe_log(prior_weights[class_name])
        for feature_name in features:
            score += _gaussian_logpdf(observed[feature_name], reference_features[feature_name], _feature_sigma(feature_name))
        scores[class_name] = score
    selected_duration = truncated.times[-1] - truncated.times[0] if len(truncated.times) >= 2 else 0.0
    return scores, observed, len(truncated.times), selected_duration


def _evaluate_executable_pairs(
    *,
    config: CommonExperimentConfig,
    pair_specs: tuple[ExecutablePairSpec, ...],
    trajectories: tuple[ExecutableTrajectory, ...],
) -> tuple[
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
]:
    classifier_manifest = load_classifier_manifest(config.classifier_manifest_path)
    feature_manifest = load_feature_set_manifest(config.feature_sets_path)
    pair_lookup = {spec.pair_id: spec for spec in pair_specs}
    feature_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    posterior_rows: list[dict[str, object]] = []
    likelihood_rows: list[dict[str, object]] = []
    metrics_by_pair: list[dict[str, object]] = []
    prior_rows: list[dict[str, object]] = []
    metrics_by_feature_set: list[dict[str, object]] = []

    trajectory_feature_cache: dict[tuple[str, bool], dict[str, float]] = {}
    for trajectory in trajectories:
        for feature_set_id in feature_manifest:
            robust = feature_set_id == "robust_extrema"
            key = (trajectory.trajectory_id, robust)
            if key not in trajectory_feature_cache:
                trajectory_feature_cache[key] = _trajectory_features(trajectory, robust=robust)
            feature_values = trajectory_feature_cache[key]
            row = {
                "trajectory_id": trajectory.trajectory_id,
                "class_pair_id": trajectory.class_pair_id,
                "scenario_id": trajectory.scenario_id,
                "scenario_family": _scenario_family(trajectory.scenario_id),
                "dataset_tier": _scenario_tier(trajectory.scenario_id),
                "true_class": trajectory.true_class,
                "feature_set_id": feature_set_id,
            }
            row.update(feature_values)
            feature_rows.append(row)

    grouped_predictions: dict[tuple[str, str], list[dict[str, object]]] = {}
    for classifier_entry in classifier_manifest:
        classifier_id = str(classifier_entry["id"])
        sensor_regime_id = "position_only"
        feature_set_id = str(
            classifier_entry.get(
                "feature_set_id",
                classifier_entry.get("requires_feature_set", "instantaneous"),
            )
        )
        for trajectory in trajectories:
            pair_spec = pair_lookup[trajectory.class_pair_id]
            prior = _pair_priors(pair_spec.class_a, pair_spec.class_b, "uniform")
            final_scores = _classifier_scores_for_prefix(
                classifier_entry,
                pair_spec,
                trajectory,
                len(trajectory.times),
                prior,
                feature_manifest,
            )
            final_weights = _normalize_scores(final_scores)
            predicted_class = max(final_weights, key=final_weights.get)
            run_id = f"{classifier_id}:{trajectory.trajectory_id}"
            prediction_row = {
                "run_id": run_id,
                "classifier_id": classifier_id,
                "feature_set_id": feature_set_id,
                "sensor_regime_id": sensor_regime_id,
                "measurement_dim": trajectory.measurement_dim,
                "coordinate_frame": trajectory.coordinate_frame,
                "class_pair_id": trajectory.class_pair_id,
                "class_a": pair_spec.class_a,
                "class_b": pair_spec.class_b,
                "trajectory_id": trajectory.trajectory_id,
                "scenario_id": trajectory.scenario_id,
                "scenario_family": _scenario_family(trajectory.scenario_id),
                "dataset_tier": _scenario_tier(trajectory.scenario_id),
                "time": trajectory.times[-1],
                "true_class": trajectory.true_class,
                "predicted_class": predicted_class,
                "confidence": final_weights[predicted_class],
                "posterior_class_a": final_weights[pair_spec.class_a],
                "posterior_class_b": final_weights[pair_spec.class_b],
            }
            prediction_rows.append(prediction_row)
            grouped_predictions.setdefault((classifier_id, trajectory.class_pair_id), []).append(prediction_row)

            for prefix_length in range(1, len(trajectory.times) + 1):
                scores = _classifier_scores_for_prefix(
                    classifier_entry,
                    pair_spec,
                    trajectory,
                    prefix_length,
                    prior,
                    feature_manifest,
                )
                weights = _normalize_scores(scores)
                posterior_rows.append(
                    {
                        "run_id": run_id,
                        "classifier_id": classifier_id,
                        "feature_set_id": feature_set_id,
                        "sensor_regime_id": sensor_regime_id,
                        "class_pair_id": trajectory.class_pair_id,
                        "class_a": pair_spec.class_a,
                        "class_b": pair_spec.class_b,
                        "trajectory_id": trajectory.trajectory_id,
                        "scenario_id": trajectory.scenario_id,
                        "scenario_family": _scenario_family(trajectory.scenario_id),
                        "dataset_tier": _scenario_tier(trajectory.scenario_id),
                        "time": trajectory.times[prefix_length - 1],
                        "true_class": trajectory.true_class,
                        "posterior_class_a": weights[pair_spec.class_a],
                        "posterior_class_b": weights[pair_spec.class_b],
                    }
                )
                likelihood_rows.append(
                    {
                        "run_id": run_id,
                        "classifier_id": classifier_id,
                        "feature_set_id": feature_set_id,
                        "sensor_regime_id": sensor_regime_id,
                        "class_pair_id": trajectory.class_pair_id,
                        "trajectory_id": trajectory.trajectory_id,
                        "scenario_id": trajectory.scenario_id,
                        "scenario_family": _scenario_family(trajectory.scenario_id),
                        "dataset_tier": _scenario_tier(trajectory.scenario_id),
                        "time": trajectory.times[prefix_length - 1],
                        "score_type": "log_likelihood_proxy",
                        "class_a": pair_spec.class_a,
                        "class_b": pair_spec.class_b,
                        "log_likelihood_class_a": scores[pair_spec.class_a],
                        "log_likelihood_class_b": scores[pair_spec.class_b],
                    }
                )

        # prior study rows per classifier/pair
        for pair_spec in pair_specs:
            pair_trajectories = [trajectory for trajectory in trajectories if trajectory.class_pair_id == pair_spec.pair_id]
            for prior_id in ("uniform", "mild_bias", "strong_bias"):
                prior = _pair_priors(pair_spec.class_a, pair_spec.class_b, prior_id)
                accuracy_hits = 0
                for trajectory in pair_trajectories:
                    weights = _normalize_scores(
                        _classifier_scores_for_prefix(
                            classifier_entry,
                            pair_spec,
                            trajectory,
                            len(trajectory.times),
                            prior,
                            feature_manifest,
                        )
                    )
                    predicted = max(weights, key=weights.get)
                    accuracy_hits += 1 if predicted == trajectory.true_class else 0
                prior_rows.append(
                    {
                        "classifier_id": classifier_id,
                        "class_pair_id": pair_spec.pair_id,
                        "prior_id": prior_id,
                        "accuracy": accuracy_hits / max(len(pair_trajectories), 1),
                    }
                )

    for (classifier_id, pair_id), rows in sorted(grouped_predictions.items()):
        accuracy = sum(1 for row in rows if row["predicted_class"] == row["true_class"]) / max(len(rows), 1)
        metrics_by_pair.append(
            {
                "classifier_id": classifier_id,
                "class_pair": pair_id,
                "overall_accuracy": accuracy,
                "status": "executed",
            }
        )

    feature_set_accuracy: dict[tuple[str, str], list[float]] = {}
    for row in prediction_rows:
        key = (str(row["classifier_id"]), str(row["feature_set_id"]))
        feature_set_accuracy.setdefault(key, []).append(1.0 if row["predicted_class"] == row["true_class"] else 0.0)
    for (classifier_id, feature_set_id), values in sorted(feature_set_accuracy.items()):
        metrics_by_feature_set.append(
            {
                "classifier_id": classifier_id,
                "feature_set_id": feature_set_id,
                "overall_accuracy": _mean(values),
            }
        )

    return (
        tuple(prediction_rows),
        tuple(posterior_rows),
        tuple(likelihood_rows),
        tuple(feature_rows),
        tuple(metrics_by_pair),
        tuple(prior_rows),
        tuple(metrics_by_feature_set),
    )


def _metrics_by_classifier(prediction_rows: tuple[dict[str, object], ...]) -> tuple[dict[str, object], ...]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in prediction_rows:
        grouped.setdefault(str(row["classifier_id"]), []).append(row)
    rows: list[dict[str, object]] = []
    for classifier_id, classifier_rows in sorted(grouped.items()):
        accuracy = sum(1 for row in classifier_rows if row["predicted_class"] == row["true_class"]) / max(len(classifier_rows), 1)
        rows.append(
            {
                "classifier_id": classifier_id,
                "overall_accuracy": accuracy,
                "num_predictions": len(classifier_rows),
            }
        )
    return tuple(rows)


def _metrics_by_sensor_regime(prediction_rows: tuple[dict[str, object], ...]) -> tuple[dict[str, object], ...]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in prediction_rows:
        grouped.setdefault(str(row["sensor_regime_id"]), []).append(row)
    rows: list[dict[str, object]] = []
    for sensor_regime_id, regime_rows in sorted(grouped.items()):
        hits = [1.0 if row["predicted_class"] == row["true_class"] else 0.0 for row in regime_rows]
        confidences = [float(row["confidence"]) for row in regime_rows]
        classifier_ids = {str(row["classifier_id"]) for row in regime_rows}
        measurement_dims = sorted({int(row["measurement_dim"]) for row in regime_rows})
        coordinate_frames = sorted({str(row["coordinate_frame"]) for row in regime_rows})
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


def _covariate_rows(trajectories: tuple[ExecutableTrajectory, ...]) -> tuple[dict[str, object], ...]:
    grouped: dict[tuple[str, str, str, str], list[ExecutableTrajectory]] = {}
    pair_tier_values: dict[tuple[str, str], dict[str, list[float]]] = {}
    for trajectory in trajectories:
        dataset_tier = _scenario_tier(trajectory.scenario_id)
        scenario_family = _scenario_family(trajectory.scenario_id)
        grouped.setdefault(
            (trajectory.class_pair_id, dataset_tier, scenario_family, trajectory.true_class),
            [],
        ).append(trajectory)
        pair_tier_key = (trajectory.class_pair_id, dataset_tier)
        pair_tier_values.setdefault(pair_tier_key, {})
        covariates = _trajectory_covariates(trajectory)
        for name, value in covariates.items():
            pair_tier_values[pair_tier_key].setdefault(name, []).append(value)

    rows: list[dict[str, object]] = []
    audited_covariates = (
        "duration",
        "sample_count",
        "measurement_std",
        "outlier_fraction",
        "sampling_irregularity",
    )
    for (pair_id, dataset_tier, scenario_family, true_class), selected in sorted(grouped.items()):
        covariate_rows = [_trajectory_covariates(trajectory) for trajectory in selected]
        mean_values = {
            name: _mean([row[name] for row in covariate_rows])
            for name in covariate_rows[0]
        }
        baseline = {
            name: _mean(pair_tier_values[(pair_id, dataset_tier)][name])
            for name in covariate_rows[0]
        }
        delta_ratios = {
            name: abs(mean_values[name] - baseline[name]) / max(abs(baseline[name]), 1e-6)
            for name in audited_covariates
        }
        max_delta_name = max(delta_ratios, key=delta_ratios.get)
        max_delta_ratio = delta_ratios[max_delta_name]
        status = "pass" if max_delta_ratio <= 0.20 else ("warn" if max_delta_ratio <= 0.40 else "fail")
        rows.append(
            {
                "class_pair_id": pair_id,
                "dataset_tier": dataset_tier,
                "scenario_family": scenario_family,
                "true_class": true_class,
                "num_trajectories": len(selected),
                "mean_duration": mean_values["duration"],
                "mean_sample_count": mean_values["sample_count"],
                "mean_dt": mean_values["mean_dt"],
                "std_dt": mean_values["std_dt"],
                "max_dt": mean_values["max_dt"],
                "sampling_irregularity": mean_values["sampling_irregularity"],
                "measurement_std": mean_values["measurement_std"],
                "outlier_fraction": mean_values["outlier_fraction"],
                "max_covariate_delta_name": max_delta_name,
                "max_covariate_delta_ratio": max_delta_ratio,
                "status": status,
            }
        )
    return tuple(rows)


def _feature_excitation_rows(feature_rows: tuple[dict[str, object], ...]) -> tuple[dict[str, object], ...]:
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
    rows: list[dict[str, object]] = []
    for (pair_id, dataset_tier, scenario_family, feature_set_id), selected in sorted(grouped.items()):
        row: dict[str, object] = {
            "class_pair_id": pair_id,
            "dataset_tier": dataset_tier,
            "scenario_family": scenario_family,
            "feature_set_id": feature_set_id,
            "num_rows": len(selected),
        }
        for feature_name in feature_names:
            values = [float(item[feature_name]) for item in selected]
            row[f"{feature_name}_mean_abs"] = _mean([abs(value) for value in values])
            row[f"{feature_name}_std"] = _std(values)
        rows.append(row)
    return tuple(rows)


def _feature_set_comparison_rows(
    *,
    config: CommonExperimentConfig,
    pair_specs: tuple[ExecutablePairSpec, ...],
    trajectories: tuple[ExecutableTrajectory, ...],
) -> tuple[dict[str, object], ...]:
    feature_manifest = load_feature_set_manifest(config.feature_sets_path)
    pair_lookup = {spec.pair_id: spec for spec in pair_specs}
    rows: list[dict[str, object]] = []
    for feature_set_id, entry in sorted(feature_manifest.items()):
        features = entry.get("features")
        if not isinstance(features, list) or not features:
            continue
        hits: list[float] = []
        pair_hits: dict[str, list[float]] = {}
        confidence_values: list[float] = []
        for trajectory in trajectories:
            pair_spec = pair_lookup[trajectory.class_pair_id]
            prior = _pair_priors(pair_spec.class_a, pair_spec.class_b, "uniform")
            scores = _feature_set_scores_for_prefix(
                feature_set_id=feature_set_id,
                feature_entry=entry,
                pair_spec=pair_spec,
                trajectory=trajectory,
                prefix_length=len(trajectory.times),
                prior_weights=prior,
            )
            weights = _normalize_scores(scores)
            predicted = max(weights, key=weights.get)
            hit = 1.0 if predicted == trajectory.true_class else 0.0
            hits.append(hit)
            pair_hits.setdefault(trajectory.class_pair_id, []).append(hit)
            confidence_values.append(max(weights.values()))
        pair_accuracies = [_mean(values) for values in pair_hits.values()]
        rows.append(
            {
                "feature_set_id": feature_set_id,
                "history_behavior": str(entry.get("history_behavior", "unknown")),
                "num_features": len(features),
                "overall_accuracy": _mean(hits),
                "min_pair_accuracy": min(pair_accuracies) if pair_accuracies else 0.0,
                "max_pair_accuracy": max(pair_accuracies) if pair_accuracies else 0.0,
                "mean_confidence": _mean(confidence_values),
            }
        )
    return tuple(rows)


def _irregular_window_comparison_rows(
    *,
    config: CommonExperimentConfig,
    pair_specs: tuple[ExecutablePairSpec, ...],
    trajectories: tuple[ExecutableTrajectory, ...],
) -> tuple[dict[str, object], ...]:
    feature_manifest = load_feature_set_manifest(config.feature_sets_path)
    pair_lookup = {spec.pair_id: spec for spec in pair_specs}
    irregular_trajectories = [trajectory for trajectory in trajectories if trajectory.scenario_id == "irregular"]
    if not irregular_trajectories:
        return ()

    window_sample_count = 4
    window_duration = 3.0
    per_def_rows: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    paired_results: dict[tuple[str, str, str], dict[str, object]] = {}

    for feature_set_id, entry in sorted(feature_manifest.items()):
        try:
            resolve_feature_names(feature_set=feature_set_id, manifest=feature_manifest)
        except (KeyError, ValueError):
            continue
        for trajectory in irregular_trajectories:
            pair_spec = pair_lookup[trajectory.class_pair_id]
            prior = _pair_priors(pair_spec.class_a, pair_spec.class_b, "uniform")
            trajectory_results: dict[str, dict[str, object]] = {}
            for window_definition in ("sample_count", "elapsed_time"):
                scores, observed, selected_count, selected_duration = _feature_set_scores_for_window(
                    feature_set_id=feature_set_id,
                    feature_manifest=feature_manifest,
                    pair_spec=pair_spec,
                    trajectory=trajectory,
                    window_definition=window_definition,
                    window_sample_count=window_sample_count,
                    window_duration=window_duration,
                    prior_weights=prior,
                )
                weights = _normalize_scores(scores)
                predicted_class = max(weights, key=weights.get)
                trajectory_results[window_definition] = {
                    "predicted_class": predicted_class,
                    "confidence": weights[predicted_class],
                    "selected_sample_count": selected_count,
                    "selected_duration": selected_duration,
                    "features": observed,
                }
                per_def_rows.setdefault((trajectory.class_pair_id, feature_set_id, window_definition), []).append(
                    {
                        "hit": 1.0 if predicted_class == trajectory.true_class else 0.0,
                        "confidence": weights[predicted_class],
                        "selected_sample_count": float(selected_count),
                        "selected_duration": selected_duration,
                    }
                )
            paired_results[(trajectory.class_pair_id, feature_set_id, trajectory.trajectory_id)] = trajectory_results

    rows: list[dict[str, object]] = []
    for (class_pair_id, feature_set_id, window_definition), selected in sorted(per_def_rows.items()):
        paired = [
            result
            for (pair_id, feature_id, _), result in paired_results.items()
            if pair_id == class_pair_id and feature_id == feature_set_id
        ]
        disagreement_flags = []
        feature_delta_values = []
        for result in paired:
            sample_result = result["sample_count"]
            elapsed_result = result["elapsed_time"]
            disagreement_flags.append(
                1.0 if sample_result["predicted_class"] != elapsed_result["predicted_class"] else 0.0
            )
            shared_feature_names = set(sample_result["features"]) & set(elapsed_result["features"])
            deltas = [
                abs(float(sample_result["features"][name]) - float(elapsed_result["features"][name]))
                for name in shared_feature_names
            ]
            feature_delta_values.append(_mean(deltas) if deltas else 0.0)
        rows.append(
            {
                "class_pair_id": class_pair_id,
                "feature_set_id": feature_set_id,
                "history_behavior": str(feature_manifest[feature_set_id].get("history_behavior", "unknown")),
                "window_definition": window_definition,
                "window_sample_count": window_sample_count,
                "window_duration": window_duration,
                "num_predictions": len(selected),
                "overall_accuracy": _mean([float(row["hit"]) for row in selected]),
                "mean_confidence": _mean([float(row["confidence"]) for row in selected]),
                "mean_selected_sample_count": _mean([float(row["selected_sample_count"]) for row in selected]),
                "mean_selected_duration": _mean([float(row["selected_duration"]) for row in selected]),
                "cross_window_prediction_disagreement_rate": _mean(disagreement_flags),
                "mean_cross_window_feature_delta": _mean(feature_delta_values),
            }
        )
    return tuple(rows)


def _class_pair_duration_rows(
    posterior_rows: tuple[dict[str, object], ...],
) -> tuple[dict[str, object], ...]:
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
    rows: list[dict[str, object]] = []
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
            {
                "classifier_id": classifier_id,
                "class_pair_id": class_pair_id,
                "time": time_value,
                "num_prefixes": len(selected),
                "prefix_accuracy": hits / max(len(selected), 1),
                "mean_confidence": confidence_sum / max(len(selected), 1),
                "posterior_margin": margin_sum / max(len(selected), 1),
            }
        )
    return tuple(rows)


def _class_pair_scenario_rows(
    prediction_rows: tuple[dict[str, object], ...],
) -> tuple[dict[str, object], ...]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    scenario_family = {
        "easy": "nominal",
        "irregular": "irregular_sampling",
        "endpoint_match": "boundary",
        "short": "short_horizon",
        "short_noisy": "noise_stress",
        "outlier": "outlier_stress",
    }
    for row in prediction_rows:
        grouped.setdefault(
            (
                str(row["classifier_id"]),
                str(row["class_pair_id"]),
                str(row["scenario_id"]),
            ),
            [],
        ).append(row)
    rows: list[dict[str, object]] = []
    for (classifier_id, class_pair_id, scenario_id), selected in sorted(grouped.items()):
        rows.append(
            {
                "classifier_id": classifier_id,
                "class_pair_id": class_pair_id,
                "scenario_id": scenario_id,
                "scenario_family": scenario_family.get(scenario_id, "other"),
                "overall_accuracy": _mean(
                    [1.0 if row["predicted_class"] == row["true_class"] else 0.0 for row in selected]
                ),
                "mean_confidence": _mean([float(row["confidence"]) for row in selected]),
                "num_predictions": len(selected),
            }
        )
    return tuple(rows)


def _identifiability_rows(
    feature_rows: tuple[dict[str, object], ...],
    *,
    feature_manifest: dict[str, dict[str, object]],
) -> tuple[dict[str, object], ...]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in feature_rows:
        grouped.setdefault((str(row["class_pair_id"]), str(row["feature_set_id"])), []).append(row)

    rows: list[dict[str, object]] = []
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


def _oracle_rows(
    feature_rows: tuple[dict[str, object], ...],
    *,
    feature_manifest: dict[str, dict[str, object]],
) -> tuple[dict[str, object], ...]:
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
        }
        pair_rows.setdefault(pair_id, []).append(feature_row)

    rows: list[dict[str, object]] = []
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


def analyze_common_experiment(
    *,
    config_path: str | Path | None = None,
    seed: int = 7,
    trajectories_per_case: int = 8,
) -> CommonExperimentResult:
    config = load_common_experiment_config(config_path)
    study_adapter = resolve_common_study_adapter(config)
    comparison = analyze_common_dataset_comparison(seed=seed, trajectories_per_case=trajectories_per_case)
    pair_specs = study_adapter.pair_spec_builder(config)
    trajectories = study_adapter.trajectory_generator(pair_specs, seed, trajectories_per_case)
    return _analyze_common_trajectory_corpus(
        config=config,
        comparison=comparison,
        pair_specs=pair_specs,
        trajectories=trajectories,
        trajectories_per_case=trajectories_per_case,
    )


def _analyze_common_trajectory_corpus(
    *,
    config: CommonExperimentConfig,
    comparison: CommonComparisonResult,
    pair_specs: tuple[ExecutablePairSpec, ...],
    trajectories: tuple[ExecutableTrajectory, ...],
    trajectories_per_case: int,
) -> CommonExperimentResult:
    feature_manifest = load_feature_set_manifest(config.feature_sets_path)
    (
        pair_prediction_rows,
        posterior_history_rows,
        likelihood_history_rows,
        feature_rows,
        metrics_by_class_pair_rows,
        prior_sensitivity_rows,
        metrics_by_classifier_and_feature_set_rows,
    ) = _evaluate_executable_pairs(config=config, pair_specs=pair_specs, trajectories=trajectories)
    metrics_by_classifier_rows = _metrics_by_classifier(pair_prediction_rows)
    metrics_by_sensor_regime_rows = _metrics_by_sensor_regime(pair_prediction_rows)
    feature_set_comparison_rows = _feature_set_comparison_rows(
        config=config,
        pair_specs=pair_specs,
        trajectories=trajectories,
    )
    irregular_window_rows = _irregular_window_comparison_rows(
        config=config,
        pair_specs=pair_specs,
        trajectories=trajectories,
    )
    class_pair_duration_rows = _class_pair_duration_rows(posterior_history_rows)
    class_pair_scenario_rows = _class_pair_scenario_rows(pair_prediction_rows)
    covariate_rows = _covariate_rows(trajectories)
    feature_excitation_rows = _feature_excitation_rows(feature_rows)
    identifiability_rows = _identifiability_rows(feature_rows, feature_manifest=feature_manifest)
    oracle_rows = _oracle_rows(feature_rows, feature_manifest=feature_manifest)
    summary = CommonExperimentSummary(
        experiment_name=config.experiment_name,
        study_adapter_id=config.study_adapter_id,
        executable_class_pairs=tuple(spec.pair_id for spec in pair_specs),
        trajectories_per_case=trajectories_per_case,
        num_pair_trajectories=len(trajectories),
        num_pair_predictions=len(pair_prediction_rows),
    )
    return CommonExperimentResult(
        config=config,
        summary=summary,
        comparison=comparison,
        pair_prediction_rows=pair_prediction_rows,
        posterior_history_rows=posterior_history_rows,
        likelihood_history_rows=likelihood_history_rows,
        feature_rows=feature_rows,
        metrics_by_classifier_rows=metrics_by_classifier_rows,
        metrics_by_sensor_regime_rows=metrics_by_sensor_regime_rows,
        metrics_by_classifier_and_feature_set_rows=metrics_by_classifier_and_feature_set_rows,
        metrics_by_class_pair_rows=metrics_by_class_pair_rows,
        prior_sensitivity_rows=prior_sensitivity_rows,
        feature_set_comparison_rows=feature_set_comparison_rows,
        irregular_window_rows=irregular_window_rows,
        class_pair_duration_rows=class_pair_duration_rows,
        class_pair_scenario_rows=class_pair_scenario_rows,
        covariate_rows=covariate_rows,
        feature_excitation_rows=feature_excitation_rows,
        identifiability_rows=identifiability_rows,
        oracle_rows=oracle_rows,
    )


def analyze_common_trajectory_corpus(
    *,
    pair_specs: tuple[ExecutablePairSpec, ...],
    trajectories: tuple[ExecutableTrajectory, ...],
    config_path: str | Path | None = None,
    seed: int = 7,
    trajectories_per_case: int | None = None,
) -> CommonExperimentResult:
    config = load_common_experiment_config(config_path)
    comparison = analyze_common_dataset_comparison(
        seed=seed,
        trajectories_per_case=trajectories_per_case or max(len(trajectories), 1),
    )
    return _analyze_common_trajectory_corpus(
        config=config,
        comparison=comparison,
        pair_specs=pair_specs,
        trajectories=trajectories,
        trajectories_per_case=trajectories_per_case or max(len(trajectories), 1),
    )


def render_common_experiment_report(result: CommonExperimentResult) -> str:
    executed_pairs = ", ".join(result.summary.executable_class_pairs)
    classifier_lines = "\n".join(
        f"| {row['classifier_id']} | {float(row['overall_accuracy']):.3f} | {int(row['num_predictions'])} |"
        for row in result.metrics_by_classifier_rows
    )
    sensor_regime_lines = "\n".join(
        f"| {row['sensor_regime_id']} | {row['same_sensor_fairness_bucket']} | {float(row['overall_accuracy']):.3f} | {float(row['mean_confidence']):.3f} | {int(row['num_predictions'])} | {int(row['num_classifiers'])} |"
        for row in result.metrics_by_sensor_regime_rows
    )
    pair_lines = "\n".join(
        f"| {row['classifier_id']} | {row['class_pair']} | {float(row['overall_accuracy']):.3f} | {row['status']} |"
        for row in result.metrics_by_class_pair_rows
    )
    feature_set_lines = "\n".join(
        f"| {row['feature_set_id']} | {row['history_behavior']} | {int(row['num_features'])} | {float(row['overall_accuracy']):.3f} | {float(row['min_pair_accuracy']):.3f} |"
        for row in result.feature_set_comparison_rows
    )
    irregular_window_lines = "\n".join(
        f"| {row['class_pair_id']} | {row['feature_set_id']} | {row['window_definition']} | {float(row['overall_accuracy']):.3f} | {float(row['mean_selected_duration']):.2f} | {float(row['cross_window_prediction_disagreement_rate']):.3f} |"
        for row in result.irregular_window_rows[:18]
    )
    duration_lines = "\n".join(
        f"| {row['classifier_id']} | {row['class_pair_id']} | {float(row['time']):.2f} | {float(row['prefix_accuracy']):.3f} | {float(row['mean_confidence']):.3f} |"
        for row in result.class_pair_duration_rows[:12]
    )
    scenario_lines = "\n".join(
        f"| {row['classifier_id']} | {row['class_pair_id']} | {row['scenario_id']} | {row['scenario_family']} | {float(row['overall_accuracy']):.3f} |"
        for row in result.class_pair_scenario_rows[:12]
    )
    covariate_lines = "\n".join(
        f"| {row['class_pair_id']} | {row['dataset_tier']} | {row['scenario_family']} | {row['true_class']} | {float(row['mean_duration']):.2f} | {float(row['measurement_std']):.2f} | {float(row['max_covariate_delta_ratio']):.2f} | {row['status']} |"
        for row in result.covariate_rows[:12]
    )
    excitation_lines = "\n".join(
        f"| {row['class_pair_id']} | {row['dataset_tier']} | {row['scenario_family']} | {row['feature_set_id']} | {int(row['num_rows'])} | {float(row['position_range_mean_abs']):.2f} | {float(row['curvature_proxy_mean_abs']):.2f} |"
        for row in result.feature_excitation_rows[:12]
    )
    identifiability_lines = "\n".join(
        f"| {row['class_pair_id']} | {row['feature_set_id']} | {float(row['mean_standardized_feature_distance']):.3f} | {float(row['overlap_estimate']):.3f} | {row['identifiability_status']} |"
        for row in result.identifiability_rows[:18]
    )
    oracle_lines = "\n".join(
        f"| {row['class_pair_id']} | {row['feature_set_id']} | {float(row['oracle_accuracy']):.3f} | {float(row['mean_confidence']):.3f} | {float(row['mean_posterior_margin']):.3f} | {row['is_best_feature_set']} |"
        for row in result.oracle_rows[:18]
    )
    return "\n".join(
        [
            "# Common Experiment Harness",
            "",
            "Milestone 10 and Milestone 11 executable common-study subset for the manifest-driven 1D classifier study.",
            "",
            "## Summary",
            "",
            f"- Experiment: `{result.summary.experiment_name}`",
            f"- Study adapter: `{result.summary.study_adapter_id}`",
            f"- Executable class pairs: `{executed_pairs}`",
            f"- Pair trajectories: `{result.summary.num_pair_trajectories}`",
            f"- Pair predictions: `{result.summary.num_pair_predictions}`",
            f"- Shared-comparison methods: `{len(result.comparison.rows)}`",
            "",
            "## Metrics By Classifier",
            "",
            "| classifier_id | overall_accuracy | num_predictions |",
            "| --- | ---: | ---: |",
            classifier_lines,
            "",
            "## Metrics By Sensor Regime",
            "",
            "| sensor_regime_id | same_sensor_fairness_bucket | overall_accuracy | mean_confidence | num_predictions | num_classifiers |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
            sensor_regime_lines,
            "",
            "## Metrics By Class Pair",
            "",
            "| classifier_id | class_pair | overall_accuracy | status |",
            "| --- | --- | ---: | --- |",
            pair_lines,
            "",
            "## Feature-Set Study",
            "",
            "| feature_set_id | history_behavior | num_features | overall_accuracy | min_pair_accuracy |",
            "| --- | --- | ---: | ---: | ---: |",
            feature_set_lines,
            "",
            "## Irregular Window Comparison",
            "",
            "| class_pair_id | feature_set_id | window_definition | overall_accuracy | mean_selected_duration | cross_window_prediction_disagreement_rate |",
            "| --- | --- | --- | ---: | ---: | ---: |",
            irregular_window_lines or "| n/a | n/a | n/a | 0.000 | 0.00 | 0.000 |",
            "",
            "## Class-Pair Duration Study",
            "",
            "| classifier_id | class_pair_id | time | prefix_accuracy | mean_confidence |",
            "| --- | --- | ---: | ---: | ---: |",
            duration_lines,
            "",
            "## Class-Pair Scenario Study",
            "",
            "| classifier_id | class_pair_id | scenario_id | scenario_family | overall_accuracy |",
            "| --- | --- | --- | --- | ---: |",
            scenario_lines,
            "",
            "## Covariate Leakage Audit",
            "",
            "| class_pair_id | dataset_tier | scenario_family | true_class | mean_duration | measurement_std | max_covariate_delta_ratio | status |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
            covariate_lines,
            "",
            "## Feature Excitation Matrix",
            "",
            "| class_pair_id | dataset_tier | scenario_family | feature_set_id | num_rows | position_range_mean_abs | curvature_proxy_mean_abs |",
            "| --- | --- | --- | --- | ---: | ---: | ---: |",
            excitation_lines,
            "",
            "## Identifiability Matrix",
            "",
            "| class_pair_id | feature_set_id | mean_standardized_feature_distance | overlap_estimate | identifiability_status |",
            "| --- | --- | ---: | ---: | --- |",
            identifiability_lines,
            "",
            "## Oracle Separability Baseline",
            "",
            "| class_pair_id | feature_set_id | oracle_accuracy | mean_confidence | mean_posterior_margin | is_best_feature_set |",
            "| --- | --- | ---: | ---: | ---: | --- |",
            oracle_lines,
            "",
            "## Notes",
            "",
            "- This artifact keeps the M10 contract honest by emitting one unified run folder from the experiment manifests.",
            "- The executable subset now includes a hard shape pair: `maneuver_vs_bounded_acceleration`.",
            "- `unified_likelihood_history.csv` currently stores standardized log-likelihood proxies so every classifier family can share one artifact surface.",
            "- The shared `comparison` block remains the common binary study used elsewhere in the repo; the pairwise outputs here are the manifest-aligned executable subset.",
            "- The M11 additions promote feature-bundle comparison and pairwise duration/scenario slices to explicit artifacts instead of leaving them implicit in the raw prediction table.",
            "- The M13 additions audit class-linked covariates by tier and preserve dataset-tier context in the feature excitation matrix.",
            "- The common run folder now emits `identifiability_matrix.csv`, `report.md`, and a minimal `plots/` pack so the experiment artifact surface matches the roadmap contract more closely.",
            "- The M14 oracle rows are now feature-only leave-one-out separability baselines, independent of the production classifier ladder.",
            "- The M15 irregular-window study compares fixed sample-count windows against elapsed-time windows on irregularly sampled trajectories.",
        ]
    )


def _write_plot(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_figure_to_png(fig))


def _write_common_experiment_plot_pack(
    run_dir: Path,
    *,
    result: CommonExperimentResult,
) -> Path:
    plots_dir = run_dir / "plots"
    overview_dir = plots_dir / "overview"
    trajectory_dir = plots_dir / "single_trajectory_examples"
    posterior_dir = plots_dir / "posteriors"
    likelihood_dir = plots_dir / "likelihoods"
    confusion_dir = plots_dir / "confusion_matrices"
    monte_carlo_dir = plots_dir / "monte_carlo"
    feature_dir = plots_dir / "feature_space"
    prior_dir = plots_dir / "priors"
    pca_dir = plots_dir / "pca"
    class_pair_dir = plots_dir / "class_pair_reports"
    for directory in (
        overview_dir,
        trajectory_dir,
        posterior_dir,
        likelihood_dir,
        confusion_dir,
        monte_carlo_dir,
        feature_dir,
        prior_dir,
        pca_dir,
        class_pair_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    plt = _prepare_matplotlib()

    scenario_counts: dict[str, int] = {}
    for row in result.pair_prediction_rows:
        scenario_counts.setdefault(str(row["scenario_id"]), 0)
        scenario_counts[str(row["scenario_id"])] += 1
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    scenario_names = list(sorted(scenario_counts))
    ax.bar(scenario_names, [scenario_counts[name] for name in scenario_names], color="#4C78A8")
    ax.set_title("Dataset Balance by Scenario")
    ax.set_ylabel("Predictions")
    ax.tick_params(axis="x", rotation=25)
    _write_plot(fig, overview_dir / "dataset_balance.png")

    if result.pair_prediction_rows:
        selected_run_id = str(result.pair_prediction_rows[0]["run_id"])
        selected_posteriors = [row for row in result.posterior_history_rows if str(row["run_id"]) == selected_run_id]
        selected_likelihoods = [row for row in result.likelihood_history_rows if str(row["run_id"]) == selected_run_id]
        selected_prediction = next(row for row in result.pair_prediction_rows if str(row["run_id"]) == selected_run_id)
        selected_trajectory_id = str(selected_prediction["trajectory_id"])
        selected_features = [row for row in result.feature_rows if str(row["trajectory_id"]) == selected_trajectory_id]

        fig, ax = plt.subplots(figsize=(9.5, 4.6))
        ax.plot(
            [float(row["time"]) for row in selected_posteriors],
            [float(row["posterior_class_a"]) for row in selected_posteriors],
            label=str(selected_prediction["class_a"]),
            linewidth=2.0,
        )
        ax.plot(
            [float(row["time"]) for row in selected_posteriors],
            [float(row["posterior_class_b"]) for row in selected_posteriors],
            label=str(selected_prediction["class_b"]),
            linewidth=2.0,
        )
        ax.set_title(f"Posterior Example: {selected_run_id}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Posterior")
        ax.legend()
        _write_plot(fig, posterior_dir / "posterior_example.png")

        fig, ax = plt.subplots(figsize=(9.5, 4.6))
        ax.plot(
            [float(row["time"]) for row in selected_likelihoods],
            [float(row["log_likelihood_class_a"]) for row in selected_likelihoods],
            label=str(selected_prediction["class_a"]),
            linewidth=2.0,
        )
        ax.plot(
            [float(row["time"]) for row in selected_likelihoods],
            [float(row["log_likelihood_class_b"]) for row in selected_likelihoods],
            label=str(selected_prediction["class_b"]),
            linewidth=2.0,
        )
        ax.set_title(f"Likelihood Example: {selected_run_id}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Log-likelihood proxy")
        ax.legend()
        _write_plot(fig, likelihood_dir / "likelihood_example.png")

        if selected_features:
            feature_row = selected_features[0]
            feature_names = [
                "position_range",
                "speed_range",
                "acceleration_range",
                "curvature_proxy",
                "linear_fit_residual",
                "outlier_score",
            ]
            fig, ax = plt.subplots(figsize=(10.0, 4.8))
            ax.bar(feature_names, [float(feature_row[name]) for name in feature_names], color="#72B7B2")
            ax.set_title(f"Single-Trajectory Feature Snapshot: {selected_trajectory_id}")
            ax.tick_params(axis="x", rotation=25)
            _write_plot(fig, trajectory_dir / "feature_snapshot.png")

    classifier_ids = [str(row["classifier_id"]) for row in result.metrics_by_class_pair_rows]
    class_pairs = sorted({str(row["class_pair"]) for row in result.metrics_by_class_pair_rows})
    classifier_order = sorted({str(row["classifier_id"]) for row in result.metrics_by_class_pair_rows})
    heatmap = []
    for classifier_id in classifier_order:
        row_values = []
        for class_pair in class_pairs:
            matched = next(
                (
                    float(row["overall_accuracy"])
                    for row in result.metrics_by_class_pair_rows
                    if str(row["classifier_id"]) == classifier_id and str(row["class_pair"]) == class_pair
                ),
                0.0,
            )
            row_values.append(matched)
        heatmap.append(row_values)
    fig, ax = plt.subplots(figsize=(max(7.5, 1.5 * len(class_pairs)), max(4.0, 0.5 * len(classifier_order) + 2.0)))
    image = ax.imshow(heatmap, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(class_pairs)))
    ax.set_xticklabels(class_pairs, rotation=25, ha="right")
    ax.set_yticks(range(len(classifier_order)))
    ax.set_yticklabels(classifier_order)
    ax.set_title("Classifier Accuracy by Class Pair")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    _write_plot(fig, confusion_dir / "classifier_pair_accuracy_heatmap.png")

    grouped_duration: dict[str, list[dict[str, object]]] = {}
    for row in result.class_pair_duration_rows:
        grouped_duration.setdefault(str(row["classifier_id"]), []).append(row)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for classifier_id, rows in sorted(grouped_duration.items()):
        ordered = sorted(rows, key=lambda item: float(item["time"]))
        ax.plot(
            [float(row["time"]) for row in ordered],
            [float(row["prefix_accuracy"]) for row in ordered],
            label=classifier_id,
            linewidth=1.6,
        )
    ax.set_title("Prefix Accuracy vs Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Prefix accuracy")
    ax.legend(fontsize=7, ncol=2)
    _write_plot(fig, monte_carlo_dir / "prefix_accuracy_curve.png")

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ordered_ident = sorted(
        result.identifiability_rows,
        key=lambda row: (str(row["class_pair_id"]), float(row["mean_standardized_feature_distance"])),
    )
    labels = [f"{row['class_pair_id']}:{row['feature_set_id']}" for row in ordered_ident]
    ax.barh(labels, [float(row["mean_standardized_feature_distance"]) for row in ordered_ident], color="#F58518")
    ax.set_title("Identifiability by Class Pair and Feature Set")
    ax.set_xlabel("Mean standardized feature distance")
    _write_plot(fig, feature_dir / "identifiability_summary.png")

    prior_grouped: dict[str, list[dict[str, object]]] = {}
    prior_order = {"uniform": 0, "mild_bias": 1, "strong_bias": 2}
    for row in result.prior_sensitivity_rows:
        prior_grouped.setdefault(f"{row['classifier_id']}:{row['class_pair_id']}", []).append(row)
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    for label, rows in sorted(prior_grouped.items()):
        ordered = sorted(rows, key=lambda item: prior_order.get(str(item["prior_id"]), 99))
        ax.plot(
            [str(row["prior_id"]) for row in ordered],
            [float(row["accuracy"]) for row in ordered],
            marker="o",
            linewidth=1.5,
            label=label,
        )
    ax.set_title("Prior Sensitivity by Classifier and Pair")
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=6, ncol=2)
    _write_plot(fig, prior_dir / "prior_sensitivity.png")

    pca_feature_names = [
        "position_range",
        "speed_range",
        "acceleration_range",
        "curvature_proxy",
        "linear_fit_residual",
        "quadratic_fit_residual",
    ]
    centered_rows = []
    metadata = []
    for row in result.feature_rows:
        centered_rows.append([float(row[name]) for name in pca_feature_names])
        metadata.append(str(row["true_class"]))
    means = [_mean([values[index] for values in centered_rows]) for index in range(len(pca_feature_names))]
    stds = [max(_std([values[index] for values in centered_rows]), 1e-9) for index in range(len(pca_feature_names))]
    standardized = [
        [(values[index] - means[index]) / stds[index] for index in range(len(pca_feature_names))]
        for values in centered_rows
    ]
    covariance = [[0.0 for _ in pca_feature_names] for _ in pca_feature_names]
    for values in standardized:
        for row_index in range(len(pca_feature_names)):
            for col_index in range(len(pca_feature_names)):
                covariance[row_index][col_index] += values[row_index] * values[col_index]
    denom = max(len(standardized) - 1, 1)
    for row_index in range(len(pca_feature_names)):
        for col_index in range(len(pca_feature_names)):
            covariance[row_index][col_index] /= denom

    def _matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
        return [sum(row[index] * vector[index] for index in range(len(vector))) for row in matrix]

    def _norm(vector: list[float]) -> float:
        return sum(value * value for value in vector) ** 0.5

    def _normalize(vector: list[float]) -> list[float]:
        norm = max(_norm(vector), 1e-12)
        return [value / norm for value in vector]

    def _dot(left: list[float], right: list[float]) -> float:
        return sum(left[index] * right[index] for index in range(len(left)))

    def _power_iteration(matrix: list[list[float]]) -> tuple[float, list[float]]:
        vector = _normalize([1.0 + index for index in range(len(matrix))])
        previous = 0.0
        for _ in range(200):
            vector = _normalize(_matvec(matrix, vector))
            value = _dot(vector, _matvec(matrix, vector))
            if abs(value - previous) <= 1e-9:
                return value, vector
            previous = value
        return previous, vector

    eig1, vec1 = _power_iteration(covariance)
    deflated = [
        [covariance[row][col] - eig1 * vec1[row] * vec1[col] for col in range(len(covariance))]
        for row in range(len(covariance))
    ]
    _, vec2 = _power_iteration(deflated)
    coords = [(_dot(values, vec1), _dot(values, vec2)) for values in standardized]
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    class_names = sorted(set(metadata))
    palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]
    for index, class_name in enumerate(class_names):
        xs = [coord[0] for coord, label in zip(coords, metadata) if label == class_name]
        ys = [coord[1] for coord, label in zip(coords, metadata) if label == class_name]
        ax.scatter(xs, ys, s=28, alpha=0.75, label=class_name, color=palette[index % len(palette)])
    ax.set_title("Feature PCA Snapshot")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=7)
    _write_plot(fig, pca_dir / "feature_pca_snapshot.png")

    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    ordered_pairs = sorted({str(row["class_pair_id"]) for row in result.class_pair_scenario_rows})
    best_by_pair = []
    for pair_id in ordered_pairs:
        rows = [row for row in result.metrics_by_class_pair_rows if str(row["class_pair"]) == pair_id]
        best_by_pair.append(max((float(row["overall_accuracy"]) for row in rows), default=0.0))
    ax.bar(ordered_pairs, best_by_pair, color="#54A24B")
    ax.set_title("Best Accuracy by Class Pair")
    ax.set_ylabel("Accuracy")
    ax.tick_params(axis="x", rotation=25)
    _write_plot(fig, class_pair_dir / "best_accuracy_by_pair.png")

    return plots_dir


def write_common_experiment_artifacts(
    output_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    seed: int | None = None,
    trajectories_per_case: int = 8,
    result: CommonExperimentResult | None = None,
) -> CommonExperimentArtifacts:
    analysis = result or analyze_common_experiment(
        config_path=config_path,
        seed=7 if seed is None else seed,
        trajectories_per_case=trajectories_per_case,
    )
    run_dir = Path(output_dir) / analysis.config.output_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)

    output_filenames = analysis.config.output_filenames
    config_path = run_dir / "config.yaml"
    dataset_manifest_path = run_dir / "dataset_manifest.json"
    class_definitions_path = run_dir / "class_definitions.json"
    feature_manifest_path = run_dir / "feature_manifest.json"
    feature_sets_path = run_dir / "feature_sets.json"
    class_pair_manifest_path = run_dir / "class_pair_manifest.json"
    classifier_manifest_path = run_dir / "classifier_manifest.json"
    sensor_regimes_path = run_dir / "sensor_regimes.json"
    predictions_path = run_dir / output_filenames.get("predictions_path", "unified_predictions.csv")
    posterior_history_path = run_dir / output_filenames.get("posterior_history_path", "unified_posterior_history.csv")
    likelihood_history_path = run_dir / output_filenames.get("likelihood_history_path", "unified_likelihood_history.csv")
    feature_matrix_path = run_dir / output_filenames.get("feature_matrix_path", "unified_feature_matrix.csv")
    metrics_by_classifier_path = run_dir / output_filenames.get("metrics_by_classifier_path", "metrics_by_classifier.csv")
    metrics_by_sensor_regime_path = run_dir / output_filenames.get(
        "metrics_by_sensor_regime_path",
        "metrics_by_sensor_regime.csv",
    )
    metrics_by_classifier_and_feature_set_path = run_dir / output_filenames.get(
        "metrics_by_classifier_and_feature_set_path",
        "metrics_by_classifier_and_feature_set.csv",
    )
    metrics_by_class_pair_path = run_dir / output_filenames.get("metrics_by_class_pair_path", "metrics_by_class_pair.csv")
    prior_sensitivity_by_class_pair_path = run_dir / output_filenames.get(
        "prior_sensitivity_by_class_pair_path",
        "prior_sensitivity_by_class_pair.csv",
    )
    feature_set_comparison_path = run_dir / output_filenames.get("feature_set_comparison_path", "feature_set_comparison.csv")
    irregular_window_comparison_path = run_dir / output_filenames.get(
        "irregular_window_comparison_path",
        "irregular_window_comparison.csv",
    )
    class_pair_duration_study_path = run_dir / output_filenames.get(
        "class_pair_duration_study_path",
        "class_pair_duration_study.csv",
    )
    class_pair_scenario_study_path = run_dir / output_filenames.get(
        "class_pair_scenario_study_path",
        "class_pair_scenario_study.csv",
    )
    covariate_leakage_audit_path = run_dir / output_filenames.get("covariate_leakage_audit_path", "covariate_leakage_audit.csv")
    feature_excitation_matrix_path = run_dir / output_filenames.get("feature_excitation_matrix_path", "feature_excitation_matrix.csv")
    identifiability_matrix_path = run_dir / output_filenames.get("identifiability_matrix_path", "identifiability_matrix.csv")
    oracle_classifier_results_path = run_dir / output_filenames.get("oracle_classifier_results_path", "oracle_classifier_results.csv")
    report_path = run_dir / output_filenames.get("report_path", "common_experiment_report.md")
    canonical_report_path = run_dir / "report.md"

    shutil.copyfile(analysis.config.config_path, config_path)
    shutil.copyfile(analysis.config.feature_sets_path, feature_sets_path)
    shutil.copyfile(analysis.config.class_pair_manifest_path, class_pair_manifest_path)
    shutil.copyfile(analysis.config.classifier_manifest_path, classifier_manifest_path)

    feature_manifest = load_feature_set_manifest(analysis.config.feature_sets_path)
    feature_manifest_path.write_text(json.dumps(feature_manifest, indent=2), encoding="utf-8")
    class_definitions = [
        {
            "name": definition.name,
            "kind": definition.kind,
            "description": definition.description,
            "nominal_steps": list(definition.nominal_steps),
            "dt_range": list(definition.dt_range),
            "measurement_std_range": list(definition.measurement_std_range),
        }
        for definition in default_trajectory_class_definitions()
    ]
    class_definitions_path.write_text(json.dumps({"classes": class_definitions}, indent=2), encoding="utf-8")
    dataset_manifest_path.write_text(
        json.dumps(
            {
                "experiment_name": analysis.summary.experiment_name,
                "executable_class_pairs": list(analysis.summary.executable_class_pairs),
                "trajectories_per_case": analysis.summary.trajectories_per_case,
                "num_pair_trajectories": analysis.summary.num_pair_trajectories,
                "num_pair_predictions": analysis.summary.num_pair_predictions,
                "scenario_ids": list(SCENARIO_TIMES.keys()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    sensor_regimes_path.write_text(json.dumps(sensor_regime_summary_rows(analysis.comparison.runs), indent=2), encoding="utf-8")

    _write_csv(
        predictions_path,
        list(analysis.pair_prediction_rows),
        [
            "run_id",
            "classifier_id",
            "feature_set_id",
            "sensor_regime_id",
            "measurement_dim",
            "coordinate_frame",
            "class_pair_id",
            "class_a",
            "class_b",
            "trajectory_id",
            "scenario_id",
            "scenario_family",
            "dataset_tier",
            "time",
            "true_class",
            "predicted_class",
            "confidence",
            "posterior_class_a",
            "posterior_class_b",
        ],
    )
    _write_csv(
        posterior_history_path,
        list(analysis.posterior_history_rows),
        [
            "run_id",
            "classifier_id",
            "feature_set_id",
            "sensor_regime_id",
            "class_pair_id",
            "class_a",
            "class_b",
            "trajectory_id",
            "scenario_id",
            "scenario_family",
            "dataset_tier",
            "time",
            "true_class",
            "posterior_class_a",
            "posterior_class_b",
        ],
    )
    _write_csv(
        likelihood_history_path,
        list(analysis.likelihood_history_rows),
        [
            "run_id",
            "classifier_id",
            "feature_set_id",
            "sensor_regime_id",
            "class_pair_id",
            "trajectory_id",
            "scenario_id",
            "scenario_family",
            "dataset_tier",
            "time",
            "score_type",
            "class_a",
            "class_b",
            "log_likelihood_class_a",
            "log_likelihood_class_b",
        ],
    )
    _write_csv(
        feature_matrix_path,
        list(analysis.feature_rows),
        [
            "trajectory_id",
            "class_pair_id",
            "scenario_id",
            "scenario_family",
            "dataset_tier",
            "true_class",
            "feature_set_id",
            "duration",
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
        ],
    )
    _write_csv(
        metrics_by_classifier_path,
        list(analysis.metrics_by_classifier_rows),
        ["classifier_id", "overall_accuracy", "num_predictions"],
    )
    _write_csv(
        metrics_by_sensor_regime_path,
        list(analysis.metrics_by_sensor_regime_rows),
        [
            "sensor_regime_id",
            "same_sensor_fairness_bucket",
            "overall_accuracy",
            "mean_confidence",
            "num_predictions",
            "num_classifiers",
            "measurement_dims",
            "coordinate_frames",
        ],
    )
    _write_csv(
        metrics_by_classifier_and_feature_set_path,
        list(analysis.metrics_by_classifier_and_feature_set_rows),
        ["classifier_id", "feature_set_id", "overall_accuracy"],
    )
    _write_csv(
        metrics_by_class_pair_path,
        list(analysis.metrics_by_class_pair_rows),
        ["classifier_id", "class_pair", "overall_accuracy", "status"],
    )
    _write_csv(
        prior_sensitivity_by_class_pair_path,
        list(analysis.prior_sensitivity_rows),
        ["classifier_id", "class_pair_id", "prior_id", "accuracy"],
    )
    _write_csv(
        feature_set_comparison_path,
        list(analysis.feature_set_comparison_rows),
        ["feature_set_id", "history_behavior", "num_features", "overall_accuracy", "min_pair_accuracy", "max_pair_accuracy", "mean_confidence"],
    )
    _write_csv(
        irregular_window_comparison_path,
        list(analysis.irregular_window_rows),
        [
            "class_pair_id",
            "feature_set_id",
            "history_behavior",
            "window_definition",
            "window_sample_count",
            "window_duration",
            "num_predictions",
            "overall_accuracy",
            "mean_confidence",
            "mean_selected_sample_count",
            "mean_selected_duration",
            "cross_window_prediction_disagreement_rate",
            "mean_cross_window_feature_delta",
        ],
    )
    _write_csv(
        class_pair_duration_study_path,
        list(analysis.class_pair_duration_rows),
        ["classifier_id", "class_pair_id", "time", "num_prefixes", "prefix_accuracy", "mean_confidence", "posterior_margin"],
    )
    _write_csv(
        class_pair_scenario_study_path,
        list(analysis.class_pair_scenario_rows),
        ["classifier_id", "class_pair_id", "scenario_id", "scenario_family", "overall_accuracy", "mean_confidence", "num_predictions"],
    )
    _write_csv(
        covariate_leakage_audit_path,
        list(analysis.covariate_rows),
        [
            "class_pair_id",
            "dataset_tier",
            "scenario_family",
            "true_class",
            "num_trajectories",
            "mean_duration",
            "mean_sample_count",
            "mean_dt",
            "std_dt",
            "max_dt",
            "sampling_irregularity",
            "measurement_std",
            "outlier_fraction",
            "max_covariate_delta_name",
            "max_covariate_delta_ratio",
            "status",
        ],
    )
    _write_csv(
        feature_excitation_matrix_path,
        list(analysis.feature_excitation_rows),
        [
            "class_pair_id",
            "dataset_tier",
            "scenario_family",
            "feature_set_id",
            "num_rows",
            "position_range_mean_abs",
            "position_range_std",
            "speed_range_mean_abs",
            "speed_range_std",
            "acceleration_range_mean_abs",
            "acceleration_range_std",
            "acceleration_variance_mean_abs",
            "acceleration_variance_std",
            "curvature_proxy_mean_abs",
            "curvature_proxy_std",
            "velocity_sign_changes_mean_abs",
            "velocity_sign_changes_std",
            "acceleration_sign_changes_mean_abs",
            "acceleration_sign_changes_std",
            "monotonicity_mean_abs",
            "monotonicity_std",
            "linear_fit_residual_mean_abs",
            "linear_fit_residual_std",
            "quadratic_fit_residual_mean_abs",
            "quadratic_fit_residual_std",
            "outlier_score_mean_abs",
            "outlier_score_std",
        ],
    )
    _write_csv(
        identifiability_matrix_path,
        list(analysis.identifiability_rows),
        [
            "class_pair_id",
            "feature_set_id",
            "history_behavior",
            "class_a",
            "class_b",
            "num_examples",
            "num_features",
            "mean_absolute_feature_distance",
            "mean_standardized_feature_distance",
            "overlap_estimate",
            "confusability_score",
            "identifiability_status",
        ],
    )
    _write_csv(
        oracle_classifier_results_path,
        list(analysis.oracle_rows),
        [
            "class_pair_id",
            "feature_set_id",
            "oracle_accuracy",
            "mean_confidence",
            "mean_posterior_margin",
            "num_examples",
            "history_behavior",
            "best_feature_set_for_pair",
            "best_oracle_accuracy_for_pair",
            "is_best_feature_set",
        ],
    )
    report_text = render_common_experiment_report(analysis)
    report_path.write_text(report_text, encoding="utf-8")
    canonical_report_path.write_text(report_text, encoding="utf-8")
    plots_dir = _write_common_experiment_plot_pack(run_dir, result=analysis)

    return CommonExperimentArtifacts(
        run_dir=run_dir,
        config_path=config_path,
        dataset_manifest_path=dataset_manifest_path,
        class_definitions_path=class_definitions_path,
        feature_manifest_path=feature_manifest_path,
        feature_sets_path=feature_sets_path,
        class_pair_manifest_path=class_pair_manifest_path,
        classifier_manifest_path=classifier_manifest_path,
        sensor_regimes_path=sensor_regimes_path,
        predictions_path=predictions_path,
        posterior_history_path=posterior_history_path,
        likelihood_history_path=likelihood_history_path,
        feature_matrix_path=feature_matrix_path,
        metrics_by_classifier_path=metrics_by_classifier_path,
        metrics_by_sensor_regime_path=metrics_by_sensor_regime_path,
        metrics_by_classifier_and_feature_set_path=metrics_by_classifier_and_feature_set_path,
        metrics_by_class_pair_path=metrics_by_class_pair_path,
        prior_sensitivity_by_class_pair_path=prior_sensitivity_by_class_pair_path,
        feature_set_comparison_path=feature_set_comparison_path,
        irregular_window_comparison_path=irregular_window_comparison_path,
        class_pair_duration_study_path=class_pair_duration_study_path,
        class_pair_scenario_study_path=class_pair_scenario_study_path,
        covariate_leakage_audit_path=covariate_leakage_audit_path,
        feature_excitation_matrix_path=feature_excitation_matrix_path,
        identifiability_matrix_path=identifiability_matrix_path,
        oracle_classifier_results_path=oracle_classifier_results_path,
        report_path=report_path,
        canonical_report_path=canonical_report_path,
        plots_dir=plots_dir,
    )
