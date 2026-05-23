from __future__ import annotations

from dataclasses import dataclass, asdict
from math import exp, log, sqrt, erf, pi
import csv
import io
import json
import os
from pathlib import Path
from statistics import median

from .trajectory_generator import (
    GeneratedTrajectoryDataset,
    TrajectoryGeneratorArtifacts,
    default_dataset_tiers,
    generate_trajectory_datasets,
)


def _prepare_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp/kinematic-classifier-sandbox-mpl")))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_value = _mean(values)
    return sqrt(sum((value - mean_value) ** 2 for value in values) / (len(values) - 1))


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = _clamp(q, 0.0, 1.0) * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _safe_log(value: float) -> float:
    return log(max(value, 1e-12))


def _erf_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def _gaussian_logpdf(value: float, mean: float, variance: float) -> float:
    safe_variance = max(variance, 1e-9)
    return -0.5 * (log(2.0 * pi * safe_variance) + ((value - mean) ** 2) / safe_variance)


def _histogram_overlap(a: list[float], b: list[float], bins: int = 20) -> float:
    if not a or not b:
        return 0.0
    lo = min(min(a), min(b))
    hi = max(max(a), max(b))
    if hi <= lo:
        return 1.0
    width = (hi - lo) / bins
    if width <= 0.0:
        return 1.0
    hist_a = [0.0] * bins
    hist_b = [0.0] * bins
    for value in a:
        index = min(int((value - lo) / width), bins - 1)
        hist_a[index] += 1.0
    for value in b:
        index = min(int((value - lo) / width), bins - 1)
        hist_b[index] += 1.0
    total_a = sum(hist_a)
    total_b = sum(hist_b)
    if total_a == 0.0 or total_b == 0.0:
        return 0.0
    return sum(min(hist_a[index] / total_a, hist_b[index] / total_b) for index in range(bins))


def _js_divergence(a: list[float], b: list[float], bins: int = 20) -> float:
    if not a or not b:
        return 0.0
    lo = min(min(a), min(b))
    hi = max(max(a), max(b))
    if hi <= lo:
        return 0.0
    width = (hi - lo) / bins
    if width <= 0.0:
        return 0.0
    hist_a = [0.0] * bins
    hist_b = [0.0] * bins
    for value in a:
        index = min(int((value - lo) / width), bins - 1)
        hist_a[index] += 1.0
    for value in b:
        index = min(int((value - lo) / width), bins - 1)
        hist_b[index] += 1.0
    total_a = sum(hist_a)
    total_b = sum(hist_b)
    if total_a == 0.0 or total_b == 0.0:
        return 0.0
    p = [value / total_a for value in hist_a]
    q = [value / total_b for value in hist_b]
    m = [(p[index] + q[index]) / 2.0 for index in range(bins)]
    def _kl(x: list[float], y: list[float]) -> float:
        return sum(value * log(value / y[index]) for index, value in enumerate(x) if value > 0.0 and y[index] > 0.0)
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _quadratic_solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [row[:] + [vector[index]] for index, row in enumerate(matrix)]
    for pivot_index in range(size):
        pivot_row = max(range(pivot_index, size), key=lambda row: abs(augmented[row][pivot_index]))
        if abs(augmented[pivot_row][pivot_index]) < 1e-12:
            return [0.0 for _ in range(size)]
        if pivot_row != pivot_index:
            augmented[pivot_index], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_index]
        pivot = augmented[pivot_index][pivot_index]
        for column in range(pivot_index, size + 1):
            augmented[pivot_index][column] /= pivot
        for row in range(size):
            if row == pivot_index:
                continue
            factor = augmented[row][pivot_index]
            for column in range(pivot_index, size + 1):
                augmented[row][column] -= factor * augmented[pivot_index][column]
    return [augmented[index][size] for index in range(size)]


def _polynomial_fit(times: list[float], values: list[float], degree: int) -> list[float]:
    size = degree + 1
    normal_matrix = [[0.0 for _ in range(size)] for _ in range(size)]
    rhs = [0.0 for _ in range(size)]
    for time, value in zip(times, values):
        powers = [1.0]
        for _ in range(degree):
            powers.append(powers[-1] * time)
        for row in range(size):
            rhs[row] += powers[row] * value
            for col in range(size):
                normal_matrix[row][col] += powers[row] * powers[col]
    return _quadratic_solve(normal_matrix, rhs)


def _polynomial_residual_rms(times: list[float], values: list[float], degree: int) -> float:
    if not times:
        return 0.0
    coeffs = _polynomial_fit(times, values, degree)
    residuals = []
    for time, value in zip(times, values):
        prediction = 0.0
        power = 1.0
        for coeff in coeffs:
            prediction += coeff * power
            power *= time
        residuals.append(value - prediction)
    return sqrt(sum(residual * residual for residual in residuals) / max(len(residuals), 1))


def _sign(value: float, threshold: float = 1e-9) -> int:
    if value > threshold:
        return 1
    if value < -threshold:
        return -1
    return 0


def _sign_changes(values: list[float]) -> int:
    changes = 0
    previous = _sign(values[0]) if values else 0
    for value in values[1:]:
        current = _sign(value)
        if previous and current and previous != current:
            changes += 1
        if current:
            previous = current
    return changes


def _monotonicity_score(values: list[float]) -> float:
    if len(values) < 2:
        return 1.0
    diffs = [values[index] - values[index - 1] for index in range(1, len(values))]
    nonzero = [value for value in diffs if abs(value) > 1e-9]
    if not nonzero:
        return 1.0
    same_sign = sum(1 for value in nonzero if value > 0.0) if sum(nonzero) >= 0.0 else sum(1 for value in nonzero if value < 0.0)
    return same_sign / len(nonzero)


def _trend_residual(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    first = values[0]
    last = values[-1]
    trend = [first + (last - first) * index / max(len(values) - 1, 1) for index in range(len(values))]
    residuals = [value - trend[index] for index, value in enumerate(values)]
    return sqrt(sum(residual * residual for residual in residuals) / max(len(residuals), 1))


def _project_scores(rows: list[dict[str, float]], feature_names: tuple[str, ...]) -> dict[str, list[float]]:
    class_values: dict[str, list[float]] = {}
    for row in rows:
        class_values.setdefault(row["true_class"], []).append(sum(row[name] for name in feature_names))
    return class_values


def _pairwise_auc(scores_a: list[float], scores_b: list[float]) -> float:
    if not scores_a or not scores_b:
        return 0.5
    wins = 0.0
    total = 0.0
    for score_a in scores_a:
        for score_b in scores_b:
            if score_a > score_b:
                wins += 1.0
            elif score_a == score_b:
                wins += 0.5
            total += 1.0
    return wins / max(total, 1.0)


def _quantile_interpolated(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    return _percentile(sorted_values, q)


def _wasserstein_1d(scores_a: list[float], scores_b: list[float]) -> float:
    if not scores_a or not scores_b:
        return 0.0
    quantiles = [index / 49.0 for index in range(50)]
    return sum(abs(_quantile_interpolated(scores_a, q) - _quantile_interpolated(scores_b, q)) for q in quantiles) / len(quantiles)


def _pooled_covariance(rows_a: list[list[float]], rows_b: list[list[float]]) -> list[list[float]]:
    dimension = len(rows_a[0]) if rows_a else len(rows_b[0])
    if dimension == 0:
        return []
    combined = rows_a + rows_b
    means = [0.0 for _ in range(dimension)]
    for row in combined:
        for index, value in enumerate(row):
            means[index] += value
    means = [value / max(len(combined), 1) for value in means]
    covariance = [[0.0 for _ in range(dimension)] for _ in range(dimension)]
    for row in combined:
        for i in range(dimension):
            for j in range(dimension):
                covariance[i][j] += (row[i] - means[i]) * (row[j] - means[j])
    denom = max(len(combined) - 1, 1)
    for i in range(dimension):
        for j in range(dimension):
            covariance[i][j] /= denom
    for i in range(dimension):
        covariance[i][i] += 1e-6
    return covariance


def _matrix_inverse(matrix: list[list[float]]) -> list[list[float]]:
    size = len(matrix)
    augmented = [row[:] + [1.0 if index == row_index else 0.0 for index in range(size)] for row_index, row in enumerate(matrix)]
    for pivot_index in range(size):
        pivot_row = max(range(pivot_index, size), key=lambda row: abs(augmented[row][pivot_index]))
        if abs(augmented[pivot_row][pivot_index]) < 1e-12:
            return [[0.0 for _ in range(size)] for _ in range(size)]
        if pivot_row != pivot_index:
            augmented[pivot_index], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_index]
        pivot = augmented[pivot_index][pivot_index]
        for column in range(2 * size):
            augmented[pivot_index][column] /= pivot
        for row in range(size):
            if row == pivot_index:
                continue
            factor = augmented[row][pivot_index]
            for column in range(2 * size):
                augmented[row][column] -= factor * augmented[pivot_index][column]
    return [row[size:] for row in augmented]


def _matrix_determinant(matrix: list[list[float]]) -> float:
    size = len(matrix)
    working = [row[:] for row in matrix]
    determinant = 1.0
    for pivot_index in range(size):
        pivot_row = max(range(pivot_index, size), key=lambda row: abs(working[row][pivot_index]))
        pivot = working[pivot_row][pivot_index]
        if abs(pivot) < 1e-12:
            return 0.0
        if pivot_row != pivot_index:
            working[pivot_index], working[pivot_row] = working[pivot_row], working[pivot_index]
            determinant *= -1.0
        determinant *= working[pivot_index][pivot_index]
        pivot = working[pivot_index][pivot_index]
        for row in range(pivot_index + 1, size):
            factor = working[row][pivot_index] / pivot
            for column in range(pivot_index, size):
                working[row][column] -= factor * working[pivot_index][column]
    return determinant


def _matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(row[index] * vector[index] for index in range(len(vector))) for row in matrix]


def _dot(left: list[float], right: list[float]) -> float:
    return sum(left[index] * right[index] for index in range(len(left)))


@dataclass(frozen=True, slots=True)
class FeatureRow:
    trajectory_id: str
    tier: str
    scenario_id: str
    true_class: str
    seed: int
    duration: float
    mean_dt: float
    std_dt: float
    max_dt: float
    position_range: float
    speed_range: float
    acceleration_variance: float
    acceleration_range: float
    velocity_sign_changes: int
    acceleration_sign_changes: int
    monotonicity: float
    linear_fit_residual: float
    quadratic_fit_residual: float
    outlier_score: float
    sampling_irregularity: float


@dataclass(frozen=True, slots=True)
class FeatureAnalysisSummary:
    total_trajectories: int
    class_counts: dict[str, int]
    feature_names: tuple[str, ...]
    excitation_totals: dict[str, dict[str, int]]
    top_features: tuple[str, ...]
    top_separating_pairs: tuple[tuple[str, str], ...]
    top_confusing_pairs: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class FeatureAnalysisResult:
    datasets: tuple[GeneratedTrajectoryDataset, ...]
    feature_rows: tuple[FeatureRow, ...]
    excitation_rows: tuple[dict[str, object], ...]
    summary_rows: tuple[dict[str, object], ...]
    feature_separation_rows: tuple[dict[str, object], ...]
    pairwise_rows: tuple[dict[str, object], ...]
    summary: FeatureAnalysisSummary


@dataclass(frozen=True, slots=True)
class FeatureAnalysisArtifacts:
    run_dir: Path
    report_path: Path
    feature_matrix_path: Path
    feature_summary_path: Path
    feature_excitation_path: Path
    feature_excitation_summary_path: Path
    feature_separation_scores_path: Path
    identifiability_matrix_path: Path
    pairwise_distance_matrix_path: Path
    pairwise_overlap_matrix_path: Path
    pairwise_auc_matrix_path: Path
    plot_excitation_svg_path: Path
    plot_excitation_png_path: Path
    plot_distance_svg_path: Path
    plot_distance_png_path: Path
    plot_overlap_svg_path: Path
    plot_overlap_png_path: Path
    plot_scatter_svg_path: Path
    plot_scatter_png_path: Path
    plot_confusability_svg_path: Path
    plot_confusability_png_path: Path


FEATURE_NAMES = (
    "duration",
    "mean_dt",
    "std_dt",
    "max_dt",
    "position_range",
    "speed_range",
    "acceleration_variance",
    "acceleration_range",
    "velocity_sign_changes",
    "acceleration_sign_changes",
    "monotonicity",
    "linear_fit_residual",
    "quadratic_fit_residual",
    "outlier_score",
    "sampling_irregularity",
)

FEATURE_ROW_FIELDNAMES = (
    "trajectory_id",
    "tier",
    "scenario_id",
    "true_class",
    "seed",
    "duration",
    "mean_dt",
    "std_dt",
    "max_dt",
    "position_range",
    "speed_range",
    "acceleration_variance",
    "acceleration_range",
    "velocity_sign_changes",
    "acceleration_sign_changes",
    "monotonicity",
    "linear_fit_residual",
    "quadratic_fit_residual",
    "outlier_score",
    "sampling_irregularity",
)


def _feature_row_from_trajectory(dataset: GeneratedTrajectoryDataset, trajectory) -> FeatureRow:
    times = list(trajectory.times)
    measurements = list(trajectory.measurements)
    velocities = list(trajectory.true_velocity or ())
    accelerations = list(trajectory.true_acceleration or ())
    durations = times[-1] - times[0] if len(times) >= 2 else 0.0
    dt_values = [times[index] - times[index - 1] for index in range(1, len(times))]
    mean_dt = _mean(dt_values) if dt_values else 0.0
    std_dt = _std(dt_values)
    max_dt = max(dt_values) if dt_values else 0.0
    position_range = max(measurements) - min(measurements) if measurements else 0.0
    speed_range = (max(velocities) - min(velocities)) if velocities else 0.0
    acceleration_variance = _std(accelerations) ** 2
    acceleration_range = (max(accelerations) - min(accelerations)) if accelerations else 0.0
    velocity_sign_changes = _sign_changes(velocities)
    acceleration_sign_changes = _sign_changes(accelerations)
    monotonicity = _monotonicity_score(measurements)
    linear_fit_residual = _polynomial_residual_rms(times, measurements, degree=1)
    quadratic_fit_residual = _polynomial_residual_rms(times, measurements, degree=2)
    outlier_score = 0.0
    if measurements:
        quadratic_coeffs = _polynomial_fit(times, measurements, degree=2)
        residuals = []
        for time, value in zip(times, measurements):
            prediction = 0.0
            power = 1.0
            for coeff in quadratic_coeffs:
                prediction += coeff * power
                power *= time
            residuals.append(value - prediction)
        outlier_score = max(abs(value) for value in residuals) / max(quadratic_fit_residual, 1e-6)
    sampling_irregularity = std_dt / max(mean_dt, 1e-6)
    return FeatureRow(
        trajectory_id=trajectory.trajectory_id,
        tier=dataset.tier,
        scenario_id=trajectory.scenario_id,
        true_class=trajectory.true_class,
        seed=trajectory.seed,
        duration=durations,
        mean_dt=mean_dt,
        std_dt=std_dt,
        max_dt=max_dt,
        position_range=position_range,
        speed_range=speed_range,
        acceleration_variance=acceleration_variance,
        acceleration_range=acceleration_range,
        velocity_sign_changes=velocity_sign_changes,
        acceleration_sign_changes=acceleration_sign_changes,
        monotonicity=monotonicity,
        linear_fit_residual=linear_fit_residual,
        quadratic_fit_residual=quadratic_fit_residual,
        outlier_score=outlier_score,
        sampling_irregularity=sampling_irregularity,
    )


def _excitation_level(value: float, thresholds: tuple[float, float, float]) -> str:
    low, medium, high = thresholds
    if value >= high:
        return "strong"
    if value >= medium:
        return "moderate"
    if value >= low:
        return "weak"
    return "not_excited"


def _numeric_features(row: FeatureRow) -> dict[str, float]:
    return {name: getattr(row, name) if isinstance(getattr(row, name), (int, float)) else 0.0 for name in FEATURE_NAMES}


def _standardize_rows(rows: list[dict[str, float]], feature_names: tuple[str, ...]) -> list[list[float]]:
    means = {name: _mean([row[name] for row in rows]) for name in feature_names}
    stds = {name: max(_std([row[name] for row in rows]), 1e-6) for name in feature_names}
    return [[(row[name] - means[name]) / stds[name] for name in feature_names] for row in rows]


def _project_row(row: list[float], basis: list[float]) -> float:
    return _dot(row, basis)


def _class_feature_values(rows: list[dict[str, float]], class_name: str, feature_name: str) -> list[float]:
    return [row[feature_name] for row in rows if row["true_class"] == class_name]


def _pairwise_metrics(
    rows: list[dict[str, float]],
    feature_names: tuple[str, ...],
    class_a: str,
    class_b: str,
) -> dict[str, object]:
    feature_rows_a = [row for row in rows if row["true_class"] == class_a]
    feature_rows_b = [row for row in rows if row["true_class"] == class_b]
    matrix_rows = _standardize_rows(rows, feature_names)
    class_vectors = {
        class_a: [vector for vector, row in zip(matrix_rows, rows) if row["true_class"] == class_a],
        class_b: [vector for vector, row in zip(matrix_rows, rows) if row["true_class"] == class_b],
    }
    if not class_vectors[class_a] or not class_vectors[class_b]:
        return {
            "class_a": class_a,
            "class_b": class_b,
            "mean_feature_distance": 0.0,
            "standardized_mean_difference": 0.0,
            "mahalanobis_distance": 0.0,
            "bhattacharyya_distance": 0.0,
            "js_divergence": 0.0,
            "wasserstein_distance": 0.0,
            "overlap_estimate": 0.0,
            "pairwise_classifier_accuracy": 0.0,
            "average_log_likelihood_ratio": 0.0,
            "pairwise_auc": 0.5,
        }
    mean_a = [_mean([vector[index] for vector in class_vectors[class_a]]) for index in range(len(feature_names))]
    mean_b = [_mean([vector[index] for vector in class_vectors[class_b]]) for index in range(len(feature_names))]
    diff = [mean_a[index] - mean_b[index] for index in range(len(feature_names))]
    mean_feature_distance = sqrt(sum(value * value for value in diff))
    standardized_mean_difference = _mean([abs(diff[index]) for index in range(len(diff))])
    pooled_covariance = _pooled_covariance(class_vectors[class_a], class_vectors[class_b])
    inv_pooled = _matrix_inverse(pooled_covariance) if pooled_covariance else []
    mahalanobis_distance = sqrt(max(_dot(diff, _matvec(inv_pooled, diff)) if inv_pooled else 0.0, 0.0))
    det_pooled = max(_matrix_determinant(pooled_covariance), 1e-12) if pooled_covariance else 1.0
    cov_a = _pooled_covariance(class_vectors[class_a], class_vectors[class_a])
    cov_b = _pooled_covariance(class_vectors[class_b], class_vectors[class_b])
    det_a = max(_matrix_determinant(cov_a), 1e-12) if cov_a else 1.0
    det_b = max(_matrix_determinant(cov_b), 1e-12) if cov_b else 1.0
    bhattacharyya_distance = 0.125 * (mahalanobis_distance ** 2) + 0.5 * log(det_pooled / sqrt(det_a * det_b))
    projection_basis = diff if any(diff) else [1.0] + [0.0] * (len(feature_names) - 1)
    basis_norm = sqrt(sum(value * value for value in projection_basis)) or 1.0
    projection_basis = [value / basis_norm for value in projection_basis]
    projected_a = [_project_row(vector, projection_basis) for vector in class_vectors[class_a]]
    projected_b = [_project_row(vector, projection_basis) for vector in class_vectors[class_b]]
    js_divergence = _js_divergence(projected_a, projected_b)
    wasserstein_distance = _wasserstein_1d(projected_a, projected_b)
    overlap_estimate = _histogram_overlap(projected_a, projected_b)
    mean_a_score = _mean(projected_a)
    mean_b_score = _mean(projected_b)
    pairwise_classifier_accuracy = _mean(
        [
            1.0 if abs(score - mean_a_score) <= abs(score - mean_b_score) else 0.0
            for score in projected_a
        ]
        + [
            1.0 if abs(score - mean_b_score) < abs(score - mean_a_score) else 0.0
            for score in projected_b
        ]
    )
    class_a_model_mean = _mean(projected_a)
    class_b_model_mean = _mean(projected_b)
    class_a_model_variance = max(_std(projected_a) ** 2, 1e-6)
    class_b_model_variance = max(_std(projected_b) ** 2, 1e-6)
    average_log_likelihood_ratio = _mean(
        [
            _gaussian_logpdf(score, class_a_model_mean, class_a_model_variance)
            - _gaussian_logpdf(score, class_b_model_mean, class_b_model_variance)
            for score in projected_a
        ]
        + [
            _gaussian_logpdf(score, class_b_model_mean, class_b_model_variance)
            - _gaussian_logpdf(score, class_a_model_mean, class_a_model_variance)
            for score in projected_b
        ]
    )
    pairwise_auc = _pairwise_auc(projected_a, projected_b)
    return {
        "class_a": class_a,
        "class_b": class_b,
        "mean_feature_distance": mean_feature_distance,
        "standardized_mean_difference": standardized_mean_difference,
        "mahalanobis_distance": mahalanobis_distance,
        "bhattacharyya_distance": bhattacharyya_distance,
        "js_divergence": js_divergence,
        "wasserstein_distance": wasserstein_distance,
        "overlap_estimate": overlap_estimate,
        "pairwise_classifier_accuracy": pairwise_classifier_accuracy,
        "average_log_likelihood_ratio": average_log_likelihood_ratio,
        "pairwise_auc": pairwise_auc,
    }


def analyze_feature_datasets(
    *,
    seed: int = 7,
    trajectories_per_class: int = 5,
) -> FeatureAnalysisResult:
    datasets = generate_trajectory_datasets(seed=seed, trajectories_per_class=trajectories_per_class)
    feature_rows: list[FeatureRow] = []
    for dataset in datasets:
        for trajectory in dataset.trajectories:
            feature_rows.append(_feature_row_from_trajectory(dataset, trajectory))
    feature_rows_tuple = tuple(feature_rows)
    numeric_rows = [dict(true_class=row.true_class, **_numeric_features(row)) for row in feature_rows_tuple]

    threshold_map = {
        name: (
            _percentile(sorted([row[name] for row in numeric_rows]), 0.25),
            _percentile(sorted([row[name] for row in numeric_rows]), 0.50),
            _percentile(sorted([row[name] for row in numeric_rows]), 0.75),
        )
        for name in FEATURE_NAMES
    }
    excitation_rows: list[dict[str, object]] = []
    for row in feature_rows_tuple:
        row_dict = asdict(row)
        for feature_name in FEATURE_NAMES:
            row_dict[f"{feature_name}_level"] = _excitation_level(getattr(row, feature_name), threshold_map[feature_name])
        excitation_rows.append(row_dict)

    summary_rows: list[dict[str, object]] = []
    class_names = sorted({row.true_class for row in feature_rows_tuple})
    for class_name in class_names:
        class_rows = [row for row in feature_rows_tuple if row.true_class == class_name]
        for feature_name in FEATURE_NAMES:
            values = [getattr(row, feature_name) for row in class_rows]
            sorted_values = sorted(values)
            summary_rows.append(
                {
                    "true_class": class_name,
                    "feature": feature_name,
                    "mean": _mean(values),
                    "std": _std(values),
                    "median": median(values),
                    "iqr": _percentile(sorted_values, 0.75) - _percentile(sorted_values, 0.25),
                    "min": min(values),
                    "max": max(values),
                    "p05": _percentile(sorted_values, 0.05),
                    "p95": _percentile(sorted_values, 0.95),
                    "missing_rate": 0.0,
                }
            )

    pairwise_rows: list[dict[str, object]] = []
    for index, class_a in enumerate(class_names):
        for class_b in class_names[index + 1 :]:
            pairwise_rows.append(_pairwise_metrics(numeric_rows, FEATURE_NAMES, class_a, class_b))

    feature_separation_rows: list[dict[str, object]] = []
    for feature_name in FEATURE_NAMES:
        values_by_class = {class_name: [getattr(row, feature_name) for row in feature_rows_tuple if row.true_class == class_name] for class_name in class_names}
        pairwise_auc_values = []
        effect_sizes = []
        for index, class_a in enumerate(class_names):
            for class_b in class_names[index + 1 :]:
                values_a = values_by_class[class_a]
                values_b = values_by_class[class_b]
                pooled_std = sqrt(((_std(values_a) ** 2) + (_std(values_b) ** 2)) / 2.0) if values_a and values_b else 0.0
                if pooled_std > 0.0:
                    effect_sizes.append(abs(_mean(values_a) - _mean(values_b)) / pooled_std)
                pairwise_auc_values.append(_pairwise_auc(values_a, values_b))
        feature_separation_rows.append(
            {
                "feature": feature_name,
                "mean_abs_cohens_d": _mean(effect_sizes) if effect_sizes else 0.0,
                "avg_pairwise_auc": _mean(pairwise_auc_values) if pairwise_auc_values else 0.5,
                "max_pairwise_auc": max(pairwise_auc_values) if pairwise_auc_values else 0.5,
                "min_pairwise_auc": min(pairwise_auc_values) if pairwise_auc_values else 0.5,
            }
        )

    excitation_totals = {
        feature_name: {level: 0 for level in ("not_excited", "weak", "moderate", "strong")}
        for feature_name in FEATURE_NAMES
    }
    for row in excitation_rows:
        for feature_name in FEATURE_NAMES:
            excitation_totals[feature_name][row[f"{feature_name}_level"]] += 1

    top_features = tuple(
        row["feature"]
        for row in sorted(feature_separation_rows, key=lambda item: (item["avg_pairwise_auc"], item["mean_abs_cohens_d"]), reverse=True)[:3]
    )
    sorted_pairs = sorted(pairwise_rows, key=lambda row: row["pairwise_auc"], reverse=True)
    top_separating_pairs = tuple((row["class_a"], row["class_b"]) for row in sorted_pairs[:3])
    top_confusing_pairs = tuple((row["class_a"], row["class_b"]) for row in sorted(pairwise_rows, key=lambda row: row["pairwise_auc"])[:3])
    summary = FeatureAnalysisSummary(
        total_trajectories=len(feature_rows_tuple),
        class_counts={class_name: sum(1 for row in feature_rows_tuple if row.true_class == class_name) for class_name in class_names},
        feature_names=FEATURE_NAMES,
        excitation_totals=excitation_totals,
        top_features=top_features,
        top_separating_pairs=top_separating_pairs,
        top_confusing_pairs=top_confusing_pairs,
    )
    return FeatureAnalysisResult(
        datasets=datasets,
        feature_rows=feature_rows_tuple,
        excitation_rows=tuple(excitation_rows),
        summary_rows=tuple(summary_rows),
        feature_separation_rows=tuple(feature_separation_rows),
        pairwise_rows=tuple(pairwise_rows),
        summary=summary,
    )


def _render_heatmap(
    matrix: list[list[float]],
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    cmap: str = "Blues",
):
    plt = _prepare_matplotlib()
    fig, ax = plt.subplots(figsize=(8.5, 6.8))
    image = ax.imshow(matrix, cmap=cmap)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            ax.text(col_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def _render_feature_scatter(result: FeatureAnalysisResult):
    plt = _prepare_matplotlib()
    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    x_feature = result.summary.top_features[0]
    y_feature = result.summary.top_features[1] if len(result.summary.top_features) > 1 else result.summary.top_features[0]
    class_names = sorted({row.true_class for row in result.feature_rows})
    colors = {
        name: color
        for name, color in zip(
            class_names,
            ("#2563eb", "#dc2626", "#16a34a", "#7c3aed", "#d97706", "#0f766e", "#db2777"),
        )
    }
    top_confusing = set(result.summary.top_confusing_pairs[0]) if result.summary.top_confusing_pairs else set()
    for class_name in class_names:
        class_rows = [row for row in result.feature_rows if row.true_class == class_name]
        xs = [getattr(row, x_feature) for row in class_rows]
        ys = [getattr(row, y_feature) for row in class_rows]
        alpha = 0.95 if class_name in top_confusing else 0.45
        size = 52 if class_name in top_confusing else 28
        label = f"{class_name} (top confusing pair)" if class_name in top_confusing else class_name
        ax.scatter(xs, ys, s=size, alpha=alpha, color=colors[class_name], edgecolors="white", linewidths=0.4, label=label)
    ax.set_title("Feature Space Map for Confusable Classes", loc="left", fontweight="bold")
    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, fontsize=8, ncols=2)
    fig.tight_layout()
    return fig


def _render_confusability_heatmap(class_names: list[str], confusability_matrix: list[list[float]]):
    return _render_heatmap(
        confusability_matrix,
        class_names,
        class_names,
        "Class Confusability Map (1 - Pairwise AUC)",
        cmap="Reds",
    )


def _figure_to_svg(fig) -> str:
    plt = _prepare_matplotlib()
    buffer = io.StringIO()
    try:
        fig.savefig(buffer, format="svg", bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def _figure_to_png(fig) -> bytes:
    plt = _prepare_matplotlib()
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def _render_report(result: FeatureAnalysisResult) -> str:
    lines = [
        "# Feature Excitation and Identifiability",
        "",
        "This report summarizes feature excitation coverage and pairwise class separability from the synthetic trajectory generator.",
        "",
        "## Summary",
        "",
        f"- Trajectories analyzed: {result.summary.total_trajectories}",
        f"- Top features: {', '.join(result.summary.top_features)}",
        f"- Top separating pairs: {', '.join(f'{a} vs {b}' for a, b in result.summary.top_separating_pairs)}",
        f"- Top confusing pairs: {', '.join(f'{a} vs {b}' for a, b in result.summary.top_confusing_pairs)}",
        "",
        "## Feature Excitation",
        "",
        "| feature | not_excited | weak | moderate | strong |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for feature_name in FEATURE_NAMES:
        counts = result.summary.excitation_totals[feature_name]
        lines.append(
            f"| {feature_name} | {counts['not_excited']} | {counts['weak']} | {counts['moderate']} | {counts['strong']} |"
        )
    lines.extend(
        [
            "",
            "## Pairwise Separability",
            "",
            "| class_a | class_b | pairwise_auc | overlap | mahalanobis | bhattacharyya | js_divergence | wasserstein |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(result.pairwise_rows, key=lambda item: (item["class_a"], item["class_b"])):
        lines.append(
            f"| {row['class_a']} | {row['class_b']} | {row['pairwise_auc']:.3f} | {row['overlap_estimate']:.3f} | {row['mahalanobis_distance']:.3f} | {row['bhattacharyya_distance']:.3f} | {row['js_divergence']:.3f} | {row['wasserstein_distance']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Feature Ranking",
            "",
            "| feature | mean_abs_cohens_d | avg_pairwise_auc | max_pairwise_auc | min_pairwise_auc |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(result.feature_separation_rows, key=lambda item: item["avg_pairwise_auc"], reverse=True):
        lines.append(
            f"| {row['feature']} | {row['mean_abs_cohens_d']:.3f} | {row['avg_pairwise_auc']:.3f} | {row['max_pairwise_auc']:.3f} | {row['min_pairwise_auc']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Validation Notes",
            "",
            "- Excitation matrices show which synthetic scenarios actually stress each feature.",
            "- Pairwise metrics distinguish clearly separated classes from intentionally confusable ones.",
            "- The same extracted features can be reused by downstream classifier experiments.",
        ]
    )
    return "\n".join(lines)


def write_feature_analysis_artifacts(
    output_dir: str | Path,
    *,
    seed: int = 7,
    trajectories_per_class: int = 5,
) -> FeatureAnalysisArtifacts:
    result = analyze_feature_datasets(seed=seed, trajectories_per_class=trajectories_per_class)
    output_root = Path(output_dir)
    run_dir = output_root / "feature_analysis_v1"
    run_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_dir / "feature_analysis_report.md"
    feature_matrix_path = run_dir / "feature_matrix.csv"
    feature_summary_path = run_dir / "feature_summary_by_class.csv"
    feature_excitation_path = run_dir / "feature_excitation_matrix.csv"
    feature_excitation_summary_path = run_dir / "feature_excitation_summary.json"
    feature_separation_scores_path = run_dir / "feature_separation_scores.csv"
    identifiability_matrix_path = run_dir / "identifiability_matrix.csv"
    pairwise_distance_matrix_path = run_dir / "pairwise_distance_matrix.csv"
    pairwise_overlap_matrix_path = run_dir / "pairwise_overlap_matrix.csv"
    pairwise_auc_matrix_path = run_dir / "pairwise_auc_matrix.csv"
    plot_excitation_svg_path = run_dir / "feature_excitation_heatmap.svg"
    plot_excitation_png_path = run_dir / "feature_excitation_heatmap.png"
    plot_distance_svg_path = run_dir / "pairwise_distance_heatmap.svg"
    plot_distance_png_path = run_dir / "pairwise_distance_heatmap.png"
    plot_overlap_svg_path = run_dir / "pairwise_overlap_heatmap.svg"
    plot_overlap_png_path = run_dir / "pairwise_overlap_heatmap.png"
    plot_scatter_svg_path = run_dir / "feature_space_confusion_map.svg"
    plot_scatter_png_path = run_dir / "feature_space_confusion_map.png"
    plot_confusability_svg_path = run_dir / "class_confusability_heatmap.svg"
    plot_confusability_png_path = run_dir / "class_confusability_heatmap.png"

    report_path.write_text(_render_report(result), encoding="utf-8")
    _write_csv(feature_matrix_path, [asdict(row) for row in result.feature_rows], list(FEATURE_ROW_FIELDNAMES))
    _write_csv(feature_summary_path, [dict(row) for row in result.summary_rows], ["true_class", "feature", "mean", "std", "median", "iqr", "min", "max", "p05", "p95", "missing_rate"])
    _write_csv(feature_excitation_path, [dict(row) for row in result.excitation_rows], list(FEATURE_ROW_FIELDNAMES) + [f"{name}_level" for name in FEATURE_NAMES])
    feature_excitation_summary_path.write_text(json.dumps(asdict(result.summary), indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(feature_separation_scores_path, [dict(row) for row in result.feature_separation_rows], ["feature", "mean_abs_cohens_d", "avg_pairwise_auc", "max_pairwise_auc", "min_pairwise_auc"])
    _write_csv(identifiability_matrix_path, [dict(row) for row in result.pairwise_rows], ["class_a", "class_b", "mean_feature_distance", "standardized_mean_difference", "mahalanobis_distance", "bhattacharyya_distance", "js_divergence", "wasserstein_distance", "overlap_estimate", "pairwise_classifier_accuracy", "average_log_likelihood_ratio", "pairwise_auc"])

    class_names = sorted({row.true_class for row in result.feature_rows})
    distance_matrix = [[0.0 for _ in class_names] for _ in class_names]
    overlap_matrix = [[0.0 for _ in class_names] for _ in class_names]
    auc_matrix = [[0.5 for _ in class_names] for _ in class_names]
    confusability_matrix = [[0.0 for _ in class_names] for _ in class_names]
    row_lookup = {(row["class_a"], row["class_b"]): row for row in result.pairwise_rows}
    row_lookup.update({(row["class_b"], row["class_a"]): row for row in result.pairwise_rows})
    for i, class_a in enumerate(class_names):
        for j, class_b in enumerate(class_names):
            if class_a == class_b:
                auc_matrix[i][j] = 1.0
                continue
            row = row_lookup[(class_a, class_b)]
            distance_matrix[i][j] = float(row["mahalanobis_distance"])
            overlap_matrix[i][j] = float(row["overlap_estimate"])
            auc_matrix[i][j] = float(row["pairwise_auc"]) if row["class_a"] == class_a else 1.0 - float(row["pairwise_auc"])
            confusability_matrix[i][j] = 1.0 - auc_matrix[i][j]
    _write_csv(
        pairwise_distance_matrix_path,
        [{"class": class_names[index], **{class_names[col_index]: distance_matrix[index][col_index] for col_index in range(len(class_names))}} for index in range(len(class_names))],
        ["class", *class_names],
    )
    _write_csv(
        pairwise_overlap_matrix_path,
        [{"class": class_names[index], **{class_names[col_index]: overlap_matrix[index][col_index] for col_index in range(len(class_names))}} for index in range(len(class_names))],
        ["class", *class_names],
    )
    _write_csv(
        pairwise_auc_matrix_path,
        [{"class": class_names[index], **{class_names[col_index]: auc_matrix[index][col_index] for col_index in range(len(class_names))}} for index in range(len(class_names))],
        ["class", *class_names],
    )

    excitation_matrix = [[float(result.summary.excitation_totals[feature][level]) for feature in FEATURE_NAMES] for level in ("not_excited", "weak", "moderate", "strong")]
    plot_excitation_svg_path.write_text(
        _figure_to_svg(
            _render_heatmap(
                excitation_matrix,
                ["not_excited", "weak", "moderate", "strong"],
                list(FEATURE_NAMES),
                "Feature Excitation Totals",
                cmap="viridis",
            )
        ),
        encoding="utf-8",
    )
    plot_excitation_png_path.write_bytes(
        _figure_to_png(
            _render_heatmap(
                excitation_matrix,
                ["not_excited", "weak", "moderate", "strong"],
                list(FEATURE_NAMES),
                "Feature Excitation Totals",
                cmap="viridis",
            )
        )
    )
    plot_distance_svg_path.write_text(
        _figure_to_svg(
            _render_heatmap(distance_matrix, class_names, class_names, "Pairwise Mahalanobis Distance", cmap="Blues")
        ),
        encoding="utf-8",
    )
    plot_distance_png_path.write_bytes(
        _figure_to_png(
            _render_heatmap(distance_matrix, class_names, class_names, "Pairwise Mahalanobis Distance", cmap="Blues")
        )
    )
    plot_overlap_svg_path.write_text(
        _figure_to_svg(
            _render_heatmap(overlap_matrix, class_names, class_names, "Pairwise Overlap Estimate", cmap="Oranges")
        ),
        encoding="utf-8",
    )
    plot_overlap_png_path.write_bytes(
        _figure_to_png(
            _render_heatmap(overlap_matrix, class_names, class_names, "Pairwise Overlap Estimate", cmap="Oranges")
        )
    )
    plot_scatter_svg_path.write_text(_figure_to_svg(_render_feature_scatter(result)), encoding="utf-8")
    plot_scatter_png_path.write_bytes(_figure_to_png(_render_feature_scatter(result)))
    plot_confusability_svg_path.write_text(
        _figure_to_svg(_render_confusability_heatmap(class_names, confusability_matrix)),
        encoding="utf-8",
    )
    plot_confusability_png_path.write_bytes(
        _figure_to_png(_render_confusability_heatmap(class_names, confusability_matrix))
    )

    return FeatureAnalysisArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        feature_matrix_path=feature_matrix_path,
        feature_summary_path=feature_summary_path,
        feature_excitation_path=feature_excitation_path,
        feature_excitation_summary_path=feature_excitation_summary_path,
        feature_separation_scores_path=feature_separation_scores_path,
        identifiability_matrix_path=identifiability_matrix_path,
        pairwise_distance_matrix_path=pairwise_distance_matrix_path,
        pairwise_overlap_matrix_path=pairwise_overlap_matrix_path,
        pairwise_auc_matrix_path=pairwise_auc_matrix_path,
        plot_excitation_svg_path=plot_excitation_svg_path,
        plot_excitation_png_path=plot_excitation_png_path,
        plot_distance_svg_path=plot_distance_svg_path,
        plot_distance_png_path=plot_distance_png_path,
        plot_overlap_svg_path=plot_overlap_svg_path,
        plot_overlap_png_path=plot_overlap_png_path,
        plot_scatter_svg_path=plot_scatter_svg_path,
        plot_scatter_png_path=plot_scatter_png_path,
        plot_confusability_svg_path=plot_confusability_svg_path,
        plot_confusability_png_path=plot_confusability_png_path,
    )
