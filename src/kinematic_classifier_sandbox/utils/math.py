from __future__ import annotations

from collections.abc import Sequence
from math import exp, isfinite, log, pi, sqrt
from typing import NamedTuple

import numpy.linalg as linalg
from numpy import asarray, eye, ndarray, zeros

from .io import union_fieldnames as _io_union_fieldnames


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _clamp(value: float, lower: float, upper: float) -> float:
    return clamp(value, lower, upper)


def _is_finite(value: float) -> bool:
    return isfinite(value)


def _safe_log(value: float) -> float:
    return safe_log(value)


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / max(len(values), 1)


def _entropy(
    weights: dict[str, float] | Sequence[float],
    *,
    epsilon: float = 1e-12,
    normalize_by_n: bool = False,
) -> float:
    values: Sequence[float] | list[float]
    if isinstance(weights, Sequence) and not isinstance(weights, str):
        values = [float(value) for value in weights]
        if not values:
            return 0.0
    else:
        values = [float(value) for value in weights.values()]
        if not values:
            return 0.0
    total = -sum(value * log(max(value, epsilon)) for value in values)
    if normalize_by_n:
        divisor = max(len(values), 2)
        return total / log(divisor)
    return total


def _union_fieldnames(rows: tuple[dict[str, object], ...] | list[dict[str, object]]) -> list[str]:
    return _io_union_fieldnames(rows)


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = clamp(q, 0.0, 1.0) * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_value = mean(values)
    return sqrt(sum((value - mean_value) ** 2 for value in values) / (len(values) - 1))


def safe_log(value: float) -> float:
    return log(max(value, 1e-12))


def logsumexp(values: list[float]) -> float:
    pivot = max(values)
    return pivot + log(sum(exp(value - pivot) for value in values))


def _logsumexp(values: Sequence[float]) -> float:
    pivot = max(values)
    return pivot + log(sum(exp(value - pivot) for value in values))


def _normalize_log_scores(log_scores: dict[str, float]) -> dict[str, float]:
    normalizer = _logsumexp(list(log_scores.values()))
    return {name: exp(score - normalizer) for name, score in log_scores.items()}


def _normalize_posterior(log_scores: dict[str, float]) -> dict[str, float]:
    return _normalize_log_scores(log_scores)


def _normalize(log_scores: dict[str, float]) -> dict[str, float]:
    return _normalize_log_scores(log_scores)


def _running_mean(values: tuple[float, ...]) -> tuple[float, ...]:
    running: list[float] = []
    total = 0.0
    for index, value in enumerate(values, start=1):
        total += value
        running.append(total / index)
    return tuple(running)


def _running_range(values: tuple[float, ...]) -> tuple[float, ...]:
    if not values:
        return ()
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
    for index, _time in enumerate(times):
        if index == 0:
            slopes.append(0.0)
            continue
        dt = max(times[index] - times[0], 1e-9)
        slopes.append((values[index] - values[0]) / dt)
    return tuple(slopes)


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_value = sum(values) / len(values)
    return sqrt(sum((value - mean_value) ** 2 for value in values) / (len(values) - 1))


def _trimmed_quantile(values: list[float], q: float, trim_fraction: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    trim = int(len(sorted_values) * trim_fraction)
    trimmed = sorted_values[trim : len(sorted_values) - trim] if len(sorted_values) > 2 * trim else sorted_values
    if not trimmed:
        trimmed = sorted_values
    position = max(0.0, min(1.0, q)) * (len(trimmed) - 1)
    lower = int(position)
    upper = min(lower + 1, len(trimmed) - 1)
    fraction = position - lower
    return trimmed[lower] * (1.0 - fraction) + trimmed[upper] * fraction


def _monotonicity_score(values: list[float]) -> float:
    if len(values) < 2:
        return 1.0
    deltas = [values[index] - values[index - 1] for index in range(1, len(values))]
    signs = [1 if delta > 1e-9 else -1 if delta < -1e-9 else 0 for delta in deltas]
    if not any(signs):
        return 1.0
    dominant = abs(sum(signs))
    return dominant / len(signs)


def _sign_change_count(values: list[float]) -> int:
    if len(values) < 3:
        return 0
    deltas = [values[index] - values[index - 1] for index in range(1, len(values))]
    signs = [1 if delta > 1e-9 else -1 if delta < -1e-9 else 0 for delta in deltas]
    filtered = [sign for sign in signs if sign != 0]
    return sum(1 for index in range(1, len(filtered)) if filtered[index] != filtered[index - 1])


def _prefix_running_min(values: list[float]) -> list[float]:
    if not values:
        return []
    running: list[float] = []
    current = values[0]
    for value in values:
        current = min(current, value)
        running.append(current)
    return running


def _prefix_running_max(values: list[float]) -> list[float]:
    if not values:
        return []
    running: list[float] = []
    current = values[0]
    for value in values:
        current = max(current, value)
        running.append(current)
    return running


def _outer(left: list[float], right: list[float]) -> list[list[float]]:
    return [[left[row] * right[col] for col in range(len(right))] for row in range(len(left))]


def _innovation_log_likelihood(innovation: float, variance: float) -> float:
    safe_variance = max(variance, 1e-9)
    return -0.5 * (log(2.0 * pi * safe_variance) + (innovation * innovation) / safe_variance)


def _as_array(values: Sequence[float] | ndarray) -> ndarray:
    return values if isinstance(values, ndarray) else asarray(values, dtype=float)


def _as_tuple(values: ndarray | Sequence[float]) -> tuple[float, ...]:
    array = asarray(values, dtype=float)
    return tuple(float(value) for value in array.tolist())


def _as_tuple_matrix(values: ndarray | Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    array = asarray(values, dtype=float)
    return tuple(tuple(float(value) for value in row.tolist()) for row in array)


def _block_diag(block: ndarray, repeats: int) -> ndarray:
    blocks = [block.copy() for _ in range(repeats)]
    if not blocks:
        return zeros((0, 0), dtype=float)
    block_size = block.shape[0]
    total_size = block_size * repeats
    result = zeros((total_size, total_size), dtype=float)
    for index, current in enumerate(blocks):
        start = index * block_size
        result[start : start + block_size, start : start + block_size] = current
    return result


def normalize_log_scores(log_scores: dict[str, float]) -> dict[str, float]:
    normalizer = logsumexp(list(log_scores.values()))
    return {name: exp(score - normalizer) for name, score in log_scores.items()}


def _gaussian_logpdf(
    value: float | Sequence[float] | ndarray,
    mean_value: float | Sequence[Sequence[float]] | ndarray | None = None,
    variance: float | Sequence[Sequence[float]] | ndarray | None = None,
    *,
    mean: float | None = None,
    var: float | None = None,
) -> float:
    if variance is None and mean_value is not None:
        covariance = asarray(mean_value, dtype=float)
        if covariance.ndim >= 2:
            residual = _as_array(value).reshape(-1, 1)
            covariance = 0.5 * (covariance + covariance.T)
            covariance += eye(covariance.shape[0], dtype=float) * 1e-9
            sign, logdet = linalg.slogdet(covariance)
            if sign <= 0:
                covariance += eye(covariance.shape[0], dtype=float) * 1e-6
                sign, logdet = linalg.slogdet(covariance)
            quadratic = float((residual.T @ linalg.solve(covariance, residual)).item())
            return float(-0.5 * (quadratic + logdet + covariance.shape[0] * log(2.0 * pi)))
    if mean_value is None:
        mean_value = mean
    if variance is None:
        variance = var
    if mean_value is None or variance is None:
        raise TypeError("gaussian_logpdf requires a mean and variance")
    safe_variance = max(float(variance), 1e-9)
    value_float = float(value) if not isinstance(value, (list, tuple, ndarray)) else float(asarray(value, dtype=float).reshape(-1)[0])
    mean_float = float(mean_value)
    return -0.5 * (log(2.0 * pi * safe_variance) + ((value_float - mean_float) ** 2) / safe_variance)


def _covariance_aware_state_log_likelihood(
    state_mean: ndarray,
    state_covariance: ndarray,
    class_mean: ndarray,
    class_covariance: ndarray,
) -> float:
    residual = state_mean - class_mean
    covariance = state_covariance + class_covariance
    return _gaussian_logpdf(residual, covariance)


def _matmul(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    rows = len(left)
    cols = len(right[0]) if right else 0
    inner = len(right)
    return [
        [sum(left[row][index] * right[index][col] for index in range(inner)) for col in range(cols)]
        for row in range(rows)
    ]


def _matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(row[index] * vector[index] for index in range(len(vector))) for row in matrix]


def _transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [list(column) for column in zip(*matrix)]


def _least_squares_slope(times: list[float], values: list[float]) -> float:
    count = len(values)
    if count < 2:
        return 0.0
    mean_t = sum(times) / count
    mean_y = sum(values) / count
    denominator = sum((time - mean_t) ** 2 for time in times)
    if denominator <= 1e-9:
        return 0.0
    numerator = sum((time - mean_t) * (value - mean_y) for time, value in zip(times, values))
    return numerator / denominator


class QuadraticFitResult(NamedTuple):
    intercept: float
    slope: float
    curvature: float


def _quadratic_fit(times: list[float], values: list[float]) -> QuadraticFitResult:
    if len(times) < 3:
        return QuadraticFitResult(intercept=0.0, slope=_least_squares_slope(times, values), curvature=0.0)
    shifted = [time - times[0] for time in times]
    s1 = len(shifted)
    s_t = sum(shifted)
    s_t2 = sum(time * time for time in shifted)
    s_t3 = sum(time * time * time for time in shifted)
    s_t4 = sum(time * time * time * time for time in shifted)
    s_y = sum(values)
    s_ty = sum(time * value for time, value in zip(shifted, values))
    s_t2y = sum(time * time * value for time, value in zip(shifted, values))
    augmented = [
        [float(s1), s_t, s_t2, s_y],
        [s_t, s_t2, s_t3, s_ty],
        [s_t2, s_t3, s_t4, s_t2y],
    ]
    for pivot_index in range(3):
        pivot_row = max(range(pivot_index, 3), key=lambda row: abs(augmented[row][pivot_index]))
        if abs(augmented[pivot_row][pivot_index]) < 1e-12:
            return QuadraticFitResult(intercept=0.0, slope=_least_squares_slope(times, values), curvature=0.0)
        augmented[pivot_index], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_index]
        pivot = augmented[pivot_index][pivot_index]
        for col in range(pivot_index, 4):
            augmented[pivot_index][col] /= pivot
        for row in range(3):
            if row == pivot_index:
                continue
            factor = augmented[row][pivot_index]
            for col in range(pivot_index, 4):
                augmented[row][col] -= factor * augmented[pivot_index][col]
    return QuadraticFitResult(intercept=augmented[0][3], slope=augmented[1][3], curvature=augmented[2][3])


def _add_matrices(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [[left[row][col] + right[row][col] for col in range(len(left[row]))] for row in range(len(left))]


def _subtract_matrices(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [[left[row][col] - right[row][col] for col in range(len(left[row]))] for row in range(len(left))]


def identity_matrix(size: int) -> list[list[float]]:
    return [[1.0 if row == col else 0.0 for col in range(size)] for row in range(size)]


def _identity(size: int) -> list[list[float]]:
    return identity_matrix(size)


def outer_product_vectors(left: list[float], right: list[float]) -> list[list[float]]:
    return [[left_row * right_col for right_col in right] for left_row in left]


def innovation_log_likelihood(innovation: float, variance: float) -> float:
    safe_variance = max(variance, 1e-9)
    return -0.5 * (log(2.0 * pi * safe_variance) + (innovation * innovation) / safe_variance)


def effective_measurement_sigma(
    predicted_covariance: list[list[float]],
    measurement: float,
    predicted_mean: list[float],
    base_measurement_sigma: float,
) -> float:
    residual = abs(measurement - predicted_mean[0]) if predicted_mean else abs(measurement)
    covariance_scale = max(predicted_covariance[0][0] if predicted_covariance else 0.0, 0.0)
    inflated = base_measurement_sigma + 0.5 * sqrt(covariance_scale) + 0.25 * residual
    return max(inflated, 1e-6)


def adaptive_process_scale(
    *,
    previous_scale: float,
    innovation: float,
    innovation_variance: float,
) -> float:
    signal = abs(innovation) / max(sqrt(max(innovation_variance, 1e-9)), 1e-6)
    next_scale = previous_scale * (1.0 + 0.12 * clamp(signal - 1.0, 0.0, 5.0))
    return clamp(next_scale, 0.05, 10.0)


def kalman_transition_and_noise(
    state_dim: int,
    process_sigma: float,
    dt: float,
    process_scale: float = 1.0,
) -> tuple[list[list[float]], list[list[float]]]:
    if state_dim <= 1:
        transition = [[1.0]]
        q = (process_sigma * process_sigma) * process_scale * max(dt, 1e-9)
        return transition, [[q]]
    if state_dim == 2:
        transition = [[1.0, dt], [0.0, 1.0]]
        q = (process_sigma * process_sigma) * process_scale
        dt2 = dt * dt
        dt3 = dt2 * dt
        noise = [
            [q * (dt3 / 3.0), q * (dt2 / 2.0)],
            [q * (dt2 / 2.0), q * dt],
        ]
        return transition, noise
    if state_dim == 3:
        transition = [
            [1.0, dt, 0.5 * dt * dt],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0],
        ]
        q = (process_sigma * process_sigma) * process_scale
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        dt5 = dt4 * dt
        noise = [
            [q * (dt5 / 20.0), q * (dt4 / 8.0), q * (dt3 / 6.0)],
            [q * (dt4 / 8.0), q * (dt3 / 3.0), q * (dt2 / 2.0)],
            [q * (dt3 / 6.0), q * (dt2 / 2.0), q * dt],
        ]
        return transition, noise
    transition = identity_matrix(state_dim)
    for index in range(state_dim - 1):
        transition[index][index + 1] = dt
    q = (process_sigma * process_sigma) * process_scale
    noise = [[0.0 for _ in range(state_dim)] for _ in range(state_dim)]
    for index in range(state_dim):
        noise[index][index] = q * max(dt, 1e-9)
    return transition, noise


def kalman_predict(
    mean: list[float],
    covariance: list[list[float]],
    state_dim: int,
    process_sigma: float,
    dt: float,
    process_scale: float = 1.0,
) -> tuple[list[float], list[list[float]]]:
    transition, noise = kalman_transition_and_noise(state_dim, process_sigma, dt, process_scale)
    predicted_mean = _matvec(transition, mean)
    predicted_covariance = _matmul(_matmul(transition, covariance), _transpose(transition))
    predicted_covariance = _add_matrices(predicted_covariance, noise)
    return predicted_mean, predicted_covariance


def kalman_update_scalar(
    predicted_mean: list[float],
    predicted_covariance: list[list[float]],
    measurement: float,
    measurement_variance: float,
    h_vector: list[float],
) -> tuple[list[float], list[list[float]], float, float]:
    innovation = measurement - sum(h_vector[index] * predicted_mean[index] for index in range(len(h_vector)))
    innovation_variance = 0.0
    for row_index in range(len(h_vector)):
        for col_index in range(len(h_vector)):
            innovation_variance += h_vector[row_index] * predicted_covariance[row_index][col_index] * h_vector[col_index]
    innovation_variance += measurement_variance
    safe_variance = max(innovation_variance, 1e-9)
    kalman_gain = [
        sum(predicted_covariance[row_index][col_index] * h_vector[col_index] for col_index in range(len(h_vector))) / safe_variance
        for row_index in range(len(predicted_mean))
    ]
    updated_mean = [predicted_mean[index] + kalman_gain[index] * innovation for index in range(len(predicted_mean))]
    kh = [[kalman_gain[row] * h_vector[col] for col in range(len(h_vector))] for row in range(len(kalman_gain))]
    identity = identity_matrix(len(predicted_mean))
    i_minus_kh = _subtract_matrices(identity, kh)
    updated_covariance = _matmul(i_minus_kh, predicted_covariance)
    return updated_mean, updated_covariance, innovation, innovation_variance


def kalman_update(
    predicted_mean: list[float],
    predicted_covariance: list[list[float]],
    measurement: float,
    measurement_sigma: float,
) -> tuple[list[float], list[list[float]], float, float]:
    h_vector = [1.0] + [0.0] * (len(predicted_mean) - 1)
    return kalman_update_scalar(
        predicted_mean,
        predicted_covariance,
        measurement,
        measurement_sigma * measurement_sigma,
        h_vector,
    )


def normalize_prior(prior: dict[str, float] | None, keys: tuple[str, ...] | list[str]) -> dict[str, float]:
    if prior is None:
        return {name: 1.0 / len(keys) for name in keys}
    total = sum(max(prior.get(name, 0.0), 0.0) for name in keys)
    if total <= 1e-12:
        return {name: 1.0 / len(keys) for name in keys}
    return {name: max(prior.get(name, 0.0), 0.0) / total for name in keys}


def gaussian_logpdf(
    value: float,
    mean_value: float | None = None,
    variance: float | None = None,
    *,
    mean: float | None = None,
    var: float | None = None,
) -> float:
    return _gaussian_logpdf(value, mean_value, variance, mean=mean, var=var)


def median3(values: list[float], index: int) -> float:
    start = max(0, index - 1)
    stop = min(len(values), index + 2)
    window = sorted(values[start:stop])
    return window[len(window) // 2]


def _median3(values: list[float], index: int) -> float:
    return median3(values, index)


def _local_quadratic_acceleration(times: list[float], values: list[float]) -> tuple[float, float]:
    if len(times) < 3 or len(values) < 3:
        return 0.0, 1e9
    local_times = times[-3:]
    local_values = values[-3:]
    weights: list[float] = []
    for index in range(3):
        time_i = local_times[index]
        other_times = [local_times[other] for other in range(3) if other != index]
        denominator = (time_i - other_times[0]) * (time_i - other_times[1])
        if abs(denominator) <= 1e-9:
            return 0.0, 1e9
        weights.append(2.0 / denominator)
    acceleration = sum(weight * value for weight, value in zip(weights, local_values))
    variance_scale = sum(weight * weight for weight in weights)
    return acceleration, variance_scale


def _log_odds_from_prior(prior_a: float, prior_b: float) -> float:
    return log(max(prior_a, 1e-12) / max(prior_b, 1e-12))


def _binary_log_odds(weights: dict[str, float], class_a: str, class_b: str) -> float:
    return log(max(weights[class_a], 1e-12) / max(weights[class_b], 1e-12))


def dot_product(left: list[float], right: list[float]) -> float:
    return sum(left[index] * right[index] for index in range(len(left)))


def matrix_vector_multiply(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(row[index] * vector[index] for index in range(len(vector))) for row in matrix]


def vector_norm(vector: list[float]) -> float:
    return sqrt(sum(value * value for value in vector))


def normalize_vector(vector: list[float]) -> list[float]:
    norm = vector_norm(vector)
    if norm <= 1e-12:
        return [0.0 for _ in vector]
    return [value / norm for value in vector]


def outer_product(vector: list[float]) -> list[list[float]]:
    return [[vector[row] * vector[col] for col in range(len(vector))] for row in range(len(vector))]


def matrix_deflation(matrix: list[list[float]], eigenvalue: float, eigenvector: list[float]) -> list[list[float]]:
    outer = outer_product(eigenvector)
    return [
        [matrix[row][col] - eigenvalue * outer[row][col] for col in range(len(matrix[row]))]
        for row in range(len(matrix))
    ]


def covariance_matrix(rows: list[list[float]]) -> list[list[float]]:
    if not rows:
        return []
    dimension = len(rows[0])
    means_vec = [mean([row[index] for row in rows]) for index in range(dimension)]
    covariance = [[0.0 for _ in range(dimension)] for _ in range(dimension)]
    for row in rows:
        centered = [row[index] - means_vec[index] for index in range(dimension)]
        for i in range(dimension):
            for j in range(dimension):
                covariance[i][j] += centered[i] * centered[j]
    denom = max(len(rows) - 1, 1)
    for i in range(dimension):
        for j in range(dimension):
            covariance[i][j] /= denom
    return covariance


def power_iteration(matrix: list[list[float]], max_iter: int = 200, tolerance: float = 1e-9) -> tuple[float, list[float]]:
    if not matrix:
        return 0.0, []
    dimension = len(matrix)
    vector = normalize_vector([1.0 + index for index in range(dimension)])
    previous_value = 0.0
    for _ in range(max_iter):
        next_vector = matrix_vector_multiply(matrix, vector)
        vector = normalize_vector(next_vector)
        value = dot_product(vector, matrix_vector_multiply(matrix, vector))
        if abs(value - previous_value) <= tolerance:
            break
        previous_value = value
    return previous_value, vector


def euclidean_distance(left: list[float], right: list[float]) -> float:
    return sqrt(sum((left[index] - right[index]) ** 2 for index in range(len(left))))


def project_rows(rows: list[list[float]], vectors: list[list[float]], k: int) -> list[list[float]]:
    active = vectors[:k]
    return [[dot_product(row, vector) for vector in active] for row in rows]


def reconstruct_rows(projected_rows: list[list[float]], vectors: list[list[float]], k: int) -> list[list[float]]:
    active = vectors[:k]
    if not active:
        return [[0.0 for _ in range(len(vectors[0]))] for _ in projected_rows] if vectors else []
    reconstructed: list[list[float]] = []
    for projected in projected_rows:
        row = [0.0 for _ in range(len(active[0]))]
        for component_index, vector in enumerate(active):
            weight = projected[component_index]
            for feature_index, loading in enumerate(vector):
                row[feature_index] += weight * loading
        reconstructed.append(row)
    return reconstructed


def row_mean(rows: list[list[float]]) -> list[float]:
    if not rows:
        return []
    dimension = len(rows[0])
    return [mean([row[index] for row in rows]) for index in range(dimension)]


def centroid(rows: list[list[float]]) -> list[float]:
    return row_mean(rows)


def farthest_first_initialization(rows: list[list[float]], k: int) -> list[list[float]]:
    if not rows or k <= 0:
        return []
    centroids = [rows[0][:]]
    for _ in range(1, k):
        next_point = max(
            rows,
            key=lambda point: min(euclidean_distance(point, c) for c in centroids),
        )
        if any(all(abs(next_point[i] - c[i]) <= 1e-12 for i in range(len(next_point))) for c in centroids):
            break
        centroids.append(next_point[:])
    while len(centroids) < k:
        centroids.append(centroids[-1][:])
    return centroids


def kmeans(rows: list[list[float]], k: int, max_iter: int = 100) -> tuple[list[int], list[list[float]], float]:
    if not rows or k <= 0:
        return [], [], 0.0
    centroids = farthest_first_initialization(rows, k)
    labels = [0 for _ in rows]
    for _ in range(max_iter):
        changed = False
        for row_index, row in enumerate(rows):
            label = min(range(k), key=lambda centroid_index: euclidean_distance(row, centroids[centroid_index]))
            if labels[row_index] != label:
                labels[row_index] = label
                changed = True
        new_centroids: list[list[float]] = []
        for centroid_index in range(k):
            cluster_rows = [row for row, label in zip(rows, labels) if label == centroid_index]
            if cluster_rows:
                new_centroids.append(centroid(cluster_rows))
            else:
                new_centroids.append(centroids[centroid_index][:])
        if not changed and all(
            euclidean_distance(old, new) <= 1e-9 for old, new in zip(centroids, new_centroids)
        ):
            centroids = new_centroids
            break
        centroids = new_centroids
    inertia = sum(euclidean_distance(row, centroids[label]) ** 2 for row, label in zip(rows, labels))
    return labels, centroids, inertia


def silhouette_score(rows: list[list[float]], labels: list[int]) -> float:
    if len(rows) < 2 or len(set(labels)) < 2:
        return 0.0
    scores: list[float] = []
    for index, row in enumerate(rows):
        same_cluster = [other for other, label in zip(rows, labels) if label == labels[index] and other is not row]
        if same_cluster:
            a = mean([euclidean_distance(row, other) for other in same_cluster])
        else:
            a = 0.0
        b_values = []
        for cluster_id in sorted(set(labels)):
            if cluster_id == labels[index]:
                continue
            cluster_rows = [other for other, label in zip(rows, labels) if label == cluster_id]
            if cluster_rows:
                b_values.append(mean([euclidean_distance(row, other) for other in cluster_rows]))
        if not b_values:
            continue
        b = min(b_values)
        scores.append((b - a) / max(a, b, 1e-12))
    return mean(scores)


def cluster_purity(labels: list[int], truth: list[str]) -> float:
    clusters: dict[int, list[str]] = {}
    for cluster_id, label in zip(labels, truth):
        clusters.setdefault(cluster_id, []).append(label)
    correct = 0
    for cluster_labels in clusters.values():
        counts: dict[str, int] = {}
        for label in cluster_labels:
            counts[label] = counts.get(label, 0) + 1
        correct += max(counts.values())
    return correct / max(len(truth), 1)


def cluster_balance(labels: list[int]) -> float:
    counts = [labels.count(cluster_id) for cluster_id in sorted(set(labels))]
    if not counts:
        return 0.0
    return min(counts) / max(counts)


def linear_fit(times: list[float], values: list[float]) -> tuple[float, float]:
    if len(times) != len(values) or not times:
        return 0.0, 0.0
    mean_t = mean(times)
    mean_y = mean(values)
    denominator = sum((time - mean_t) ** 2 for time in times)
    if abs(denominator) <= 1e-12:
        return mean_y, 0.0
    slope = sum((time - mean_t) * (value - mean_y) for time, value in zip(times, values, strict=True)) / denominator
    intercept = mean_y - slope * mean_t
    return intercept, slope
