from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def logsumexp(values: FloatArray) -> float:
    max_value = float(np.max(values))
    return max_value + float(np.log(np.sum(np.exp(values - max_value))))


def normalize_log_weights(log_weights: FloatArray) -> tuple[FloatArray, FloatArray, float]:
    log_norm = logsumexp(log_weights)
    normalized_log_weights = log_weights - log_norm
    weights = np.exp(normalized_log_weights)
    return weights, normalized_log_weights, log_norm


def effective_sample_size(weights: FloatArray) -> float:
    return 1.0 / float(np.sum(weights**2))


def systematic_resample(weights: FloatArray, rng: np.random.Generator) -> IntArray:
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    indexes = np.zeros(n, dtype=np.int64)
    i = 0
    j = 0
    while i < n:
        if positions[i] <= cumulative[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
            if j >= n:
                j = n - 1
    return indexes
