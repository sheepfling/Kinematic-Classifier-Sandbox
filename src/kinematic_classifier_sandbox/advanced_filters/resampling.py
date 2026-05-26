from __future__ import annotations

from typing import NamedTuple

import numpy.random as random
from numpy import arange, cumsum, exp, int64, log, zeros
from numpy import max as amax
from numpy import sum as nsum

from ..utils.types import FloatArray, IntArray


class LogWeightNormalization(NamedTuple):
    weights: FloatArray
    normalized_log_weights: FloatArray
    log_norm: float


def logsumexp(values: FloatArray) -> float:
    max_value = float(amax(values))
    return max_value + float(log(nsum(exp(values - max_value))))


def normalize_log_weights(log_weights: FloatArray) -> LogWeightNormalization:
    log_norm = logsumexp(log_weights)
    normalized_log_weights = log_weights - log_norm
    weights = exp(normalized_log_weights)
    return LogWeightNormalization(weights=weights, normalized_log_weights=normalized_log_weights, log_norm=log_norm)


def effective_sample_size(weights: FloatArray) -> float:
    return 1.0 / float(nsum(weights**2))


def systematic_resample(weights: FloatArray, rng: random.Generator) -> IntArray:
    n = len(weights)
    positions = (rng.random() + arange(n)) / n
    cumulative = cumsum(weights)
    indexes = zeros(n, dtype=int64)
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
