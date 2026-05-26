from __future__ import annotations

from typing import Callable

import numpy.random as random
from numpy import float64, int64
from numpy.typing import NDArray

FloatArray = NDArray[float64]
IntArray = NDArray[int64]
TransitionFn = Callable[[FloatArray, float, random.Generator], FloatArray]
LogLikelihoodFn = Callable[[FloatArray, FloatArray], FloatArray]

__all__ = [
    "FloatArray",
    "IntArray",
    "TransitionFn",
    "LogLikelihoodFn",
]
