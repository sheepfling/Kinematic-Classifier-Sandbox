from __future__ import annotations

from math import log


def histogram_overlap(a: list[float], b: list[float], bins: int = 20) -> float:
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


def js_divergence(a: list[float], b: list[float], bins: int = 20) -> float:
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
