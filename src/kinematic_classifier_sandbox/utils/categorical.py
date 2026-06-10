from __future__ import annotations

from collections.abc import Sequence


def status_score(
    status: str,
    *,
    green: float = 1.0,
    yellow: float = 0.5,
    red: float = 0.0,
    default: float = 0.0,
) -> float:
    return {
        "green": green,
        "yellow": yellow,
        "red": red,
    }.get(status, default)


def bucket_thresholds(value: float, thresholds: Sequence[float], labels: Sequence[str] = ("low", "medium", "high")) -> str:
    if len(labels) != len(thresholds) + 1:
        raise ValueError("bucket labels must have exactly one more element than thresholds")
    for index, threshold in enumerate(thresholds):
        if value < threshold:
            return str(labels[index])
    return str(labels[-1])


def bucket2(value: float, low: float, high: float) -> str:
    return bucket_thresholds(value, (low, high))


__all__ = [
    "bucket2",
    "bucket_thresholds",
    "status_score",
]
