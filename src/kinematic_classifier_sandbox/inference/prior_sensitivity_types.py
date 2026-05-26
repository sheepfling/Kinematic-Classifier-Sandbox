from __future__ import annotations

from typing import NamedTuple


class PriorSweepPredictions(NamedTuple):
    rows: tuple[tuple[float, str, float], ...]
