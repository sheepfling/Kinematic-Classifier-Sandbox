from __future__ import annotations

from typing import NamedTuple


class KinematicSeries(NamedTuple):
    position: tuple[float, ...]
    velocity: tuple[float, ...]
    acceleration: tuple[float, ...]
