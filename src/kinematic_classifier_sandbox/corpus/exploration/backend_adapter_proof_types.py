from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..trajectory_backend_contract import TrajectoryRun


@dataclass(frozen=True, slots=True)
class BackendCandidateSpec:
    candidate_id: str
    scenario_id: str
    scenario_family: str
    target_class: str
    difficulty_tier: str
    seed: int
    duration: float
    sample_period: float
    initial_position: float
    initial_velocity: float
    acceleration: float
    measurement_std: float
    switch_time: float | None = None
    acceleration_after_switch: float | None = None
    drag_coefficient: float | None = None
    density_scale: float | None = None
    wind_bias: float | None = None
    input_deck_hash: str | None = None
    longitudinal_command: tuple[float, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AdapterExecutionRecord:
    backend_id: str
    candidate_id: str
    cache_key: str
    cache_hit: bool
    input_bundle: dict[str, Any]
    raw_output: dict[str, Any]
    trajectory_run: TrajectoryRun
    validation_errors: tuple[str, ...]
