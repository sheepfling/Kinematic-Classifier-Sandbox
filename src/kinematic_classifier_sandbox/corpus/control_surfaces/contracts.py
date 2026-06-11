from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ...schema.artifacts import TrajectoryArtifact


@dataclass(frozen=True, slots=True)
class ControlSurfaceMetadata:
    backend_id: str
    display_name: str
    control_surface_type: str
    state_variables: tuple[str, ...]
    control_variables: tuple[str, ...]
    constraints: dict[str, float]
    supports: dict[str, bool]
    classifier_allowed_fields: tuple[str, ...]
    hidden_fields: tuple[str, ...]
    best_use: str
    lift_to_3d: str


@dataclass(frozen=True, slots=True)
class TrajectoryCandidate:
    candidate_id: str
    backend_id: str
    trajectory: TrajectoryArtifact
    params: dict[str, float]
    control_trace: dict[str, tuple[float, ...]] = field(default_factory=dict)
    generation_metadata: dict[str, object] = field(default_factory=dict)


class ControlSurfaceBackend(Protocol):
    backend_id: str

    def metadata(self) -> ControlSurfaceMetadata: ...

    def sample_params(self, seed: int, *, accel_magnitude: float | None = None) -> dict[str, float]: ...

    def rollout(self, params: dict[str, float], *, seed: int, candidate_id: str) -> TrajectoryCandidate: ...

