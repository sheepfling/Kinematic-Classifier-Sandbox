from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol


@dataclass(frozen=True, slots=True)
class SharedClassifierRun:
    method_name: str
    sensor_regime_id: str
    trajectory_id: str
    true_class: str
    scenario_name: str
    final_predicted_class: str
    final_confidence: float
    final_weights: dict[str, float]
    measurement_dim: int = 1
    coordinate_frame: str = "scalar_line"


class SharedTrajectoryClassifier(Protocol):
    method_name: str
    sensor_regime_id: str

    def predict_trajectory(
        self,
        trajectory: Any,
        *,
        prior: dict[str, float] | None = None,
    ) -> SharedClassifierRun:
        ...


@dataclass(frozen=True, slots=True)
class CallableSharedClassifierAdapter:
    method_name: str
    sensor_regime_id: str
    predict_fn: Callable[[Any, dict[str, float] | None], SharedClassifierRun]

    def predict_trajectory(
        self,
        trajectory: Any,
        *,
        prior: dict[str, float] | None = None,
    ) -> SharedClassifierRun:
        return self.predict_fn(trajectory, prior)


__all__ = [
    "CallableSharedClassifierAdapter",
    "SharedClassifierRun",
    "SharedTrajectoryClassifier",
]
