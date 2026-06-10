from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Protocol


SharedScenarioFamily = Literal[
    "shared_binary_dynamics",
    "nonlinear_drag_outlier",
    "latent_maneuver_onset",
    "ou_mean_reversion",
]


@dataclass(frozen=True, slots=True)
class SharedMethodCapabilities:
    local_feature: bool
    recursive: bool
    model_based: bool
    switching_aware: bool
    nonlinear_nongaussian: bool
    sampled_latent: bool
    stochastic_mean_reversion: bool


@dataclass(frozen=True, slots=True)
class SharedClassifierMethodSpec:
    method_name: str
    sensor_regime_id: str
    primary_evaluation_family: SharedScenarioFamily
    supported_scenario_families: tuple[SharedScenarioFamily, ...]
    witness_artifact: str | None
    capabilities: SharedMethodCapabilities


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
    method_spec: SharedClassifierMethodSpec

    @property
    def method_name(self) -> str:
        ...

    @property
    def sensor_regime_id(self) -> str:
        ...

    def predict_trajectory(
        self,
        trajectory: Any,
        *,
        prior: dict[str, float] | None = None,
    ) -> SharedClassifierRun:
        ...


@dataclass(frozen=True, slots=True)
class CallableSharedClassifierAdapter:
    method_spec: SharedClassifierMethodSpec
    predict_fn: Callable[[Any, dict[str, float] | None], SharedClassifierRun]

    @property
    def method_name(self) -> str:
        return self.method_spec.method_name

    @property
    def sensor_regime_id(self) -> str:
        return self.method_spec.sensor_regime_id

    def predict_trajectory(
        self,
        trajectory: Any,
        *,
        prior: dict[str, float] | None = None,
    ) -> SharedClassifierRun:
        return self.predict_fn(trajectory, prior)


__all__ = [
    "CallableSharedClassifierAdapter",
    "SharedClassifierMethodSpec",
    "SharedClassifierRun",
    "SharedMethodCapabilities",
    "SharedScenarioFamily",
    "SharedTrajectoryClassifier",
]
