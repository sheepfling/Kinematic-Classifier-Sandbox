from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .contracts import CommonExperimentConfig, ExecutablePairSpec, ExecutableTrajectory


class FeatureExtractor(Protocol):
    def __call__(self, trajectory: "ExecutableTrajectory", robust: bool) -> dict[str, float]: ...


class FeatureSigma(Protocol):
    def __call__(self, feature_name: str) -> float: ...


class ReferenceBuilder(Protocol):
    def __call__(
        self,
        pair_spec: "ExecutablePairSpec",
        class_name: str,
        scenario_id: str,
        times: tuple[float, ...],
    ) -> "ExecutableTrajectory": ...


class GaussianLogPdf(Protocol):
    def __call__(self, value: float, mean: float, variance: float) -> float: ...


class SafeLog(Protocol):
    def __call__(self, value: float) -> float: ...


class MeasurementSigma(Protocol):
    def __call__(self, scenario_id: str) -> float: ...


class PairSpecBuilder(Protocol):
    def __call__(self, config: "CommonExperimentConfig") -> tuple["ExecutablePairSpec", ...]: ...


class TrajectoryGenerator(Protocol):
    def __call__(
        self,
        pair_specs: tuple["ExecutablePairSpec", ...],
        seed: int,
        trajectories_per_case: int,
    ) -> tuple[object, ...]: ...
