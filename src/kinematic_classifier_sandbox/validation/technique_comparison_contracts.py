from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from .shared_evaluation_contracts import SharedClassifierMethodSpec, SharedScenarioFamily


TechniqueApplicabilityStatus = Literal["supported", "not_applicable", "witness_only"]

TechniqueScenarioFamily = Literal[
    "easy",
    "boundary",
    "outlier",
    "transition",
    "long_history",
    "irregular_dt",
    "acceleration",
    "nonlinear_drag_outlier",
    "latent_maneuver_onset",
    "ou_mean_reversion",
]


@dataclass(frozen=True, slots=True)
class TechniqueComparisonRow:
    method_name: str
    sensor_regime_id: str
    applicability_status: TechniqueApplicabilityStatus
    primary_evaluation_family: SharedScenarioFamily
    witness_artifact: str | None
    overall_accuracy: float | None
    prior_flip_fraction: float | None
    median_flip_threshold: float | None
    easy_accuracy: float | None
    boundary_accuracy: float | None
    outlier_accuracy: float | None
    transition_accuracy: float | None
    long_history_accuracy: float | None
    irregular_dt_accuracy: float | None
    acceleration_accuracy: float | None


@dataclass(frozen=True, slots=True)
class TechniqueScenarioSupportRow:
    method_name: str
    scenario_family: TechniqueScenarioFamily
    applicability_status: TechniqueApplicabilityStatus
    metric_name: str | None
    metric_value: float | None
    note: str


@dataclass(frozen=True, slots=True)
class TechniqueComparisonResult:
    rows: tuple[TechniqueComparisonRow, ...]
    method_specs: tuple[SharedClassifierMethodSpec, ...]
    scenario_support_rows: tuple[TechniqueScenarioSupportRow, ...]


@dataclass(frozen=True, slots=True)
class TechniqueDefinition:
    method_spec: SharedClassifierMethodSpec
    build_row: Callable[[int], TechniqueComparisonRow]

    @property
    def method_name(self) -> str:
        return self.method_spec.method_name

    @property
    def sensor_regime_id(self) -> str:
        return self.method_spec.sensor_regime_id


@dataclass(frozen=True, slots=True)
class TechniqueComparisonArtifacts:
    run_dir: Path
    report_path: Path
    summary_csv_path: Path
    scenario_csv_path: Path
    capability_csv_path: Path
    metric_heatmap_png_path: Path
    scatter_png_path: Path
    capability_png_path: Path


__all__ = [
    "TechniqueApplicabilityStatus",
    "TechniqueComparisonArtifacts",
    "TechniqueComparisonResult",
    "TechniqueComparisonRow",
    "TechniqueDefinition",
    "TechniqueScenarioFamily",
    "TechniqueScenarioSupportRow",
]
