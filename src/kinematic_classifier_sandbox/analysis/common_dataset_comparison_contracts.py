from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..validation.shared_evaluation import SharedClassifierMethodSpec, SharedClassifierRun
from ..validation.technique_comparison_contracts import TechniqueApplicabilityStatus


@dataclass(frozen=True, slots=True)
class SharedDynamicsTrajectory:
    trajectory_id: str
    true_class: str
    scenario_name: str
    seed: int
    times: tuple[float, ...]
    measurements: tuple[float, ...]
    true_position: tuple[float, ...]
    true_velocity: tuple[float, ...]
    true_acceleration: tuple[float, ...]
    measurement_dim: int = 1
    coordinate_frame: str = "scalar_line"


CommonMethodRun = SharedClassifierRun


@dataclass(frozen=True, slots=True)
class CommonComparisonRow:
    method_name: str
    sensor_regime_id: str
    applicability_status: TechniqueApplicabilityStatus
    primary_evaluation_family: str
    witness_artifact: str | None
    overall_accuracy: float | None
    easy_accuracy: float | None
    irregular_accuracy: float | None
    endpoint_match_accuracy: float | None
    short_accuracy: float | None
    noisy_accuracy: float | None
    outlier_accuracy: float | None
    prior_flip_fraction: float | None


@dataclass(frozen=True, slots=True)
class CommonComparisonResult:
    trajectories: tuple[SharedDynamicsTrajectory, ...] = field(default_factory=tuple)
    runs: tuple[CommonMethodRun, ...] = field(default_factory=tuple)
    rows: tuple[CommonComparisonRow, ...] = field(default_factory=tuple)
    method_specs: tuple[SharedClassifierMethodSpec, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class CommonComparisonArtifacts:
    run_dir: Path
    report_path: Path
    trajectory_path: Path
    run_summary_path: Path
    method_summary_path: Path
    sensor_regimes_path: Path
    sensor_regime_metrics_path: Path
    heatmap_png_path: Path
    confusion_png_path: Path
    plots_dir: Path
    overview_balance_png_path: Path
    overview_covariates_png_path: Path
    scenario_profile_png_path: Path
    prior_sensitivity_png_path: Path
    trajectory_examples_png_path: Path
    final_confusion_png_path: Path


__all__ = [
    "CommonComparisonArtifacts",
    "CommonComparisonResult",
    "CommonComparisonRow",
    "CommonMethodRun",
    "SharedDynamicsTrajectory",
]
