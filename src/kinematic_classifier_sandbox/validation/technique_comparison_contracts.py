from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True, slots=True)
class TechniqueComparisonRow:
    method_name: str
    sensor_regime_id: str
    overall_accuracy: float
    prior_flip_fraction: float
    median_flip_threshold: float | None
    easy_accuracy: float | None
    boundary_accuracy: float | None
    outlier_accuracy: float | None
    transition_accuracy: float | None
    long_history_accuracy: float | None
    irregular_dt_accuracy: float | None
    acceleration_accuracy: float | None
    uses_temporal_history: float
    model_based: float
    irregular_dt_native: float
    outlier_aware: float
    stronger_sensor_stream: float


@dataclass(frozen=True, slots=True)
class TechniqueComparisonResult:
    rows: tuple[TechniqueComparisonRow, ...]


@dataclass(frozen=True, slots=True)
class TechniqueDefinition:
    method_name: str
    sensor_regime_id: str
    build_row: Callable[[int], TechniqueComparisonRow]


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
    "TechniqueComparisonArtifacts",
    "TechniqueComparisonResult",
    "TechniqueComparisonRow",
    "TechniqueDefinition",
]
