from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class VelocityAidedRow:
    measurement_mode: str
    overall_accuracy: float
    endpoint_match_accuracy: float
    short_accuracy: float
    short_noisy_accuracy: float
    outlier_accuracy: float


@dataclass(frozen=True, slots=True)
class VelocityAidedTrace:
    measurement_mode: str
    trajectory_id: str
    scenario_name: str
    true_class: str
    final_predicted_class: str
    final_confidence: float
    times: tuple[float, ...]
    measurements: tuple[float, ...]
    velocity_measurements: tuple[float, ...]
    true_class_posterior: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class VelocityAidedComparisonResult:
    rows: tuple[VelocityAidedRow, ...]
    traces: tuple[VelocityAidedTrace, ...]


@dataclass(frozen=True, slots=True)
class VelocityAidedComparisonArtifacts:
    run_dir: Path
    report_path: Path
    summary_csv_path: Path
    trace_csv_path: Path
    heatmap_png_path: Path
    diagnostics_png_path: Path
