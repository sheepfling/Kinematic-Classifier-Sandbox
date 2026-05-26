from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..utils.types import FloatArray

from .protocols import validate_advanced_filter_step
from ..inference.transition_matrix_accumulator import SwitchingScenario

DiagnosticValue = float | int | str | bool

@dataclass(frozen=True, slots=True)
class AdvancedFilterStep:
    trajectory_id: str
    time: float
    filter_id: str
    predicted_label: str
    confidence: float
    posterior_by_label: dict[str, float]
    log_evidence_by_label: dict[str, float]
    diagnostics: dict[str, DiagnosticValue]

@dataclass(frozen=True, slots=True)
class AdvancedStateSummary:
    trajectory_id: str
    time: float
    filter_id: str
    state_mean: FloatArray
    state_covariance: FloatArray | None
    diagnostics: dict[str, DiagnosticValue]


@dataclass(frozen=True, slots=True)
class IMMSwitchingRun:
    trajectory_id: str
    scenario_name: str
    true_modes: tuple[str, ...]
    times: tuple[float, ...]
    measurements: tuple[float, ...]
    steps: tuple[AdvancedFilterStep, ...]
    state_means: tuple[tuple[float, float, float], ...]
    state_covariances: tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class IMMBenchmarkResult:
    scenarios: tuple[SwitchingScenario, ...]
    runs: tuple[IMMSwitchingRun, ...]
    metrics: dict[str, float | int | str]
    method_comparison: tuple[dict[str, float | int | str], ...]


@dataclass(frozen=True, slots=True)
class IMMArtifacts:
    run_dir: Path
    config_path: Path
    report_path: Path
    mode_probability_history_path: Path
    mixing_probability_history_path: Path
    mode_likelihood_history_path: Path
    state_estimate_history_path: Path
    posterior_history_path: Path
    switching_detection_metrics_path: Path
    method_comparison_path: Path
    decision_matrix_path: Path
    plot_dir: Path
    mode_probability_plot_path: Path
    state_plot_path: Path
