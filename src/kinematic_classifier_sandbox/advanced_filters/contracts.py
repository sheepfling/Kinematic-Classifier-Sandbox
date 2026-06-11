from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..inference.transition_matrix_accumulator import SwitchingScenario
from ..utils.types import FloatArray

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
    mode_ids: tuple[str, ...]
    true_modes: tuple[str, ...]
    times: tuple[float, ...]
    measurements: tuple[float, ...]
    steps: tuple[AdvancedFilterStep, ...]
    state_means: tuple[tuple[float, float, float], ...]
    state_covariances: tuple[tuple[float, ...], ...]
    mode_state_means: tuple[dict[str, tuple[float, float, float]], ...]
    mode_state_covariances: tuple[dict[str, tuple[float, ...]], ...]
    mixing_probabilities: tuple[dict[str, tuple[float, ...]], ...]


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
    trace_dir: Path
    filter_step_trace_path: Path
    per_method_diagnostics_path: Path
    intermediate_plot_dir: Path
    posterior_timeline_plot_path: Path
    likelihood_strip_plot_path: Path
    waterfall_plot_path: Path
    mixing_heatmap_plot_path: Path
    mode_conditioned_state_plot_path: Path
    switch_recovery_plot_path: Path
    step_card_dir: Path
    step_card_paths: tuple[Path, ...]
