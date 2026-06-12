from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SwitchingModeSpec:
    name: str
    mean_speed: float
    sigma_speed: float
    mean_accel: float
    sigma_accel: float
    mean_abs_accel: float
    sigma_abs_accel: float
    prior_weight: float


@dataclass(frozen=True, slots=True)
class SwitchingScenario:
    trajectory_id: str
    scenario_name: str
    seed: int
    times: tuple[float, ...]
    measurements: tuple[float, ...]
    true_mode_by_step: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TransitionPosteriorStep:
    step: int
    time: float
    measurement: float
    estimated_speed: float
    estimated_accel: float
    prior_weights: dict[str, float]
    posterior_weights: dict[str, float]
    predicted_mode: str
    true_mode: str
    confidence: float
    emission_log_scores: dict[str, float] | None = None
    log_normalizer: float | None = None


@dataclass(frozen=True, slots=True)
class TransitionRun:
    trajectory_id: str
    scenario_name: str
    mode: str
    steps: tuple[TransitionPosteriorStep, ...]
    final_weights: dict[str, float]
    final_predicted_mode: str
    accuracy: float
    post_switch_accuracy: float


@dataclass(frozen=True, slots=True)
class TransitionBenchmarkSummary:
    num_scenarios: int
    static_accuracy: float
    transition_accuracy: float
    kalman_accuracy: float
    static_post_switch_accuracy: float
    transition_post_switch_accuracy: float
    kalman_post_switch_accuracy: float
    improved_scenarios: int


@dataclass(frozen=True, slots=True)
class TransitionBenchmarkResult:
    scenarios: tuple[SwitchingScenario, ...]
    static_runs: tuple[TransitionRun, ...]
    transition_runs: tuple[TransitionRun, ...]
    kalman_runs: tuple[TransitionRun, ...]
    summary: TransitionBenchmarkSummary


@dataclass(frozen=True, slots=True)
class TransitionBenchmarkArtifacts:
    run_dir: Path
    report_path: Path
    numeric_walkthrough_path: Path
    posterior_history_path: Path
    scenario_summary_path: Path
    method_evaluation_summary_path: Path
    config_path: Path
    dataset_manifest_path: Path
    plot_png_path: Path
    trace_dir: Path
    filter_step_trace_path: Path
    per_method_diagnostics_path: Path
    intermediate_plot_dir: Path
    posterior_timeline_plot_path: Path
    likelihood_strip_plot_path: Path
    waterfall_plot_path: Path
    static_vs_transition_plot_path: Path
    step_card_dir: Path
    step_card_paths: tuple[Path, ...]
