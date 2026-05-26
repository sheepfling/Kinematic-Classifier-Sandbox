from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PriorSweepRow:
    trajectory_id: str
    scenario_name: str
    true_class: str
    prior_a: float
    prior_b: float
    log_prior_odds: float
    final_class: str
    final_confidence: float
    abstained: bool
    posterior_a: float
    posterior_b: float
    final_log_posterior_odds: float
    cumulative_log_likelihood_ratio: float


@dataclass(frozen=True, slots=True)
class PriorFlipThreshold:
    trajectory_id: str
    scenario_name: str
    true_class: str
    uniform_prior_class: str
    uniform_prior_confidence: float
    min_prior_a_for_a: float | None
    max_prior_a_for_b: float | None
    smallest_prior_shift_to_flip: float | None
    smallest_log_prior_shift_to_flip: float | None


@dataclass(frozen=True, slots=True)
class PriorSensitivitySummary:
    trajectory_count: int
    sweep_count: int
    flipped_by_small_prior_fraction: float
    median_smallest_prior_shift_to_flip: float | None
    median_smallest_log_prior_shift_to_flip: float | None
    ambiguous_uniform_class: str
    ambiguous_flip_threshold_for_a: float | None


@dataclass(frozen=True, slots=True)
class PriorSensitivityResult:
    method_name: str
    class_names: tuple[str, ...]
    trajectories: tuple[object, ...]
    sweep_rows: tuple[PriorSweepRow, ...]
    flip_thresholds: tuple[PriorFlipThreshold, ...]
    summary: PriorSensitivitySummary
    prior_dominance_metrics: dict[str, object]


@dataclass(frozen=True, slots=True)
class PriorSensitivityArtifacts:
    run_dir: Path
    report_path: Path
    sweep_path: Path
    flip_thresholds_path: Path
    metrics_path: Path
    config_path: Path
    plot_posterior_png_path: Path
    plot_flip_png_path: Path
    plot_heatmap_png_path: Path
    plot_decision_png_path: Path
    plot_decomposition_png_path: Path
    plot_pairwise_flip_png_path: Path
    plot_fragility_png_path: Path


@dataclass(frozen=True, slots=True)
class CrossMethodPriorComparisonResult:
    method_results: tuple[PriorSensitivityResult, ...]
    scenario_names: tuple[str, ...]
    rows: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class CrossMethodPriorComparisonArtifacts:
    run_dir: Path
    report_path: Path
    comparison_csv_path: Path
    status_csv_path: Path
    plot_png_path: Path
