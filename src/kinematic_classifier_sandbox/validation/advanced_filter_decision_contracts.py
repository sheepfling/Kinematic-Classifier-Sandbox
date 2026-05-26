from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AdvancedFilterDecisionResult:
    imm_justified: bool
    particle_filter_justified: bool
    transition_post_switch_gain: float
    transition_overall_gain: float
    transition_vs_kalman_post_switch_gain: float
    transition_vs_kalman_overall_gain: float
    short_horizon_mean_gap_sigma: float
    short_horizon_final_gap_sigma: float
    velocity_aided_short_noisy_gain: float
    best_kalman_outlier_accuracy: float
    evidence_rows: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class AdvancedFilterDecisionArtifacts:
    run_dir: Path
    report_path: Path
    summary_path: Path
    evidence_path: Path
    numeric_walkthrough_path: Path


__all__ = ["AdvancedFilterDecisionArtifacts", "AdvancedFilterDecisionResult"]
