from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..autodevelopment_types import CorpusCandidateEvaluation


@dataclass(frozen=True, slots=True)
class FeatureGapRow:
    iteration: int
    gap_id: str
    gap_kind: str
    target_id: str
    status: str
    severity: float
    observed_value: float
    target_value: float
    recommendation_hint: str


@dataclass(frozen=True, slots=True)
class FeatureGapRecommendation:
    iteration: int
    recommendation_id: str
    source_gap_id: str
    source_gap_kind: str
    trajectory_family: str
    sampler_name: str
    priority: float
    description: str
    expected_effect: str
    tier_counts: dict[str, int]
    measurement_scale: float
    irregularity_scale: float
    outlier_scale: float
    step_scale: float


@dataclass(frozen=True, slots=True)
class FeatureGapIterationSummary:
    iteration: int
    starting_candidate_id: str
    selected_candidate_id: str
    selected_recommendation_id: str
    accepted: bool
    stop_reason: str
    starting_q_corpus: float
    selected_q_corpus: float
    q_corpus_delta: float
    starting_feature_excitation: float
    selected_feature_excitation: float
    feature_excitation_delta: float
    starting_boundary_coverage: float
    selected_boundary_coverage: float
    boundary_coverage_delta: float
    starting_overall_score: float
    selected_overall_score: float
    overall_score_delta: float


@dataclass(frozen=True, slots=True)
class FeatureGapTrajectoryExplorerResult:
    initial_candidate_id: str
    final_candidate_id: str
    stop_reason: str
    selected_candidate_ids: tuple[str, ...]
    gap_rows: tuple[FeatureGapRow, ...]
    recommendation_rows: tuple[FeatureGapRecommendation, ...]
    iteration_rows: tuple[FeatureGapIterationSummary, ...]
    candidate_score_rows: tuple[dict[str, object], ...]
    selected_evaluations: tuple[CorpusCandidateEvaluation, ...]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class FeatureGapTrajectoryExplorerArtifacts:
    run_dir: Path
    summary_path: Path
    gap_rows_path: Path
    recommendation_rows_path: Path
    iteration_rows_path: Path
    candidate_scores_path: Path
    selected_manifest_path: Path
    report_path: Path
    q_corpus_progression_png_path: Path
    gap_priority_png_path: Path
    recommendation_family_png_path: Path
