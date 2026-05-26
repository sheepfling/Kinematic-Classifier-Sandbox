from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .autodevelopment_utils import CorpusCandidateSpec
from ..analysis.feature_analysis import FeatureAnalysisResult
from .adequacy_audit import CorpusAdequacyResult


@dataclass(frozen=True, slots=True)
class CorpusCandidateEvaluation:
    spec: CorpusCandidateSpec
    feature_analysis: FeatureAnalysisResult
    adequacy: CorpusAdequacyResult
    manifest_row: dict[str, object]
    score_row: dict[str, object]
    adequacy_row: dict[str, object]
    feature_excitation_rows: tuple[dict[str, object], ...]
    leakage_rows: tuple[dict[str, object], ...]
    pareto_objectives: dict[str, float]


@dataclass(frozen=True, slots=True)
class CorpusAutodevelopmentResult:
    objectives_path: Path
    objectives: dict[str, object]
    candidate_evaluations: tuple[CorpusCandidateEvaluation, ...]
    selected_candidate_id: str
    candidate_manifest_rows: tuple[dict[str, object], ...]
    candidate_score_rows: tuple[dict[str, object], ...]
    rejected_candidate_rows: tuple[dict[str, object], ...]
    pareto_front_rows: tuple[dict[str, object], ...]
    adequacy_comparison_rows: tuple[dict[str, object], ...]
    feature_excitation_comparison_rows: tuple[dict[str, object], ...]
    leakage_comparison_rows: tuple[dict[str, object], ...]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class CorpusAutodevelopmentArtifacts:
    run_dir: Path
    objectives_path: Path
    candidate_manifest_path: Path
    candidate_scores_path: Path
    selected_manifest_path: Path
    rejected_manifest_path: Path
    pareto_front_path: Path
    adequacy_comparison_path: Path
    feature_excitation_comparison_path: Path
    leakage_comparison_path: Path
    report_path: Path
    numeric_walkthrough_path: Path
    corpus_score_pareto_path: Path
    feature_excitation_heatmap_path: Path
    leakage_by_candidate_path: Path
    difficulty_distribution_by_candidate_path: Path
