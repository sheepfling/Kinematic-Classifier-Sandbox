from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class GenericCorpusExplorationResult:
    exploration_manifest: dict[str, Any]
    candidate_score_rows: tuple[dict[str, Any], ...]
    archive_cell_rows: tuple[dict[str, Any], ...]
    selected_corpus_manifest: dict[str, Any]
    backend_comparison_rows: tuple[dict[str, Any], ...]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class GenericCorpusExplorationArtifacts:
    run_dir: Path
    exploration_manifest_path: Path
    candidate_scores_path: Path
    archive_cells_path: Path
    selected_corpus_manifest_path: Path
    backend_comparison_path: Path
    report_path: Path
    numeric_walkthrough_path: Path
    backend_coverage_png_path: Path
    archive_heatmap_png_path: Path
    score_parallel_png_path: Path
    selected_gallery_png_path: Path
    provenance_dashboard_png_path: Path


@dataclass(frozen=True, slots=True)
class GenericCorpusExplorationWeights:
    validity: float = 0.22
    coverage_novelty: float = 0.18
    boundary: float = 0.18
    stress: float = 0.18
    environment: float = 0.12
    provenance: float = 0.12


@dataclass(frozen=True, slots=True)
class GenericCorpusExplorationSweepVariant:
    variant_id: str
    description: str
    weights: GenericCorpusExplorationWeights


@dataclass(frozen=True, slots=True)
class GenericCorpusExplorationSweepConfig:
    baseline_variant_id: str
    variants: tuple[GenericCorpusExplorationSweepVariant, ...]
    config_path: Path | None = None


@dataclass(frozen=True, slots=True)
class GenericCorpusExplorationSweepRow:
    variant_id: str
    description: str
    weight_validity: float
    weight_coverage_novelty: float
    weight_boundary: float
    weight_stress: float
    weight_environment: float
    weight_provenance: float
    selected_coverage: int
    random_baseline_coverage: int
    coverage_delta_vs_random: int
    coverage_delta_vs_baseline: int
    selected_backend_count: int
    selected_scenario_count: int
    selected_candidate_count: int
    selected_cell_count: int
    mean_total_utility: float
    mean_total_utility_delta_vs_baseline: float
    mean_provenance_completeness: float
    candidate_jaccard_vs_baseline: float
    cell_jaccard_vs_baseline: float
    selected_candidate_ids: tuple[str, ...]
    selected_cell_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GenericCorpusExplorationSweepResult:
    baseline_variant_id: str
    variants: tuple[GenericCorpusExplorationSweepVariant, ...]
    rows: tuple[GenericCorpusExplorationSweepRow, ...]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class GenericCorpusExplorationSweepArtifacts:
    run_dir: Path
    config_path: Path
    report_path: Path
    summary_path: Path
    rows_path: Path
    overlap_matrix_path: Path
    weight_matrix_path: Path
    tradeoff_png_path: Path
    selected_set_png_path: Path
    baseline_manifest_path: Path
