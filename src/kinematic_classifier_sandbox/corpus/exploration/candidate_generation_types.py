from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict


class CandidateGenerationRow(TypedDict):
    candidate_id: str
    objective_id: str
    target_type: str
    sampler_name: str
    search_method: str
    backend_id: str
    scenario_family: str
    target_class: str
    difficulty_tier: str
    seed: int
    duration: float
    sample_period: float
    initial_velocity: float
    acceleration: float
    measurement_std: float
    environment_id: str
    parent_candidate_id: str
    selected: int
    feature_excitation: float
    coverage_gain: float
    boundary_closeness: float
    total_utility: float


@dataclass(frozen=True, slots=True)
class CandidateGenerationResult:
    sampler_manifest: dict[str, Any]
    generated_candidate_rows: tuple[CandidateGenerationRow, ...]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class CandidateGenerationArtifacts:
    run_dir: Path
    sampler_manifest_path: Path
    generated_candidates_path: Path
    report_path: Path
    sampler_comparison_png_path: Path
    candidate_coverage_png_path: Path
    mutation_lineage_png_path: Path
