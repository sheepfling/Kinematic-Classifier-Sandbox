from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class CandidateGenerationResult:
    sampler_manifest: dict[str, Any]
    generated_candidate_rows: tuple[dict[str, Any], ...]
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
