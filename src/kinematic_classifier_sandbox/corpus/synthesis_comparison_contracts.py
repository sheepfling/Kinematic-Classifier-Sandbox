from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CorpusSynthesisComparisonResult:
    generator_rows: tuple[dict[str, object], ...]
    corpus_quality_rows: tuple[dict[str, object], ...]
    feature_excitation_rows: tuple[dict[str, object], ...]
    classifier_stress_rows: tuple[dict[str, object], ...]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class CorpusSynthesisComparisonArtifacts:
    run_dir: Path
    generator_comparison_path: Path
    corpus_quality_path: Path
    feature_excitation_path: Path
    classifier_stress_path: Path
    report_path: Path
