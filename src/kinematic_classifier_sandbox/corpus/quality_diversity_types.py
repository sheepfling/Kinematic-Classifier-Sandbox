from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class QualityDiversityCorpusResult:
    config: dict[str, object]
    archive_cell_rows: tuple[dict[str, object], ...]
    archive_elite_rows: tuple[dict[str, object], ...]
    archive_coverage_rows: tuple[dict[str, object], ...]
    corpus_manifest: dict[str, object]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class QualityDiversityCorpusArtifacts:
    run_dir: Path
    config_path: Path
    archive_cells_path: Path
    archive_elites_path: Path
    archive_coverage_path: Path
    manifest_path: Path
    report_path: Path
    archive_coverage_heatmap_path: Path
    elite_score_distribution_path: Path
    feature_cell_examples_path: Path
