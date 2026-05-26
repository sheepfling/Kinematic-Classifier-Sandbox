from __future__ import annotations

from .quality_diversity_rendering import write_quality_diversity_corpus_artifacts
from .quality_diversity_types import QualityDiversityCorpusArtifacts, QualityDiversityCorpusResult
from .quality_diversity_utils import build_quality_diversity_corpus


def analyze_quality_diversity_corpus(
    *,
    seed: int = 7,
    iterations: int = 42,
) -> QualityDiversityCorpusResult:
    config, archive_cell_rows, archive_elite_rows, archive_coverage_rows, corpus_manifest, report_markdown = (
        build_quality_diversity_corpus(seed=seed, iterations=iterations)
    )
    return QualityDiversityCorpusResult(
        config=config,
        archive_cell_rows=archive_cell_rows,
        archive_elite_rows=archive_elite_rows,
        archive_coverage_rows=archive_coverage_rows,
        corpus_manifest=corpus_manifest,
        report_markdown=report_markdown,
    )
