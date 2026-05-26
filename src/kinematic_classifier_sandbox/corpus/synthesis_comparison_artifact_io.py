from __future__ import annotations

from pathlib import Path

from kinematic_classifier_sandbox.utils.io import write_csv

from ..utils.math import _union_fieldnames
from .synthesis_comparison_contracts import (
    CorpusSynthesisComparisonArtifacts,
    CorpusSynthesisComparisonResult,
)


def write_corpus_synthesis_comparison_artifacts(
    output_dir: str | Path,
    *,
    result: CorpusSynthesisComparisonResult | None = None,
) -> CorpusSynthesisComparisonArtifacts:
    from .synthesis_comparison import analyze_corpus_synthesis_comparison

    analysis = result or analyze_corpus_synthesis_comparison()
    run_dir = Path(output_dir) / "corpus_synthesis_comparison"
    run_dir.mkdir(parents=True, exist_ok=True)

    generator_comparison_path = run_dir / "generator_comparison.csv"
    corpus_quality_path = run_dir / "corpus_quality_by_generator.csv"
    feature_excitation_path = run_dir / "feature_excitation_by_generator.csv"
    classifier_stress_path = run_dir / "classifier_stress_by_generator.csv"
    report_path = run_dir / "corpus_synthesis_comparison_report.md"

    write_csv(generator_comparison_path, list(analysis.generator_rows), _union_fieldnames(analysis.generator_rows))
    write_csv(corpus_quality_path, list(analysis.corpus_quality_rows), list(analysis.corpus_quality_rows[0].keys()))
    write_csv(feature_excitation_path, list(analysis.feature_excitation_rows), list(analysis.feature_excitation_rows[0].keys()))
    write_csv(classifier_stress_path, list(analysis.classifier_stress_rows), list(analysis.classifier_stress_rows[0].keys()))
    report_path.write_text(analysis.report_markdown, encoding="utf-8")

    return CorpusSynthesisComparisonArtifacts(
        run_dir=run_dir,
        generator_comparison_path=generator_comparison_path,
        corpus_quality_path=corpus_quality_path,
        feature_excitation_path=feature_excitation_path,
        classifier_stress_path=classifier_stress_path,
        report_path=report_path,
    )
