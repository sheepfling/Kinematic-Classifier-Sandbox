from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..corpus.adequacy_audit import CorpusAdequacyArtifacts
from ..corpus.coverage_report import CoverageReportArtifacts
from .feature_analysis import FeatureAnalysisArtifacts
from .pca_analysis import PcaAnalysisArtifacts


@dataclass(frozen=True, slots=True)
class AbstractInspectionArtifacts:
    run_dir: Path
    index_path: Path
    manifest_path: Path
    machine_summary_path: Path
    summary_table_path: Path
    summary_chart_path: Path
    class_pair_summary_table_path: Path
    class_pair_summary_chart_path: Path
    feature_analysis_runs: tuple[FeatureAnalysisArtifacts, ...]
    pca_runs: tuple[PcaAnalysisArtifacts, ...]
    corpus_adequacy: CorpusAdequacyArtifacts
    coverage_report: CoverageReportArtifacts
