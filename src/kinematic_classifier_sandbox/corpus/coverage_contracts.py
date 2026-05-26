from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CoverageReportSummary:
    overall_status: str
    corpus_status: str
    classifier_support_status: str
    feature_set_count: int
    classifier_count: int
    green_classifier_count: int
    yellow_classifier_count: int
    red_classifier_count: int


@dataclass(frozen=True, slots=True)
class CoverageReportResult:
    corpus_adequacy: object
    feature_set_summary_rows: tuple[dict[str, object], ...]
    feature_group_rows: tuple[dict[str, object], ...]
    classifier_support_rows: tuple[dict[str, object], ...]
    summary: CoverageReportSummary


@dataclass(frozen=True, slots=True)
class CoverageReportArtifacts:
    run_dir: Path
    report_path: Path
    summary_path: Path
    feature_set_summary_path: Path
    feature_group_summary_path: Path
    classifier_support_path: Path
