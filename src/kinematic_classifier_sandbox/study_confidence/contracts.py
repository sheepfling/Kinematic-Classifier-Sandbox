from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class StudyConfidenceArtifacts:
    run_dir: Path
    components_path: Path
    classifier_scores_path: Path
    summary_path: Path
    report_path: Path
    dashboard_path: Path


@dataclass(frozen=True, slots=True)
class StudyConfidenceResult:
    component_rows: tuple[dict[str, object], ...]
    classifier_rows: tuple[dict[str, object], ...]
    summary: dict[str, object]
    report_markdown: str

