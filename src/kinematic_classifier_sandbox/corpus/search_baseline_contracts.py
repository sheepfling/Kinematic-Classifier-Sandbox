from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CorpusSearchBaselineResult:
    config: dict[str, object]
    generated_candidate_rows: tuple[dict[str, object], ...]
    candidate_score_rows: tuple[dict[str, object], ...]
    selected_candidate_rows: tuple[dict[str, object], ...]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class CorpusSearchBaselineArtifacts:
    run_dir: Path
    search_config_path: Path
    generated_candidates_path: Path
    candidate_scores_path: Path
    selected_candidates_path: Path
    report_path: Path
