from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class CorpusClassifierScoringResult:
    candidate_score_rows: tuple[dict[str, Any], ...]
    posterior_rows: tuple[dict[str, Any], ...]
    prior_sensitivity_rows: tuple[dict[str, Any], ...]
    disagreement_rows: tuple[dict[str, Any], ...]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class CorpusClassifierScoringArtifacts:
    run_dir: Path
    candidate_scores_path: Path
    posterior_history_path: Path
    prior_sensitivity_path: Path
    disagreement_path: Path
    report_path: Path
    posterior_plot_path: Path
    disagreement_plot_path: Path
    stress_plot_path: Path
