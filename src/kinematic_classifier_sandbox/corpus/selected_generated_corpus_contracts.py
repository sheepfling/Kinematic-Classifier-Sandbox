from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class SelectedGeneratedCorpusResult:
    corpus_manifest: dict[str, Any]
    trajectory_rows: tuple[dict[str, Any], ...]
    observation_rows: tuple[dict[str, Any], ...]
    truth_state_rows: tuple[dict[str, Any], ...]
    event_rows: tuple[dict[str, Any], ...]
    environment_rows: tuple[dict[str, Any], ...]
    feature_rows: tuple[dict[str, Any], ...]
    class_validity_rows: tuple[dict[str, Any], ...]
    classifier_score_rows: tuple[dict[str, Any], ...]
    posterior_rows: tuple[dict[str, Any], ...]
    adequacy_result: Any
    adequacy_summary: dict[str, Any]
    adequacy_recommendations: tuple[str, ...]
    regression_rows: tuple[dict[str, Any], ...]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class SelectedGeneratedCorpusArtifacts:
    run_dir: Path
    manifest_path: Path
    trajectories_path: Path
    observations_path: Path
    truth_states_path: Path
    events_path: Path
    environment_traces_path: Path
    feature_matrix_path: Path
    class_validity_scores_path: Path
    classifier_scores_path: Path
    posterior_history_path: Path
    report_path: Path
    adequacy_run_dir: Path
    adequacy_summary_path: Path
    adequacy_regressions_path: Path
    summary_plot_path: Path
    validity_plot_path: Path
    score_gallery_path: Path
