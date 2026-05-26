from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AdaptiveStressCorpusResult:
    config: dict[str, object]
    stress_case_rows: tuple[dict[str, object], ...]
    stress_score_rows: tuple[dict[str, object], ...]
    report_markdown: str
    posterior_trace_payloads: tuple[dict[str, object], ...]
    feature_trace_payloads: tuple[dict[str, object], ...]
    prior_flip_payloads: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class AdaptiveStressCorpusArtifacts:
    run_dir: Path
    config_path: Path
    stress_cases_path: Path
    stress_scores_path: Path
    report_path: Path
    posterior_timelines_path: Path
    feature_traces_path: Path
    prior_flip_examples_path: Path
