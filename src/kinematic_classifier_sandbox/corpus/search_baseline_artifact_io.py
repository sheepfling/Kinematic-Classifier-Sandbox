from __future__ import annotations

from pathlib import Path

import yaml

from kinematic_classifier_sandbox.utils.io import write_csv

from .search_baseline_contracts import (
    CorpusSearchBaselineArtifacts,
    CorpusSearchBaselineResult,
)


def write_corpus_search_baseline_artifacts(
    output_dir: str | Path,
    *,
    result: CorpusSearchBaselineResult | None = None,
) -> CorpusSearchBaselineArtifacts:
    from .search_baseline import analyze_corpus_search_baseline

    baseline = result or analyze_corpus_search_baseline()
    run_dir = Path(output_dir) / "corpus_search_baseline"
    run_dir.mkdir(parents=True, exist_ok=True)
    search_config_path = run_dir / "search_config.yaml"
    generated_candidates_path = run_dir / "generated_candidates.csv"
    candidate_scores_path = run_dir / "candidate_scores.csv"
    selected_candidates_path = run_dir / "selected_candidates.csv"
    report_path = run_dir / "search_baseline_report.md"

    search_config_path.write_text(yaml.safe_dump(baseline.config, sort_keys=False), encoding="utf-8")
    write_csv(
        generated_candidates_path,
        list(baseline.generated_candidate_rows),
        list(baseline.generated_candidate_rows[0].keys()),
    )
    write_csv(
        candidate_scores_path,
        list(baseline.candidate_score_rows),
        list(baseline.candidate_score_rows[0].keys()),
    )
    write_csv(
        selected_candidates_path,
        list(baseline.selected_candidate_rows),
        list(baseline.selected_candidate_rows[0].keys()),
    )
    report_path.write_text(baseline.report_markdown, encoding="utf-8")

    return CorpusSearchBaselineArtifacts(
        run_dir=run_dir,
        search_config_path=search_config_path,
        generated_candidates_path=generated_candidates_path,
        candidate_scores_path=candidate_scores_path,
        selected_candidates_path=selected_candidates_path,
        report_path=report_path,
    )
