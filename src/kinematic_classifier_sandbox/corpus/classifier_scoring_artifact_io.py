from __future__ import annotations

from pathlib import Path

from kinematic_classifier_sandbox.utils.io import write_csv

from .classifier_scoring_reporting import (
    render_classifier_scoring_disagreement_plot,
    render_classifier_scoring_posterior_plot,
    render_classifier_scoring_stress_plot,
)
from .classifier_scoring_types import CorpusClassifierScoringArtifacts, CorpusClassifierScoringResult


def write_corpus_classifier_scoring_artifacts(
    base_dir: str | Path,
    *,
    result: CorpusClassifierScoringResult | None = None,
) -> CorpusClassifierScoringArtifacts:
    from .classifier_scoring import analyze_corpus_classifier_scoring

    run_dir = Path(base_dir) / "corpus_classifier_scoring"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = result or analyze_corpus_classifier_scoring()

    candidate_scores_path = run_dir / "classifier_candidate_scores.csv"
    posterior_history_path = run_dir / "posterior_history.csv"
    prior_sensitivity_path = run_dir / "prior_sensitivity_scores.csv"
    disagreement_path = run_dir / "method_disagreement_scores.csv"
    report_path = run_dir / "classifier_scoring_report.md"
    posterior_plot_path = run_dir / "posterior_confidence_preview.png"
    disagreement_plot_path = run_dir / "method_disagreement_preview.png"
    stress_plot_path = run_dir / "classifier_stress_by_method.png"

    write_csv(candidate_scores_path, list(payload.candidate_score_rows), list(payload.candidate_score_rows[0].keys()) if payload.candidate_score_rows else [])
    write_csv(posterior_history_path, list(payload.posterior_rows), list(payload.posterior_rows[0].keys()) if payload.posterior_rows else [])
    write_csv(prior_sensitivity_path, list(payload.prior_sensitivity_rows), list(payload.prior_sensitivity_rows[0].keys()) if payload.prior_sensitivity_rows else [])
    write_csv(disagreement_path, list(payload.disagreement_rows), list(payload.disagreement_rows[0].keys()) if payload.disagreement_rows else [])
    report_path.write_text(payload.report_markdown, encoding="utf-8")
    posterior_plot_path.write_bytes(render_classifier_scoring_posterior_plot(payload.posterior_rows))
    disagreement_plot_path.write_bytes(render_classifier_scoring_disagreement_plot(payload.disagreement_rows))
    stress_plot_path.write_bytes(render_classifier_scoring_stress_plot(payload.candidate_score_rows))

    return CorpusClassifierScoringArtifacts(
        run_dir=run_dir,
        candidate_scores_path=candidate_scores_path,
        posterior_history_path=posterior_history_path,
        prior_sensitivity_path=prior_sensitivity_path,
        disagreement_path=disagreement_path,
        report_path=report_path,
        posterior_plot_path=posterior_plot_path,
        disagreement_plot_path=disagreement_plot_path,
        stress_plot_path=stress_plot_path,
    )
