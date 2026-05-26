from __future__ import annotations

import json
from pathlib import Path

from kinematic_classifier_sandbox.utils.io import write_csv

from .selected_generated_corpus_contracts import (
    SelectedGeneratedCorpusArtifacts,
    SelectedGeneratedCorpusResult,
)


def write_selected_generated_corpus_artifacts(
    base_dir: str | Path,
    *,
    result: SelectedGeneratedCorpusResult | None = None,
) -> SelectedGeneratedCorpusArtifacts:
    from .selected_generated_corpus import (
        _render_score_gallery,
        _render_summary,
        _render_validity,
        analyze_selected_generated_corpus,
    )

    run_dir = Path(base_dir) / "selected_generated_corpus"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = result or analyze_selected_generated_corpus()

    manifest_path = run_dir / "corpus_manifest.json"
    trajectories_path = run_dir / "trajectories.csv"
    observations_path = run_dir / "observations.csv"
    truth_states_path = run_dir / "truth_states.csv"
    events_path = run_dir / "events.csv"
    environment_traces_path = run_dir / "environment_traces.csv"
    feature_matrix_path = run_dir / "feature_matrix.csv"
    class_validity_scores_path = run_dir / "class_validity_scores.csv"
    classifier_scores_path = run_dir / "classifier_scores.csv"
    posterior_history_path = run_dir / "posterior_history.csv"
    report_path = run_dir / "selected_corpus_report.md"
    summary_plot_path = run_dir / "selected_corpus_summary_dashboard.png"
    validity_plot_path = run_dir / "class_validity_breakdown.png"
    score_gallery_path = run_dir / "feature_classifier_score_gallery.png"

    manifest_path.write_text(json.dumps(payload.corpus_manifest, indent=2), encoding="utf-8")
    write_csv(trajectories_path, list(payload.trajectory_rows), list(payload.trajectory_rows[0].keys()) if payload.trajectory_rows else [])
    write_csv(observations_path, list(payload.observation_rows), list(payload.observation_rows[0].keys()) if payload.observation_rows else [])
    write_csv(truth_states_path, list(payload.truth_state_rows), list(payload.truth_state_rows[0].keys()) if payload.truth_state_rows else [])
    write_csv(events_path, list(payload.event_rows), list(payload.event_rows[0].keys()) if payload.event_rows else [])
    write_csv(environment_traces_path, list(payload.environment_rows), list(payload.environment_rows[0].keys()) if payload.environment_rows else [])
    write_csv(feature_matrix_path, list(payload.feature_rows), list(payload.feature_rows[0].keys()) if payload.feature_rows else [])
    write_csv(class_validity_scores_path, list(payload.class_validity_rows), list(payload.class_validity_rows[0].keys()) if payload.class_validity_rows else [])
    write_csv(classifier_scores_path, list(payload.classifier_score_rows), list(payload.classifier_score_rows[0].keys()) if payload.classifier_score_rows else [])
    write_csv(posterior_history_path, list(payload.posterior_rows), list(payload.posterior_rows[0].keys()) if payload.posterior_rows else [])
    report_path.write_text(payload.report_markdown, encoding="utf-8")
    summary_plot_path.write_bytes(_render_summary(payload.trajectory_rows))
    validity_plot_path.write_bytes(_render_validity(payload.class_validity_rows))
    score_gallery_path.write_bytes(_render_score_gallery(payload.feature_rows, payload.classifier_score_rows))

    return SelectedGeneratedCorpusArtifacts(
        run_dir=run_dir,
        manifest_path=manifest_path,
        trajectories_path=trajectories_path,
        observations_path=observations_path,
        truth_states_path=truth_states_path,
        events_path=events_path,
        environment_traces_path=environment_traces_path,
        feature_matrix_path=feature_matrix_path,
        class_validity_scores_path=class_validity_scores_path,
        classifier_scores_path=classifier_scores_path,
        posterior_history_path=posterior_history_path,
        report_path=report_path,
        summary_plot_path=summary_plot_path,
        validity_plot_path=validity_plot_path,
        score_gallery_path=score_gallery_path,
    )
