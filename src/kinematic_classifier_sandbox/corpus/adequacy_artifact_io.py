from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from kinematic_classifier_sandbox.utils.io import write_csv

from ..utils.plotting import _figure_to_png
from .adequacy_contracts import (
    CorpusAdequacyArtifacts,
    CorpusAdequacyResult,
    CorpusAdequacyThresholds,
)


def write_corpus_adequacy_artifacts(
    output_dir: str | Path,
    *,
    seed: int = 7,
    trajectories_per_class: int = 5,
    thresholds: CorpusAdequacyThresholds | None = None,
) -> CorpusAdequacyArtifacts:
    from .adequacy_audit import (
        _render_covariate_leakage_plot,
        _render_pair_status_heatmap,
        analyze_corpus_adequacy,
        render_corpus_adequacy_report,
    )

    result = analyze_corpus_adequacy(
        seed=seed,
        trajectories_per_class=trajectories_per_class,
        thresholds=thresholds,
    )
    output_root = Path(output_dir)
    run_dir = output_root / "corpus_adequacy_audit_v1"
    run_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_dir / "corpus_adequacy_report.md"
    summary_path = run_dir / "corpus_adequacy_summary.json"
    feature_set_coverage_path = run_dir / "feature_set_coverage.csv"
    class_pair_coverage_path = run_dir / "class_pair_coverage.csv"
    class_balance_path = run_dir / "class_balance.csv"
    covariate_leakage_path = run_dir / "covariate_leakage_audit.csv"
    scorecard_path = run_dir / "corpus_adequacy_scorecard.csv"
    validity_audit_path = run_dir / "class_validity_audit.csv"
    degeneracy_report_path = run_dir / "corpus_degeneracy_report.csv"
    pair_status_heatmap_path = run_dir / "class_pair_coverage_heatmap.png"
    covariate_leakage_plot_path = run_dir / "covariate_leakage_audit.png"

    report_path.write_text(render_corpus_adequacy_report(result), encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "summary": asdict(result.summary),
                "scorecard": asdict(result.scorecard),
                "thresholds": asdict(result.thresholds),
                "recommendations": list(result.recommendations),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    write_csv(
        feature_set_coverage_path,
        [dict(row) for row in result.feature_set_rows],
        [
            "feature_set",
            "feature",
            "moderate_or_strong_count",
            "moderate_or_strong_fraction",
            "strong_count",
            "supporting_tier_count",
            "supporting_class_count",
            "supporting_tiers",
            "supporting_classes",
            "status",
            "recommendation",
        ],
    )
    pair_fieldnames = [
        "class_a",
        "class_b",
        "expected_difficulty",
        "required_tiers",
        "satisfied_tiers",
        "pairwise_auc",
        "overlap_estimate",
        "pairwise_classifier_accuracy",
        "mahalanobis_distance",
        "required_tier_min_examples",
    ]
    pair_dynamic_fields = sorted(
        {
            key
            for row in result.class_pair_rows
            for key in row.keys()
            if str(key).startswith("count_")
        }
    )
    write_csv(
        class_pair_coverage_path,
        [dict(row) for row in result.class_pair_rows],
        [*pair_fieldnames, *pair_dynamic_fields, "status", "recommendation"],
    )
    write_csv(
        class_balance_path,
        [dict(row) for row in result.class_balance_rows],
        ["tier", "true_class", "count", "expected_count", "delta_from_expected", "status", "recommendation"],
    )
    write_csv(
        covariate_leakage_path,
        [dict(row) for row in result.covariate_rows],
        ["covariate", "max_pairwise_auc", "worst_pair", "spread_ratio", "normalized_wasserstein", "min_class_mean", "max_class_mean", "status", "recommendation"],
    )
    write_csv(
        scorecard_path,
        [dict(row) for row in result.scorecard_rows],
        sorted({key for row in result.scorecard_rows for key in row.keys()}) if result.scorecard_rows else ["term", "score", "desired_direction", "artifact"],
    )
    write_csv(
        validity_audit_path,
        [dict(row) for row in result.validity_rows],
        ["trajectory_id", "target_class", "assigned_class", "validity_score", "alternate_class_similarity", "label_status"],
    )
    write_csv(
        degeneracy_report_path,
        [dict(row) for row in result.degeneracy_rows],
        ["term", "value", "interpretation"],
    )
    pair_status_heatmap_path.write_bytes(_figure_to_png(_render_pair_status_heatmap(result)))
    covariate_leakage_plot_path.write_bytes(_figure_to_png(_render_covariate_leakage_plot(result)))
    return CorpusAdequacyArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        summary_path=summary_path,
        feature_set_coverage_path=feature_set_coverage_path,
        class_pair_coverage_path=class_pair_coverage_path,
        class_balance_path=class_balance_path,
        covariate_leakage_path=covariate_leakage_path,
        scorecard_path=scorecard_path,
        validity_audit_path=validity_audit_path,
        degeneracy_report_path=degeneracy_report_path,
        pair_status_heatmap_path=pair_status_heatmap_path,
        covariate_leakage_plot_path=covariate_leakage_plot_path,
    )
