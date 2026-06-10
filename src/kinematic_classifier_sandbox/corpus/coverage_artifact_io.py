from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from kinematic_classifier_sandbox.utils.io import write_csv

from .coverage_contracts import CoverageReportArtifacts, CoverageReportResult


def write_coverage_report_artifacts(
    output_dir: str | Path,
    *,
    seed: int = 7,
    trajectories_per_class: int = 5,
    thresholds=None,
    result: CoverageReportResult | None = None,
) -> CoverageReportArtifacts:
    from .coverage_report import analyze_coverage_report, render_coverage_report

    result = result or analyze_coverage_report(
        seed=seed,
        trajectories_per_class=trajectories_per_class,
        thresholds=thresholds,
    )
    output_root = Path(output_dir)
    run_dir = output_root / "coverage_report_v1"
    run_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_dir / "coverage_report.md"
    summary_path = run_dir / "coverage_report_summary.json"
    feature_set_summary_path = run_dir / "feature_set_summary.csv"
    feature_group_summary_path = run_dir / "feature_group_summary.csv"
    classifier_support_path = run_dir / "classifier_support.csv"

    report_path.write_text(render_coverage_report(result), encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "summary": asdict(result.summary),
                "corpus_adequacy_summary": asdict(result.corpus_adequacy.summary),
                "thresholds": asdict(result.corpus_adequacy.thresholds),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    write_csv(
        feature_set_summary_path,
        [dict(row) for row in result.feature_set_summary_rows],
        [
            "feature_set",
            "feature_count",
            "history_behavior",
            "description",
            "status",
            "green_features",
            "yellow_features",
            "red_features",
            "avg_moderate_or_strong_fraction",
        ],
    )
    write_csv(
        feature_group_summary_path,
        [dict(row) for row in result.feature_group_rows],
        ["feature_group", "feature_count", "status", "green_features", "yellow_features", "red_features"],
    )
    write_csv(
        classifier_support_path,
        [dict(row) for row in result.classifier_support_rows],
        [
            "classifier_id",
            "family",
            "feature_set",
            "produces",
            "feature_set_status",
            "corpus_global_status",
            "status",
            "ready_for_evaluation",
            "recommendation",
        ],
    )
    return CoverageReportArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        summary_path=summary_path,
        feature_set_summary_path=feature_set_summary_path,
        feature_group_summary_path=feature_group_summary_path,
        classifier_support_path=classifier_support_path,
    )
