from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kinematic_classifier_sandbox.reports.markdown import MarkdownDocument
from kinematic_classifier_sandbox.utils.io import write_csv

from .environment_aware_corpus_core import (
    _candidate_rows,
    _coverage_rows,
    _environment_adapter,
    _environment_regimes,
    _leakage_rows,
    _trajectory_summary_row,
)
from .environment_aware_corpus_rendering import (
    _render_coverage_heatmap_png,
    _render_leakage_plot_png,
    _render_trajectory_gallery_png,
)
from .environment_aware_corpus_types import (
    EnvironmentAwareCorpusArtifacts,
    EnvironmentAwareCorpusResult,
)


def analyze_environment_aware_corpus() -> EnvironmentAwareCorpusResult:
    adapter = _environment_adapter()
    candidates = _candidate_rows()
    summary_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        record = adapter.run(candidate)
        run = record.trajectory_run
        summary_rows.append(_trajectory_summary_row(candidate, {"times": run.times, "truth_state": run.truth_state, "environment_trace": run.environment_trace}))

    coverage_rows = _coverage_rows(tuple(summary_rows))
    leakage_rows = _leakage_rows(tuple(summary_rows))
    selected_flagged = sum(1 for row in leakage_rows if row["slice_id"] == "selected_corpus" and bool(row["flagged_class_linkage"]))
    control_flagged = sum(1 for row in leakage_rows if row["slice_id"] == "biased_control_slice" and bool(row["flagged_class_linkage"]))

    environment_manifest = {
        "environment_regimes": _environment_regimes(),
        "candidate_count": len(candidates),
        "selected_classes": sorted({candidate.target_class for candidate in candidates}),
        "feature_views": {
            "agnostic": "truth_state-only summaries remain available even when environment fields are ignored",
            "environment_aware": "density, wind, and drag metadata are available from normalized environment_trace",
        },
        "search_objective": "target environment regimes while preserving class balance and traceable environment metadata",
    }
    
    doc = MarkdownDocument("Atmosphere-Like 1D Environment Corpus")
    doc.heading("Summary", level=2)
    doc.bullet_list(
        [
            f"environment regimes targeted: `{len(_environment_regimes())}`",
            f"generated trajectories: `{len(candidates)}`",
            f"selected corpus leakage flags: `{selected_flagged}`",
            f"biased control leakage flags: `{control_flagged}`",
        ]
    )

    doc.heading("Environment Coverage", level=2)
    doc.table(
        ["Environment", "Class", "Count", "Mean Density", "Mean Wind Bias"],
        [
            (
                f"`{row['environment_id']}`",
                f"`{row['true_class']}`",
                f"`{row['trajectory_count']}`",
                f"`{row['mean_density']:.3f}`",
                f"`{row['mean_wind_bias']:.3f}`",
            )
            for row in coverage_rows
        ]
    )

    doc.heading("Leakage Audit", level=2)
    doc.table(
        ["Slice", "Variable", "Delta Ratio", "Flagged"],
        [
            (f"`{row['slice_id']}`", f"`{row['variable_name']}`", f"`{row['delta_ratio']:.3f}`", str(row['flagged_class_linkage']))
            for row in leakage_rows
        ]
    )

    doc.heading("Notes", level=2)
    doc.bullet_list(
        [
            "The selected corpus is balanced across environment regimes and classes, so the main slice should remain mostly unflagged.",
            "A biased control slice is audited alongside it to prove that the leakage logic can surface class-linked environment variables when they are present.",
            "The same normalized runs support both environment-agnostic and environment-aware feature views because `truth_state` and `environment_trace` are preserved separately.",
        ]
    )

    return EnvironmentAwareCorpusResult(
        environment_manifest=environment_manifest,
        environment_coverage_rows=coverage_rows,
        environment_leakage_rows=leakage_rows,
        report_markdown=doc.text(),
    )


def write_environment_aware_corpus_artifacts(
    base_dir: str | Path,
    *,
    result: EnvironmentAwareCorpusResult | None = None,
) -> EnvironmentAwareCorpusArtifacts:
    run_dir = Path(base_dir) / "environment_aware_corpus"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = result or analyze_environment_aware_corpus()

    environment_manifest_path = run_dir / "environment_manifest.json"
    environment_coverage_path = run_dir / "environment_coverage.csv"
    environment_leakage_audit_path = run_dir / "environment_leakage_audit.csv"
    report_path = run_dir / "atmosphere_like_1d_report.md"
    coverage_heatmap_png_path = run_dir / "environment_regime_coverage_heatmap.png"
    leakage_plot_png_path = run_dir / "environment_variable_leakage_by_class.png"
    trajectory_gallery_png_path = run_dir / "environment_conditioned_trajectory_gallery.png"

    environment_manifest_path.write_text(json.dumps(payload.environment_manifest, indent=2), encoding="utf-8")
    report_path.write_text(payload.report_markdown, encoding="utf-8")

    coverage_fieldnames = list(payload.environment_coverage_rows[0].keys()) if payload.environment_coverage_rows else []
    leakage_fieldnames = list(payload.environment_leakage_rows[0].keys()) if payload.environment_leakage_rows else []
    write_csv(environment_coverage_path, list(payload.environment_coverage_rows), coverage_fieldnames)
    write_csv(environment_leakage_audit_path, list(payload.environment_leakage_rows), leakage_fieldnames)

    coverage_heatmap_png_path.write_bytes(_render_coverage_heatmap_png(payload.environment_coverage_rows))
    leakage_plot_png_path.write_bytes(_render_leakage_plot_png(payload.environment_leakage_rows))
    trajectory_gallery_png_path.write_bytes(_render_trajectory_gallery_png())

    return EnvironmentAwareCorpusArtifacts(
        run_dir=run_dir,
        environment_manifest_path=environment_manifest_path,
        environment_coverage_path=environment_coverage_path,
        environment_leakage_audit_path=environment_leakage_audit_path,
        report_path=report_path,
        coverage_heatmap_png_path=coverage_heatmap_png_path,
        leakage_plot_png_path=leakage_plot_png_path,
        trajectory_gallery_png_path=trajectory_gallery_png_path,
    )
