from __future__ import annotations

import json
from dataclasses import asdict
from io import BytesIO
from pathlib import Path

from kinematic_classifier_sandbox.utils.io import write_csv

from ...utils.plotting import plt
from .feature_gap_trajectory_explorer_core import analyze_feature_gap_trajectory_explorer
from .feature_gap_trajectory_explorer_types import (
    FeatureGapTrajectoryExplorerArtifacts,
    FeatureGapTrajectoryExplorerResult,
)


def _gap_row_dicts(result: FeatureGapTrajectoryExplorerResult) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "iteration": row.iteration,
            "gap_id": row.gap_id,
            "gap_kind": row.gap_kind,
            "target_id": row.target_id,
            "status": row.status,
            "severity": row.severity,
            "observed_value": row.observed_value,
            "target_value": row.target_value,
            "recommendation_hint": row.recommendation_hint,
        }
        for row in result.gap_rows
    )


def _recommendation_row_dicts(result: FeatureGapTrajectoryExplorerResult) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "iteration": row.iteration,
            "recommendation_id": row.recommendation_id,
            "source_gap_id": row.source_gap_id,
            "source_gap_kind": row.source_gap_kind,
            "trajectory_family": row.trajectory_family,
            "sampler_name": row.sampler_name,
            "priority": row.priority,
            "description": row.description,
            "expected_effect": row.expected_effect,
            "measurement_scale": row.measurement_scale,
            "irregularity_scale": row.irregularity_scale,
            "outlier_scale": row.outlier_scale,
            "step_scale": row.step_scale,
        }
        | {f"{tier}_count": count for tier, count in sorted(row.tier_counts.items())}
        for row in result.recommendation_rows
    )


def _iteration_row_dicts(result: FeatureGapTrajectoryExplorerResult) -> tuple[dict[str, object], ...]:
    return tuple(asdict(row) for row in result.iteration_rows)


def _render_q_corpus_progression_png(result: FeatureGapTrajectoryExplorerResult) -> bytes:
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    if result.iteration_rows:
        xs = [row.iteration for row in result.iteration_rows]
        ax.plot(xs, [row.starting_q_corpus for row in result.iteration_rows], label="starting", linewidth=1.6, color="#64748b")
        ax.plot(xs, [row.selected_q_corpus for row in result.iteration_rows], label="selected", linewidth=2.1, color="#15803d")
    ax.set_title("Q_corpus Progression", loc="left", fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Q_corpus")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    return buffer.getvalue()


def _render_gap_priority_png(result: FeatureGapTrajectoryExplorerResult) -> bytes:
    rows = list(result.gap_rows[:12])
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    labels = [f"{row.gap_kind}:{row.target_id}" for row in rows]
    values = [row.severity for row in rows]
    colors = [{"red": "#dc2626", "yellow": "#d97706", "green": "#16a34a"}.get(row.status, "#64748b") for row in rows]
    ax.barh(range(len(rows)), values[::-1], color=colors[::-1])
    ax.set_yticks(range(len(rows)), labels[::-1])
    ax.set_xlabel("severity")
    ax.set_title("Top Gap Priorities", loc="left", fontweight="bold")
    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    return buffer.getvalue()


def _render_recommendation_family_png(result: FeatureGapTrajectoryExplorerResult) -> bytes:
    families = sorted({row.trajectory_family for row in result.recommendation_rows})
    counts = [sum(1 for row in result.recommendation_rows if row.trajectory_family == family) for family in families]
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.bar(families, counts, color="#2563eb")
    ax.set_ylabel("recommendation count")
    ax.set_title("Recommendation Families", loc="left", fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    return buffer.getvalue()


def write_feature_gap_trajectory_explorer_artifacts(
    output_dir: str | Path,
    *,
    result: FeatureGapTrajectoryExplorerResult | None = None,
) -> FeatureGapTrajectoryExplorerArtifacts:
    payload = result or analyze_feature_gap_trajectory_explorer()
    run_dir = Path(output_dir) / "feature_gap_trajectory_explorer_v1"
    plots_dir = run_dir / "plots"
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / "summary.json"
    gap_rows_path = run_dir / "gap_rows.csv"
    recommendation_rows_path = run_dir / "recommendation_rows.csv"
    iteration_rows_path = run_dir / "iteration_rows.csv"
    candidate_scores_path = run_dir / "candidate_scores.csv"
    selected_manifest_path = run_dir / "selected_manifest.json"
    report_path = run_dir / "feature_gap_trajectory_explorer_report.md"
    q_corpus_progression_png_path = plots_dir / "q_corpus_progression.png"
    gap_priority_png_path = plots_dir / "gap_priority.png"
    recommendation_family_png_path = plots_dir / "recommendation_family_counts.png"

    gap_row_dicts = _gap_row_dicts(payload)
    recommendation_row_dicts = _recommendation_row_dicts(payload)
    iteration_row_dicts = _iteration_row_dicts(payload)

    summary_path.write_text(
        json.dumps(
            {
                "initial_candidate_id": payload.initial_candidate_id,
                "final_candidate_id": payload.final_candidate_id,
                "stop_reason": payload.stop_reason,
                "selected_candidate_ids": list(payload.selected_candidate_ids),
                "iteration_count": len(payload.iteration_rows),
                "accepted_iteration_count": sum(1 for row in payload.iteration_rows if row.accepted),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_csv(gap_rows_path, list(gap_row_dicts), list(gap_row_dicts[0].keys()) if gap_row_dicts else ["iteration", "gap_id"])
    write_csv(
        recommendation_rows_path,
        list(recommendation_row_dicts),
        list(recommendation_row_dicts[0].keys()) if recommendation_row_dicts else ["iteration", "recommendation_id"],
    )
    write_csv(iteration_rows_path, list(iteration_row_dicts), list(iteration_row_dicts[0].keys()) if iteration_row_dicts else ["iteration"])
    write_csv(candidate_scores_path, list(payload.candidate_score_rows), list(payload.candidate_score_rows[0].keys()) if payload.candidate_score_rows else ["iteration"])
    selected_manifest_path.write_text(
        json.dumps(
            {
                "initial_candidate_id": payload.initial_candidate_id,
                "final_candidate_id": payload.final_candidate_id,
                "stop_reason": payload.stop_reason,
                "selected_candidates": [
                    {
                        "candidate_id": evaluation.spec.candidate_id,
                        "sampling_method": evaluation.spec.sampling_method,
                        "score_row": evaluation.score_row,
                        "adequacy_summary": asdict(evaluation.adequacy.summary),
                    }
                    for evaluation in payload.selected_evaluations
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    report_path.write_text(payload.report_markdown, encoding="utf-8")
    q_corpus_progression_png_path.write_bytes(_render_q_corpus_progression_png(payload))
    gap_priority_png_path.write_bytes(_render_gap_priority_png(payload))
    recommendation_family_png_path.write_bytes(_render_recommendation_family_png(payload))
    return FeatureGapTrajectoryExplorerArtifacts(
        run_dir=run_dir,
        summary_path=summary_path,
        gap_rows_path=gap_rows_path,
        recommendation_rows_path=recommendation_rows_path,
        iteration_rows_path=iteration_rows_path,
        candidate_scores_path=candidate_scores_path,
        selected_manifest_path=selected_manifest_path,
        report_path=report_path,
        q_corpus_progression_png_path=q_corpus_progression_png_path,
        gap_priority_png_path=gap_priority_png_path,
        recommendation_family_png_path=recommendation_family_png_path,
    )
