from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .corpus_classifier_scoring import analyze_corpus_classifier_scoring
from .corpus_policy import CorpusPolicySpec, load_corpus_policy_spec, score_qd_archive_elite
from .generated_corpus_features import collect_generated_corpus_records


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fieldnames(rows: tuple[dict[str, Any], ...]) -> list[str]:
    ordered: list[str] = []
    for row in rows:
        for key in row:
            if key not in ordered:
                ordered.append(key)
    return ordered


def _bucket(value: float, low: float, high: float) -> str:
    if value < low:
        return "low"
    if value < high:
        return "medium"
    return "high"


@dataclass(frozen=True, slots=True)
class ObjectiveDrivenQdArchiveResult:
    archive_cell_rows: tuple[dict[str, Any], ...]
    archive_elite_rows: tuple[dict[str, Any], ...]
    coverage_rows: tuple[dict[str, Any], ...]
    lineage_rows: tuple[dict[str, Any], ...]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class ObjectiveDrivenQdArchiveArtifacts:
    run_dir: Path
    archive_cells_path: Path
    archive_elites_path: Path
    coverage_path: Path
    lineage_path: Path
    report_path: Path
    coverage_plot_path: Path
    elite_distribution_path: Path
    lineage_plot_path: Path


def analyze_objective_driven_qd_archive(policy: CorpusPolicySpec | None = None) -> ObjectiveDrivenQdArchiveResult:
    resolved_policy = policy or load_corpus_policy_spec()
    records = collect_generated_corpus_records()
    scoring = analyze_corpus_classifier_scoring()
    classifier_rows = list(scoring.candidate_score_rows)
    prior_rows = list(scoring.prior_sensitivity_rows)

    aggregate_by_trajectory: dict[str, dict[str, float]] = {}
    for trajectory_id in {str(row["trajectory_id"]) for row in classifier_rows}:
        selected = [row for row in classifier_rows if str(row["trajectory_id"]) == trajectory_id]
        prior_selected = [row for row in prior_rows if str(row["trajectory_id"]) == trajectory_id]
        aggregate_by_trajectory[trajectory_id] = {
            "mean_entropy": mean(float(row["posterior_entropy"]) for row in selected) if selected else 0.0,
            "mean_margin": mean(float(row["top_two_margin"]) for row in selected) if selected else 0.0,
            "max_stress": max((float(row["measured_classifier_stress"]) for row in selected), default=0.0),
            "min_prior_flip": min((float(row["prior_flip_threshold"]) for row in prior_selected), default=1.0),
            "mean_confidence": mean(float(row["confidence"]) for row in selected) if selected else 0.0,
        }

    success_archive: dict[tuple[str, ...], dict[str, Any]] = {}
    failed_archive: dict[tuple[str, ...], dict[str, Any]] = {}
    lineage_rows: list[dict[str, Any]] = []
    raw_coverage_rows: list[dict[str, Any]] = []

    for iteration, record in enumerate(records, start=1):
        run = record.execution.trajectory_run
        trajectory_id = run.run_id
        metrics = aggregate_by_trajectory.get(
            trajectory_id,
            {
                "mean_entropy": 0.0,
                "mean_margin": 0.0,
                "max_stress": 0.0,
                "min_prior_flip": 1.0,
                "mean_confidence": 0.0,
            },
        )
        duration = float(run.times[-1] - run.times[0]) if len(run.times) >= 2 else 0.0
        truth_accel = [float(value) for value in run.truth_state.get("acceleration", ())]
        accel_range = max(truth_accel) - min(truth_accel) if truth_accel else 0.0
        successful = run.success and record.label_status == "valid_target_class"
        if successful:
            cell_id = (
                record.assigned_class,
                record.candidate.difficulty_tier,
                str(record.candidate.provenance.get("backend_id", "")),
                _bucket(duration, 2.0, 4.0),
                _bucket(accel_range, 0.12, 0.40),
                _bucket(metrics["mean_entropy"], 0.20, 0.55),
                _bucket(metrics["min_prior_flip"], 0.25, 0.65),
            )
            utility = score_qd_archive_elite(
                resolved_policy,
                validity_score=record.validity_score,
                acceleration_range_pressure=min(accel_range / 0.40, 1.0),
                classifier_stress=metrics["max_stress"],
                mean_margin_pressure=1.0 - metrics["mean_margin"],
            )
            incumbent = success_archive.get(cell_id)
            action = "new_cell"
            if incumbent is None or utility > float(incumbent["archive_utility"]):
                if incumbent is not None:
                    action = "replaced_elite"
                success_archive[cell_id] = {
                    "cell_id": "|".join(cell_id),
                    "archive_status": "successful",
                    "trajectory_id": trajectory_id,
                    "candidate_id": record.candidate.candidate_id,
                    "parent_candidate_id": str(record.candidate.provenance.get("parent_candidate_id", "")),
                    "assigned_class": record.assigned_class,
                    "difficulty_tier": record.candidate.difficulty_tier,
                    "backend_id": str(record.candidate.provenance.get("backend_id", "")),
                    "duration_bucket": cell_id[3],
                    "acceleration_bucket": cell_id[4],
                    "entropy_bucket": cell_id[5],
                    "prior_bucket": cell_id[6],
                    "archive_utility": utility,
                    "policy_id": resolved_policy.policy_id,
                    "validity_score": record.validity_score,
                    "max_classifier_stress": metrics["max_stress"],
                    "mean_entropy": metrics["mean_entropy"],
                    "min_prior_flip": metrics["min_prior_flip"],
                }
            else:
                action = "not_elite"
        else:
            failed_cell = (
                str(record.candidate.provenance.get("backend_id", "")),
                record.candidate.difficulty_tier,
                record.label_status,
            )
            failed_archive.setdefault(
                failed_cell,
                {
                    "cell_id": "|".join(failed_cell),
                    "archive_status": "failed",
                    "backend_id": failed_cell[0],
                    "difficulty_tier": failed_cell[1],
                    "failure_label_status": failed_cell[2],
                    "count": 0,
                },
            )
            failed_archive[failed_cell]["count"] = int(failed_archive[failed_cell]["count"]) + 1
            action = "failed"

        lineage_rows.append(
            {
                "iteration": iteration,
                "candidate_id": record.candidate.candidate_id,
                "trajectory_id": trajectory_id,
                "parent_candidate_id": str(record.candidate.provenance.get("parent_candidate_id", "")),
                "backend_id": str(record.candidate.provenance.get("backend_id", "")),
                "label_status": record.label_status,
                "run_success": run.success,
                "archive_action": action,
                "archive_status": "successful" if successful else "failed",
            }
        )
        raw_coverage_rows.append(
            {
                "iteration": iteration,
                "successful_cells": len(success_archive),
                "failed_cells": len(failed_archive),
                "successful_elites": len(success_archive),
            }
        )

    final_success_cells = max(len(success_archive), 1)
    final_failed_cells = max(len(failed_archive), 1)
    coverage_rows = tuple(
        {
            **row,
            "successful_coverage_fraction": float(row["successful_cells"]) / final_success_cells,
            "failed_coverage_fraction": float(row["failed_cells"]) / final_failed_cells,
        }
        for row in raw_coverage_rows
    )
    archive_elite_rows = tuple(sorted(success_archive.values(), key=lambda row: float(row["archive_utility"]), reverse=True))
    archive_cell_rows = tuple(
        sorted(
            list(success_archive.values()) + list(failed_archive.values()),
            key=lambda row: (str(row["archive_status"]), str(row["cell_id"])),
        )
    )
    report_markdown = "\n".join(
        [
            "# Objective-Driven Quality-Diversity Archive",
            "",
            "## Summary",
            f"- generated candidate records processed: `{len(records)}`",
            f"- successful archive cells: `{len(success_archive)}`",
            f"- failed archive cells: `{len(failed_archive)}`",
            f"- final successful coverage fraction: `{coverage_rows[-1]['successful_coverage_fraction']:.3f}`",
            f"- final failed coverage fraction: `{coverage_rows[-1]['failed_coverage_fraction']:.3f}`",
            "",
            "## Notes",
            "- Successful and failed coverage are tracked separately so failed mock/backend or invalid-label cases do not inflate successful corpus coverage.",
            "- Elite replacement is iterative and mutation lineage is preserved from sampler provenance.",
        ]
    )
    return ObjectiveDrivenQdArchiveResult(
        archive_cell_rows=archive_cell_rows,
        archive_elite_rows=archive_elite_rows,
        coverage_rows=coverage_rows,
        lineage_rows=tuple(lineage_rows),
        report_markdown=report_markdown,
    )


def _render_coverage(rows: tuple[dict[str, Any], ...]) -> bytes:
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.plot([row["iteration"] for row in rows], [row["successful_coverage_fraction"] for row in rows], label="successful", linewidth=1.8)
    ax.plot([row["iteration"] for row in rows], [row["failed_coverage_fraction"] for row in rows], label="failed", linewidth=1.2)
    ax.set_title("Archive Coverage By Iteration")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Coverage Fraction")
    ax.legend()
    fig.tight_layout()
    from io import BytesIO

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    return buffer.getvalue()


def _render_elites(rows: tuple[dict[str, Any], ...]) -> bytes:
    selected = list(rows[:16])
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.bar(range(len(selected)), [float(row["archive_utility"]) for row in selected], color="#4d8f77")
    ax.set_title("Elite Utility Distribution")
    ax.set_ylabel("Archive Utility")
    ax.set_xlabel("Elite Index")
    fig.tight_layout()
    from io import BytesIO

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    return buffer.getvalue()


def _render_lineage(rows: tuple[dict[str, Any], ...]) -> bytes:
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.axis("off")
    ax.text(0.50, 0.96, "Archive Mutation Lineage Preview", ha="center", va="center", fontsize=11)
    y = 0.84
    for row in list(rows[:10]):
        parent = str(row["parent_candidate_id"]) or "root"
        child = str(row["candidate_id"])
        action = str(row["archive_action"])
        ax.text(0.22, y, parent, ha="center", va="center", fontsize=8, bbox={"boxstyle": "round,pad=0.2", "facecolor": "#eee"})
        ax.text(0.50, y, action, ha="center", va="center", fontsize=8)
        ax.text(0.78, y, child, ha="center", va="center", fontsize=8, bbox={"boxstyle": "round,pad=0.2", "facecolor": "#e6f2ea"})
        ax.annotate("", xy=(0.70, y), xytext=(0.30, y), arrowprops={"arrowstyle": "->", "linewidth": 1.0})
        y -= 0.09
    fig.tight_layout()
    from io import BytesIO

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    return buffer.getvalue()


def write_objective_driven_qd_archive_artifacts(
    base_dir: str | Path,
    *,
    result: ObjectiveDrivenQdArchiveResult | None = None,
) -> ObjectiveDrivenQdArchiveArtifacts:
    run_dir = Path(base_dir) / "quality_diversity_corpus_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = result or analyze_objective_driven_qd_archive()

    archive_cells_path = run_dir / "archive_cells.csv"
    archive_elites_path = run_dir / "archive_elites.csv"
    coverage_path = run_dir / "archive_coverage_by_iteration.csv"
    lineage_path = run_dir / "archive_lineage.csv"
    report_path = run_dir / "qd_report.md"
    coverage_plot_path = run_dir / "archive_coverage_by_iteration.png"
    elite_distribution_path = run_dir / "elite_score_distribution.png"
    lineage_plot_path = run_dir / "mutation_lineage_graph.png"

    _write_csv(archive_cells_path, list(payload.archive_cell_rows), _fieldnames(payload.archive_cell_rows))
    _write_csv(archive_elites_path, list(payload.archive_elite_rows), _fieldnames(payload.archive_elite_rows))
    _write_csv(coverage_path, list(payload.coverage_rows), _fieldnames(payload.coverage_rows))
    _write_csv(lineage_path, list(payload.lineage_rows), _fieldnames(payload.lineage_rows))
    report_path.write_text(payload.report_markdown, encoding="utf-8")
    coverage_plot_path.write_bytes(_render_coverage(payload.coverage_rows))
    elite_distribution_path.write_bytes(_render_elites(payload.archive_elite_rows))
    lineage_plot_path.write_bytes(_render_lineage(payload.lineage_rows))

    return ObjectiveDrivenQdArchiveArtifacts(
        run_dir=run_dir,
        archive_cells_path=archive_cells_path,
        archive_elites_path=archive_elites_path,
        coverage_path=coverage_path,
        lineage_path=lineage_path,
        report_path=report_path,
        coverage_plot_path=coverage_plot_path,
        elite_distribution_path=elite_distribution_path,
        lineage_plot_path=lineage_plot_path,
    )
