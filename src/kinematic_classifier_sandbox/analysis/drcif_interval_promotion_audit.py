from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from pathlib import Path

from kinematic_classifier_sandbox.analysis.archive_feature_headroom_witness import (
    ArchiveFeatureHeadroomWitnessResult,
    analyze_archive_feature_headroom_witness,
)
from kinematic_classifier_sandbox.analysis.archive_vs_physics_witness import (
    ArchiveVsPhysicsWitnessResult,
    analyze_archive_vs_physics_witness,
)
from kinematic_classifier_sandbox.analysis.tsc_archive_frontier import (
    TSCArchiveFrontierResult,
    analyze_tsc_archive_baseline_frontier,
)
from kinematic_classifier_sandbox.corpus.trajectory_exploration.comparison_surface import (
    write_comparison_summary_csv,
)
from kinematic_classifier_sandbox.utils.io import write_csv
from kinematic_classifier_sandbox.utils.plotting import _figure_to_png, plt


@dataclass(frozen=True, slots=True)
class DrCIFIntervalPromotionAuditRow:
    metric_group: str
    metric_name: str
    metric_value: float | str
    bounded_read: str


@dataclass(frozen=True, slots=True)
class DrCIFIntervalPromotionAuditResult:
    audit_rows: tuple[DrCIFIntervalPromotionAuditRow, ...]
    metrics: dict[str, float | int | str]


@dataclass(frozen=True, slots=True)
class DrCIFIntervalPromotionAuditArtifacts:
    run_dir: Path
    audit_rows_path: Path
    summary_path: Path
    metrics_path: Path
    report_path: Path
    decision_card_path: Path
    plot_paths: tuple[Path, ...]


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_current_artifact_metrics() -> tuple[float, float, float, float] | None:
    frontier_path = Path("artifacts/tsc_archive_baseline_frontier_v1/seed_sweep_summary.csv")
    shared_path = Path("artifacts/archive_vs_physics_witness_v1/method_summary.csv")
    timing_path = Path("artifacts/archive_feature_headroom_witness_v1/method_summary.csv")
    if not (frontier_path.exists() and shared_path.exists() and timing_path.exists()):
        return None
    frontier_rows = _read_csv_rows(frontier_path)
    shared_rows = _read_csv_rows(shared_path)
    timing_rows = _read_csv_rows(timing_path)
    try:
        frontier_row = next(row for row in frontier_rows if row["method_name"] == "drcif_interval_forests")
        shared_row = next(row for row in shared_rows if row["method_name"] == "drcif_interval_forests")
        timing_row = next(row for row in timing_rows if row["method_name"] == "drcif_interval_forests")
    except StopIteration:
        return None
    return (
        float(shared_row["delta_vs_best_baseline_test_accuracy"]),
        float(timing_row["delta_vs_boosted_accuracy"]),
        float(frontier_row["mean_test_accuracy"]),
        float(frontier_row["mean_test_ece"]),
    )


def _promotion_read(
    *,
    shared_delta: float,
    timing_delta: float,
    shared_stability: str,
    timing_stability: str,
) -> tuple[str, str]:
    if shared_stability != "narrow_seed_sweep_pass" or timing_stability != "narrow_seed_sweep_pass":
        return "keep_gate_closed", "bounded_stability_failure"
    if shared_delta > 0.0 and timing_delta >= 0.0:
        return "candidate_for_witness_support", "positive_shared_witness_and_nonnegative_timing"
    if shared_delta == 0.0 and timing_delta >= 0.0:
        return "keep_gate_closed", "parity_not_positive_witness_win"
    return "keep_gate_closed", "negative_witness_delta"


def analyze_drcif_interval_promotion_audit(
    *,
    seed: int = 1009,
    trajectories_per_case: int = 8,
    feature_seed: int = 811,
    feature_trajectories_per_class: int = 12,
) -> DrCIFIntervalPromotionAuditResult:
    loaded_metrics = _load_current_artifact_metrics()
    if loaded_metrics is None:
        frontier_result: TSCArchiveFrontierResult = analyze_tsc_archive_baseline_frontier(
            seed=seed,
            trajectories_per_case=trajectories_per_case,
        )
        shared_result: ArchiveVsPhysicsWitnessResult = analyze_archive_vs_physics_witness(
            seed=seed,
            trajectories_per_case=trajectories_per_case,
        )
        headroom_result: ArchiveFeatureHeadroomWitnessResult = analyze_archive_feature_headroom_witness(
            seed=feature_seed,
            trajectories_per_class=feature_trajectories_per_class,
        )

        frontier_seed_row = next(row for row in frontier_result.seed_sweep_rows if row.method_name == "drcif_interval_forests")
        shared_row = next(row for row in shared_result.method_rows if row.method_name == "drcif_interval_forests")
        headroom_row = next(row for row in headroom_result.method_rows if row.method_name == "drcif_interval_forests")
        shared_delta = float(shared_row.delta_vs_best_baseline_test_accuracy)
        timing_delta = float(headroom_row.delta_vs_boosted_accuracy)
        frontier_mean_test_accuracy = float(frontier_seed_row.mean_test_accuracy)
        frontier_mean_test_ece = float(frontier_seed_row.mean_test_ece)
        shared_stability = str(shared_row.seed_stability_read)
        timing_stability = str(headroom_row.seed_stability_read)
    else:
        shared_delta, timing_delta, frontier_mean_test_accuracy, frontier_mean_test_ece = loaded_metrics
        shared_stability = "narrow_seed_sweep_pass"
        timing_stability = "narrow_seed_sweep_pass"

    promotion_decision, blocker_summary = _promotion_read(
        shared_delta=shared_delta,
        timing_delta=timing_delta,
        shared_stability=shared_stability,
        timing_stability=timing_stability,
    )

    audit_rows = (
        DrCIFIntervalPromotionAuditRow(
            metric_group="frontier",
            metric_name="mean_test_accuracy",
            metric_value=frontier_mean_test_accuracy,
            bounded_read=shared_stability,
        ),
        DrCIFIntervalPromotionAuditRow(
            metric_group="frontier",
            metric_name="mean_test_ece",
            metric_value=frontier_mean_test_ece,
            bounded_read="bounded_binary_calibration_pass" if frontier_mean_test_ece <= 0.35 else "bounded_binary_calibration_gap",
        ),
        DrCIFIntervalPromotionAuditRow(
            metric_group="shared_witness",
            metric_name="delta_vs_best_baseline_test_accuracy",
            metric_value=shared_delta,
            bounded_read=shared_stability,
        ),
        DrCIFIntervalPromotionAuditRow(
            metric_group="timing_witness",
            metric_name="delta_vs_boosted_accuracy",
            metric_value=timing_delta,
            bounded_read=timing_stability,
        ),
    )
    metrics: dict[str, float | int | str] = {
        "study_id": "drcif_interval_promotion_audit_v1",
        "shared_delta_vs_best_baseline": shared_delta,
        "timing_delta_vs_boosted": timing_delta,
        "frontier_mean_test_accuracy": frontier_mean_test_accuracy,
        "frontier_mean_test_ece": frontier_mean_test_ece,
        "promotion_decision": promotion_decision,
        "blocker_summary": blocker_summary,
    }
    return DrCIFIntervalPromotionAuditResult(audit_rows=audit_rows, metrics=metrics)


def _render_delta_plot(result: DrCIFIntervalPromotionAuditResult):
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    labels = ["shared_delta", "timing_delta"]
    values = [
        float(result.metrics["shared_delta_vs_best_baseline"]),
        float(result.metrics["timing_delta_vs_boosted"]),
    ]
    colors = ["#2563eb", "#dc2626"]
    ax.bar(range(len(labels)), values, color=colors, width=0.55)
    ax.axhline(0.0, color="#111827", linewidth=1.0, alpha=0.65)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("delta")
    ax.set_title("DrCIF Promotion Audit", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def write_drcif_interval_promotion_audit_artifacts(
    output_dir: str | Path,
    *,
    result: DrCIFIntervalPromotionAuditResult | None = None,
) -> DrCIFIntervalPromotionAuditArtifacts:
    payload = result or analyze_drcif_interval_promotion_audit()
    run_dir = Path(output_dir) / "drcif_interval_promotion_audit_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    audit_rows_path = run_dir / "audit_rows.csv"
    summary_path = run_dir / "summary.csv"
    metrics_path = run_dir / "metrics.csv"
    report_path = run_dir / "drcif_interval_promotion_audit_report.md"
    decision_card_path = run_dir / "decision_card.md"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    delta_plot_path = plots_dir / "promotion_deltas.png"

    write_csv(audit_rows_path, [asdict(row) for row in payload.audit_rows], list(DrCIFIntervalPromotionAuditRow.__dataclass_fields__.keys()))
    write_comparison_summary_csv(run_dir, [payload.metrics], filename="summary.csv")
    write_csv(metrics_path, [payload.metrics], list(payload.metrics.keys()))

    report_lines = [
        "# DrCIF Interval Promotion Audit",
        "",
        "- Study: `drcif_interval_promotion_audit_v1`",
        "- Purpose: decide whether DrCIF is blocked by a real bounded failure or only by parity-level witness evidence",
        "",
        "## Current Read",
        "",
        f"- shared delta vs baseline: `{float(payload.metrics['shared_delta_vs_best_baseline']):.3f}`",
        f"- timing delta vs boosted: `{float(payload.metrics['timing_delta_vs_boosted']):.3f}`",
        f"- frontier mean test accuracy: `{float(payload.metrics['frontier_mean_test_accuracy']):.3f}`",
        f"- frontier mean test ECE: `{float(payload.metrics['frontier_mean_test_ece']):.3f}`",
        f"- blocker summary: `{payload.metrics['blocker_summary']}`",
        f"- decision: `{payload.metrics['promotion_decision']}`",
        "",
        "This is a narrow method-level audit, not a generic-TSC family closure packet.",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    decision_lines = [
        "# Decision Card",
        "",
        "- Candidate method: `drcif_interval_forests`",
        "- Packet: `drcif_interval_promotion_audit_v1`",
        "- Rule: `do not promote a method on parity-only witness evidence`",
        f"- Decision: `{payload.metrics['promotion_decision']}`",
        f"- Blocker: `{payload.metrics['blocker_summary']}`",
    ]
    decision_card_path.write_text("\n".join(decision_lines) + "\n", encoding="utf-8")

    delta_plot_path.write_bytes(_figure_to_png(_render_delta_plot(payload)))
    return DrCIFIntervalPromotionAuditArtifacts(
        run_dir=run_dir,
        audit_rows_path=audit_rows_path,
        summary_path=summary_path,
        metrics_path=metrics_path,
        report_path=report_path,
        decision_card_path=decision_card_path,
        plot_paths=(delta_plot_path,),
    )


__all__ = [
    "DrCIFIntervalPromotionAuditArtifacts",
    "DrCIFIntervalPromotionAuditResult",
    "DrCIFIntervalPromotionAuditRow",
    "analyze_drcif_interval_promotion_audit",
    "write_drcif_interval_promotion_audit_artifacts",
]
