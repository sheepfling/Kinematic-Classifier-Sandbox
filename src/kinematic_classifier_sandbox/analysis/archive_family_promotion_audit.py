from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv

from kinematic_classifier_sandbox.analysis.archive_backend_diagnosis import (
    ArchiveBackendDiagnosisResult,
    analyze_archive_backend_diagnosis,
)
from kinematic_classifier_sandbox.analysis.archive_feature_headroom_witness import (
    ArchiveFeatureHeadroomWitnessResult,
    analyze_archive_feature_headroom_witness,
)
from kinematic_classifier_sandbox.analysis.archive_vs_physics_witness import (
    ArchiveVsPhysicsWitnessResult,
    analyze_archive_vs_physics_witness,
)
from kinematic_classifier_sandbox.corpus.trajectory_exploration.comparison_surface import (
    write_comparison_summary_csv,
)
from kinematic_classifier_sandbox.utils.io import write_csv
from kinematic_classifier_sandbox.utils.plotting import _figure_to_png, plt


ARCHIVE_METHODS = (
    "minirocket_family",
    "drcif_interval_forests",
    "dictionary_tde_family",
    "hive_cote",
)


@dataclass(frozen=True, slots=True)
class ArchiveFamilyPromotionAuditRow:
    method_name: str
    shared_witness_test_accuracy: float
    shared_witness_delta_vs_baseline: float
    timing_witness_test_accuracy: float
    timing_witness_delta_vs_boosted: float
    best_diagnosis_accuracy: float
    diagnosis_warning_dominated: str
    diagnosis_best_variant: str
    diagnosis_best_resample_length: int
    promotion_read: str
    blocker_summary: str


@dataclass(frozen=True, slots=True)
class ArchiveFamilyPromotionAuditResult:
    method_rows: tuple[ArchiveFamilyPromotionAuditRow, ...]
    metrics: dict[str, float | int | str]


@dataclass(frozen=True, slots=True)
class ArchiveFamilyPromotionAuditArtifacts:
    run_dir: Path
    method_summary_path: Path
    summary_path: Path
    metrics_path: Path
    report_path: Path
    decision_card_path: Path
    plot_paths: tuple[Path, ...]


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _field(row: object, name: str) -> object:
    if isinstance(row, dict):
        return row[name]
    return getattr(row, name)


def _promotion_read(
    *,
    shared_delta: float,
    timing_delta: float,
    warning_dominated: str,
) -> tuple[str, str]:
    blockers: list[str] = []
    if shared_delta < 0.0:
        blockers.append("shared_witness_loss")
    if timing_delta < 0.0:
        blockers.append("timing_witness_loss")
    if warning_dominated == "yes":
        blockers.append("warning_dominated_diagnosis")
    if not blockers:
        return "candidate_for_followon_review", "no_bounded_blocker_detected"
    if len(blockers) == 1 and blockers[0] == "shared_witness_loss":
        return "closest_archive_candidate", blockers[0]
    return "keep_gate_closed", ",".join(blockers)


def analyze_archive_family_promotion_audit(
    *,
    shared_seed: int = 1009,
    shared_trajectories_per_case: int = 8,
    feature_seed: int = 811,
    feature_trajectories_per_class: int = 12,
) -> ArchiveFamilyPromotionAuditResult:
    shared_path = Path("artifacts/archive_vs_physics_witness_v1/method_summary.csv")
    timing_path = Path("artifacts/archive_feature_headroom_witness_v1/method_summary.csv")
    diagnosis_path = Path("artifacts/archive_backend_diagnosis_v1/summary.csv")

    if shared_path.exists() and timing_path.exists() and diagnosis_path.exists():
        shared_rows = _read_csv_rows(shared_path)
        timing_rows = _read_csv_rows(timing_path)
        diagnosis_summary_rows = _read_csv_rows(diagnosis_path)
        shared_map = {row["method_name"]: row for row in shared_rows}
        timing_map = {row["method_name"]: row for row in timing_rows}
        diagnosis_rows: dict[str, list[dict[str, str]]] = {}
        for row in diagnosis_summary_rows:
            diagnosis_rows.setdefault(row["method_name"], []).append(row)
    else:
        shared_result: ArchiveVsPhysicsWitnessResult = analyze_archive_vs_physics_witness(
            seed=shared_seed,
            trajectories_per_case=shared_trajectories_per_case,
        )
        timing_result: ArchiveFeatureHeadroomWitnessResult = analyze_archive_feature_headroom_witness(
            seed=feature_seed,
            trajectories_per_class=feature_trajectories_per_class,
        )
        diagnosis_result: ArchiveBackendDiagnosisResult = analyze_archive_backend_diagnosis(
            shared_seed=shared_seed,
            shared_trajectories_per_case=shared_trajectories_per_case,
            feature_seed=feature_seed,
            feature_trajectories_per_class=feature_trajectories_per_class,
        )

        shared_map = {row.method_name: row for row in shared_result.method_rows}
        timing_map = {row.method_name: row for row in timing_result.method_rows}
        diagnosis_rows = {}
        for row in diagnosis_result.summary_rows:
            diagnosis_rows.setdefault(row.method_name, []).append(row)

    method_rows: list[ArchiveFamilyPromotionAuditRow] = []
    for method_name in ARCHIVE_METHODS:
        best_diagnosis = max(
            diagnosis_rows[method_name],
            key=lambda row: float(_field(row, "best_test_accuracy")),
        )
        promotion_read, blocker_summary = _promotion_read(
            shared_delta=float(_field(shared_map[method_name], "delta_vs_best_baseline_test_accuracy")),
            timing_delta=float(_field(timing_map[method_name], "delta_vs_boosted_accuracy")),
            warning_dominated=str(_field(best_diagnosis, "warning_dominated")),
        )
        method_rows.append(
            ArchiveFamilyPromotionAuditRow(
                method_name=method_name,
                shared_witness_test_accuracy=float(_field(shared_map[method_name], "test_accuracy")),
                shared_witness_delta_vs_baseline=float(_field(shared_map[method_name], "delta_vs_best_baseline_test_accuracy")),
                timing_witness_test_accuracy=float(_field(timing_map[method_name], "test_accuracy")),
                timing_witness_delta_vs_boosted=float(_field(timing_map[method_name], "delta_vs_boosted_accuracy")),
                best_diagnosis_accuracy=float(_field(best_diagnosis, "best_test_accuracy")),
                diagnosis_warning_dominated=str(_field(best_diagnosis, "warning_dominated")),
                diagnosis_best_variant=str(_field(best_diagnosis, "best_panel_variant")),
                diagnosis_best_resample_length=int(_field(best_diagnosis, "best_resample_length")),
                promotion_read=promotion_read,
                blocker_summary=blocker_summary,
            )
        )

    closest_method = max(
        method_rows,
        key=lambda row: (
            row.shared_witness_delta_vs_baseline,
            row.timing_witness_delta_vs_boosted,
            row.best_diagnosis_accuracy,
        ),
    )
    metrics: dict[str, float | int | str] = {
        "study_id": "archive_family_promotion_audit_v1",
        "closest_method": closest_method.method_name,
        "closest_shared_delta_vs_baseline": closest_method.shared_witness_delta_vs_baseline,
        "closest_timing_delta_vs_boosted": closest_method.timing_witness_delta_vs_boosted,
        "closest_best_diagnosis_accuracy": closest_method.best_diagnosis_accuracy,
        "closest_blocker_summary": closest_method.blocker_summary,
        "promotion_decision": (
            "archive_family_candidate_exists"
            if closest_method.promotion_read == "candidate_for_followon_review"
            else "keep_generic_tsc_gate_closed"
        ),
    }
    return ArchiveFamilyPromotionAuditResult(
        method_rows=tuple(method_rows),
        metrics=metrics,
    )


def _render_delta_panel(result: ArchiveFamilyPromotionAuditResult):
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    labels = [row.method_name for row in result.method_rows]
    x = list(range(len(labels)))
    width = 0.35
    ax.bar(
        [value - width / 2 for value in x],
        [row.shared_witness_delta_vs_baseline for row in result.method_rows],
        width=width,
        label="shared delta vs baseline",
        color="#2563eb",
    )
    ax.bar(
        [value + width / 2 for value in x],
        [row.timing_witness_delta_vs_boosted for row in result.method_rows],
        width=width,
        label="timing delta vs boosted",
        color="#dc2626",
    )
    ax.axhline(0.0, color="#111827", linewidth=1.0, alpha=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("delta")
    ax.set_title("Archive Family Promotion Audit", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _render_diagnosis_panel(result: ArchiveFamilyPromotionAuditResult):
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    labels = [row.method_name for row in result.method_rows]
    values = [row.best_diagnosis_accuracy for row in result.method_rows]
    colors = ["#dc2626" if row.diagnosis_warning_dominated == "yes" else "#16a34a" for row in result.method_rows]
    ax.bar(range(len(labels)), values, color=colors, width=0.65)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("best diagnosis accuracy")
    ax.set_title("Archive Diagnosis Best Variant Accuracy", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def write_archive_family_promotion_audit_artifacts(
    output_dir: str | Path,
    *,
    result: ArchiveFamilyPromotionAuditResult | None = None,
) -> ArchiveFamilyPromotionAuditArtifacts:
    payload = result or analyze_archive_family_promotion_audit()
    run_dir = Path(output_dir) / "archive_family_promotion_audit_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    method_summary_path = run_dir / "method_summary.csv"
    summary_path = run_dir / "summary.csv"
    metrics_path = run_dir / "metrics.csv"
    report_path = run_dir / "archive_family_promotion_audit_report.md"
    decision_card_path = run_dir / "decision_card.md"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    delta_plot_path = plots_dir / "promotion_deltas.png"
    diagnosis_plot_path = plots_dir / "diagnosis_best_accuracy.png"

    write_csv(method_summary_path, [asdict(row) for row in payload.method_rows], list(ArchiveFamilyPromotionAuditRow.__dataclass_fields__.keys()))
    write_comparison_summary_csv(run_dir, [payload.metrics], filename="summary.csv")
    write_csv(metrics_path, [payload.metrics], list(payload.metrics.keys()))

    report_lines = [
        "# Archive Family Promotion Audit",
        "",
        "- Study: `archive_family_promotion_audit_v1`",
        "- Purpose: summarize which archive family is closest to promotion and why the gate still stays closed",
        "",
        "## Current Read",
        "",
        f"- closest method: `{payload.metrics['closest_method']}`",
        f"- shared delta vs baseline: `{float(payload.metrics['closest_shared_delta_vs_baseline']):.3f}`",
        f"- timing delta vs boosted: `{float(payload.metrics['closest_timing_delta_vs_boosted']):.3f}`",
        f"- diagnosis best accuracy: `{float(payload.metrics['closest_best_diagnosis_accuracy']):.3f}`",
        f"- blocker summary: `{payload.metrics['closest_blocker_summary']}`",
        f"- decision: `{payload.metrics['promotion_decision']}`",
        "",
        "This is a bounded method-level ranking surface, not a family promotion packet.",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    decision_lines = [
        "# Decision Card",
        "",
        "- Candidate family: `generic_time_series_benchmark_classifiers`",
        "- Packet: `archive_family_promotion_audit_v1`",
        "- Rule: `do not stop at family-wide negative narratives; identify the closest archive method and state exactly why it still fails`",
        f"- Closest method: `{payload.metrics['closest_method']}`",
        f"- Blocker summary: `{payload.metrics['closest_blocker_summary']}`",
        f"- Decision: `{payload.metrics['promotion_decision']}`",
    ]
    decision_card_path.write_text("\n".join(decision_lines) + "\n", encoding="utf-8")

    delta_plot_path.write_bytes(_figure_to_png(_render_delta_panel(payload)))
    diagnosis_plot_path.write_bytes(_figure_to_png(_render_diagnosis_panel(payload)))
    return ArchiveFamilyPromotionAuditArtifacts(
        run_dir=run_dir,
        method_summary_path=method_summary_path,
        summary_path=summary_path,
        metrics_path=metrics_path,
        report_path=report_path,
        decision_card_path=decision_card_path,
        plot_paths=(delta_plot_path, diagnosis_plot_path),
    )


__all__ = [
    "ArchiveFamilyPromotionAuditArtifacts",
    "ArchiveFamilyPromotionAuditResult",
    "ArchiveFamilyPromotionAuditRow",
    "analyze_archive_family_promotion_audit",
    "write_archive_family_promotion_audit_artifacts",
]
