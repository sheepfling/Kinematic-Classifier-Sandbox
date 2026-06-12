from __future__ import annotations

import csv
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from kinematic_classifier_sandbox.advanced_filters.evaluation import (
    write_advanced_filter_comparison_artifacts,
)
from kinematic_classifier_sandbox.advanced_filters.oracle_ukf_1d import (
    analyze_ukf_nonlinear_unimodal_witness,
)
from kinematic_classifier_sandbox.advanced_filters.runner import run_imm_switching_benchmark
from kinematic_classifier_sandbox.analysis.gsf_multimodal_promotion_audit import (
    analyze_gsf_multimodal_promotion_audit,
)
from kinematic_classifier_sandbox.analysis.imm_switching_promotion_audit import (
    analyze_imm_switching_promotion_audit,
)
from kinematic_classifier_sandbox.analysis.ukf_nonlinear_promotion_audit import (
    analyze_ukf_nonlinear_promotion_audit,
)
from kinematic_classifier_sandbox.corpus.trajectory_exploration.comparison_surface import (
    write_comparison_summary_csv,
)
from kinematic_classifier_sandbox.registry.method_validation_os import (
    analyze_method_validation_os,
)
from kinematic_classifier_sandbox.utils.io import write_csv
from kinematic_classifier_sandbox.utils.plotting import _figure_to_png, plt


@dataclass(frozen=True, slots=True)
class PhysicsFamilyPromotionAuditRow:
    method_name: str
    current_status: str
    current_failure_status: str
    witness_id: str
    witness_decision: str
    primary_metric_name: str
    primary_metric_value: float | str
    blocker_summary: str
    promotion_read: str


@dataclass(frozen=True, slots=True)
class PhysicsFamilyPromotionAuditResult:
    method_rows: tuple[PhysicsFamilyPromotionAuditRow, ...]
    metrics: dict[str, float | int | str]


@dataclass(frozen=True, slots=True)
class PhysicsFamilyPromotionAuditArtifacts:
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


def analyze_physics_family_promotion_audit() -> PhysicsFamilyPromotionAuditResult:
    method_validation = analyze_method_validation_os()
    method_map = {row.method_id: row for row in method_validation.method_rows}

    imm_result = run_imm_switching_benchmark()
    imm_audit = analyze_imm_switching_promotion_audit()
    ukf_result = analyze_ukf_nonlinear_unimodal_witness()
    ukf_audit = analyze_ukf_nonlinear_promotion_audit()
    gsf_audit = analyze_gsf_multimodal_promotion_audit()
    with tempfile.TemporaryDirectory() as temp_dir:
        comparison_artifacts = write_advanced_filter_comparison_artifacts(temp_dir)
        gate_rows = _read_csv_rows(comparison_artifacts.gate_matrix_path)
        rbpf_row = next(row for row in gate_rows if row["method_id"] == "rbpf_v1")

    rows = [
        PhysicsFamilyPromotionAuditRow(
            method_name="imm",
            current_status=method_map["imm"].current_status,
            current_failure_status=str(method_map["imm"].current_failure_status),
            witness_id="markov_switching_acceleration",
            witness_decision=str(imm_audit.metrics["promotion_decision"]),
            primary_metric_name="post_switch_accuracy",
            primary_metric_value=float(imm_result.metrics["post_switch_accuracy"]),
            blocker_summary=str(imm_audit.metrics["blocker_summary"]),
            promotion_read=(
                "study_justified_switching_state_mixing_blocker"
                if method_map["imm"].current_status == "study_justified"
                else "witness_supported_but_not_family_closed"
            ),
        ),
        PhysicsFamilyPromotionAuditRow(
            method_name="ukf",
            current_status=method_map["ukf"].current_status,
            current_failure_status=str(method_map["ukf"].current_failure_status),
            witness_id="nonlinear_unimodal_sensor",
            witness_decision=str(ukf_audit.metrics["promotion_decision"]),
            primary_metric_name="mean_oracle_to_ukf_kl",
            primary_metric_value=float(ukf_result.metrics["mean_oracle_to_ukf_kl"]),
            blocker_summary=str(ukf_audit.metrics["blocker_summary"]),
            promotion_read=(
                "study_justified_nonlinear_gaussian_blocker"
                if method_map["ukf"].current_status == "study_justified"
                else "witness_supported_but_not_family_closed"
            ),
        ),
        PhysicsFamilyPromotionAuditRow(
            method_name="gaussian_sum_filter",
            current_status=method_map["gaussian_sum_filter"].current_status,
            current_failure_status=str(method_map["gaussian_sum_filter"].current_failure_status),
            witness_id="abs_range_multimodal_1d",
            witness_decision=str(gsf_audit.metrics["promotion_decision"]),
            primary_metric_name="gsf_to_pf_kl_ratio",
            primary_metric_value=float(gsf_audit.metrics["gsf_to_pf_kl_ratio"]),
            blocker_summary=str(gsf_audit.metrics["blocker_summary"]),
            promotion_read=(
                "study_justified_multimodal_blocker"
                if method_map["gaussian_sum_filter"].current_status == "study_justified"
                else "candidate_for_broader_comparison"
            ),
        ),
        PhysicsFamilyPromotionAuditRow(
            method_name="rbpf",
            current_status=method_map["rbpf"].current_status,
            current_failure_status=str(method_map["rbpf"].current_failure_status),
            witness_id="latent_maneuver_onset_duration",
            witness_decision=str(rbpf_row["decision_card_status"]),
            primary_metric_name="status_level",
            primary_metric_value=str(rbpf_row["status_level"]),
            blocker_summary=(
                "bounded_compute_normalized_frontier_cleared"
                if method_map["rbpf"].current_status == "study_justified"
                else "advanced_method_improves_no_robustness_not_yet_not_complexity_justified"
            ),
            promotion_read=(
                "study_justified_structured_latent_blocker"
                if method_map["rbpf"].current_status == "study_justified"
                else "witness_supported_but_not_family_closed"
            ),
        ),
    ]

    still_open = sum(1 for row in rows if row.current_status != "study_justified")
    metrics: dict[str, float | int | str] = {
        "study_id": "physics_family_promotion_audit_v1",
        "method_count": len(rows),
        "study_justified_count": sum(1 for row in rows if row.current_status == "study_justified"),
        "still_open_count": still_open,
        "family_decision": (
            "physics_family_advanced_filter_blockers_not_cleared"
            if still_open > 0
            else "physics_family_advanced_filter_blockers_cleared"
        ),
        "primary_blocker": (
            "rbpf_not_all_study_justified"
            if still_open > 0
            else "advanced_filter_core_blockers_cleared"
        ),
    }
    return PhysicsFamilyPromotionAuditResult(
        method_rows=tuple(rows),
        metrics=metrics,
    )


def _render_status_plot(result: PhysicsFamilyPromotionAuditResult):
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    labels = [row.method_name for row in result.method_rows]
    values = [1.0 if row.current_status == "study_justified" else 0.7 if row.current_status == "witness_supported" else 0.3 for row in result.method_rows]
    colors = ["#16a34a" if row.current_status == "study_justified" else "#d97706" for row in result.method_rows]
    ax.bar(range(len(labels)), values, color=colors, width=0.65)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("closure level proxy")
    ax.set_title("Physics Family Promotion Audit", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _render_blocker_plot(result: PhysicsFamilyPromotionAuditResult):
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    labels = [row.method_name for row in result.method_rows]
    text_values = [row.blocker_summary for row in result.method_rows]
    ax.axis("off")
    table = ax.table(
        cellText=[[label, blocker] for label, blocker in zip(labels, text_values, strict=True)],
        colLabels=["method", "blocker_summary"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    ax.set_title("Physics Family Remaining Blockers", loc="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def write_physics_family_promotion_audit_artifacts(
    output_dir: str | Path,
    *,
    result: PhysicsFamilyPromotionAuditResult | None = None,
) -> PhysicsFamilyPromotionAuditArtifacts:
    payload = result or analyze_physics_family_promotion_audit()
    run_dir = Path(output_dir) / "physics_family_promotion_audit_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    method_summary_path = run_dir / "method_summary.csv"
    summary_path = run_dir / "summary.csv"
    metrics_path = run_dir / "metrics.csv"
    report_path = run_dir / "physics_family_promotion_audit_report.md"
    decision_card_path = run_dir / "decision_card.md"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    status_plot_path = plots_dir / "closure_status.png"
    blocker_plot_path = plots_dir / "blocker_summary.png"

    write_csv(method_summary_path, [asdict(row) for row in payload.method_rows], list(PhysicsFamilyPromotionAuditRow.__dataclass_fields__.keys()))
    write_comparison_summary_csv(run_dir, [payload.metrics], filename="summary.csv")
    write_csv(metrics_path, [payload.metrics], list(payload.metrics.keys()))

    report_lines = [
        "# Physics Family Promotion Audit",
        "",
        "- Study: `physics_family_promotion_audit_v1`",
        "- Purpose: summarize whether the advanced-filter blocker set is still holding back the physics-aware family",
        "",
        "## Current Read",
        "",
        f"- study-justified count: `{int(payload.metrics['study_justified_count'])}`",
        f"- still-open count: `{int(payload.metrics['still_open_count'])}`",
        f"- family decision: `{payload.metrics['family_decision']}`",
        f"- primary blocker: `{payload.metrics['primary_blocker']}`",
        "",
        "This is a bounded advanced-filter family audit, not a new witness packet.",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    decision_lines = [
        "# Decision Card",
        "",
        "- Candidate family: `physics_aware_inference_classifiers`",
        "- Packet: `physics_family_promotion_audit_v1`",
        "- Rule: `do not treat advanced filters as promoted until the bounded blocker set is study-justified`",
        f"- Family decision: `{payload.metrics['family_decision']}`",
        f"- Primary blocker: `{payload.metrics['primary_blocker']}`",
    ]
    decision_card_path.write_text("\n".join(decision_lines) + "\n", encoding="utf-8")

    status_plot_path.write_bytes(_figure_to_png(_render_status_plot(payload)))
    blocker_plot_path.write_bytes(_figure_to_png(_render_blocker_plot(payload)))
    return PhysicsFamilyPromotionAuditArtifacts(
        run_dir=run_dir,
        method_summary_path=method_summary_path,
        summary_path=summary_path,
        metrics_path=metrics_path,
        report_path=report_path,
        decision_card_path=decision_card_path,
        plot_paths=(status_plot_path, blocker_plot_path),
    )


__all__ = [
    "PhysicsFamilyPromotionAuditArtifacts",
    "PhysicsFamilyPromotionAuditResult",
    "PhysicsFamilyPromotionAuditRow",
    "analyze_physics_family_promotion_audit",
    "write_physics_family_promotion_audit_artifacts",
]
