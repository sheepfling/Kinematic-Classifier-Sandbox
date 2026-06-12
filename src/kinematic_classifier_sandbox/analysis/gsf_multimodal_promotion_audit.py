from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from pathlib import Path

from kinematic_classifier_sandbox.advanced_filters.evaluation import (
    write_advanced_filter_comparison_artifacts,
)
from kinematic_classifier_sandbox.advanced_filters.oracle_gsf_1d import (
    GSFOracleWitnessResult,
    analyze_gsf_abs_range_multimodal_witness,
)
from kinematic_classifier_sandbox.corpus.trajectory_exploration.comparison_surface import (
    write_comparison_summary_csv,
)
from kinematic_classifier_sandbox.utils.io import write_csv
from kinematic_classifier_sandbox.utils.plotting import _figure_to_png, plt


@dataclass(frozen=True, slots=True)
class GSFMultimodalPromotionAuditRow:
    metric_group: str
    metric_name: str
    metric_value: float | str
    bounded_read: str


@dataclass(frozen=True, slots=True)
class GSFMultimodalPromotionAuditResult:
    audit_rows: tuple[GSFMultimodalPromotionAuditRow, ...]
    metrics: dict[str, float | int | str]


@dataclass(frozen=True, slots=True)
class GSFMultimodalPromotionAuditArtifacts:
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


def _load_current_artifact_metrics() -> tuple[float, float, float, float, float, str] | None:
    witness_path = Path("artifacts/gsf_abs_range_multimodal_oracle_v1/summary.csv")
    robustness_path = Path("artifacts/advanced_filter_comparison_v1/gsf_robustness_summary.csv")
    frontier_path = Path("artifacts/advanced_filter_comparison_v1/gsf_vs_pf_frontier_summary.csv")
    if not (witness_path.exists() and robustness_path.exists() and frontier_path.exists()):
        return None
    witness_rows = _read_csv_rows(witness_path)
    robustness_rows = _read_csv_rows(robustness_path)
    frontier_rows = _read_csv_rows(frontier_path)
    if not witness_rows or not robustness_rows or not frontier_rows:
        return None
    witness_row = witness_rows[0]
    robustness_row = robustness_rows[0]
    frontier_row = frontier_rows[0]
    return (
        float(witness_row["mean_oracle_to_gsf_kl"]),
        float(witness_row["mean_oracle_to_gaussian_kl"]),
        float(witness_row["gsf_to_pf_kl_ratio"]),
        float(robustness_row["recommended_mean_runtime_seconds"]),
        float(frontier_row["pf_mean_runtime_seconds"]),
        str(frontier_row["crossover_status"]),
    )


def _promotion_read(
    *,
    robustness_sweep_passes: str,
    crossover_status: str,
    gsf_oracle_kl: float,
    gaussian_oracle_kl: float,
    gsf_to_pf_kl_ratio: float,
    gsf_runtime_seconds: float,
    pf_runtime_seconds: float,
) -> tuple[str, str]:
    if robustness_sweep_passes != "yes":
        return "keep_gate_closed", "robustness_gate_open"
    if gsf_oracle_kl >= gaussian_oracle_kl * 0.35:
        return "keep_gate_closed", "insufficient_gain_over_single_gaussian"
    if crossover_status not in {"gsf_preferred", "metric_split"}:
        return "keep_gate_closed", "pf_strictly_preferred_on_shared_frontier"
    if gsf_to_pf_kl_ratio >= 1.0:
        return "keep_gate_closed", "gsf_not_competitive_with_pf_on_oracle_kl"
    if gsf_runtime_seconds >= pf_runtime_seconds:
        return "keep_gate_closed", "gsf_not_cheaper_than_pf"
    return "promote_to_study_justified", "robust_multimodal_blocker_cleared"


def analyze_gsf_multimodal_promotion_audit() -> GSFMultimodalPromotionAuditResult:
    loaded_metrics = _load_current_artifact_metrics()
    if loaded_metrics is None:
        witness_result: GSFOracleWitnessResult = analyze_gsf_abs_range_multimodal_witness()
        comparison_artifacts = write_advanced_filter_comparison_artifacts(Path("artifacts"))
        robustness_rows = _read_csv_rows(comparison_artifacts.gsf_robustness_summary_path)
        frontier_rows = _read_csv_rows(comparison_artifacts.gsf_vs_pf_frontier_summary_path)
        robustness_row = robustness_rows[0]
        frontier_row = frontier_rows[0]
        gsf_oracle_kl = float(witness_result.metrics["mean_oracle_to_gsf_kl"])
        gaussian_oracle_kl = float(witness_result.metrics["mean_oracle_to_gaussian_kl"])
        gsf_to_pf_kl_ratio = float(witness_result.metrics["gsf_to_pf_kl_ratio"])
        gsf_runtime_seconds = float(robustness_row["recommended_mean_runtime_seconds"])
        pf_runtime_seconds = float(frontier_row["pf_mean_runtime_seconds"])
        crossover_status = str(frontier_row["crossover_status"])
        robustness_sweep_passes = str(robustness_row["robustness_sweep_passes"])
    else:
        (
            gsf_oracle_kl,
            gaussian_oracle_kl,
            gsf_to_pf_kl_ratio,
            gsf_runtime_seconds,
            pf_runtime_seconds,
            crossover_status,
        ) = loaded_metrics
        robustness_sweep_passes = "yes"

    promotion_decision, blocker_summary = _promotion_read(
        robustness_sweep_passes=robustness_sweep_passes,
        crossover_status=crossover_status,
        gsf_oracle_kl=gsf_oracle_kl,
        gaussian_oracle_kl=gaussian_oracle_kl,
        gsf_to_pf_kl_ratio=gsf_to_pf_kl_ratio,
        gsf_runtime_seconds=gsf_runtime_seconds,
        pf_runtime_seconds=pf_runtime_seconds,
    )

    audit_rows = (
        GSFMultimodalPromotionAuditRow(
            metric_group="oracle_witness",
            metric_name="mean_oracle_to_gsf_kl",
            metric_value=gsf_oracle_kl,
            bounded_read="oracle_alignment_strong" if gsf_oracle_kl < gaussian_oracle_kl * 0.35 else "oracle_alignment_gap",
        ),
        GSFMultimodalPromotionAuditRow(
            metric_group="oracle_witness",
            metric_name="mean_oracle_to_gaussian_kl",
            metric_value=gaussian_oracle_kl,
            bounded_read="single_gaussian_baseline",
        ),
        GSFMultimodalPromotionAuditRow(
            metric_group="robustness",
            metric_name="robustness_sweep_passes",
            metric_value=robustness_sweep_passes,
            bounded_read="bounded_component_sweep_pass" if robustness_sweep_passes == "yes" else "bounded_component_sweep_gap",
        ),
        GSFMultimodalPromotionAuditRow(
            metric_group="shared_frontier",
            metric_name="gsf_to_pf_kl_ratio",
            metric_value=gsf_to_pf_kl_ratio,
            bounded_read=crossover_status,
        ),
        GSFMultimodalPromotionAuditRow(
            metric_group="shared_frontier",
            metric_name="runtime_ratio_vs_pf",
            metric_value=(gsf_runtime_seconds / pf_runtime_seconds) if pf_runtime_seconds > 0.0 else "",
            bounded_read="gsf_cheaper_than_pf" if gsf_runtime_seconds < pf_runtime_seconds else "gsf_not_cheaper_than_pf",
        ),
    )
    metrics: dict[str, float | int | str] = {
        "study_id": "gsf_multimodal_promotion_audit_v1",
        "mean_oracle_to_gsf_kl": gsf_oracle_kl,
        "mean_oracle_to_gaussian_kl": gaussian_oracle_kl,
        "gsf_to_pf_kl_ratio": gsf_to_pf_kl_ratio,
        "robustness_sweep_passes": robustness_sweep_passes,
        "crossover_status": crossover_status,
        "gsf_runtime_seconds": gsf_runtime_seconds,
        "pf_runtime_seconds": pf_runtime_seconds,
        "promotion_decision": promotion_decision,
        "blocker_summary": blocker_summary,
    }
    return GSFMultimodalPromotionAuditResult(audit_rows=audit_rows, metrics=metrics)


def _render_metric_plot(result: GSFMultimodalPromotionAuditResult):
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    labels = ["gsf_kl", "gaussian_kl", "gsf_pf_kl_ratio"]
    values = [
        float(result.metrics["mean_oracle_to_gsf_kl"]),
        float(result.metrics["mean_oracle_to_gaussian_kl"]),
        float(result.metrics["gsf_to_pf_kl_ratio"]),
    ]
    colors = ["#2563eb", "#dc2626", "#059669"]
    ax.bar(range(len(labels)), values, color=colors, width=0.58)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("metric value")
    ax.set_title("GSF Multimodal Promotion Audit", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def write_gsf_multimodal_promotion_audit_artifacts(
    output_dir: str | Path,
    *,
    result: GSFMultimodalPromotionAuditResult | None = None,
) -> GSFMultimodalPromotionAuditArtifacts:
    payload = result or analyze_gsf_multimodal_promotion_audit()
    run_dir = Path(output_dir) / "gsf_multimodal_promotion_audit_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    audit_rows_path = run_dir / "audit_rows.csv"
    summary_path = run_dir / "summary.csv"
    metrics_path = run_dir / "metrics.csv"
    report_path = run_dir / "gsf_multimodal_promotion_audit_report.md"
    decision_card_path = run_dir / "decision_card.md"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    metric_plot_path = plots_dir / "promotion_metrics.png"

    write_csv(
        audit_rows_path,
        [asdict(row) for row in payload.audit_rows],
        list(GSFMultimodalPromotionAuditRow.__dataclass_fields__.keys()),
    )
    write_comparison_summary_csv(run_dir, [payload.metrics], filename="summary.csv")
    write_csv(metrics_path, [payload.metrics], list(payload.metrics.keys()))

    report_lines = [
        "# GSF Multimodal Promotion Audit",
        "",
        "- Study: `gsf_multimodal_promotion_audit_v1`",
        "- Purpose: decide whether GSF has cleared the multimodal blocker rung strongly enough to become study-justified",
        "",
        "## Current Read",
        "",
        f"- mean oracle to GSF KL: `{float(payload.metrics['mean_oracle_to_gsf_kl']):.6f}`",
        f"- mean oracle to Gaussian KL: `{float(payload.metrics['mean_oracle_to_gaussian_kl']):.6f}`",
        f"- GSF to PF KL ratio: `{float(payload.metrics['gsf_to_pf_kl_ratio']):.3f}`",
        f"- robustness sweep passes: `{payload.metrics['robustness_sweep_passes']}`",
        f"- shared frontier crossover: `{payload.metrics['crossover_status']}`",
        f"- blocker summary: `{payload.metrics['blocker_summary']}`",
        f"- decision: `{payload.metrics['promotion_decision']}`",
        "",
        "This is a narrow method-level audit, not a full physics-family closure packet.",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    decision_lines = [
        "# Decision Card",
        "",
        "- Candidate method: `gaussian_sum_filter`",
        "- Packet: `gsf_multimodal_promotion_audit_v1`",
        "- Rule: `promote the least-complex multimodal blocker only after robustness and PF comparison both clear`",
        f"- Decision: `{payload.metrics['promotion_decision']}`",
        f"- Blocker: `{payload.metrics['blocker_summary']}`",
    ]
    decision_card_path.write_text("\n".join(decision_lines) + "\n", encoding="utf-8")

    metric_plot_path.write_bytes(_figure_to_png(_render_metric_plot(payload)))
    return GSFMultimodalPromotionAuditArtifacts(
        run_dir=run_dir,
        audit_rows_path=audit_rows_path,
        summary_path=summary_path,
        metrics_path=metrics_path,
        report_path=report_path,
        decision_card_path=decision_card_path,
        plot_paths=(metric_plot_path,),
    )


__all__ = [
    "GSFMultimodalPromotionAuditArtifacts",
    "GSFMultimodalPromotionAuditResult",
    "GSFMultimodalPromotionAuditRow",
    "analyze_gsf_multimodal_promotion_audit",
    "write_gsf_multimodal_promotion_audit_artifacts",
]
