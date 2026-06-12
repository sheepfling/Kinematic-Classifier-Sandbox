from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from pathlib import Path

from kinematic_classifier_sandbox.advanced_filters.oracle_ukf_1d import (
    UKFOracleWitnessResult,
    analyze_ukf_nonlinear_unimodal_witness,
)
from kinematic_classifier_sandbox.corpus.trajectory_exploration.comparison_surface import (
    write_comparison_summary_csv,
)
from kinematic_classifier_sandbox.utils.io import write_csv
from kinematic_classifier_sandbox.utils.plotting import _figure_to_png, plt


@dataclass(frozen=True, slots=True)
class UKFNonlinearPromotionAuditRow:
    seed: int
    measurement_offset: float
    mean_oracle_to_ukf_kl: float
    mean_oracle_to_kalman_kl: float
    ukf_rmse: float
    kalman_rmse: float
    ukf_coverage_95: float
    kalman_coverage_95: float
    runtime_seconds: float
    promotion_decision: str


@dataclass(frozen=True, slots=True)
class UKFNonlinearPromotionAuditResult:
    audit_rows: tuple[UKFNonlinearPromotionAuditRow, ...]
    metrics: dict[str, float | int | str]


@dataclass(frozen=True, slots=True)
class UKFNonlinearPromotionAuditArtifacts:
    run_dir: Path
    audit_rows_path: Path
    summary_path: Path
    metrics_path: Path
    report_path: Path
    decision_card_path: Path
    plot_paths: tuple[Path, ...]


def _seed_std(values: list[float]) -> float:
    if not values:
        return 0.0
    average = sum(values) / len(values)
    return float(sqrt(sum((value - average) ** 2 for value in values) / len(values)))


def _promotion_read(
    *,
    promotion_rate: float,
    mean_kl_ratio: float,
    mean_rmse_ratio: float,
    mean_coverage_gap: float,
    kl_ratio_std: float,
    runtime_mean: float,
) -> tuple[str, str]:
    if promotion_rate < 0.80:
        return "keep_gate_closed", "promotion_rate_below_threshold"
    if mean_kl_ratio >= 0.35:
        return "keep_gate_closed", "ukf_not_far_enough_ahead_on_oracle_kl"
    if mean_rmse_ratio >= 0.85:
        return "keep_gate_closed", "ukf_rmse_gain_not_large_enough"
    if mean_coverage_gap <= 0.40:
        return "keep_gate_closed", "coverage_gain_not_large_enough"
    if kl_ratio_std > 0.08:
        return "keep_gate_closed", "seed_or_geometry_variance_too_high"
    if runtime_mean <= 0.0:
        return "keep_gate_closed", "missing_runtime_accounting"
    return "promote_to_study_justified", "bounded_nonlinear_gaussian_blocker_cleared"


def analyze_ukf_nonlinear_promotion_audit(
    *,
    seeds: tuple[int, ...] = (307, 311, 313),
    measurement_offsets: tuple[float, ...] = (0.55, 0.75, 1.05),
) -> UKFNonlinearPromotionAuditResult:
    rows: list[UKFNonlinearPromotionAuditRow] = []
    results: list[UKFOracleWitnessResult] = []
    for seed in seeds:
        for measurement_offset in measurement_offsets:
            result = analyze_ukf_nonlinear_unimodal_witness(seed=seed, measurement_offset=measurement_offset)
            results.append(result)
            rows.append(
                UKFNonlinearPromotionAuditRow(
                    seed=seed,
                    measurement_offset=measurement_offset,
                    mean_oracle_to_ukf_kl=float(result.metrics["mean_oracle_to_ukf_kl"]),
                    mean_oracle_to_kalman_kl=float(result.metrics["mean_oracle_to_kalman_kl"]),
                    ukf_rmse=float(result.metrics["ukf_rmse"]),
                    kalman_rmse=float(result.metrics["kalman_rmse"]),
                    ukf_coverage_95=float(result.metrics["ukf_coverage_95"]),
                    kalman_coverage_95=float(result.metrics["kalman_coverage_95"]),
                    runtime_seconds=float(result.metrics["runtime_seconds"]),
                    promotion_decision=str(result.metrics["promotion_decision"]),
                )
            )

    kl_ratios = [row.mean_oracle_to_ukf_kl / max(row.mean_oracle_to_kalman_kl, 1.0e-12) for row in rows]
    rmse_ratios = [row.ukf_rmse / max(row.kalman_rmse, 1.0e-12) for row in rows]
    coverage_gaps = [row.ukf_coverage_95 - row.kalman_coverage_95 for row in rows]
    runtimes = [row.runtime_seconds for row in rows]
    promotion_rate = sum(1.0 for row in rows if row.promotion_decision == "promote_ukf_for_nonlinear_unimodal_measurement") / max(len(rows), 1)
    mean_kl_ratio = sum(kl_ratios) / max(len(kl_ratios), 1)
    mean_rmse_ratio = sum(rmse_ratios) / max(len(rmse_ratios), 1)
    mean_coverage_gap = sum(coverage_gaps) / max(len(coverage_gaps), 1)
    runtime_mean = sum(runtimes) / max(len(runtimes), 1)
    promotion_decision, blocker_summary = _promotion_read(
        promotion_rate=promotion_rate,
        mean_kl_ratio=mean_kl_ratio,
        mean_rmse_ratio=mean_rmse_ratio,
        mean_coverage_gap=mean_coverage_gap,
        kl_ratio_std=_seed_std(kl_ratios),
        runtime_mean=runtime_mean,
    )

    metrics: dict[str, float | int | str] = {
        "study_id": "ukf_nonlinear_promotion_audit_v1",
        "case_count": len(rows),
        "seed_count": len(seeds),
        "offset_count": len(measurement_offsets),
        "promotion_rate": promotion_rate,
        "mean_kl_ratio": mean_kl_ratio,
        "mean_rmse_ratio": mean_rmse_ratio,
        "mean_coverage_gap": mean_coverage_gap,
        "kl_ratio_std": _seed_std(kl_ratios),
        "mean_runtime_seconds": runtime_mean,
        "promotion_decision": promotion_decision,
        "blocker_summary": blocker_summary,
    }
    return UKFNonlinearPromotionAuditResult(audit_rows=tuple(rows), metrics=metrics)


def _render_ratio_plot(result: UKFNonlinearPromotionAuditResult):
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    labels = ["mean_kl_ratio", "mean_rmse_ratio", "mean_coverage_gap"]
    values = [
        float(result.metrics["mean_kl_ratio"]),
        float(result.metrics["mean_rmse_ratio"]),
        float(result.metrics["mean_coverage_gap"]),
    ]
    colors = ["#2563eb", "#dc2626", "#059669"]
    ax.bar(range(len(labels)), values, color=colors, width=0.58)
    ax.axhline(0.0, color="#111827", linewidth=1.0, alpha=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("summary value")
    ax.set_title("UKF Nonlinear Promotion Audit", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def write_ukf_nonlinear_promotion_audit_artifacts(
    output_dir: str | Path,
    *,
    result: UKFNonlinearPromotionAuditResult | None = None,
) -> UKFNonlinearPromotionAuditArtifacts:
    payload = result or analyze_ukf_nonlinear_promotion_audit()
    run_dir = Path(output_dir) / "ukf_nonlinear_promotion_audit_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    audit_rows_path = run_dir / "audit_rows.csv"
    summary_path = run_dir / "summary.csv"
    metrics_path = run_dir / "metrics.csv"
    report_path = run_dir / "ukf_nonlinear_promotion_audit_report.md"
    decision_card_path = run_dir / "decision_card.md"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    ratio_plot_path = plots_dir / "promotion_ratios.png"

    write_csv(
        audit_rows_path,
        [asdict(row) for row in payload.audit_rows],
        list(UKFNonlinearPromotionAuditRow.__dataclass_fields__.keys()),
    )
    write_comparison_summary_csv(run_dir, [payload.metrics], filename="summary.csv")
    write_csv(metrics_path, [payload.metrics], list(payload.metrics.keys()))

    report_lines = [
        "# UKF Nonlinear Promotion Audit",
        "",
        "- Study: `ukf_nonlinear_promotion_audit_v1`",
        "- Purpose: decide whether UKF has cleared the nonlinear-but-unimodal blocker strongly enough to become study-justified",
        "",
        "## Current Read",
        "",
        f"- case count: `{int(payload.metrics['case_count'])}`",
        f"- promotion rate: `{float(payload.metrics['promotion_rate']):.3f}`",
        f"- mean KL ratio: `{float(payload.metrics['mean_kl_ratio']):.6f}`",
        f"- mean RMSE ratio: `{float(payload.metrics['mean_rmse_ratio']):.6f}`",
        f"- mean coverage gap: `{float(payload.metrics['mean_coverage_gap']):.6f}`",
        f"- KL ratio std: `{float(payload.metrics['kl_ratio_std']):.6f}`",
        f"- blocker summary: `{payload.metrics['blocker_summary']}`",
        f"- decision: `{payload.metrics['promotion_decision']}`",
        "",
        "This is a narrow method-level audit, not a full physics-family closure packet.",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    decision_lines = [
        "# Decision Card",
        "",
        "- Candidate method: `ukf`",
        "- Packet: `ukf_nonlinear_promotion_audit_v1`",
        "- Rule: `promote the nonlinear Gaussian blocker only after the witness survives bounded seed and geometry sweeps`",
        f"- Decision: `{payload.metrics['promotion_decision']}`",
        f"- Blocker: `{payload.metrics['blocker_summary']}`",
    ]
    decision_card_path.write_text("\n".join(decision_lines) + "\n", encoding="utf-8")

    ratio_plot_path.write_bytes(_figure_to_png(_render_ratio_plot(payload)))
    return UKFNonlinearPromotionAuditArtifacts(
        run_dir=run_dir,
        audit_rows_path=audit_rows_path,
        summary_path=summary_path,
        metrics_path=metrics_path,
        report_path=report_path,
        decision_card_path=decision_card_path,
        plot_paths=(ratio_plot_path,),
    )


__all__ = [
    "UKFNonlinearPromotionAuditArtifacts",
    "UKFNonlinearPromotionAuditResult",
    "UKFNonlinearPromotionAuditRow",
    "analyze_ukf_nonlinear_promotion_audit",
    "write_ukf_nonlinear_promotion_audit_artifacts",
]
