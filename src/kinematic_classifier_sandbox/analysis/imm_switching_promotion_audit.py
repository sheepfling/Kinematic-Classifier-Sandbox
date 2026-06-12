from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from pathlib import Path

from kinematic_classifier_sandbox.advanced_filters.runner import run_imm_switching_benchmark
from kinematic_classifier_sandbox.corpus.trajectory_exploration.comparison_surface import (
    write_comparison_summary_csv,
)
from kinematic_classifier_sandbox.utils.io import write_csv
from kinematic_classifier_sandbox.utils.plotting import _figure_to_png, plt


@dataclass(frozen=True, slots=True)
class IMMSwitchingPromotionAuditRow:
    seed: int
    replicas: int
    imm_post_switch_accuracy: float
    transition_post_switch_accuracy: float
    static_post_switch_accuracy: float
    imm_minus_transition: float
    switch_detection_delay_median: float
    mode_nll: float
    runtime_seconds: float
    promotion_decision: str


@dataclass(frozen=True, slots=True)
class IMMSwitchingPromotionAuditResult:
    audit_rows: tuple[IMMSwitchingPromotionAuditRow, ...]
    metrics: dict[str, float | int | str]


@dataclass(frozen=True, slots=True)
class IMMSwitchingPromotionAuditArtifacts:
    run_dir: Path
    audit_rows_path: Path
    summary_path: Path
    metrics_path: Path
    report_path: Path
    decision_card_path: Path
    plot_paths: tuple[Path, ...]


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    average = sum(values) / len(values)
    return float(sqrt(sum((value - average) ** 2 for value in values) / len(values)))


def _promotion_read(
    *,
    promotion_rate: float,
    mean_delta: float,
    delta_std: float,
    mean_delay: float,
    mean_runtime_seconds: float,
) -> tuple[str, str]:
    if promotion_rate < 0.80:
        return "keep_gate_closed", "promotion_rate_below_threshold"
    if mean_delta <= 0.05:
        return "keep_gate_closed", "imm_margin_over_transition_too_small"
    if delta_std > 0.03:
        return "keep_gate_closed", "switching_gain_not_stable_enough"
    if mean_delay > 0.25:
        return "keep_gate_closed", "switch_recovery_delay_too_high"
    if mean_runtime_seconds <= 0.0:
        return "keep_gate_closed", "missing_runtime_accounting"
    return "promote_to_study_justified", "bounded_switching_state_mixing_blocker_cleared"


def analyze_imm_switching_promotion_audit(
    *,
    seeds: tuple[int, ...] = (17, 23, 29),
    replica_counts: tuple[int, ...] = (4, 8, 12),
) -> IMMSwitchingPromotionAuditResult:
    rows: list[IMMSwitchingPromotionAuditRow] = []
    for seed in seeds:
        for replicas in replica_counts:
            result = run_imm_switching_benchmark(seed=seed, replicas=replicas)
            comparison = {row["method_id"]: row for row in result.method_comparison}
            imm_post = float(comparison["imm_v1"]["post_switch_accuracy"])
            transition_post = float(comparison["transition_matrix_accumulator"]["post_switch_accuracy"])
            static_post = float(comparison["static_mode_likelihood"]["post_switch_accuracy"])
            rows.append(
                IMMSwitchingPromotionAuditRow(
                    seed=seed,
                    replicas=replicas,
                    imm_post_switch_accuracy=imm_post,
                    transition_post_switch_accuracy=transition_post,
                    static_post_switch_accuracy=static_post,
                    imm_minus_transition=imm_post - transition_post,
                    switch_detection_delay_median=float(result.metrics["switch_detection_delay_median"]),
                    mode_nll=float(result.metrics["mode_nll"]),
                    runtime_seconds=float(result.metrics["runtime_seconds"]),
                    promotion_decision=str(result.metrics["promotion_decision"]),
                )
            )

    deltas = [row.imm_minus_transition for row in rows]
    delays = [row.switch_detection_delay_median for row in rows]
    runtimes = [row.runtime_seconds for row in rows]
    promotion_rate = sum(1.0 for row in rows if row.promotion_decision == "promote") / max(len(rows), 1)
    mean_delta = sum(deltas) / max(len(deltas), 1)
    mean_delay = sum(delays) / max(len(delays), 1)
    mean_runtime = sum(runtimes) / max(len(runtimes), 1)
    delta_std = _std(deltas)
    promotion_decision, blocker_summary = _promotion_read(
        promotion_rate=promotion_rate,
        mean_delta=mean_delta,
        delta_std=delta_std,
        mean_delay=mean_delay,
        mean_runtime_seconds=mean_runtime,
    )

    metrics: dict[str, float | int | str] = {
        "study_id": "imm_switching_promotion_audit_v1",
        "case_count": len(rows),
        "seed_count": len(seeds),
        "replica_count_options": len(replica_counts),
        "promotion_rate": promotion_rate,
        "mean_delta_vs_transition": mean_delta,
        "delta_std": delta_std,
        "mean_switch_detection_delay_median": mean_delay,
        "mean_runtime_seconds": mean_runtime,
        "promotion_decision": promotion_decision,
        "blocker_summary": blocker_summary,
    }
    return IMMSwitchingPromotionAuditResult(audit_rows=tuple(rows), metrics=metrics)


def _render_delta_plot(result: IMMSwitchingPromotionAuditResult):
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    labels = ["mean_delta_vs_transition", "delta_std", "mean_switch_delay"]
    values = [
        float(result.metrics["mean_delta_vs_transition"]),
        float(result.metrics["delta_std"]),
        float(result.metrics["mean_switch_detection_delay_median"]),
    ]
    colors = ["#2563eb", "#dc2626", "#059669"]
    ax.bar(range(len(labels)), values, color=colors, width=0.58)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("summary value")
    ax.set_title("IMM Switching Promotion Audit", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def write_imm_switching_promotion_audit_artifacts(
    output_dir: str | Path,
    *,
    result: IMMSwitchingPromotionAuditResult | None = None,
) -> IMMSwitchingPromotionAuditArtifacts:
    payload = result or analyze_imm_switching_promotion_audit()
    run_dir = Path(output_dir) / "imm_switching_promotion_audit_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    audit_rows_path = run_dir / "audit_rows.csv"
    summary_path = run_dir / "summary.csv"
    metrics_path = run_dir / "metrics.csv"
    report_path = run_dir / "imm_switching_promotion_audit_report.md"
    decision_card_path = run_dir / "decision_card.md"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    delta_plot_path = plots_dir / "promotion_deltas.png"

    write_csv(
        audit_rows_path,
        [asdict(row) for row in payload.audit_rows],
        list(IMMSwitchingPromotionAuditRow.__dataclass_fields__.keys()),
    )
    write_comparison_summary_csv(run_dir, [payload.metrics], filename="summary.csv")
    write_csv(metrics_path, [payload.metrics], list(payload.metrics.keys()))

    report_lines = [
        "# IMM Switching Promotion Audit",
        "",
        "- Study: `imm_switching_promotion_audit_v1`",
        "- Purpose: decide whether IMM has cleared the switching-state-mixing blocker strongly enough to become study-justified",
        "",
        "## Current Read",
        "",
        f"- case count: `{int(payload.metrics['case_count'])}`",
        f"- promotion rate: `{float(payload.metrics['promotion_rate']):.3f}`",
        f"- mean delta vs transition: `{float(payload.metrics['mean_delta_vs_transition']):.6f}`",
        f"- delta std: `{float(payload.metrics['delta_std']):.6f}`",
        f"- mean switch delay: `{float(payload.metrics['mean_switch_detection_delay_median']):.6f}`",
        f"- blocker summary: `{payload.metrics['blocker_summary']}`",
        f"- decision: `{payload.metrics['promotion_decision']}`",
        "",
        "This is a narrow method-level audit, not a full physics-family closure packet.",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    decision_lines = [
        "# Decision Card",
        "",
        "- Candidate method: `imm`",
        "- Packet: `imm_switching_promotion_audit_v1`",
        "- Rule: `promote IMM only after switching gains survive bounded seed and witness-size sweeps`",
        f"- Decision: `{payload.metrics['promotion_decision']}`",
        f"- Blocker: `{payload.metrics['blocker_summary']}`",
    ]
    decision_card_path.write_text("\n".join(decision_lines) + "\n", encoding="utf-8")

    delta_plot_path.write_bytes(_figure_to_png(_render_delta_plot(payload)))
    return IMMSwitchingPromotionAuditArtifacts(
        run_dir=run_dir,
        audit_rows_path=audit_rows_path,
        summary_path=summary_path,
        metrics_path=metrics_path,
        report_path=report_path,
        decision_card_path=decision_card_path,
        plot_paths=(delta_plot_path,),
    )


__all__ = [
    "IMMSwitchingPromotionAuditArtifacts",
    "IMMSwitchingPromotionAuditResult",
    "IMMSwitchingPromotionAuditRow",
    "analyze_imm_switching_promotion_audit",
    "write_imm_switching_promotion_audit_artifacts",
]
