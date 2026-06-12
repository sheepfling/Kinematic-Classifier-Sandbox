from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from kinematic_classifier_sandbox.analysis.tsc_archive_frontier import (
    TSCArchiveFrontierResult,
    analyze_tsc_archive_baseline_frontier,
)
from kinematic_classifier_sandbox.corpus.trajectory_exploration.comparison_surface import (
    write_comparison_summary_csv,
)
from kinematic_classifier_sandbox.utils.io import write_csv
from kinematic_classifier_sandbox.utils.plotting import _figure_to_png
from kinematic_classifier_sandbox.utils.plotting import plt


ARCHIVE_METHODS = (
    "minirocket_family",
    "drcif_interval_forests",
    "dictionary_tde_family",
    "hive_cote",
)
BASELINE_METHODS = ("windowed_robust", "kalman_bank")
SCENARIO_FIELDS = (
    ("test_accuracy", "test"),
    ("short_noisy_accuracy", "short_noisy"),
    ("endpoint_match_accuracy", "endpoint_match"),
    ("outlier_accuracy", "outlier"),
)


@dataclass(frozen=True, slots=True)
class ArchiveVsPhysicsMethodRow:
    method_name: str
    family_group: str
    backend_name: str
    claim_level: str
    test_accuracy: float
    short_noisy_accuracy: float
    endpoint_match_accuracy: float
    outlier_accuracy: float
    test_nll: float
    test_ece: float
    seed_stability_read: str
    delta_vs_best_baseline_test_accuracy: float
    delta_vs_best_baseline_short_noisy_accuracy: float
    delta_vs_best_baseline_endpoint_match_accuracy: float
    delta_vs_best_baseline_outlier_accuracy: float


@dataclass(frozen=True, slots=True)
class ArchiveVsPhysicsScenarioWinnerRow:
    metric_name: str
    winner_method: str
    winner_family_group: str
    winner_value: float
    runner_up_method: str
    runner_up_family_group: str
    runner_up_value: float
    winner_margin: float


@dataclass(frozen=True, slots=True)
class ArchiveVsPhysicsWitnessResult:
    frontier_result: TSCArchiveFrontierResult
    method_rows: tuple[ArchiveVsPhysicsMethodRow, ...]
    scenario_winner_rows: tuple[ArchiveVsPhysicsScenarioWinnerRow, ...]
    metrics: dict[str, float | int | str]


@dataclass(frozen=True, slots=True)
class ArchiveVsPhysicsWitnessArtifacts:
    run_dir: Path
    method_summary_path: Path
    scenario_winners_path: Path
    summary_path: Path
    metrics_path: Path
    report_path: Path
    decision_card_path: Path
    plot_paths: tuple[Path, ...]


def _family_group(method_name: str) -> str:
    if method_name in ARCHIVE_METHODS:
        return "archive_family"
    if method_name == "windowed_robust":
        return "interpretable_baseline"
    if method_name == "kalman_bank":
        return "physics_baseline"
    return "other"


def analyze_archive_vs_physics_witness(
    *,
    seed: int = 1009,
    trajectories_per_case: int = 8,
    backend_smoke_timeout_seconds: float = 20.0,
    seed_sweep: tuple[int, ...] | None = None,
) -> ArchiveVsPhysicsWitnessResult:
    frontier_result = analyze_tsc_archive_baseline_frontier(
        seed=seed,
        trajectories_per_case=trajectories_per_case,
        backend_smoke_timeout_seconds=backend_smoke_timeout_seconds,
        seed_sweep=seed_sweep,
    )
    metric_map = {row.method_name: row for row in frontier_result.metric_rows}
    best_baseline_test_accuracy = max(metric_map[name].test_accuracy for name in BASELINE_METHODS)
    best_baseline_short_noisy_accuracy = max(metric_map[name].short_noisy_accuracy for name in BASELINE_METHODS)
    best_baseline_endpoint_match_accuracy = max(metric_map[name].endpoint_match_accuracy for name in BASELINE_METHODS)
    best_baseline_outlier_accuracy = max(metric_map[name].outlier_accuracy for name in BASELINE_METHODS)

    ordered_methods = BASELINE_METHODS + ARCHIVE_METHODS
    method_rows = tuple(
        ArchiveVsPhysicsMethodRow(
            method_name=method_name,
            family_group=_family_group(method_name),
            backend_name=metric_map[method_name].backend_name,
            claim_level=metric_map[method_name].claim_level,
            test_accuracy=metric_map[method_name].test_accuracy,
            short_noisy_accuracy=metric_map[method_name].short_noisy_accuracy,
            endpoint_match_accuracy=metric_map[method_name].endpoint_match_accuracy,
            outlier_accuracy=metric_map[method_name].outlier_accuracy,
            test_nll=metric_map[method_name].test_nll,
            test_ece=metric_map[method_name].test_ece,
            seed_stability_read=metric_map[method_name].seed_stability_read,
            delta_vs_best_baseline_test_accuracy=metric_map[method_name].test_accuracy - best_baseline_test_accuracy,
            delta_vs_best_baseline_short_noisy_accuracy=metric_map[method_name].short_noisy_accuracy - best_baseline_short_noisy_accuracy,
            delta_vs_best_baseline_endpoint_match_accuracy=metric_map[method_name].endpoint_match_accuracy - best_baseline_endpoint_match_accuracy,
            delta_vs_best_baseline_outlier_accuracy=metric_map[method_name].outlier_accuracy - best_baseline_outlier_accuracy,
        )
        for method_name in ordered_methods
    )

    scenario_winner_rows: list[ArchiveVsPhysicsScenarioWinnerRow] = []
    for field_name, metric_name in SCENARIO_FIELDS:
        sorted_rows = sorted(
            method_rows,
            key=lambda row: getattr(row, field_name),
            reverse=True,
        )
        winner = sorted_rows[0]
        runner_up = sorted_rows[1]
        scenario_winner_rows.append(
            ArchiveVsPhysicsScenarioWinnerRow(
                metric_name=metric_name,
                winner_method=winner.method_name,
                winner_family_group=winner.family_group,
                winner_value=float(getattr(winner, field_name)),
                runner_up_method=runner_up.method_name,
                runner_up_family_group=runner_up.family_group,
                runner_up_value=float(getattr(runner_up, field_name)),
                winner_margin=float(getattr(winner, field_name) - getattr(runner_up, field_name)),
            )
        )

    archive_rows = [row for row in method_rows if row.family_group == "archive_family"]
    archive_champion = max(archive_rows, key=lambda row: row.test_accuracy)
    baseline_rows = [row for row in method_rows if row.family_group != "archive_family"]
    baseline_champion = max(baseline_rows, key=lambda row: row.test_accuracy)
    archive_scenario_win_count = sum(1 for row in scenario_winner_rows if row.winner_family_group == "archive_family")
    all_archive_external = all(row.backend_name != "local_proxy" for row in archive_rows)
    robust_enough = frontier_result.metrics["archive_seed_robustness_read"] == "narrow_seed_sweep_passes"
    calibrated_enough = frontier_result.metrics["archive_calibration_read"] == "all_methods_within_bounded_binary_ece_band"
    promotion_decision = (
        "promote_archive_vs_physics_witness_for_followon_review"
        if all_archive_external
        and robust_enough
        and calibrated_enough
        and archive_champion.test_accuracy > baseline_champion.test_accuracy
        and archive_scenario_win_count >= 1
        else "hold_archive_vs_physics_witness_until_nonfallback_external_execution"
        if not all_archive_external
        else "record_archive_vs_physics_witness_keep_gate_closed"
    )

    metrics: dict[str, float | int | str] = {
        "study_id": "archive_vs_physics_witness_v1",
        "seed": seed,
        "trajectory_count": frontier_result.metrics["trajectory_count"],
        "archive_external_family_count": frontier_result.metrics["archive_external_family_count"],
        "archive_integration_read": frontier_result.metrics["archive_integration_read"],
        "archive_seed_robustness_read": frontier_result.metrics["archive_seed_robustness_read"],
        "archive_calibration_read": frontier_result.metrics["archive_calibration_read"],
        "archive_champion_method": archive_champion.method_name,
        "archive_champion_test_accuracy": archive_champion.test_accuracy,
        "baseline_champion_method": baseline_champion.method_name,
        "baseline_champion_test_accuracy": baseline_champion.test_accuracy,
        "archive_champion_delta_vs_baseline": archive_champion.test_accuracy - baseline_champion.test_accuracy,
        "archive_scenario_win_count": archive_scenario_win_count,
        "promotion_decision": promotion_decision,
        "next_gate": (
            "run_named_witness_with_all_external_archive_backends"
            if not all_archive_external
            else "review_archive_family_promotion_against_broader_named_witnesses"
            if promotion_decision == "promote_archive_vs_physics_witness_for_followon_review"
            else "broaden_named_witness_coverage_keep_gate_closed"
        ),
    }
    return ArchiveVsPhysicsWitnessResult(
        frontier_result=frontier_result,
        method_rows=method_rows,
        scenario_winner_rows=tuple(scenario_winner_rows),
        metrics=metrics,
    )


def _render_accuracy_frontier(result: ArchiveVsPhysicsWitnessResult):
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    labels = [row.method_name for row in result.method_rows]
    values = [row.test_accuracy for row in result.method_rows]
    colors = [
        "#2563eb" if row.family_group == "physics_baseline" else "#9ca3af" if row.family_group == "interpretable_baseline" else "#7c3aed"
        for row in result.method_rows
    ]
    ax.bar(range(len(labels)), values, color=colors, width=0.65)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("test accuracy")
    ax.set_title("Archive vs Physics Witness", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _render_archive_gaps(result: ArchiveVsPhysicsWitnessResult):
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    archive_rows = [row for row in result.method_rows if row.family_group == "archive_family"]
    labels = [row.method_name for row in archive_rows]
    x = list(range(len(labels)))
    width = 0.18
    ax.bar([value - 1.5 * width for value in x], [row.delta_vs_best_baseline_test_accuracy for row in archive_rows], width=width, label="test", color="#2563eb")
    ax.bar([value - 0.5 * width for value in x], [row.delta_vs_best_baseline_short_noisy_accuracy for row in archive_rows], width=width, label="short_noisy", color="#dc2626")
    ax.bar([value + 0.5 * width for value in x], [row.delta_vs_best_baseline_endpoint_match_accuracy for row in archive_rows], width=width, label="endpoint_match", color="#16a34a")
    ax.bar([value + 1.5 * width for value in x], [row.delta_vs_best_baseline_outlier_accuracy for row in archive_rows], width=width, label="outlier", color="#d97706")
    ax.axhline(0.0, color="#111827", linewidth=1.0, alpha=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("archive minus best baseline")
    ax.set_title("Archive Family Gaps vs Best Baseline", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def write_archive_vs_physics_witness_artifacts(
    output_dir: str | Path,
    *,
    result: ArchiveVsPhysicsWitnessResult | None = None,
    seed: int = 1009,
    trajectories_per_case: int = 8,
    backend_smoke_timeout_seconds: float = 20.0,
    seed_sweep: tuple[int, ...] | None = None,
) -> ArchiveVsPhysicsWitnessArtifacts:
    payload = result or analyze_archive_vs_physics_witness(
        seed=seed,
        trajectories_per_case=trajectories_per_case,
        backend_smoke_timeout_seconds=backend_smoke_timeout_seconds,
        seed_sweep=seed_sweep,
    )
    run_dir = Path(output_dir) / "archive_vs_physics_witness_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    method_summary_path = run_dir / "method_summary.csv"
    scenario_winners_path = run_dir / "scenario_winners.csv"
    summary_path = run_dir / "summary.csv"
    metrics_path = run_dir / "metrics.csv"
    report_path = run_dir / "archive_vs_physics_witness_report.md"
    decision_card_path = run_dir / "decision_card.md"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    accuracy_plot_path = plots_dir / "archive_vs_physics_accuracy.png"
    gaps_plot_path = plots_dir / "archive_vs_physics_gaps.png"

    write_csv(
        method_summary_path,
        [asdict(row) for row in payload.method_rows],
        list(ArchiveVsPhysicsMethodRow.__dataclass_fields__.keys()),
    )
    write_csv(
        scenario_winners_path,
        [asdict(row) for row in payload.scenario_winner_rows],
        list(ArchiveVsPhysicsScenarioWinnerRow.__dataclass_fields__.keys()),
    )
    write_comparison_summary_csv(run_dir, [asdict(row) for row in payload.method_rows], filename="summary.csv")
    write_csv(metrics_path, [payload.metrics], list(payload.metrics.keys()))

    report_lines = [
        "# Archive vs Physics Witness",
        "",
        "- Study: `archive_vs_physics_witness_v1`",
        "- Shared witness family: `shared_binary_dynamics`",
        "- Archive methods: `minirocket_family`, `drcif_interval_forests`, `dictionary_tde_family`, `hive_cote`",
        "- Baselines: `windowed_robust`, `kalman_bank`",
        "",
        "## Claim Boundary",
        "",
        "This packet is a named archive-versus-baseline witness on the shared 1D corpus.",
        "It does not promote the archive lane merely because a method beats a baseline under fallback proxies.",
        "Archive-family promotion requires non-fallback external execution, bounded robustness, bounded calibration, and follow-on witness breadth.",
        "",
        f"- archive integration read: `{payload.metrics['archive_integration_read']}`",
        f"- archive seed robustness read: `{payload.metrics['archive_seed_robustness_read']}`",
        f"- archive calibration read: `{payload.metrics['archive_calibration_read']}`",
        f"- archive champion: `{payload.metrics['archive_champion_method']}` @ `{float(payload.metrics['archive_champion_test_accuracy']):.4f}`",
        f"- baseline champion: `{payload.metrics['baseline_champion_method']}` @ `{float(payload.metrics['baseline_champion_test_accuracy']):.4f}`",
        f"- archive champion delta vs baseline: `{float(payload.metrics['archive_champion_delta_vs_baseline']):.4f}`",
        f"- archive scenario win count: `{payload.metrics['archive_scenario_win_count']}`",
        f"- decision: `{payload.metrics['promotion_decision']}`",
        f"- next gate: `{payload.metrics['next_gate']}`",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    decision_lines = [
        "# Decision Card",
        "",
        "- Candidate family: `generic_time_series_benchmark_classifiers`",
        "- Comparison packet: `archive_vs_physics_witness_v1`",
        f"- Archive integration read: `{payload.metrics['archive_integration_read']}`",
        f"- Decision: `{payload.metrics['promotion_decision']}`",
        f"- Next gate: `{payload.metrics['next_gate']}`",
        "- Promotion rule: `do not promote archive families from proxy-backed witness wins; require real external archive execution plus bounded robustness/calibration and broader named witness coverage`",
    ]
    decision_card_path.write_text("\n".join(decision_lines) + "\n", encoding="utf-8")

    accuracy_plot_path.write_bytes(_figure_to_png(_render_accuracy_frontier(payload)))
    gaps_plot_path.write_bytes(_figure_to_png(_render_archive_gaps(payload)))
    return ArchiveVsPhysicsWitnessArtifacts(
        run_dir=run_dir,
        method_summary_path=method_summary_path,
        scenario_winners_path=scenario_winners_path,
        summary_path=summary_path,
        metrics_path=metrics_path,
        report_path=report_path,
        decision_card_path=decision_card_path,
        plot_paths=(accuracy_plot_path, gaps_plot_path),
    )


__all__ = [
    "ArchiveVsPhysicsMethodRow",
    "ArchiveVsPhysicsScenarioWinnerRow",
    "ArchiveVsPhysicsWitnessArtifacts",
    "ArchiveVsPhysicsWitnessResult",
    "analyze_archive_vs_physics_witness",
    "write_archive_vs_physics_witness_artifacts",
]
