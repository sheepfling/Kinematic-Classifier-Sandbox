from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev

from kinematic_classifier_sandbox.analysis.neural_sequence_frontier import (
    NeuralSequenceFrontierResult,
    analyze_neural_sequence_vs_physics_frontier,
)
from kinematic_classifier_sandbox.corpus.trajectory_exploration.comparison_surface import (
    write_comparison_summary_csv,
)
from kinematic_classifier_sandbox.utils.io import write_csv
from kinematic_classifier_sandbox.utils.plotting import _figure_to_png, plt


@dataclass(frozen=True, slots=True)
class NeuralSequenceRobustnessSeedRow:
    seed: int
    method_name: str
    test_accuracy: float
    short_noisy_accuracy: float
    endpoint_match_accuracy: float
    test_nll: float
    test_ece: float
    calibration_delta_nll: float


@dataclass(frozen=True, slots=True)
class NeuralSequenceRobustnessSummaryRow:
    method_name: str
    mean_test_accuracy: float
    std_test_accuracy: float
    mean_short_noisy_accuracy: float
    mean_endpoint_match_accuracy: float
    mean_test_nll: float
    mean_test_ece: float
    mean_calibration_delta_nll: float
    winner_count: int
    claim_read: str


@dataclass(frozen=True, slots=True)
class NeuralSequenceRobustnessResult:
    seed_rows: tuple[NeuralSequenceRobustnessSeedRow, ...]
    summary_rows: tuple[NeuralSequenceRobustnessSummaryRow, ...]
    metrics: dict[str, float | int | str]


@dataclass(frozen=True, slots=True)
class NeuralSequenceRobustnessArtifacts:
    run_dir: Path
    seed_summary_path: Path
    metric_summary_path: Path
    summary_path: Path
    metrics_path: Path
    report_path: Path
    decision_card_path: Path
    plot_paths: tuple[Path, ...]


def _collect_seed_rows(result: NeuralSequenceFrontierResult, *, seed: int) -> list[NeuralSequenceRobustnessSeedRow]:
    rows: list[NeuralSequenceRobustnessSeedRow] = []
    for metric_row in result.metric_rows:
        rows.append(
            NeuralSequenceRobustnessSeedRow(
                seed=seed,
                method_name=metric_row.method_name,
                test_accuracy=metric_row.test_accuracy,
                short_noisy_accuracy=metric_row.short_noisy_accuracy,
                endpoint_match_accuracy=metric_row.endpoint_match_accuracy,
                test_nll=metric_row.test_nll,
                test_ece=metric_row.test_ece,
                calibration_delta_nll=metric_row.calibration_delta_nll,
            )
        )
    return rows


def analyze_neural_sequence_robustness_frontier(
    *,
    seeds: tuple[int, ...] = (907, 1013, 1117),
    trajectories_per_case: int = 8,
) -> NeuralSequenceRobustnessResult:
    seed_rows: list[NeuralSequenceRobustnessSeedRow] = []
    winner_counts: dict[str, int] = {}
    for seed in seeds:
        result = analyze_neural_sequence_vs_physics_frontier(seed=seed, trajectories_per_case=trajectories_per_case)
        seed_rows.extend(_collect_seed_rows(result, seed=seed))
        best_row = max(result.metric_rows, key=lambda row: row.test_accuracy)
        winner_counts[best_row.method_name] = winner_counts.get(best_row.method_name, 0) + 1

    method_names: list[str] = []
    for row in seed_rows:
        if row.method_name not in method_names:
            method_names.append(row.method_name)

    summary_rows: list[NeuralSequenceRobustnessSummaryRow] = []
    for method_name in method_names:
        rows = [row for row in seed_rows if row.method_name == method_name]
        mean_test_accuracy = mean(row.test_accuracy for row in rows)
        std_test_accuracy = pstdev(row.test_accuracy for row in rows) if len(rows) > 1 else 0.0
        claim_read = (
            "bounded_neural_candidate_signal"
            if method_name in {"tcn", "inceptiontime"} and mean_test_accuracy >= 0.75 and std_test_accuracy <= 0.10
            else "bounded_baseline_reference"
            if method_name in {"windowed_robust", "kalman_bank"}
            else "bounded_proxy_reference"
            if method_name == "rocket_proxy"
            else "unstable_or_underperforming"
        )
        summary_rows.append(
            NeuralSequenceRobustnessSummaryRow(
                method_name=method_name,
                mean_test_accuracy=mean_test_accuracy,
                std_test_accuracy=std_test_accuracy,
                mean_short_noisy_accuracy=mean(row.short_noisy_accuracy for row in rows),
                mean_endpoint_match_accuracy=mean(row.endpoint_match_accuracy for row in rows),
                mean_test_nll=mean(row.test_nll for row in rows),
                mean_test_ece=mean(row.test_ece for row in rows),
                mean_calibration_delta_nll=mean(row.calibration_delta_nll for row in rows),
                winner_count=winner_counts.get(method_name, 0),
                claim_read=claim_read,
            )
        )

    summary_map = {row.method_name: row for row in summary_rows}
    best_neural = max(summary_map["tcn"].mean_test_accuracy, summary_map["inceptiontime"].mean_test_accuracy)
    best_baseline = max(
        summary_map["windowed_robust"].mean_test_accuracy,
        summary_map["rocket_proxy"].mean_test_accuracy,
        summary_map["kalman_bank"].mean_test_accuracy,
    )
    neural_wins = winner_counts.get("tcn", 0) + winner_counts.get("inceptiontime", 0)
    metrics: dict[str, float | int | str] = {
        "study_id": "neural_sequence_robustness_frontier_v1",
        "seed_count": len(seeds),
        "trajectories_per_case": trajectories_per_case,
        "best_neural_mean_test_accuracy": best_neural,
        "best_baseline_mean_test_accuracy": best_baseline,
        "neural_seed_wins": neural_wins,
        "promotion_decision": (
            "bounded_neural_robustness_candidate"
            if best_neural >= best_baseline and neural_wins >= max(1, len(seeds) // 2)
            else "hold_neural_robustness_frontier"
        ),
    }
    return NeuralSequenceRobustnessResult(
        seed_rows=tuple(seed_rows),
        summary_rows=tuple(summary_rows),
        metrics=metrics,
    )


def _render_accuracy_plot(result: NeuralSequenceRobustnessResult):
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    labels = [row.method_name for row in result.summary_rows]
    values = [row.mean_test_accuracy for row in result.summary_rows]
    errors = [row.std_test_accuracy for row in result.summary_rows]
    colors = ["#2563eb", "#16a34a", "#0f766e", "#7c3aed", "#dc2626"]
    ax.bar(range(len(labels)), values, yerr=errors, color=colors, width=0.65, capsize=4)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("mean test accuracy")
    ax.set_title("Neural Sequence Robustness Frontier", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _render_seed_winner_plot(result: NeuralSequenceRobustnessResult):
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    labels = [row.method_name for row in result.summary_rows]
    values = [row.winner_count for row in result.summary_rows]
    ax.bar(range(len(labels)), values, color="#d97706", width=0.65)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("seed wins")
    ax.set_title("Neural Sequence Seed Winners", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def write_neural_sequence_robustness_frontier_artifacts(
    output_dir: str | Path,
    *,
    result: NeuralSequenceRobustnessResult | None = None,
    seeds: tuple[int, ...] = (907, 1013, 1117),
    trajectories_per_case: int = 8,
) -> NeuralSequenceRobustnessArtifacts:
    payload = result or analyze_neural_sequence_robustness_frontier(
        seeds=seeds,
        trajectories_per_case=trajectories_per_case,
    )
    run_dir = Path(output_dir) / "neural_sequence_robustness_frontier_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_summary_path = run_dir / "seed_summary.csv"
    metric_summary_path = run_dir / "metric_summary.csv"
    summary_path = run_dir / "summary.csv"
    metrics_path = run_dir / "metrics.csv"
    report_path = run_dir / "neural_sequence_robustness_frontier_report.md"
    decision_card_path = run_dir / "decision_card.md"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    accuracy_plot_path = plots_dir / "mean_accuracy_with_variance.png"
    seed_winner_plot_path = plots_dir / "seed_winner_counts.png"

    write_csv(seed_summary_path, [asdict(row) for row in payload.seed_rows], list(NeuralSequenceRobustnessSeedRow.__dataclass_fields__.keys()))
    write_csv(metric_summary_path, [asdict(row) for row in payload.summary_rows], list(NeuralSequenceRobustnessSummaryRow.__dataclass_fields__.keys()))
    write_comparison_summary_csv(run_dir, [payload.metrics], filename="summary.csv")
    write_csv(metrics_path, [payload.metrics], list(payload.metrics.keys()))

    report_lines = [
        "# Neural Sequence Robustness Frontier",
        "",
        "- Study: `neural_sequence_robustness_frontier_v1`",
        "- Purpose: add a bounded multi-seed robustness read for the learned-sequence lane",
        "",
        "## Read",
        "",
        f"- seed count: `{payload.metrics['seed_count']}`",
        f"- best neural mean test accuracy: `{float(payload.metrics['best_neural_mean_test_accuracy']):.3f}`",
        f"- best baseline mean test accuracy: `{float(payload.metrics['best_baseline_mean_test_accuracy']):.3f}`",
        f"- neural seed wins: `{int(payload.metrics['neural_seed_wins'])}`",
        f"- decision: `{payload.metrics['promotion_decision']}`",
        "",
        "This is a bounded robustness packet, not a broad generalization claim.",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    decision_lines = [
        "# Decision Card",
        "",
        "- Candidate family: `learned_sequence_and_embedding_classifiers`",
        "- Packet: `neural_sequence_robustness_frontier_v1`",
        "- Rule: `do not treat a single-seed trained frontier as family closure; require at least a bounded multi-seed read`",
        f"- Best neural mean test accuracy: `{float(payload.metrics['best_neural_mean_test_accuracy']):.3f}`",
        f"- Best baseline mean test accuracy: `{float(payload.metrics['best_baseline_mean_test_accuracy']):.3f}`",
        f"- Decision: `{payload.metrics['promotion_decision']}`",
    ]
    decision_card_path.write_text("\n".join(decision_lines) + "\n", encoding="utf-8")

    accuracy_plot_path.write_bytes(_figure_to_png(_render_accuracy_plot(payload)))
    seed_winner_plot_path.write_bytes(_figure_to_png(_render_seed_winner_plot(payload)))
    return NeuralSequenceRobustnessArtifacts(
        run_dir=run_dir,
        seed_summary_path=seed_summary_path,
        metric_summary_path=metric_summary_path,
        summary_path=summary_path,
        metrics_path=metrics_path,
        report_path=report_path,
        decision_card_path=decision_card_path,
        plot_paths=(accuracy_plot_path, seed_winner_plot_path),
    )


__all__ = [
    "NeuralSequenceRobustnessArtifacts",
    "NeuralSequenceRobustnessResult",
    "NeuralSequenceRobustnessSeedRow",
    "NeuralSequenceRobustnessSummaryRow",
    "analyze_neural_sequence_robustness_frontier",
    "write_neural_sequence_robustness_frontier_artifacts",
]
