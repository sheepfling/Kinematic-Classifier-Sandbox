from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from kinematic_classifier_sandbox.corpus.trajectory_exploration.runner import (
    analyze_trajectory_exploration_benchmarks,
)
from kinematic_classifier_sandbox.corpus.trajectory_exploration.comparison_surface import write_comparison_summary_csv
from kinematic_classifier_sandbox.utils.io import write_csv
from kinematic_classifier_sandbox.utils.plotting import _figure_to_png
from kinematic_classifier_sandbox.utils.plotting import plt


@dataclass(frozen=True, slots=True)
class CmaesFrontierRow:
    objective_id: str
    heuristic_mean_total_utility: float
    blackbox_mean_total_utility: float
    cmaes_mean_total_utility: float
    cmaes_minus_heuristic: float
    cmaes_minus_blackbox: float
    recommended_backend: str


@dataclass(frozen=True, slots=True)
class CmaesGeneratorWitnessResult:
    frontier_rows: tuple[CmaesFrontierRow, ...]
    metrics: dict[str, float | int | str]


@dataclass(frozen=True, slots=True)
class CmaesGeneratorWitnessArtifacts:
    run_dir: Path
    frontier_summary_path: Path
    summary_path: Path
    metrics_path: Path
    report_path: Path
    decision_card_path: Path
    plot_paths: tuple[Path, ...]


def analyze_continuous_generator_frontier(*, seed: int = 7) -> CmaesGeneratorWitnessResult:
    benchmark = analyze_trajectory_exploration_benchmarks(seed=seed)
    metrics_rows = list(benchmark.metrics_rows)
    recommendation_by_objective = {
        str(row["objective_id"]): str(row["recommended_backend"])
        for row in benchmark.backend_recommendation_rows
    }
    objective_ids = sorted({str(row["objective_id"]) for row in metrics_rows})
    frontier_rows: list[CmaesFrontierRow] = []
    for objective_id in objective_ids:
        heuristic = next(row for row in metrics_rows if str(row["objective_id"]) == objective_id and str(row["backend_id"]) == "heuristic_search")
        blackbox = next(row for row in metrics_rows if str(row["objective_id"]) == objective_id and str(row["backend_id"]) == "blackbox_optimizer")
        cmaes = next(row for row in metrics_rows if str(row["objective_id"]) == objective_id and str(row["backend_id"]) == "cmaes")
        frontier_rows.append(
            CmaesFrontierRow(
                objective_id=objective_id,
                heuristic_mean_total_utility=float(heuristic["mean_total_utility"]),
                blackbox_mean_total_utility=float(blackbox["mean_total_utility"]),
                cmaes_mean_total_utility=float(cmaes["mean_total_utility"]),
                cmaes_minus_heuristic=float(cmaes["mean_total_utility"]) - float(heuristic["mean_total_utility"]),
                cmaes_minus_blackbox=float(cmaes["mean_total_utility"]) - float(blackbox["mean_total_utility"]),
                recommended_backend=recommendation_by_objective[objective_id],
            )
        )
    mean_cmaes = mean(row.cmaes_mean_total_utility for row in frontier_rows)
    mean_heuristic = mean(row.heuristic_mean_total_utility for row in frontier_rows)
    mean_blackbox = mean(row.blackbox_mean_total_utility for row in frontier_rows)
    promotion_decision = (
        "promote_cmaes_for_continuous_generator_frontier"
        if mean_cmaes > mean_heuristic + 0.02
        else "revise_cmaes_generator_witness"
    )
    metrics: dict[str, float | int | str] = {
        "study_id": "continuous_generator_frontier_v1",
        "seed": seed,
        "objective_count": len(frontier_rows),
        "mean_heuristic_total_utility": mean_heuristic,
        "mean_blackbox_total_utility": mean_blackbox,
        "mean_cmaes_total_utility": mean_cmaes,
        "mean_cmaes_minus_heuristic": mean(row.cmaes_minus_heuristic for row in frontier_rows),
        "mean_cmaes_minus_blackbox": mean(row.cmaes_minus_blackbox for row in frontier_rows),
        "promotion_decision": promotion_decision,
    }
    return CmaesGeneratorWitnessResult(
        frontier_rows=tuple(frontier_rows),
        metrics=metrics,
    )


def _render_frontier_bars(result: CmaesGeneratorWitnessResult):
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    labels = [row.objective_id for row in result.frontier_rows]
    x = list(range(len(labels)))
    width = 0.24
    heuristic = [row.heuristic_mean_total_utility for row in result.frontier_rows]
    blackbox = [row.blackbox_mean_total_utility for row in result.frontier_rows]
    cmaes = [row.cmaes_mean_total_utility for row in result.frontier_rows]
    ax.bar([value - width for value in x], heuristic, width=width, label="heuristic_search", color="#9ca3af")
    ax.bar(x, blackbox, width=width, label="blackbox_optimizer", color="#16a34a")
    ax.bar([value + width for value in x], cmaes, width=width, label="cmaes", color="#2563eb")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("mean total utility")
    ax.set_title("Continuous Generator Frontier", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _render_delta_bars(result: CmaesGeneratorWitnessResult):
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    labels = [row.objective_id for row in result.frontier_rows]
    x = list(range(len(labels)))
    heuristic_delta = [row.cmaes_minus_heuristic for row in result.frontier_rows]
    blackbox_delta = [row.cmaes_minus_blackbox for row in result.frontier_rows]
    width = 0.32
    ax.bar([value - width / 2 for value in x], heuristic_delta, width=width, label="cmaes - heuristic", color="#2563eb")
    ax.bar([value + width / 2 for value in x], blackbox_delta, width=width, label="cmaes - blackbox", color="#7c3aed")
    ax.axhline(0.0, color="#6b7280", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("utility delta")
    ax.set_title("CMA-ES Comparative Deltas", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def write_continuous_generator_frontier_artifacts(
    output_dir: str | Path,
    *,
    result: CmaesGeneratorWitnessResult | None = None,
    seed: int = 7,
) -> CmaesGeneratorWitnessArtifacts:
    payload = result or analyze_continuous_generator_frontier(seed=seed)
    run_dir = Path(output_dir) / "continuous_generator_frontier_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    frontier_summary_path = run_dir / "frontier_summary.csv"
    summary_path = run_dir / "summary.csv"
    metrics_path = run_dir / "metrics.csv"
    report_path = run_dir / "continuous_generator_frontier_report.md"
    decision_card_path = run_dir / "decision_card.md"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    frontier_plot_path = plots_dir / "frontier_bars.png"
    delta_plot_path = plots_dir / "delta_bars.png"

    write_csv(frontier_summary_path, [asdict(row) for row in payload.frontier_rows], list(CmaesFrontierRow.__dataclass_fields__.keys()))
    write_comparison_summary_csv(run_dir, [asdict(row) for row in payload.frontier_rows], filename="summary.csv")
    write_csv(metrics_path, [payload.metrics], list(payload.metrics.keys()))

    report_lines = [
        "# Continuous Generator Frontier",
        "",
        "- Study: `continuous_generator_frontier_v1`",
        "- Compared backends: `heuristic_search`, `blackbox_optimizer`, `cmaes`",
        "",
        f"- mean heuristic total utility: `{float(payload.metrics['mean_heuristic_total_utility']):.4f}`",
        f"- mean blackbox total utility: `{float(payload.metrics['mean_blackbox_total_utility']):.4f}`",
        f"- mean cmaes total utility: `{float(payload.metrics['mean_cmaes_total_utility']):.4f}`",
        f"- mean cmaes minus heuristic: `{float(payload.metrics['mean_cmaes_minus_heuristic']):.4f}`",
        f"- decision: `{payload.metrics['promotion_decision']}`",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    decision_lines = [
        "# Decision Card",
        "",
        "- Previous methods: `heuristic_search`, `blackbox_optimizer`",
        "- Candidate method: `cmaes`",
        f"- Improvement over heuristic: `{float(payload.metrics['mean_cmaes_minus_heuristic']):.4f}` mean total utility",
        f"- Improvement over blackbox: `{float(payload.metrics['mean_cmaes_minus_blackbox']):.4f}` mean total utility",
        f"- Decision: `{payload.metrics['promotion_decision']}`",
    ]
    decision_card_path.write_text("\n".join(decision_lines) + "\n", encoding="utf-8")

    frontier_plot_path.write_bytes(_figure_to_png(_render_frontier_bars(payload)))
    delta_plot_path.write_bytes(_figure_to_png(_render_delta_bars(payload)))

    return CmaesGeneratorWitnessArtifacts(
        run_dir=run_dir,
        frontier_summary_path=frontier_summary_path,
        summary_path=summary_path,
        metrics_path=metrics_path,
        report_path=report_path,
        decision_card_path=decision_card_path,
        plot_paths=(frontier_plot_path, delta_plot_path),
    )


__all__ = [
    "CmaesFrontierRow",
    "CmaesGeneratorWitnessArtifacts",
    "CmaesGeneratorWitnessResult",
    "analyze_continuous_generator_frontier",
    "write_continuous_generator_frontier_artifacts",
]
