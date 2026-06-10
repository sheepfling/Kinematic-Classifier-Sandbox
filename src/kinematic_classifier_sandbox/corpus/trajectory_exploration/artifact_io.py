from __future__ import annotations

from pathlib import Path

from ...utils.io import _write_json, _write_text, write_csv
from ...utils.plotting import plt, write_plot
from .backends import BlackBoxOptimizerBackend
from .contracts import TrajectoryExplorationArtifacts
from .objectives import (
    default_trajectory_exploration_objectives,
    trajectory_exploration_evaluation_schema,
    trajectory_exploration_objective_schema,
)
from .runner import analyze_trajectory_exploration_benchmarks, run_trajectory_exploration_backend


def _render_objective_progress(rows: tuple[dict[str, object], ...]):
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["objective_id"]), []).append(row)
    for objective_id, selected in grouped.items():
        ordered = list(selected)
        ax.plot(range(len(ordered)), [float(row["best_total_utility"]) for row in ordered], label=objective_id)
    ax.set_title("Black-box objective progress", loc="left", fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best total utility")
    ax.legend()
    fig.tight_layout()
    return fig


def write_trajectory_exploration_artifacts(
    output_dir: str | Path,
    *,
    seed: int = 7,
) -> TrajectoryExplorationArtifacts:
    root = Path(output_dir)
    contract_dir = root / "trajectory_exploration_contract"
    benchmarks_dir = root / "trajectory_exploration_benchmarks"
    rl_dir = root / "trajectory_exploration_rl"
    blackbox_dir = root / "trajectory_exploration_blackbox"
    for path in (contract_dir, benchmarks_dir, rl_dir, blackbox_dir):
        path.mkdir(parents=True, exist_ok=True)

    benchmark = analyze_trajectory_exploration_benchmarks(seed=seed)
    blackbox_runs = [
        run_trajectory_exploration_backend(BlackBoxOptimizerBackend(), objective, seed=seed + index * 10)
        for index, objective in enumerate(default_trajectory_exploration_objectives())
    ]
    optimizer_trace_rows = tuple(
        {
            "objective_id": run.objective.objective_id,
            **trace_row,
        }
        for run in blackbox_runs
        for trace_row in run.objective_summary.get("optimizer_trace_rows", ())
    ) or tuple(
        {
            "objective_id": run.objective.objective_id,
            "iteration": index,
            "best_total_utility": row["total_utility"],
            "mean_duration_scale": row["duration_scale"],
            "mean_measurement_scale": row["measurement_scale"],
            "mean_irregularity_scale": row["irregularity_scale"],
            "mean_outlier_scale": row["outlier_scale"],
            "mean_step_scale": row["step_scale"],
        }
        for run in blackbox_runs
        for index, row in enumerate(run.selected_rows[:4])
    )
    elite_frontier_rows = tuple(row for run in blackbox_runs for row in run.frontier_rows)

    backend_contract_path = contract_dir / "backend_contract.json"
    objective_schema_path = contract_dir / "objective_schema.json"
    evaluation_schema_path = contract_dir / "evaluation_schema.json"
    comparison_report_path = contract_dir / "comparison_report.md"
    metrics_by_backend_path = benchmarks_dir / "metrics_by_backend.csv"
    coverage_gain_by_backend_path = benchmarks_dir / "coverage_gain_by_backend.csv"
    excitation_gain_by_backend_path = benchmarks_dir / "excitation_gain_by_backend.csv"
    overlap_reduction_by_backend_path = benchmarks_dir / "overlap_reduction_by_backend.csv"
    failure_witness_gain_by_backend_path = benchmarks_dir / "failure_witness_gain_by_backend.csv"
    budget_efficiency_path = benchmarks_dir / "budget_efficiency.csv"
    rl_decision_report_path = rl_dir / "rl_decision_report.md"
    rl_vs_blackbox_path = rl_dir / "rl_vs_blackbox_comparison.csv"
    optimizer_trace_path = blackbox_dir / "optimizer_trace.csv"
    elite_frontier_path = blackbox_dir / "elite_frontier.csv"
    objective_progress_path = blackbox_dir / "objective_progress.png"

    _write_json(backend_contract_path, benchmark.contract_payload)
    _write_json(objective_schema_path, trajectory_exploration_objective_schema())
    _write_json(evaluation_schema_path, trajectory_exploration_evaluation_schema())
    _write_text(comparison_report_path, benchmark.comparison_report_markdown)
    write_csv(metrics_by_backend_path, list(benchmark.metrics_rows), list(benchmark.metrics_rows[0].keys()))
    write_csv(coverage_gain_by_backend_path, list(benchmark.coverage_gain_rows), list(benchmark.coverage_gain_rows[0].keys()))
    write_csv(excitation_gain_by_backend_path, list(benchmark.excitation_gain_rows), list(benchmark.excitation_gain_rows[0].keys()))
    write_csv(overlap_reduction_by_backend_path, list(benchmark.overlap_reduction_rows), list(benchmark.overlap_reduction_rows[0].keys()))
    write_csv(failure_witness_gain_by_backend_path, list(benchmark.failure_witness_gain_rows), list(benchmark.failure_witness_gain_rows[0].keys()))
    write_csv(budget_efficiency_path, list(benchmark.budget_efficiency_rows), list(benchmark.budget_efficiency_rows[0].keys()))
    _write_text(rl_decision_report_path, benchmark.rl_decision_report_markdown)
    write_csv(rl_vs_blackbox_path, list(benchmark.rl_vs_blackbox_rows), list(benchmark.rl_vs_blackbox_rows[0].keys()))
    write_csv(optimizer_trace_path, list(optimizer_trace_rows), list(optimizer_trace_rows[0].keys()))
    write_csv(elite_frontier_path, list(elite_frontier_rows), list(elite_frontier_rows[0].keys()))
    write_plot(_render_objective_progress(optimizer_trace_rows), objective_progress_path)

    return TrajectoryExplorationArtifacts(
        contract_dir=contract_dir,
        benchmarks_dir=benchmarks_dir,
        rl_dir=rl_dir,
        blackbox_dir=blackbox_dir,
        backend_contract_path=backend_contract_path,
        objective_schema_path=objective_schema_path,
        evaluation_schema_path=evaluation_schema_path,
        comparison_report_path=comparison_report_path,
        metrics_by_backend_path=metrics_by_backend_path,
        coverage_gain_by_backend_path=coverage_gain_by_backend_path,
        excitation_gain_by_backend_path=excitation_gain_by_backend_path,
        overlap_reduction_by_backend_path=overlap_reduction_by_backend_path,
        failure_witness_gain_by_backend_path=failure_witness_gain_by_backend_path,
        budget_efficiency_path=budget_efficiency_path,
        rl_decision_report_path=rl_decision_report_path,
        rl_vs_blackbox_path=rl_vs_blackbox_path,
        optimizer_trace_path=optimizer_trace_path,
        elite_frontier_path=elite_frontier_path,
        objective_progress_path=objective_progress_path,
    )
