from __future__ import annotations

from pathlib import Path

from ...utils.io import _write_json, _write_text, write_csv
from ...utils.plotting import plt, write_plot
from .comparison_surface import write_comparison_summary_csv, write_decision_card
from .backends import (
    BayesianOptimizationBackend,
    BlackBoxOptimizerBackend,
    CmaEsBackend,
    HeuristicSearchBackend,
    LatinHypercubeBackend,
    MapElitesBackend,
)
from .contracts import TrajectoryExplorationArtifacts
from .objectives import (
    default_trajectory_exploration_objectives,
    trajectory_exploration_evaluation_schema,
    trajectory_exploration_objective_schema,
)
from .ppo_boundary_control import write_sequential_ppo_boundary_control_artifacts
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


def _render_search_backend_progress(rows: tuple[dict[str, object], ...]):
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["backend_id"]), []).append(row)
    for backend_id, selected in grouped.items():
        ordered = sorted(selected, key=lambda row: (str(row["objective_id"]), int(row["iteration"])))
        ax.plot(range(len(ordered)), [float(row["best_total_utility"]) for row in ordered], label=backend_id)
    ax.set_title("Search backend progress traces", loc="left", fontweight="bold")
    ax.set_xlabel("Trace row")
    ax.set_ylabel("Best total utility")
    ax.legend()
    fig.tight_layout()
    return fig


def _render_bayesopt_report(rows: tuple[dict[str, object], ...]) -> str:
    if not rows:
        return "# Bayesian Optimization Trace Report\n\nNo Bayesian-optimization trace rows were recorded."
    mode_counts: dict[str, int] = {}
    objective_modes: dict[str, str] = {}
    best_by_objective: dict[str, float] = {}
    for row in rows:
        mode = str(row.get("acquisition_mode", ""))
        objective_id = str(row.get("objective_id", ""))
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        if objective_id and mode and objective_id not in objective_modes:
            objective_modes[objective_id] = mode
        best_by_objective[objective_id] = max(
            float(row.get("global_best_total_utility", row.get("best_total_utility", 0.0))),
            best_by_objective.get(objective_id, float("-inf")),
        )
    lines = [
        "# Bayesian Optimization Trace Report",
        "",
        "This report summarizes the acquisition modes and trace evolution for the Bayesian-optimization backend.",
        "",
        "## Acquisition Modes",
        *[f"- `{mode}`: `{count}` trace rows" for mode, count in sorted(mode_counts.items())],
        "",
        "## Objective Mapping",
        *[
            f"- `{objective_id}` -> `{objective_modes[objective_id]}` with best observed utility `{best_by_objective[objective_id]:.3f}`"
            for objective_id in sorted(objective_modes)
        ],
    ]
    return "\n".join(lines)


def _render_map_elites_report(rows: tuple[dict[str, object], ...]) -> str:
    if not rows:
        return "# MAP-Elites Trace Report\n\nNo MAP-Elites trace rows were recorded."
    best_archive_size = 0
    best_utility = float("-inf")
    objective_summaries: dict[str, dict[str, float]] = {}
    for row in rows:
        objective_id = str(row.get("objective_id", ""))
        archive_size = int(row.get("archive_size", 0))
        visited_cells = int(row.get("visited_cells", 0))
        cells_added = int(row.get("cells_added", 0))
        cells_replaced = int(row.get("cells_replaced", 0))
        best_total_utility = float(row.get("best_total_utility", 0.0))
        best_archive_size = max(best_archive_size, archive_size)
        best_utility = max(best_utility, best_total_utility)
        summary = objective_summaries.setdefault(
            objective_id,
            {
                "max_archive_size": 0.0,
                "max_visited_cells": 0.0,
                "cells_added": 0.0,
                "cells_replaced": 0.0,
                "best_total_utility": float("-inf"),
            },
        )
        summary["max_archive_size"] = max(summary["max_archive_size"], archive_size)
        summary["max_visited_cells"] = max(summary["max_visited_cells"], visited_cells)
        summary["cells_added"] += cells_added
        summary["cells_replaced"] += cells_replaced
        summary["best_total_utility"] = max(summary["best_total_utility"], best_total_utility)
    lines = [
        "# MAP-Elites Trace Report",
        "",
        "This report summarizes archive growth, sparse-cell exploration, and elite replacement behavior for the MAP-Elites backend.",
        "",
        "## Global Summary",
        f"- best archive size observed: `{best_archive_size}`",
        f"- best total utility observed: `{best_utility:.3f}`",
        "",
        "## Objective Mapping",
        *[
            (
                f"- `{objective_id}` -> archive max `{int(summary['max_archive_size'])}`, "
                f"visited cells `{int(summary['max_visited_cells'])}`, "
                f"cells added `{int(summary['cells_added'])}`, "
                f"cells replaced `{int(summary['cells_replaced'])}`, "
                f"best utility `{summary['best_total_utility']:.3f}`"
            )
            for objective_id, summary in sorted(objective_summaries.items())
        ],
    ]
    return "\n".join(lines)


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
    objectives = default_trajectory_exploration_objectives()
    blackbox_runs = [
        run_trajectory_exploration_backend(BlackBoxOptimizerBackend(), objective, seed=seed + index * 10)
        for index, objective in enumerate(objectives)
    ]
    bayesopt_runs = [
        run_trajectory_exploration_backend(BayesianOptimizationBackend(), objective, seed=seed + index * 10)
        for index, objective in enumerate(objectives)
    ]
    map_elites_runs = [
        run_trajectory_exploration_backend(MapElitesBackend(), objective, seed=seed + index * 10)
        for index, objective in enumerate(objectives)
    ]
    search_backend_factories = (
        HeuristicSearchBackend,
        LatinHypercubeBackend,
        BlackBoxOptimizerBackend,
        CmaEsBackend,
        BayesianOptimizationBackend,
    )
    search_backend_runs = [
        run_trajectory_exploration_backend(factory(), objective, seed=seed + index * 10)
        for index, objective in enumerate(objectives)
        for factory in search_backend_factories
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
    search_backend_comparison_rows = tuple(
        {
            "backend_id": run.backend_id,
            "objective_id": run.objective.objective_id,
            "best_total_utility": run.objective_summary["best_total_utility"],
            "mean_total_utility": run.objective_summary["mean_total_utility"],
            "mean_geometry_score": run.objective_summary["mean_geometry_score"],
            "selected_count": run.objective_summary["selected_count"],
            "candidate_count": run.objective_summary["candidate_count"],
        }
        for run in search_backend_runs
    )
    search_backend_trace_rows = tuple(
        {
            "backend_id": run.backend_id,
            "objective_id": run.objective.objective_id,
            **trace_row,
        }
        for run in search_backend_runs
        for trace_key in (
            "optimizer_trace_rows",
            "cmaes_trace_rows",
            "bayesopt_trace_rows",
            "lhs_trace_rows",
            "map_elites_trace_rows",
        )
        for trace_row in run.objective_summary.get(trace_key, ())
    ) or tuple(
        {
            "backend_id": run.backend_id,
            "objective_id": run.objective.objective_id,
            "iteration": 0,
            "best_total_utility": run.objective_summary["best_total_utility"],
            "mean_total_utility": run.objective_summary["mean_total_utility"],
        }
        for run in search_backend_runs
    )
    bayesopt_trace_rows = tuple(
        {
            "objective_id": run.objective.objective_id,
            **trace_row,
        }
        for run in bayesopt_runs
        for trace_row in run.objective_summary.get("bayesopt_trace_rows", ())
    )
    map_elites_trace_rows = tuple(
        {
            "objective_id": run.objective.objective_id,
            **trace_row,
        }
        for run in map_elites_runs
        for trace_row in run.objective_summary.get("map_elites_trace_rows", ())
    )

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
    search_backend_comparison_path = benchmarks_dir / "search_backend_comparison.csv"
    search_backend_trace_path = benchmarks_dir / "search_backend_traces.csv"
    search_backend_progress_path = benchmarks_dir / "search_backend_progress.png"
    backend_recommendation_path = benchmarks_dir / "backend_recommendations.csv"
    bayesopt_trace_path = benchmarks_dir / "bayesopt_trace_rows.csv"
    bayesopt_report_path = benchmarks_dir / "bayesopt_trace_report.md"
    map_elites_trace_path = benchmarks_dir / "map_elites_trace_rows.csv"
    map_elites_report_path = benchmarks_dir / "map_elites_trace_report.md"

    _write_json(backend_contract_path, benchmark.contract_payload)
    _write_json(objective_schema_path, trajectory_exploration_objective_schema())
    _write_json(evaluation_schema_path, trajectory_exploration_evaluation_schema())
    _write_text(comparison_report_path, benchmark.comparison_report_markdown)
    write_csv(metrics_by_backend_path, list(benchmark.metrics_rows), list(benchmark.metrics_rows[0].keys()))
    write_comparison_summary_csv(benchmarks_dir, benchmark.metrics_rows, filename="summary.csv")
    write_csv(coverage_gain_by_backend_path, list(benchmark.coverage_gain_rows), list(benchmark.coverage_gain_rows[0].keys()))
    write_csv(excitation_gain_by_backend_path, list(benchmark.excitation_gain_rows), list(benchmark.excitation_gain_rows[0].keys()))
    write_csv(overlap_reduction_by_backend_path, list(benchmark.overlap_reduction_rows), list(benchmark.overlap_reduction_rows[0].keys()))
    write_csv(failure_witness_gain_by_backend_path, list(benchmark.failure_witness_gain_rows), list(benchmark.failure_witness_gain_rows[0].keys()))
    write_csv(budget_efficiency_path, list(benchmark.budget_efficiency_rows), list(benchmark.budget_efficiency_rows[0].keys()))
    _write_text(rl_decision_report_path, benchmark.rl_decision_report_markdown)
    write_decision_card(rl_dir, benchmark.rl_decision_report_markdown)
    write_csv(rl_vs_blackbox_path, list(benchmark.rl_vs_blackbox_rows), list(benchmark.rl_vs_blackbox_rows[0].keys()))
    write_csv(optimizer_trace_path, list(optimizer_trace_rows), list(optimizer_trace_rows[0].keys()))
    write_csv(elite_frontier_path, list(elite_frontier_rows), list(elite_frontier_rows[0].keys()))
    write_csv(search_backend_comparison_path, list(search_backend_comparison_rows), list(search_backend_comparison_rows[0].keys()))
    write_csv(search_backend_trace_path, list(search_backend_trace_rows), list(search_backend_trace_rows[0].keys()))
    write_csv(backend_recommendation_path, list(benchmark.backend_recommendation_rows), list(benchmark.backend_recommendation_rows[0].keys()))
    write_csv(bayesopt_trace_path, list(bayesopt_trace_rows), list(bayesopt_trace_rows[0].keys()))
    _write_text(bayesopt_report_path, _render_bayesopt_report(bayesopt_trace_rows))
    write_csv(map_elites_trace_path, list(map_elites_trace_rows), list(map_elites_trace_rows[0].keys()))
    _write_text(map_elites_report_path, _render_map_elites_report(map_elites_trace_rows))
    write_plot(_render_objective_progress(optimizer_trace_rows), objective_progress_path)
    write_plot(_render_search_backend_progress(search_backend_trace_rows), search_backend_progress_path)
    ppo_result = write_sequential_ppo_boundary_control_artifacts(root)

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
        search_backend_comparison_path=search_backend_comparison_path,
        search_backend_trace_path=search_backend_trace_path,
        search_backend_progress_path=search_backend_progress_path,
        backend_recommendation_path=backend_recommendation_path,
        bayesopt_trace_path=bayesopt_trace_path,
        bayesopt_report_path=bayesopt_report_path,
        map_elites_trace_path=map_elites_trace_path,
        map_elites_report_path=map_elites_report_path,
        rl_algorithm_decision_report_path=ppo_result.artifacts.rl_algorithm_decision_report_path if ppo_result.artifacts else None,
        ppo_environment_contract_path=ppo_result.artifacts.environment_contract_path if ppo_result.artifacts else None,
        ppo_training_config_path=ppo_result.artifacts.training_config_path if ppo_result.artifacts else None,
        ppo_training_summary_path=ppo_result.artifacts.training_summary_path if ppo_result.artifacts else None,
        ppo_evaluation_rows_path=ppo_result.artifacts.evaluation_rows_path if ppo_result.artifacts else None,
        ppo_selected_rollouts_path=ppo_result.artifacts.selected_rollouts_path if ppo_result.artifacts else None,
        ppo_control_sequences_path=ppo_result.artifacts.control_sequences_path if ppo_result.artifacts else None,
        ppo_training_curve_path=ppo_result.artifacts.training_curve_path if ppo_result.artifacts else None,
        ppo_rollout_gallery_path=ppo_result.artifacts.rollout_gallery_path if ppo_result.artifacts else None,
        ppo_vs_heuristics_path=ppo_result.artifacts.ppo_vs_heuristics_path if ppo_result.artifacts else None,
        ppo_report_path=ppo_result.artifacts.report_path if ppo_result.artifacts else None,
    )
