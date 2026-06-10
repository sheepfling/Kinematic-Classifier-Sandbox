from __future__ import annotations

from statistics import mean

from ...utils.io import union_fieldnames
from ..adaptive_stress import analyze_adaptive_stress_corpus
from ..gym import CorpusGymEnvironment
from ..policy import load_corpus_policy_spec
from ..quality_diversity import analyze_quality_diversity_corpus
from ..rl_backend_decision import analyze_rl_backend_decision
from ..search_baseline import analyze_corpus_search_baseline
from ..exploration.feature_gap_trajectory_explorer import analyze_feature_gap_trajectory_explorer
from .backends import BlackBoxOptimizerBackend, HeuristicSearchBackend, StatelessRlPolicyBackend
from .contracts import (
    TrajectoryExplorationBackend,
    TrajectoryExplorationBenchmarkResult,
    TrajectoryExplorationEvaluation,
    TrajectoryExplorationObjective,
    TrajectoryExplorationResult,
)
from .objectives import (
    default_trajectory_exploration_objectives,
    evaluate_proposal,
    objective_as_row,
)


def _select_rows(candidate_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    ranked = sorted(candidate_rows, key=lambda row: float(row["total_utility"]), reverse=True)
    selected_count = max(1, len(ranked) // 4)
    selected_ids = {str(row["proposal_id"]) for row in ranked[:selected_count]}
    rows: list[dict[str, object]] = []
    for row in ranked:
        payload = dict(row)
        payload["selected"] = payload["proposal_id"] in selected_ids
        rows.append(payload)
    return rows


def _coverage_rows(rows: list[dict[str, object]]) -> tuple[dict[str, object], ...]:
    if not rows:
        return ()
    totals: list[dict[str, object]] = []
    seen_trajectories: set[str] = set()
    ordered_rows = sorted(rows, key=lambda item: (int(item["iteration"]), -float(item["total_utility"])))
    for index, row in enumerate(ordered_rows, start=1):
        seen_trajectories.add(str(row["trajectory_id"]))
        totals.append(
            {
                "iteration": row["iteration"],
                "evaluations_seen": index,
                "unique_trajectories": len(seen_trajectories),
                "coverage_fraction": len(seen_trajectories) / max(len(rows), 1),
                "best_total_utility_so_far": max(float(item["total_utility"]) for item in ordered_rows[:index]),
            }
        )
    return tuple(totals)


def _frontier_rows(rows: list[dict[str, object]]) -> tuple[dict[str, object], ...]:
    frontier = sorted(rows, key=lambda row: (float(row["geometry_score"]), float(row["total_utility"])), reverse=True)[:8]
    return tuple(
        {
            "proposal_id": row["proposal_id"],
            "backend_id": row["backend_id"],
            "objective_id": row["objective_id"],
            "trajectory_id": row["trajectory_id"],
            "total_utility": row["total_utility"],
            "geometry_score": row["geometry_score"],
            "feature_excitation": row["feature_excitation"],
        }
        for row in frontier
    )


def run_trajectory_exploration_backend(
    backend: TrajectoryExplorationBackend,
    objective: TrajectoryExplorationObjective,
    *,
    seed: int = 7,
    batch_size: int = 6,
) -> TrajectoryExplorationResult:
    environment = CorpusGymEnvironment(policy=load_corpus_policy_spec())
    backend.initialize(objective, seed)
    evaluation_budget = objective.evaluation_budget
    all_rows: list[dict[str, object]] = []
    evaluations_total: list[TrajectoryExplorationEvaluation] = []
    remaining = evaluation_budget
    while remaining > 0:
        proposals = backend.propose_batch(min(batch_size, remaining))
        batch_evaluations: list[TrajectoryExplorationEvaluation] = []
        for proposal in proposals:
            environment.reset(objective.target)
            episode = environment.simulate(proposal.action)
            evaluation = evaluate_proposal(objective, proposal, episode)
            batch_evaluations.append(evaluation)
            all_rows.append(evaluation.as_row())
        backend.observe(tuple(batch_evaluations))
        evaluations_total.extend(batch_evaluations)
        remaining -= len(proposals)
    selected_rows = _select_rows(all_rows)
    candidate_rows = tuple(sorted(selected_rows, key=lambda row: (int(row["iteration"]), int(row["candidate_index"]))))
    selected_only = tuple(row for row in candidate_rows if bool(row["selected"]))
    summary = {
        "backend_id": backend.backend_id,
        "objective_id": objective.objective_id,
        "evaluation_budget": objective.evaluation_budget,
        "candidate_count": len(candidate_rows),
        "selected_count": len(selected_only),
        "best_total_utility": max(float(row["total_utility"]) for row in candidate_rows),
        "mean_total_utility": mean(float(row["total_utility"]) for row in candidate_rows),
        "mean_geometry_score": mean(float(row["geometry_score"]) for row in candidate_rows),
        **backend.state_summary(),
        **backend.diagnostics(),
    }
    report_markdown = "\n".join(
        [
            f"# Trajectory Exploration Backend: {backend.backend_id}",
            "",
            f"- objective: `{objective.objective_id}`",
            f"- budget: `{objective.evaluation_budget}`",
            f"- best total utility: `{summary['best_total_utility']:.3f}`",
            f"- mean total utility: `{summary['mean_total_utility']:.3f}`",
            f"- mean geometry score: `{summary['mean_geometry_score']:.3f}`",
        ]
    )
    return TrajectoryExplorationResult(
        backend_id=backend.backend_id,
        objective=objective,
        candidate_rows=candidate_rows,
        selected_rows=selected_only,
        coverage_rows=_coverage_rows(list(candidate_rows)),
        frontier_rows=_frontier_rows(list(candidate_rows)),
        objective_summary=summary,
        report_markdown=report_markdown,
    )


def _aggregate_backend_metrics(result: TrajectoryExplorationResult) -> dict[str, object]:
    rows = result.selected_rows or result.candidate_rows
    return {
        "backend_id": result.backend_id,
        "objective_id": result.objective.objective_id,
        "selected_count": len(rows),
        "mean_total_utility": mean(float(row["total_utility"]) for row in rows),
        "best_total_utility": max(float(row["total_utility"]) for row in rows),
        "mean_coverage_gain": mean(float(row["coverage_gain"]) for row in rows),
        "mean_feature_excitation": mean(float(row["feature_excitation"]) for row in rows),
        "mean_overlap_reduction": mean(float(row["class_pair_overlap_reduction"]) for row in rows),
        "mean_failure_witness_score": mean(float(row["confusion_witness_score"]) for row in rows),
        "budget_efficiency": max(float(row["total_utility"]) for row in rows) / max(result.objective.evaluation_budget, 1),
    }


def _status_rows(metrics_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    statuses: list[dict[str, object]] = []
    by_backend: dict[str, list[dict[str, object]]] = {}
    for row in metrics_rows:
        by_backend.setdefault(str(row["backend_id"]), []).append(row)
    heuristic_mean = mean(float(row["mean_total_utility"]) for row in by_backend["heuristic_search"])
    blackbox_mean = mean(float(row["mean_total_utility"]) for row in by_backend["blackbox_optimizer"])
    rl_mean = mean(float(row["mean_total_utility"]) for row in by_backend["rl_policy"])
    statuses.append({"backend_id": "heuristic_search", "status": "promote" if heuristic_mean >= 0.20 else "experimental", "justification": "Baseline search clears the minimum utility floor."})
    statuses.append({"backend_id": "blackbox_optimizer", "status": "promote" if blackbox_mean > heuristic_mean + 0.02 else "experimental", "justification": "Promote when fixed-budget utility materially exceeds heuristic search."})
    statuses.append({"backend_id": "rl_policy", "status": "experimental" if rl_mean >= heuristic_mean else "no-go", "justification": "RL remains experimental unless it materially beats heuristic and black-box search at matched budget."})
    return statuses


def analyze_trajectory_exploration_benchmarks(*, seed: int = 7) -> TrajectoryExplorationBenchmarkResult:
    objectives = default_trajectory_exploration_objectives()
    results: list[TrajectoryExplorationResult] = []
    for objective_index, objective in enumerate(objectives):
        results.append(run_trajectory_exploration_backend(HeuristicSearchBackend(), objective, seed=seed + objective_index * 10))
        results.append(run_trajectory_exploration_backend(BlackBoxOptimizerBackend(), objective, seed=seed + objective_index * 10))
        results.append(run_trajectory_exploration_backend(StatelessRlPolicyBackend(), objective, seed=seed + objective_index * 10))
    metrics_rows = [_aggregate_backend_metrics(result) for result in results]
    coverage_gain_rows = [{"backend_id": row["backend_id"], "objective_id": row["objective_id"], "coverage_gain": row["mean_coverage_gain"]} for row in metrics_rows]
    excitation_gain_rows = [{"backend_id": row["backend_id"], "objective_id": row["objective_id"], "feature_excitation_gain": row["mean_feature_excitation"]} for row in metrics_rows]
    overlap_reduction_rows = [{"backend_id": row["backend_id"], "objective_id": row["objective_id"], "overlap_reduction": row["mean_overlap_reduction"]} for row in metrics_rows]
    failure_witness_gain_rows = [{"backend_id": row["backend_id"], "objective_id": row["objective_id"], "failure_witness_gain": row["mean_failure_witness_score"]} for row in metrics_rows]
    budget_efficiency_rows = [{"backend_id": row["backend_id"], "objective_id": row["objective_id"], "budget_efficiency": row["budget_efficiency"]} for row in metrics_rows]
    backend_status_rows = _status_rows(metrics_rows)
    rl_vs_blackbox_rows = []
    for objective in objectives:
        rl_row = next(row for row in metrics_rows if row["objective_id"] == objective.objective_id and row["backend_id"] == "rl_policy")
        bb_row = next(row for row in metrics_rows if row["objective_id"] == objective.objective_id and row["backend_id"] == "blackbox_optimizer")
        rl_vs_blackbox_rows.append(
            {
                "objective_id": objective.objective_id,
                "rl_mean_total_utility": rl_row["mean_total_utility"],
                "blackbox_mean_total_utility": bb_row["mean_total_utility"],
                "rl_minus_blackbox": float(rl_row["mean_total_utility"]) - float(bb_row["mean_total_utility"]),
            }
        )
    comparison_report_markdown = "\n".join(
        [
            "# Trajectory Exploration Comparison Report",
            "",
            "This bundle compares heuristic, black-box, and RL-shaped parameter-proposal backends using one shared exploration contract.",
            "",
            "## Backend Status",
            *[
                f"- `{row['backend_id']}`: `{row['status']}` — {row['justification']}"
                for row in backend_status_rows
            ],
        ]
    )
    rl_decision = analyze_rl_backend_decision()
    rl_decision_report_markdown = "\n".join(
        [
            "# Unified RL Decision Report",
            "",
            f"- repo-level RL justified now: `{rl_decision.rl_justified}`",
            "- unified exploration comparison keeps RL experimental unless it exceeds both heuristic and black-box baselines at matched budget.",
            f"- current legacy gate success metric: {rl_decision.success_metric}",
        ]
    )
    evaluation_rows = tuple(row for result in results for row in result.candidate_rows)
    contract_payload = {
        "backend_methods": ["initialize", "propose_batch", "observe", "state_summary", "diagnostics"],
        "shared_candidate_schema": union_fieldnames(evaluation_rows),
        "objective_count": len(objectives),
        "backend_ids": ["heuristic_search", "blackbox_optimizer", "rl_policy"],
    }
    return TrajectoryExplorationBenchmarkResult(
        contract_payload=contract_payload,
        objective_rows=tuple(objective_as_row(objective) for objective in objectives),
        evaluation_rows=evaluation_rows,
        metrics_rows=tuple(metrics_rows),
        coverage_gain_rows=tuple(coverage_gain_rows),
        excitation_gain_rows=tuple(excitation_gain_rows),
        overlap_reduction_rows=tuple(overlap_reduction_rows),
        failure_witness_gain_rows=tuple(failure_witness_gain_rows),
        budget_efficiency_rows=tuple(budget_efficiency_rows),
        backend_status_rows=tuple(backend_status_rows),
        rl_vs_blackbox_rows=tuple(rl_vs_blackbox_rows),
        comparison_report_markdown=comparison_report_markdown,
        rl_decision_report_markdown=rl_decision_report_markdown,
    )


def adapt_search_baseline_result(objective: TrajectoryExplorationObjective | None = None) -> TrajectoryExplorationResult:
    result = analyze_corpus_search_baseline(seed=7)
    resolved = objective or default_trajectory_exploration_objectives()[0]
    rows = []
    for row in result.candidate_score_rows:
        rows.append(
            {
                "proposal_id": str(row["candidate_id"]),
                "backend_id": "heuristic_search",
                "objective_id": resolved.objective_id,
                "iteration": 0,
                "candidate_index": len(rows),
                "target_id": row["target_id"],
                "trajectory_id": row["trajectory_id"],
                "true_class": row["generated_class"],
                "total_utility": row["total_utility"],
                "class_validity": row["class_validity"],
                "feature_excitation": row["feature_excitation"],
                "coverage_gain": row["coverage_gain"],
                "boundary_closeness": row["boundary_closeness"],
                "classifier_stress": row["classifier_stress"],
                "prior_sensitivity": row["prior_sensitivity"],
                "leakage_penalty": row["leakage_penalty"],
                "physical_invalidity_penalty": row["physical_invalidity_penalty"],
                "feature_cell_coverage_gain": row["feature_excitation"],
                "class_pair_overlap_reduction": max(0.0, float(row["class_validity"]) - float(row["boundary_closeness"]) * 0.25),
                "pairwise_auc_gain": max(0.0, 0.5 * float(row["class_validity"]) + 0.5 * float(row["feature_excitation"])),
                "pca_margin_gain": row["position_range"],
                "confusion_witness_score": row["classifier_stress"],
                "feature_dependency_stress": row["sampling_irregularity"],
                "prior_flip_witness_score": row["prior_sensitivity"],
                "geometry_score": row["boundary_closeness"],
                "selected": any(str(selected["candidate_id"]) == str(row["candidate_id"]) for selected in result.selected_candidate_rows),
                "search_method": row["search_method"],
            }
        )
    candidate_rows = tuple(rows)
    selected_rows = tuple(row for row in candidate_rows if bool(row["selected"]))
    return TrajectoryExplorationResult(
        backend_id="heuristic_search",
        objective=resolved,
        candidate_rows=candidate_rows,
        selected_rows=selected_rows,
        coverage_rows=_coverage_rows(list(candidate_rows)),
        frontier_rows=_frontier_rows(list(candidate_rows)),
        objective_summary={"selected_count": len(selected_rows), "candidate_count": len(candidate_rows)},
        report_markdown=result.report_markdown,
    )


def adapt_quality_diversity_result(objective: TrajectoryExplorationObjective | None = None) -> TrajectoryExplorationResult:
    result = analyze_quality_diversity_corpus(seed=7, iterations=18)
    resolved = objective or default_trajectory_exploration_objectives()[0]
    candidate_rows = tuple(
        {
            "proposal_id": str(row["elite_candidate_id"]),
            "backend_id": "quality_diversity",
            "objective_id": resolved.objective_id,
            "iteration": index,
            "candidate_index": index,
            "target_id": resolved.target.target_id,
            "trajectory_id": str(row["elite_candidate_id"]),
            "true_class": row["generated_class"],
            "total_utility": row["elite_total_utility"],
            "feature_excitation": result.corpus_manifest["feature_target_elite_excitation_mean"],
            "coverage_gain": result.archive_coverage_rows[min(index, len(result.archive_coverage_rows) - 1)]["coverage_fraction"],
            "boundary_closeness": 0.0,
            "class_validity": 0.45,
            "classifier_stress": 0.0,
            "prior_sensitivity": 0.0,
            "leakage_penalty": 0.0,
            "physical_invalidity_penalty": 0.0,
            "feature_cell_coverage_gain": result.corpus_manifest["best_feature_target_excitation"],
            "class_pair_overlap_reduction": 0.0,
            "pairwise_auc_gain": 0.0,
            "pca_margin_gain": 0.0,
            "confusion_witness_score": 0.0,
            "feature_dependency_stress": 0.0,
            "prior_flip_witness_score": 0.0,
            "geometry_score": result.corpus_manifest["best_feature_target_excitation"],
            "selected": True,
        }
        for index, row in enumerate(result.archive_cell_rows)
    )
    return TrajectoryExplorationResult(
        backend_id="quality_diversity",
        objective=resolved,
        candidate_rows=candidate_rows,
        selected_rows=candidate_rows,
        coverage_rows=tuple(result.archive_coverage_rows),
        frontier_rows=_frontier_rows(list(candidate_rows)),
        objective_summary=result.corpus_manifest,
        report_markdown=result.report_markdown,
    )


def adapt_adaptive_stress_result(objective: TrajectoryExplorationObjective | None = None) -> TrajectoryExplorationResult:
    result = analyze_adaptive_stress_corpus(seed=7, random_candidates_per_mode=4, guided_candidates_per_mode=6)
    resolved = objective or default_trajectory_exploration_objectives()[1]
    candidate_rows = tuple(
        {
            "proposal_id": str(row["candidate_id"]),
            "backend_id": "adaptive_stress",
            "objective_id": resolved.objective_id,
            "iteration": 0,
            "candidate_index": index,
            "target_id": str(row.get("failure_mode", resolved.target.target_id)),
            "trajectory_id": row["candidate_id"],
            "true_class": row["true_class"],
            "total_utility": row["stress_score"],
            "feature_excitation": row["feature_excitation"],
            "coverage_gain": 0.0,
            "boundary_closeness": row["boundary_closeness"],
            "class_validity": row["class_validity"],
            "classifier_stress": row["stress_score"],
            "prior_sensitivity": row.get("prior_sensitivity", 0.0),
            "leakage_penalty": row["leakage_penalty"],
            "physical_invalidity_penalty": 0.0,
            "feature_cell_coverage_gain": row["feature_excitation"],
            "class_pair_overlap_reduction": max(0.0, float(row["class_validity"]) - float(row["boundary_closeness"]) * 0.25),
            "pairwise_auc_gain": 0.0,
            "pca_margin_gain": 0.0,
            "confusion_witness_score": row["stress_score"],
            "feature_dependency_stress": row.get("sampling_irregularity", 0.0),
            "prior_flip_witness_score": row.get("prior_sensitivity", 0.0),
            "geometry_score": row["stress_score"],
            "selected": True,
            "failure_mode": row["failure_mode"],
        }
        for index, row in enumerate(result.stress_case_rows)
    )
    return TrajectoryExplorationResult(
        backend_id="adaptive_stress",
        objective=resolved,
        candidate_rows=candidate_rows,
        selected_rows=candidate_rows,
        coverage_rows=(),
        frontier_rows=_frontier_rows(list(candidate_rows)),
        objective_summary={"stress_case_count": len(candidate_rows)},
        report_markdown=result.report_markdown,
    )


def adapt_feature_gap_result(objective: TrajectoryExplorationObjective | None = None) -> TrajectoryExplorationResult:
    result = analyze_feature_gap_trajectory_explorer(seed=7, max_iterations=2)
    resolved = objective or default_trajectory_exploration_objectives()[0]
    candidate_rows = tuple(
        {
            "proposal_id": str(row["candidate_id"]),
            "backend_id": "feature_gap_explorer",
            "objective_id": resolved.objective_id,
            "iteration": 0,
            "candidate_index": index,
            "target_id": resolved.target.target_id,
            "trajectory_id": str(row["candidate_id"]),
            "true_class": "",
            "total_utility": row["overall_score"],
            "feature_excitation": row["feature_excitation_score"],
            "coverage_gain": row["boundary_coverage_score"],
            "boundary_closeness": row["boundary_coverage_score"],
            "class_validity": 1.0,
            "classifier_stress": 0.0,
            "prior_sensitivity": 0.0,
            "leakage_penalty": row["leakage_penalty"],
            "physical_invalidity_penalty": 0.0,
            "feature_cell_coverage_gain": row["feature_excitation_score"],
            "class_pair_overlap_reduction": row["boundary_coverage_score"],
            "pairwise_auc_gain": row["balance_score"],
            "pca_margin_gain": row["difficulty_diversity_score"],
            "confusion_witness_score": 0.0,
            "feature_dependency_stress": 0.0,
            "prior_flip_witness_score": 0.0,
            "geometry_score": max(float(row["feature_excitation_score"]), float(row["boundary_coverage_score"])),
            "selected": any(str(selected.spec.candidate_id) == str(row["candidate_id"]) for selected in result.selected_evaluations),
        }
        for index, row in enumerate(result.candidate_score_rows)
    )
    selected_rows = tuple(row for row in candidate_rows if bool(row["selected"]))
    return TrajectoryExplorationResult(
        backend_id="feature_gap_explorer",
        objective=resolved,
        candidate_rows=candidate_rows,
        selected_rows=selected_rows,
        coverage_rows=(),
        frontier_rows=_frontier_rows(list(candidate_rows)),
        objective_summary={"final_candidate_id": result.final_candidate_id, "stop_reason": result.stop_reason},
        report_markdown=result.report_markdown,
    )
