from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from kinematic_classifier_sandbox.corpus.gym import CorpusGymEnvironment
from kinematic_classifier_sandbox.corpus.trajectory_exploration.artifact_io import (
    write_trajectory_exploration_artifacts,
)
from kinematic_classifier_sandbox.corpus.trajectory_exploration.backends import (
    BayesianOptimizationBackend,
    BlackBoxOptimizerBackend,
    CmaEsBackend,
    HeuristicSearchBackend,
    LatinHypercubeBackend,
    MapElitesBackend,
    StatelessRlPolicyBackend,
    _seed_action_for_objective,
)
from kinematic_classifier_sandbox.corpus.trajectory_exploration.objectives import (
    default_trajectory_exploration_objectives,
    evaluate_proposal,
)
from kinematic_classifier_sandbox.corpus.trajectory_exploration.runner import (
    adapt_adaptive_stress_result,
    adapt_feature_gap_result,
    adapt_quality_diversity_result,
    adapt_search_baseline_result,
    analyze_trajectory_exploration_benchmarks,
    run_trajectory_exploration_backend,
)
from kinematic_classifier_sandbox.corpus.trajectory_exploration.contracts import (
    TrajectoryExplorationEvaluation,
    TrajectoryExplorationProposal,
)


class TrajectoryExplorationTests(unittest.TestCase):
    def test_backend_state_summary_reflects_objective_policy(self) -> None:
        objectives = {objective.objective_id: objective for objective in default_trajectory_exploration_objectives()}
        lhs = LatinHypercubeBackend()
        lhs.initialize(objectives["feature_cell_repair"], seed=7)
        map_elites = MapElitesBackend()
        map_elites.initialize(objectives["feature_cell_repair"], seed=7)
        cmaes = CmaEsBackend()
        cmaes.initialize(objectives["class_pair_boundary_refinement"], seed=7)
        bayesopt = BayesianOptimizationBackend()
        bayesopt.initialize(objectives["prior_flip_witness_search"], seed=7)

        self.assertEqual(lhs.state_summary()["selection_policy"], "coverage_first")
        self.assertEqual(map_elites.state_summary()["selection_policy"], "coverage_first")
        self.assertEqual(cmaes.state_summary()["selection_policy"], "adaptive_continuous")
        self.assertEqual(bayesopt.state_summary()["selection_policy"], "adversarial_witness")
        self.assertTrue(bool(bayesopt.state_summary()["prefer_novelty"]))

    def test_bayesopt_emits_objective_aware_acquisition_modes(self) -> None:
        objectives = {objective.objective_id: objective for objective in default_trajectory_exploration_objectives()}
        coverage_result = run_trajectory_exploration_backend(BayesianOptimizationBackend(), objectives["feature_cell_repair"], seed=7)
        boundary_result = run_trajectory_exploration_backend(BayesianOptimizationBackend(), objectives["class_pair_boundary_refinement"], seed=7)
        witness_result = run_trajectory_exploration_backend(BayesianOptimizationBackend(), objectives["prior_flip_witness_search"], seed=7)

        self.assertTrue(any(str(row.get("acquisition_mode", "")) == "coverage_ucb" for row in coverage_result.candidate_rows))
        self.assertTrue(any(str(row.get("acquisition_mode", "")) == "boundary_ucb" for row in boundary_result.candidate_rows))
        self.assertTrue(any(str(row.get("acquisition_mode", "")) == "witness_novelty_ucb" for row in witness_result.candidate_rows))

    def test_map_elites_emits_archive_trace_and_strategy_metadata(self) -> None:
        objective = default_trajectory_exploration_objectives()[0]
        result = run_trajectory_exploration_backend(MapElitesBackend(), objective, seed=7)
        self.assertEqual(result.backend_id, "map_elites")
        self.assertTrue(result.objective_summary["map_elites_trace_rows"])
        self.assertTrue(
            any(
                str(row.get("map_strategy", "")) in {"sparse_cell_explore", "elite_crossover", "elite_mutation", "seed_mutation"}
                for row in result.candidate_rows
            )
        )

    def test_cmaes_restarts_after_stagnation(self) -> None:
        objective = default_trajectory_exploration_objectives()[1]
        backend = CmaEsBackend()
        backend.initialize(objective, seed=7)

        def _evaluation(iteration: int, utility: float) -> TrajectoryExplorationEvaluation:
            return TrajectoryExplorationEvaluation(
                proposal_id=f"p{iteration}",
                backend_id="cmaes",
                objective_id=objective.objective_id,
                iteration=iteration,
                candidate_index=0,
                target_id=objective.target.target_id,
                trajectory_id=f"traj_{iteration}",
                true_class="constant_velocity",
                total_utility=utility,
                class_validity=0.6,
                feature_excitation=0.5,
                coverage_gain=0.4,
                boundary_closeness=0.5,
                classifier_stress=0.2,
                prior_sensitivity=0.2,
                leakage_penalty=0.1,
                physical_invalidity_penalty=0.1,
                feature_cell_coverage_gain=0.4,
                class_pair_overlap_reduction=0.3,
                pairwise_auc_gain=0.3,
                pca_margin_gain=0.3,
                confusion_witness_score=0.4,
                feature_dependency_stress=0.2,
                prior_flip_witness_score=0.2,
                geometry_score=0.5,
                diagnostics={
                    "duration_scale": 1.0,
                    "measurement_scale": 1.0,
                    "irregularity_scale": 1.0,
                    "outlier_scale": 1.0,
                    "step_scale": 1.0,
                },
            )

        for iteration in range(3):
            backend._iteration = iteration + 1
            backend.observe((_evaluation(iteration, 0.5), _evaluation(iteration, 0.49)))

        trace_rows = backend.diagnostics()["cmaes_trace_rows"]
        self.assertTrue(trace_rows)
        self.assertGreaterEqual(int(trace_rows[-1]["restart_count"]), 1)
        self.assertEqual(str(trace_rows[-1]["restarted"]), "yes")

    def test_shared_backend_runs_are_deterministic(self) -> None:
        objective = default_trajectory_exploration_objectives()[0]
        first = run_trajectory_exploration_backend(HeuristicSearchBackend(), objective, seed=7)
        second = run_trajectory_exploration_backend(HeuristicSearchBackend(), objective, seed=7)

        self.assertEqual(first.objective_summary["best_total_utility"], second.objective_summary["best_total_utility"])
        self.assertEqual(first.candidate_rows, second.candidate_rows)

    def test_search_backends_emit_same_candidate_schema(self) -> None:
        objective = default_trajectory_exploration_objectives()[1]
        lhs = run_trajectory_exploration_backend(LatinHypercubeBackend(), objective, seed=7)
        map_elites = run_trajectory_exploration_backend(MapElitesBackend(), objective, seed=7)
        blackbox = run_trajectory_exploration_backend(BlackBoxOptimizerBackend(), objective, seed=7)
        cmaes = run_trajectory_exploration_backend(CmaEsBackend(), objective, seed=7)
        bayesopt = run_trajectory_exploration_backend(BayesianOptimizationBackend(), objective, seed=7)
        rl = run_trajectory_exploration_backend(StatelessRlPolicyBackend(), objective, seed=7)

        self.assertEqual(set(lhs.candidate_rows[0].keys()), set(blackbox.candidate_rows[0].keys()))
        self.assertEqual(set(map_elites.candidate_rows[0].keys()), set(blackbox.candidate_rows[0].keys()))
        self.assertEqual(set(blackbox.candidate_rows[0].keys()), set(cmaes.candidate_rows[0].keys()))
        self.assertEqual(set(blackbox.candidate_rows[0].keys()), set(bayesopt.candidate_rows[0].keys()))
        self.assertEqual(set(blackbox.candidate_rows[0].keys()), set(rl.candidate_rows[0].keys()))
        self.assertEqual(len(lhs.candidate_rows), objective.evaluation_budget)
        self.assertEqual(len(map_elites.candidate_rows), objective.evaluation_budget)
        self.assertEqual(len(blackbox.candidate_rows), objective.evaluation_budget)
        self.assertEqual(len(cmaes.candidate_rows), objective.evaluation_budget)
        self.assertEqual(len(bayesopt.candidate_rows), objective.evaluation_budget)
        self.assertEqual(len(rl.candidate_rows), objective.evaluation_budget)

    def test_blackbox_beats_heuristic_on_at_least_one_objective(self) -> None:
        benchmark = analyze_trajectory_exploration_benchmarks(seed=7)
        deltas = []
        for objective in default_trajectory_exploration_objectives():
            heuristic = next(
                row for row in benchmark.metrics_rows if row["objective_id"] == objective.objective_id and row["backend_id"] == "heuristic_search"
            )
            blackbox = next(
                row for row in benchmark.metrics_rows if row["objective_id"] == objective.objective_id and row["backend_id"] == "blackbox_optimizer"
            )
            deltas.append(float(blackbox["best_total_utility"]) - float(heuristic["best_total_utility"]))
        self.assertTrue(any(delta > 0.0 for delta in deltas))

    def test_new_search_backends_are_included_in_benchmark_contract(self) -> None:
        benchmark = analyze_trajectory_exploration_benchmarks(seed=7)
        backend_ids = set(benchmark.contract_payload["backend_ids"])
        metric_backend_ids = {str(row["backend_id"]) for row in benchmark.metrics_rows}
        status_backend_ids = {str(row["backend_id"]) for row in benchmark.backend_status_rows}
        recommendation_objective_ids = {str(row["objective_id"]) for row in benchmark.backend_recommendation_rows}
        recommendation_rows = {str(row["objective_id"]): row for row in benchmark.backend_recommendation_rows}
        self.assertIn("latin_hypercube", backend_ids)
        self.assertIn("map_elites", backend_ids)
        self.assertIn("cmaes", backend_ids)
        self.assertIn("bayesian_optimization", backend_ids)
        self.assertIn("latin_hypercube", metric_backend_ids)
        self.assertIn("map_elites", metric_backend_ids)
        self.assertIn("cmaes", metric_backend_ids)
        self.assertIn("bayesian_optimization", metric_backend_ids)
        self.assertIn("latin_hypercube", status_backend_ids)
        self.assertIn("map_elites", status_backend_ids)
        self.assertIn("cmaes", status_backend_ids)
        self.assertIn("bayesian_optimization", status_backend_ids)
        self.assertEqual(
            recommendation_objective_ids,
            {objective.objective_id for objective in default_trajectory_exploration_objectives()},
        )
        self.assertEqual(
            recommendation_rows["feature_cell_repair"]["selection_policy"],
            "coverage_first",
        )
        self.assertIn(
            recommendation_rows["feature_cell_repair"]["recommended_backend"],
            {"map_elites", "latin_hypercube", "heuristic_search"},
        )
        self.assertEqual(
            recommendation_rows["class_pair_boundary_refinement"]["selection_policy"],
            "adaptive_continuous",
        )
        self.assertIn(
            recommendation_rows["class_pair_boundary_refinement"]["recommended_backend"],
            {"blackbox_optimizer", "cmaes", "bayesian_optimization"},
        )
        self.assertEqual(
            recommendation_rows["prior_flip_witness_search"]["selection_policy"],
            "adversarial_witness",
        )
        self.assertIn(
            recommendation_rows["prior_flip_witness_search"]["recommended_backend"],
            {"blackbox_optimizer", "cmaes", "bayesian_optimization"},
        )

    def test_existing_backends_adapt_into_shared_contract(self) -> None:
        search = adapt_search_baseline_result()
        qd = adapt_quality_diversity_result()
        stress = adapt_adaptive_stress_result()
        feature_gap = adapt_feature_gap_result()

        self.assertGreater(len(search.candidate_rows), 0)
        self.assertGreater(len(search.selected_rows), 0)
        self.assertGreater(len(qd.coverage_rows), 0)
        self.assertGreater(len(stress.selected_rows), 0)
        self.assertGreater(len(feature_gap.selected_rows), 0)

    def test_leakage_penalty_changes_total_utility(self) -> None:
        base = default_trajectory_exploration_objectives()[0]
        heavy_penalty = replace(
            base,
            reward_weights={**base.reward_weights, "leakage_penalty": 0.80, "feature_excitation": 0.15},
        )
        environment = CorpusGymEnvironment()
        action = _seed_action_for_objective(__import__("random").Random(7), base, 7)
        proposal = TrajectoryExplorationProposal(
            proposal_id="p0",
            backend_id="manual",
            iteration=0,
            candidate_index=0,
            action=action,
        )
        environment.reset(base.target)
        episode = environment.simulate(action)
        baseline_eval = evaluate_proposal(base, proposal, episode)
        penalized_eval = evaluate_proposal(heavy_penalty, proposal, episode)

        self.assertLessEqual(penalized_eval.total_utility, baseline_eval.total_utility)

    def test_artifacts_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_trajectory_exploration_artifacts(temp_dir, seed=7)
            self.assertEqual(artifacts.contract_dir, Path(temp_dir) / "trajectory_exploration_contract")
            self.assertTrue(artifacts.backend_contract_path.exists())
            self.assertTrue(artifacts.objective_schema_path.exists())
            self.assertTrue(artifacts.evaluation_schema_path.exists())
            self.assertTrue(artifacts.comparison_report_path.exists())
            self.assertTrue(artifacts.metrics_by_backend_path.exists())
            self.assertTrue(artifacts.coverage_gain_by_backend_path.exists())
            self.assertTrue(artifacts.excitation_gain_by_backend_path.exists())
            self.assertTrue(artifacts.overlap_reduction_by_backend_path.exists())
            self.assertTrue(artifacts.failure_witness_gain_by_backend_path.exists())
            self.assertTrue(artifacts.budget_efficiency_path.exists())
            self.assertTrue(artifacts.rl_decision_report_path.exists())
            self.assertTrue(artifacts.rl_vs_blackbox_path.exists())
            self.assertTrue(artifacts.optimizer_trace_path.exists())
            self.assertTrue(artifacts.elite_frontier_path.exists())
            self.assertTrue(artifacts.objective_progress_path.exists())
            self.assertTrue(artifacts.search_backend_comparison_path.exists())
            self.assertTrue(artifacts.search_backend_trace_path.exists())
            self.assertTrue(artifacts.search_backend_progress_path.exists())
            self.assertTrue(artifacts.backend_recommendation_path.exists())
            self.assertTrue(artifacts.bayesopt_trace_path.exists())
            self.assertTrue(artifacts.bayesopt_report_path.exists())
            self.assertTrue(artifacts.map_elites_trace_path.exists())
            self.assertTrue(artifacts.map_elites_report_path.exists())


if __name__ == "__main__":
    unittest.main()
