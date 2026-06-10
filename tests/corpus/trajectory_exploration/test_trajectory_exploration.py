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
    BlackBoxOptimizerBackend,
    HeuristicSearchBackend,
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
    TrajectoryExplorationProposal,
)


class TrajectoryExplorationTests(unittest.TestCase):
    def test_shared_backend_runs_are_deterministic(self) -> None:
        objective = default_trajectory_exploration_objectives()[0]
        first = run_trajectory_exploration_backend(HeuristicSearchBackend(), objective, seed=7)
        second = run_trajectory_exploration_backend(HeuristicSearchBackend(), objective, seed=7)

        self.assertEqual(first.objective_summary["best_total_utility"], second.objective_summary["best_total_utility"])
        self.assertEqual(first.candidate_rows, second.candidate_rows)

    def test_blackbox_and_rl_emit_same_candidate_schema(self) -> None:
        objective = default_trajectory_exploration_objectives()[1]
        blackbox = run_trajectory_exploration_backend(BlackBoxOptimizerBackend(), objective, seed=7)
        rl = run_trajectory_exploration_backend(StatelessRlPolicyBackend(), objective, seed=7)

        self.assertEqual(set(blackbox.candidate_rows[0].keys()), set(rl.candidate_rows[0].keys()))
        self.assertEqual(len(blackbox.candidate_rows), objective.evaluation_budget)
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


if __name__ == "__main__":
    unittest.main()
