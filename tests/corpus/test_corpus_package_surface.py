from __future__ import annotations

import unittest

from kinematic_classifier_sandbox import corpus


class CorpusPackageSurfaceTests(unittest.TestCase):
    def test_curated_surface_exposes_core_entrypoints(self) -> None:
        self.assertTrue(callable(corpus.analyze_corpus_adequacy))
        self.assertTrue(callable(corpus.write_corpus_adequacy_artifacts))
        self.assertTrue(callable(corpus.analyze_coverage_report))
        self.assertTrue(callable(corpus.write_coverage_report_artifacts))
        self.assertTrue(callable(corpus.analyze_corpus_search_baseline))
        self.assertTrue(callable(corpus.write_corpus_search_baseline_artifacts))
        self.assertTrue(callable(corpus.analyze_corpus_synthesis_comparison))
        self.assertTrue(callable(corpus.write_corpus_synthesis_comparison_artifacts))
        self.assertTrue(callable(corpus.analyze_generated_trajectory_objective_suite))
        self.assertTrue(callable(corpus.write_generated_trajectory_objective_artifacts))
        self.assertTrue(callable(corpus.analyze_selected_generated_corpus))
        self.assertTrue(callable(corpus.write_selected_generated_corpus_artifacts))
        self.assertTrue(callable(corpus.analyze_rl_backend_decision))
        self.assertTrue(callable(corpus.analyze_sequential_ppo_boundary_control))
        self.assertTrue(callable(corpus.write_rl_backend_decision_artifacts))
        self.assertTrue(callable(corpus.write_sequential_ppo_boundary_control_artifacts))
        self.assertTrue(callable(corpus.analyze_trajectory_exploration_benchmarks))
        self.assertTrue(callable(corpus.write_trajectory_exploration_artifacts))

    def test_curated_surface_exposes_core_contracts(self) -> None:
        self.assertEqual(corpus.CorpusAdequacyResult.__name__, "CorpusAdequacyResult")
        self.assertEqual(corpus.CoverageReportResult.__name__, "CoverageReportResult")
        self.assertEqual(corpus.CorpusSearchBaselineResult.__name__, "CorpusSearchBaselineResult")
        self.assertEqual(corpus.CorpusSynthesisComparisonResult.__name__, "CorpusSynthesisComparisonResult")
        self.assertEqual(corpus.GeneratedTrajectoryObjectiveSuite.__name__, "GeneratedTrajectoryObjectiveSuite")
        self.assertEqual(corpus.SelectedGeneratedCorpusResult.__name__, "SelectedGeneratedCorpusResult")
        self.assertEqual(corpus.RlBackendDecisionResult.__name__, "RlBackendDecisionResult")
        self.assertEqual(corpus.SequentialPpoResult.__name__, "SequentialPpoResult")
        self.assertEqual(corpus.TrajectoryExplorationResult.__name__, "TrajectoryExplorationResult")
        self.assertEqual(corpus.TrajectoryExplorationObjective.__name__, "TrajectoryExplorationObjective")


if __name__ == "__main__":
    unittest.main()
