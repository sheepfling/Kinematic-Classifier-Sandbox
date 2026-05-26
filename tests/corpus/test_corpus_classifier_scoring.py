from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.api import *


class CorpusClassifierScoringTests(unittest.TestCase):
    def test_analysis_uses_real_classifier_methods(self) -> None:
        result = analyze_corpus_classifier_scoring()
        methods = {row["method_name"] for row in result.candidate_score_rows}
        self.assertEqual(
            methods,
            {"pointwise", "sequential_bayes", "windowed_raw", "windowed_robust", "kalman_bank"},
        )
        self.assertGreater(len(result.posterior_rows), 0)
        measured = {round(float(row["measured_classifier_stress"]), 4) for row in result.candidate_score_rows}
        heuristic = {round(float(row["heuristic_stress_reference"]), 4) for row in result.candidate_score_rows}
        self.assertNotEqual(measured, heuristic)
        self.assertIn("measured from real posterior outputs", result.report_markdown)

    def test_artifacts_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_corpus_classifier_scoring_artifacts(temp_dir)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "corpus_classifier_scoring")
            self.assertTrue(artifacts.candidate_scores_path.exists())
            self.assertTrue(artifacts.posterior_history_path.exists())
            self.assertTrue(artifacts.prior_sensitivity_path.exists())
            self.assertTrue(artifacts.disagreement_path.exists())
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.posterior_plot_path.exists())
            self.assertTrue(artifacts.disagreement_plot_path.exists())
            self.assertTrue(artifacts.stress_plot_path.exists())


if __name__ == "__main__":
    unittest.main()
