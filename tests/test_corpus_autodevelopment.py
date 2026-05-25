from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox import (
    analyze_corpus_autodevelopment,
    render_corpus_autodevelopment_numeric_walkthrough_markdown,
    write_corpus_autodevelopment_artifacts,
)


class CorpusAutodevelopmentTests(unittest.TestCase):
    def test_corpus_autodevelopment_generates_ranked_candidates(self) -> None:
        result = analyze_corpus_autodevelopment(seed=7)

        self.assertGreater(len(result.candidate_evaluations), 1)
        self.assertGreater(len(result.candidate_score_rows), 1)
        self.assertGreater(len(result.rejected_candidate_rows), 0)
        self.assertGreater(len(result.pareto_front_rows), 0)
        self.assertIn(result.selected_candidate_id, [row["candidate_id"] for row in result.candidate_score_rows])

        selected_score = next(
            float(row["overall_score"]) for row in result.candidate_score_rows if row["candidate_id"] == result.selected_candidate_id
        )
        rejected_scores = [float(row["overall_score"]) for row in result.rejected_candidate_rows]
        self.assertTrue(any(selected_score > value for value in rejected_scores))
        self.assertIn("Corpus Autodevelopment", result.report_markdown)
        walkthrough = render_corpus_autodevelopment_numeric_walkthrough_markdown(result)
        self.assertIn("Corpus Autodevelopment Numeric Walkthrough", walkthrough)
        self.assertIn("Difficulty-Diversity Subscore", walkthrough)
        self.assertIn("Why This Candidate Beats A Rejected One", walkthrough)

    def test_corpus_autodevelopment_writes_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_corpus_autodevelopment_artifacts(temp_dir, result=analyze_corpus_autodevelopment(seed=7))
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "corpus_autodevelopment_v1")
            self.assertTrue(artifacts.objectives_path.exists())
            self.assertTrue(artifacts.candidate_manifest_path.exists())
            self.assertTrue(artifacts.candidate_scores_path.exists())
            self.assertTrue(artifacts.selected_manifest_path.exists())
            self.assertTrue(artifacts.rejected_manifest_path.exists())
            self.assertTrue(artifacts.pareto_front_path.exists())
            self.assertTrue(artifacts.adequacy_comparison_path.exists())
            self.assertTrue(artifacts.feature_excitation_comparison_path.exists())
            self.assertTrue(artifacts.leakage_comparison_path.exists())
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.numeric_walkthrough_path.exists())
            self.assertTrue(artifacts.corpus_score_pareto_path.exists())
            self.assertTrue(artifacts.feature_excitation_heatmap_path.exists())
            self.assertTrue(artifacts.leakage_by_candidate_path.exists())
            self.assertTrue(artifacts.difficulty_distribution_by_candidate_path.exists())

            selected_payload = json.loads(artifacts.selected_manifest_path.read_text(encoding="utf-8"))
            self.assertIn("selected_candidate_id", selected_payload)
            self.assertIn("selected_score", selected_payload)
            self.assertIn("selected_adequacy_summary", selected_payload)
            walkthrough = artifacts.numeric_walkthrough_path.read_text(encoding="utf-8")
            self.assertIn("Corpus Autodevelopment Numeric Walkthrough", walkthrough)
            self.assertIn("Overall score", walkthrough)


if __name__ == "__main__":
    unittest.main()
