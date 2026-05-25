from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox import (
    analyze_corpus_search_baseline,
    write_corpus_search_baseline_artifacts,
)


class CorpusSearchBaselineTests(unittest.TestCase):
    def test_search_baseline_generates_and_selects_candidates(self) -> None:
        result = analyze_corpus_search_baseline(seed=7, random_candidates_per_target=6, rejection_pool_per_target=8)

        self.assertGreater(len(result.generated_candidate_rows), 0)
        self.assertGreater(len(result.candidate_score_rows), 0)
        self.assertGreater(len(result.selected_candidate_rows), 0)

        random_rows = [row for row in result.candidate_score_rows if row["search_method"] == "random"]
        selected_rows = list(result.selected_candidate_rows)
        mean_random = sum(float(row["total_utility"]) for row in random_rows) / len(random_rows)
        mean_selected = sum(float(row["total_utility"]) for row in selected_rows) / len(selected_rows)
        self.assertGreater(mean_selected, mean_random)
        self.assertTrue(any(float(row["feature_excitation_gain_vs_random"]) > 0.0 for row in selected_rows))
        self.assertGreater(len({row["generated_class"] for row in selected_rows}), 1)

    def test_search_baseline_artifacts_are_written(self) -> None:
        result = analyze_corpus_search_baseline(seed=7, random_candidates_per_target=4, rejection_pool_per_target=6)
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_corpus_search_baseline_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "corpus_search_baseline")
            self.assertTrue(artifacts.search_config_path.exists())
            self.assertTrue(artifacts.generated_candidates_path.exists())
            self.assertTrue(artifacts.candidate_scores_path.exists())
            self.assertTrue(artifacts.selected_candidates_path.exists())
            self.assertTrue(artifacts.report_path.exists())
            report_text = artifacts.report_path.read_text(encoding="utf-8")
            self.assertIn("Corpus Search Baseline", report_text)
            self.assertIn("Selected candidates beat the random-average utility", report_text)


if __name__ == "__main__":
    unittest.main()
