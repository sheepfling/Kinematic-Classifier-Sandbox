from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.corpus.adaptive_stress import (
    analyze_adaptive_stress_corpus,
    write_adaptive_stress_corpus_artifacts,
)


class AdaptiveStressCorpusTests(unittest.TestCase):
    def test_stress_search_generates_cases_for_all_failure_modes(self) -> None:
        result = analyze_adaptive_stress_corpus(seed=7, random_candidates_per_mode=4, guided_candidates_per_mode=6)

        self.assertGreater(len(result.stress_score_rows), 0)
        self.assertGreater(len(result.stress_case_rows), 0)

        failure_modes = {str(row["failure_mode"]) for row in result.stress_score_rows}
        self.assertEqual(
            failure_modes,
            {
                "wrong_classification",
                "high_entropy",
                "prior_flip",
                "raw_extrema_failure",
                "irregular_window_failure",
                "kalman_mismatch",
                "transition_delay",
            },
        )

        rows_by_mode: dict[str, list[dict[str, object]]] = {}
        for row in result.stress_score_rows:
            rows_by_mode.setdefault(str(row["failure_mode"]), []).append(row)
        improved_modes = 0
        for mode, rows in rows_by_mode.items():
            random_rows = [row for row in rows if row["search_method"] == "random"]
            guided_rows = [row for row in rows if row["search_method"] == "guided"]
            mean_random = sum(float(row["stress_score"]) for row in random_rows) / max(len(random_rows), 1)
            best_guided = max(float(row["stress_score"]) for row in guided_rows)
            if best_guided > mean_random:
                improved_modes += 1
        self.assertGreaterEqual(improved_modes, 5)
        for mode in ("high_entropy", "prior_flip", "raw_extrema_failure"):
            rows = rows_by_mode[mode]
            random_rows = [row for row in rows if row["search_method"] == "random"]
            guided_rows = [row for row in rows if row["search_method"] == "guided"]
            mean_random = sum(float(row["stress_score"]) for row in random_rows) / max(len(random_rows), 1)
            best_guided = max(float(row["stress_score"]) for row in guided_rows)
            self.assertGreater(best_guided, mean_random)

    def test_stress_artifacts_are_written(self) -> None:
        result = analyze_adaptive_stress_corpus(seed=7, random_candidates_per_mode=3, guided_candidates_per_mode=5)
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_adaptive_stress_corpus_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "adaptive_stress_corpus")
            self.assertTrue(artifacts.config_path.exists())
            self.assertTrue(artifacts.stress_cases_path.exists())
            self.assertTrue(artifacts.stress_scores_path.exists())
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.posterior_timelines_path.exists())
            self.assertTrue(artifacts.feature_traces_path.exists())
            self.assertTrue(artifacts.prior_flip_examples_path.exists())
            report_text = artifacts.report_path.read_text(encoding="utf-8")
            self.assertIn("Adaptive Stress Corpus", report_text)
            self.assertIn("wrong_classification", report_text)


if __name__ == "__main__":
    unittest.main()
