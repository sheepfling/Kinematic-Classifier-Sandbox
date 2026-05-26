from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.api import (
    analyze_corpus_synthesis_comparison,
    write_corpus_synthesis_comparison_artifacts,
)


class CorpusSynthesisComparisonTests(unittest.TestCase):
    def test_corpus_synthesis_comparison_generates_expected_method_rows(self) -> None:
        result = analyze_corpus_synthesis_comparison(seed=7)
        method_names = {str(row["method_name"]) for row in result.generator_rows}
        self.assertEqual(
            method_names,
            {
                "manual_generator",
                "random_search",
                "doe_search",
                "rejection_search",
                "quality_diversity",
                "adaptive_stress",
                "rl_backend",
            },
        )
        rl_row = next(row for row in result.generator_rows if row["method_name"] == "rl_backend")
        self.assertFalse(rl_row["rl_justified"])
        self.assertIn("Corpus Synthesis Comparison", result.report_markdown)

    def test_corpus_synthesis_comparison_artifacts_are_written(self) -> None:
        result = analyze_corpus_synthesis_comparison(seed=7)
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_corpus_synthesis_comparison_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "corpus_synthesis_comparison")
            self.assertTrue(artifacts.generator_comparison_path.exists())
            self.assertTrue(artifacts.corpus_quality_path.exists())
            self.assertTrue(artifacts.feature_excitation_path.exists())
            self.assertTrue(artifacts.classifier_stress_path.exists())
            self.assertTrue(artifacts.report_path.exists())
            report_text = artifacts.report_path.read_text(encoding="utf-8")
            self.assertIn("RL backend justified now", report_text)
            self.assertIn("manual_generator", artifacts.generator_comparison_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
