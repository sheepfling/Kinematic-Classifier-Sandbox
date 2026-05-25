from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox import (
    analyze_generic_corpus_exploration,
    render_generic_corpus_exploration_numeric_walkthrough_markdown,
    write_generic_corpus_exploration_artifacts,
)


class GenericCorpusExplorationTests(unittest.TestCase):
    def test_selected_corpus_meets_acceptance_shape(self) -> None:
        result = analyze_generic_corpus_exploration(seed=7)

        manifest = result.selected_corpus_manifest
        self.assertTrue(result.exploration_manifest["coverage_improves_over_random"])
        self.assertGreaterEqual(manifest["selected_backend_count"], 2)
        self.assertTrue(manifest["includes_boundary_examples"])
        self.assertTrue(manifest["includes_stress_examples"])

        for row in manifest["selected_rows"]:
            self.assertIn("backend_id", row)
            self.assertIn("candidate_id", row)
            self.assertIn("provenance_completeness", row)

        walkthrough = render_generic_corpus_exploration_numeric_walkthrough_markdown(result)
        self.assertIn("Generic Corpus Explorer Numeric Walkthrough", walkthrough)
        self.assertIn("Numeric Substitution", walkthrough)
        self.assertIn("Coverage Comparison Against Random Baseline", walkthrough)

    def test_artifacts_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_generic_corpus_exploration_artifacts(temp_dir)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "generic_corpus_exploration")
            self.assertTrue(artifacts.exploration_manifest_path.exists())
            self.assertTrue(artifacts.candidate_scores_path.exists())
            self.assertTrue(artifacts.archive_cells_path.exists())
            self.assertTrue(artifacts.selected_corpus_manifest_path.exists())
            self.assertTrue(artifacts.backend_comparison_path.exists())
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.numeric_walkthrough_path.exists())
            self.assertTrue(artifacts.backend_coverage_png_path.exists())
            self.assertTrue(artifacts.archive_heatmap_png_path.exists())
            self.assertTrue(artifacts.score_parallel_png_path.exists())
            self.assertTrue(artifacts.selected_gallery_png_path.exists())
            self.assertTrue(artifacts.provenance_dashboard_png_path.exists())

            payload = json.loads(artifacts.exploration_manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["coverage_improves_over_random"])
            walkthrough = artifacts.numeric_walkthrough_path.read_text(encoding="utf-8")
            self.assertIn("Generic Corpus Explorer Numeric Walkthrough", walkthrough)
            self.assertIn("Archive Cell Interpretation", walkthrough)


if __name__ == "__main__":
    unittest.main()
