from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.analysis.generated_corpus_features import (
    analyze_generated_corpus_features,
    write_generated_corpus_feature_artifacts,
)


class GeneratedCorpusFeaturesTests(unittest.TestCase):
    def test_analysis_generates_real_feature_rows(self) -> None:
        result = analyze_generated_corpus_features()
        self.assertGreater(len(result.feature_rows), 0)
        self.assertIn("acceleration_range", result.feature_rows[0])
        self.assertIn("routes objective-driven selected trajectories through the real feature pipeline", result.report_markdown)
        self.assertTrue(all(row["backend_id"] == "corpus_gym" for row in result.record_rows))

    def test_artifacts_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_generated_corpus_feature_artifacts(temp_dir)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "generated_corpus_features")
            self.assertTrue(artifacts.feature_matrix_path.exists())
            self.assertTrue(artifacts.feature_manifest_path.exists())
            self.assertTrue(artifacts.excitation_scores_path.exists())
            self.assertTrue(artifacts.record_manifest_path.exists())
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.excitation_heatmap_path.exists())
            self.assertTrue(artifacts.coverage_plot_path.exists())
            self.assertTrue(artifacts.gallery_plot_path.exists())

            payload = json.loads(artifacts.feature_manifest_path.read_text(encoding="utf-8"))
            self.assertIn("feature_names", payload)
            self.assertGreater(len(payload["feature_names"]), 0)


if __name__ == "__main__":
    unittest.main()
