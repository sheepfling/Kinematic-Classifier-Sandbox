from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox import (
    analyze_environment_aware_corpus,
    write_environment_aware_corpus_artifacts,
)


class EnvironmentAwareCorpusTests(unittest.TestCase):
    def test_analysis_targets_environment_regimes_and_detects_biased_control(self) -> None:
        result = analyze_environment_aware_corpus()
        self.assertEqual(len(result.environment_manifest["environment_regimes"]), 3)
        self.assertGreaterEqual(len(result.environment_coverage_rows), 6)
        self.assertIn("biased_control_slice", {row["slice_id"] for row in result.environment_leakage_rows})

        biased_flags = [
            row for row in result.environment_leakage_rows
            if row["slice_id"] == "biased_control_slice" and bool(row["flagged_class_linkage"])
        ]
        self.assertGreaterEqual(len(biased_flags), 1)
        self.assertIn("Environment Coverage", result.report_markdown)

    def test_artifacts_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_environment_aware_corpus_artifacts(temp_dir)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "environment_aware_corpus")
            self.assertTrue(artifacts.environment_manifest_path.exists())
            self.assertTrue(artifacts.environment_coverage_path.exists())
            self.assertTrue(artifacts.environment_leakage_audit_path.exists())
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.coverage_heatmap_png_path.exists())
            self.assertTrue(artifacts.leakage_plot_png_path.exists())
            self.assertTrue(artifacts.trajectory_gallery_png_path.exists())

            manifest = json.loads(artifacts.environment_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["environment_regimes"]), 3)


if __name__ == "__main__":
    unittest.main()
