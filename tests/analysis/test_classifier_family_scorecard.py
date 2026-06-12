from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.analysis.classifier_family_scorecard import (
    analyze_classifier_family_scorecard,
    write_classifier_family_scorecard_artifacts,
)


class ClassifierFamilyScorecardTests(unittest.TestCase):
    def test_classifier_family_scorecard_runs(self) -> None:
        result = analyze_classifier_family_scorecard()

        self.assertEqual(result.metrics["study_id"], "classifier_family_scorecard_v1")
        capability_method_ids = {row.method_id for row in result.capability_rows}
        self.assertIn("pointwise", capability_method_ids)
        self.assertIn("kalman_bank", capability_method_ids)
        self.assertIn("minirocket_family", capability_method_ids)
        self.assertIn("ts2vec", capability_method_ids)
        ceiling_rows = {row.method_id: row for row in result.ceiling_rows}
        self.assertIn("pointwise", ceiling_rows)
        self.assertIn("minirocket_family", ceiling_rows)
        self.assertEqual(ceiling_rows["minirocket_family"].ceiling_status, "no_named_ceiling_alignment")
        self.assertTrue(result.atlas_markdown.startswith("# Classifier Family Atlas"))
        self.assertIn("Classifier Family Scorecard", result.report_markdown)

    def test_classifier_family_scorecard_artifacts_are_written(self) -> None:
        result = analyze_classifier_family_scorecard()
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_classifier_family_scorecard_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "classifier_family_scorecard_v1")
            self.assertTrue(artifacts.capability_matrix_path.exists())
            self.assertTrue(artifacts.ceiling_efficiency_path.exists())
            self.assertTrue(artifacts.family_summary_path.exists())
            self.assertTrue(artifacts.atlas_path.exists())
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertEqual(len(artifacts.plot_paths), 1)
            self.assertTrue(all(path.exists() for path in artifacts.plot_paths))

            summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["study_id"], "classifier_family_scorecard_v1")
            self.assertGreater(summary["method_count"], 5)


if __name__ == "__main__":
    unittest.main()
