from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.api import (
    analyze_coverage_report,
    write_coverage_report_artifacts,
)


class CoverageReportTests(unittest.TestCase):
    def test_coverage_report_summarizes_feature_and_classifier_space(self) -> None:
        result = analyze_coverage_report(seed=7, trajectories_per_class=5)

        self.assertGreater(len(result.feature_set_summary_rows), 0)
        self.assertGreater(len(result.feature_group_rows), 0)
        self.assertGreater(len(result.classifier_support_rows), 0)
        self.assertEqual(result.summary.classifier_count, len(result.classifier_support_rows))
        classifier_lookup = {
            row["classifier_id"]: row
            for row in result.classifier_support_rows
        }
        self.assertIn("pointwise", classifier_lookup)
        self.assertIn("kalman_bank", classifier_lookup)
        self.assertTrue(all(not row["ready_for_evaluation"] for row in result.classifier_support_rows))
        self.assertEqual(result.summary.overall_status, "red")

    def test_coverage_report_writes_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_coverage_report_artifacts(temp_dir, seed=7, trajectories_per_class=5)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "coverage_report_v1")
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.feature_set_summary_path.exists())
            self.assertTrue(artifacts.feature_group_summary_path.exists())
            self.assertTrue(artifacts.classifier_support_path.exists())
            report = artifacts.report_path.read_text(encoding="utf-8")
            self.assertIn("Corpus Coverage Report", report)
            self.assertIn("Classifier Support", report)


if __name__ == "__main__":
    unittest.main()
