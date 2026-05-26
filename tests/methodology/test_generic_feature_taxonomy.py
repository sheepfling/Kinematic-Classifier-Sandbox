from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.methodology.feature_taxonomy import (
    analyze_generic_feature_taxonomy,
    write_generic_feature_taxonomy_artifacts,
)


class GenericFeatureTaxonomyTests(unittest.TestCase):
    def test_generic_feature_taxonomy_artifacts_are_generated(self) -> None:
        result = analyze_generic_feature_taxonomy()

        self.assertEqual(result.validation_results["overall_status"], "pass")
        self.assertTrue(result.validation_results["all_features_have_metadata"])
        self.assertTrue(result.validation_results["cumulative_features_labeled"])
        self.assertTrue(result.validation_results["all_feature_sets_resolve"])
        self.assertTrue(result.taxonomy_rows)
        self.assertTrue(result.feature_set_rows)
        self.assertTrue(result.sensitivity_rows)
        self.assertTrue(result.dependency_rows)
        self.assertIn("Generic Feature Taxonomy", result.transfer_report_markdown)

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_generic_feature_taxonomy_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "feature_taxonomy")
            self.assertTrue(artifacts.taxonomy_path.exists())
            self.assertTrue(artifacts.feature_sets_path.exists())
            self.assertTrue(artifacts.sensitivity_matrix_path.exists())
            self.assertTrue(artifacts.dependency_matrix_path.exists())
            self.assertTrue(artifacts.transfer_report_path.exists())
            self.assertTrue(artifacts.validation_results_path.exists())

            taxonomy_rows = json.loads(artifacts.taxonomy_path.read_text(encoding="utf-8"))
            self.assertTrue(any(row["name"] == "speed_range" for row in taxonomy_rows))
            speed_range = next(row for row in taxonomy_rows if row["name"] == "speed_range")
            self.assertEqual(speed_range["role"], "kinematics")
            self.assertEqual(speed_range["history_behavior"], "windowed")

            validation_results = json.loads(artifacts.validation_results_path.read_text(encoding="utf-8"))
            self.assertEqual(validation_results["overall_status"], "pass")
            self.assertIn("sampling_features", validation_results["feature_set_tag_selection_examples"])

            report_text = artifacts.transfer_report_path.read_text(encoding="utf-8")
            self.assertIn("Feature Taxonomy", report_text)
            self.assertIn("Tag Selection Examples", report_text)


if __name__ == "__main__":
    unittest.main()
