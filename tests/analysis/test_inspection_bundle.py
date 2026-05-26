from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.api import write_abstract_inspection_artifacts
from kinematic_classifier_sandbox.api import recommend_feature_set, recommend_hardest_class_pair


class AbstractInspectionBundleTests(unittest.TestCase):
    def test_bundle_writes_index_and_subset_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_abstract_inspection_artifacts(
                temp_dir,
                seed=7,
                trajectories_per_class=3,
                n_components=2,
                feature_sets=("all_engineered", "shape_window", "model_residuals"),
            )
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "abstract_inspection_v1")
            self.assertTrue(artifacts.index_path.exists())
            self.assertTrue(artifacts.manifest_path.exists())
            self.assertTrue(artifacts.machine_summary_path.exists())
            self.assertTrue(artifacts.summary_table_path.exists())
            self.assertTrue(artifacts.summary_chart_path.exists())
            self.assertTrue(artifacts.class_pair_summary_table_path.exists())
            self.assertTrue(artifacts.class_pair_summary_chart_path.exists())
            self.assertEqual(len(artifacts.feature_analysis_runs), 3)
            self.assertEqual(len(artifacts.pca_runs), 3)
            self.assertTrue(artifacts.corpus_adequacy.report_path.exists())
            self.assertTrue(artifacts.coverage_report.report_path.exists())

            index = artifacts.index_path.read_text(encoding="utf-8")
            self.assertIn("Abstract Inspection Bundle", index)
            self.assertIn("Feature-Set Summary", index)
            self.assertIn("feature_set_inspection_summary.csv", index)
            self.assertIn("feature_set_inspection_summary.png", index)
            self.assertIn("Min Pairwise AUC", index)
            self.assertIn("Max Overlap", index)
            self.assertIn("Hardest Class Boundaries", index)
            self.assertIn("hardest_class_pairs.csv", index)
            self.assertIn("hardest_class_pairs.png", index)
            self.assertIn("feature_analysis_shape_window_v1/feature_analysis_report.md", index)
            self.assertIn("pca_analysis_model_residuals_v1/pca_report.md", index)
            self.assertIn("corpus_adequacy_audit_v1/corpus_adequacy_report.md", index)
            self.assertIn("coverage_report_v1/coverage_report.md", index)

            summary_payload = json.loads(artifacts.machine_summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary_payload["feature_sets"], ["all_engineered", "shape_window", "model_residuals"])
            self.assertGreater(len(summary_payload["feature_set_summary"]), 0)
            self.assertGreater(len(summary_payload["hardest_class_pairs"]), 0)
            self.assertIn("corpus_adequacy", summary_payload)
            self.assertIn("coverage_report", summary_payload)

            recommended_feature_set = recommend_feature_set(summary_payload)
            hardest_class_pair = recommend_hardest_class_pair(summary_payload)
            self.assertIn("feature_set", recommended_feature_set)
            self.assertIn("feature_set_status", recommended_feature_set)
            self.assertIn("class_pair", hardest_class_pair)
            self.assertIn("pairwise_auc", hardest_class_pair)


if __name__ == "__main__":
    unittest.main()
