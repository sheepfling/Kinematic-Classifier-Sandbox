from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox import (
    analyze_pca_dimensionality,
    write_pca_dimensionality_audit_artifacts,
)


class PcaDimensionalityAuditTests(unittest.TestCase):
    def test_pca_dimensionality_audit_reports_dimension_and_clusterability(self) -> None:
        result = analyze_pca_dimensionality(seed=7, trajectories_per_class=5, max_components=4)

        self.assertEqual(result.feature_set_name, "all_engineered")
        self.assertGreaterEqual(len(result.component_rows), 1)
        self.assertGreaterEqual(result.recommendation["recommended_components_for_95pct_variance"], 1)
        self.assertGreaterEqual(result.recommendation["best_clusterability_component_count"], 1)
        self.assertLessEqual(result.recommendation["best_clusterability_component_count"], 4)

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_pca_dimensionality_audit_artifacts(
                temp_dir,
                seed=7,
                trajectories_per_class=5,
                max_components=4,
            )
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "pca_dimensionality_audit_v1")
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.component_sweep_path.exists())
            self.assertTrue(artifacts.clusterability_path.exists())
            self.assertTrue(artifacts.plot_variance_path.exists())
            self.assertTrue(artifacts.plot_clusterability_path.exists())
            self.assertTrue(artifacts.plot_separation_path.exists())

            report = artifacts.report_path.read_text(encoding="utf-8")
            self.assertIn("PCA Dimensionality Audit", report)
            self.assertIn("95% variance", report)
            self.assertIn("Clusterability", report)
            self.assertIn("k-means", report)

            summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["feature_set_name"], "all_engineered")
            self.assertIn("recommended_components_for_95pct_variance", summary["recommendation"])

    def test_pca_dimensionality_audit_respects_feature_subset(self) -> None:
        result = analyze_pca_dimensionality(
            seed=7,
            trajectories_per_class=5,
            max_components=3,
            feature_set="model_residuals",
        )

        self.assertEqual(result.feature_set_name, "model_residuals")
        self.assertGreaterEqual(len(result.feature_names), 1)
        self.assertLessEqual(len(result.component_rows), 3)

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_pca_dimensionality_audit_artifacts(
                temp_dir,
                seed=7,
                trajectories_per_class=5,
                feature_set="model_residuals",
                max_components=3,
            )
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "pca_dimensionality_audit_model_residuals_v1")
            report = artifacts.report_path.read_text(encoding="utf-8")
            self.assertIn("Feature set: model_residuals", report)


if __name__ == "__main__":
    unittest.main()
