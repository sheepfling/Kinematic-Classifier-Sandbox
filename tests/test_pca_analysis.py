from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox import (
    analyze_feature_pca,
    write_pca_analysis_artifacts,
)


class PcaAnalysisTests(unittest.TestCase):
    def test_pca_analysis_produces_components_and_artifacts(self) -> None:
        result = analyze_feature_pca(seed=7, trajectories_per_class=5, n_components=3)

        self.assertEqual(len(result.components), 3)
        self.assertGreater(result.components[0].explained_variance, 0.0)
        self.assertGreaterEqual(
            result.components[0].explained_variance_ratio,
            result.components[1].explained_variance_ratio,
        )
        self.assertGreaterEqual(
            result.components[1].explained_variance_ratio,
            result.components[2].explained_variance_ratio,
        )
        self.assertEqual(len(result.coordinates), len(result.feature_analysis.feature_rows))
        self.assertIn("pc1", result.coordinates[0])
        self.assertIn("pc2", result.coordinates[0])

        total_ratio = sum(component.explained_variance_ratio for component in result.components)
        self.assertGreater(total_ratio, 0.5)

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_pca_analysis_artifacts(temp_dir, seed=7, trajectories_per_class=5, n_components=3)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "pca_analysis_v1")
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.coordinates_path.exists())
            self.assertTrue(artifacts.loadings_path.exists())
            self.assertTrue(artifacts.explained_variance_path.exists())
            self.assertTrue(artifacts.config_path.exists())
            self.assertTrue(artifacts.plot_scatter_png_path.exists())
            self.assertTrue(artifacts.plot_variance_png_path.exists())
            self.assertTrue(artifacts.plot_loadings_png_path.exists())
            report = artifacts.report_path.read_text(encoding="utf-8")
            self.assertIn("Explained Variance", report)
            self.assertIn("Dominant Loadings", report)

    def test_pca_analysis_respects_feature_subset(self) -> None:
        result = analyze_feature_pca(
            seed=7,
            trajectories_per_class=5,
            n_components=2,
            feature_set="model_residuals",
        )

        self.assertEqual(result.feature_set_name, "model_residuals")
        self.assertEqual(
            result.feature_names,
            (
                "acceleration_variance",
                "linear_fit_residual",
                "quadratic_fit_residual",
                "outlier_score",
            ),
        )
        self.assertEqual(len(result.components), 2)
        self.assertTrue(all(set(component.loadings) == set(result.feature_names) for component in result.components))

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_pca_analysis_artifacts(
                temp_dir,
                seed=7,
                trajectories_per_class=5,
                n_components=2,
                feature_set="model_residuals",
            )
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "pca_analysis_model_residuals_v1")
            report = artifacts.report_path.read_text(encoding="utf-8")
            self.assertIn("Feature set: model_residuals", report)


if __name__ == "__main__":
    unittest.main()
