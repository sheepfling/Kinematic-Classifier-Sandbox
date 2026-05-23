from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox import (
    analyze_feature_datasets,
    write_feature_analysis_artifacts,
)


class FeatureAnalysisTests(unittest.TestCase):
    def test_feature_analysis_reports_excitation_and_separability(self) -> None:
        result = analyze_feature_datasets(seed=7, trajectories_per_class=5)

        self.assertGreater(len(result.feature_rows), 0)
        self.assertIn("position_range", result.summary.top_features)
        self.assertGreater(result.summary.excitation_totals["position_range"]["strong"], 0)

        pairwise_lookup = {
            (row["class_a"], row["class_b"]): row["pairwise_auc"]
            for row in result.pairwise_rows
        }
        self.assertLess(pairwise_lookup[("constant_acceleration", "maneuver")], pairwise_lookup[("constant_velocity", "stationary")])
        self.assertLess(min(pairwise_lookup.values()), 0.85)

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_feature_analysis_artifacts(temp_dir, seed=7, trajectories_per_class=5)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "feature_analysis_v1")
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.feature_matrix_path.exists())
            self.assertTrue(artifacts.feature_summary_path.exists())
            self.assertTrue(artifacts.feature_excitation_path.exists())
            self.assertTrue(artifacts.feature_excitation_summary_path.exists())
            self.assertTrue(artifacts.feature_separation_scores_path.exists())
            self.assertTrue(artifacts.identifiability_matrix_path.exists())
            self.assertTrue(artifacts.pairwise_distance_matrix_path.exists())
            self.assertTrue(artifacts.pairwise_overlap_matrix_path.exists())
            self.assertTrue(artifacts.pairwise_auc_matrix_path.exists())
            self.assertTrue(artifacts.plot_excitation_svg_path.exists())
            self.assertTrue(artifacts.plot_excitation_png_path.exists())
            self.assertTrue(artifacts.plot_distance_svg_path.exists())
            self.assertTrue(artifacts.plot_distance_png_path.exists())
            self.assertTrue(artifacts.plot_overlap_svg_path.exists())
            self.assertTrue(artifacts.plot_overlap_png_path.exists())
            self.assertTrue(artifacts.plot_scatter_svg_path.exists())
            self.assertTrue(artifacts.plot_scatter_png_path.exists())
            self.assertTrue(artifacts.plot_confusability_svg_path.exists())
            self.assertTrue(artifacts.plot_confusability_png_path.exists())
            report = artifacts.report_path.read_text(encoding="utf-8")
            self.assertIn("Feature Excitation", report)
            self.assertIn("Pairwise Separability", report)


if __name__ == "__main__":
    unittest.main()
