from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.registry.functional_surface_catalog import (
    analyze_functional_surface_catalog,
    write_functional_surface_catalog_artifacts,
)


class FunctionalSurfaceCatalogTests(unittest.TestCase):
    def test_catalog_reports_pure_surface_inventory(self) -> None:
        result = analyze_functional_surface_catalog()

        self.assertGreaterEqual(result.summary["surface_count"], 10)
        self.assertGreaterEqual(result.summary["analysis_callable_count"], 10)
        self.assertGreaterEqual(result.summary["artifact_callable_count"], 10)
        self.assertIn("Functional Surface Catalog", result.report_markdown)
        self.assertIn("analyze -> render -> write artifacts", result.report_markdown)
        self.assertTrue(any(row.surface_id == "feature_analysis" for row in result.surface_rows))
        self.assertTrue(any(row.surface_id == "pca_dimensionality_audit" for row in result.surface_rows))
        self.assertTrue(any(row.surface_id == "methodology_latex" for row in result.surface_rows))
        self.assertTrue(any(row.surface_id == "algorithm_coverage_matrix" for row in result.surface_rows))
        self.assertTrue(any(row.surface_id == "classifier_family_scorecard" for row in result.surface_rows))
        self.assertTrue(any(row.surface_id == "method_validation_os" for row in result.surface_rows))
        self.assertTrue(any(row.surface_id == "trajectory_exploration_backend_registry" for row in result.surface_rows))
        self.assertTrue(any(row.surface_id == "embedding_baseline_frontier" for row in result.surface_rows))
        self.assertTrue(any(row.surface_id == "ts2vec_backend_parity" for row in result.surface_rows))
        self.assertTrue(any(row.surface_id == "shapelet_maneuver_motif" for row in result.surface_rows))
        self.assertTrue(any(row.surface_id == "tsc_archive_baseline_frontier" for row in result.surface_rows))
        self.assertTrue(any(row.surface_id == "archive_vs_physics_witness" for row in result.surface_rows))
        self.assertTrue(any(row.surface_id == "archive_feature_headroom_witness" for row in result.surface_rows))
        self.assertTrue(any(row.surface_id == "archive_backend_diagnosis" for row in result.surface_rows))
        self.assertTrue(any(row.surface_id == "archive_family_promotion_audit" for row in result.surface_rows))
        self.assertTrue(any(row.surface_id == "drcif_interval_promotion_audit" for row in result.surface_rows))
        self.assertTrue(any(row.surface_id == "gsf_multimodal_promotion_audit" for row in result.surface_rows))
        self.assertTrue(any(row.surface_id == "ukf_nonlinear_promotion_audit" for row in result.surface_rows))
        self.assertTrue(any(row.surface_id == "imm_switching_promotion_audit" for row in result.surface_rows))
        self.assertTrue(any(row.surface_id == "tsc_archive_backend_smoke" for row in result.surface_rows))
        self.assertTrue(any(row.surface_id == "neural_sequence_vs_physics_frontier" for row in result.surface_rows))
        self.assertTrue(any(row.surface_id == "neural_sequence_robustness_frontier" for row in result.surface_rows))
        self.assertTrue(any(row.surface_id == "physics_family_promotion_audit" for row in result.surface_rows))
        self.assertTrue(any(row.surface_id == "static_feature_class_prior_audit" for row in result.surface_rows))

    def test_catalog_writes_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_functional_surface_catalog_artifacts(temp_dir)

            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "functional_surface_catalog_v1")
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.catalog_path.exists())
            self.assertTrue(artifacts.plot_path.exists())

            report = artifacts.report_path.read_text(encoding="utf-8")
            self.assertIn("Functional Surface Catalog", report)
            self.assertIn("feature_analysis", report)
            self.assertIn("pca_analysis", report)
            self.assertIn("algorithm_coverage_matrix", report)
            self.assertIn("classifier_family_scorecard", report)
            self.assertIn("trajectory_exploration_backend_registry", report)
            self.assertIn("embedding_baseline_frontier", report)
            self.assertIn("ts2vec_backend_parity", report)
            self.assertIn("shapelet_maneuver_motif", report)
            self.assertIn("tsc_archive_baseline_frontier", report)
            self.assertIn("archive_vs_physics_witness", report)
            self.assertIn("archive_feature_headroom_witness", report)
            self.assertIn("archive_backend_diagnosis", report)
            self.assertIn("archive_family_promotion_audit", report)
            self.assertIn("drcif_interval_promotion_audit", report)
            self.assertIn("gsf_multimodal_promotion_audit", report)
            self.assertIn("ukf_nonlinear_promotion_audit", report)
            self.assertIn("imm_switching_promotion_audit", report)
            self.assertIn("tsc_archive_backend_smoke", report)
            self.assertIn("neural_sequence_vs_physics_frontier", report)
            self.assertIn("neural_sequence_robustness_frontier", report)
            self.assertIn("physics_family_promotion_audit", report)
            self.assertIn("static_feature_class_prior_audit", report)

            summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
            self.assertGreaterEqual(summary["surface_count"], 10)
            self.assertGreaterEqual(summary["high_priority_count"], 5)


if __name__ == "__main__":
    unittest.main()
