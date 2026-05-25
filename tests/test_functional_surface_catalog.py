from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox import (
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

            summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
            self.assertGreaterEqual(summary["surface_count"], 10)
            self.assertGreaterEqual(summary["high_priority_count"], 5)


if __name__ == "__main__":
    unittest.main()
