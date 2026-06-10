from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.registry.exported_surface_coverage import (
    EXPORT_SCRIPT_PATH,
    analyze_exported_surface_coverage,
    build_exported_surface_inventory,
    write_exported_surface_coverage_artifacts,
)


class ExportedSurfaceCoverageTests(unittest.TestCase):
    def test_inventory_matches_export_script_surface_list(self) -> None:
        inventory = build_exported_surface_inventory()

        self.assertGreaterEqual(len(inventory), 70)
        self.assertEqual(len({spec.surface_id for spec in inventory}), len(inventory))
        self.assertTrue(all(spec.writer_name for spec in inventory))
        self.assertTrue(all(spec.module for spec in inventory))
        self.assertEqual(EXPORT_SCRIPT_PATH.name, "export_artifacts.py")

    def test_static_audit_resolves_writers_and_flags_known_contract_gaps(self) -> None:
        result = analyze_exported_surface_coverage()
        by_surface = {row.surface_id: row for row in result.surface_rows}

        self.assertIn("feature_analysis", by_surface)
        self.assertIn("method_survey", by_surface)
        self.assertTrue(by_surface["feature_analysis"].writer_callable)
        self.assertTrue(by_surface["feature_analysis"].rerun_command_target_exists)
        self.assertIn("missing_machine_class", by_surface["method_survey"].missing_requirements)
        self.assertGreater(result.summary["surface_count"], 70)
        self.assertGreaterEqual(result.summary["writer_callable_count"], 70)

    def test_materialized_subset_observes_real_artifact_classes(self) -> None:
        result = analyze_exported_surface_coverage(
            surface_ids=("feature_analysis", "functional_surface_catalog"),
            materialize=True,
        )
        by_surface = {row.surface_id: row for row in result.surface_rows}

        self.assertEqual(set(by_surface), {"feature_analysis", "functional_surface_catalog"})
        self.assertIn("report", by_surface["feature_analysis"].observed_artifact_classes)
        self.assertIn("tabular", by_surface["feature_analysis"].observed_artifact_classes)
        self.assertIn("visual", by_surface["feature_analysis"].observed_artifact_classes)
        self.assertEqual(by_surface["functional_surface_catalog"].run_scope_name, "functional_surface_catalog_v1")

    def test_artifact_bundle_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_exported_surface_coverage_artifacts(temp_dir)

            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "exported_surface_coverage_v1")
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.coverage_matrix_path.exists())
            self.assertTrue(artifacts.missing_coverage_path.exists())
            self.assertTrue(artifacts.visualization_exemptions_path.exists())
            self.assertTrue(artifacts.rerun_commands_path.exists())
            self.assertTrue(artifacts.category_plot_path.exists())
            self.assertTrue(artifacts.inventory_path.exists())

            summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
            self.assertGreater(summary["surface_count"], 70)
            report = artifacts.report_path.read_text(encoding="utf-8")
            self.assertIn("Exported Surface Coverage Audit", report)
            self.assertIn("feature_analysis", report)


if __name__ == "__main__":
    unittest.main()
