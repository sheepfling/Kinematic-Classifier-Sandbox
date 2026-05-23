from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox import (
    analyze_technique_comparison,
    render_technique_comparison_report,
    write_technique_comparison_artifacts,
)


class TechniqueComparisonTests(unittest.TestCase):
    def test_technique_comparison_artifacts_are_generated(self) -> None:
        result = analyze_technique_comparison(seed=7)
        report = render_technique_comparison_report(result)

        self.assertIn("Technique Comparison Study", report)
        method_names = [row.method_name for row in result.rows]
        self.assertEqual(
            method_names,
            ["pointwise", "windowed_raw", "windowed_robust", "accumulator", "kalman_bank"],
        )
        self.assertTrue(any(row.boundary_accuracy is not None for row in result.rows))
        self.assertTrue(any(row.irregular_dt_accuracy is not None for row in result.rows))

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_technique_comparison_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "technique_comparison_v1")
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.summary_csv_path.exists())
            self.assertTrue(artifacts.scenario_csv_path.exists())
            self.assertTrue(artifacts.capability_csv_path.exists())
            self.assertTrue(artifacts.metric_heatmap_svg_path.exists())
            self.assertTrue(artifacts.metric_heatmap_png_path.exists())
            self.assertTrue(artifacts.scatter_svg_path.exists())
            self.assertTrue(artifacts.scatter_png_path.exists())
            self.assertTrue(artifacts.capability_svg_path.exists())
            self.assertTrue(artifacts.capability_png_path.exists())


if __name__ == "__main__":
    unittest.main()
