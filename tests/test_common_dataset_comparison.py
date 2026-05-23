from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox import (
    analyze_common_dataset_comparison,
    render_common_dataset_comparison_report,
    write_common_dataset_comparison_artifacts,
)


class CommonDatasetComparisonTests(unittest.TestCase):
    def test_common_dataset_comparison_artifacts_are_generated(self) -> None:
        result = analyze_common_dataset_comparison(seed=7, trajectories_per_case=4)
        report = render_common_dataset_comparison_report(result)

        self.assertIn("Common-Dataset Technique Comparison", report)
        self.assertEqual(len(result.trajectories), 24)
        method_names = [row.method_name for row in result.rows]
        self.assertEqual(
            method_names,
            ["pointwise", "windowed_raw", "windowed_robust", "accumulator", "kalman_bank"],
        )
        self.assertTrue(all(0.0 <= row.prior_flip_fraction <= 1.0 for row in result.rows))
        kalman_row = next(row for row in result.rows if row.method_name == "kalman_bank")
        self.assertGreaterEqual(kalman_row.irregular_accuracy, 0.0)

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_common_dataset_comparison_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "common_dataset_comparison_v1")
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.trajectory_path.exists())
            self.assertTrue(artifacts.run_summary_path.exists())
            self.assertTrue(artifacts.method_summary_path.exists())
            self.assertTrue(artifacts.heatmap_svg_path.exists())
            self.assertTrue(artifacts.heatmap_png_path.exists())
            self.assertTrue(artifacts.confusion_svg_path.exists())
            self.assertTrue(artifacts.confusion_png_path.exists())


if __name__ == "__main__":
    unittest.main()
