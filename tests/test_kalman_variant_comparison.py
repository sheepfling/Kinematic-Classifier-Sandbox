from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox import (
    analyze_kalman_variant_comparison,
    render_kalman_variant_comparison_report,
    write_kalman_variant_comparison_artifacts,
)


class KalmanVariantComparisonTests(unittest.TestCase):
    def test_kalman_variant_comparison_artifacts_are_generated(self) -> None:
        result = analyze_kalman_variant_comparison(seed=7, trajectories_per_case=4)
        report = render_kalman_variant_comparison_report(result)

        self.assertIn("Kalman Variant Comparison", report)
        variant_names = [row.variant_name for row in result.rows]
        self.assertEqual(
            variant_names,
            ["plain", "robust_measurement", "robust_plus_adaptive_process"],
        )
        self.assertEqual(len(result.traces), 6)
        scenario_names = sorted({trace.scenario_name for trace in result.traces})
        self.assertEqual(scenario_names, ["outlier", "short_noisy"])

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_kalman_variant_comparison_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "kalman_variant_comparison_v1")
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.summary_csv_path.exists())
            self.assertTrue(artifacts.trace_csv_path.exists())
            self.assertTrue(artifacts.heatmap_png_path.exists())
            self.assertTrue(artifacts.diagnostics_png_path.exists())
            self.assertTrue(artifacts.delta_png_path.exists())


if __name__ == "__main__":
    unittest.main()
