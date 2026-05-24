from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox import (
    analyze_velocity_aided_kalman_comparison,
    render_velocity_aided_kalman_comparison_report,
    write_velocity_aided_kalman_comparison_artifacts,
)


class VelocityAidedKalmanComparisonTests(unittest.TestCase):
    def test_velocity_aided_comparison_artifacts_are_generated(self) -> None:
        result = analyze_velocity_aided_kalman_comparison(seed=7, trajectories_per_case=4)
        report = render_velocity_aided_kalman_comparison_report(result)

        self.assertIn("Velocity-Aided Kalman Comparison", report)
        measurement_modes = [row.measurement_mode for row in result.rows]
        self.assertEqual(measurement_modes, ["position_only", "position_plus_direct_velocity"])
        self.assertEqual(len(result.traces), 6)
        scenario_names = sorted({trace.scenario_name for trace in result.traces})
        self.assertEqual(scenario_names, ["endpoint_match", "outlier", "short_noisy"])

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_velocity_aided_kalman_comparison_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "velocity_aided_kalman_comparison_v1")
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.summary_csv_path.exists())
            self.assertTrue(artifacts.trace_csv_path.exists())
            self.assertTrue(artifacts.heatmap_png_path.exists())
            self.assertTrue(artifacts.diagnostics_png_path.exists())


if __name__ == "__main__":
    unittest.main()
