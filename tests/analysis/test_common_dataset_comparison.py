from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.analysis.common_dataset_comparison import (
    analyze_common_dataset_comparison,
    render_common_dataset_comparison_report,
    write_common_dataset_comparison_artifacts,
)


class CommonDatasetComparisonTests(unittest.TestCase):
    def test_common_dataset_comparison_artifacts_are_generated(self) -> None:
        result = analyze_common_dataset_comparison(seed=7, trajectories_per_case=4)
        report = render_common_dataset_comparison_report(result)

        self.assertIn("Common-Dataset Technique Comparison", report)
        self.assertEqual(len(result.trajectories), 48)
        method_names = [row.method_name for row in result.rows]
        self.assertEqual(
            method_names,
            ["pointwise", "windowed_raw", "windowed_robust", "accumulator", "kalman_bank", "kalman_bank_velocity_aided"],
        )
        self.assertTrue(all(0.0 <= row.prior_flip_fraction <= 1.0 for row in result.rows))
        self.assertTrue(any(row.prior_flip_fraction > 0.0 for row in result.rows))
        self.assertTrue(any(row.short_accuracy < 1.0 for row in result.rows))
        self.assertTrue(any(row.noisy_accuracy < 1.0 for row in result.rows))
        kalman_row = next(row for row in result.rows if row.method_name == "kalman_bank")
        kalman_velocity_row = next(row for row in result.rows if row.method_name == "kalman_bank_velocity_aided")
        pointwise_row = next(row for row in result.rows if row.method_name == "pointwise")
        raw_row = next(row for row in result.rows if row.method_name == "windowed_raw")
        robust_row = next(row for row in result.rows if row.method_name == "windowed_robust")
        self.assertGreaterEqual(kalman_row.irregular_accuracy, 0.0)
        self.assertGreaterEqual(robust_row.outlier_accuracy, raw_row.outlier_accuracy)
        self.assertGreater(kalman_row.endpoint_match_accuracy, pointwise_row.endpoint_match_accuracy)
        self.assertGreater(kalman_velocity_row.noisy_accuracy, kalman_row.noisy_accuracy)

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_common_dataset_comparison_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "common_dataset_comparison_v1")
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.trajectory_path.exists())
            self.assertTrue(artifacts.run_summary_path.exists())
            self.assertTrue(artifacts.method_summary_path.exists())
            self.assertTrue(artifacts.sensor_regimes_path.exists())
            self.assertTrue(artifacts.sensor_regime_metrics_path.exists())
            self.assertTrue(artifacts.heatmap_png_path.exists())
            self.assertTrue(artifacts.confusion_png_path.exists())
            self.assertTrue(artifacts.plots_dir.exists())
            self.assertTrue(artifacts.overview_balance_png_path.exists())
            self.assertTrue(artifacts.overview_covariates_png_path.exists())
            self.assertTrue(artifacts.scenario_profile_png_path.exists())
            self.assertTrue(artifacts.prior_sensitivity_png_path.exists())
            self.assertTrue(artifacts.trajectory_examples_png_path.exists())
            self.assertTrue(artifacts.final_confusion_png_path.exists())
            trajectory_header = artifacts.trajectory_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("measurement_dim", trajectory_header)
            self.assertIn("coordinate_frame", trajectory_header)
            run_header = artifacts.run_summary_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("measurement_dim", run_header)
            self.assertIn("coordinate_frame", run_header)
            sensor_metric_header = artifacts.sensor_regime_metrics_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("measurement_dims", sensor_metric_header)
            self.assertIn("coordinate_frames", sensor_metric_header)


if __name__ == "__main__":
    unittest.main()
