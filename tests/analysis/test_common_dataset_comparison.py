from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.analysis.common_dataset_comparison import (
    analyze_common_dataset_comparison,
    render_common_dataset_comparison_report,
)
from kinematic_classifier_sandbox.analysis.common_dataset_comparison_artifact_io import write_common_dataset_comparison_artifacts


class CommonDatasetComparisonTests(unittest.TestCase):
    def test_common_dataset_comparison_artifacts_are_generated(self) -> None:
        result = analyze_common_dataset_comparison(seed=7, trajectories_per_case=4)
        report = render_common_dataset_comparison_report(result)

        self.assertIn("Common-Dataset Technique Comparison", report)
        self.assertEqual(len(result.trajectories), 48)
        method_names = [row.method_name for row in result.rows]
        self.assertEqual(method_names[:8], ["pointwise", "windowed_raw", "windowed_robust", "rocket_proxy", "ts2vec_proxy", "accumulator", "kalman_bank", "kalman_bank_velocity_aided"])
        self.assertEqual(method_names[8:], ["particle_filter_bank", "rbpf", "ornstein_uhlenbeck_pf_v1"])
        supported_rows = [row for row in result.rows if row.applicability_status == "supported"]
        self.assertTrue(all(row.prior_flip_fraction is not None and 0.0 <= row.prior_flip_fraction <= 1.0 for row in supported_rows))
        self.assertTrue(any((row.prior_flip_fraction or 0.0) > 0.0 for row in supported_rows))
        self.assertTrue(any((row.short_accuracy or 1.0) < 1.0 for row in supported_rows))
        self.assertTrue(any((row.noisy_accuracy or 1.0) < 1.0 for row in supported_rows))
        kalman_row = next(row for row in result.rows if row.method_name == "kalman_bank")
        kalman_velocity_row = next(row for row in result.rows if row.method_name == "kalman_bank_velocity_aided")
        pointwise_row = next(row for row in result.rows if row.method_name == "pointwise")
        raw_row = next(row for row in result.rows if row.method_name == "windowed_raw")
        robust_row = next(row for row in result.rows if row.method_name == "windowed_robust")
        rocket_row = next(row for row in result.rows if row.method_name == "rocket_proxy")
        ts2vec_row = next(row for row in result.rows if row.method_name == "ts2vec_proxy")
        self.assertGreaterEqual(kalman_row.irregular_accuracy or 0.0, 0.0)
        self.assertGreaterEqual(robust_row.outlier_accuracy or 0.0, raw_row.outlier_accuracy or 0.0)
        self.assertGreater(kalman_row.endpoint_match_accuracy or 0.0, pointwise_row.endpoint_match_accuracy or 0.0)
        self.assertGreater(kalman_velocity_row.noisy_accuracy or 0.0, kalman_row.noisy_accuracy or 0.0)
        self.assertGreaterEqual(rocket_row.overall_accuracy or 0.0, raw_row.overall_accuracy or 0.0)
        self.assertGreaterEqual(ts2vec_row.overall_accuracy or 0.0, rocket_row.overall_accuracy or 0.0)
        self.assertEqual(next(row for row in result.rows if row.method_name == "particle_filter_bank").applicability_status, "witness_only")
        self.assertEqual(next(row for row in result.rows if row.method_name == "rbpf").primary_evaluation_family, "latent_maneuver_onset")
        self.assertEqual(next(row for row in result.rows if row.method_name == "ornstein_uhlenbeck_pf_v1").primary_evaluation_family, "ou_mean_reversion")

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
            method_header = artifacts.method_summary_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("applicability_status", method_header)
            self.assertIn("witness_artifact", method_header)
            sensor_metric_header = artifacts.sensor_regime_metrics_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("measurement_dims", sensor_metric_header)
            self.assertIn("coordinate_frames", sensor_metric_header)


if __name__ == "__main__":
    unittest.main()
