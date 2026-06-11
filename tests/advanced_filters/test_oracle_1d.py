from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.advanced_filters.oracle_1d import (
    analyze_linear_gaussian_negative_control,
    linear_gaussian_negative_control_surface,
    write_linear_gaussian_negative_control_artifacts,
)
from kinematic_classifier_sandbox.witnesses.surface import WitnessSurface


class Oracle1DTests(unittest.TestCase):
    def test_linear_gaussian_negative_control_matches_kalman(self) -> None:
        result = analyze_linear_gaussian_negative_control(seed=101)
        self.assertEqual(result.metrics["promotion_decision"], "do_not_escalate_beyond_kalman")
        self.assertLess(float(result.metrics["mean_oracle_to_kalman_kl"]), 0.02)
        self.assertLess(abs(float(result.metrics["oracle_rmse"]) - float(result.metrics["kalman_rmse"])), 0.02)
        self.assertGreaterEqual(float(result.metrics["kalman_95_coverage"]), 0.80)
        self.assertEqual(len(result.truth_rows), len(result.measurement_rows))
        self.assertEqual(len(result.truth_rows), len(result.state_rows))
        self.assertGreater(len(result.oracle_posterior_rows), len(result.state_rows))

    def test_negative_control_artifacts_are_rerunnable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_linear_gaussian_negative_control_artifacts(temp_dir)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "linear_gaussian_negative_control_v1")
            self.assertTrue(artifacts.truth_path.exists())
            self.assertTrue(artifacts.measurement_path.exists())
            self.assertTrue(artifacts.grid_oracle_posterior_path.exists())
            self.assertTrue(artifacts.method_posterior_path.exists())
            self.assertTrue(artifacts.state_estimate_history_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.metrics_path.exists())
            self.assertTrue(artifacts.decision_card_path.exists())
            for plot in artifacts.plot_paths:
                self.assertTrue(plot.exists())

            with artifacts.metrics_path.open(encoding="utf-8", newline="") as handle:
                metrics_rows = list(csv.DictReader(handle))
            self.assertEqual(len(metrics_rows), 1)
            self.assertEqual(metrics_rows[0]["promotion_decision"], "do_not_escalate_beyond_kalman")
            self.assertIn("Kalman filter should match the grid oracle", artifacts.decision_card_path.read_text(encoding="utf-8"))

    def test_surface_exposes_expected_study_id(self) -> None:
        surface = linear_gaussian_negative_control_surface()
        self.assertIsInstance(surface, WitnessSurface)
        self.assertEqual(surface.study_id, "linear_gaussian_negative_control_v1")
        self.assertEqual(surface.metadata["study_kind"], "1d_oracle_negative_control")


if __name__ == "__main__":
    unittest.main()
