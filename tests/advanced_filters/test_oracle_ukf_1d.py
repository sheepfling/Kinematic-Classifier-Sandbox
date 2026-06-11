from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.advanced_filters.oracle_ukf_1d import (
    analyze_ukf_nonlinear_unimodal_witness,
    ukf_nonlinear_unimodal_witness_surface,
    write_ukf_nonlinear_unimodal_witness_artifacts,
)


class UKFOracleWitnessTests(unittest.TestCase):
    def test_ukf_nonlinear_unimodal_witness_beats_linear_proxy(self) -> None:
        result = analyze_ukf_nonlinear_unimodal_witness(seed=307)
        self.assertEqual(
            result.metrics["promotion_decision"],
            "promote_ukf_for_nonlinear_unimodal_measurement",
        )
        self.assertLess(
            float(result.metrics["mean_oracle_to_ukf_kl"]),
            float(result.metrics["mean_oracle_to_kalman_kl"]),
        )
        self.assertLess(
            float(result.metrics["ukf_rmse"]),
            float(result.metrics["kalman_rmse"]),
        )

    def test_ukf_oracle_artifacts_are_rerunnable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_ukf_nonlinear_unimodal_witness_artifacts(temp_dir)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "ukf_nonlinear_unimodal_oracle_v1")
            self.assertTrue(artifacts.truth_path.exists())
            self.assertTrue(artifacts.measurement_path.exists())
            self.assertTrue(artifacts.grid_oracle_posterior_path.exists())
            self.assertTrue(artifacts.method_posterior_path.exists())
            self.assertTrue(artifacts.kalman_baseline_posterior_path.exists())
            self.assertTrue(artifacts.state_estimate_history_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.metrics_path.exists())
            self.assertTrue(artifacts.decision_card_path.exists())
            for plot in artifacts.plot_paths:
                self.assertTrue(plot.exists())
            with artifacts.metrics_path.open(encoding="utf-8", newline="") as handle:
                metrics_rows = list(csv.DictReader(handle))
            self.assertEqual(len(metrics_rows), 1)
            self.assertEqual(
                metrics_rows[0]["promotion_decision"],
                "promote_ukf_for_nonlinear_unimodal_measurement",
            )

    def test_ukf_surface_exposes_expected_study_id(self) -> None:
        surface = ukf_nonlinear_unimodal_witness_surface()
        self.assertEqual(surface.study_id, "ukf_nonlinear_unimodal_oracle_v1")


if __name__ == "__main__":
    unittest.main()
