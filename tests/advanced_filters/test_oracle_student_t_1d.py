from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.advanced_filters.oracle_student_t_1d import (
    analyze_student_t_heavy_tail_witness,
    student_t_heavy_tail_witness_surface,
    write_student_t_heavy_tail_witness_artifacts,
)


class StudentTOracleWitnessTests(unittest.TestCase):
    def test_student_t_heavy_tail_witness_beats_gaussian_proxy(self) -> None:
        result = analyze_student_t_heavy_tail_witness(seed=409)
        self.assertEqual(
            result.metrics["promotion_decision"],
            "promote_student_t_for_heavy_tail_measurements",
        )
        self.assertLess(
            float(result.metrics["mean_oracle_to_robust_kl"]),
            float(result.metrics["mean_oracle_to_gaussian_kl"]),
        )
        self.assertLess(
            float(result.metrics["robust_rmse"]),
            float(result.metrics["gaussian_rmse"]),
        )
        self.assertGreater(
            float(result.metrics["robust_coverage_95"]),
            float(result.metrics["gaussian_coverage_95"]),
        )

    def test_student_t_oracle_artifacts_are_rerunnable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_student_t_heavy_tail_witness_artifacts(temp_dir)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "student_t_heavy_tail_oracle_v1")
            self.assertTrue(artifacts.truth_path.exists())
            self.assertTrue(artifacts.measurement_path.exists())
            self.assertTrue(artifacts.grid_oracle_posterior_path.exists())
            self.assertTrue(artifacts.robust_posterior_path.exists())
            self.assertTrue(artifacts.gaussian_posterior_path.exists())
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
                "promote_student_t_for_heavy_tail_measurements",
            )

    def test_student_t_surface_exposes_expected_study_id(self) -> None:
        surface = student_t_heavy_tail_witness_surface()
        self.assertEqual(surface.study_id, "student_t_heavy_tail_oracle_v1")


if __name__ == "__main__":
    unittest.main()
