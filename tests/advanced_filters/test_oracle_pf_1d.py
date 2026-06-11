from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.advanced_filters.oracle_pf_1d import (
    analyze_pf_abs_range_multimodal_witness,
    pf_abs_range_multimodal_witness_surface,
    write_pf_abs_range_multimodal_witness_artifacts,
)
from kinematic_classifier_sandbox.witnesses.surface import WitnessSurface


class OraclePF1DTests(unittest.TestCase):
    def test_pf_abs_range_multimodal_witness_justifies_pf(self) -> None:
        result = analyze_pf_abs_range_multimodal_witness(seed=211)
        self.assertEqual(result.metrics["promotion_decision"], "promote_pf_for_multimodal_posterior")
        self.assertLess(
            float(result.metrics["mean_oracle_to_pf_kl"]),
            float(result.metrics["mean_oracle_to_gaussian_kl"]),
        )
        self.assertLess(
            float(result.metrics["mean_pf_positive_mass_error"]),
            float(result.metrics["mean_gaussian_positive_mass_error"]),
        )
        self.assertGreater(float(result.metrics["mean_ess_fraction"]), 0.30)
        self.assertEqual(len(result.truth_rows), len(result.measurement_rows))
        self.assertEqual(len(result.truth_rows), len(result.state_rows))
        self.assertEqual(len(result.truth_rows), len(result.particle_diagnostic_rows))

    def test_pf_oracle_artifacts_are_rerunnable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_pf_abs_range_multimodal_witness_artifacts(temp_dir)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "pf_abs_range_multimodal_oracle_v1")
            self.assertTrue(artifacts.truth_path.exists())
            self.assertTrue(artifacts.measurement_path.exists())
            self.assertTrue(artifacts.grid_oracle_posterior_path.exists())
            self.assertTrue(artifacts.method_posterior_path.exists())
            self.assertTrue(artifacts.gaussian_baseline_posterior_path.exists())
            self.assertTrue(artifacts.state_estimate_history_path.exists())
            self.assertTrue(artifacts.particle_diagnostics_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.metrics_path.exists())
            self.assertTrue(artifacts.decision_card_path.exists())
            self.assertTrue(artifacts.gaussian_collapse_panel_path.exists())
            for plot in artifacts.plot_paths:
                self.assertTrue(plot.exists())

            with artifacts.metrics_path.open(encoding="utf-8", newline="") as handle:
                metrics_rows = list(csv.DictReader(handle))
            self.assertEqual(len(metrics_rows), 1)
            self.assertEqual(metrics_rows[0]["promotion_decision"], "promote_pf_for_multimodal_posterior")
            self.assertIn("non-injective", artifacts.decision_card_path.read_text(encoding="utf-8"))

    def test_pf_surface_exposes_expected_study_id(self) -> None:
        surface = pf_abs_range_multimodal_witness_surface()
        self.assertIsInstance(surface, WitnessSurface)
        self.assertEqual(surface.study_id, "pf_abs_range_multimodal_oracle_v1")
        self.assertEqual(surface.metadata["study_kind"], "1d_oracle_positive_witness")


if __name__ == "__main__":
    unittest.main()
