from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.analysis.common_dataset_comparison import default_shared_classifier_adapters
from kinematic_classifier_sandbox.validation.technique_comparison import (
    analyze_technique_comparison,
    default_technique_definitions,
    render_technique_comparison_report,
    write_technique_comparison_artifacts,
)


class TechniqueComparisonTests(unittest.TestCase):
    def test_technique_comparison_artifacts_are_generated(self) -> None:
        result = analyze_technique_comparison(seed=7)
        report = render_technique_comparison_report(result)

        self.assertIn("Technique Comparison Study", report)
        method_names = [row.method_name for row in result.rows]
        sensor_regimes = [row.sensor_regime_id for row in result.rows]
        self.assertEqual(method_names[:6], ["pointwise", "windowed_raw", "windowed_robust", "accumulator", "kalman_bank", "kalman_bank_velocity_aided"])
        self.assertIn("particle_filter_bank", method_names)
        self.assertIn("rbpf", method_names)
        self.assertIn("ornstein_uhlenbeck_pf_v1", method_names)
        self.assertEqual(sensor_regimes[:6], ["position_only", "position_only", "position_only", "position_only", "position_only", "position_plus_direct_velocity"])
        self.assertTrue(any(row.boundary_accuracy is not None for row in result.rows))
        self.assertTrue(any(row.irregular_dt_accuracy is not None for row in result.rows))
        self.assertEqual(next(row for row in result.rows if row.method_name == "particle_filter_bank").applicability_status, "witness_only")
        self.assertEqual(next(row for row in result.rows if row.method_name == "rbpf").primary_evaluation_family, "latent_maneuver_onset")
        pf_drag_row = next(row for row in result.scenario_support_rows if row.method_name == "particle_filter_bank" and row.scenario_family == "nonlinear_drag_outlier")
        pf_ou_row = next(row for row in result.scenario_support_rows if row.method_name == "particle_filter_bank" and row.scenario_family == "ou_mean_reversion")
        rbpf_row = next(row for row in result.scenario_support_rows if row.method_name == "rbpf" and row.scenario_family == "latent_maneuver_onset")
        ou_row = next(row for row in result.scenario_support_rows if row.method_name == "ornstein_uhlenbeck_pf_v1" and row.scenario_family == "ou_mean_reversion")
        self.assertEqual(pf_drag_row.applicability_status, "witness_only")
        self.assertEqual(pf_drag_row.metric_name, "position_rmse")
        self.assertAlmostEqual(pf_drag_row.metric_value or 0.0, 0.05611251409383978, places=12)
        self.assertEqual(pf_ou_row.applicability_status, "witness_only")
        self.assertEqual(pf_ou_row.metric_name, "final_mean_reverting_posterior")
        self.assertAlmostEqual(pf_ou_row.metric_value or 0.0, 0.9999945192428745, places=12)
        self.assertEqual(rbpf_row.applicability_status, "witness_only")
        self.assertEqual(rbpf_row.metric_name, "post_onset_mode_accuracy")
        self.assertAlmostEqual(rbpf_row.metric_value or 0.0, 1.0, places=12)
        self.assertEqual(ou_row.applicability_status, "witness_only")
        self.assertEqual(ou_row.metric_name, "final_mean_reverting_posterior")
        self.assertAlmostEqual(ou_row.metric_value or 0.0, 0.9999945192428745, places=12)
        self.assertEqual([(definition.method_name, definition.sensor_regime_id) for definition in default_technique_definitions()], [(adapter.method_name, adapter.sensor_regime_id) for adapter in default_shared_classifier_adapters()])
        self.assertEqual([spec.method_name for spec in result.method_specs], [adapter.method_name for adapter in default_shared_classifier_adapters()])

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_technique_comparison_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "technique_comparison_v1")
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.summary_csv_path.exists())
            self.assertTrue(artifacts.scenario_csv_path.exists())
            self.assertTrue(artifacts.capability_csv_path.exists())
            self.assertTrue(artifacts.metric_heatmap_png_path.exists())
            self.assertTrue(artifacts.scatter_png_path.exists())
            self.assertTrue(artifacts.capability_png_path.exists())


if __name__ == "__main__":
    unittest.main()
