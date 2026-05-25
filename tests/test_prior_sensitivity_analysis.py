from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.api import *


class PriorSensitivityAnalysisTests(unittest.TestCase):
    def test_ambiguous_symmetric_trajectory_tracks_the_prior(self) -> None:
        result = analyze_prior_sensitivity(seed=7)
        row = next(
            sweep_row
            for sweep_row in result.sweep_rows
            if sweep_row.trajectory_id == "ambiguous_mid" and abs(sweep_row.prior_a - 0.25) < 1e-9
        )
        self.assertAlmostEqual(row.posterior_a, 0.25, places=6)
        self.assertAlmostEqual(row.posterior_b, 0.75, places=6)
        self.assertAlmostEqual(row.cumulative_log_likelihood_ratio, 0.0, places=6)
        self.assertEqual(row.final_class, "B")

    def test_easy_trajectory_is_not_flipped_by_small_prior_shift(self) -> None:
        result = analyze_prior_sensitivity(seed=7)
        row = next(threshold for threshold in result.flip_thresholds if threshold.trajectory_id == "easy_A_0")
        self.assertEqual(row.uniform_prior_class, "A")
        self.assertIsNone(row.smallest_prior_shift_to_flip)

    def test_pointwise_and_windowed_share_the_prior_sensitivity_contract(self) -> None:
        pointwise = analyze_pointwise_prior_sensitivity(seed=7)
        windowed_raw = analyze_windowed_prior_sensitivity(seed=7, feature_mode="raw")
        windowed_robust = analyze_windowed_prior_sensitivity(seed=7, feature_mode="robust")

        self.assertEqual(pointwise.method_name, "pointwise")
        self.assertEqual(windowed_raw.method_name, "windowed_raw")
        self.assertEqual(windowed_robust.method_name, "windowed_robust")
        self.assertEqual(pointwise.class_names, ("A", "B"))
        self.assertEqual(windowed_raw.class_names, ("low", "high"))
        self.assertGreater(len(pointwise.sweep_rows), 0)
        self.assertGreater(len(windowed_raw.sweep_rows), 0)
        self.assertGreater(len(windowed_robust.sweep_rows), 0)
        self.assertTrue(any(row.scenario_name == "overlap" for row in pointwise.sweep_rows))
        self.assertTrue(any(row.scenario_name == "low_spike" for row in windowed_raw.sweep_rows))

    def test_cross_method_prior_comparison_artifacts_are_generated(self) -> None:
        result = analyze_cross_method_prior_comparison(seed=7)
        report = render_cross_method_prior_comparison_report(result)
        svg = render_cross_method_prior_comparison_svg(result)
        png = render_cross_method_prior_comparison_png_bytes(result)

        self.assertIn("Cross-Method Prior Sensitivity Comparison", report)
        self.assertIn("<svg", svg)
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
        method_names = {row["method_name"] for row in result.rows}
        self.assertEqual(method_names, {"accumulator", "pointwise", "windowed_raw", "windowed_robust"})
        self.assertEqual(result.scenario_names, ("easy", "boundary", "outlier", "transition", "long_history"))
        pointwise_row = next(row for row in result.rows if row["method_name"] == "pointwise")
        self.assertEqual(pointwise_row["easy_status"], "stable")
        self.assertEqual(pointwise_row["boundary_status"], "flips")

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_cross_method_prior_comparison_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "prior_sensitivity_cross_method_v1")
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.comparison_csv_path.exists())
            self.assertTrue(artifacts.status_csv_path.exists())
            self.assertTrue(artifacts.plot_png_path.exists())

    def test_artifacts_are_generated(self) -> None:
        result = analyze_prior_sensitivity(seed=7)
        report = render_prior_sensitivity_report(result)
        posterior_svg = render_prior_sensitivity_posterior_svg(result)
        posterior_png = render_prior_sensitivity_posterior_png_bytes(result)
        flip_svg = render_prior_sensitivity_flip_svg(result)
        flip_png = render_prior_sensitivity_flip_png_bytes(result)
        heatmap_svg = render_prior_sensitivity_heatmap_svg(result)
        heatmap_png = render_prior_sensitivity_heatmap_png_bytes(result)
        decision_svg = render_prior_sensitivity_decision_svg(result)
        decision_png = render_prior_sensitivity_decision_png_bytes(result)
        decomposition_svg = render_prior_sensitivity_decomposition_svg(result)
        decomposition_png = render_prior_sensitivity_decomposition_png_bytes(result)
        pairwise_flip_svg = render_prior_sensitivity_pairwise_flip_svg(result)
        pairwise_flip_png = render_prior_sensitivity_pairwise_flip_png_bytes(result)
        fragility_svg = render_prior_sensitivity_fragility_svg(result)
        fragility_png = render_prior_sensitivity_fragility_png_bytes(result)

        self.assertIn("Prior Sensitivity and Bias Study", report)
        self.assertIn("<svg", posterior_svg)
        self.assertIn("<svg", flip_svg)
        self.assertIn("<svg", heatmap_svg)
        self.assertIn("<svg", decision_svg)
        self.assertIn("<svg", decomposition_svg)
        self.assertIn("<svg", pairwise_flip_svg)
        self.assertIn("<svg", fragility_svg)
        self.assertTrue(posterior_png.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertTrue(flip_png.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertTrue(heatmap_png.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertTrue(decision_png.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertTrue(decomposition_png.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertTrue(pairwise_flip_png.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertTrue(fragility_png.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertGreater(result.summary.flipped_by_small_prior_fraction, 0.0)

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_prior_sensitivity_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "prior_sensitivity_v1")
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.sweep_path.exists())
            self.assertTrue(artifacts.flip_thresholds_path.exists())
            self.assertTrue(artifacts.metrics_path.exists())
            self.assertTrue(artifacts.config_path.exists())
            self.assertTrue(artifacts.plot_posterior_png_path.exists())
            self.assertTrue(artifacts.plot_flip_png_path.exists())
            self.assertTrue(artifacts.plot_heatmap_png_path.exists())
            self.assertTrue(artifacts.plot_decision_png_path.exists())
            self.assertTrue(artifacts.plot_decomposition_png_path.exists())
            self.assertTrue(artifacts.plot_pairwise_flip_png_path.exists())
            self.assertTrue(artifacts.plot_fragility_png_path.exists())
            metrics = json.loads(artifacts.metrics_path.read_text(encoding="utf-8"))
            self.assertIn("fraction_flipped_by_small_prior_perturbation", metrics)


if __name__ == "__main__":
    unittest.main()
