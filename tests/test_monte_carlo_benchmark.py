from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox import (
    render_monte_carlo_accuracy_png_bytes,
    render_monte_carlo_accuracy_svg,
    render_monte_carlo_calibration_png_bytes,
    render_monte_carlo_calibration_svg,
    render_monte_carlo_confusion_png_bytes,
    render_monte_carlo_confusion_svg,
    render_monte_carlo_report,
    render_monte_carlo_posterior_png_bytes,
    render_monte_carlo_posterior_svg,
    render_monte_carlo_time_to_confidence_png_bytes,
    render_monte_carlo_time_to_confidence_svg,
    render_monte_carlo_time_to_correct_png_bytes,
    render_monte_carlo_time_to_correct_svg,
    run_accumulator_monte_carlo_benchmark,
    write_monte_carlo_artifacts,
)


class MonteCarloBenchmarkTests(unittest.TestCase):
    def test_monte_carlo_pack_builds_metrics_and_artifacts(self) -> None:
        result = run_accumulator_monte_carlo_benchmark(seed=7, trajectories_per_class=8)

        report = render_monte_carlo_report(result)
        accuracy_svg = render_monte_carlo_accuracy_svg(result)
        accuracy_png = render_monte_carlo_accuracy_png_bytes(result)
        calibration_svg = render_monte_carlo_calibration_svg(result)
        calibration_png = render_monte_carlo_calibration_png_bytes(result)
        confusion_svg = render_monte_carlo_confusion_svg(result)
        confusion_png = render_monte_carlo_confusion_png_bytes(result)
        posterior_svg = render_monte_carlo_posterior_svg(result)
        posterior_png = render_monte_carlo_posterior_png_bytes(result)
        time_to_confidence_svg = render_monte_carlo_time_to_confidence_svg(result)
        time_to_confidence_png = render_monte_carlo_time_to_confidence_png_bytes(result)
        time_to_correct_svg = render_monte_carlo_time_to_correct_svg(result)
        time_to_correct_png = render_monte_carlo_time_to_correct_png_bytes(result)

        self.assertIn("Monte Carlo Accumulator Report", report)
        self.assertGreater(len(result.metrics_by_time), 0)
        self.assertGreater(len(result.calibration_bins), 0)
        self.assertGreaterEqual(result.summary.final_accuracy, 0.5)
        self.assertGreaterEqual(result.summary.expected_calibration_error, 0.0)
        self.assertGreaterEqual(result.summary.confidence_reached_rate, 0.0)
        self.assertLessEqual(result.summary.confidence_reached_rate, 1.0)
        self.assertGreaterEqual(result.summary.mean_brier_score, 0.0)
        self.assertIn("<svg", accuracy_svg)
        self.assertTrue(accuracy_png.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertIn("<svg", calibration_svg)
        self.assertTrue(calibration_png.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertIn("<svg", confusion_svg)
        self.assertTrue(confusion_png.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertIn("<svg", posterior_svg)
        self.assertTrue(posterior_png.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertIn("<svg", time_to_confidence_svg)
        self.assertTrue(time_to_confidence_png.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertIn("<svg", time_to_correct_svg)
        self.assertTrue(time_to_correct_png.startswith(b"\x89PNG\r\n\x1a\n"))

        first_accuracy = result.metrics_by_time[0]["accuracy"]
        last_accuracy = result.metrics_by_time[-1]["accuracy"]
        self.assertLessEqual(first_accuracy, last_accuracy)

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_monte_carlo_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "monte_carlo_accumulator")
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.summary_json_path.exists())
            self.assertTrue(artifacts.metrics_by_time_path.exists())
            self.assertTrue(artifacts.time_to_confidence_path.exists())
            self.assertTrue(artifacts.time_to_correct_classification_path.exists())
            self.assertTrue(artifacts.calibration_bins_path.exists())
            self.assertTrue(artifacts.confusion_final_path.exists())
            self.assertTrue(artifacts.confusion_confidence_gated_path.exists())
            self.assertTrue(artifacts.plot_accuracy_png_path.exists())
            self.assertTrue(artifacts.plot_posterior_png_path.exists())
            self.assertTrue(artifacts.plot_time_to_confidence_png_path.exists())
            self.assertTrue(artifacts.plot_time_to_correct_png_path.exists())
            self.assertTrue(artifacts.plot_calibration_png_path.exists())
            self.assertTrue(artifacts.plot_confusion_png_path.exists())


if __name__ == "__main__":
    unittest.main()
