from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.api import (
    GaussianPointwiseClassifier,
    PointwiseClassSpec,
    PointwiseTrajectory,
    default_pointwise_class_specs,
    generate_pointwise_benchmark_trajectories,
    render_pointwise_benchmark_report,
    render_pointwise_benchmark_png_bytes,
    render_pointwise_benchmark_svg,
    run_pointwise_benchmark,
    run_pointwise_classifier,
    write_pointwise_benchmark_artifacts,
)


class PointwiseBaselineTests(unittest.TestCase):
    def test_pointwise_classifier_respects_obvious_measurements(self) -> None:
        specs = (
            PointwiseClassSpec("A", mean=0.0, sigma=1.0, prior_weight=0.5),
            PointwiseClassSpec("B", mean=5.0, sigma=1.0, prior_weight=0.5),
        )
        classifier = GaussianPointwiseClassifier(specs)

        step_a = classifier.update(0.0, 0.0)
        self.assertGreater(step_a.posterior_weights["A"], 0.9)
        self.assertEqual(step_a.predicted_class, "A")

        classifier.reset()
        step_b = classifier.update(0.0, 5.0)
        self.assertGreater(step_b.posterior_weights["B"], 0.9)
        self.assertEqual(step_b.predicted_class, "B")

        classifier.reset()
        step_mid = classifier.update(0.0, 2.5)
        self.assertAlmostEqual(step_mid.posterior_weights["A"], 0.5, delta=0.1)
        self.assertAlmostEqual(step_mid.posterior_weights["B"], 0.5, delta=0.1)

    def test_prior_shift_changes_ambiguous_decision(self) -> None:
        specs = (
            PointwiseClassSpec("A", mean=0.0, sigma=1.0, prior_weight=0.8),
            PointwiseClassSpec("B", mean=5.0, sigma=1.0, prior_weight=0.2),
        )
        classifier = GaussianPointwiseClassifier(specs)
        step = classifier.update(0.0, 2.5)
        self.assertGreater(step.posterior_weights["A"], step.posterior_weights["B"])

    def test_generated_benchmark_separates_easy_from_overlap(self) -> None:
        result = run_pointwise_benchmark(seed=7)
        easy_runs = [run for run in result.runs if run.scenario_name == "easy"]
        overlap_runs = [run for run in result.runs if run.scenario_name == "overlap"]

        easy_accuracy = sum(run.final_predicted_class == run.true_class for run in easy_runs) / len(easy_runs)
        overlap_accuracy = sum(run.final_predicted_class == run.true_class for run in overlap_runs) / len(overlap_runs)

        self.assertGreater(easy_accuracy, 0.95)
        self.assertLess(overlap_accuracy, easy_accuracy)
        self.assertGreater(result.summary.final_accuracy, 0.7)
        self.assertIn("A", result.summary.confusion_counts)
        self.assertIn("B", result.summary.confusion_counts)

    def test_pointwise_artifacts_are_generated(self) -> None:
        result = run_pointwise_benchmark(seed=7)
        report = render_pointwise_benchmark_report(result)
        svg = render_pointwise_benchmark_svg(result)
        png = render_pointwise_benchmark_png_bytes(result)

        self.assertIn("Pointwise Gaussian Benchmark", report)
        self.assertIn("Easy", report)
        self.assertIn("<svg", svg)
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_pointwise_benchmark_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "pointwise_baseline")
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.posterior_history_path.exists())
            self.assertTrue(artifacts.confusion_matrix_path.exists())
            self.assertTrue(artifacts.plot_png_path.exists())


if __name__ == "__main__":
    unittest.main()
