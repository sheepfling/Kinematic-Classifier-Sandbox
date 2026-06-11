from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.inference.transition_matrix.reporting import (
    render_transition_benchmark_report,
    render_transition_numeric_walkthrough_markdown,
)
from kinematic_classifier_sandbox.inference.transition_matrix_accumulator import (
    run_transition_benchmark,
    write_transition_benchmark_artifacts,
)


class TransitionMatrixAccumulatorTests(unittest.TestCase):
    def test_transition_matrix_improves_post_switch_accuracy(self) -> None:
        result = run_transition_benchmark(seed=7, replicas=6)
        self.assertGreaterEqual(
            result.summary.transition_post_switch_accuracy,
            result.summary.static_post_switch_accuracy,
        )
        self.assertGreater(result.summary.improved_scenarios, 0)
        self.assertGreaterEqual(result.summary.kalman_post_switch_accuracy, 0.0)
        self.assertEqual(len(result.kalman_runs), len(result.static_runs))

    def test_transition_matrix_artifacts_are_generated(self) -> None:
        result = run_transition_benchmark(seed=7, replicas=4)
        report = render_transition_benchmark_report(result)
        walkthrough = render_transition_numeric_walkthrough_markdown(result)
        self.assertIn("Milestone 16", report)
        self.assertIn("transition-matrix accumulator", report)
        self.assertIn("Kalman mode bank", report)
        self.assertIn("Transition-Matrix Numeric Walkthrough", walkthrough)
        self.assertIn("propagated prior", walkthrough)
        self.assertIn("log numerator", walkthrough)
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_transition_benchmark_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "transition_matrix_accumulator_v1")
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.numeric_walkthrough_path.exists())
            self.assertTrue(artifacts.posterior_history_path.exists())
            self.assertTrue(artifacts.scenario_summary_path.exists())
            self.assertTrue(artifacts.config_path.exists())
            self.assertTrue(artifacts.dataset_manifest_path.exists())
            self.assertTrue(artifacts.plot_png_path.exists())
            self.assertTrue(artifacts.filter_step_trace_path.exists())
            self.assertTrue(artifacts.per_method_diagnostics_path.exists())
            self.assertTrue(artifacts.posterior_timeline_plot_path.exists())
            self.assertTrue(artifacts.likelihood_strip_plot_path.exists())
            self.assertTrue(artifacts.waterfall_plot_path.exists())
            self.assertTrue(artifacts.static_vs_transition_plot_path.exists())
            self.assertGreaterEqual(len(artifacts.step_card_paths), 2)
            self.assertIn("Transition-Matrix Numeric Walkthrough", artifacts.numeric_walkthrough_path.read_text(encoding="utf-8"))
            header = artifacts.scenario_summary_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("static_post_switch_accuracy", header)
            self.assertIn("transition_post_switch_accuracy", header)
            self.assertIn("kalman_post_switch_accuracy", header)
            trace_header = artifacts.filter_step_trace_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("predicted_probability", trace_header)
            self.assertIn("posterior_probability", trace_header)


if __name__ == "__main__":
    unittest.main()
