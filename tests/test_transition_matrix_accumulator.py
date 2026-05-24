from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox import (
    render_transition_benchmark_report,
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
        self.assertIn("Milestone 16", report)
        self.assertIn("transition-matrix accumulator", report)
        self.assertIn("Kalman mode bank", report)
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_transition_benchmark_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "transition_matrix_accumulator_v1")
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.posterior_history_path.exists())
            self.assertTrue(artifacts.scenario_summary_path.exists())
            self.assertTrue(artifacts.config_path.exists())
            self.assertTrue(artifacts.dataset_manifest_path.exists())
            self.assertTrue(artifacts.plot_png_path.exists())
            header = artifacts.scenario_summary_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("static_post_switch_accuracy", header)
            self.assertIn("transition_post_switch_accuracy", header)
            self.assertIn("kalman_post_switch_accuracy", header)


if __name__ == "__main__":
    unittest.main()
