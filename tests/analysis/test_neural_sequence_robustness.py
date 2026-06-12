from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.analysis.neural_sequence_robustness import (
    analyze_neural_sequence_robustness_frontier,
    write_neural_sequence_robustness_frontier_artifacts,
)


class NeuralSequenceRobustnessTests(unittest.TestCase):
    def test_bounded_multi_seed_robustness_packet_runs(self) -> None:
        result = analyze_neural_sequence_robustness_frontier(
            seeds=(907, 1013),
            trajectories_per_case=6,
        )

        self.assertEqual(result.metrics["study_id"], "neural_sequence_robustness_frontier_v1")
        self.assertEqual(result.metrics["seed_count"], 2)
        method_names = {row.method_name for row in result.summary_rows}
        self.assertEqual(method_names, {"windowed_robust", "rocket_proxy", "kalman_bank", "tcn", "inceptiontime"})
        self.assertGreaterEqual(float(result.metrics["best_neural_mean_test_accuracy"]), 0.0)
        self.assertGreaterEqual(float(result.metrics["best_baseline_mean_test_accuracy"]), 0.0)
        self.assertIn(
            result.metrics["promotion_decision"],
            {"bounded_neural_robustness_candidate", "hold_neural_robustness_frontier"},
        )

    def test_artifacts_write_expected_outputs(self) -> None:
        result = analyze_neural_sequence_robustness_frontier(
            seeds=(907, 1013),
            trajectories_per_case=6,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_neural_sequence_robustness_frontier_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "neural_sequence_robustness_frontier_v1")
            self.assertTrue(artifacts.seed_summary_path.exists())
            self.assertTrue(artifacts.metric_summary_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.metrics_path.exists())
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.decision_card_path.exists())
            self.assertEqual(len(artifacts.plot_paths), 2)
            self.assertTrue(all(path.exists() for path in artifacts.plot_paths))


if __name__ == "__main__":
    unittest.main()
