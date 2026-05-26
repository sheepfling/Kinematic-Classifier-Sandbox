from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from kinematic_classifier_sandbox.advanced_filters.imm_filter import IMMFilter
from kinematic_classifier_sandbox.advanced_filters.runner import (
    default_imm_mode_specs,
    default_imm_transition_matrix,
    run_imm_switching_benchmark,
    write_imm_artifacts,
)
from kinematic_classifier_sandbox.advanced_filters.protocols import validate_advanced_filter_step


class IMMFilterTests(unittest.TestCase):
    def test_imm_mode_probabilities_sum_to_one(self) -> None:
        imm = IMMFilter(default_imm_mode_specs(), default_imm_transition_matrix())
        imm.reset("traj", np.array([0.0], dtype=np.float64))
        for index in range(5):
            step = imm.update(float(index), np.array([float(index)], dtype=np.float64))
            validate_advanced_filter_step(step)
            assert imm.state is not None
            self.assertAlmostEqual(float(np.sum(imm.state.mode_probabilities)), 1.0)

    def test_imm_mixing_probabilities_sum_to_one_by_destination(self) -> None:
        imm = IMMFilter(default_imm_mode_specs(), default_imm_transition_matrix())
        imm.reset("traj", np.array([0.0], dtype=np.float64))
        imm.update(0.0, np.array([0.0], dtype=np.float64))
        imm.update(1.0, np.array([1.0], dtype=np.float64))
        assert imm.state is not None
        for values in imm.state.latest_mixing_probabilities.values():
            self.assertAlmostEqual(float(np.sum(values)), 1.0)

    def test_imm_switching_witness_improves_over_transition_baseline(self) -> None:
        result = run_imm_switching_benchmark(seed=17, replicas=4)
        rows = {row["method_id"]: row for row in result.method_comparison}
        self.assertGreater(
            float(rows["imm_v1"]["post_switch_accuracy"]),
            float(rows["transition_matrix_accumulator"]["post_switch_accuracy"]),
        )
        self.assertIn(result.metrics["promotion_decision"], {"promote", "revise"})

    def test_imm_outputs_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_imm_artifacts(temp_dir, result=run_imm_switching_benchmark(seed=17, replicas=2))
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "imm_filter_v1")
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.mode_probability_history_path.exists())
            self.assertTrue(artifacts.mixing_probability_history_path.exists())
            self.assertTrue(artifacts.mode_likelihood_history_path.exists())
            self.assertTrue(artifacts.state_estimate_history_path.exists())
            self.assertTrue(artifacts.posterior_history_path.exists())
            self.assertTrue(artifacts.switching_detection_metrics_path.exists())
            self.assertTrue(artifacts.method_comparison_path.exists())
            self.assertTrue(artifacts.decision_matrix_path.exists())
            self.assertTrue(artifacts.mode_probability_plot_path.exists())
            self.assertTrue(artifacts.state_plot_path.exists())
            report = artifacts.report_path.read_text(encoding="utf-8")
            self.assertIn("IMM Filter V1 Report", report)
            self.assertIn("promote/revise/reject/defer", report)


if __name__ == "__main__":
    unittest.main()
