from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.common_experiment.runner import analyze_common_experiment
from kinematic_classifier_sandbox.witnesses.toy_1d.bayesian_walkthroughs import (
    analyze_bayesian_walkthroughs,
    write_bayesian_walkthrough_artifacts,
)


class BayesianWalkthroughTests(unittest.TestCase):
    def test_walkthrough_suite_emits_real_examples_and_hand_checkable_demo(self) -> None:
        result = analyze_bayesian_walkthroughs(seed=7, trajectories_per_case=6)

        self.assertGreater(len(result.bayesian_step_rows), 0)
        self.assertGreater(len(result.prior_sweep_rows), 0)
        self.assertGreater(len(result.feature_contribution_rows), 0)
        self.assertGreater(len(result.posterior_flip_threshold_rows), 0)
        self.assertIn("Bayesian Evidence Walkthroughs", result.report_markdown)

        hand_row = next(row for row in result.bayesian_step_rows if str(row["example_id"]) == "hand_checkable_binary_demo")
        self.assertAlmostEqual(float(hand_row["prior_a"]), 0.5, places=6)
        self.assertAlmostEqual(float(hand_row["posterior_a"]), 0.8, places=6)
        self.assertAlmostEqual(float(hand_row["posterior_b"]), 0.2, places=6)

        common = analyze_common_experiment(seed=7, trajectories_per_case=6)
        trajectory_ids = {str(row["trajectory_id"]) for row in common.pair_prediction_rows}
        class_pair_ids = {str(row["class_pair_id"]) for row in common.pair_prediction_rows}

        self.assertIn(str(result.selected_walkthrough["trajectory_id"]), trajectory_ids)
        self.assertIn(str(result.selected_walkthrough["class_pair_id"]), class_pair_ids)
        self.assertIn(str(result.feature_example["trajectory_id"]), trajectory_ids)
        self.assertIn(str(result.feature_example["class_pair_id"]), class_pair_ids)

    def test_walkthrough_suite_writes_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_bayesian_walkthrough_artifacts(
                temp_dir,
                result=analyze_bayesian_walkthroughs(seed=7, trajectories_per_case=6),
            )
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "bayesian_walkthroughs")
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.bayesian_step_tables_path.exists())
            self.assertTrue(artifacts.prior_sweep_examples_path.exists())
            self.assertTrue(artifacts.feature_contribution_examples_path.exists())
            self.assertTrue(artifacts.posterior_flip_thresholds_path.exists())
            self.assertTrue(artifacts.prior_to_posterior_single_step_path.exists())
            self.assertTrue(artifacts.likelihood_curves_with_feature_value_path.exists())
            self.assertTrue(artifacts.posterior_timeline_path.exists())
            self.assertTrue(artifacts.log_odds_timeline_path.exists())
            self.assertTrue(artifacts.bayes_factor_timeline_path.exists())
            self.assertTrue(artifacts.prior_sensitivity_curve_path.exists())
            self.assertTrue(artifacts.feature_ablation_posterior_path.exists())
            self.assertTrue(artifacts.confidence_threshold_crossing_path.exists())

            report_text = artifacts.report_path.read_text(encoding="utf-8")
            self.assertIn("Representative Sequential Walkthrough", report_text)
            self.assertIn("Feature Evidence Example", report_text)

            with artifacts.bayesian_step_tables_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(any(row["example_id"] == "hand_checkable_binary_demo" for row in rows))
            self.assertTrue(any(row["example_type"] == "trajectory_walkthrough" for row in rows))


if __name__ == "__main__":
    unittest.main()
