from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.analysis.conformal_wrapper_witness import (
    analyze_coverage_control_under_shift,
    write_coverage_control_under_shift_artifacts,
)


class ConformalWrapperWitnessTests(unittest.TestCase):
    def test_conformal_wrapper_improves_coverage_under_shift(self) -> None:
        result = analyze_coverage_control_under_shift(seed=1209, trajectories_per_case=8)

        self.assertEqual(result.metrics["promotion_decision"], "promote_conformal_wrapper_for_coverage_control_under_shift")
        self.assertGreater(result.metrics["conformal_coverage"], result.metrics["singleton_coverage"])
        self.assertGreaterEqual(result.metrics["conformal_coverage"], 0.875)
        self.assertLessEqual(result.metrics["mean_prediction_set_size"], 2.0)

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_coverage_control_under_shift_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "coverage_control_under_shift_v1")
            self.assertTrue(artifacts.prediction_summary_path.exists())
            self.assertTrue(artifacts.coverage_summary_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.metrics_path.exists())
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.decision_card_path.exists())
            self.assertEqual(len(artifacts.plot_paths), 2)
            self.assertTrue(all(path.exists() for path in artifacts.plot_paths))


if __name__ == "__main__":
    unittest.main()
