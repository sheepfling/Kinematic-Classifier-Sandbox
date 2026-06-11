from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.analysis.temperature_scaling_witness import (
    analyze_confidence_calibration_shift,
    write_confidence_calibration_shift_artifacts,
)


class TemperatureScalingWitnessTests(unittest.TestCase):
    def test_temperature_scaling_improves_shifted_calibration(self) -> None:
        result = analyze_confidence_calibration_shift(seed=1109, trajectories_per_case=8)

        self.assertEqual(result.metrics["promotion_decision"], "promote_temperature_scaling_for_confidence_calibration_shift")
        self.assertLess(result.metrics["scaled_ece"], result.metrics["raw_ece"])
        self.assertLessEqual(result.metrics["scaled_brier"], result.metrics["raw_brier"])
        self.assertLessEqual(result.metrics["scaled_nll"], result.metrics["raw_nll"])

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_confidence_calibration_shift_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "confidence_calibration_shift_v1")
            self.assertTrue(artifacts.prediction_summary_path.exists())
            self.assertTrue(artifacts.calibration_bins_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.metrics_path.exists())
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.decision_card_path.exists())
            self.assertEqual(len(artifacts.plot_paths), 2)
            self.assertTrue(all(path.exists() for path in artifacts.plot_paths))


if __name__ == "__main__":
    unittest.main()
