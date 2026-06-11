from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.analysis.shapelet_motif_witness import (
    analyze_shapelet_maneuver_motif_witness,
    write_shapelet_maneuver_motif_witness_artifacts,
)


class ShapeletMotifWitnessTests(unittest.TestCase):
    def test_shapelet_witness_promotes_localized_motif_lane(self) -> None:
        result = analyze_shapelet_maneuver_motif_witness(seed=709, trajectories_per_class=12)

        self.assertEqual(result.metrics["promotion_decision"], "promote_shapelet_for_localized_maneuver_motif")
        self.assertGreater(result.metrics["shapelet_accuracy"], result.metrics["windowed_accuracy"])
        self.assertGreaterEqual(result.metrics["shapelet_alignment_rate"], 0.80)
        self.assertEqual(len(result.prediction_rows), 24)
        self.assertTrue(all(row.shapelet_alignment_error >= 0 for row in result.prediction_rows))

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_shapelet_maneuver_motif_witness_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "shapelet_maneuver_motif_v1")
            self.assertTrue(artifacts.trajectory_path.exists())
            self.assertTrue(artifacts.prediction_path.exists())
            self.assertTrue(artifacts.activation_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.metrics_path.exists())
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.decision_card_path.exists())
            self.assertEqual(len(artifacts.plot_paths), 3)
            self.assertTrue(all(path.exists() for path in artifacts.plot_paths))


if __name__ == "__main__":
    unittest.main()
