from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.analysis.gradient_boosted_feature_witness import (
    analyze_feature_headroom_frontier,
    write_feature_headroom_frontier_artifacts,
)


class GradientBoostedFeatureWitnessTests(unittest.TestCase):
    def test_feature_headroom_frontier_promotes_boosted_feature_lane(self) -> None:
        result = analyze_feature_headroom_frontier(seed=811, trajectories_per_class=12)

        self.assertEqual(result.metrics["promotion_decision"], "promote_gradient_boosted_features_for_feature_headroom")
        self.assertGreater(result.metrics["boosted_test_accuracy"], result.metrics["windowed_test_accuracy"])
        self.assertGreaterEqual(result.metrics["boosted_train_accuracy"], 0.85)
        self.assertEqual(len(result.stumps), 4)

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_feature_headroom_frontier_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "feature_headroom_frontier_v1")
            self.assertTrue(artifacts.trajectory_path.exists())
            self.assertTrue(artifacts.feature_matrix_path.exists())
            self.assertTrue(artifacts.stump_summary_path.exists())
            self.assertTrue(artifacts.prediction_summary_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.metrics_path.exists())
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.decision_card_path.exists())
            self.assertEqual(len(artifacts.plot_paths), 3)
            self.assertTrue(all(path.exists() for path in artifacts.plot_paths))


if __name__ == "__main__":
    unittest.main()
