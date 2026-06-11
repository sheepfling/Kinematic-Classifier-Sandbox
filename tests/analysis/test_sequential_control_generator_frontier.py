from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.analysis.sequential_control_generator_frontier import (
    analyze_sequential_control_generator_frontier,
    write_sequential_control_generator_frontier_artifacts,
)
from kinematic_classifier_sandbox.corpus.trajectory_exploration.ppo_boundary_control import (
    has_stable_baselines3_support,
)


class SequentialControlGeneratorFrontierTests(unittest.TestCase):
    @unittest.skipUnless(has_stable_baselines3_support(), "stable-baselines3 is optional")
    def test_sequential_control_frontier_artifacts_are_generated(self) -> None:
        result = analyze_sequential_control_generator_frontier(seed=1309)

        self.assertEqual(result.metrics["study_id"], "sequential_control_generator_frontier_v1")
        self.assertGreaterEqual(int(result.metrics["objective_count"]), 3)
        self.assertTrue(result.frontier_rows)
        self.assertIn(
            result.metrics["promotion_decision"],
            {
                "promote_ppo_proxy_for_sequential_control_frontier",
                "revise_sequential_control_proxy",
            },
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_sequential_control_generator_frontier_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "sequential_control_generator_frontier_v1")
            self.assertTrue(artifacts.frontier_summary_path.exists())
            self.assertTrue(artifacts.metrics_path.exists())
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.decision_card_path.exists())
            self.assertEqual(len(artifacts.plot_paths), 2)
            self.assertTrue(all(path.exists() for path in artifacts.plot_paths))


if __name__ == "__main__":
    unittest.main()
