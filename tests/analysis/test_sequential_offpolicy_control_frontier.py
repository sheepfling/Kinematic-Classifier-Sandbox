from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.analysis.sequential_offpolicy_control_frontier import (
    analyze_sequential_offpolicy_control_frontier,
    has_sequential_offpolicy_support,
    write_sequential_offpolicy_control_frontier_artifacts,
)


class SequentialOffPolicyControlFrontierTests(unittest.TestCase):
    @unittest.skipUnless(has_sequential_offpolicy_support(), "stable-baselines3 off-policy support is optional")
    def test_offpolicy_frontier_artifacts_are_generated(self) -> None:
        result = analyze_sequential_offpolicy_control_frontier(seed=1409, budget_sweep_timesteps=(32, 64), eval_episodes=1)

        self.assertEqual(result.metrics["study_id"], "sequential_offpolicy_control_frontier_v1")
        self.assertGreaterEqual(int(result.metrics["objective_count"]), 3)
        self.assertGreaterEqual(int(result.metrics["budget_count"]), 2)
        self.assertGreaterEqual(int(result.metrics["seed_count"]), 2)
        self.assertTrue(result.frontier_rows)
        self.assertTrue(result.budget_sweep_rows)
        self.assertTrue(result.seed_sweep_rows)
        self.assertIn(
            result.metrics["promotion_decision"],
            {
                "promote_offpolicy_sequential_frontier",
                "revise_sequential_offpolicy_frontier",
            },
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_sequential_offpolicy_control_frontier_artifacts(
                temp_dir,
                result=result,
                budget_sweep_timesteps=(32, 64),
                eval_episodes=1,
            )
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "sequential_offpolicy_control_frontier_v1")
            self.assertTrue(artifacts.frontier_summary_path.exists())
            self.assertTrue((Path(temp_dir) / "sequential_offpolicy_control_frontier_v1" / "summary.csv").exists())
            self.assertTrue(artifacts.budget_sweep_path.exists())
            self.assertTrue(artifacts.seed_sweep_path.exists())
            self.assertTrue(artifacts.metrics_path.exists())
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.decision_card_path.exists())
            self.assertEqual(len(artifacts.plot_paths), 2)
            self.assertTrue(all(path.exists() for path in artifacts.plot_paths))


if __name__ == "__main__":
    unittest.main()
