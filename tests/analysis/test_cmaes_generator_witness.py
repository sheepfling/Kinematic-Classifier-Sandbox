from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.analysis.cmaes_generator_witness import (
    analyze_continuous_generator_frontier,
    write_continuous_generator_frontier_artifacts,
)


class CmaesGeneratorWitnessTests(unittest.TestCase):
    def test_cmaes_generator_frontier_promotes_cmaes_lane(self) -> None:
        result = analyze_continuous_generator_frontier(seed=7)

        self.assertEqual(result.metrics["promotion_decision"], "promote_cmaes_for_continuous_generator_frontier")
        self.assertGreater(result.metrics["mean_cmaes_total_utility"], result.metrics["mean_heuristic_total_utility"])
        self.assertGreater(result.metrics["mean_cmaes_minus_heuristic"], 0.02)

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_continuous_generator_frontier_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "continuous_generator_frontier_v1")
            self.assertTrue(artifacts.frontier_summary_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.metrics_path.exists())
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.decision_card_path.exists())
            self.assertEqual(len(artifacts.plot_paths), 2)
            self.assertTrue(all(path.exists() for path in artifacts.plot_paths))


if __name__ == "__main__":
    unittest.main()
