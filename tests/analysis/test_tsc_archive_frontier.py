from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.analysis.tsc_archive_frontier import (
    analyze_tsc_archive_baseline_frontier,
    write_tsc_archive_baseline_frontier_artifacts,
)


class TSCArchiveFrontierTests(unittest.TestCase):
    def test_tsc_archive_proxy_frontier_artifacts_are_generated(self) -> None:
        result = analyze_tsc_archive_baseline_frontier(seed=1009, trajectories_per_case=8)

        self.assertEqual(result.metrics["promotion_decision"], "hold_modern_tsc_at_proxy_stage")
        method_names = {row.method_name for row in result.metric_rows}
        self.assertEqual(method_names, {"windowed_robust", "rocket_proxy", "kalman_bank", "drcif_proxy", "dictionary_proxy", "hive_cote_proxy"})
        self.assertGreaterEqual(result.metrics["rocket_overall_accuracy"], 0.0)
        self.assertGreaterEqual(result.metrics["drcif_overall_accuracy"], 0.0)
        self.assertGreaterEqual(result.metrics["dictionary_overall_accuracy"], 0.0)
        self.assertGreaterEqual(result.metrics["hive_overall_accuracy"], 0.0)

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_tsc_archive_baseline_frontier_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "tsc_archive_baseline_frontier_v1")
            self.assertTrue(artifacts.prediction_summary_path.exists())
            self.assertTrue(artifacts.metric_summary_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.metrics_path.exists())
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.decision_card_path.exists())
            self.assertEqual(len(artifacts.plot_paths), 2)
            self.assertTrue(all(path.exists() for path in artifacts.plot_paths))


if __name__ == "__main__":
    unittest.main()
