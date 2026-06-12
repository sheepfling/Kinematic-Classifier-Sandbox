from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.analysis.embedding_baseline_frontier import (
    analyze_embedding_baseline_frontier,
    write_embedding_baseline_frontier_artifacts,
)


class EmbeddingBaselineFrontierTests(unittest.TestCase):
    def test_analysis_builds_embedding_frontier(self) -> None:
        result = analyze_embedding_baseline_frontier()
        self.assertGreater(len(result.view_rows), 0)
        self.assertGreater(len(result.embedding_rows), 0)
        self.assertGreater(len(result.prediction_rows), 0)
        self.assertGreater(len(result.metric_rows), 0)
        self.assertIn("mean_canonical_correlation", result.metrics)
        self.assertIn("ts2vec_centroid_test_accuracy", result.metrics)
        self.assertIn("ts2vec_nn_test_accuracy", result.metrics)
        self.assertIn("online_ts2vec_test_accuracy", result.metrics)
        self.assertIn("online_ts2vec_mean_confidence", result.metrics)
        self.assertIn("online_route_win_rate", result.metrics)
        self.assertIn("ts2vec_backend", result.metrics)
        self.assertGreaterEqual(float(result.metrics["mean_canonical_correlation"]), 0.0)
        self.assertGreaterEqual(
            float(result.metrics["ts2vec_nn_test_accuracy"]),
            float(result.metrics["windowed_test_accuracy"]),
        )
        self.assertEqual(result.metrics["promotion_decision"], "promote_embedding_baseline_frontier")
        method_names = {row.method_name for row in result.prediction_rows}
        self.assertIn("ts2vec_centroid", method_names)
        self.assertIn("ts2vec_nn", method_names)
        self.assertGreater(len(result.online_route_rows), 0)
        self.assertEqual(result.online_route_rows[0].prefix_fraction, 0.25)

    def test_artifacts_write_expected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_embedding_baseline_frontier_artifacts(temp_dir)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "embedding_baseline_frontier_v1")
            self.assertTrue(artifacts.view_summary_path.exists())
            self.assertTrue(artifacts.embedding_summary_path.exists())
            self.assertTrue(artifacts.prediction_summary_path.exists())
            self.assertTrue(artifacts.metric_summary_path.exists())
            self.assertTrue(artifacts.online_route_summary_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.metrics_path.exists())
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.decision_card_path.exists())
            self.assertEqual(len(artifacts.plot_paths), 4)
            for path in artifacts.plot_paths:
                self.assertTrue(path.exists())
            report = artifacts.report_path.read_text(encoding="utf-8")
            if "ts2vec_external" in report:
                self.assertIn("optional external TS2Vec backend", report)
            else:
                self.assertIn("TS2Vec-style proxy frontier", report)


if __name__ == "__main__":
    unittest.main()
