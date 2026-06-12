from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.analysis.ts2vec_backend_parity import (
    analyze_ts2vec_backend_parity,
    write_ts2vec_backend_parity_artifacts,
)


class Ts2VecBackendParityTests(unittest.TestCase):
    def test_analysis_builds_proxy_and_external_parity_rows(self) -> None:
        result = analyze_ts2vec_backend_parity()

        self.assertGreater(len(result.prediction_rows), 0)
        self.assertGreater(len(result.metric_rows), 0)
        self.assertIn("proxy_best_test_accuracy", result.metrics)
        self.assertIn("external_backend_available", result.metrics)
        self.assertIn("best_baseline_test_accuracy", result.metrics)
        method_names = {row.method_name for row in result.metric_rows}
        self.assertIn("ts2vec_proxy_centroid", method_names)
        self.assertIn("ts2vec_proxy_nn", method_names)
        if result.metrics["external_backend_available"] == "yes":
            self.assertIn("ts2vec_external_centroid", method_names)
            self.assertIn("ts2vec_external_nn", method_names)

    def test_artifacts_write_expected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_ts2vec_backend_parity_artifacts(temp_dir)

            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "ts2vec_backend_parity_v1")
            self.assertTrue(artifacts.prediction_summary_path.exists())
            self.assertTrue(artifacts.metric_summary_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.metrics_path.exists())
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.decision_card_path.exists())
            self.assertEqual(len(artifacts.plot_paths), 2)
            for path in artifacts.plot_paths:
                self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
