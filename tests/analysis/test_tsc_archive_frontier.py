from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.analysis.tsc_archive_frontier import (
    analyze_tsc_archive_baseline_frontier,
    write_tsc_archive_baseline_frontier_artifacts,
)


class TSCArchiveFrontierTests(unittest.TestCase):
    def test_tsc_archive_frontier_artifacts_are_generated(self) -> None:
        result = analyze_tsc_archive_baseline_frontier(seed=1009, trajectories_per_case=8)

        self.assertIn(
            result.metrics["promotion_decision"],
            {
                "hold_modern_tsc_at_optional_wrapper_stage",
                "record_partial_external_execution_keep_gate_closed",
                "promote_faithful_archive_wrappers",
            },
        )
        self.assertEqual(result.metrics["archive_seed_count"], 2)
        self.assertIn(
            result.metrics["next_gate_decision"],
            {"keep_generic_tsc_gate_closed", "ready_for_named_witness_comparison"},
        )
        method_names = {row.method_name for row in result.metric_rows}
        self.assertEqual(
            method_names,
            {
                "windowed_robust",
                "kalman_bank",
                "minirocket_family",
                "drcif_interval_forests",
                "dictionary_tde_family",
                "hive_cote",
            },
        )
        self.assertGreaterEqual(result.metrics["minirocket_test_accuracy"], 0.0)
        self.assertGreaterEqual(result.metrics["drcif_test_accuracy"], 0.0)
        self.assertGreaterEqual(result.metrics["dictionary_test_accuracy"], 0.0)
        self.assertGreaterEqual(result.metrics["hive_test_accuracy"], 0.0)
        self.assertIn(result.metrics["minirocket_backend"], {"local_proxy", "aeon.classification.convolution_based:MiniRocketClassifier", "aeon.classification.convolution_based:MultiRocketClassifier", "aeon.classification.convolution_based:MultiRocketHydraClassifier", "aeon.classification.convolution_based:HydraClassifier"})
        self.assertTrue(str(result.metrics["hive_backend"]))
        self.assertIn(result.metrics["archive_integration_read"], {"wrapper_stage_only", "mixed_external_and_fallback", "all_external"})
        self.assertGreaterEqual(int(result.metrics["archive_attempted_family_count"]), 0)
        self.assertGreaterEqual(int(result.metrics["archive_external_family_count"]), 0)
        self.assertGreaterEqual(int(result.metrics["archive_fallback_family_count"]), 0)
        self.assertGreaterEqual(int(result.metrics["archive_failed_external_family_count"]), 0)
        self.assertEqual(len(result.seed_sweep_rows), 6)
        minirocket_metric = next(row for row in result.metric_rows if row.method_name == "minirocket_family")
        self.assertGreaterEqual(minirocket_metric.test_nll, 0.0)
        self.assertGreaterEqual(minirocket_metric.test_ece, 0.0)
        self.assertIn(minirocket_metric.seed_stability_read, {"narrow_seed_sweep_pass", "narrow_seed_sweep_flags_instability"})

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_tsc_archive_baseline_frontier_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "tsc_archive_baseline_frontier_v1")
            self.assertTrue(artifacts.prediction_summary_path.exists())
            self.assertTrue(artifacts.metric_summary_path.exists())
            self.assertTrue(artifacts.seed_sweep_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.metrics_path.exists())
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.decision_card_path.exists())
            self.assertEqual(len(artifacts.plot_paths), 4)
            self.assertTrue(all(path.exists() for path in artifacts.plot_paths))
            report_text = artifacts.report_path.read_text(encoding="utf-8")
            self.assertIn("integration read", report_text)
            self.assertIn("Backend Provenance", report_text)
            self.assertIn("failed external family count", report_text)
            self.assertIn("seed robustness read", report_text)
            self.assertIn("calibration read", report_text)


if __name__ == "__main__":
    unittest.main()
