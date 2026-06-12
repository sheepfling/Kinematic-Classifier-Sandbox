from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.analysis.archive_feature_headroom_witness import (
    analyze_archive_feature_headroom_witness,
    write_archive_feature_headroom_witness_artifacts,
)


class ArchiveFeatureHeadroomWitnessTests(unittest.TestCase):
    def test_archive_feature_headroom_witness_artifacts_are_generated(self) -> None:
        result = analyze_archive_feature_headroom_witness(seed=811, trajectories_per_class=12)

        self.assertEqual(result.metrics["study_id"], "archive_feature_headroom_witness_v1")
        method_names = {row.method_name for row in result.method_rows}
        self.assertEqual(
            method_names,
            {
                "windowed_feature_summary",
                "gradient_boosted_features",
                "minirocket_family",
                "drcif_interval_forests",
                "dictionary_tde_family",
                "hive_cote",
            },
        )
        self.assertEqual(len(result.seed_sweep_rows), 6)
        self.assertIn(
            result.metrics["promotion_decision"],
            {
                "promote_archive_feature_headroom_witness_for_followon_review",
                "record_archive_feature_headroom_witness_keep_gate_closed",
            },
        )
        boosted_row = next(row for row in result.method_rows if row.method_name == "gradient_boosted_features")
        self.assertGreaterEqual(boosted_row.test_accuracy, 0.5)

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_archive_feature_headroom_witness_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "archive_feature_headroom_witness_v1")
            self.assertTrue(artifacts.method_summary_path.exists())
            self.assertTrue(artifacts.seed_sweep_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.metrics_path.exists())
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.decision_card_path.exists())
            self.assertEqual(len(artifacts.plot_paths), 2)
            self.assertTrue(all(path.exists() for path in artifacts.plot_paths))
            report_text = artifacts.report_path.read_text(encoding="utf-8")
            self.assertIn("Archive Feature Headroom Witness", report_text)
            self.assertIn("archive champion", report_text)


if __name__ == "__main__":
    unittest.main()
