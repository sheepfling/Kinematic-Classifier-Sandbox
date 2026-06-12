from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.analysis.archive_family_promotion_audit import (
    analyze_archive_family_promotion_audit,
    write_archive_family_promotion_audit_artifacts,
)


class ArchiveFamilyPromotionAuditTests(unittest.TestCase):
    def test_archive_family_promotion_audit_runs(self) -> None:
        result = analyze_archive_family_promotion_audit(
            shared_trajectories_per_case=8,
            feature_trajectories_per_class=12,
        )

        self.assertEqual(result.metrics["study_id"], "archive_family_promotion_audit_v1")
        method_names = {row.method_name for row in result.method_rows}
        self.assertEqual(
            method_names,
            {
                "minirocket_family",
                "drcif_interval_forests",
                "dictionary_tde_family",
                "hive_cote",
            },
        )
        self.assertIn(
            result.metrics["promotion_decision"],
            {"archive_family_candidate_exists", "keep_generic_tsc_gate_closed"},
        )
        self.assertTrue(str(result.metrics["closest_method"]))

    def test_archive_family_promotion_audit_artifacts_are_written(self) -> None:
        result = analyze_archive_family_promotion_audit(
            shared_trajectories_per_case=8,
            feature_trajectories_per_class=12,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_archive_family_promotion_audit_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "archive_family_promotion_audit_v1")
            self.assertTrue(artifacts.method_summary_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.metrics_path.exists())
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.decision_card_path.exists())
            self.assertEqual(len(artifacts.plot_paths), 2)
            self.assertTrue(all(path.exists() for path in artifacts.plot_paths))


if __name__ == "__main__":
    unittest.main()
