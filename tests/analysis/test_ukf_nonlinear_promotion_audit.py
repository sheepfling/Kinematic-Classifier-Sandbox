from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.analysis.ukf_nonlinear_promotion_audit import (
    analyze_ukf_nonlinear_promotion_audit,
    write_ukf_nonlinear_promotion_audit_artifacts,
)


class UKFNonlinearPromotionAuditTests(unittest.TestCase):
    def test_ukf_nonlinear_promotion_audit_runs(self) -> None:
        result = analyze_ukf_nonlinear_promotion_audit()

        self.assertEqual(result.metrics["study_id"], "ukf_nonlinear_promotion_audit_v1")
        self.assertIn(
            result.metrics["promotion_decision"],
            {"promote_to_study_justified", "keep_gate_closed"},
        )
        self.assertTrue(str(result.metrics["blocker_summary"]))
        self.assertEqual(len(result.audit_rows), 9)

    def test_ukf_nonlinear_promotion_audit_artifacts_are_written(self) -> None:
        result = analyze_ukf_nonlinear_promotion_audit()

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_ukf_nonlinear_promotion_audit_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "ukf_nonlinear_promotion_audit_v1")
            self.assertTrue(artifacts.audit_rows_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.metrics_path.exists())
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.decision_card_path.exists())
            self.assertEqual(len(artifacts.plot_paths), 1)
            self.assertTrue(all(path.exists() for path in artifacts.plot_paths))


if __name__ == "__main__":
    unittest.main()
