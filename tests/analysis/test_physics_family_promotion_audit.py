from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.analysis.physics_family_promotion_audit import (
    analyze_physics_family_promotion_audit,
    write_physics_family_promotion_audit_artifacts,
)


class PhysicsFamilyPromotionAuditTests(unittest.TestCase):
    def test_physics_family_promotion_audit_runs(self) -> None:
        result = analyze_physics_family_promotion_audit()

        self.assertEqual(result.metrics["study_id"], "physics_family_promotion_audit_v1")
        method_names = {row.method_name for row in result.method_rows}
        self.assertEqual(method_names, {"imm", "ukf", "gaussian_sum_filter", "rbpf"})
        self.assertEqual(result.metrics["family_decision"], "physics_family_advanced_filter_blockers_cleared")
        self.assertEqual(result.metrics["study_justified_count"], 4)
        self.assertEqual(result.metrics["still_open_count"], 0)
        self.assertEqual(result.metrics["primary_blocker"], "advanced_filter_core_blockers_cleared")

    def test_physics_family_promotion_audit_artifacts_are_written(self) -> None:
        result = analyze_physics_family_promotion_audit()
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_physics_family_promotion_audit_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "physics_family_promotion_audit_v1")
            self.assertTrue(artifacts.method_summary_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.metrics_path.exists())
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.decision_card_path.exists())
            self.assertEqual(len(artifacts.plot_paths), 2)
            self.assertTrue(all(path.exists() for path in artifacts.plot_paths))


if __name__ == "__main__":
    unittest.main()
