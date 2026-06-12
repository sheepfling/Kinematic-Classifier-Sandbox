from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.analysis.tsc_archive_backend_smoke import (
    analyze_tsc_archive_backend_smoke,
    write_tsc_archive_backend_smoke_artifacts,
)


class TSCArchiveBackendSmokeTests(unittest.TestCase):
    def test_backend_smoke_artifacts_are_generated(self) -> None:
        result = analyze_tsc_archive_backend_smoke(timeout_seconds=1.0)
        self.assertEqual(result.metrics["study_id"], "tsc_archive_backend_smoke_v1")
        self.assertEqual(result.metrics["family_count"], 4)
        self.assertIn(
            result.metrics["integration_read"],
            {"no_external_backend_available", "mixed_smoke_outcomes", "all_external_smoke_succeeded"},
        )
        self.assertEqual(len(result.rows), 4)

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_tsc_archive_backend_smoke_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "tsc_archive_backend_smoke_v1")
            self.assertTrue(artifacts.row_summary_path.exists())
            self.assertTrue(artifacts.metrics_path.exists())
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.decision_card_path.exists())
            report_text = artifacts.report_path.read_text(encoding="utf-8")
            self.assertIn("TSC Archive Backend Smoke", report_text)
            self.assertIn("Integration read", report_text)


if __name__ == "__main__":
    unittest.main()
