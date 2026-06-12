from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.analysis.archive_backend_diagnosis import (
    analyze_archive_backend_diagnosis,
    write_archive_backend_diagnosis_artifacts,
)


class ArchiveBackendDiagnosisTests(unittest.TestCase):
    def test_archive_backend_diagnosis_artifacts_are_generated(self) -> None:
        result = analyze_archive_backend_diagnosis(
            shared_trajectories_per_case=4,
            feature_trajectories_per_class=12,
            methods=("minirocket_family",),
            panel_variants=("normalized_position",),
            resample_lengths=(32,),
        )

        self.assertEqual(result.metrics["study_id"], "archive_backend_diagnosis_v1")
        self.assertEqual(len(result.diagnosis_rows), 2)
        self.assertEqual(len(result.summary_rows), 2)
        self.assertIn(
            result.metrics["diagnosis_read"],
            {"bounded_variants_do_not_recover_archive_lane", "some_variants_recover_archive_rows"},
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_archive_backend_diagnosis_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "archive_backend_diagnosis_v1")
            self.assertTrue(artifacts.diagnosis_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.metrics_path.exists())
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.decision_card_path.exists())
            self.assertEqual(len(artifacts.plot_paths), 2)
            self.assertTrue(all(path.exists() for path in artifacts.plot_paths))
            report_text = artifacts.report_path.read_text(encoding="utf-8")
            self.assertIn("Archive Backend Diagnosis", report_text)
            self.assertIn("diagnosis read", report_text)


if __name__ == "__main__":
    unittest.main()
