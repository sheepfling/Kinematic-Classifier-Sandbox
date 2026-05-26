from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.api import (
    analyze_class_validity,
    write_class_validity_artifacts,
)


class ClassValidityTests(unittest.TestCase):
    def test_analysis_flags_ambiguous_and_relabel_candidates(self) -> None:
        result = analyze_class_validity()
        statuses = {row["label_status"] for row in result.score_rows}
        self.assertIn("ambiguous", statuses)
        self.assertIn("invalid", statuses)
        self.assertIn("relabel_candidate", statuses)
        self.assertIn("Class Validity Scoring", result.report_markdown)

    def test_artifacts_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_class_validity_artifacts(temp_dir)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "class_validity")
            self.assertTrue(artifacts.class_definition_schema_path.exists())
            self.assertTrue(artifacts.class_validity_scores_path.exists())
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.confusion_png_path.exists())
            self.assertTrue(artifacts.status_distribution_png_path.exists())
            self.assertTrue(artifacts.alternate_similarity_png_path.exists())

            payload = json.loads(artifacts.class_definition_schema_path.read_text(encoding="utf-8"))
            self.assertIn("properties", payload)


if __name__ == "__main__":
    unittest.main()
