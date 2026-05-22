from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox import render_method_survey_markdown, write_method_survey_artifact


class ArtifactTests(unittest.TestCase):
    def test_render_mentions_expected_sections(self) -> None:
        markdown = render_method_survey_markdown()
        self.assertIn("Kinematic Method Survey Summary", markdown)
        self.assertIn("Traditional", markdown)
        self.assertIn("Advanced", markdown)
        self.assertIn("Current Recommended Sandbox Baseline", markdown)

    def test_write_artifact_creates_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = write_method_survey_artifact(temp_dir)
            self.assertEqual(output_path, Path(temp_dir) / "method_survey_summary.md")
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
