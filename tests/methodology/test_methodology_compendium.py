from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.methodology.compendium import (
    analyze_methodology_compendium,
    write_methodology_compendium_artifacts,
)


class MethodologyCompendiumTests(unittest.TestCase):
    def test_analysis_combines_all_survey_parts(self) -> None:
        result = analyze_methodology_compendium()

        self.assertIn("# Kinematic Classifier Methodology Compendium", result.markdown)
        self.assertIn("## Part 1. Posterior Update Math", result.markdown)
        self.assertIn("## Part 2. Methodology Evaluation Framework", result.markdown)
        self.assertIn("## Part 3. Classifier Ladder and Contracts", result.markdown)
        self.assertIn("## Part 4. Corpus Generation and Search", result.markdown)
        self.assertIn("## Part 5. Dimensional Lift and Advanced Filter Gates", result.markdown)
        self.assertIn("This note documents the posterior update math", result.markdown)
        self.assertIn("This note covers the evaluation side of the repo", result.markdown)
        self.assertIn("This note documents the repo's classifier ladder", result.markdown)

    def test_artifacts_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_methodology_compendium_artifacts(Path(temp_dir))
            self.assertTrue(artifacts.source_markdown_path.exists())
            self.assertTrue(artifacts.artifact_markdown_path.exists())

            artifact_text = artifacts.artifact_markdown_path.read_text(encoding="utf-8")
            self.assertIn("Included Documents", artifact_text)
            self.assertIn("kinematic_classifier_methodology.pdf", artifact_text)
            self.assertIn("methodology_evaluation_framework.pdf", artifact_text)

    def test_build_script_exists(self) -> None:
        root = Path(__file__).resolve().parents[2]
        script_path = root / "src" / "kinematic_classifier_sandbox" / "meta" / "__init__.py"
        self.assertTrue(script_path.exists())
        script_text = script_path.read_text(encoding="utf-8")
        self.assertIn("write_methodology_compendium_artifacts", script_text)


if __name__ == "__main__":
    unittest.main()
