from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.artifacts import (
    render_method_survey_markdown,
    render_posterior_numeric_walkthrough_markdown,
    render_posterior_numeric_walkthrough_png_bytes,
    render_posterior_math_markdown,
    render_posterior_math_png_bytes,
    render_probability_primitives_markdown,
    render_probability_primitives_png_bytes,
    write_method_survey_artifact,
    write_posterior_numeric_walkthrough_artifacts,
    write_posterior_math_artifacts,
    write_probability_primitives_artifacts,
)


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

    def test_posterior_math_artifacts_are_generated(self) -> None:
        markdown = render_posterior_math_markdown()
        png = render_posterior_math_png_bytes()

        self.assertIn("Posterior Update Math for Toy and Identity Benchmarks", markdown)
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))

        with tempfile.TemporaryDirectory() as temp_dir:
            markdown_path, png_path = write_posterior_math_artifacts(temp_dir)
            self.assertEqual(markdown_path, Path(temp_dir) / "posterior_update_math.md")
            self.assertEqual(png_path, Path(temp_dir) / "posterior_update_math.png")
            self.assertTrue(markdown_path.exists())
            self.assertTrue(png_path.exists())

    def test_posterior_numeric_walkthrough_artifacts_are_generated(self) -> None:
        markdown = render_posterior_numeric_walkthrough_markdown()
        png = render_posterior_numeric_walkthrough_png_bytes()

        self.assertIn("Posterior Equation Numeric Walkthrough", markdown)
        self.assertIn("Toy numeric substitution", markdown)
        self.assertIn("Identity numeric substitution", markdown)
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))

        with tempfile.TemporaryDirectory() as temp_dir:
            markdown_path, png_path = write_posterior_numeric_walkthrough_artifacts(temp_dir)
            self.assertEqual(markdown_path, Path(temp_dir) / "posterior_numeric_walkthrough.md")
            self.assertEqual(png_path, Path(temp_dir) / "posterior_numeric_walkthrough.png")
            self.assertTrue(markdown_path.exists())
            self.assertTrue(png_path.exists())

    def test_posterior_math_tex_and_build_script_exist(self) -> None:
        root = Path(__file__).resolve().parents[1]
        tex_path = root / "docs" / "surveys" / "posterior_update_math.tex"
        script_path = root / "scripts" / "build_posterior_math.sh"

        self.assertTrue(tex_path.exists())
        self.assertTrue(script_path.exists())
        self.assertIn("Posterior Update Math for Toy and Identity Benchmarks", tex_path.read_text(encoding="utf-8"))
        self.assertIn("latexmk", script_path.read_text(encoding="utf-8"))

    def test_probability_primitives_artifacts_are_generated(self) -> None:
        markdown = render_probability_primitives_markdown()
        png = render_probability_primitives_png_bytes()

        self.assertIn("Probability Primitive Charts", markdown)
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))

        with tempfile.TemporaryDirectory() as temp_dir:
            markdown_path, png_path = write_probability_primitives_artifacts(temp_dir)
            self.assertEqual(markdown_path, Path(temp_dir) / "probability_primitives.md")
            self.assertEqual(png_path, Path(temp_dir) / "probability_primitives.png")
            self.assertTrue(markdown_path.exists())
            self.assertTrue(png_path.exists())


if __name__ == "__main__":
    unittest.main()
