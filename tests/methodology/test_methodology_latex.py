from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.methodology_latex import (
    analyze_methodology_latex,
    write_methodology_latex_artifacts,
)


class MethodologyLatexTests(unittest.TestCase):
    def test_methodology_latex_analysis_emits_core_tables(self) -> None:
        result = analyze_methodology_latex(seed=7, trajectories_per_case=6)

        self.assertGreater(len(result.toy_problem_rows), 0)
        self.assertGreater(len(result.algorithm_ladder_rows), 0)
        self.assertGreater(len(result.bayesian_table_rows), 0)
        self.assertIn("Kinematic Classifier Methodology", result.methodology_tex)
        self.assertIn("Bayesian Evidence Model", result.methodology_tex)
        self.assertIn("Corpus Adequacy-Driven Corpus Synthesis", result.methodology_tex)
        self.assertIn("Filtering Taxonomy and Advanced-Method Gates", result.methodology_tex)
        self.assertIn("Algorithm ladder proof summary", result.algorithm_ladder_table_tex)
        self.assertIn("Representative Bayesian walkthrough steps", result.bayesian_update_walkthrough_table_tex)
        self.assertIn("Witness problems used to prove", result.toy_problem_summary_table_tex)
        self.assertIn(r"\begin{equation}", result.methodology_tex)
        self.assertIn(r"\begin{enumerate}", result.corpus_synthesis_algorithm_tex)

    def test_methodology_latex_artifacts_write_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_methodology_latex_artifacts(
                temp_dir,
                result=analyze_methodology_latex(seed=7, trajectories_per_case=6),
                build_pdf=False,
            )
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "latex")
            self.assertTrue(artifacts.source_tex_path.exists())
            self.assertTrue(artifacts.artifact_tex_path.exists())
            self.assertTrue(artifacts.algorithm_ladder_csv_path.exists())
            self.assertTrue(artifacts.toy_problem_summary_csv_path.exists())
            self.assertTrue(artifacts.corpus_synthesis_algorithm_path.exists())
            self.assertTrue(artifacts.algorithm_ladder_table_path.exists())
            self.assertTrue(artifacts.bayesian_update_walkthrough_table_path.exists())
            self.assertTrue(artifacts.toy_problem_summary_table_path.exists())
            self.assertTrue(artifacts.study_candidate_generation_algorithm_path.exists())
            self.assertIsNone(artifacts.pdf_path)

            tex_text = artifacts.artifact_tex_path.read_text(encoding="utf-8")
            self.assertIn("\\section{Algorithm Ladder}", tex_text)
            self.assertIn("\\input{tables/algorithm_ladder_table.tex}", tex_text)
            self.assertIn("\\input{tables/corpus_synthesis_algorithm.tex}", tex_text)

            self.assertTrue((Path.cwd() / "docs" / "latex" / "figures").exists())
            self.assertTrue((Path.cwd() / "docs" / "latex" / "tables").exists())
            self.assertTrue((artifacts.run_dir / "figures").exists())

    def test_methodology_latex_can_build_pdf_when_latexmk_is_available(self) -> None:
        if shutil.which("latexmk") is None:
            self.skipTest("latexmk not available")
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                artifacts = write_methodology_latex_artifacts(
                    temp_dir,
                    result=analyze_methodology_latex(seed=7, trajectories_per_case=6),
                    build_pdf=True,
                )
            except subprocess.CalledProcessError as exc:
                self.skipTest(f"latexmk failed: {exc}")
            self.assertIsNotNone(artifacts.pdf_path)
            assert artifacts.pdf_path is not None
            self.assertTrue(artifacts.pdf_path.exists())

    def test_methodology_build_script_exists(self) -> None:
        root = Path(__file__).resolve().parents[2]
        script_path = root / "src" / "kinematic_classifier_sandbox" / "meta" / "__init__.py"
        self.assertTrue(script_path.exists())
        script_text = script_path.read_text(encoding="utf-8")
        self.assertIn("write_methodology_latex_artifacts", script_text)


if __name__ == "__main__":
    unittest.main()
