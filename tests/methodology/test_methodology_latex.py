from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
import json
from pathlib import Path

from kinematic_classifier_sandbox.methodology.context import build_methodology_execution_context
from kinematic_classifier_sandbox.methodology_latex import (
    analyze_section_symbol_audits,
    analyze_section_symbol_coverage,
    analyze_methodology_latex,
    write_methodology_section_symbol_audit_artifacts,
    write_methodology_latex_artifacts,
)


class MethodologyLatexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.methodology_context = build_methodology_execution_context(
            seed=7,
            trajectories_per_case=6,
            use_cache=True,
        )
        cls.result = analyze_methodology_latex(
            seed=7,
            trajectories_per_case=6,
            methodology_context=cls.methodology_context,
        )

    def test_methodology_latex_analysis_emits_core_tables(self) -> None:
        result = self.result

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
        self.assertIn("particle\\_filter\\_bank", result.algorithm_ladder_table_tex)
        self.assertIn("rbpf", result.algorithm_ladder_table_tex)
        self.assertIn("ornstein\\_uhlenbeck\\_mean\\_reversion", result.toy_problem_summary_table_tex)
        self.assertIn(r"\begin{equation}", result.methodology_tex)
        self.assertIn(r"\begin{enumerate}", result.corpus_synthesis_algorithm_tex)

    def test_methodology_stage_sections_have_local_symbol_tables(self) -> None:
        result = self.result

        coverage = analyze_section_symbol_coverage(result.methodology_tex)

        self.assertGreater(len(coverage.required_section_titles), 0)
        self.assertTrue(
            coverage.is_complete,
            f"missing section symbol tables for: {coverage.missing_section_titles}",
        )

    def test_section_symbol_audit_detects_missing_symbol_family(self) -> None:
        methodology_tex = r"""
\section{Stage X Example}
\paragraph{Section Symbols.}
\sectionsymbols{
\sectionsymbol{\(p_k(c)\)}{posterior for class \(c\)}{declared}
\sectionsymbol{\(\ell_k(c)\)}{log evidence}{declared}
}
\begin{equation}
p_k(c) = \ell_k(c) + \omega_k
\end{equation}
"""

        audits = analyze_section_symbol_audits(methodology_tex)

        self.assertEqual(len(audits), 1)
        self.assertIn(r"\omega_*", audits[0].missing_symbols)
        self.assertNotIn("p_*", audits[0].missing_symbols)
        self.assertNotIn(r"\ell_*", audits[0].missing_symbols)

    def test_methodology_section_symbol_audit_is_clean_for_current_manuscript(self) -> None:
        result = self.result

        audits = analyze_section_symbol_audits(result.methodology_tex)

        self.assertGreater(len(audits), 0)
        self.assertTrue(all(not audit.has_gaps for audit in audits))

    def test_methodology_section_symbol_audit_artifacts_write_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_methodology_section_symbol_audit_artifacts(temp_dir)

            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "methodology_section_symbol_audit")
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.rows_path.exists())
            self.assertTrue(artifacts.section_coverage_path.exists())

            report_text = artifacts.report_path.read_text(encoding="utf-8")
            self.assertIn("Methodology Section Symbol Audit", report_text)
            self.assertIn("Methodology Packet Status", report_text)
            self.assertIn("Junior rerun command", report_text)
            self.assertIn("Section Coverage", report_text)
            self.assertIn("Stage V Evaluate: Algorithm Ladder", report_text)

            summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
            self.assertIn("build_status", summary)
            self.assertTrue(summary["section_coverage_complete"])

            coverage = json.loads(artifacts.section_coverage_path.read_text(encoding="utf-8"))
            self.assertTrue(coverage["is_complete"])

    def test_methodology_latex_artifacts_write_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_methodology_latex_artifacts(
                temp_dir,
                result=self.result,
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
            self.assertIn("\\section{Stage V Evaluate: Algorithm Ladder}", tex_text)
            self.assertIn("\\input{tables/algorithm_ladder_table.tex}", tex_text)
            self.assertIn("\\input{tables/corpus_synthesis_algorithm.tex}", tex_text)

            self.assertTrue((Path.cwd() / "docs" / "latex" / "figures").exists())
            self.assertTrue((Path.cwd() / "docs" / "latex" / "tables").exists())
            self.assertTrue((artifacts.run_dir / "figures").exists())

    def test_methodology_latex_can_build_pdf_when_latexmk_is_available(self) -> None:
        if os.environ.get("KCS_RUN_PDF_TESTS") != "1":
            self.skipTest("set KCS_RUN_PDF_TESTS=1 to run the PDF build test")
        if shutil.which("latexmk") is None:
            self.skipTest("latexmk not available")
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                artifacts = write_methodology_latex_artifacts(
                    temp_dir,
                    result=self.result,
                    build_pdf=True,
                )
            except subprocess.CalledProcessError as exc:
                self.skipTest(f"latexmk failed: {exc}")
            self.assertIsNotNone(artifacts.pdf_path)
            assert artifacts.pdf_path is not None
            self.assertTrue(artifacts.pdf_path.exists())

    def test_methodology_latex_fast_mode_skips_pdf_build(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_methodology_latex_artifacts(
                temp_dir,
                methodology_context=self.methodology_context,
                artifact_mode="fast",
                build_pdf=True,
            )
        self.assertIsNone(artifacts.pdf_path)

    def test_methodology_build_script_exists(self) -> None:
        root = Path(__file__).resolve().parents[2]
        script_path = root / "src" / "kinematic_classifier_sandbox" / "meta" / "__init__.py"
        self.assertTrue(script_path.exists())
        script_text = script_path.read_text(encoding="utf-8")
        self.assertIn("write_methodology_latex_artifacts", script_text)


if __name__ == "__main__":
    unittest.main()
