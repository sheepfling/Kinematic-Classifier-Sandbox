from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

from kinematic_classifier_sandbox.methodology.context import build_methodology_execution_context
from kinematic_classifier_sandbox.methodology import latex as methodology_latex_module
from kinematic_classifier_sandbox.methodology.latex import (
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
        self.assertGreater(len(result.algorithm_lane_rows), 0)
        self.assertGreater(len(result.bayesian_table_rows), 0)
        self.assertIn("Kinematic Classifier Methodology", result.methodology_tex)
        self.assertIn("Bayesian Evidence Model", result.methodology_tex)
        self.assertIn("Corpus Adequacy-Driven Corpus Synthesis", result.methodology_tex)
        self.assertIn("Filtering Taxonomy and Advanced-Method Gates", result.methodology_tex)
        self.assertIn("Algorithm ladder proof summary", result.algorithm_ladder_table_tex)
        self.assertIn("Broader algorithm lane map for the repository", result.algorithm_lane_table_tex)
        self.assertIn("Representative Bayesian walkthrough steps", result.bayesian_update_walkthrough_table_tex)
        self.assertIn("Witness problems used to prove", result.toy_problem_summary_table_tex)
        self.assertIn("particle\\_filter\\_bank", result.algorithm_ladder_table_tex)
        self.assertIn("rbpf", result.algorithm_ladder_table_tex)
        self.assertIn("learning\\_evidence", result.algorithm_lane_table_tex)
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
            self.assertTrue(artifacts.algorithm_lane_csv_path.exists())
            self.assertTrue(artifacts.toy_problem_summary_csv_path.exists())
            self.assertTrue(artifacts.corpus_synthesis_algorithm_path.exists())
            self.assertTrue(artifacts.algorithm_ladder_table_path.exists())
            self.assertTrue(artifacts.algorithm_lane_table_path.exists())
            self.assertTrue(artifacts.bayesian_update_walkthrough_table_path.exists())
            self.assertTrue(artifacts.toy_problem_summary_table_path.exists())
            self.assertTrue(artifacts.study_candidate_generation_algorithm_path.exists())
            self.assertIsNone(artifacts.pdf_path)

            tex_text = artifacts.artifact_tex_path.read_text(encoding="utf-8")
            self.assertIn("\\section{Stage V Evaluate: Algorithm Ladder}", tex_text)
            self.assertIn("\\input{tables/algorithm_ladder_table.tex}", tex_text)
            self.assertIn("\\input{tables/algorithm_lane_table.tex}", tex_text)
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

    def test_methodology_latex_pickled_cache_reuses_result_after_process_local_clear(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            previous = os.environ.get("KINEMATIC_CLASSIFIER_RUNTIME_ROOT")
            os.environ["KINEMATIC_CLASSIFIER_RUNTIME_ROOT"] = temp_dir
            try:
                methodology_latex_module._METHODOLOGY_LATEX_CACHE.clear()
                first = analyze_methodology_latex(seed=7, trajectories_per_case=6, use_cache=True)
                methodology_latex_module._METHODOLOGY_LATEX_CACHE.clear()
                with patch(
                    "kinematic_classifier_sandbox.methodology.context.build_methodology_execution_context",
                    side_effect=AssertionError("pickled methodology cache should satisfy second call"),
                ), patch(
                    "kinematic_classifier_sandbox.witnesses.toy_1d.bayesian_walkthroughs.analyze_bayesian_walkthroughs",
                    side_effect=AssertionError("pickled methodology cache should satisfy second call"),
                ), patch(
                    "kinematic_classifier_sandbox.inference.transition_matrix_accumulator.run_transition_benchmark",
                    side_effect=AssertionError("pickled methodology cache should satisfy second call"),
                ), patch(
                    "kinematic_classifier_sandbox.validation.advanced_filter_decision.analyze_advanced_filter_decision",
                    side_effect=AssertionError("pickled methodology cache should satisfy second call"),
                ):
                    second = analyze_methodology_latex(seed=7, trajectories_per_case=6, use_cache=True)
            finally:
                methodology_latex_module._METHODOLOGY_LATEX_CACHE.clear()
                if previous is None:
                    os.environ.pop("KINEMATIC_CLASSIFIER_RUNTIME_ROOT", None)
                else:
                    os.environ["KINEMATIC_CLASSIFIER_RUNTIME_ROOT"] = previous
        self.assertEqual(first.methodology_tex, second.methodology_tex)

    def test_methodology_build_script_exists(self) -> None:
        root = Path(__file__).resolve().parents[2]
        script_path = root / "src" / "kinematic_classifier_sandbox" / "meta" / "__init__.py"
        self.assertTrue(script_path.exists())
        script_text = script_path.read_text(encoding="utf-8")
        self.assertIn("write_methodology_latex_artifacts", script_text)

    def test_main_cli_exposes_methodology_latex_command(self) -> None:
        root = Path(__file__).resolve().parents[2]
        main_path = root / "src" / "kinematic_classifier_sandbox" / "__main__.py"
        main_text = main_path.read_text(encoding="utf-8")
        self.assertIn('"methodology-latex"', main_text)
        self.assertIn("write_methodology_latex_artifacts", main_text)

    def test_narrow_rerun_commands_work_via_subprocess(self) -> None:
        root = Path(__file__).resolve().parents[2]
        env = {**os.environ, "PYTHONPATH": str(root / "src")}
        with tempfile.TemporaryDirectory() as temp_dir:
            latex_run = subprocess.run(
                [
                    "python3",
                    "scripts/render/render_methodology_latex.py",
                    "--fast",
                    "--output-dir",
                    temp_dir,
                ],
                cwd=root,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn(str(Path(temp_dir) / "latex"), latex_run.stdout)
            self.assertTrue((Path(temp_dir) / "latex" / "kinematic_classifier_methodology.tex").exists())

            symbol_run = subprocess.run(
                [
                    "python3",
                    "scripts/render/render_methodology_section_symbol_audit.py",
                    "--output-dir",
                    temp_dir,
                ],
                cwd=root,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn(str(Path(temp_dir) / "methodology_section_symbol_audit"), symbol_run.stdout)
            self.assertTrue(
                (
                    Path(temp_dir)
                    / "methodology_section_symbol_audit"
                    / "methodology_section_symbol_audit_summary.json"
                ).exists()
            )

            front_door_run = subprocess.run(
                [
                    "python3",
                    "scripts/export_artifacts.py",
                    "--scope",
                    "front-door",
                    "--fast",
                    "--output-dir",
                    temp_dir,
                ],
                cwd=root,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn(str(Path(temp_dir) / "feature_analysis_v1"), front_door_run.stdout)
            self.assertIn(str(Path(temp_dir) / "latex"), front_door_run.stdout)
            self.assertIn(str(Path(temp_dir) / "repo_story"), front_door_run.stdout)
            self.assertIn("analysis_cache[front-door]:", front_door_run.stdout)
            self.assertIn("hits=", front_door_run.stdout)
            self.assertIn("misses=", front_door_run.stdout)
            self.assertTrue((Path(temp_dir) / "coverage_report_v1" / "coverage_report.md").exists())
            self.assertTrue((Path(temp_dir) / "latex" / "kinematic_classifier_methodology.tex").exists())
            self.assertTrue((Path(temp_dir) / "repo_story" / "artifact_manifest.json").exists())
            self.assertFalse((Path(temp_dir) / "showcase").exists())


if __name__ == "__main__":
    unittest.main()
