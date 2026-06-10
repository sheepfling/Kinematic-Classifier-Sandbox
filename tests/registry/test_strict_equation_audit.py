from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.registry.strict_equation_audit import analyze_strict_equation_audit


class StrictEquationAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]

    def test_audit_labels_every_equation_strictly(self) -> None:
        result = analyze_strict_equation_audit()
        self.assertEqual(result.summary["equation_count"], 23)
        self.assertEqual(result.summary["implemented_count"], 23)
        self.assertEqual(result.summary["illustrative_count"], 0)
        self.assertEqual(result.summary["missing_count"], 0)
        self.assertEqual(len(result.rows), 23)
        self.assertTrue(all(row.exact_artifacts for row in result.rows))
        self.assertTrue(all(row.source_data for row in result.rows))

    def test_report_mentions_strict_labels(self) -> None:
        result = analyze_strict_equation_audit()
        self.assertIn("Formal Math Strict Audit", result.report_markdown)
        self.assertIn("illustrative", result.report_markdown)
        self.assertIn("missing", result.report_markdown)

    def test_writer_and_cli_emit_artifacts(self) -> None:
        script_path = self.root / "scripts" / "render" / "render_strict_equation_audit.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.run(
                ["python3", str(script_path), "--output-dir", temp_dir],
                check=True,
                cwd=self.root,
                env={**os.environ, "PYTHONPATH": str(self.root / "src")},
            )
            outputs = (
                Path(temp_dir) / "formal_math_strict_audit_v1" / "formal_math_strict_audit_report.md",
                Path(temp_dir) / "formal_math_strict_audit_v1" / "formal_math_strict_audit_summary.json",
                Path(temp_dir) / "formal_math_strict_audit_v1" / "formal_math_strict_audit.csv",
                Path(temp_dir) / "formal_math_strict_audit_v1" / "formal_math_strict_audit_status.png",
            )
            for path in outputs:
                self.assertTrue(path.exists(), path)
                self.assertGreater(path.stat().st_size, 0, path)

            module_run = subprocess.run(
                [
                    "python3",
                    "-m",
                    "kinematic_classifier_sandbox",
                    "strict-equation-audit",
                    "--output-dir",
                    temp_dir,
                ],
                check=True,
                cwd=self.root,
                env={**os.environ, "PYTHONPATH": str(self.root / "src")},
                capture_output=True,
                text=True,
            )
            self.assertIn("formal_math_strict_audit_v1", module_run.stdout)


if __name__ == "__main__":
    unittest.main()
