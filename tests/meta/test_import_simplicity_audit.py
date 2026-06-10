from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.meta.import_simplicity_audit import (
    analyze_import_simplicity,
    write_import_simplicity_audit_artifacts,
)


class ImportSimplicityAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]

    def test_import_simplicity_audit_reports_current_tree(self) -> None:
        result = analyze_import_simplicity()
        self.assertIn("passes", result.summary)
        self.assertGreaterEqual(result.summary["issue_count"], result.summary["violation_count"])
        self.assertIn("Import Simplicity Audit", result.report_markdown)

    def test_import_simplicity_audit_exposes_cleanup_debt(self) -> None:
        result = analyze_import_simplicity()
        self.assertTrue(result.summary["passes"])
        self.assertEqual(result.summary["violation_count"], 0)
        for key in (
            "root_wrapper_count",
        ):
            self.assertIn(key, result.summary)
            self.assertEqual(result.summary[key], 0, key)
        self.assertEqual(result.summary["debt_count"], 0)
        self.assertEqual(result.summary["accidental_export_count"], 0)
        self.assertEqual(result.summary["broad_package_surface_count"], 0)
        self.assertEqual(result.summary["import_cycle_count"], 0)

        kinds = {row["kind"] for row in result.issue_rows}
        self.assertNotIn("root_wrapper_surface", kinds)
        self.assertNotIn("accidental_export", kinds)
        self.assertNotIn("broad_package_surface", kinds)
        self.assertNotIn("import_cycle", kinds)

    def test_artifacts_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_import_simplicity_audit_artifacts(temp_dir)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "import_simplicity_audit_v1")
            for path in (artifacts.report_path, artifacts.summary_path, artifacts.issues_path):
                self.assertTrue(path.exists(), path)
            summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
            self.assertIn("passes", summary)

    def test_cli_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env = os.environ.copy()
            env["PYTHONPATH"] = "src"
            completed = subprocess.run(
                [
                    "python3",
                    "scripts/audit/audit_import_simplicity.py",
                    "--write-artifacts",
                    "--output-dir",
                    temp_dir,
                ],
                cwd=self.root,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("import_simplicity_audit_v1", completed.stdout)


if __name__ == "__main__":
    unittest.main()
