from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.meta.repo_shape_audit import (
    analyze_repo_shape,
    write_repo_shape_audit_artifacts,
)


class RepoShapeAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]

    def _remove_local_bytecode_caches(self) -> None:
        for base_name in ("src", "docs", "tests", "scripts", "experiments", "templates"):
            base = self.root / base_name
            if not base.exists():
                continue
            for path in sorted(base.rglob("__pycache__"), reverse=True):
                for child in path.rglob("*"):
                    if child.is_file():
                        child.unlink()
                for child in sorted(path.rglob("*"), reverse=True):
                    if child.is_dir():
                        child.rmdir()
                path.rmdir()

    def test_repo_shape_audit_passes_current_tree(self) -> None:
        self._remove_local_bytecode_caches()
        result = analyze_repo_shape()
        self.assertTrue(result.summary["passes"], result.issue_rows[:5])
        self.assertEqual(result.summary["duplicate_script_count"], 0)
        self.assertEqual(result.summary["generated_cruft_count"], 0)
        self.assertGreater(result.summary["root_module_count"], 0)
        self.assertGreater(result.summary["oversized_module_count"], 0)

    def test_artifacts_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_repo_shape_audit_artifacts(temp_dir)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "repo_shape_audit_v1")
            for path in (
                artifacts.report_path,
                artifacts.summary_path,
                artifacts.root_module_inventory_path,
                artifacts.duplicate_module_inventory_path,
                artifacts.duplicate_script_inventory_path,
                artifacts.generated_cruft_inventory_path,
                artifacts.oversized_module_inventory_path,
                artifacts.issues_path,
            ):
                self.assertTrue(path.exists(), path)
            summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
            self.assertIn("passes", summary)

    def test_cli_returns_success(self) -> None:
        self._remove_local_bytecode_caches()
        completed = subprocess.run(
            ["python3", "scripts/audit/audit_repo_shape.py"],
            cwd=self.root,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn("Repo Shape Audit", completed.stdout)


if __name__ == "__main__":
    unittest.main()
