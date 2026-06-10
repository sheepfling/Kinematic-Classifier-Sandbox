from __future__ import annotations

import ast
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

from kinematic_classifier_sandbox.registry.formal_math_registry import analyze_formal_math_registry


class FormalMathRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cls.src_dir = cls.root / "src" / "kinematic_classifier_sandbox"
        cls.equation_registry_path = cls.root / "docs" / "math" / "equation_registry.yaml"
        cls.equations = yaml.safe_load(cls.equation_registry_path.read_text(encoding="utf-8"))

    def _scan_source_functions(self) -> set[tuple[str, str]]:
        rows: set[tuple[str, str]] = set()
        for path in sorted(self.src_dir.rglob("*.py")):
            module_path = str(path.relative_to(self.root)).replace("\\", "/")
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    rows.add((module_path, node.name))
        return rows

    def test_registry_matches_source_tree(self) -> None:
        result = analyze_formal_math_registry()
        registry_rows = {(row.module_path, row.function_name) for row in result.function_rows}
        self.assertEqual(self._scan_source_functions(), registry_rows)

    def test_equations_link_to_registry_functions(self) -> None:
        result = analyze_formal_math_registry()
        function_rows = {(row.module_path, row.function_name): row for row in result.function_rows}
        for equation in self.equations:
            implementation = equation["implementation"]
            key = (implementation["module"], implementation["function"])
            if equation["status"] == "implemented":
                self.assertIn(key, function_rows)
                self.assertIn(equation["id"], function_rows[key].equation_ids)

    def test_report_mentions_registry_surface(self) -> None:
        result = analyze_formal_math_registry()
        self.assertIn("Formal Math Registry", result.report_markdown)
        self.assertIn("equation registry", result.report_markdown.lower())
        self.assertIn("function registry", result.report_markdown.lower())

    def test_writer_emits_artifacts(self) -> None:
        script_path = self.root / "scripts" / "render" / "render_formal_math_registry.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.run(
                ["python3", str(script_path), "--output-dir", temp_dir],
                check=True,
                cwd=self.root,
                env={**os.environ, "PYTHONPATH": str(self.root / "src")},
            )

            outputs = (
                Path(temp_dir) / "formal_math_registry_v1" / "formal_math_registry_report.md",
                Path(temp_dir) / "formal_math_registry_v1" / "formal_math_registry_summary.json",
                Path(temp_dir) / "formal_math_registry_v1" / "function_registry.csv",
                Path(temp_dir) / "formal_math_registry_v1" / "equation_registry.csv",
                Path(temp_dir) / "formal_math_registry_v1" / "function_equation_crosswalk.csv",
                Path(temp_dir) / "formal_math_registry_v1" / "formal_math_registry_role_counts.png",
            )
            for path in outputs:
                self.assertTrue(path.exists(), path)
                self.assertGreater(path.stat().st_size, 0, path)


if __name__ == "__main__":
    unittest.main()
