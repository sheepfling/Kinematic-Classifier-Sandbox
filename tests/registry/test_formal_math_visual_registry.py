from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

from kinematic_classifier_sandbox.formal_math_visual_registry import (
    analyze_formal_math_visual_registry,
)


class FormalMathVisualRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cls.equation_registry_path = cls.root / "docs" / "math" / "equation_registry.yaml"
        cls.equations = yaml.safe_load(cls.equation_registry_path.read_text(encoding="utf-8"))

    def test_every_implemented_equation_has_a_visual(self) -> None:
        result = analyze_formal_math_visual_registry()
        visual_ids = {row.equation_id for row in result.rows}
        implemented_ids = {row["id"] for row in self.equations if row["status"] == "implemented"}
        self.assertTrue(implemented_ids.issubset(visual_ids))

    def test_report_mentions_gallery_surface(self) -> None:
        result = analyze_formal_math_visual_registry()
        self.assertIn("Formal Math Visual Registry", result.report_markdown)
        self.assertIn("Gallery", result.report_markdown)
        self.assertIn("Provenance", result.report_markdown)
        self.assertIn("rerun command", result.report_markdown.lower())

    def test_registry_uses_explicit_provenance(self) -> None:
        result = analyze_formal_math_visual_registry()
        self.assertEqual(len(result.rows), len(self.equations))
        self.assertTrue(all(row.source_data_artifacts for row in result.rows))
        self.assertTrue(all(row.rerun_command for row in result.rows))
        self.assertTrue(all(row.visual_status in {"implemented", "illustrative", "missing"} for row in result.rows))
        self.assertFalse(any(row.visual_kind == "fallback" for row in result.rows))

    def test_writer_emits_artifacts(self) -> None:
        script_path = self.root / "scripts" / "render" / "render_formal_math_visual_registry.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.run(
                ["python3", str(script_path), "--output-dir", temp_dir],
                check=True,
                cwd=self.root,
            )

            outputs = (
                Path(temp_dir) / "formal_math_visual_registry_v1" / "formal_math_visual_registry_report.md",
                Path(temp_dir) / "formal_math_visual_registry_v1" / "formal_math_visual_registry_summary.json",
                Path(temp_dir) / "formal_math_visual_registry_v1" / "formal_math_visual_registry.csv",
                Path(temp_dir) / "formal_math_visual_registry_v1" / "formal_math_visual_registry_provenance.csv",
                Path(temp_dir) / "formal_math_visual_registry_v1" / "formal_math_visual_registry_runbook.md",
                Path(temp_dir) / "formal_math_visual_registry_v1" / "formal_math_visual_registry_coverage.png",
                Path(temp_dir) / "formal_math_visual_registry_v1" / "assets",
            )
            for path in outputs:
                self.assertTrue(path.exists(), path)
            self.assertGreater(len(list(outputs[-1].glob("*.png"))), 0)


if __name__ == "__main__":
    unittest.main()
