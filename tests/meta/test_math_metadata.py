from __future__ import annotations

import json
import os
import subprocess
import unittest
from pathlib import Path

import yaml


class MathMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cls.symbol_glossary_path = cls.root / "docs" / "math" / "symbol_glossary.tex"
        cls.equation_registry_path = cls.root / "docs" / "math" / "equation_registry.yaml"
        cls.crosswalk_path = cls.root / "docs" / "math" / "code_equation_crosswalk.md"
        cls.registry = yaml.safe_load(cls.equation_registry_path.read_text(encoding="utf-8"))

    def test_equation_registry_has_required_fields(self) -> None:
        self.assertIsInstance(self.registry, list)
        seen: set[str] = set()
        for row in self.registry:
            self.assertIn("id", row)
            self.assertIn("status", row)
            self.assertIn("latex", row)
            self.assertIn("symbols", row)
            self.assertIn("implementation", row)
            self.assertIn("artifacts", row)
            self.assertIn("tests", row)
            self.assertNotIn(row["id"], seen)
            seen.add(row["id"])
            self.assertIn(row["status"], {"implemented", "conceptual"})
            implementation = row["implementation"]
            self.assertIn("module", implementation)
            self.assertIn("function", implementation)

    def test_implemented_entries_point_to_existing_repo_files(self) -> None:
        for row in self.registry:
            if row["status"] != "implemented":
                continue
            module_path = self.root / row["implementation"]["module"]
            self.assertTrue(module_path.exists(), module_path)
            for test_path in row["tests"]:
                self.assertTrue((self.root / test_path).exists(), test_path)

    def test_symbol_glossary_mentions_core_symbols(self) -> None:
        text = self.symbol_glossary_path.read_text(encoding="utf-8")
        for symbol in (r"\(z_k\)", r"\(\phi_k\)", r"\(y_k\)", r"\(\ell_k(c)\)", r"\(p_k(c)\)", r"\(Q_{\text{static}}\)"):
            self.assertIn(symbol, text)

    def test_crosswalk_mentions_core_equations(self) -> None:
        text = self.crosswalk_path.read_text(encoding="utf-8")
        self.assertIn("Bayes recursive update", text)
        self.assertIn("CorpusGym reward", text)
        self.assertIn("Advanced-filter gate", text)

    def test_renderer_emits_generated_math_metadata_artifacts(self) -> None:
        script_path = self.root / "scripts" / "render" / "render_math_metadata.py"
        subprocess.run(
            ["python3", str(script_path)],
            check=True,
            cwd=self.root,
            env={**os.environ, "PYTHONPATH": str(self.root / "src")},
        )

        outputs = (
            self.root / "artifacts" / "latex" / "symbol_glossary.json",
            self.root / "artifacts" / "latex" / "symbol_glossary.md",
            self.root / "artifacts" / "latex" / "equation_registry.json",
            self.root / "artifacts" / "latex" / "equation_registry.md",
            self.root / "artifacts" / "latex" / "code_equation_crosswalk.md",
        )
        for path in outputs:
            self.assertTrue(path.exists(), path)
            self.assertGreater(path.stat().st_size, 0, path)

        payload = json.loads((self.root / "artifacts" / "latex" / "equation_registry.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(payload), 10)
        implemented = [row for row in payload if row["status"] == "implemented"]
        conceptual = [row for row in payload if row["status"] == "conceptual"]
        self.assertGreaterEqual(len(implemented), 8)
        self.assertEqual(len(conceptual), 0)


if __name__ == "__main__":
    unittest.main()
