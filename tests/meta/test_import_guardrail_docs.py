from __future__ import annotations

import unittest
from pathlib import Path


class ImportGuardrailDocsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]

    def test_agent_instructions_contain_import_surface_policy(self) -> None:
        text = (self.root / "AGENTS.md").read_text(encoding="utf-8")

        for required in (
            "Import and Package Surface Policy",
            "owning module directly",
            "package-root compatibility wrappers",
            "Do not mutate `sys.path`",
            "dynamic `__all__`",
            "audit_import_simplicity.py --strict",
        ):
            self.assertIn(required, text)

    def test_package_readme_rejects_root_wrappers_and_broad_reexports(self) -> None:
        text = (self.root / "src" / "kinematic_classifier_sandbox" / "README.md").read_text(
            encoding="utf-8"
        )

        for required in (
            "Root-level compatibility wrappers are not allowed",
            "import it from that owning module",
            "Do not add new files like",
            "broad convenience reexports",
            "audit_import_simplicity.py --strict",
        ):
            self.assertIn(required, text)

    def test_scripts_readme_documents_explicit_script_import_contract(self) -> None:
        text = (self.root / "scripts" / "README.md").read_text(encoding="utf-8")

        for required in (
            "Scripts do not mutate `sys.path`",
            "PYTHONPATH=src",
            "Internal code imports concrete owner modules",
            "Do not add script-local `sys.path.insert(...)` bootstraps",
            "audit_import_simplicity.py --strict",
        ):
            self.assertIn(required, text)

    def test_check_script_runs_import_simplicity_in_strict_mode(self) -> None:
        text = (self.root / "scripts" / "check.py").read_text(encoding="utf-8")

        self.assertIn("scripts/audit/audit_import_simplicity.py", text)
        self.assertIn("--strict", text)
        self.assertIn("--write-artifacts", text)


if __name__ == "__main__":
    unittest.main()
