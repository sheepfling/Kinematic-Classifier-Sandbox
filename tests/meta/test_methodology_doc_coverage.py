from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path


class MethodologyDocCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cls.manifest_path = cls.root / "docs" / "surveys" / "methodology_doc_coverage.yaml"
        cls.payload = json.loads(cls.manifest_path.read_text(encoding="utf-8"))
        cls.modules = cls.payload["modules"]

    def test_every_src_module_is_accounted_for(self) -> None:
        src_modules = sorted(
            str(path.relative_to(self.root)).replace("\\", "/")
            for path in (self.root / "src" / "kinematic_classifier_sandbox").rglob("*.py")
        )
        manifest_modules = sorted(row["module_path"] for row in self.modules)
        coverage_debt = self.payload.get("coverage_debt", [])
        debt_modules = sorted(row["module_path"] for row in coverage_debt)
        self.assertEqual(set(manifest_modules).intersection(debt_modules), set())
        self.assertEqual(set(src_modules), set(manifest_modules).union(debt_modules))
        for row in coverage_debt:
            self.assertIn("module_path", row)
            self.assertIn("reason", row)
            self.assertIn("status", row)
            self.assertTrue(row["reason"])
            self.assertTrue(row["status"].endswith("_inventory"))

    def test_manifest_rows_have_required_fields(self) -> None:
        seen = set()
        for row in self.modules:
            self.assertIn("module_path", row)
            self.assertIn("primary_doc", row)
            self.assertIn("primary_section", row)
            self.assertIn("coverage_kind", row)
            self.assertIn("artifact_outputs", row)
            self.assertIn("status", row)
            self.assertNotIn(row["module_path"], seen)
            seen.add(row["module_path"])

    def test_build_scripts_smoke(self) -> None:
        scripts = (
            "scripts/build/build_posterior_math.sh",
            "scripts/build/build_methodology_evaluation_framework.sh",
            "scripts/build/build_classifier_ladder_and_contracts.sh",
            "scripts/build/build_corpus_generation_and_search.sh",
            "scripts/build/build_dimensional_lift_and_advanced_filter_gates.sh",
            "scripts/build/build_methodology_docs.sh",
        )
        for script in scripts:
            subprocess.run(["bash", str(self.root / script)], check=True, cwd=self.root)

    def test_generated_docs_mention_expected_subsystems(self) -> None:
        artifact_texts = {
            "posterior": (self.root / "artifacts" / "posterior_update_math.md").read_text(encoding="utf-8"),
            "evaluation": (self.root / "artifacts" / "methodology_evaluation_framework.md").read_text(encoding="utf-8"),
            "ladder": (self.root / "artifacts" / "classifier_ladder_and_contracts.md").read_text(encoding="utf-8"),
            "corpus": (self.root / "artifacts" / "corpus_generation_and_search.md").read_text(encoding="utf-8"),
            "dimensional": (self.root / "artifacts" / "dimensional_lift_and_advanced_filter_gates.md").read_text(encoding="utf-8"),
        }
        self.assertIn("run_class_bank", artifact_texts["posterior"])
        self.assertIn("run_identity_benchmark", artifact_texts["posterior"])
        self.assertIn("prior_sensitivity_analysis", artifact_texts["evaluation"])
        self.assertIn("feature_analysis", artifact_texts["evaluation"])
        self.assertIn("corpus_adequacy_audit", artifact_texts["evaluation"])
        self.assertIn("pointwise", artifact_texts["ladder"])
        self.assertIn("windowed", artifact_texts["ladder"])
        self.assertIn("accumulator", artifact_texts["ladder"])
        self.assertIn("kalman", artifact_texts["ladder"])
        self.assertIn("transition_matrix", artifact_texts["ladder"])
        self.assertIn("trajectory_generator", artifact_texts["corpus"])
        self.assertIn("corpus_autodevelopment", artifact_texts["corpus"])
        self.assertIn("study_candidate_generation", artifact_texts["corpus"])
        self.assertIn("dimensional_lift_audit", artifact_texts["dimensional"])
        self.assertIn("advanced_filter_decision", artifact_texts["dimensional"])
        self.assertIn("RBPF", artifact_texts["dimensional"])

    def test_generated_artifacts_exist_and_are_nonempty(self) -> None:
        outputs = (
            self.root / "artifacts" / "posterior_update_math.pdf",
            self.root / "artifacts" / "posterior_update_math.md",
            self.root / "artifacts" / "methodology_evaluation_framework.pdf",
            self.root / "artifacts" / "methodology_evaluation_framework.md",
            self.root / "artifacts" / "classifier_ladder_and_contracts.pdf",
            self.root / "artifacts" / "classifier_ladder_and_contracts.md",
            self.root / "artifacts" / "corpus_generation_and_search.pdf",
            self.root / "artifacts" / "corpus_generation_and_search.md",
            self.root / "artifacts" / "dimensional_lift_and_advanced_filter_gates.pdf",
            self.root / "artifacts" / "dimensional_lift_and_advanced_filter_gates.md",
            self.root / "artifacts" / "latex" / "methodology_doc_coverage.json",
            self.root / "artifacts" / "latex" / "methodology_doc_coverage.md",
        )
        for path in outputs:
            self.assertTrue(path.exists(), path)
            self.assertGreater(path.stat().st_size, 0, path)


if __name__ == "__main__":
    unittest.main()
