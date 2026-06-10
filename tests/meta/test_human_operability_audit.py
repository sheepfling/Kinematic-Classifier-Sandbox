from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from kinematic_classifier_sandbox.meta.human_operability_audit import (
    analyze_human_operability_audit,
    write_human_operability_audit_artifacts,
)
from kinematic_classifier_sandbox.story.repo_story import ArtifactManifestEntry, ClaimEvidence


class HumanOperabilityAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]

    def test_current_repo_passes_hard_gate_subset(self) -> None:
        result = analyze_human_operability_audit()
        self.assertEqual(result.summary["hard_fail_count"], 0, result.issue_rows[:10])
        self.assertTrue(any(row["surface_id"] == "repo_checks" for row in result.rerun_command_rows))

    def test_artifacts_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_human_operability_audit_artifacts(temp_dir)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "human_operability_audit_v1")
            for path in (
                artifacts.report_path,
                artifacts.summary_path,
                artifacts.issues_path,
                artifacts.front_door_coverage_path,
                artifacts.claim_traceability_path,
                artifacts.equation_graph_link_audit_path,
                artifacts.rerun_command_coverage_path,
            ):
                self.assertTrue(path.exists(), path)
            summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
            self.assertIn("hard_fail_count", summary)

    def test_missing_front_door_is_a_hard_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_minimal_repo(root, include_repo_story=False)
            with mock.patch(
                "kinematic_classifier_sandbox.meta.human_operability_audit.validate_repo_story_references",
                return_value={"status": "pass", "missing": (), "claim_count": 0, "witness_count": 0, "artifact_manifest_count": 0},
            ), mock.patch(
                "kinematic_classifier_sandbox.meta.human_operability_audit.CLAIMS",
                (),
            ), mock.patch(
                "kinematic_classifier_sandbox.meta.human_operability_audit.ARTIFACT_MANIFEST",
                (),
            ), mock.patch(
                "kinematic_classifier_sandbox.meta.human_operability_audit.load_equation_registry",
                return_value=[],
            ), mock.patch(
                "kinematic_classifier_sandbox.meta.human_operability_audit.analyze_repo_shape",
                return_value=mock.Mock(root_module_rows=()),
            ):
                result = analyze_human_operability_audit(source_root=root, output_dir=root / "artifacts")
            codes = {row["code"] for row in result.issue_rows if row["severity"] == "fail"}
            self.assertIn("missing_front_door", codes)

    def test_claim_rows_require_limitations_next_work_and_generating_code(self) -> None:
        bad_claim = ClaimEvidence(
            claim_id="CXX",
            claim="Synthetic claim",
            pillar="Audit",
            evidence_doc=("docs/story/00_repo_story.md",),
            artifact_paths=("artifacts/showcase/story_index.md",),
            test_paths=("tests/README.md",),
            current_status="draft",
            limitations="",
            next_work="",
            showcase_plot="plots/example.png",
            showcase_table="tables/example.csv",
            supporting_equation="eq",
        )
        manifest = (
            ArtifactManifestEntry(
                path="artifacts/showcase/story_index.md",
                generated_by="src/kinematic_classifier_sandbox/meta/human_operability_audit.py",
                depends_on=("docs/story/00_repo_story.md",),
                question_answered="q",
                claim_supported="CXX",
                status="implemented",
                known_limitation="l",
            ),
        )
        with mock.patch(
            "kinematic_classifier_sandbox.meta.human_operability_audit.CLAIMS",
            (bad_claim,),
        ), mock.patch(
            "kinematic_classifier_sandbox.meta.human_operability_audit.ARTIFACT_MANIFEST",
            manifest,
        ), mock.patch(
            "kinematic_classifier_sandbox.meta.human_operability_audit.validate_repo_story_references",
            return_value={"status": "pass", "missing": (), "claim_count": 1, "witness_count": 0, "artifact_manifest_count": 1},
        ):
            result = analyze_human_operability_audit()
        failing = [row for row in result.claim_traceability_rows if row["claim_id"] == "CXX"]
        self.assertEqual(len(failing), 1)
        self.assertEqual(failing[0]["status"], "fail")

    def test_cli_returns_success_on_warning_only_repo(self) -> None:
        completed = subprocess.run(
            ["python3", "scripts/audit/audit_human_operability.py"],
            cwd=self.root,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn("Human Operability Audit", completed.stdout)

    def _write_minimal_repo(self, root: Path, *, include_repo_story: bool) -> None:
        for rel_path, text in (
            ("docs/story/02_reading_order.md", "[repo](00_repo_story.md)\n"),
            ("docs/story/claim_evidence_matrix.md", "claim\n"),
            ("docs/story/document_roles.md", "roles\n"),
            ("src/kinematic_classifier_sandbox/README.md", "[story](../../docs/story/00_repo_story.md)\n"),
            ("tests/README.md", "[scripts](../scripts/README.md)\n"),
            ("scripts/README.md", "`python3 scripts/check.py`\n`python3 scripts/audit/audit_repo_shape.py`\n`python3 scripts/audit/validate_artifacts.py`\n`python3 scripts/audit/audit_corpus.py`\n`python3 scripts/audit/audit_dimensions.py`\n`python3 scripts/render/render_methodology_latex.py`\n"),
            ("docs/math/code_equation_crosswalk.md", "Bayes recursive update\nCorpusGym reward\nAdvanced-filter gate\n"),
            ("docs/latex/kinematic_classifier_methodology.tex", "\\section{Test}\n"),
            ("artifacts/showcase/story_index.md", "index\n"),
            ("artifacts/showcase/proof_gallery.md", "gallery\n"),
        ):
            path = root / rel_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        if include_repo_story:
            path = root / "docs/story/00_repo_story.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("[read](02_reading_order.md)\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
