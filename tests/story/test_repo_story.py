from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.repo_story import (
    ARTIFACT_MANIFEST,
    CLAIMS,
    WITNESSES,
    render_proof_gallery,
    render_repo_story_index,
    render_story_index,
    render_team_packet_index,
    validate_repo_story_references,
    write_repo_story_artifacts,
)


class RepoStoryTests(unittest.TestCase):
    def test_static_repo_story_references_exist(self) -> None:
        validation = validate_repo_story_references()
        self.assertEqual(validation["status"], "pass", validation["missing"])
        self.assertEqual(validation["claim_count"], len(CLAIMS))
        self.assertGreaterEqual(validation["claim_count"], 8)
        self.assertEqual(validation["witness_count"], 6)
        self.assertGreaterEqual(validation["artifact_manifest_count"], 20)

    def test_claims_have_docs_artifacts_tests_and_limitations(self) -> None:
        for claim in CLAIMS:
            self.assertTrue(claim.evidence_doc, claim.claim_id)
            self.assertTrue(claim.artifact_paths, claim.claim_id)
            self.assertTrue(claim.test_paths, claim.claim_id)
            self.assertTrue(claim.limitations.strip(), claim.claim_id)
            self.assertTrue(claim.next_work.strip(), claim.claim_id)

    def test_witnesses_have_plot_and_table(self) -> None:
        for witness in WITNESSES:
            self.assertTrue(witness.key_plot.endswith(".png"), witness.witness)
            self.assertTrue(witness.key_table.endswith((".csv", ".json")), witness.witness)
            self.assertTrue(Path(witness.key_plot).exists(), witness.witness)
            self.assertTrue(Path(witness.key_table).exists(), witness.witness)

    def test_artifact_manifest_has_required_fields(self) -> None:
        for entry in ARTIFACT_MANIFEST:
            self.assertTrue(entry.path)
            self.assertTrue(entry.generated_by)
            self.assertTrue(entry.depends_on)
            self.assertTrue(entry.question_answered)
            self.assertTrue(entry.claim_supported)
            self.assertTrue(entry.status)
            self.assertTrue(entry.known_limitation)

    def test_write_repo_story_artifacts_generates_valid_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            docs_root = Path(temp_dir) / "docs"
            artifacts = write_repo_story_artifacts(temp_dir, docs_root=docs_root, write_showcase=True)
            self.assertTrue(artifacts.claim_matrix_path.exists())
            self.assertTrue(artifacts.artifact_manifest_path.exists())
            self.assertTrue(artifacts.artifact_graph_path.exists())
            self.assertTrue(artifacts.repo_layer_diagram_path.exists())
            self.assertTrue(artifacts.artifact_dependency_graph_path.exists())
            self.assertTrue((docs_root / "story" / "claim_evidence_matrix.md").exists())
            self.assertTrue((docs_root / "story" / "artifact_graph.md").exists())
            self.assertTrue((Path(temp_dir) / "showcase" / "proof_gallery.md").exists())
            self.assertTrue((Path(temp_dir) / "showcase" / "story_index.md").exists())
            self.assertTrue((Path(temp_dir) / "team_packet" / "index.md").exists())

            with artifacts.claim_matrix_path.open(encoding="utf-8", newline="") as handle:
                claims = list(csv.DictReader(handle))
            self.assertEqual(len(claims), len(CLAIMS))
            manifest = json.loads(artifacts.artifact_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(len(manifest), len(ARTIFACT_MANIFEST))

    def test_proof_gallery_uses_pln024_claim_story(self) -> None:
        text = render_proof_gallery()
        self.assertIn("## Claim 1: Corpus quality is evaluated before classifier claims.", text)
        self.assertIn("3D transition is a controlled lift, not a full rewrite.", text)
        self.assertIn("plots/corpus_adequacy_scorecard.png", text)
        self.assertIn("tables/advanced_filter_method_comparison.csv", text)

    def test_repo_story_indices_reference_new_method_surfaces(self) -> None:
        repo_index = render_repo_story_index()
        showcase_index = render_story_index()
        team_packet = render_team_packet_index()
        self.assertIn("artifacts/algorithm_coverage_matrix_v1/algorithm_coverage_matrix_report.md", repo_index)
        self.assertIn("artifacts/method_validation_os_v1/method_validation_os_report.md", repo_index)
        self.assertIn("artifacts/trajectory_exploration_backend_registry_v1/report.md", repo_index)
        self.assertIn("Tracked Method Surfaces", showcase_index)
        self.assertIn("trajectory_exploration_backend_registry_v1/report.md", showcase_index)
        self.assertIn("algorithm_coverage_matrix_v1/algorithm_coverage_matrix_report.md", team_packet)


if __name__ == "__main__":
    unittest.main()
