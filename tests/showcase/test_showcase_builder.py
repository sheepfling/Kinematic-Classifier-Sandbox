from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.methodology.context import build_methodology_execution_context
from kinematic_classifier_sandbox.showcase.builder import (
    build_showcase_artifacts,
    validate_showcase_artifacts,
)


class ShowcaseBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.methodology_context = build_methodology_execution_context(
            seed=7,
            trajectories_per_case=6,
            use_cache=True,
        )

    def test_showcase_packet_builds_and_validates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = build_showcase_artifacts(temp_dir, refresh=False, create_zip=True)

            self.assertEqual(artifacts.showcase_dir, Path(temp_dir) / "showcase")
            self.assertTrue(artifacts.index_path.exists())
            self.assertTrue(artifacts.proof_gallery_path.exists())
            self.assertTrue(artifacts.artifact_manifest_path.exists())
            self.assertTrue(artifacts.summary_metrics_path.exists())
            self.assertTrue(artifacts.validation_path.exists())
            self.assertTrue(artifacts.team_packet_dir.exists())
            self.assertIsNotNone(artifacts.zip_path)
            assert artifacts.zip_path is not None
            self.assertTrue(artifacts.zip_path.exists())

            manifest = json.loads(artifacts.artifact_manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(manifest["items"])
            kinds = {item["kind"] for item in manifest["items"]}
            self.assertTrue({"report", "plot", "table", "run_card"}.issubset(kinds))

            validation = validate_showcase_artifacts(artifacts.showcase_dir)
            self.assertEqual(validation.overall_status, "pass")
            self.assertTrue(validation.proof_gallery_complete)
            self.assertTrue(validation.proof_gallery_references_exist)
            self.assertTrue(validation.gallery_annotations_complete)
            self.assertTrue(validation.class_pair_identifiability_complete)
            self.assertTrue(validation.advanced_filter_go_no_go_present)
            self.assertTrue(validation.dimensional_status_present)

            proof_gallery_text = artifacts.proof_gallery_path.read_text(encoding="utf-8")
            self.assertIn("## Claim 1: Corpus quality is evaluated before classifier claims.", proof_gallery_text)
            self.assertIn("## Claim 8: 3D transition is a controlled lift, not a full rewrite.", proof_gallery_text)
            self.assertTrue((artifacts.showcase_dir / "story_index.md").exists())
            self.assertIn(
                "Study Candidate Evaluator",
                (artifacts.team_packet_dir / "index.md").read_text(encoding="utf-8"),
            )
            self.assertIn("plots/pointwise_vs_accumulator_posterior_timelines.png", proof_gallery_text)
            self.assertIn("plots/corpus_adequacy_scorecard.png", proof_gallery_text)
            self.assertIn("plots/dimension_lift_audit_chart.png", proof_gallery_text)
            self.assertIn("tables/advanced_filter_method_comparison.csv", proof_gallery_text)
            self.assertTrue((artifacts.showcase_dir / "plots" / "pointwise_vs_accumulator_posterior_timelines.png").exists())
            self.assertTrue((artifacts.showcase_dir / "plots" / "feature_correlation_heatmap.png").exists())
            self.assertTrue((artifacts.showcase_dir / "plots" / "kalman_innovation_likelihood_timeline.png").exists())
            self.assertTrue((artifacts.showcase_dir / "plots" / "generic_vs_1d_specific_layer_diagram.png").exists())
            self.assertTrue((artifacts.showcase_dir / "tables" / "advanced_filter_method_comparison.csv").exists())
            self.assertTrue((artifacts.showcase_dir / "tables" / "pf_rbpf_go_no_go_table.csv").exists())
            self.assertTrue((artifacts.showcase_dir / "tables" / "filter_trace_method_matrix.csv").exists())
            self.assertTrue((artifacts.showcase_dir / "tables" / "filter_trace_requirement_matrix.csv").exists())
            algorithm_report = (artifacts.showcase_dir / "reports" / "03_algorithm_ladder.md").read_text(encoding="utf-8")
            self.assertIn("capability-aware", algorithm_report)
            self.assertIn("witness_only", algorithm_report)
            filtering_report = (artifacts.showcase_dir / "reports" / "05_filtering_taxonomy.md").read_text(encoding="utf-8")
            self.assertIn("witness-specific promotions", filtering_report)
            self.assertIn("shared-corpus benchmark", filtering_report)
            self.assertIn("trace-validation packet", filtering_report)

    def test_showcase_fast_mode_accepts_precomputed_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = build_showcase_artifacts(
                temp_dir,
                refresh=False,
                create_zip=False,
                methodology_context=self.methodology_context,
                artifact_mode="fast",
            )
            self.assertTrue(artifacts.index_path.exists())
            self.assertTrue(artifacts.proof_gallery_path.exists())


if __name__ == "__main__":
    unittest.main()
