from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.registry.corpus_evaluation_gap_matrix import (
    CAPABILITY_SPECS,
    analyze_corpus_evaluation_gap_matrix,
    write_corpus_evaluation_gap_matrix_artifacts,
)


class CorpusEvaluationGapMatrixTests(unittest.TestCase):
    def test_inventory_contains_required_capabilities(self) -> None:
        capability_ids = {spec.capability_id for spec in CAPABILITY_SPECS}

        self.assertEqual(len(capability_ids), len(CAPABILITY_SPECS))
        self.assertIn("corpus_adequacy_scoring", capability_ids)
        self.assertIn("feature_class_confusability", capability_ids)
        self.assertIn("selected_corpus_closed_loop_rerun", capability_ids)
        self.assertIn("arbitrary_given_corpus_evaluation", capability_ids)

    def test_static_audit_recognizes_current_support_boundary(self) -> None:
        result = analyze_corpus_evaluation_gap_matrix()
        by_capability = {row.capability_id: row for row in result.capability_rows}

        self.assertEqual(by_capability["corpus_adequacy_scoring"].current_status, "implemented")
        self.assertEqual(by_capability["selected_corpus_closed_loop_rerun"].current_status, "implemented")
        self.assertEqual(by_capability["arbitrary_given_corpus_evaluation"].current_status, "partial")
        self.assertTrue(by_capability["corpus_adequacy_scoring"].artifact_writer_callable)
        self.assertTrue(by_capability["classifier_support_coverage_report"].tests_exist)
        self.assertTrue(by_capability["feature_excitation_coverage"].markdown_docs_coherent)
        self.assertTrue(by_capability["corpus_adequacy_scoring"].latex_docs_coherent)

    def test_materialized_subset_observes_real_artifact_classes(self) -> None:
        result = analyze_corpus_evaluation_gap_matrix(
            capability_ids=(
                "corpus_adequacy_scoring",
                "feature_excitation_coverage",
                "classifier_support_coverage_report",
                "selected_corpus_closed_loop_rerun",
            ),
            materialize=True,
        )
        by_capability = {row.capability_id: row for row in result.capability_rows}

        self.assertIn("report", by_capability["corpus_adequacy_scoring"].observed_artifact_classes)
        self.assertIn("tabular", by_capability["feature_excitation_coverage"].observed_artifact_classes)
        self.assertIn("summary", by_capability["classifier_support_coverage_report"].observed_artifact_classes)
        self.assertIn("visual", by_capability["selected_corpus_closed_loop_rerun"].observed_artifact_classes)

    def test_artifact_bundle_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_corpus_evaluation_gap_matrix_artifacts(temp_dir)

            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "corpus_evaluation_gap_matrix_v1")
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.matrix_path.exists())
            self.assertTrue(artifacts.coherence_issues_path.exists())
            self.assertTrue(artifacts.inventory_path.exists())
            self.assertTrue(artifacts.status_plot_path.exists())

            summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["automatic_evaluation_verdict"], "yes for default/common-study and selected-corpus paths")
            self.assertEqual(summary["arbitrary_given_corpus_verdict"], "partial")

            report_text = artifacts.report_path.read_text(encoding="utf-8")
            self.assertIn("Corpus Evaluation Gap Matrix", report_text)
            self.assertIn("automatic evaluation", report_text)


if __name__ == "__main__":
    unittest.main()
