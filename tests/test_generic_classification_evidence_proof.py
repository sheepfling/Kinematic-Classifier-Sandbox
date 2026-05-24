from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox import (
    EvidenceStep,
    analyze_generic_classification_evidence_proof,
    posterior_history_from_evidence_stream,
    write_generic_classification_evidence_proof_artifacts,
)


class GenericClassificationEvidenceProofTests(unittest.TestCase):
    def test_identical_evidence_streams_produce_identical_posteriors(self) -> None:
        stream = (
            EvidenceStep(time=0.0, log_likelihoods={"A": -0.2, "B": -1.2}),
            EvidenceStep(time=1.0, log_likelihoods={"A": -0.3, "B": -0.9}),
        )
        history_a = posterior_history_from_evidence_stream(
            class_names=("A", "B"),
            prior={"A": 0.5, "B": 0.5},
            evidence_stream=stream,
        )
        history_b = posterior_history_from_evidence_stream(
            class_names=("A", "B"),
            prior={"A": 0.5, "B": 0.5},
            evidence_stream=tuple(EvidenceStep(time=step.time, log_likelihoods=dict(step.log_likelihoods)) for step in stream),
        )
        self.assertEqual(history_a, history_b)

    def test_generic_classification_evidence_artifacts_are_generated(self) -> None:
        result = analyze_generic_classification_evidence_proof()

        self.assertTrue(result.evidence_provider_manifest)
        self.assertTrue(result.method_equivalence_tests)
        self.assertIn("Classification Evidence Proof", result.classification_principles_report)
        self.assertTrue(all(row["status"] == "pass" for row in result.method_equivalence_tests))

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_generic_classification_evidence_proof_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "classification_evidence_proof")
            self.assertTrue(artifacts.evidence_provider_manifest_path.exists())
            self.assertTrue(artifacts.method_equivalence_tests_path.exists())
            self.assertTrue(artifacts.classification_principles_report_path.exists())

            manifest = json.loads(artifacts.evidence_provider_manifest_path.read_text(encoding="utf-8"))
            provider_ids = [row["provider_id"] for row in manifest]
            self.assertIn("pointwise", provider_ids)
            self.assertIn("kalman_bank", provider_ids)

            tests = json.loads(artifacts.method_equivalence_tests_path.read_text(encoding="utf-8"))
            test_ids = [row["test_id"] for row in tests]
            self.assertIn("identical_likelihood_streams_imply_identical_posteriors", test_ids)
            self.assertIn("different_evidence_providers_share_artifact_shape", test_ids)

            report = artifacts.classification_principles_report_path.read_text(encoding="utf-8")
            self.assertIn("Evidence Providers", report)
            self.assertIn("Equivalence Tests", report)


if __name__ == "__main__":
    unittest.main()
