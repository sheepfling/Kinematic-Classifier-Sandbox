from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kinematic_classifier_sandbox.common_experiment.runner import analyze_common_experiment
from kinematic_classifier_sandbox.corpus.autodevelopment import analyze_corpus_autodevelopment
from kinematic_classifier_sandbox.validation.study_candidate.generation import (
    analyze_study_candidate_generation,
    write_study_candidate_generation_artifacts,
)
from kinematic_classifier_sandbox.validation.study_candidate.protocol import (
    analyze_study_candidate_protocol,
)


class StudyCandidateGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol_result = analyze_study_candidate_protocol()
        cls.common_result = analyze_common_experiment(seed=7, trajectories_per_case=6)
        cls.corpus_result = analyze_corpus_autodevelopment(seed=7)
        cls.result = analyze_study_candidate_generation(
            seed=7,
            trajectories_per_case=6,
            protocol_result=cls.protocol_result,
            common_result=cls.common_result,
            corpus_result=cls.corpus_result,
        )

    def test_study_candidates_are_generated_and_scored(self) -> None:
        result = self.result

        self.assertGreater(len(result.generated_candidates), 0)
        self.assertGreater(len(result.static_score_rows), 0)
        self.assertGreater(len(result.monte_carlo_score_rows), 0)
        self.assertGreater(len(result.feature_evidence_rows), 0)
        self.assertGreater(len(result.prior_sensitivity_explanation_rows), 0)
        self.assertGreater(len(result.promoted_rows), 0)
        self.assertGreater(len(result.rejected_rows), 0)
        self.assertIn("generated_candidates", json.dumps({"generated_candidates": list(result.generated_candidates)}))

        required = set(result.schema["required"])
        for candidate in result.generated_candidates[:10]:
            self.assertTrue(required.issubset(candidate.keys()))
            self.assertIn(candidate["prior_spec"]["prior_ids"][0], {"uniform", "mild_bias", "strong_bias"})

        promoted_ids = {row["study_id"] for row in result.promoted_rows}
        rejected_ids = {row["study_id"] for row in result.rejected_rows}
        self.assertTrue(promoted_ids.isdisjoint(rejected_ids))

    def test_study_candidate_generation_uses_injected_results(self) -> None:
        with (
            patch("kinematic_classifier_sandbox.validation.study_candidate.generation.analyze_study_candidate_protocol", side_effect=AssertionError("protocol should not be recomputed")),
            patch("kinematic_classifier_sandbox.validation.study_candidate.generation.analyze_common_experiment", side_effect=AssertionError("common result should not be recomputed")),
            patch("kinematic_classifier_sandbox.validation.study_candidate.generation.analyze_corpus_autodevelopment", side_effect=AssertionError("corpus result should not be recomputed")),
        ):
            result = analyze_study_candidate_generation(
                seed=7,
                trajectories_per_case=6,
                protocol_result=self.protocol_result,
                common_result=self.common_result,
                corpus_result=self.corpus_result,
            )
        self.assertEqual(result.promoted_rows, self.result.promoted_rows)
        self.assertEqual(result.rejected_rows, self.result.rejected_rows)

    def test_study_candidate_generation_writes_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_study_candidate_generation_artifacts(
                temp_dir,
                result=self.result,
            )
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "study_candidate_generation")
            self.assertTrue(artifacts.schema_path.exists())
            self.assertTrue(artifacts.generated_candidates_path.exists())
            self.assertTrue(artifacts.static_scores_path.exists())
            self.assertTrue(artifacts.feature_evidence_table_path.exists())
            self.assertTrue(artifacts.prior_sensitivity_explanation_table_path.exists())
            self.assertTrue(artifacts.promoted_candidates_path.exists())
            self.assertTrue(artifacts.rejected_candidates_path.exists())
            self.assertTrue(artifacts.monte_carlo_scores_path.exists())
            self.assertTrue(artifacts.decision_report_path.exists())
            self.assertTrue(artifacts.static_vs_statistical_score_path.exists())
            self.assertTrue(artifacts.candidate_promotion_matrix_path.exists())
            self.assertTrue(artifacts.classifier_feature_class_heatmap_path.exists())

            generated_payload = json.loads(artifacts.generated_candidates_path.read_text(encoding="utf-8"))
            self.assertGreater(len(generated_payload["generated_candidates"]), 0)
            self.assertIn("feature_name", artifacts.feature_evidence_table_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertIn(
                "baseline_prior",
                artifacts.prior_sensitivity_explanation_table_path.read_text(encoding="utf-8").splitlines()[0],
            )

            report_text = artifacts.decision_report_path.read_text(encoding="utf-8")
            self.assertIn("Study Candidate Generation", report_text)
            self.assertIn("Top Promoted Candidates", report_text)


if __name__ == "__main__":
    unittest.main()
