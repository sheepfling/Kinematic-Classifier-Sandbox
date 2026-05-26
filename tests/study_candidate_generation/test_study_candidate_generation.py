from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.study_candidate_generation import (
    analyze_study_candidate_generation,
    write_study_candidate_generation_artifacts,
)


class StudyCandidateGenerationTests(unittest.TestCase):
    def test_study_candidates_are_generated_and_scored(self) -> None:
        result = analyze_study_candidate_generation(seed=7, trajectories_per_case=6)

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

    def test_study_candidate_generation_writes_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_study_candidate_generation_artifacts(
                temp_dir,
                result=analyze_study_candidate_generation(seed=7, trajectories_per_case=6),
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
