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
)
from kinematic_classifier_sandbox.validation.study_candidate.protocol import (
    analyze_study_candidate_protocol,
)
from kinematic_classifier_sandbox.validation.validation_ladder import (
    analyze_validation_ladder,
    write_validation_ladder_artifacts,
)


class ValidationLadderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol_result = analyze_study_candidate_protocol()
        cls.common_result = analyze_common_experiment(seed=7, trajectories_per_case=6)
        cls.corpus_result = analyze_corpus_autodevelopment(seed=7)
        cls.study_generation_result = analyze_study_candidate_generation(
            seed=7,
            trajectories_per_case=6,
            protocol_result=cls.protocol_result,
            common_result=cls.common_result,
            corpus_result=cls.corpus_result,
        )
        cls.result = analyze_validation_ladder(
            seed=7,
            trajectories_per_case=6,
            protocol_result=cls.protocol_result,
            common_result=cls.common_result,
            corpus_result=cls.corpus_result,
            study_generation_result=cls.study_generation_result,
        )

    def test_validation_ladder_emits_scores_and_decisions(self) -> None:
        result = self.result

        self.assertGreater(len(result.score_rows), 0)
        self.assertGreater(len(result.decision_rows), 0)
        decisions = {row["final_decision"] for row in result.decision_rows}
        self.assertTrue(decisions.issubset({"promote", "revise", "reject", "defer"}))
        self.assertIn("promote", decisions)
        self.assertIn("reject", decisions)

        study_ids = {row["study_id"] for row in result.decision_rows}
        for study_id in list(study_ids)[:10]:
            level_rows = [row for row in result.score_rows if row["study_id"] == study_id]
            self.assertEqual(len(level_rows), 10)

        self.assertIn("Validation Ladder", result.report_markdown)
        self.assertIn("Decision Counts", result.report_markdown)

    def test_validation_ladder_uses_injected_results(self) -> None:
        with (
            patch("kinematic_classifier_sandbox.validation.validation_ladder_runner.analyze_study_candidate_protocol", side_effect=AssertionError("protocol should not be recomputed")),
            patch("kinematic_classifier_sandbox.validation.validation_ladder_runner.analyze_common_experiment", side_effect=AssertionError("common result should not be recomputed")),
            patch("kinematic_classifier_sandbox.validation.validation_ladder_runner.analyze_corpus_autodevelopment", side_effect=AssertionError("corpus result should not be recomputed")),
            patch("kinematic_classifier_sandbox.validation.validation_ladder_runner.analyze_study_candidate_generation", side_effect=AssertionError("study generation should not be recomputed")),
        ):
            result = analyze_validation_ladder(
                seed=7,
                trajectories_per_case=6,
                protocol_result=self.protocol_result,
                common_result=self.common_result,
                corpus_result=self.corpus_result,
                study_generation_result=self.study_generation_result,
            )
        self.assertEqual(result.decision_rows, self.result.decision_rows)

    def test_validation_ladder_writes_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_validation_ladder_artifacts(
                temp_dir,
                result=self.result,
            )
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "validation_ladder")
            self.assertTrue(artifacts.schema_path.exists())
            self.assertTrue(artifacts.scores_path.exists())
            self.assertTrue(artifacts.decisions_path.exists())
            self.assertTrue(artifacts.report_path.exists())

            schema = json.loads(artifacts.schema_path.read_text(encoding="utf-8"))
            self.assertEqual(schema["title"], "ValidationLadder")

            report_text = artifacts.report_path.read_text(encoding="utf-8")
            self.assertIn("Top Decisions", report_text)


if __name__ == "__main__":
    unittest.main()
