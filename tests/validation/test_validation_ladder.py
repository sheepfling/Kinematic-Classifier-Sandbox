from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import kinematic_classifier_sandbox.api as api


class ValidationLadderTests(unittest.TestCase):
    def test_validation_ladder_emits_scores_and_decisions(self) -> None:
        result = api.analyze_validation_ladder(seed=7, trajectories_per_case=6)

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

    def test_validation_ladder_writes_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = api.write_validation_ladder_artifacts(
                temp_dir,
                result=api.analyze_validation_ladder(seed=7, trajectories_per_case=6),
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
