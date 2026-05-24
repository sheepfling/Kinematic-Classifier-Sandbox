from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox import (
    analyze_study_candidate_protocol,
    write_study_candidate_protocol_artifacts,
)


class StudyCandidateProtocolTests(unittest.TestCase):
    def test_m18_protocol_and_schema_artifacts_are_generated(self) -> None:
        result = analyze_study_candidate_protocol()

        self.assertTrue(result.validation_summary["protocol_has_ten_steps"])
        self.assertTrue(result.validation_summary["study_candidate_has_core_specs"])
        self.assertTrue(result.validation_summary["study_candidate_has_optional_filter_spec"])
        self.assertTrue(result.validation_summary["decision_vocab_complete"])
        self.assertTrue(result.validation_summary["validation_ladder_has_ten_levels"])
        self.assertEqual(result.validation_summary["overall_status"], "pass")
        self.assertIn("StudyCandidate", result.study_candidate_schema["title"])
        self.assertEqual(
            result.validation_ladder_schema["properties"]["final_decision"]["enum"],
            ["promote", "revise", "reject", "defer"],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_study_candidate_protocol_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "study_candidate_generation")
            self.assertTrue(artifacts.protocol_path.exists())
            self.assertTrue(artifacts.study_candidate_schema_path.exists())
            self.assertTrue(artifacts.validation_ladder_schema_path.exists())
            self.assertTrue(artifacts.validation_summary_path.exists())

            study_schema = json.loads(artifacts.study_candidate_schema_path.read_text(encoding="utf-8"))
            self.assertIn("corpus_spec", study_schema["properties"])
            self.assertIn("decision_policy", study_schema["properties"])

            ladder_schema = json.loads(artifacts.validation_ladder_schema_path.read_text(encoding="utf-8"))
            level_names = ladder_schema["properties"]["levels"]["items"]["properties"]["level_name"]["enum"]
            self.assertEqual(len(level_names), 10)
            self.assertIn("prior_sensitivity", level_names)

            protocol_text = artifacts.protocol_path.read_text(encoding="utf-8")
            self.assertIn("Ten-Step Protocol", protocol_text)
            self.assertIn("promote", protocol_text)
            self.assertIn("revise", protocol_text)
            self.assertIn("reject", protocol_text)
            self.assertIn("defer", protocol_text)


if __name__ == "__main__":
    unittest.main()
