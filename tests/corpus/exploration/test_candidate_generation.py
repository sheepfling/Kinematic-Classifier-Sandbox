from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.api import (
    analyze_candidate_generation,
    generate_candidates_from_objective_file,
    write_candidate_generation_artifacts,
    write_corpus_objective_artifacts,
)


class CandidateGenerationTests(unittest.TestCase):
    def test_candidates_are_generated_from_objective_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            objective_artifacts = write_corpus_objective_artifacts(temp_dir)
            candidates = generate_candidates_from_objective_file(objective_artifacts.example_objectives_path)
            self.assertGreater(len(candidates), 0)
            objective_ids = {candidate.provenance["objective_id"] for candidate in candidates}
            self.assertIn("cv_vs_ca_boundary_entropy", objective_ids)
            self.assertIn("switching_transition_delay", objective_ids)

    def test_analysis_uses_multiple_samplers(self) -> None:
        result = analyze_candidate_generation()
        sampler_names = {row["sampler_name"] for row in result.generated_candidate_rows}
        self.assertEqual(
            sampler_names,
            {"random", "grid", "lhs", "boundary_mutation", "archive_mutation", "stress_mutation"},
        )
        self.assertIn("Candidate generation is now objective-driven", result.report_markdown)

    def test_artifacts_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_candidate_generation_artifacts(temp_dir)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "candidate_generation")
            self.assertTrue(artifacts.sampler_manifest_path.exists())
            self.assertTrue(artifacts.generated_candidates_path.exists())
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.sampler_comparison_png_path.exists())
            self.assertTrue(artifacts.candidate_coverage_png_path.exists())
            self.assertTrue(artifacts.mutation_lineage_png_path.exists())

            payload = json.loads(artifacts.sampler_manifest_path.read_text(encoding="utf-8"))
            self.assertIn("samplers", payload)


if __name__ == "__main__":
    unittest.main()
