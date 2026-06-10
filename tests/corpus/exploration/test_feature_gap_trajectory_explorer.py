from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.corpus.exploration.feature_gap_trajectory_explorer import (
    analyze_feature_gap_trajectory_explorer,
    write_feature_gap_trajectory_explorer_artifacts,
)


class FeatureGapTrajectoryExplorerTests(unittest.TestCase):
    def test_gap_rows_recommendations_and_iterations_are_generated(self) -> None:
        result = analyze_feature_gap_trajectory_explorer(seed=7, max_iterations=3)

        self.assertGreater(len(result.gap_rows), 0)
        self.assertGreater(len(result.recommendation_rows), 0)
        self.assertGreater(len(result.iteration_rows), 0)
        self.assertIn(result.stop_reason, {"iteration_budget_exhausted", "no_improving_candidate", "plateau_reached", "no_gaps_remaining"})
        self.assertEqual(result.initial_candidate_id, "baseline_uniform")
        self.assertTrue(any(row.gap_kind == "feature_set" for row in result.gap_rows))
        self.assertTrue(any(row.gap_kind == "class_pair" for row in result.gap_rows))
        self.assertIn("Feature Gap Trajectory Explorer", result.report_markdown)

    def test_loop_accepts_an_improving_candidate_and_tracks_progress(self) -> None:
        result = analyze_feature_gap_trajectory_explorer(seed=7, max_iterations=3)

        accepted_rows = [row for row in result.iteration_rows if row.accepted]
        self.assertGreater(len(accepted_rows), 0)
        self.assertGreater(len(result.selected_candidate_ids), 1)
        self.assertNotEqual(result.final_candidate_id, result.initial_candidate_id)
        self.assertTrue(
            accepted_rows[0].selected_q_corpus > accepted_rows[0].starting_q_corpus
            or accepted_rows[0].selected_feature_excitation > accepted_rows[0].starting_feature_excitation
            or accepted_rows[0].selected_boundary_coverage > accepted_rows[0].starting_boundary_coverage
            or accepted_rows[0].selected_overall_score > accepted_rows[0].starting_overall_score
        )
        self.assertTrue(any(row.trajectory_family == "stress_feature_excitation" for row in result.recommendation_rows))
        self.assertTrue(any(row.trajectory_family == "boundary_pair_focus" for row in result.recommendation_rows))

    def test_artifacts_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_feature_gap_trajectory_explorer_artifacts(temp_dir)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "feature_gap_trajectory_explorer_v1")
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.gap_rows_path.exists())
            self.assertTrue(artifacts.recommendation_rows_path.exists())
            self.assertTrue(artifacts.iteration_rows_path.exists())
            self.assertTrue(artifacts.candidate_scores_path.exists())
            self.assertTrue(artifacts.selected_manifest_path.exists())
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.q_corpus_progression_png_path.exists())
            self.assertTrue(artifacts.gap_priority_png_path.exists())
            self.assertTrue(artifacts.recommendation_family_png_path.exists())

            summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
            self.assertIn("stop_reason", summary)
            self.assertGreater(summary["accepted_iteration_count"], 0)
            selected_manifest = json.loads(artifacts.selected_manifest_path.read_text(encoding="utf-8"))
            self.assertIn("selected_candidates", selected_manifest)
            self.assertGreater(len(selected_manifest["selected_candidates"]), 1)


if __name__ == "__main__":
    unittest.main()
