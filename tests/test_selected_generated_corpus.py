from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox import (
    analyze_selected_generated_corpus,
    write_selected_generated_corpus_artifacts,
)


class SelectedGeneratedCorpusTests(unittest.TestCase):
    def test_selected_corpus_is_consumable_by_common_harness(self) -> None:
        result = analyze_selected_generated_corpus()
        self.assertGreater(len(result.trajectory_rows), 0)
        self.assertGreater(len(result.feature_rows), 0)
        self.assertGreater(len(result.classifier_score_rows), 0)
        self.assertGreater(len(result.posterior_rows), 0)
        self.assertTrue(result.corpus_manifest["consumable_by_common_harness"])
        self.assertGreater(int(result.corpus_manifest["harness_prediction_rows"]), 0)
        self.assertIn("routing the selected executable trajectories through the common experiment scoring path", result.report_markdown)

    def test_artifacts_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_selected_generated_corpus_artifacts(temp_dir)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "selected_generated_corpus")
            self.assertTrue(artifacts.manifest_path.exists())
            self.assertTrue(artifacts.trajectories_path.exists())
            self.assertTrue(artifacts.observations_path.exists())
            self.assertTrue(artifacts.truth_states_path.exists())
            self.assertTrue(artifacts.events_path.exists())
            self.assertTrue(artifacts.environment_traces_path.exists())
            self.assertTrue(artifacts.feature_matrix_path.exists())
            self.assertTrue(artifacts.class_validity_scores_path.exists())
            self.assertTrue(artifacts.classifier_scores_path.exists())
            self.assertTrue(artifacts.posterior_history_path.exists())
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.summary_plot_path.exists())
            self.assertTrue(artifacts.validity_plot_path.exists())
            self.assertTrue(artifacts.score_gallery_path.exists())

            manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(manifest["consumable_by_common_harness"])


if __name__ == "__main__":
    unittest.main()
