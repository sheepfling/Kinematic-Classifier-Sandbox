from __future__ import annotations

import tempfile
import unittest

from kinematic_classifier_sandbox.corpus.trajectory_exploration import (
    default_objective_generation_spec,
    generate_trajectory_exploration_objective_suite,
    write_generated_trajectory_objective_artifacts,
)


class TrajectoryObjectiveGenerationTests(unittest.TestCase):
    def test_generated_suite_covers_cells_rows_pairs_and_novelty(self) -> None:
        suite = generate_trajectory_exploration_objective_suite()
        spec = default_objective_generation_spec()
        self.assertEqual(
            len(suite.objectives),
            4 + len(spec.feature_rows) + len(spec.class_pair_regions) + len(spec.novelty_regions),
        )
        scopes = {str(row["generation_scope"]) for row in suite.objective_rows}
        self.assertEqual(
            scopes,
            {"feature_cell", "feature_row", "class_pair_region", "novelty_region"},
        )
        self.assertEqual(len({objective.objective_id for objective in suite.objectives}), len(suite.objectives))

    def test_generated_feature_row_objective_is_mechanical(self) -> None:
        suite = generate_trajectory_exploration_objective_suite()
        row_objective = next(objective for objective in suite.objectives if objective.mode == "feature_space_row")
        self.assertEqual(row_objective.geometry_target, "sweep_feature_row_novelty")
        self.assertTrue(bool(row_objective.backend_constraints["mechanically_generated"]))
        self.assertEqual(row_objective.target.target_type, "target_feature_space_row")

    def test_generated_objective_artifacts_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_generated_trajectory_objective_artifacts(temp_dir)
            self.assertTrue(artifacts.spec_path.exists())
            self.assertTrue(artifacts.manifest_path.exists())
            self.assertTrue(artifacts.objectives_path.exists())
            self.assertTrue(artifacts.objective_table_path.exists())
            self.assertTrue(artifacts.report_path.exists())


if __name__ == "__main__":
    unittest.main()
