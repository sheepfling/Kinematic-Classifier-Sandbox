from __future__ import annotations

import tempfile
import unittest

from kinematic_classifier_sandbox.corpus.trajectory_exploration.objective_scoring import (
    objective_score_schema,
    objective_spec_schema,
    posterior_target_objective_spec,
    score_posterior_target_distribution,
    write_objective_proof_artifacts,
)
from kinematic_classifier_sandbox.trajectory_generator import _make_manual_trajectory


def _trajectory_with_constant_acceleration(acceleration: float, *, trajectory_id: str):
    times = tuple(index * 0.25 for index in range(9))
    velocities = tuple(acceleration * time for time in times)
    positions = tuple(0.5 * acceleration * time * time for time in times)
    accelerations = tuple(acceleration for _ in times)
    return _make_manual_trajectory(
        trajectory_id=trajectory_id,
        true_class="constant_acceleration",
        tier="boundary_v1",
        scenario_family="posterior_target_test",
        measurements=positions,
        times=times,
        true_position=positions,
        true_velocity=velocities,
        true_acceleration=accelerations,
        measurement_std=0.01,
        outlier_indices=[],
        seed=7,
        generator_parameters={},
    )


class ObjectiveScoringTests(unittest.TestCase):
    def test_posterior_target_score_is_bounded_and_prefers_ambiguous_case(self) -> None:
        spec = posterior_target_objective_spec()
        ambiguous = _trajectory_with_constant_acceleration(0.15, trajectory_id="ambiguous")
        clear_cv = _trajectory_with_constant_acceleration(0.0, trajectory_id="clear_cv")
        ambiguous_score = score_posterior_target_distribution(spec, ambiguous)
        clear_score = score_posterior_target_distribution(spec, clear_cv)
        self.assertGreaterEqual(ambiguous_score.score, 0.0)
        self.assertLessEqual(ambiguous_score.score, 1.0)
        self.assertGreater(ambiguous_score.primary_terms["posterior_entropy"], 0.0)
        self.assertLess(ambiguous_score.primary_terms["posterior_tv_error"], clear_score.primary_terms["posterior_tv_error"])
        self.assertGreater(ambiguous_score.score, clear_score.score)

    def test_objective_schemas_and_proof_artifacts_are_written(self) -> None:
        self.assertEqual(objective_spec_schema()["title"], "ObjectiveSpec")
        self.assertEqual(objective_score_schema()["title"], "ObjectiveScore")
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_objective_proof_artifacts(temp_dir)
            self.assertTrue(artifacts.objective_spec_schema_path.exists())
            self.assertTrue(artifacts.objective_score_schema_path.exists())
            self.assertTrue(artifacts.objective_family_catalog_path.exists())
            self.assertTrue(artifacts.proof_ladder_path.exists())
            self.assertTrue(artifacts.posterior_target_spec_path.exists())
            self.assertTrue(artifacts.report_path.exists())


if __name__ == "__main__":
    unittest.main()
