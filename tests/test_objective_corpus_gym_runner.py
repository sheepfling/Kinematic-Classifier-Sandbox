from __future__ import annotations

import unittest

from kinematic_classifier_sandbox import (
    execute_objective_candidates_via_corpus_gym,
    objective_to_corpus_gym_target,
    default_corpus_objectives,
)


class ObjectiveCorpusGymRunnerTests(unittest.TestCase):
    def test_runner_executes_objective_candidates_through_corpus_gym(self) -> None:
        records = execute_objective_candidates_via_corpus_gym()
        self.assertGreater(len(records), 0)
        self.assertTrue(all(record.execution.backend_id == "corpus_gym" for record in records))
        self.assertTrue(all(record.execution.trajectory_run.metadata["adapter_family"] == "corpus_gym" for record in records))
        self.assertTrue(any(bool(record.action.metadata.get("candidate_id")) for record in records))

    def test_objective_mapping_preserves_target_shape(self) -> None:
        objectives = default_corpus_objectives()
        pair_target = objective_to_corpus_gym_target(objectives[0])
        self.assertEqual(pair_target.target_type, "target_feature_cell")
        self.assertEqual(pair_target.class_pair, ("constant_velocity", "constant_acceleration"))
        class_target = objective_to_corpus_gym_target(objectives[2])
        self.assertEqual(class_target.target_type, "target_feature_cell")
        self.assertEqual(class_target.class_name, "constant_acceleration")


if __name__ == "__main__":
    unittest.main()
