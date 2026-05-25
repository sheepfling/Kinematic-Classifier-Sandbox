from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox import api


class CorpusGymTests(unittest.TestCase):
    def test_environment_resets_and_steps_with_target(self) -> None:
        environment = api.CorpusGymEnvironment()
        target = api.default_corpus_gym_targets()[0]
        observation = environment.reset(target)

        self.assertEqual(observation["target_id"], target.target_id)
        self.assertEqual(observation["target_type"], target.target_type)
        self.assertFalse(observation["done"])

        step_result = environment.step(
            api.CorpusGymAction(
                seed=11,
                tier_name=target.target_tier or "realistic_v1",
            )
        )
        self.assertTrue(step_result["done"])
        self.assertIn("reward", step_result)
        self.assertIn("class_validity", step_result["reward"])
        self.assertIsNotNone(environment.trajectory())

    def test_simulation_is_reproducible_from_seed_and_action(self) -> None:
        environment = api.CorpusGymEnvironment()
        target = api.default_corpus_gym_targets()[1]
        action = api.CorpusGymAction(
            seed=23,
            tier_name=target.target_tier or "boundary_v1",
            duration_scale=0.9,
            measurement_scale=1.1,
            irregularity_scale=1.2,
            outlier_scale=1.05,
            step_scale=0.95,
        )

        environment.reset(target)
        first = environment.simulate(action)
        environment.reset(target)
        second = environment.simulate(action)

        self.assertEqual(first.trajectory.times, second.trajectory.times)
        self.assertEqual(first.trajectory.measurements, second.trajectory.measurements)
        self.assertEqual(first.reward.total_utility, second.reward.total_utility)

    def test_analysis_and_artifacts_are_generated(self) -> None:
        result = api.analyze_corpus_gym_contract()
        self.assertIn("target_types", result.environment_contract)
        self.assertGreater(len(result.example_targets), 0)
        self.assertIn("Reward Breakdown", result.report_markdown)
        walkthrough = api.render_corpus_gym_numeric_walkthrough_markdown(result)
        self.assertIn("Corpus Gym Numeric Walkthrough", walkthrough)
        self.assertIn("Numeric Substitution", walkthrough)
        self.assertIn("total_utility", walkthrough)

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = api.write_corpus_gym_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "corpus_gym")
            self.assertTrue(artifacts.environment_contract_path.exists())
            self.assertTrue(artifacts.example_targets_path.exists())
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.numeric_walkthrough_path.exists())

            contract = json.loads(artifacts.environment_contract_path.read_text(encoding="utf-8"))
            self.assertIn("reward_components", contract)
            targets = json.loads(artifacts.example_targets_path.read_text(encoding="utf-8"))
            self.assertGreater(len(targets["targets"]), 0)
            walkthrough_text = artifacts.numeric_walkthrough_path.read_text(encoding="utf-8")
            self.assertIn("Corpus Gym Numeric Walkthrough", walkthrough_text)
            self.assertIn("Implemented Reward Equation", walkthrough_text)


if __name__ == "__main__":
    unittest.main()
