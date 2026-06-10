from __future__ import annotations

import tempfile
import unittest

import numpy as np

from kinematic_classifier_sandbox.corpus.trajectory_exploration import (
    SequentialBoundaryControlConfig,
    SequentialPpoConfig,
    SequentialTrajectoryGym,
    analyze_sequential_ppo_boundary_control,
    evaluate_control_sequence,
    has_stable_baselines3_support,
    write_sequential_ppo_boundary_control_artifacts,
)


class SequentialPpoBoundaryControlTests(unittest.TestCase):
    def test_environment_is_deterministic_for_fixed_seed(self) -> None:
        config = SequentialBoundaryControlConfig(episode_horizon=8)
        env_a = SequentialTrajectoryGym(config=config, seed=17)
        env_b = SequentialTrajectoryGym(config=config, seed=17)
        obs_a, _ = env_a.reset(seed=17)
        obs_b, _ = env_b.reset(seed=17)
        self.assertTrue(np.allclose(obs_a, obs_b))
        for _ in range(4):
            action = np.asarray([0.25], dtype=np.float32)
            obs_a, reward_a, term_a, trunc_a, _ = env_a.step(action)
            obs_b, reward_b, term_b, trunc_b, _ = env_b.step(action)
            self.assertTrue(np.allclose(obs_a, obs_b))
            self.assertAlmostEqual(reward_a, reward_b)
            self.assertEqual(term_a, term_b)
            self.assertEqual(trunc_a, trunc_b)

    def test_control_sequence_normalizes_into_shared_schema(self) -> None:
        summary = evaluate_control_sequence((0.2, 0.4, -0.1, -0.3, 0.1, 0.0), seed=29)
        row = summary.evaluation.as_row()
        self.assertEqual(summary.proposal.metadata["action_mode"], "sequential_control")
        self.assertEqual(int(row["control_sequence_length"]), len(summary.control_sequence))
        self.assertIn("rollout_return", row)
        self.assertGreaterEqual(float(row["class_validity"]), 0.0)

    def test_control_actions_obey_bounds(self) -> None:
        summary = evaluate_control_sequence((2.5, -2.5, 0.5, -0.5), seed=31)
        self.assertTrue(all(-1.0 <= value <= 1.0 for value in summary.control_sequence))
        self.assertTrue(all(abs(value) <= 1.25 + 1e-6 for value in summary.accelerations))

    def test_artifacts_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = write_sequential_ppo_boundary_control_artifacts(
                temp_dir,
                config=SequentialBoundaryControlConfig(episode_horizon=8),
                ppo_config=SequentialPpoConfig(total_timesteps=256, n_steps=32, batch_size=32, eval_episodes=4),
            )
            self.assertIsNotNone(result.artifacts)
            artifacts = result.artifacts
            assert artifacts is not None
            self.assertTrue(artifacts.checkpoints_dir.exists())
            self.assertTrue(artifacts.environment_contract_path.exists())
            self.assertTrue(artifacts.training_config_path.exists())
            self.assertTrue(artifacts.checkpoint_manifest_path.exists())
            self.assertTrue(artifacts.training_summary_path.exists())
            self.assertTrue(artifacts.evaluation_rows_path.exists())
            self.assertTrue(artifacts.selected_rollouts_path.exists())
            self.assertTrue(artifacts.control_sequences_path.exists())
            self.assertTrue(artifacts.training_trace_rows_path.exists())
            self.assertTrue(artifacts.snapshot_rows_path.exists())
            self.assertTrue(artifacts.training_curve_path.exists())
            self.assertTrue(artifacts.rollout_gallery_path.exists())
            self.assertTrue(artifacts.utility_progress_path.exists())
            self.assertTrue(artifacts.feature_progress_path.exists())
            self.assertTrue(artifacts.class_space_progress_path.exists())
            self.assertTrue(artifacts.ppo_vs_heuristics_path.exists())
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.rl_algorithm_decision_report_path.exists())

    @unittest.skipUnless(has_stable_baselines3_support(), "stable-baselines3 is optional")
    def test_writer_can_resume_existing_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            first = write_sequential_ppo_boundary_control_artifacts(
                temp_dir,
                config=SequentialBoundaryControlConfig(episode_horizon=8),
                ppo_config=SequentialPpoConfig(
                    total_timesteps=256,
                    n_steps=32,
                    batch_size=32,
                    eval_episodes=4,
                    checkpoint_interval_timesteps=128,
                    snapshot_interval_timesteps=128,
                ),
            )
            second = write_sequential_ppo_boundary_control_artifacts(
                temp_dir,
                config=SequentialBoundaryControlConfig(episode_horizon=8),
                ppo_config=SequentialPpoConfig(
                    total_timesteps=256,
                    n_steps=32,
                    batch_size=32,
                    eval_episodes=4,
                    checkpoint_interval_timesteps=128,
                    snapshot_interval_timesteps=128,
                ),
            )
            self.assertEqual(first.checkpoint_manifest["target_total_timesteps"], second.checkpoint_manifest["target_total_timesteps"])
            self.assertTrue(bool(second.checkpoint_manifest["latest_checkpoint_path"]))
            self.assertGreaterEqual(int(second.training_summary["timesteps_completed"]), 256)

    @unittest.skipUnless(has_stable_baselines3_support(), "stable-baselines3 is optional")
    def test_ppo_smoke_run_beats_random_and_writes_comparison(self) -> None:
        result = analyze_sequential_ppo_boundary_control(
            config=SequentialBoundaryControlConfig(episode_horizon=10),
            ppo_config=SequentialPpoConfig(total_timesteps=768, n_steps=32, batch_size=32, eval_episodes=6),
        )
        self.assertEqual(result.training_summary["status"], "experimental")
        self.assertGreater(len(result.evaluation_rows), 0)
        self.assertGreater(len(result.ppo_vs_heuristics_rows), 0)
        self.assertGreater(len(result.snapshot_rows), 0)
        self.assertTrue(bool(result.training_summary["beats_random_control"]))
        self.assertTrue(bool(result.training_summary["beats_scripted_mean"]) or int(result.training_summary["novel_rollout_count"]) > 0)


if __name__ == "__main__":
    unittest.main()
