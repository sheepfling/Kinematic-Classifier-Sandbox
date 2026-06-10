from __future__ import annotations

import tempfile
import unittest

from kinematic_classifier_sandbox.corpus.trajectory_exploration.ppo_boundary_control import (
    SequentialPpoConfig,
)
from kinematic_classifier_sandbox.corpus.trajectory_exploration.sequential_comparison import (
    SequentialCemConfig,
    analyze_sequential_objective_sweep_comparison,
    analyze_sequential_ppo_vs_cem_comparison,
    write_sequential_objective_sweep_comparison_artifacts,
    write_sequential_ppo_vs_cem_comparison_artifacts,
)
from kinematic_classifier_sandbox.corpus.trajectory_exploration.sequential_gym import (
    SequentialBoundaryControlConfig,
)


class SequentialComparisonTests(unittest.TestCase):
    def test_analysis_produces_backend_metrics_and_report(self) -> None:
        result = analyze_sequential_ppo_vs_cem_comparison(
            config=SequentialBoundaryControlConfig(episode_horizon=8),
            ppo_config=SequentialPpoConfig(total_timesteps=128, n_steps=32, batch_size=32, eval_episodes=2, progress_eval_episodes=2),
            cem_config=SequentialCemConfig(iterations=3, population_size=8),
            seed_count=2,
        )
        backend_ids = {str(row["backend_id"]) for row in result.backend_metrics_rows}
        self.assertIn("cem_open_loop", backend_ids)
        self.assertIn("ppo_policy", backend_ids)
        self.assertGreater(len(result.evaluation_rows), 0)
        self.assertGreater(len(result.progress_rows), 0)
        self.assertGreater(len(result.aggregate_backend_metrics_rows), 0)
        self.assertGreater(len(result.backend_decision_rows), 0)
        self.assertEqual(len(result.seed_run_rows), 2)
        self.assertIn("Sequential PPO vs CEM Comparison Report", result.report_markdown)

    def test_writer_persists_comparison_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = write_sequential_ppo_vs_cem_comparison_artifacts(
                temp_dir,
                config=SequentialBoundaryControlConfig(episode_horizon=8),
                ppo_config=SequentialPpoConfig(total_timesteps=128, n_steps=32, batch_size=32, eval_episodes=2, progress_eval_episodes=2),
                cem_config=SequentialCemConfig(iterations=3, population_size=8),
                seed_count=2,
            )
            self.assertIsNotNone(result.artifacts)
            artifacts = result.artifacts
            assert artifacts is not None
            self.assertTrue(artifacts.config_path.exists())
            self.assertTrue(artifacts.artifact_manifest_path.exists())
            self.assertTrue(artifacts.backend_metrics_path.exists())
            self.assertTrue(artifacts.aggregate_backend_metrics_path.exists())
            self.assertTrue(artifacts.backend_decisions_path.exists())
            self.assertTrue(artifacts.seed_runs_path.exists())
            self.assertTrue(artifacts.evaluation_rows_path.exists())
            self.assertTrue(artifacts.selected_rollouts_path.exists())
            self.assertTrue(artifacts.control_sequences_path.exists())
            self.assertTrue(artifacts.progress_rows_path.exists())
            self.assertTrue(artifacts.strengths_limits_path.exists())
            self.assertTrue(artifacts.progress_plot_path.exists())
            self.assertTrue(artifacts.backend_metrics_plot_path.exists())
            self.assertTrue(artifacts.control_gallery_path.exists())
            self.assertTrue(artifacts.report_path.exists())

    def test_objective_sweep_handles_larger_feature_class_space(self) -> None:
        result = analyze_sequential_objective_sweep_comparison(
            config=SequentialBoundaryControlConfig(episode_horizon=8),
            ppo_config=SequentialPpoConfig(total_timesteps=64, n_steps=32, batch_size=32, eval_episodes=2, progress_eval_episodes=1),
            cem_config=SequentialCemConfig(iterations=2, population_size=6),
            objective_ids=("feature_row__accel_high_row", "class_pair__cv_vs_ca"),
            seed_count=1,
        )
        self.assertEqual(len(result.objective_summary_rows), 2)
        self.assertGreater(len(result.backend_summary_rows), 0)
        self.assertGreater(len(result.objective_backend_matrix_rows), 0)
        self.assertIn("Feature/Class Space", result.report_markdown)

    def test_objective_sweep_writer_persists_summary_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = write_sequential_objective_sweep_comparison_artifacts(
                temp_dir,
                config=SequentialBoundaryControlConfig(episode_horizon=8),
                ppo_config=SequentialPpoConfig(total_timesteps=64, n_steps=32, batch_size=32, eval_episodes=2, progress_eval_episodes=1),
                cem_config=SequentialCemConfig(iterations=2, population_size=6),
                objective_ids=("feature_row__accel_high_row", "class_pair__cv_vs_ca"),
                seed_count=1,
            )
            self.assertIsNotNone(result.artifacts)
            artifacts = result.artifacts
            assert artifacts is not None
            self.assertTrue(artifacts.artifact_manifest_path.exists())
            self.assertTrue(artifacts.objective_summary_path.exists())
            self.assertTrue(artifacts.backend_summary_path.exists())
            self.assertTrue(artifacts.decision_summary_path.exists())
            self.assertTrue(artifacts.objective_backend_matrix_path.exists())
            self.assertTrue(artifacts.objective_backend_heatmap_path.exists())
            self.assertTrue(artifacts.report_path.exists())


if __name__ == "__main__":
    unittest.main()
