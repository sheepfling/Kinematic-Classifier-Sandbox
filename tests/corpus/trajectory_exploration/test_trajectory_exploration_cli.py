from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from kinematic_classifier_sandbox.__main__ import main


class TrajectoryExplorationCliTests(unittest.TestCase):
    def test_trajectory_exploration_objectives_command_writes_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "trajectory-exploration-objectives",
                        "--output-dir",
                        temp_dir,
                    ]
                )
            self.assertEqual(exit_code, 0)
            lines = [line.strip() for line in stdout.getvalue().splitlines() if line.strip()]
            self.assertGreaterEqual(len(lines), 6)
            self.assertTrue((Path(temp_dir) / "trajectory_exploration_objectives" / "objective_manifest.json").exists())
            self.assertTrue((Path(temp_dir) / "trajectory_exploration_objectives" / "objective_table.csv").exists())

    def test_trajectory_exploration_ppo_command_writes_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "trajectory-exploration-ppo",
                        "--output-dir",
                        temp_dir,
                        "--timesteps",
                        "256",
                        "--episode-horizon",
                        "8",
                        "--eval-episodes",
                        "4",
                        "--seed",
                        "7",
                    ]
                )
            self.assertEqual(exit_code, 0)
            lines = [line.strip() for line in stdout.getvalue().splitlines() if line.strip()]
            self.assertGreaterEqual(len(lines), 10)
            self.assertTrue((Path(temp_dir) / "trajectory_exploration_rl" / "ppo_boundary_control" / "checkpoint_manifest.json").exists())
            self.assertTrue((Path(temp_dir) / "trajectory_exploration_rl" / "ppo_boundary_control" / "snapshot_rows.csv").exists())
            self.assertTrue((Path(temp_dir) / "trajectory_exploration_rl" / "ppo_boundary_control" / "report.md").exists())
            self.assertTrue((Path(temp_dir) / "trajectory_exploration_rl" / "rl_algorithm_decision_report.md").exists())

    def test_generated_objective_ppo_sweep_command_writes_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "trajectory-exploration-ppo-sweep-generated",
                        "--output-dir",
                        temp_dir,
                        "--timesteps",
                        "128",
                        "--episode-horizon",
                        "8",
                        "--eval-episodes",
                        "2",
                        "--objective-id",
                        "feature_row__accel_high_row",
                    ]
                )
            self.assertEqual(exit_code, 0)
            self.assertTrue((Path(temp_dir) / "trajectory_exploration_rl" / "generated_objective_sweep" / "summary_rows.csv").exists())
            self.assertTrue((Path(temp_dir) / "trajectory_exploration_rl" / "generated_objective_sweep" / "report.md").exists())

    def test_trajectory_exploration_ppo_vs_cem_command_writes_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "trajectory-exploration-ppo-vs-cem",
                        "--output-dir",
                        temp_dir,
                        "--timesteps",
                        "128",
                        "--episode-horizon",
                        "8",
                        "--eval-episodes",
                        "2",
                        "--progress-eval-episodes",
                        "2",
                        "--cem-iterations",
                        "3",
                        "--cem-population",
                        "8",
                        "--seed-count",
                        "2",
                    ]
                )
            self.assertEqual(exit_code, 0)
            bundle = Path(temp_dir) / "trajectory_exploration_rl" / "ppo_vs_cem_boundary_control"
            self.assertTrue((bundle / "metrics_by_backend.csv").exists())
            self.assertTrue((bundle / "aggregate_metrics_by_backend.csv").exists())
            self.assertTrue((bundle / "backend_decisions.csv").exists())
            self.assertTrue((bundle / "artifact_manifest.json").exists())
            self.assertTrue((bundle / "progress_rows.csv").exists())
            self.assertTrue((bundle / "backend_metrics.png").exists())
            self.assertTrue((bundle / "report.md").exists())

    def test_trajectory_exploration_ppo_vs_cem_sweep_command_writes_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "trajectory-exploration-ppo-vs-cem-sweep",
                        "--output-dir",
                        temp_dir,
                        "--timesteps",
                        "64",
                        "--episode-horizon",
                        "8",
                        "--eval-episodes",
                        "2",
                        "--progress-eval-episodes",
                        "1",
                        "--cem-iterations",
                        "2",
                        "--cem-population",
                        "6",
                        "--objective-id",
                        "feature_row__accel_high_row",
                        "--objective-id",
                        "class_pair__cv_vs_ca",
                    ]
                )
            self.assertEqual(exit_code, 0)
            bundle = Path(temp_dir) / "trajectory_exploration_rl" / "ppo_vs_cem_objective_sweep"
            self.assertTrue((bundle / "artifact_manifest.json").exists())
            self.assertTrue((bundle / "objective_summary.csv").exists())
            self.assertTrue((bundle / "backend_summary.csv").exists())
            self.assertTrue((bundle / "decision_summary.csv").exists())
            self.assertTrue((bundle / "objective_backend_heatmap.png").exists())
            self.assertTrue((bundle / "report.md").exists())


if __name__ == "__main__":
    unittest.main()
