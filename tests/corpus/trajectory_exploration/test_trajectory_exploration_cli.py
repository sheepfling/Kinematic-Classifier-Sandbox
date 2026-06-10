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


if __name__ == "__main__":
    unittest.main()
