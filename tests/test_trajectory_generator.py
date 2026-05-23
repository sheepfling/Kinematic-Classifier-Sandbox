from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox import (
    default_dataset_tiers,
    default_trajectory_class_definitions,
    generate_trajectory_dataset,
    generate_trajectory_datasets,
    write_trajectory_generator_artifacts,
)
from kinematic_classifier_sandbox import trajectory_generator as tg


class TrajectoryGeneratorTests(unittest.TestCase):
    def test_default_definitions_cover_expected_classes(self) -> None:
        class_names = [definition.name for definition in default_trajectory_class_definitions()]
        self.assertEqual(
            class_names,
            [
                "stationary",
                "constant_velocity",
                "constant_acceleration",
                "braking",
                "maneuver",
                "oscillatory",
                "bounded_acceleration",
            ],
        )
        tier_names = [tier.name for tier in default_dataset_tiers()]
        self.assertEqual(
            tier_names,
            ["easy_v1", "boundary_v1", "adversarial_v1", "stress_v1", "realistic_v1"],
        )

    def test_generator_is_deterministic_for_fixed_seed(self) -> None:
        first = generate_trajectory_dataset("easy_v1", seed=7, trajectories_per_class=2)
        second = generate_trajectory_dataset("easy_v1", seed=7, trajectories_per_class=2)
        third = generate_trajectory_dataset("easy_v1", seed=11, trajectories_per_class=2)

        self.assertEqual(first.trajectories, second.trajectories)
        self.assertNotEqual(first.trajectories[0].measurements, third.trajectories[0].measurements)

    def test_noise_free_kinematics_match_analytic_forms(self) -> None:
        times = (0.0, 1.0, 2.0)
        stationary_positions, stationary_velocities, stationary_accelerations = tg._evaluate_stationary(
            times, {"position": 3.0}
        )
        velocity_positions, velocity_velocities, velocity_accelerations = tg._evaluate_constant_velocity(
            times, {"position": 2.0, "velocity": -1.5}
        )
        accel_positions, accel_velocities, accel_accelerations = tg._evaluate_constant_acceleration(
            times, {"position": 1.0, "velocity": 0.5, "acceleration": -0.25}
        )

        self.assertEqual(stationary_positions, (3.0, 3.0, 3.0))
        self.assertEqual(stationary_velocities, (0.0, 0.0, 0.0))
        self.assertEqual(stationary_accelerations, (0.0, 0.0, 0.0))
        self.assertEqual(velocity_positions, (2.0, 0.5, -1.0))
        self.assertEqual(velocity_velocities, (-1.5, -1.5, -1.5))
        self.assertEqual(velocity_accelerations, (0.0, 0.0, 0.0))
        self.assertAlmostEqual(accel_positions[2], 1.5, places=6)
        self.assertEqual(accel_velocities, (0.5, 0.25, 0.0))
        self.assertEqual(accel_accelerations, (-0.25, -0.25, -0.25))

    def test_artifact_bundle_is_written(self) -> None:
        datasets = generate_trajectory_datasets(seed=7, trajectories_per_class=2)
        self.assertEqual(len(datasets), 5)
        self.assertGreater(len(datasets[0].trajectories), 0)

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_trajectory_generator_artifacts(temp_dir, seed=7, trajectories_per_class=2)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "trajectory_generator_v1")
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.class_definitions_path.exists())
            self.assertTrue(artifacts.config_path.exists())
            self.assertTrue(artifacts.plot_svg_path.exists())
            self.assertTrue(artifacts.plot_png_path.exists())
            self.assertEqual(set(artifacts.dataset_manifest_paths), {"easy_v1", "boundary_v1", "adversarial_v1", "stress_v1", "realistic_v1"})
            self.assertEqual(set(artifacts.generated_trajectories_paths), {"easy_v1", "boundary_v1", "adversarial_v1", "stress_v1", "realistic_v1"})
            self.assertEqual(set(artifacts.true_states_paths), {"easy_v1", "boundary_v1", "adversarial_v1", "stress_v1", "realistic_v1"})
            for path in artifacts.dataset_manifest_paths.values():
                self.assertTrue(path.exists())
            for path in artifacts.generated_trajectories_paths.values():
                self.assertTrue(path.exists())
            for path in artifacts.true_states_paths.values():
                self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
