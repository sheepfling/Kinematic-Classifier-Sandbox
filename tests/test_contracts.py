from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox import (
    ClassifierOutputArtifact,
    TrajectoryArtifact,
    validate_classifier_output_artifact,
    validate_milestone0_sample_run_artifacts,
    validate_trajectory_artifact,
    write_milestone0_sample_run_artifacts,
)


class ContractTests(unittest.TestCase):
    def test_trajectory_validation_accepts_monotonic_series(self) -> None:
        artifact = TrajectoryArtifact(
            trajectory_id="traj",
            true_class="A",
            scenario_id="scenario",
            seed=1,
            times=(0.0, 1.0, 2.0),
            measurements=(0.1, 0.2, 0.3),
        )
        self.assertEqual(validate_trajectory_artifact(artifact), [])

    def test_trajectory_validation_accepts_dimension_metadata(self) -> None:
        artifact = TrajectoryArtifact(
            trajectory_id="traj",
            true_class="A",
            scenario_id="scenario",
            seed=1,
            times=(0.0, 1.0, 2.0),
            measurements=(
                (0.1, 0.0, 0.0),
                (0.2, 0.1, 0.0),
                (0.3, 0.1, 0.1),
            ),
            measurement_dim=3,
            measurement_axes=("x", "y", "z"),
            coordinate_frame="enu",
            state_dim=9,
            state_axes=("x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az"),
            truth_series={"x": (0.0, 1.0, 2.0), "vz": (0.0, 0.0, 0.0)},
        )
        self.assertEqual(validate_trajectory_artifact(artifact), [])

    def test_trajectory_validation_rejects_non_monotonic_time(self) -> None:
        artifact = TrajectoryArtifact(
            trajectory_id="traj",
            true_class="A",
            scenario_id="scenario",
            seed=1,
            times=(0.0, 1.0, 1.0),
            measurements=(0.1, 0.2, 0.3),
        )
        errors = validate_trajectory_artifact(artifact)
        self.assertTrue(any("strictly increasing" in error for error in errors))

    def test_trajectory_validation_rejects_axis_dim_mismatch(self) -> None:
        artifact = TrajectoryArtifact(
            trajectory_id="traj",
            true_class="A",
            scenario_id="scenario",
            seed=1,
            times=(0.0, 1.0),
            measurements=(0.1, 0.2),
            measurement_dim=3,
            measurement_axes=("x", "y"),
        )
        errors = validate_trajectory_artifact(artifact)
        self.assertTrue(any("measurement_axes must match measurement_dim" in error for error in errors))

    def test_classifier_output_validation_enforces_probability_contract(self) -> None:
        artifact = ClassifierOutputArtifact(
            trajectory_id="traj",
            class_names=("A", "B"),
            rows=(
                {
                    "trajectory_id": "traj",
                    "time": 0.0,
                    "true_class": "A",
                    "predicted_class": "A",
                    "confidence": 0.8,
                    "posterior_A": 0.8,
                    "posterior_B": 0.2,
                    "log_likelihood_A": -0.1,
                    "log_likelihood_B": -1.5,
                },
            ),
        )
        self.assertEqual(validate_classifier_output_artifact(artifact), [])

    def test_sample_run_directory_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_milestone0_sample_run_artifacts(temp_dir)
            self.assertTrue(artifacts.run_dir.exists())
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "milestone0_contract_demo")

            errors = validate_milestone0_sample_run_artifacts(artifacts)
            self.assertEqual(errors, [])

            report = artifacts.report_path.read_text(encoding="utf-8")
            self.assertIn("## Experiment Summary", report)
            self.assertIn("## Artifact Contract", report)
            self.assertIn("## Validation", report)

            posterior_header = artifacts.posterior_history_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("posterior_A", posterior_header)
            self.assertIn("posterior_B", posterior_header)


if __name__ == "__main__":
    unittest.main()
