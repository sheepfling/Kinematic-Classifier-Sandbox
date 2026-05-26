from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.inference.kalman_filter_bank import (
    KalmanTrajectory,
    default_kalman_model_specs,
    render_kalman_bank_png_bytes,
    render_kalman_bank_report,
    render_kalman_bank_svg,
    run_kalman_bank_benchmark,
    run_kalman_filter_bank,
    write_kalman_bank_artifacts,
)


def _make_trajectory(
    *,
    trajectory_id: str,
    true_class: str,
    scenario_name: str,
    times: tuple[float, ...],
    position0: float,
    velocity0: float,
    acceleration: float,
) -> KalmanTrajectory:
    positions = tuple(position0 + velocity0 * time + 0.5 * acceleration * time * time for time in times)
    velocities = tuple(velocity0 + acceleration * time for time in times)
    accelerations = tuple(acceleration for _ in times)
    return KalmanTrajectory(
        trajectory_id=trajectory_id,
        true_class=true_class,
        scenario_name=scenario_name,
        seed=7,
        times=times,
        measurements=positions,
        true_position=positions,
        true_velocity=velocities,
        true_acceleration=accelerations,
    )


class KalmanFilterBankTests(unittest.TestCase):
    def test_constant_velocity_model_beats_stationary_and_estimates_velocity(self) -> None:
        trajectory = _make_trajectory(
            trajectory_id="cv_exact",
            true_class="constant_velocity",
            scenario_name="cv_exact",
            times=tuple(float(step) for step in range(10)),
            position0=0.5,
            velocity0=1.25,
            acceleration=0.0,
        )
        run = run_kalman_filter_bank(trajectory, default_kalman_model_specs())
        self.assertEqual(run.final_predicted_class, "constant_velocity")
        self.assertGreater(run.final_weights["constant_velocity"], run.final_weights["stationary"])
        self.assertAlmostEqual(run.final_states["constant_velocity"].mean[1], 1.25, delta=0.15)

    def test_constant_acceleration_model_beats_constant_velocity_after_enough_samples(self) -> None:
        trajectory = _make_trajectory(
            trajectory_id="ca_exact",
            true_class="constant_acceleration",
            scenario_name="ca_exact",
            times=tuple(float(step) for step in range(10)),
            position0=-0.2,
            velocity0=0.3,
            acceleration=0.35,
        )
        run = run_kalman_filter_bank(trajectory, default_kalman_model_specs())
        self.assertEqual(run.final_predicted_class, "constant_acceleration")
        self.assertGreater(run.final_weights["constant_acceleration"], run.final_weights["constant_velocity"])

    def test_irregular_dt_keeps_constant_velocity_classification_stable(self) -> None:
        regular = _make_trajectory(
            trajectory_id="cv_regular",
            true_class="constant_velocity",
            scenario_name="cv_regular",
            times=tuple(float(step) for step in range(10)),
            position0=1.0,
            velocity0=0.9,
            acceleration=0.0,
        )
        irregular = _make_trajectory(
            trajectory_id="cv_irregular",
            true_class="constant_velocity",
            scenario_name="cv_irregular",
            times=(0.0, 0.7, 1.6, 2.8, 4.1, 5.0, 6.6, 7.4, 8.9, 10.0),
            position0=1.0,
            velocity0=0.9,
            acceleration=0.0,
        )
        regular_run = run_kalman_filter_bank(regular, default_kalman_model_specs())
        irregular_run = run_kalman_filter_bank(irregular, default_kalman_model_specs())
        self.assertEqual(regular_run.final_predicted_class, "constant_velocity")
        self.assertEqual(irregular_run.final_predicted_class, "constant_velocity")
        self.assertAlmostEqual(
            regular_run.final_weights["constant_velocity"],
            irregular_run.final_weights["constant_velocity"],
            delta=0.20,
        )

    def test_constant_velocity_track_with_single_outlier_stays_constant_velocity(self) -> None:
        times = tuple(float(step) for step in range(8))
        true_positions = tuple(1.0 * time for time in times)
        measurements = list(true_positions)
        measurements[4] -= 2.6
        trajectory = KalmanTrajectory(
            trajectory_id="cv_outlier",
            true_class="constant_velocity",
            scenario_name="cv_outlier",
            seed=7,
            times=times,
            measurements=tuple(measurements),
            true_position=true_positions,
            true_velocity=tuple(1.0 for _ in times),
            true_acceleration=tuple(0.0 for _ in times),
        )
        run = run_kalman_filter_bank(trajectory, default_kalman_model_specs())
        self.assertEqual(run.final_predicted_class, "constant_velocity")
        self.assertGreater(run.final_weights["constant_velocity"], 0.95)

    def test_kalman_bank_artifacts_are_generated(self) -> None:
        result = run_kalman_bank_benchmark(seed=7)
        report = render_kalman_bank_report(result)
        svg = render_kalman_bank_svg(result)
        png = render_kalman_bank_png_bytes(result)

        self.assertIn("Kalman Filter Bank", report)
        self.assertIn("variance inflation", report)
        self.assertIn("<svg", svg)
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertGreater(result.summary.final_accuracy, 0.60)

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_kalman_bank_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "kalman_filter_bank")
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.innovation_history_path.exists())
            self.assertTrue(artifacts.state_estimate_history_path.exists())
            self.assertTrue(artifacts.posterior_history_path.exists())
            self.assertTrue(artifacts.confusion_matrix_path.exists())
            self.assertTrue(artifacts.config_path.exists())
            self.assertTrue(artifacts.dataset_manifest_path.exists())
            self.assertTrue(artifacts.model_definitions_path.exists())
            self.assertTrue(artifacts.plot_png_path.exists())


if __name__ == "__main__":
    unittest.main()
