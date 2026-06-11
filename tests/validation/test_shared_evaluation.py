from __future__ import annotations

import unittest

from kinematic_classifier_sandbox.analysis.common_dataset_comparison import (
    default_shared_classifier_adapters,
    generate_shared_dynamics_dataset,
)
from kinematic_classifier_sandbox.validation.shared_evaluation import (
    CallableSharedClassifierAdapter,
    SharedClassifierMethodSpec,
    SharedClassifierRun,
    SharedMethodCapabilities,
    evaluate_shared_classifier_registry,
    sensor_regime_summary_rows,
)


class SharedEvaluationTests(unittest.TestCase):
    def test_registry_evaluator_runs_registered_classifiers(self) -> None:
        trajectories = generate_shared_dynamics_dataset(seed=7, trajectories_per_case=1)
        adapters = default_shared_classifier_adapters()
        runs = evaluate_shared_classifier_registry(trajectories, adapters)

        self.assertEqual(len(runs), len(trajectories) * len(adapters))
        self.assertIn("particle_filter_bank", {adapter.method_name for adapter in adapters})
        self.assertIn("rbpf", {adapter.method_name for adapter in adapters})
        self.assertTrue(all(run.sensor_regime_id for run in runs))
        self.assertTrue(all(run.measurement_dim == 1 for run in runs))
        self.assertTrue(all(run.coordinate_frame == "scalar_line" for run in runs))
        self.assertTrue(any(adapter.method_spec.primary_evaluation_family == "abs_range_multimodal" for adapter in adapters if adapter.method_name == "particle_filter_bank"))
        self.assertTrue(any("ou_mean_reversion" in adapter.method_spec.supported_scenario_families for adapter in adapters if adapter.method_name == "particle_filter_bank"))
        summary_rows = sensor_regime_summary_rows(runs)
        self.assertTrue(any(row["sensor_regime_id"] == "position_only" for row in summary_rows))
        self.assertTrue(any(row["measurement_dims"] == "1" for row in summary_rows))
        self.assertTrue(any(row["coordinate_frames"] == "scalar_line" for row in summary_rows))

    def test_callable_adapter_contract(self) -> None:
        trajectory = generate_shared_dynamics_dataset(seed=7, trajectories_per_case=1)[0]

        adapter = CallableSharedClassifierAdapter(
            method_spec=SharedClassifierMethodSpec(
                method_name="dummy",
                sensor_regime_id="position_only",
                primary_evaluation_family="shared_binary_dynamics",
                supported_scenario_families=("shared_binary_dynamics",),
                witness_artifact=None,
                capabilities=SharedMethodCapabilities(True, False, False, False, False, False, False),
            ),
            predict_fn=lambda current_trajectory, prior=None: SharedClassifierRun(
                method_name="dummy",
                sensor_regime_id="position_only",
                trajectory_id=current_trajectory.trajectory_id,
                true_class=current_trajectory.true_class,
                scenario_name=current_trajectory.scenario_name,
                final_predicted_class=current_trajectory.true_class,
                final_confidence=1.0,
                final_weights={current_trajectory.true_class: 1.0},
                measurement_dim=3,
                coordinate_frame="enu",
            ),
        )

        run = adapter.predict_trajectory(trajectory)
        self.assertEqual(run.method_name, "dummy")
        self.assertEqual(adapter.method_spec.primary_evaluation_family, "shared_binary_dynamics")
        self.assertEqual(run.final_predicted_class, trajectory.true_class)
        self.assertEqual(run.measurement_dim, 3)
        self.assertEqual(run.coordinate_frame, "enu")


if __name__ == "__main__":
    unittest.main()
