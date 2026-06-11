from __future__ import annotations

import tempfile
import unittest

from kinematic_classifier_sandbox.corpus.control_surfaces.backend_sweep import (
    ControlSurfaceBackendSweepConfig,
    run_control_surface_backend_sweep,
    write_control_surface_backend_sweep_artifacts,
)
from kinematic_classifier_sandbox.corpus.control_surfaces.backends import default_control_surface_backends


class ControlSurfaceBackendTests(unittest.TestCase):
    def test_default_backends_declare_hidden_and_visible_fields(self) -> None:
        backends = default_control_surface_backends()
        self.assertEqual(len(backends), 6)
        backend_ids = {backend.backend_id for backend in backends}
        self.assertEqual(
            backend_ids,
            {
                "direct_kinematic_params",
                "acceleration_sequence",
                "jerk_sequence",
                "spline_knots",
                "hybrid_mode_schedule",
                "stochastic_process",
            },
        )
        for backend in backends:
            metadata = backend.metadata()
            self.assertIn("observed_position", metadata.classifier_allowed_fields)
            self.assertIn("backend_id", metadata.hidden_fields)
            self.assertTrue(metadata.supports["posterior_target"])

    def test_posterior_target_sweep_runs_across_backends(self) -> None:
        config = ControlSurfaceBackendSweepConfig(
            random_candidates_per_backend=4,
            cem_iterations=2,
            cem_population=5,
            seed=13,
        )
        rows = run_control_surface_backend_sweep(config)
        self.assertEqual(len(rows["manifest_rows"]), 6)
        self.assertGreaterEqual(len(rows["observation_surface_rows"]), 6)
        self.assertEqual(len(rows["achievability_rows"]), 6)
        self.assertEqual(len(rows["generator_probe_rows"]), 6)
        self.assertGreater(len(rows["backend_identification_probe_rows"]), 0)
        self.assertGreater(len(rows["backend_identification_confusion_rows"]), 0)
        optimizer_ids = {str(row["optimizer_id"]) for row in rows["evaluation_rows"]}
        self.assertEqual(optimizer_ids, {"random_search", "cem"})
        self.assertTrue(all(0.0 <= float(row["score"]) <= 1.0 for row in rows["evaluation_rows"]))
        self.assertTrue(any(float(row["best_score"]) > 0.80 for row in rows["achievability_rows"]))

    def test_control_surface_artifacts_are_written(self) -> None:
        config = ControlSurfaceBackendSweepConfig(
            random_candidates_per_backend=3,
            cem_iterations=1,
            cem_population=4,
            seed=17,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_control_surface_backend_sweep_artifacts(temp_dir, config=config)
            self.assertTrue(artifacts.control_surface_manifest_path.exists())
            self.assertTrue(artifacts.backend_capability_matrix_path.exists())
            self.assertTrue(artifacts.backend_objective_achievability_path.exists())
            self.assertTrue(artifacts.posterior_target_backend_sweep_path.exists())
            self.assertTrue(artifacts.target_vs_achieved_posterior_path.exists())
            self.assertTrue(artifacts.generator_identification_probe_path.exists())
            self.assertTrue(artifacts.backend_identification_probe_path.exists())
            self.assertTrue(artifacts.backend_identification_confusion_path.exists())
            self.assertTrue(artifacts.observation_surface_manifest_path.exists())
            self.assertTrue(artifacts.achievability_plot_path.exists())
            self.assertTrue(artifacts.posterior_plot_path.exists())
            self.assertTrue(artifacts.backend_probe_plot_path.exists())
            self.assertTrue(artifacts.report_path.exists())


if __name__ == "__main__":
    unittest.main()
