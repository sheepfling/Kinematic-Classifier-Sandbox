from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.advanced_filters.evaluation import (
    advanced_filter_comparison_surface,
    particle_filter_witness_surface,
    rbpf_witness_surface,
    write_advanced_filter_comparison_artifacts,
    write_particle_filter_witness_artifacts,
    write_rbpf_witness_artifacts,
)
from kinematic_classifier_sandbox.advanced_filters.runner import (
    imm_witness_surface,
    run_advanced_filter_comparison,
    write_imm_artifacts,
)
from kinematic_classifier_sandbox.witnesses.surface import WitnessSurface


class AdvancedFilterEvaluationTests(unittest.TestCase):
    def test_surfaces_expose_expected_study_ids(self) -> None:
        self.assertEqual(imm_witness_surface().study_id, "imm_filter_v1")
        self.assertEqual(particle_filter_witness_surface().study_id, "particle_filter_v1")
        self.assertEqual(rbpf_witness_surface().study_id, "rbpf_v1")
        self.assertEqual(advanced_filter_comparison_surface().study_id, "advanced_filter_comparison_v1")
        self.assertIsInstance(imm_witness_surface(), WitnessSurface)
        self.assertIsInstance(particle_filter_witness_surface(), WitnessSurface)
        self.assertIsInstance(rbpf_witness_surface(), WitnessSurface)
        self.assertIsInstance(advanced_filter_comparison_surface(), WitnessSurface)

    def test_particle_filter_witness_outputs_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_particle_filter_witness_artifacts(temp_dir)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "particle_filter_v1")
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.posterior_history_path.exists())
            self.assertTrue(artifacts.state_estimate_history_path.exists())
            self.assertTrue(artifacts.metrics_path.exists())
            for plot in artifacts.plot_paths:
                self.assertTrue(plot.exists())
            metrics_text = artifacts.metrics_path.read_text(encoding="utf-8")
            self.assertIn("promotion_decision", metrics_text)

    def test_rbpf_witness_outputs_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_rbpf_witness_artifacts(temp_dir)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "rbpf_v1")
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.posterior_history_path.exists())
            self.assertTrue(artifacts.state_estimate_history_path.exists())
            self.assertTrue(artifacts.metrics_path.exists())
            for plot in artifacts.plot_paths:
                self.assertTrue(plot.exists())
            report = artifacts.report_path.read_text(encoding="utf-8")
            self.assertIn("conditionally Kalman-filtering", report)

    def test_advanced_filter_comparison_outputs_decision_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            write_imm_artifacts(Path(temp_dir) / "imm_filter_v1")
            write_particle_filter_witness_artifacts(temp_dir)
            write_rbpf_witness_artifacts(temp_dir)
            artifacts = write_advanced_filter_comparison_artifacts(temp_dir)
            self.assertTrue(artifacts.method_comparison_path.exists())
            self.assertTrue(artifacts.nonlinear_stress_metrics_path.exists())
            self.assertTrue(artifacts.latent_maneuver_metrics_path.exists())
            self.assertTrue(artifacts.runtime_cost_metrics_path.exists())
            self.assertTrue(artifacts.decision_matrix_path.exists())
            decision_text = artifacts.decision_matrix_path.read_text(encoding="utf-8")
            self.assertIn("promote", decision_text)
            self.assertIn("supporting_artifact", decision_text)
            report = artifacts.report_path.read_text(encoding="utf-8")
            self.assertIn("advanced evidence providers", report)

    def test_runner_helper_returns_comparison_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            write_imm_artifacts(Path(temp_dir) / "imm_filter_v1")
            write_particle_filter_witness_artifacts(temp_dir)
            write_rbpf_witness_artifacts(temp_dir)
            artifacts = run_advanced_filter_comparison(temp_dir)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "advanced_filter_comparison_v1")
            self.assertTrue(artifacts.method_comparison_path.exists())
            self.assertTrue(artifacts.decision_matrix_path.exists())


if __name__ == "__main__":
    unittest.main()
