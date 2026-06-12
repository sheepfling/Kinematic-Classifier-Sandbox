from __future__ import annotations

import csv
import json
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
from kinematic_classifier_sandbox.advanced_filters.ou_witness import (
    analyze_ornstein_uhlenbeck_witness,
    ornstein_uhlenbeck_witness_surface,
    write_ornstein_uhlenbeck_witness_artifacts,
)
from kinematic_classifier_sandbox.advanced_filters.oracle_gsf_1d import (
    write_gsf_abs_range_multimodal_witness_artifacts,
)
from kinematic_classifier_sandbox.advanced_filters.oracle_pf_1d import (
    write_pf_abs_range_multimodal_witness_artifacts,
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
        self.assertEqual(ornstein_uhlenbeck_witness_surface().study_id, "ornstein_uhlenbeck_witness_v1")
        self.assertEqual(advanced_filter_comparison_surface().study_id, "advanced_filter_comparison_v1")
        self.assertIsInstance(imm_witness_surface(), WitnessSurface)
        self.assertIsInstance(particle_filter_witness_surface(), WitnessSurface)
        self.assertIsInstance(rbpf_witness_surface(), WitnessSurface)
        self.assertIsInstance(ornstein_uhlenbeck_witness_surface(), WitnessSurface)
        self.assertIsInstance(advanced_filter_comparison_surface(), WitnessSurface)

    def test_particle_filter_witness_outputs_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_particle_filter_witness_artifacts(temp_dir)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "particle_filter_v1")
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.posterior_history_path.exists())
            self.assertTrue(artifacts.state_estimate_history_path.exists())
            self.assertTrue(artifacts.metrics_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.method_evaluation_summary_path.exists())
            self.assertTrue((artifacts.run_dir / "traces" / "filter_step_trace.csv").exists())
            for plot in artifacts.plot_paths:
                self.assertTrue(plot.exists())
            metrics_text = artifacts.metrics_path.read_text(encoding="utf-8")
            self.assertIn("promotion_decision", metrics_text)
            summary_text = artifacts.method_evaluation_summary_path.read_text(encoding="utf-8")
            self.assertIn("negative_log_likelihood", summary_text)
            self.assertIn("brier_score", summary_text)

    def test_rbpf_witness_outputs_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_rbpf_witness_artifacts(temp_dir)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "rbpf_v1")
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.posterior_history_path.exists())
            self.assertTrue(artifacts.state_estimate_history_path.exists())
            self.assertTrue(artifacts.metrics_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.method_evaluation_summary_path.exists())
            self.assertTrue((artifacts.run_dir / "traces" / "filter_step_trace.csv").exists())
            for plot in artifacts.plot_paths:
                self.assertTrue(plot.exists())
            report = artifacts.report_path.read_text(encoding="utf-8")
            self.assertIn("conditionally Kalman-filtering", report)
            summary_text = artifacts.method_evaluation_summary_path.read_text(encoding="utf-8")
            self.assertIn("ece", summary_text)
            self.assertIn("posterior_margin", summary_text)

    def test_ou_witness_outputs_expected_artifacts(self) -> None:
        result = analyze_ornstein_uhlenbeck_witness(seed=37)
        self.assertGreater(len(result.posterior_rows), 0)
        self.assertGreater(len(result.state_rows), 0)
        self.assertIn("final_mean_reverting_posterior", result.metrics)
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_ornstein_uhlenbeck_witness_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "ornstein_uhlenbeck_witness_v1")
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.posterior_history_path.exists())
            self.assertTrue(artifacts.state_estimate_history_path.exists())
            self.assertTrue(artifacts.metrics_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.method_evaluation_summary_path.exists())
            self.assertTrue((artifacts.run_dir / "traces" / "filter_step_trace.csv").exists())
            report = artifacts.report_path.read_text(encoding="utf-8")
            self.assertIn("Ornstein-Uhlenbeck", report)

    def test_advanced_filter_comparison_outputs_decision_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            write_imm_artifacts(temp_dir)
            write_particle_filter_witness_artifacts(temp_dir)
            write_pf_abs_range_multimodal_witness_artifacts(temp_dir)
            write_gsf_abs_range_multimodal_witness_artifacts(temp_dir)
            write_rbpf_witness_artifacts(temp_dir)
            write_ornstein_uhlenbeck_witness_artifacts(temp_dir)
            artifacts = write_advanced_filter_comparison_artifacts(temp_dir)
            self.assertTrue(artifacts.method_comparison_path.exists())
            self.assertTrue(artifacts.gate_matrix_path.exists())
            self.assertTrue(artifacts.gate_matrix_json_path.exists())
            self.assertTrue(artifacts.nonlinear_stress_metrics_path.exists())
            self.assertTrue(artifacts.particle_filter_robustness_summary_path.exists())
            self.assertTrue(artifacts.gsf_robustness_summary_path.exists())
            self.assertTrue(artifacts.gsf_vs_pf_frontier_path.exists())
            self.assertTrue(artifacts.gsf_vs_pf_frontier_summary_path.exists())
            self.assertTrue(artifacts.latent_maneuver_metrics_path.exists())
            self.assertTrue(artifacts.rbpf_robustness_summary_path.exists())
            self.assertTrue(artifacts.mean_reverting_metrics_path.exists())
            self.assertTrue(artifacts.runtime_cost_metrics_path.exists())
            self.assertTrue(artifacts.particle_count_pareto_path.exists())
            self.assertTrue(artifacts.pf_vs_rbpf_frontier_path.exists())
            self.assertTrue(artifacts.pf_vs_rbpf_frontier_summary_path.exists())
            self.assertTrue(artifacts.advanced_method_promotion_cards_path.exists())
            self.assertTrue(artifacts.decision_matrix_path.exists())
            self.assertTrue(artifacts.particle_count_plot_path.exists())
            self.assertTrue(artifacts.gsf_vs_pf_frontier_plot_path.exists())
            self.assertTrue(artifacts.pf_vs_rbpf_frontier_plot_path.exists())
            decision_text = artifacts.decision_matrix_path.read_text(encoding="utf-8")
            self.assertIn("promote", decision_text)
            self.assertIn("supporting_artifact", decision_text)
            self.assertIn("required_by_current_evidence", decision_text)
            report = artifacts.report_path.read_text(encoding="utf-8")
            self.assertIn("advanced evidence providers", report)
            self.assertIn("Status Semantics", report)
            self.assertIn("not yet required by current repo evidence", report)
            self.assertIn("PF vs RBPF Trade", report)
            self.assertIn("metric_split", artifacts.gsf_vs_pf_frontier_summary_path.read_text(encoding="utf-8"))
            frontier_text = artifacts.pf_vs_rbpf_frontier_summary_path.read_text(encoding="utf-8")
            self.assertIn("rbpf_preferred", frontier_text)
            self.assertIn("metric_split", frontier_text)
            cards = artifacts.advanced_method_promotion_cards_path.read_text(encoding="utf-8")
            self.assertIn("witness-specific", cards)
            self.assertIn("Required by current evidence: not yet.", cards)
            nonlinear_text = artifacts.nonlinear_stress_metrics_path.read_text(encoding="utf-8")
            self.assertIn("mean_oracle_to_pf_kl", nonlinear_text)
            self.assertIn("mean_oracle_to_gaussian_kl", nonlinear_text)
            self.assertIn("gsf_vs_pf_crossover_status", nonlinear_text)
            robustness_text = artifacts.particle_filter_robustness_summary_path.read_text(encoding="utf-8")
            self.assertIn("recommended_particle_count", robustness_text)
            self.assertIn("robustness_sweep_passes", robustness_text)
            gsf_robustness_text = artifacts.gsf_robustness_summary_path.read_text(encoding="utf-8")
            self.assertIn("recommended_max_components", gsf_robustness_text)
            self.assertIn("robustness_sweep_passes", gsf_robustness_text)
            rbpf_robustness_text = artifacts.rbpf_robustness_summary_path.read_text(encoding="utf-8")
            self.assertIn("latent_crossover_status", rbpf_robustness_text)
            self.assertIn("smooth_crossover_status", rbpf_robustness_text)
            with artifacts.gate_matrix_path.open(encoding="utf-8", newline="") as handle:
                gate_rows = list(csv.DictReader(handle))
            with artifacts.method_comparison_path.open(encoding="utf-8", newline="") as handle:
                method_rows = list(csv.DictReader(handle))
            self.assertEqual(
                {"imm_v1", "particle_filter_bank_v1", "rbpf_v1", "ornstein_uhlenbeck_pf_v1"},
                {row["method_id"] for row in gate_rows},
            )
            pf_method_row = next(row for row in method_rows if row["method_id"] == "particle_filter_bank_v1")
            self.assertTrue(
                pf_method_row["artifact_path"].endswith(
                    "advanced_filter_comparison_v1/gsf_vs_pf_frontier_summary.csv"
                )
                or pf_method_row["artifact_path"].endswith(
                    "pf_abs_range_multimodal_oracle_v1/summary.csv"
                )
            )
            self.assertTrue(all(row["implemented"] == "yes" for row in gate_rows))
            self.assertTrue(all(row["contract_hooked"] == "yes" for row in gate_rows))
            self.assertTrue(all(row["generalized"] == "no" for row in gate_rows))
            self.assertTrue(any(row["status_level"] == "witness_supported" for row in gate_rows))
            self.assertTrue(all(row["intermediate_trace_packet"] == "yes" for row in gate_rows))
            pf_row = next(row for row in gate_rows if row["method_id"] == "particle_filter_bank_v1")
            self.assertEqual(pf_row["scenario_family"], "abs_range_multimodal_1d")
            self.assertEqual(pf_row["robustness_sweep_passes"], "yes")
            self.assertEqual(pf_row["status_level"], "justified_for_study")
            rbpf_row = next(row for row in gate_rows if row["method_id"] == "rbpf_v1")
            self.assertEqual(rbpf_row["robustness_sweep_passes"], "yes")
            self.assertEqual(rbpf_row["status_level"], "justified_for_study")
            gate_json = json.loads(artifacts.gate_matrix_json_path.read_text(encoding="utf-8"))
            self.assertEqual(len(gate_rows), len(gate_json))

    def test_runner_helper_returns_comparison_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            write_imm_artifacts(temp_dir)
            write_particle_filter_witness_artifacts(temp_dir)
            write_pf_abs_range_multimodal_witness_artifacts(temp_dir)
            write_gsf_abs_range_multimodal_witness_artifacts(temp_dir)
            write_rbpf_witness_artifacts(temp_dir)
            write_ornstein_uhlenbeck_witness_artifacts(temp_dir)
            artifacts = run_advanced_filter_comparison(temp_dir)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "advanced_filter_comparison_v1")
            self.assertTrue(artifacts.method_comparison_path.exists())
            self.assertTrue(artifacts.gate_matrix_path.exists())
            self.assertTrue(artifacts.decision_matrix_path.exists())
            self.assertTrue(artifacts.particle_count_pareto_path.exists())
            self.assertTrue(artifacts.gsf_vs_pf_frontier_path.exists())
            self.assertTrue(artifacts.gsf_vs_pf_frontier_summary_path.exists())
            self.assertTrue(artifacts.pf_vs_rbpf_frontier_path.exists())
            self.assertTrue(artifacts.pf_vs_rbpf_frontier_summary_path.exists())


if __name__ == "__main__":
    unittest.main()
