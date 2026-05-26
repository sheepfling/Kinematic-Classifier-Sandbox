from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.api import (
    analyze_advanced_filter_contract,
    analyze_advanced_state_inference,
    write_advanced_filter_contract_artifacts,
    write_advanced_state_inference_artifacts,
)


class AdvancedStateInferenceTests(unittest.TestCase):
    def test_contract_exposes_shared_filter_interface(self) -> None:
        result = analyze_advanced_filter_contract()
        self.assertEqual(result.contract.backend_id, "imm_1d_pva_lift_prototype")
        self.assertIn("initialize", result.contract.interface_methods)
        self.assertIn("predict", result.contract.interface_methods)
        self.assertIn("update", result.contract.interface_methods)
        self.assertIn("history", result.contract.interface_methods)
        self.assertIn("posterior_history", result.output_schema["required_rows"])
        self.assertIn("state_estimate_history", result.output_schema["required_rows"])
        self.assertIn("3D PVA", result.report_markdown)

    def test_imm_backend_emits_normalized_posteriors_and_improves_some_witnesses(self) -> None:
        result = analyze_advanced_state_inference(seed=7, replicas=3)
        self.assertGreater(result.summary.num_witnesses, 0)
        self.assertGreater(result.summary.improved_witnesses, 0)
        self.assertGreater(result.summary.imm_accuracy, 0.45)
        self.assertGreater(result.summary.mean_state_rmse, 0.0)
        run = result.runs[0]
        self.assertAlmostEqual(sum(run.final_mode_posteriors.values()), 1.0, places=6)
        self.assertAlmostEqual(sum(run.final_class_posteriors.values()), 1.0, places=6)
        self.assertIn(run.final_predicted_mode, run.mode_names)
        self.assertIn("IMM", result.report_markdown)
        self.assertIn("transition baseline", result.report_markdown.lower())

    def test_imm_artifacts_are_generated(self) -> None:
        result = analyze_advanced_state_inference(seed=7, replicas=2)
        with tempfile.TemporaryDirectory() as temp_dir:
            contract_artifacts = write_advanced_filter_contract_artifacts(temp_dir, result=result.contract_result)
            self.assertEqual(contract_artifacts.run_dir, Path(temp_dir) / "advanced_state_inference_contract")
            self.assertTrue(contract_artifacts.contract_path.exists())
            self.assertTrue(contract_artifacts.output_schema_path.exists())
            self.assertTrue(contract_artifacts.diagnostics_schema_path.exists())
            self.assertTrue(contract_artifacts.report_path.exists())

            artifacts = write_advanced_state_inference_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "advanced_state_inference_v1")
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.config_path.exists())
            self.assertTrue(artifacts.mode_probability_history_path.exists())
            self.assertTrue(artifacts.mixing_probability_history_path.exists())
            self.assertTrue(artifacts.mode_likelihood_history_path.exists())
            self.assertTrue(artifacts.state_estimate_history_path.exists())
            self.assertTrue(artifacts.posterior_history_path.exists())
            self.assertTrue(artifacts.diagnostics_history_path.exists())
            self.assertTrue(artifacts.comparison_path.exists())
            self.assertTrue(artifacts.mode_probability_plot_path.exists())
            self.assertTrue(artifacts.mixing_probability_plot_path.exists())
            self.assertTrue(artifacts.mode_likelihood_plot_path.exists())
            self.assertTrue(artifacts.state_estimate_plot_path.exists())
            self.assertTrue(artifacts.switch_delay_plot_path.exists())
            self.assertTrue(artifacts.comparison_plot_path.exists())
            self.assertTrue(artifacts.plot_png_path.exists())
            header = artifacts.comparison_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("imm_post_switch_accuracy", header)
            self.assertIn("transition_post_switch_accuracy", header)


if __name__ == "__main__":
    unittest.main()
