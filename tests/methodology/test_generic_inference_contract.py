from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.api import *


class GenericInferenceContractTests(unittest.TestCase):
    def test_generic_inference_contract_artifacts_are_generated(self) -> None:
        result = analyze_generic_inference_contract()

        self.assertEqual(result.validation_results["overall_status"], "pass")
        self.assertTrue(result.validation_results["classifier_output_contract"]["all_schema_checks_passed"])
        self.assertTrue(result.validation_results["filter_output_contract"]["step_contract_passed"])
        self.assertTrue(result.validation_results["filter_output_contract"]["state_contract_passed"])
        self.assertIn("required_run_fields", result.classifier_output_schema)
        self.assertIn("required_fields", result.evidence_provider_schema)
        self.assertIn("required_fields", result.posterior_history_schema)
        self.assertIn("required_fields_for_filter_backends", result.filter_output_schema)
        self.assertIn("Generic Inference Contract", result.report_markdown)

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_generic_inference_contract_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "generic_inference_contract")
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.classifier_output_schema_path.exists())
            self.assertTrue(artifacts.evidence_provider_schema_path.exists())
            self.assertTrue(artifacts.posterior_history_schema_path.exists())
            self.assertTrue(artifacts.filter_output_schema_path.exists())
            self.assertTrue(artifacts.validation_results_path.exists())

            classifier_schema = json.loads(artifacts.classifier_output_schema_path.read_text(encoding="utf-8"))
            self.assertIn("required_row_fields", classifier_schema)

            validation_results = json.loads(artifacts.validation_results_path.read_text(encoding="utf-8"))
            self.assertEqual(validation_results["overall_status"], "pass")
            classifier_ids = [row["classifier_id"] for row in validation_results["classifier_output_contract"]["classifiers"]]
            self.assertIn("pointwise", classifier_ids)
            self.assertIn("kalman_bank", classifier_ids)

            report_text = artifacts.report_path.read_text(encoding="utf-8")
            self.assertIn("Classifier Output Contract", report_text)
            self.assertIn("Evidence Provider Contract", report_text)
            self.assertIn("Posterior History Contract", report_text)
            self.assertIn("Filter Output Contract", report_text)


if __name__ == "__main__":
    unittest.main()
