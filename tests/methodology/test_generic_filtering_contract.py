from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.methodology.filtering_contract import (
    analyze_generic_filtering_contract,
    write_generic_filtering_contract_artifacts,
)


class GenericFilteringContractTests(unittest.TestCase):
    def test_generic_filtering_contract_artifacts_are_generated(self) -> None:
        result = analyze_generic_filtering_contract()

        self.assertIn("validation", result.filter_backend_contract)
        self.assertTrue(result.filter_backend_contract["validation"]["kalman"]["step_contract_passed"])
        self.assertTrue(result.filter_backend_contract["validation"]["kalman"]["state_contract_passed"])
        self.assertTrue(result.filter_backend_contract["validation"]["switching_kalman_mode_bank"]["step_contract_passed"])
        self.assertIn("Generic Filtering Contract", result.filtering_principles_report)
        self.assertIn("Particle Filter Decision Report", result.particle_filter_decision_report)
        self.assertIn("Rao-Blackwell Particle Filter Decision Report", result.rbpf_decision_report)

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_generic_filtering_contract_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "filtering_contract")
            self.assertTrue(artifacts.filter_backend_contract_path.exists())
            self.assertTrue(artifacts.filter_diagnostics_schema_path.exists())
            self.assertTrue(artifacts.filtering_principles_report_path.exists())
            self.assertTrue(artifacts.particle_filter_decision_report_path.exists())
            self.assertTrue(artifacts.rbpf_decision_report_path.exists())

            backend_contract = json.loads(artifacts.filter_backend_contract_path.read_text(encoding="utf-8"))
            self.assertIn("backend_interface", backend_contract)
            self.assertIn("reference_backends", backend_contract)
            self.assertIn("kalman_bank", backend_contract["reference_backends"])

            diagnostics_schema = json.loads(artifacts.filter_diagnostics_schema_path.read_text(encoding="utf-8"))
            self.assertIn("innovation_backend_fields", diagnostics_schema)
            self.assertIn("future_particle_backend_fields", diagnostics_schema)

            principles = artifacts.filtering_principles_report_path.read_text(encoding="utf-8")
            self.assertIn("Contract Summary", principles)
            self.assertIn("Validation", principles)

            particle_report = artifacts.particle_filter_decision_report_path.read_text(encoding="utf-8")
            self.assertIn("What Would Be Sampled", particle_report)

            rbpf_report = artifacts.rbpf_decision_report_path.read_text(encoding="utf-8")
            self.assertIn("What Would Be Marginalized Analytically", rbpf_report)


if __name__ == "__main__":
    unittest.main()
