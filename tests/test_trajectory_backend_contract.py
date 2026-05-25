from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox import (
    analyze_trajectory_backend_contract,
    default_backend_contract_definitions,
    validate_backend_contract_definition,
    write_trajectory_backend_contract_artifacts,
)


class TrajectoryBackendContractTests(unittest.TestCase):
    def test_default_backend_definitions_are_valid(self) -> None:
        definitions = default_backend_contract_definitions()
        self.assertEqual(len(definitions), 4)

        for definition in definitions:
            self.assertEqual(validate_backend_contract_definition(definition), [])

        backend_ids = {definition.capabilities.backend_id for definition in definitions}
        self.assertEqual(
            backend_ids,
            {
                "parameter_only_1d",
                "controlled_1d",
                "environment_aware_1d",
                "mock_file_backend_1d",
            },
        )

    def test_analysis_contains_required_schemas_and_valid_backends(self) -> None:
        result = analyze_trajectory_backend_contract()

        self.assertIn("backend_families", result.backend_contract)
        self.assertEqual(len(result.backend_contract["backend_families"]), 4)
        self.assertIn("properties", result.backend_capability_schema)
        self.assertIn("properties", result.trajectory_run_schema)
        self.assertIn("Relationship Diagram", result.report_markdown)
        self.assertEqual(len(result.capability_matrix_rows), 4)

        validation = result.backend_contract["validation"]
        self.assertTrue(all(row["valid"] for row in validation))

    def test_artifacts_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_trajectory_backend_contract_artifacts(temp_dir)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "trajectory_backend_contract")
            self.assertTrue(artifacts.backend_contract_path.exists())
            self.assertTrue(artifacts.backend_capability_schema_path.exists())
            self.assertTrue(artifacts.scenario_spec_schema_path.exists())
            self.assertTrue(artifacts.design_variable_schema_path.exists())
            self.assertTrue(artifacts.control_policy_schema_path.exists())
            self.assertTrue(artifacts.environment_spec_schema_path.exists())
            self.assertTrue(artifacts.trajectory_run_schema_path.exists())
            self.assertTrue(artifacts.capability_matrix_csv_path.exists())
            self.assertTrue(artifacts.capability_matrix_png_path.exists())
            self.assertTrue(artifacts.report_path.exists())

            payload = json.loads(artifacts.backend_contract_path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["backend_families"]), 4)
            self.assertEqual(sum(1 for row in payload["validation"] if row["valid"]), 4)


if __name__ == "__main__":
    unittest.main()
