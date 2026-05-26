from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.api import (
    analyze_backend_adapter_proof,
    write_backend_adapter_proof_artifacts,
)


class BackendAdapterProofTests(unittest.TestCase):
    def test_analysis_proves_shared_scenario_and_cache_stability(self) -> None:
        result = analyze_backend_adapter_proof()

        shared = result.backend_manifest["shared_compatible_scenario"]
        self.assertGreaterEqual(len(shared["backend_ids"]), 2)

        cache_probe = result.backend_manifest["cache_probe"]
        self.assertTrue(cache_probe["stable_cache_key"])
        self.assertTrue(cache_probe["second_run_cache_hit"])

        self.assertGreaterEqual(len(result.failure_rows), 2)
        self.assertIn("Output Equivalence", result.report_markdown)

        success_rows = [
            row
            for row in result.backend_run_rows
            if row["candidate_id"] == "shared_boundary_cv_ca" and bool(row["success"])
        ]
        self.assertGreaterEqual(len(success_rows), 2)

    def test_failure_rows_are_structured(self) -> None:
        result = analyze_backend_adapter_proof()
        failure_reasons = {str(row["failure_reason"]) for row in result.failure_rows}
        self.assertIn("missing_switch_time", failure_reasons)
        self.assertIn("missing_input_deck_hash", failure_reasons)

    def test_artifacts_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_backend_adapter_proof_artifacts(temp_dir)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "backend_adapter_proof")
            self.assertTrue(artifacts.backend_manifest_path.exists())
            self.assertTrue(artifacts.backend_run_examples_path.exists())
            self.assertTrue(artifacts.backend_output_equivalence_report_path.exists())
            self.assertTrue(artifacts.adapter_failure_cases_path.exists())
            self.assertTrue(artifacts.telemetry_comparison_png_path.exists())
            self.assertTrue(artifacts.failure_taxonomy_png_path.exists())

            payload = json.loads(artifacts.backend_manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["cache_probe"]["stable_cache_key"])
            self.assertGreaterEqual(payload["structured_failure_count"], 2)


if __name__ == "__main__":
    unittest.main()
