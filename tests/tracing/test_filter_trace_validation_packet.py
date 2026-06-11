from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.tracing.filter_trace_validation_packet import (
    analyze_filter_trace_validation_packet,
    write_filter_trace_validation_artifacts,
)


class FilterTraceValidationPacketTests(unittest.TestCase):
    def test_analysis_materializes_and_reports_trace_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = analyze_filter_trace_validation_packet(Path(temp_dir), materialize=True)
            self.assertGreater(len(result.method_rows), 0)
            method_ids = {str(row["method_id"]) for row in result.method_rows}
            self.assertIn("kalman_bank", method_ids)
            self.assertIn("transition_matrix_accumulator", method_ids)
            self.assertIn("imm_v1", method_ids)
            self.assertIn("particle_filter_bank_v1", method_ids)
            self.assertIn("rbpf_v1", method_ids)
            kalman_row = next(row for row in result.method_rows if row["method_id"] == "kalman_bank")
            self.assertEqual(kalman_row["trace_status"], "trace_validated")
            self.assertEqual(kalman_row["has_step_cards"], "yes")
            self.assertEqual(kalman_row["has_likelihood"], "yes")

    def test_artifacts_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_filter_trace_validation_artifacts(Path(temp_dir), materialize=True)
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.method_trace_matrix_path.exists())
            self.assertTrue(artifacts.trace_requirement_matrix_path.exists())
            self.assertTrue(artifacts.schema_path.exists())
            summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
            self.assertGreater(summary["method_count"], 0)


if __name__ == "__main__":
    unittest.main()
