from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.registry.method_validation_os import (
    analyze_method_validation_os,
    write_method_validation_os_artifacts,
)


class MethodValidationOSTests(unittest.TestCase):
    def test_analysis_exposes_method_and_witness_registry(self) -> None:
        result = analyze_method_validation_os()
        self.assertGreater(len(result.method_rows), 10)
        self.assertGreater(len(result.witness_rows), 10)
        method_ids = {row.method_id for row in result.method_rows}
        witness_ids = {row.witness_id for row in result.witness_rows}
        self.assertIn("particle_filter", method_ids)
        self.assertIn("rbpf", method_ids)
        self.assertIn("ukf", method_ids)
        self.assertIn("gaussian_sum_filter", method_ids)
        self.assertIn("tcn", method_ids)
        self.assertIn("temperature_scaling", method_ids)
        self.assertIn("cmaes", method_ids)
        self.assertIn("abs_range_multimodal_1d", witness_ids)
        self.assertIn("latent_maneuver_onset_duration", witness_ids)
        self.assertIn("neural_sequence_vs_physics_frontier", witness_ids)
        self.assertIn("continuous_generator_frontier", witness_ids)
        pf_row = next(row for row in result.method_rows if row.method_id == "particle_filter")
        rbpf_row = next(row for row in result.method_rows if row.method_id == "rbpf")
        gsf_row = next(row for row in result.method_rows if row.method_id == "gaussian_sum_filter")
        self.assertEqual(pf_row.current_status, "study_justified")
        self.assertEqual(rbpf_row.current_status, "witness_supported")
        self.assertEqual(rbpf_row.current_failure_status, "not_complexity_justified")
        self.assertEqual(gsf_row.current_status, "witness_supported")

    def test_artifacts_write_status_and_coverage_matrices(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_method_validation_os_artifacts(temp_dir)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "method_validation_os_v1")
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.method_specs_path.exists())
            self.assertTrue(artifacts.witness_specs_path.exists())
            self.assertTrue(artifacts.promotion_status_matrix_path.exists())
            self.assertTrue(artifacts.witness_coverage_matrix_path.exists())
            self.assertTrue(artifacts.lane_summary_path.exists())
            self.assertTrue(artifacts.promotion_status_plot_path.exists())
            self.assertTrue(artifacts.witness_coverage_plot_path.exists())

            summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
            self.assertGreater(summary["method_count"], 10)
            self.assertGreater(summary["witness_count"], 10)

            with artifacts.promotion_status_matrix_path.open(encoding="utf-8", newline="") as handle:
                status_rows = list(csv.DictReader(handle))
            with artifacts.witness_coverage_matrix_path.open(encoding="utf-8", newline="") as handle:
                coverage_rows = list(csv.DictReader(handle))
            self.assertTrue(any(row["method_id"] == "particle_filter" for row in status_rows))
            self.assertTrue(any(row["method_id"] == "rbpf" for row in status_rows))
            pf_row = next(row for row in status_rows if row["method_id"] == "particle_filter")
            self.assertEqual(pf_row["study_justified"], "yes")
            rbpf_row = next(row for row in status_rows if row["method_id"] == "rbpf")
            self.assertEqual(rbpf_row["study_justified"], "no")
            self.assertTrue(
                any(
                    row["witness_id"] == "abs_range_multimodal_1d"
                    and row["method_id"] == "particle_filter"
                    and row["designed_to_justify"] == "yes"
                    for row in coverage_rows
                )
            )


if __name__ == "__main__":
    unittest.main()
