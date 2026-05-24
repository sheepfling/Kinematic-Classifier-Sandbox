from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox import (
    TrajectoryArtifact,
    analyze_dimensional_lift_audit,
    validate_trajectory_artifact,
    write_dimensional_lift_audit_artifacts,
)


class DimensionalLiftAuditTests(unittest.TestCase):
    def test_vector_measurements_validate_when_dimension_metadata_matches(self) -> None:
        artifact = TrajectoryArtifact(
            trajectory_id="vector_demo",
            true_class="demo",
            scenario_id="vector_nominal",
            seed=1,
            times=(0.0, 1.0),
            measurements=((0.0, 0.0, 0.0), (1.0, 2.0, 3.0)),
            measurement_dim=3,
            measurement_axes=("x", "y", "z"),
            coordinate_frame="enu",
            state_dim=6,
            state_axes=("x", "y", "z", "vx", "vy", "vz"),
        )
        self.assertEqual(validate_trajectory_artifact(artifact), [])

    def test_dimensional_lift_audit_artifacts_are_generated(self) -> None:
        result = analyze_dimensional_lift_audit()

        self.assertEqual(result.validation_results["overall_status"], "pass")
        self.assertTrue(result.validation_results["vector_corpus_loaded"])
        self.assertTrue(result.validation_results["vector_features_emitted"])
        self.assertTrue(result.validation_results["vector_predictions_emitted"])
        self.assertTrue(result.validation_results["vector_posteriors_emitted"])
        self.assertTrue(result.module_rows)
        self.assertTrue(result.scalar_assumption_rows)
        self.assertIn("Dimensional Lift Audit", result.audit_markdown)

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_dimensional_lift_audit_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "dimensional_lift_audit")
            self.assertTrue(artifacts.audit_report_path.exists())
            self.assertTrue(artifacts.module_status_path.exists())
            self.assertTrue(artifacts.scalar_assumption_inventory_path.exists())
            self.assertTrue(artifacts.required_adapters_path.exists())
            self.assertTrue(artifacts.vector_predictions_path.exists())
            self.assertTrue(artifacts.vector_posterior_history_path.exists())
            self.assertTrue(artifacts.vector_feature_matrix_path.exists())
            self.assertTrue(artifacts.validation_results_path.exists())

            validation = json.loads(artifacts.validation_results_path.read_text(encoding="utf-8"))
            self.assertEqual(validation["overall_status"], "pass")

            report = artifacts.audit_report_path.read_text(encoding="utf-8")
            self.assertIn("Fake Vector Proof", report)
            self.assertIn("Module Status", report)

            module_header = artifacts.module_status_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("dimensional_status", module_header)

            prediction_header = artifacts.vector_predictions_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("measurement_dim", prediction_header)
            self.assertIn("coordinate_frame", prediction_header)


if __name__ == "__main__":
    unittest.main()
