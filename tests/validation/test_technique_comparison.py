from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.analysis.common_dataset_comparison import (
    default_shared_classifier_adapters,
)
from kinematic_classifier_sandbox.validation.technique_comparison import (
    analyze_technique_comparison,
    default_technique_definitions,
    render_technique_comparison_report,
    write_technique_comparison_artifacts,
)


class TechniqueComparisonTests(unittest.TestCase):
    def test_technique_comparison_artifacts_are_generated(self) -> None:
        result = analyze_technique_comparison(seed=7)
        report = render_technique_comparison_report(result)

        self.assertIn("Technique Comparison Study", report)
        method_names = [row.method_name for row in result.rows]
        sensor_regimes = [row.sensor_regime_id for row in result.rows]
        self.assertEqual(
            method_names,
            ["pointwise", "windowed_raw", "windowed_robust", "accumulator", "kalman_bank", "kalman_bank_velocity_aided"],
        )
        self.assertEqual(
            sensor_regimes,
            ["position_only", "position_only", "position_only", "position_only", "position_only", "position_plus_direct_velocity"],
        )
        self.assertTrue(any(row.boundary_accuracy is not None for row in result.rows))
        self.assertTrue(any(row.irregular_dt_accuracy is not None for row in result.rows))
        self.assertTrue(any(row.stronger_sensor_stream > 0.0 for row in result.rows))
        self.assertEqual(
            [definition.method_name for definition in default_technique_definitions()],
            method_names,
        )
        shared_adapters = default_shared_classifier_adapters()
        shared_adapter_pairs = [(adapter.method_name, adapter.sensor_regime_id) for adapter in shared_adapters]
        self.assertEqual(
            [(definition.method_name, definition.sensor_regime_id) for definition in default_technique_definitions()],
            shared_adapter_pairs[: len(default_technique_definitions())],
        )
        self.assertIn(("particle_filter_bank", "position_only"), shared_adapter_pairs)
        self.assertIn(("rbpf", "position_only"), shared_adapter_pairs)

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_technique_comparison_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "technique_comparison_v1")
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.summary_csv_path.exists())
            self.assertTrue(artifacts.scenario_csv_path.exists())
            self.assertTrue(artifacts.capability_csv_path.exists())
            self.assertTrue(artifacts.metric_heatmap_png_path.exists())
            self.assertTrue(artifacts.scatter_png_path.exists())
            self.assertTrue(artifacts.capability_png_path.exists())


if __name__ == "__main__":
    unittest.main()
