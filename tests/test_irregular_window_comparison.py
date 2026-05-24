from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox import (
    analyze_irregular_window_comparison,
    render_irregular_window_report,
    write_irregular_window_artifacts,
)


class IrregularWindowComparisonTests(unittest.TestCase):
    def test_duration_window_reduces_regular_irregular_gap(self) -> None:
        result = analyze_irregular_window_comparison(seed=7, replicas=8)
        gap_lookup = {
            (row.window_mode, row.feature_name): row
            for row in result.summary_rows
        }
        self.assertLessEqual(
            gap_lookup[("duration", "slope")].mean_regular_irregular_gap,
            gap_lookup[("sample_count", "slope")].mean_regular_irregular_gap,
        )
        self.assertLessEqual(
            gap_lookup[("duration", "curvature_proxy")].mean_regular_irregular_gap,
            gap_lookup[("sample_count", "curvature_proxy")].mean_regular_irregular_gap,
        )

    def test_duration_window_uses_stable_elapsed_horizon(self) -> None:
        result = analyze_irregular_window_comparison(seed=7, replicas=4, duration_window=5.0)
        irregular_duration_rows = [
            row for row in result.feature_rows
            if row.window_mode == "duration" and row.sampling_regime == "irregular"
        ]
        self.assertTrue(irregular_duration_rows)
        for row in irregular_duration_rows:
            self.assertLessEqual(row.duration, 5.0 + 1e-6)

    def test_irregular_window_artifacts_are_generated(self) -> None:
        result = analyze_irregular_window_comparison(seed=7, replicas=4)
        report = render_irregular_window_report(result)
        self.assertIn("Milestone 15", report)
        self.assertIn("sample-count windows", report)
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_irregular_window_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "irregular_window_comparison_v1")
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.features_path.exists())
            self.assertTrue(artifacts.plot_png_path.exists())
            summary_header = artifacts.summary_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("window_mode", summary_header)
            self.assertIn("mean_regular_irregular_gap", summary_header)


if __name__ == "__main__":
    unittest.main()
