from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.analysis.short_horizon_identifiability import (
    analyze_short_horizon_identifiability,
    render_short_horizon_identifiability_report,
    write_short_horizon_identifiability_artifacts,
)


class ShortHorizonIdentifiabilityTests(unittest.TestCase):
    def test_short_horizon_identifiability_artifacts_are_generated(self) -> None:
        result = analyze_short_horizon_identifiability()
        report = render_short_horizon_identifiability_report(result)

        self.assertIn("Short-Horizon Identifiability", report)
        self.assertEqual(len(result.times), 4)
        self.assertGreater(result.times[-1].absolute_gap, result.times[1].absolute_gap)
        self.assertGreater(result.noise_sweep[0].mean_normalized_gap, result.noise_sweep[-1].mean_normalized_gap)
        self.assertIsNotNone(result.duration_thresholds[0].first_time_at_1sigma)

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_short_horizon_identifiability_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "short_horizon_identifiability_v1")
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.time_series_path.exists())
            self.assertTrue(artifacts.noise_sweep_path.exists())
            self.assertTrue(artifacts.duration_thresholds_path.exists())
            self.assertTrue(artifacts.time_plot_png_path.exists())
            self.assertTrue(artifacts.noise_plot_png_path.exists())
            self.assertTrue(artifacts.duration_thresholds_path.exists())
            self.assertTrue(artifacts.duration_plot_png_path.exists())
            self.assertTrue(artifacts.duration_plot_png_path.exists())


if __name__ == "__main__":
    unittest.main()
