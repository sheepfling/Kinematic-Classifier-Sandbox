from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.inference.windowed_baseline import (
    WindowedTrajectory,
    extract_windowed_feature_rows,
    render_windowed_benchmark_png_bytes,
    render_windowed_benchmark_report,
    render_windowed_benchmark_svg,
    run_windowed_benchmark,
    write_windowed_benchmark_artifacts,
)


class WindowedBaselineTests(unittest.TestCase):
    def test_running_extrema_match_prefix_bruteforce(self) -> None:
        trajectory = WindowedTrajectory(
            trajectory_id="traj",
            true_class="low",
            scenario_name="manual",
            seed=1,
            times=(0.0, 1.0, 2.0, 3.0, 4.0),
            measurements=(0.0, -1.0, 2.0, 1.5, -0.5),
        )
        rows = extract_windowed_feature_rows(trajectory, window_size=3)

        running_min = []
        running_max = []
        current_min = trajectory.measurements[0]
        current_max = trajectory.measurements[0]
        for value in trajectory.measurements:
            current_min = min(current_min, value)
            current_max = max(current_max, value)
            running_min.append(current_min)
            running_max.append(current_max)

        self.assertEqual([row.running_min for row in rows], running_min)
        self.assertEqual([row.running_max for row in rows], running_max)

    def test_fixed_window_features_match_bruteforce(self) -> None:
        trajectory = WindowedTrajectory(
            trajectory_id="traj",
            true_class="low",
            scenario_name="manual",
            seed=1,
            times=(0.0, 1.0, 2.0, 3.0, 4.0),
            measurements=(0.0, 1.0, 4.0, 2.0, 3.0),
        )
        rows = extract_windowed_feature_rows(trajectory, window_size=3, trim_fraction=0.2)
        final_row = rows[-1]
        window_values = list(trajectory.measurements[-3:])
        self.assertEqual(final_row.window_min, min(window_values))
        self.assertEqual(final_row.window_max, max(window_values))
        self.assertAlmostEqual(final_row.window_mean, sum(window_values) / len(window_values))
        self.assertGreaterEqual(final_row.trimmed_range, 0.0)

    def test_outlier_sensitivity_is_lower_for_robust_extrema(self) -> None:
        clean = WindowedTrajectory(
            trajectory_id="clean",
            true_class="low",
            scenario_name="clean",
            seed=1,
            times=(0.0, 1.0, 2.0, 3.0, 4.0),
            measurements=(-1.0, -1.1, -0.9, -1.0, -1.05),
        )
        spiky = WindowedTrajectory(
            trajectory_id="spiky",
            true_class="low",
            scenario_name="spiky",
            seed=2,
            times=(0.0, 1.0, 2.0, 3.0, 4.0),
            measurements=(-1.0, -1.1, 2.8, -1.0, -1.05),
        )
        clean_row = extract_windowed_feature_rows(clean, window_size=5)[-1]
        spiky_row = extract_windowed_feature_rows(spiky, window_size=5)[-1]
        self.assertGreater(spiky_row.window_max - clean_row.window_max, spiky_row.robust_max - clean_row.robust_max)
        self.assertGreater(spiky_row.window_range - clean_row.window_range, spiky_row.trimmed_range - clean_row.trimmed_range)

    def test_duration_bias_is_smaller_for_robust_extrema(self) -> None:
        short = WindowedTrajectory(
            trajectory_id="short",
            true_class="low",
            scenario_name="short",
            seed=1,
            times=(0.0, 1.0, 2.0, 3.0),
            measurements=(-1.0, -0.95, -1.05, -0.90),
        )
        long = WindowedTrajectory(
            trajectory_id="long",
            true_class="low",
            scenario_name="long",
            seed=1,
            times=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0),
            measurements=(-1.0, -0.95, -1.05, -0.90, -0.92, -0.89, -0.87, -0.40),
        )
        short_row = extract_windowed_feature_rows(short, window_size=4)[-1]
        long_row = extract_windowed_feature_rows(long, window_size=4)[-1]
        self.assertGreaterEqual(long_row.window_max, short_row.window_max)
        self.assertLessEqual(long_row.robust_max - short_row.robust_max, long_row.window_max - short_row.window_max)

    def test_windowed_benchmark_artifacts_are_generated(self) -> None:
        result = run_windowed_benchmark(seed=7)
        report = render_windowed_benchmark_report(result)
        svg = render_windowed_benchmark_svg(result)
        png = render_windowed_benchmark_png_bytes(result)

        self.assertIn("Windowed Feature Baseline", report)
        self.assertIn("Raw final accuracy", report)
        self.assertIn("Robust final accuracy", report)
        self.assertIn("<svg", svg)
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))

        self.assertGreater(result.summary.robust_final_accuracy, result.summary.raw_final_accuracy - 0.15)
        self.assertGreaterEqual(result.summary.raw_final_accuracy, 0.5)
        self.assertGreaterEqual(result.summary.robust_final_accuracy, 0.5)

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_windowed_benchmark_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "windowed_baseline")
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.feature_matrix_path.exists())
            self.assertTrue(artifacts.posterior_history_path.exists())
            self.assertTrue(artifacts.confusion_raw_path.exists())
            self.assertTrue(artifacts.confusion_robust_path.exists())
            self.assertTrue(artifacts.plot_png_path.exists())


if __name__ == "__main__":
    unittest.main()
