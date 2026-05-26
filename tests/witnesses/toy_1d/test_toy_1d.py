from __future__ import annotations

import unittest

from kinematic_classifier_sandbox.witnesses.surface import WitnessSurface
from kinematic_classifier_sandbox.witnesses.toy_1d.core import (
    FEATURE_NAMES,
    ToyBenchmarkConfig,
    default_class_specs,
    gaussian_interval_probability,
    generate_toy_dataset,
    generate_toy_track,
    render_toy_benchmark_markdown,
    render_toy_benchmark_png_bytes,
    render_toy_feature_confusion_png_bytes,
    run_class_bank,
    run_toy_benchmark,
    summarize_runs,
    toy_witness_surface,
    write_toy_benchmark_artifact,
    write_toy_benchmark_plot_artifacts,
    write_toy_benchmark_trace_csv,
    write_toy_feature_confusion_artifacts,
)
from kinematic_classifier_sandbox.witnesses.toy_1d.posterior_explainer import (
    render_posterior_comparison_markdown,
    render_posterior_comparison_png_bytes,
    render_posterior_explainer_markdown,
    render_posterior_explainer_png_bytes,
    render_posterior_failure_markdown,
    render_posterior_failure_png_bytes,
    render_posterior_margin_trace_markdown,
    render_posterior_margin_trace_png_bytes,
    write_posterior_comparison_artifacts,
    write_posterior_explainer_artifacts,
    write_posterior_failure_artifacts,
    write_posterior_margin_trace_artifacts,
)


class Toy1DBenchmarkTests(unittest.TestCase):
    def test_surface_contract_exposes_problem_specific_classes_and_features(self) -> None:
        surface = toy_witness_surface()
        self.assertEqual(surface.study_id, "toy_1d")
        self.assertEqual(surface.class_specs, default_class_specs())
        self.assertEqual(surface.feature_names, FEATURE_NAMES)
        self.assertIsInstance(surface, WitnessSurface)

    def test_generator_is_reproducible_for_same_seed(self) -> None:
        spec = default_class_specs()[0]
        first = generate_toy_track(spec=spec, steps=24, dt=1.0, seed=11, obs_sigma=0.5)
        second = generate_toy_track(spec=spec, steps=24, dt=1.0, seed=11, obs_sigma=0.5)
        third = generate_toy_track(spec=spec, steps=24, dt=1.0, seed=12, obs_sigma=0.5)

        self.assertEqual(first.positions_true, second.positions_true)
        self.assertEqual(first.positions_obs, second.positions_obs)
        self.assertNotEqual(first.positions_obs, third.positions_obs)

    def test_gaussian_interval_probability_behaves_sensibly(self) -> None:
        inside = gaussian_interval_probability(mean=0.0, variance=0.04, limit=1.0)
        edge = gaussian_interval_probability(mean=1.0, variance=0.25, limit=1.0)
        outside = gaussian_interval_probability(mean=2.5, variance=0.04, limit=1.0)

        self.assertGreater(inside, edge)
        self.assertGreater(edge, outside)
        self.assertGreaterEqual(outside, 0.0)
        self.assertLessEqual(inside, 1.0)

    def test_class_bank_normalizes_without_nan_collapse(self) -> None:
        config = ToyBenchmarkConfig(steps=18, obs_sigma=0.6)
        track = generate_toy_track(
            spec=default_class_specs()[2],
            steps=config.steps,
            dt=config.dt,
            seed=23,
            obs_sigma=config.obs_sigma,
        )
        run = run_class_bank(track=track, class_specs=config.class_specs, config=config)

        for step in run.steps:
            total = sum(step.updated_class_weights.values())
            self.assertAlmostEqual(total, 1.0, places=6)
            for value in step.updated_class_weights.values():
                self.assertGreaterEqual(value, 0.0)
                self.assertTrue(value == value)
            for terms in step.log_likelihood_terms.values():
                for value in terms.values():
                    self.assertTrue(value == value)

    def test_seeded_benchmark_separates_classes_and_retains_unknown(self) -> None:
        config = ToyBenchmarkConfig(steps=32, obs_sigma=0.75)
        dataset = generate_toy_dataset(config=config, tracks_per_class=8, seed=7)
        runs = tuple(run_class_bank(track, dataset.class_specs, config) for track in dataset.tracks)
        summary = summarize_runs(runs)

        self.assertEqual(summary.total_runs, 48)
        self.assertEqual(
            set(summary.per_class_accuracy),
            {"brake", "coast", "drift", "maneuver", "powered", "unknown"},
        )
        self.assertGreaterEqual(summary.overall_accuracy, 0.40)
        self.assertGreaterEqual(summary.per_class_accuracy["drift"], 0.80)
        self.assertGreaterEqual(summary.per_class_accuracy["powered"], 0.85)
        self.assertGreaterEqual(summary.per_class_accuracy["unknown"], 0.80)
        self.assertGreater(summary.unknown_retention_mean, 0.10)
        self.assertGreaterEqual(summary.transient_accuracy, 0.30)
        self.assertGreaterEqual(summary.terminal_accuracy, 0.30)
        self.assertEqual(len(summary.entropy_mean_by_step), config.steps - 1)
        self.assertEqual(set(summary.mean_feature_probability_by_step), set(FEATURE_NAMES))
        self.assertIn("tp", summary.feature_confusion_counts["high_speed"])
        self.assertIn("coast_nominal", summary.scenario_confusion_counts)
        self.assertIn("steady", summary.phase_confusion_counts)
        self.assertIn("coast_nominal", summary.scenario_phase_hit_counts)
        self.assertIn("steady", summary.scenario_phase_total_counts["coast_nominal"])
        self.assertIn("reverse_motion", summary.class_feature_detection_counts["drift"])

    def test_benchmark_report_artifact_contains_metrics(self) -> None:
        result = run_toy_benchmark(seed=7, steps=20, tracks_per_class=2, obs_sigma=0.6)
        markdown = render_toy_benchmark_markdown(result)
        self.assertIn("1D Toy Bayesian Benchmark Report", markdown)
        self.assertIn("Overall accuracy", markdown)
        self.assertIn("Transient accuracy", markdown)
        self.assertIn("Terminal accuracy", markdown)
        self.assertIn("Class vs Detected Feature Counts", markdown)
        self.assertIn("Phase Confusion Counts", markdown)
        self.assertIn("Scenario Phase Target Hit Rates", markdown)
        self.assertIn("Feature Semantics", markdown)
        self.assertIn("Feature-Class Precision and Recall", markdown)
        self.assertIn("Feature Lift by Class", markdown)
        self.assertIn("True Feature vs Predicted Class Matrix", markdown)
        self.assertIn("Detected Feature vs Predicted Class Matrix", markdown)
        self.assertIn("Mean Posterior Entropy by Step", markdown)
        png = render_toy_benchmark_png_bytes(result)
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = write_toy_benchmark_trace_csv(result, temp_dir)
            self.assertEqual(csv_path, Path(temp_dir) / "toy_1d_benchmark_traces.csv")
            self.assertTrue(csv_path.exists())
            csv_text = csv_path.read_text(encoding="utf-8")
            self.assertIn("scenario_name", csv_text.splitlines()[0])
            self.assertIn("posterior_entropy", csv_text.splitlines()[0])
            self.assertIn("true_phase_label", csv_text.splitlines()[0])
            self.assertIn("target_phase_label", csv_text.splitlines()[0])
            self.assertIn("transient_map_class", csv_text.splitlines()[0])
            self.assertIn("agg_hard_brake", csv_text.splitlines()[0])
            png_path = write_toy_benchmark_plot_artifacts(temp_dir, result=result)
            self.assertEqual(png_path, Path(temp_dir) / "toy_1d_benchmark_posteriors.png")
            self.assertTrue(png_path.exists())
            feature_png = render_toy_feature_confusion_png_bytes(result)
            self.assertTrue(feature_png.startswith(b"\x89PNG\r\n\x1a\n"))
            feature_png_path = write_toy_feature_confusion_artifacts(temp_dir, result=result)
            self.assertEqual(feature_png_path, Path(temp_dir) / "toy_1d_feature_confusion.png")
            self.assertTrue(feature_png_path.exists())

    def test_write_benchmark_artifact_creates_markdown(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = write_toy_benchmark_artifact(temp_dir, seed=7, steps=20, tracks_per_class=2, obs_sigma=0.6)
            self.assertEqual(output_path, Path(temp_dir) / "toy_1d_benchmark_summary.md")
            self.assertTrue(output_path.exists())
            self.assertIn("Overall accuracy", output_path.read_text(encoding="utf-8"))

    def test_posterior_explainer_artifacts_are_generated(self) -> None:
        import tempfile
        from pathlib import Path

        result = run_toy_benchmark(seed=7, steps=20, tracks_per_class=2, obs_sigma=0.6)
        markdown = render_posterior_explainer_markdown(result)
        png = render_posterior_explainer_png_bytes(result)

        self.assertIn("Posterior Update Walkthrough", markdown)
        self.assertIn("Bayesian Update", markdown)
        self.assertIn("Composite Log-Likelihood Terms by Class", markdown)
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))

        with tempfile.TemporaryDirectory() as temp_dir:
            markdown_path, png_path = write_posterior_explainer_artifacts(temp_dir, result=result)
            self.assertEqual(markdown_path, Path(temp_dir) / "toy_1d_posterior_walkthrough.md")
            self.assertEqual(png_path, Path(temp_dir) / "toy_1d_posterior_walkthrough.png")
            self.assertTrue(markdown_path.exists())
            self.assertTrue(png_path.exists())

    def test_posterior_failure_artifacts_are_generated(self) -> None:
        import tempfile
        from pathlib import Path

        result = run_toy_benchmark(seed=7, steps=20, tracks_per_class=2, obs_sigma=0.6)
        markdown = render_posterior_failure_markdown(result)
        png = render_posterior_failure_png_bytes(result)

        self.assertIn("Posterior Failure Walkthrough", markdown)
        self.assertIn("Aggregate predicted class", markdown)
        self.assertIn("Composite Log-Likelihood Terms by Class", markdown)
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))

        with tempfile.TemporaryDirectory() as temp_dir:
            markdown_path, png_path = write_posterior_failure_artifacts(temp_dir, result=result)
            self.assertEqual(markdown_path, Path(temp_dir) / "toy_1d_posterior_failure_walkthrough.md")
            self.assertEqual(png_path, Path(temp_dir) / "toy_1d_posterior_failure_walkthrough.png")
            self.assertTrue(markdown_path.exists())
            self.assertTrue(png_path.exists())

    def test_posterior_comparison_artifacts_are_generated(self) -> None:
        import tempfile
        from pathlib import Path

        result = run_toy_benchmark(seed=7, steps=20, tracks_per_class=2, obs_sigma=0.6)
        markdown = render_posterior_comparison_markdown(result)
        png = render_posterior_comparison_png_bytes(result)

        self.assertIn("Toy 1D Posterior Comparison", markdown)
        self.assertIn("Side-by-Side Posterior Terms", markdown)
        self.assertIn("Decision Margins", markdown)
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))

        with tempfile.TemporaryDirectory() as temp_dir:
            markdown_path, png_path = write_posterior_comparison_artifacts(temp_dir, result=result)
            self.assertEqual(markdown_path, Path(temp_dir) / "toy_1d_posterior_comparison.md")
            self.assertEqual(png_path, Path(temp_dir) / "toy_1d_posterior_comparison.png")
            self.assertTrue(markdown_path.exists())
            self.assertTrue(png_path.exists())

    def test_posterior_margin_trace_artifacts_are_generated(self) -> None:
        import tempfile
        from pathlib import Path

        result = run_toy_benchmark(seed=7, steps=20, tracks_per_class=2, obs_sigma=0.6)
        markdown = render_posterior_margin_trace_markdown(result)
        png = render_posterior_margin_trace_png_bytes(result)

        self.assertIn("Toy 1D Posterior Margin Trace", markdown)
        self.assertIn("Posterior margin", markdown)
        self.assertIn("Stepwise Margins", markdown)
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))

        with tempfile.TemporaryDirectory() as temp_dir:
            markdown_path, png_path = write_posterior_margin_trace_artifacts(temp_dir, result=result)
            self.assertEqual(markdown_path, Path(temp_dir) / "toy_1d_posterior_margin_trace.md")
            self.assertEqual(png_path, Path(temp_dir) / "toy_1d_posterior_margin_trace.png")
            self.assertTrue(markdown_path.exists())
            self.assertTrue(png_path.exists())


if __name__ == "__main__":
    unittest.main()
