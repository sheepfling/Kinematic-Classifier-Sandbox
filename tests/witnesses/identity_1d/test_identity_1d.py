from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.witnesses.identity_1d.core import (
    FEATURE_NAMES,
    default_identity_class_specs,
    default_identity_hand_authored_scenarios,
    generate_identity_scenarios,
    identity_witness_surface,
    make_identity_scenario,
    render_identity_benchmark_markdown,
    render_identity_benchmark_png_bytes,
    render_identity_feature_confusion_png_bytes,
    run_identity_benchmark,
    run_identity_classifier,
    write_identity_benchmark_artifacts,
    write_identity_benchmark_trace_csv,
    write_identity_feature_confusion_artifacts,
)
from kinematic_classifier_sandbox.witnesses.identity_1d.posterior_explainer import (
    render_identity_posterior_comparison_markdown,
    render_identity_posterior_comparison_png_bytes,
    render_identity_posterior_explainer_markdown,
    render_identity_posterior_explainer_png_bytes,
    render_identity_posterior_failure_markdown,
    render_identity_posterior_failure_png_bytes,
    render_identity_posterior_margin_trace_markdown,
    render_identity_posterior_margin_trace_png_bytes,
    write_identity_posterior_comparison_artifacts,
    write_identity_posterior_explainer_artifacts,
    write_identity_posterior_failure_artifacts,
    write_identity_posterior_margin_trace_artifacts,
)
from kinematic_classifier_sandbox.witnesses.surface import WitnessSurface


class Identity1DBenchmarkTests(unittest.TestCase):
    def test_surface_contract_exposes_problem_specific_classes_and_features(self) -> None:
        surface = identity_witness_surface()
        self.assertEqual(surface.study_id, "identity_1d")
        self.assertEqual(surface.class_specs, default_identity_class_specs())
        self.assertEqual(surface.feature_names, FEATURE_NAMES)
        self.assertIsInstance(surface, WitnessSurface)

    def test_scenarios_are_reproducible(self) -> None:
        first = generate_identity_scenarios(steps=16, seed=9, obs_sigma_mph=1.5)
        second = generate_identity_scenarios(steps=16, seed=9, obs_sigma_mph=1.5)
        third = generate_identity_scenarios(steps=16, seed=10, obs_sigma_mph=1.5)

        self.assertEqual(first[0].speeds_true_mph, second[0].speeds_true_mph)
        self.assertEqual(first[1].speeds_obs_mph, second[1].speeds_obs_mph)
        self.assertNotEqual(first[2].speeds_obs_mph, third[2].speeds_obs_mph)
        self.assertEqual(len(first), 12)

    def test_default_specs_include_third_class(self) -> None:
        class_names = [spec.name for spec in default_identity_class_specs()]
        self.assertEqual(class_names, ["horse", "car", "bike"])

    def test_identity_scenarios_push_expected_posterior_behaviors(self) -> None:
        result = run_identity_benchmark(steps=20, seed=7, obs_sigma_mph=2.0)
        family_runs: dict[str, list] = {}
        for run in result.runs:
            family_runs.setdefault(run.family_name, []).append(run)

        self.assertEqual(result.summary.total_runs, 72)
        self.assertTrue(all(len(runs) == 6 for runs in family_runs.values()))
        self.assertGreater(sum(run.final_weights["horse"] for run in family_runs["horse_cruise"]) / 6.0, 0.70)
        self.assertGreater(sum(run.final_weights["horse"] for run in family_runs["horse_near_limit"]) / 6.0, 0.55)
        self.assertGreater(sum(run.final_weights["car"] for run in family_runs["car_cruise"]) / 6.0, 0.75)
        self.assertGreater(sum(run.final_weights["car"] for run in family_runs["car_sprint"]) / 6.0, 0.75)
        self.assertGreater(sum(run.final_weights["bike"] for run in family_runs["bike_cruise"]) / 6.0, 0.60)
        self.assertGreater(sum(run.final_weights["bike"] for run in family_runs["bike_fast_pack"]) / 6.0, 0.55)
        self.assertGreaterEqual(result.summary.overall_accuracy, 0.75)
        self.assertLess(result.summary.overall_accuracy, 0.95)
        self.assertGreater(result.summary.scenario_confusion_counts["horse_car_border"]["car"], 0)
        self.assertGreater(result.summary.scenario_confusion_counts["horse_car_border"]["horse"], 0)
        self.assertGreater(result.summary.scenario_confusion_counts["bike_horse_border"]["horse"], 0)
        self.assertGreaterEqual(result.summary.overall_accuracy, 0.75)
        self.assertIn("horse", result.summary.confusion_counts)
        self.assertIn("bike_envelope", result.summary.feature_confusion_counts)
        self.assertEqual(set(result.summary.mean_feature_probability_by_step), set(FEATURE_NAMES))

    def test_hand_authored_scenarios_can_be_run_directly(self) -> None:
        specs = default_identity_class_specs()
        scenarios = default_identity_hand_authored_scenarios()
        result = run_identity_benchmark(class_specs=specs, scenarios=scenarios)
        runs = {run.scenario_name: run for run in result.runs}

        self.assertGreater(runs["manual_horse_cruise"].final_weights["horse"], 0.75)
        self.assertGreater(runs["manual_near_horse_limit"].final_weights["horse"], 0.50)
        self.assertGreater(runs["manual_border_dance"].final_weights["car"], 0.55)
        self.assertGreater(runs["manual_car_cruise"].final_weights["car"], 0.60)
        self.assertGreater(runs["manual_bike_cruise"].final_weights["bike"], 0.55)
        self.assertGreater(runs["manual_bike_fast_pack"].final_weights["bike"], 0.55)

    def test_make_identity_scenario_accepts_hand_authored_observations(self) -> None:
        scenario = make_identity_scenario(
            name="manual_probe",
            expected_class="horse",
            seed=77,
            obs_sigma_mph=0.5,
            speeds_true_mph=(31.0, 33.0, 35.0),
            speeds_obs_mph=(31.2, 32.8, 35.4),
        )
        run = run_identity_classifier(scenario, default_identity_class_specs())

        self.assertEqual(scenario.speeds_obs_mph, (31.2, 32.8, 35.4))
        self.assertEqual(len(run.steps), 3)
        self.assertAlmostEqual(sum(run.final_weights.values()), 1.0, places=6)
        self.assertEqual(run.expected_class, "horse")
        self.assertEqual(len(run.steps[0].feature_probabilities), len(FEATURE_NAMES))

    def test_identity_artifacts_are_generated(self) -> None:
        result = run_identity_benchmark(steps=20, seed=7, obs_sigma_mph=2.0)
        markdown = render_identity_benchmark_markdown(result)
        png = render_identity_benchmark_png_bytes(result)

        self.assertIn("1D Identity Speed Benchmark", markdown)
        self.assertIn("Overall accuracy", markdown)
        self.assertIn("Scenario Family Counts", markdown)
        self.assertIn("Class Confusion Counts", markdown)
        self.assertIn("True Feature vs Predicted Class Matrix", markdown)
        self.assertIn("Detected Feature vs Predicted Class Matrix", markdown)
        self.assertIn("Mean Posterior Entropy by Step", markdown)
        self.assertIn("bike_horse_border", markdown)
        self.assertIn("bike=", markdown)
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
        feature_png = render_identity_feature_confusion_png_bytes(result)
        self.assertTrue(feature_png.startswith(b"\x89PNG\r\n\x1a\n"))

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = write_identity_benchmark_trace_csv(result, temp_dir)
            self.assertEqual(csv_path, Path(temp_dir) / "identity_1d_benchmark_traces.csv")
            self.assertTrue(csv_path.exists())
            csv_text = csv_path.read_text(encoding="utf-8")
            self.assertIn("observed_speed_mph", csv_text)
            self.assertIn("posterior_entropy", csv_text)
            self.assertIn("aggregate_map_class", csv_text.splitlines()[0])
            self.assertIn("scenario_family", csv_text.splitlines()[0])
            self.assertIn("bike", csv_text.splitlines()[0])

            artifacts = write_identity_benchmark_artifacts(temp_dir, result=result)
            markdown_path = artifacts.summary_path
            png_path = artifacts.plot_path
            trace_path = artifacts.trace_path
            self.assertEqual(markdown_path, Path(temp_dir) / "identity_1d_benchmark_summary.md")
            self.assertEqual(png_path, Path(temp_dir) / "identity_1d_benchmark_posteriors.png")
            self.assertEqual(trace_path, Path(temp_dir) / "identity_1d_benchmark_traces.csv")
            self.assertTrue(markdown_path.exists())
            self.assertTrue(png_path.exists())
            feature_png_path = write_identity_feature_confusion_artifacts(temp_dir, result=result)
            self.assertEqual(feature_png_path, Path(temp_dir) / "identity_1d_feature_confusion.png")
            self.assertTrue(feature_png_path.exists())

    def test_identity_posterior_explainer_artifacts_are_generated(self) -> None:
        result = run_identity_benchmark(steps=20, seed=7, obs_sigma_mph=2.0)
        markdown = render_identity_posterior_explainer_markdown(result)
        png = render_identity_posterior_explainer_png_bytes(result)

        self.assertIn("Identity Posterior Walkthrough", markdown)
        self.assertIn("Composite Log-Likelihood Terms by Class", markdown)
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))

        with tempfile.TemporaryDirectory() as temp_dir:
            markdown_path, png_path = write_identity_posterior_explainer_artifacts(temp_dir, result=result)
            self.assertEqual(markdown_path, Path(temp_dir) / "identity_1d_posterior_walkthrough.md")
            self.assertEqual(png_path, Path(temp_dir) / "identity_1d_posterior_walkthrough.png")
            self.assertTrue(markdown_path.exists())
            self.assertTrue(png_path.exists())

    def test_identity_posterior_failure_artifacts_are_generated(self) -> None:
        result = run_identity_benchmark(steps=20, seed=7, obs_sigma_mph=2.0)
        markdown = render_identity_posterior_failure_markdown(result)
        png = render_identity_posterior_failure_png_bytes(result)

        self.assertIn("Identity Posterior Failure Walkthrough", markdown)
        self.assertIn("bike_horse_border", markdown)
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))

        with tempfile.TemporaryDirectory() as temp_dir:
            markdown_path, png_path = write_identity_posterior_failure_artifacts(temp_dir, result=result)
            self.assertEqual(markdown_path, Path(temp_dir) / "identity_1d_posterior_failure_walkthrough.md")
            self.assertEqual(png_path, Path(temp_dir) / "identity_1d_posterior_failure_walkthrough.png")
            self.assertTrue(markdown_path.exists())
            self.assertTrue(png_path.exists())

    def test_identity_posterior_comparison_artifacts_are_generated(self) -> None:
        result = run_identity_benchmark(steps=20, seed=7, obs_sigma_mph=2.0)
        markdown = render_identity_posterior_comparison_markdown(result)
        png = render_identity_posterior_comparison_png_bytes(result)

        self.assertIn("Identity Posterior Comparison", markdown)
        self.assertIn("Side-by-Side Posterior Terms", markdown)
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))

        with tempfile.TemporaryDirectory() as temp_dir:
            markdown_path, png_path = write_identity_posterior_comparison_artifacts(temp_dir, result=result)
            self.assertEqual(markdown_path, Path(temp_dir) / "identity_1d_posterior_comparison.md")
            self.assertEqual(png_path, Path(temp_dir) / "identity_1d_posterior_comparison.png")
            self.assertTrue(markdown_path.exists())
            self.assertTrue(png_path.exists())

    def test_identity_posterior_margin_trace_artifacts_are_generated(self) -> None:
        result = run_identity_benchmark(steps=20, seed=7, obs_sigma_mph=2.0)
        markdown = render_identity_posterior_margin_trace_markdown(result)
        png = render_identity_posterior_margin_trace_png_bytes(result)

        self.assertIn("Identity Posterior Margin Trace", markdown)
        self.assertIn("Stepwise Margins", markdown)
        self.assertIn("bike_horse_border", markdown)
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))

        with tempfile.TemporaryDirectory() as temp_dir:
            markdown_path, png_path = write_identity_posterior_margin_trace_artifacts(temp_dir, result=result)
            self.assertEqual(markdown_path, Path(temp_dir) / "identity_1d_posterior_margin_trace.md")
            self.assertEqual(png_path, Path(temp_dir) / "identity_1d_posterior_margin_trace.png")
            self.assertTrue(markdown_path.exists())
            self.assertTrue(png_path.exists())


if __name__ == "__main__":
    unittest.main()
