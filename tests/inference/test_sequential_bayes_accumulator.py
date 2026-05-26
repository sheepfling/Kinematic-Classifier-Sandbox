from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.inference.sequential_bayes_accumulator import (
    SequentialBayesAccumulator,
    default_accumulator_class_specs,
    render_accumulator_png_bytes,
    render_accumulator_report,
    render_accumulator_svg,
    run_accumulator_benchmark,
    write_accumulator_artifacts,
)


class SequentialBayesAccumulatorTests(unittest.TestCase):
    def test_hand_computed_update_matches_expected_posterior(self) -> None:
        specs = default_accumulator_class_specs()
        accumulator = SequentialBayesAccumulator(specs)
        accumulator.reset({"A": 0.5, "B": 0.5})

        step = accumulator.update_with_likelihoods(0.0, {"A": 0.8, "B": 0.2})
        self.assertAlmostEqual(step.posterior_weights["A"], 0.8, places=6)
        self.assertAlmostEqual(step.posterior_weights["B"], 0.2, places=6)
        self.assertEqual(step.predicted_class, "A")

    def test_equal_likelihoods_preserve_prior(self) -> None:
        specs = default_accumulator_class_specs()
        accumulator = SequentialBayesAccumulator(specs)
        accumulator.reset({"A": 0.7, "B": 0.3})

        step = accumulator.update_with_likelihoods(0.0, {"A": 0.5, "B": 0.5})
        self.assertAlmostEqual(step.posterior_weights["A"], 0.7, places=6)
        self.assertAlmostEqual(step.posterior_weights["B"], 0.3, places=6)

    def test_repeated_evidence_monotonically_favors_the_matching_class(self) -> None:
        specs = default_accumulator_class_specs()
        accumulator = SequentialBayesAccumulator(specs)
        accumulator.reset({"A": 0.5, "B": 0.5})

        posteriors = []
        for step_index in range(3):
            step = accumulator.update_with_likelihoods(float(step_index), {"A": 0.8, "B": 0.2})
            posteriors.append(step.posterior_weights["A"])
        self.assertGreater(posteriors[1], posteriors[0])
        self.assertGreater(posteriors[2], posteriors[1])

    def test_forgetting_switches_faster_after_evidence_change(self) -> None:
        specs = default_accumulator_class_specs()
        sequence = (
            {"A": 0.9, "B": 0.1},
            {"A": 0.9, "B": 0.1},
            {"A": 0.9, "B": 0.1},
            {"A": 0.1, "B": 0.9},
            {"A": 0.1, "B": 0.9},
            {"A": 0.1, "B": 0.9},
        )

        no_forgetting = SequentialBayesAccumulator(specs, forgetting_factor=1.0)
        no_forgetting.reset({"A": 0.5, "B": 0.5})
        for index, likelihoods in enumerate(sequence):
            no_forgetting.update_with_likelihoods(float(index), likelihoods)

        forgetting = SequentialBayesAccumulator(specs, forgetting_factor=0.6)
        forgetting.reset({"A": 0.5, "B": 0.5})
        for index, likelihoods in enumerate(sequence):
            forgetting.update_with_likelihoods(float(index), likelihoods)

        self.assertGreater(forgetting.posterior()["B"], no_forgetting.posterior()["B"])

    def test_accumulator_artifacts_are_generated(self) -> None:
        result = run_accumulator_benchmark(seed=7)
        report = render_accumulator_report(result)
        svg = render_accumulator_svg(result)
        png = render_accumulator_png_bytes(result)

        self.assertIn("Sequential Bayesian Accumulator", report)
        self.assertIn("Prior Sensitivity", report)
        self.assertIn("<svg", svg)
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertGreater(result.summary.final_accuracy, 0.5)
        self.assertGreater(result.summary.confidence_crossings, 0)
        self.assertEqual(len(result.prior_sensitivity), 5)

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_accumulator_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "bayes_accumulator")
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.posterior_history_path.exists())
            self.assertTrue(artifacts.log_odds_history_path.exists())
            self.assertTrue(artifacts.confidence_crossings_path.exists())
            self.assertTrue(artifacts.prior_sensitivity_path.exists())
            self.assertTrue(artifacts.plot_png_path.exists())


if __name__ == "__main__":
    unittest.main()
