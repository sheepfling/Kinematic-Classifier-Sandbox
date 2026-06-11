from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.advanced_filters.hsmm_duration_witness import (
    analyze_hsmm_duration_limited_witness,
    hsmm_duration_limited_witness_surface,
    write_hsmm_duration_limited_witness_artifacts,
)


class HSMMDurationWitnessTests(unittest.TestCase):
    def test_hsmm_duration_witness_beats_hmm_on_exit_timing(self) -> None:
        result = analyze_hsmm_duration_limited_witness(seed=503)
        self.assertEqual(
            result.metrics["promotion_decision"],
            "promote_hsmm_for_duration_limited_maneuver",
        )
        self.assertGreater(
            float(result.metrics["hsmm_mode_accuracy"]),
            float(result.metrics["hmm_mode_accuracy"]),
        )
        self.assertLess(
            float(result.metrics["hsmm_maneuver_brier"]),
            float(result.metrics["hmm_maneuver_brier"]),
        )
        self.assertLess(
            int(result.metrics["hsmm_offset_delay_steps"]),
            int(result.metrics["hmm_offset_delay_steps"]),
        )

    def test_hsmm_duration_artifacts_are_rerunnable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_hsmm_duration_limited_witness_artifacts(temp_dir)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "hsmm_duration_limited_maneuver_v1")
            self.assertTrue(artifacts.truth_path.exists())
            self.assertTrue(artifacts.posterior_history_path.exists())
            self.assertTrue(artifacts.chain_posterior_path.exists())
            self.assertTrue(artifacts.state_estimate_history_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.metrics_path.exists())
            self.assertTrue(artifacts.decision_card_path.exists())
            for plot in artifacts.plot_paths:
                self.assertTrue(plot.exists())
            with artifacts.metrics_path.open(encoding="utf-8", newline="") as handle:
                metrics_rows = list(csv.DictReader(handle))
            self.assertEqual(len(metrics_rows), 1)
            self.assertEqual(
                metrics_rows[0]["promotion_decision"],
                "promote_hsmm_for_duration_limited_maneuver",
            )

    def test_hsmm_surface_exposes_expected_study_id(self) -> None:
        surface = hsmm_duration_limited_witness_surface()
        self.assertEqual(surface.study_id, "hsmm_duration_limited_maneuver_v1")


if __name__ == "__main__":
    unittest.main()
