from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.advanced_filters.bocpd_onset_witness import (
    analyze_bocpd_unknown_onset_witness,
    bocpd_unknown_onset_witness_surface,
    write_bocpd_unknown_onset_witness_artifacts,
)


class BOCPDOnsetWitnessTests(unittest.TestCase):
    def test_bocpd_unknown_onset_witness_beats_hmm_hsmm(self) -> None:
        result = analyze_bocpd_unknown_onset_witness(seed=607)
        self.assertEqual(
            result.metrics["promotion_decision"],
            "promote_bocpd_for_unknown_maneuver_onset",
        )
        self.assertGreaterEqual(
            float(result.metrics["bocpd_mode_accuracy"]),
            float(result.metrics["hmm_mode_accuracy"]),
        )
        self.assertLess(
            float(result.metrics["bocpd_maneuver_brier"]),
            float(result.metrics["hmm_maneuver_brier"]),
        )
        self.assertLessEqual(
            int(result.metrics["bocpd_onset_delay_steps"]),
            int(result.metrics["hmm_onset_delay_steps"]),
        )

    def test_bocpd_onset_artifacts_are_rerunnable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_bocpd_unknown_onset_witness_artifacts(temp_dir)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "bocpd_unknown_maneuver_onset_v1")
            self.assertTrue(artifacts.truth_path.exists())
            self.assertTrue(artifacts.posterior_history_path.exists())
            self.assertTrue(artifacts.onset_posterior_path.exists())
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
                "promote_bocpd_for_unknown_maneuver_onset",
            )

    def test_bocpd_surface_exposes_expected_study_id(self) -> None:
        surface = bocpd_unknown_onset_witness_surface()
        self.assertEqual(surface.study_id, "bocpd_unknown_maneuver_onset_v1")


if __name__ == "__main__":
    unittest.main()
