from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.analysis.archive_vs_physics_witness import (
    analyze_archive_vs_physics_witness,
    write_archive_vs_physics_witness_artifacts,
)


class ArchiveVsPhysicsWitnessTests(unittest.TestCase):
    def test_archive_vs_physics_witness_artifacts_are_generated(self) -> None:
        result = analyze_archive_vs_physics_witness(seed=1009, trajectories_per_case=8)

        self.assertEqual(result.metrics["study_id"], "archive_vs_physics_witness_v1")
        method_names = {row.method_name for row in result.method_rows}
        self.assertEqual(
            method_names,
            {
                "windowed_robust",
                "kalman_bank",
                "minirocket_family",
                "drcif_interval_forests",
                "dictionary_tde_family",
                "hive_cote",
            },
        )
        self.assertEqual(len(result.scenario_winner_rows), 4)
        self.assertIn(
            result.metrics["promotion_decision"],
            {
                "promote_archive_vs_physics_witness_for_followon_review",
                "hold_archive_vs_physics_witness_until_nonfallback_external_execution",
                "record_archive_vs_physics_witness_keep_gate_closed",
            },
        )
        self.assertTrue(str(result.metrics["next_gate"]))
        self.assertGreaterEqual(float(result.metrics["archive_champion_test_accuracy"]), 0.0)

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_archive_vs_physics_witness_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "archive_vs_physics_witness_v1")
            self.assertTrue(artifacts.method_summary_path.exists())
            self.assertTrue(artifacts.scenario_winners_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.metrics_path.exists())
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.decision_card_path.exists())
            self.assertEqual(len(artifacts.plot_paths), 2)
            self.assertTrue(all(path.exists() for path in artifacts.plot_paths))
            report_text = artifacts.report_path.read_text(encoding="utf-8")
            self.assertIn("Archive vs Physics Witness", report_text)
            self.assertIn("archive integration read", report_text)
            self.assertIn("next gate", report_text)


if __name__ == "__main__":
    unittest.main()
