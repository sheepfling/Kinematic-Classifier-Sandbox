from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.corpus.adequacy_audit import (
    analyze_corpus_adequacy,
    write_corpus_adequacy_artifacts,
)


class CorpusAdequacyAuditTests(unittest.TestCase):
    def test_corpus_adequacy_produces_gate_rows_and_recommendations(self) -> None:
        result = analyze_corpus_adequacy(seed=7, trajectories_per_class=5)

        self.assertEqual(result.summary.total_trajectories, 175)
        self.assertFalse(result.summary.overall_pass)
        self.assertEqual(result.summary.overall_status, "fail")
        self.assertGreater(len(result.feature_set_rows), 0)
        self.assertGreater(len(result.class_pair_rows), 0)
        self.assertGreater(len(result.class_balance_rows), 0)
        self.assertGreater(len(result.covariate_rows), 0)
        self.assertGreater(len(result.scorecard_rows), 0)
        self.assertGreaterEqual(result.summary.q_corpus, 0.0)
        self.assertLessEqual(result.summary.q_corpus, 1.0)
        self.assertGreaterEqual(result.summary.leakage_penalty, 0.0)

        pair_lookup = {
            (row["class_a"], row["class_b"]): row
            for row in result.class_pair_rows
        }
        self.assertEqual(pair_lookup[("constant_velocity", "stationary")]["status"], "green")
        self.assertEqual(pair_lookup[("constant_acceleration", "maneuver")]["status"], "red")
        self.assertTrue(all(row["status"] == "green" for row in result.class_balance_rows))
        self.assertTrue(any(row["term"] == "Q_corpus" for row in result.scorecard_rows))

    def test_corpus_adequacy_writes_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_corpus_adequacy_artifacts(temp_dir, seed=7, trajectories_per_class=5)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "corpus_adequacy_audit_v1")
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.feature_set_coverage_path.exists())
            self.assertTrue(artifacts.class_pair_coverage_path.exists())
            self.assertTrue(artifacts.class_balance_path.exists())
            self.assertTrue(artifacts.covariate_leakage_path.exists())
            self.assertTrue(artifacts.scorecard_path.exists())
            self.assertTrue(artifacts.validity_audit_path.exists())
            self.assertTrue(artifacts.degeneracy_report_path.exists())
            self.assertTrue(artifacts.pair_status_heatmap_path.exists())
            self.assertTrue(artifacts.covariate_leakage_plot_path.exists())
            report = artifacts.report_path.read_text(encoding="utf-8")
            self.assertIn("Corpus Adequacy Audit", report)
            self.assertIn("Corpus Scorecard", report)
            self.assertIn("Declared Class-Pair Boundary Coverage", report)
            self.assertIn("Covariate Leakage", report)


if __name__ == "__main__":
    unittest.main()
