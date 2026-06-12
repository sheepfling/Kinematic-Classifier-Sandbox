from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.corpus.rl_backend_decision import (
    analyze_rl_backend_decision,
    render_rl_backend_decision_report,
    write_rl_backend_decision_artifacts,
)


class RlBackendDecisionTests(unittest.TestCase):
    def test_rl_backend_decision_currently_defers_rl(self) -> None:
        result = analyze_rl_backend_decision()
        self.assertFalse(result.rl_justified)
        self.assertGreater(result.search_selected_mean_utility, 0.44)
        self.assertGreaterEqual(result.qd_final_coverage_fraction, 0.20)
        self.assertEqual(result.stress_resolved_modes, result.stress_total_modes)
        self.assertLess(result.offpolicy_mean_best_policy_minus_best_baseline, 0.0)
        self.assertLessEqual(result.offpolicy_seed_promotion_rate, 0.5)
        self.assertIn(result.offpolicy_best_policy_backend, {"ppo_policy", "sac", "td3"})
        self.assertIn("matched evaluation budget", result.success_metric.lower())
        self.assertTrue(any(row["criterion"] == "environment_requires_true_sequential_control" for row in result.decision_rows))
        self.assertTrue(any(row["criterion"] == "sequential_offpolicy_frontier_shows_promotion_signal" for row in result.decision_rows))

    def test_rl_backend_decision_artifacts_are_generated(self) -> None:
        result = analyze_rl_backend_decision()
        report = render_rl_backend_decision_report(result)
        self.assertIn("RL Backend Decision Report", report)
        self.assertIn("RL justified now: `False`", report)
        self.assertIn("Keep RL out of fielded-deployment recommendations for now.", report)
        self.assertIn("Off-policy smoke frontier", report)
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_rl_backend_decision_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "rl_corpus_agent")
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.evidence_path.exists())


if __name__ == "__main__":
    unittest.main()
