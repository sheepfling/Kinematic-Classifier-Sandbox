from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.api import (
    analyze_advanced_filter_decision,
    render_advanced_filter_decision_report,
    render_advanced_filter_decision_numeric_walkthrough_markdown,
    write_advanced_filter_decision_artifacts,
)


class AdvancedFilterDecisionTests(unittest.TestCase):
    def test_advanced_filter_decision_currently_defers_imm_and_particle_filter(self) -> None:
        result = analyze_advanced_filter_decision()
        self.assertFalse(result.imm_justified)
        self.assertFalse(result.particle_filter_justified)
        self.assertGreater(result.transition_post_switch_gain, 0.0)
        self.assertGreater(result.transition_vs_kalman_post_switch_gain, 0.0)
        self.assertGreater(result.velocity_aided_short_noisy_gain, 0.0)

    def test_advanced_filter_artifacts_are_generated(self) -> None:
        result = analyze_advanced_filter_decision()
        report = render_advanced_filter_decision_report(result)
        walkthrough = render_advanced_filter_decision_numeric_walkthrough_markdown(result)
        self.assertIn("Advanced Filter Decision Report", report)
        self.assertIn("Defer IMM", report)
        self.assertIn("Defer particle filtering", report)
        self.assertIn("Kalman mode bank", report)
        self.assertIn("Advanced Filter Decision Numeric Walkthrough", walkthrough)
        self.assertIn("Why The Decision Is `defer`", walkthrough)
        self.assertIn("Particle-Filter Gate", walkthrough)
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_advanced_filter_decision_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "advanced_filter_decision_v1")
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.evidence_path.exists())
            self.assertTrue(artifacts.numeric_walkthrough_path.exists())


if __name__ == "__main__":
    unittest.main()
