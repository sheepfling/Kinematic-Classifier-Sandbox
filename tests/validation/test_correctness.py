from __future__ import annotations

import unittest

from kinematic_classifier_sandbox.validation.correctness import (
    build_correctness_plan,
    correctness_summary,
)


class CorrectnessLadderTests(unittest.TestCase):
    def test_smoke_plan_targets_curated_smoke_lane(self) -> None:
        plan = build_correctness_plan("smoke")
        self.assertEqual(len(plan.steps), 1)
        self.assertEqual(
            plan.steps[0].argv,
            ("python3", "scripts/test.py", "--lane", "correctness-smoke"),
        )

    def test_full_plan_targets_curated_full_lane(self) -> None:
        plan = build_correctness_plan("full")
        self.assertEqual(len(plan.steps), 1)
        self.assertEqual(
            plan.steps[0].argv,
            ("python3", "scripts/test.py", "--lane", "correctness-full"),
        )

    def test_presentation_plan_includes_packet_validators(self) -> None:
        plan = build_correctness_plan("presentation")
        argv_rows = [step.argv for step in plan.steps]
        self.assertIn(("python3", "scripts/test.py", "--lane", "correctness-full"), argv_rows)
        self.assertIn(("python3", "scripts/test.py", "--lane", "correctness-presentation"), argv_rows)
        self.assertEqual(len(plan.steps), 5)
        self.assertIn(
            (
                "python3",
                "scripts/audit/validate_presentation_hero_packet.py",
                "--packet-dir",
                "artifacts/presentation_hero_charts_v5",
            ),
            argv_rows,
        )
        self.assertIn(
            (
                "python3",
                "-m",
                "kinematic_classifier_sandbox",
                "validate-packet",
                "artifacts/packets/static_admissibility_mvp",
            ),
            argv_rows,
        )
        self.assertIn(
            (
                "python3",
                "-m",
                "kinematic_classifier_sandbox",
                "validate-packet",
                "artifacts/packets/corpus_explorer_mvp",
                "--profile",
                "corpus_explorer_mvp",
            ),
            argv_rows,
        )

    def test_summary_mentions_all_ladder_levels(self) -> None:
        summary = correctness_summary("presentation")
        self.assertIn("L0 schema correctness", summary)
        self.assertIn("L1 invariant correctness", summary)
        self.assertIn("L2 toy oracle correctness", summary)
        self.assertIn("L3 statistical regression", summary)
        self.assertIn("L4 claim correctness", summary)


if __name__ == "__main__":
    unittest.main()
