from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.milestones import (
    list_milestones,
    resolve_milestone_ids,
    run_milestones,
)


class MilestoneRunnerTests(unittest.TestCase):
    def test_milestone_registry_lists_expected_surface(self) -> None:
        milestone_ids = [entry.milestone_id for entry in list_milestones()]
        self.assertEqual(milestone_ids, [f"m{index}" for index in range(10)])
        statuses = {entry.milestone_id: entry.status for entry in list_milestones()}
        self.assertEqual(statuses["m9"], "done")

    def test_resolve_milestone_ranges(self) -> None:
        self.assertEqual(resolve_milestone_ids("m1"), ("m1",))
        self.assertEqual(resolve_milestone_ids("m1-m9"), tuple(f"m{index}" for index in range(1, 10)))
        self.assertEqual(resolve_milestone_ids("all"), tuple(f"m{index}" for index in range(1, 10)))

    def test_runner_can_write_selected_milestone_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            results = run_milestones(temp_dir, selection="m1")
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].milestone_id, "m1")
            self.assertEqual(results[0].artifact_dir, Path(temp_dir) / "pointwise_baseline")
            self.assertTrue(results[0].artifact_dir.exists())
            self.assertTrue(results[0].report_path.exists())

    def test_runner_can_write_m9_surface(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            results = run_milestones(temp_dir, selection="m9")
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].artifact_dir, Path(temp_dir) / "trajectory_generator_v1")
            self.assertTrue(results[0].artifact_dir.exists())
            self.assertTrue(results[0].report_path.exists())


if __name__ == "__main__":
    unittest.main()
