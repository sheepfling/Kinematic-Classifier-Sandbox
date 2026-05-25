from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox import (
    analyze_capability_aware_search,
    write_capability_aware_search_artifacts,
)


class CapabilityAwareSearchTests(unittest.TestCase):
    def test_planner_assigns_distinct_search_policies(self) -> None:
        result = analyze_capability_aware_search()
        plans = {row["family"]: row for row in result.backend_plan_rows}

        parameter_plan = plans["parameter_only_1d"]
        self.assertFalse(parameter_plan["sequential_methods_enabled"])
        self.assertNotIn("adaptive_stress", str(parameter_plan["recommended_methods"]))

        controlled_plan = plans["controlled_1d"]
        self.assertTrue(controlled_plan["sequential_methods_enabled"])
        self.assertIn("adaptive_stress", str(controlled_plan["recommended_methods"]))

        expensive_plan = plans["future_6dof_backend"]
        self.assertTrue(expensive_plan["broad_expensive_search_avoided"])
        self.assertIn("surrogate_assisted", str(expensive_plan["recommended_methods"]))
        self.assertNotIn("random", str(expensive_plan["recommended_methods"]))

        self.assertIn("Selection Matrix", result.report_markdown)

    def test_artifacts_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_capability_aware_search_artifacts(temp_dir)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "capability_aware_search")
            self.assertTrue(artifacts.search_planner_rules_path.exists())
            self.assertTrue(artifacts.search_method_selection_matrix_path.exists())
            self.assertTrue(artifacts.backend_search_plan_path.exists())
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.selection_matrix_png_path.exists())
            self.assertTrue(artifacts.decision_tree_png_path.exists())
            self.assertTrue(artifacts.cost_coverage_frontier_png_path.exists())

            payload = json.loads(artifacts.search_planner_rules_path.read_text(encoding="utf-8"))
            self.assertIn("runtime_rules", payload)


if __name__ == "__main__":
    unittest.main()
