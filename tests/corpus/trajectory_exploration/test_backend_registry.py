from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.corpus.trajectory_exploration.backend_registry import (
    analyze_exploration_backend_registry,
    default_exploration_backend_specs,
    exploration_backend_family_summary,
    write_exploration_backend_registry_artifacts,
)


class ExplorationBackendRegistryTests(unittest.TestCase):
    def test_registry_covers_current_and_planned_backend_families(self) -> None:
        specs = default_exploration_backend_specs()
        backend_ids = {spec.backend_id for spec in specs}
        self.assertIn("heuristic_search", backend_ids)
        self.assertIn("blackbox_optimizer", backend_ids)
        self.assertIn("map_elites", backend_ids)
        self.assertIn("cmaes", backend_ids)
        self.assertIn("bayesian_optimization", backend_ids)
        self.assertIn("ppo", backend_ids)
        self.assertIn("sac", backend_ids)
        self.assertIn("td3", backend_ids)
        self.assertIn("mpc_adversarial", backend_ids)

        cmaes_spec = next(spec for spec in specs if spec.backend_id == "cmaes")
        ppo_spec = next(spec for spec in specs if spec.backend_id == "ppo")
        sac_spec = next(spec for spec in specs if spec.backend_id == "sac")
        td3_spec = next(spec for spec in specs if spec.backend_id == "td3")
        self.assertEqual(cmaes_spec.implementation_status, "implemented")
        self.assertEqual(ppo_spec.implementation_status, "implemented")
        self.assertEqual(sac_spec.implementation_status, "implemented")
        self.assertEqual(td3_spec.implementation_status, "implemented")

    def test_family_summary_reports_multiple_search_families(self) -> None:
        rows = exploration_backend_family_summary()
        families = {row["family"] for row in rows}
        self.assertEqual(
            families,
            {
                "baseline_search",
                "black_box_optimization",
                "quality_diversity",
                "reinforcement_learning",
                "trajectory_optimization",
            },
        )
        rl_row = next(row for row in rows if row["family"] == "reinforcement_learning")
        self.assertGreaterEqual(int(rl_row["backend_count"]), 3)
        self.assertGreaterEqual(int(rl_row["sequential_control_count"]), 3)

    def test_artifacts_are_written(self) -> None:
        result = analyze_exploration_backend_registry()
        self.assertGreater(result.summary["backend_count"], 0)
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_exploration_backend_registry_artifacts(Path(temp_dir), result=result)
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.spec_table_path.exists())
            self.assertTrue(artifacts.inventory_path.exists())
            self.assertTrue(artifacts.family_summary_path.exists())
            self.assertTrue(artifacts.capability_plot_path.exists())


if __name__ == "__main__":
    unittest.main()
