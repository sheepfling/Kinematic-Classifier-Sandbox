from __future__ import annotations

import unittest
from pathlib import Path

import yaml


class ProductTestMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cls.matrix_path = cls.root / "docs" / "testing" / "test_matrix.yaml"
        cls.matrix = yaml.safe_load(cls.matrix_path.read_text(encoding="utf-8"))
        cls.suites = cls.matrix["suites"]

    def test_matrix_is_structurally_complete(self) -> None:
        self.assertEqual(self.matrix["matrix_version"], "product-test-matrix-v1")
        suite_ids = [suite["suite_id"] for suite in self.suites]
        self.assertEqual(len(suite_ids), len(set(suite_ids)))
        self.assertGreaterEqual(len(self.suites), 10)

        required_fields = {
            "suite_id",
            "product",
            "title",
            "pytest_marker",
            "paths",
            "tier",
            "parallel_safe",
            "cross_product_gate",
            "artifact_scope",
            "runnable",
            "status",
            "promotion_gate",
            "notes",
        }
        allowed_products = {"product1", "product2", "product3", "product4", "shared", "cross_product", "all"}
        for suite in self.suites:
            self.assertTrue(required_fields.issubset(suite), suite["suite_id"])
            self.assertIn(suite["product"], allowed_products)
            self.assertIsInstance(suite["paths"], list)
            self.assertIsInstance(suite["parallel_safe"], bool)
            self.assertIsInstance(suite["cross_product_gate"], bool)
            self.assertIsInstance(suite["runnable"], bool)
            for relative_path in suite["paths"]:
                self.assertTrue((self.root / relative_path).exists(), relative_path)

    def test_product_boundaries_and_release_gates_are_explicit(self) -> None:
        by_id = {suite["suite_id"]: suite for suite in self.suites}
        self.assertEqual(by_id["product1_static_admissibility"]["pytest_marker"], "product1")
        self.assertEqual(by_id["product2_classifier_ladder"]["pytest_marker"], "product2")
        self.assertEqual(by_id["product3_corpus_exploration"]["pytest_marker"], "product3")
        self.assertEqual(by_id["shared_analysis_and_repository"]["pytest_marker"], "cross_product")

        lane_ids = {
            "product4_land_surface",
            "product4_sea_surface",
            "product4_sea_subsurface",
            "product4_air_atmospheric",
            "product4_space_near",
            "product4_space_orbital",
        }
        self.assertTrue(lane_ids.issubset(by_id))
        self.assertTrue(all(by_id[suite_id]["parallel_safe"] for suite_id in lane_ids))
        self.assertFalse(by_id["product4_all"]["parallel_safe"])

        analysis = by_id["product4_analysis"]
        self.assertEqual(analysis["pytest_marker"], "product4_analysis")
        self.assertTrue(analysis["parallel_safe"])
        self.assertFalse(analysis["cross_product_gate"])

        bridge = by_id["real_world_classifier_bridge"]
        self.assertFalse(bridge["runnable"])
        self.assertEqual(bridge["status"], "blocked_until_prepared_snapshot")
        self.assertTrue(bridge["cross_product_gate"])
        self.assertFalse(by_id["full_repository_gate"]["parallel_safe"])

    def test_product3_excludes_observed_real_world_corpus(self) -> None:
        product3 = next(suite for suite in self.suites if suite["suite_id"] == "product3_corpus_exploration")
        self.assertIn("tests/corpus/real_world", product3["excludes"])


if __name__ == "__main__":
    unittest.main()
