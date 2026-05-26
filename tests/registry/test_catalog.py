from __future__ import annotations

import unittest

from kinematic_classifier_sandbox.registry.catalog import METHOD_CATALOG, method_families


class CatalogTests(unittest.TestCase):
    def test_catalog_covers_expected_families(self) -> None:
        self.assertEqual(
            method_families(),
            ("advanced", "deep_learning", "model_based", "traditional"),
        )

    def test_each_entry_has_use_case(self) -> None:
        self.assertTrue(METHOD_CATALOG)
        for entry in METHOD_CATALOG:
            self.assertTrue(entry.typical_use_cases)

    def test_catalog_includes_joint_tracking_baseline(self) -> None:
        names = {entry.name for entry in METHOD_CATALOG}
        self.assertIn("Bayesian joint tracking and classification filter bank", names)


if __name__ == "__main__":
    unittest.main()
