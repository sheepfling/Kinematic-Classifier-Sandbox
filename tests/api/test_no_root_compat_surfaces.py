from __future__ import annotations

import importlib.util
import unittest


REMOVED_ROOT_COMPAT_MODULES = (
    "kinematic_classifier_sandbox.common_dataset_comparison",
    "kinematic_classifier_sandbox.dimensional_lift_audit",
    "kinematic_classifier_sandbox.feature_analysis",
    "kinematic_classifier_sandbox.inspection_bundle",
    "kinematic_classifier_sandbox.pca_analysis",
    "kinematic_classifier_sandbox.technique_comparison",
)


class NoRootCompatibilitySurfaceTests(unittest.TestCase):
    def test_legacy_root_modules_are_not_importable(self) -> None:
        for module_name in REMOVED_ROOT_COMPAT_MODULES:
            with self.subTest(module_name=module_name):
                self.assertIsNone(importlib.util.find_spec(module_name))


if __name__ == "__main__":
    unittest.main()
