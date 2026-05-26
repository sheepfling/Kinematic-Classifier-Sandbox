from __future__ import annotations

import unittest

import kinematic_classifier_sandbox.api as api


class ApiFacadeTests(unittest.TestCase):
    def test_api_exports_canonical_entry_points(self) -> None:
        self.assertTrue(hasattr(api, "analyze_corpus_adequacy"))
        self.assertTrue(hasattr(api, "analyze_generic_corpus_exploration"))
        self.assertTrue(hasattr(api, "analyze_validation_ladder"))
        self.assertGreater(len(api.__all__), 10)

    def test_meta_surface_is_not_on_the_core_api(self) -> None:
        self.assertFalse(hasattr(api, "write_repo_story_artifacts"))
        self.assertFalse(hasattr(api, "write_functional_surface_catalog_artifacts"))
        self.assertFalse(hasattr(api, "write_methodology_latex_artifacts"))


if __name__ == "__main__":
    unittest.main()
