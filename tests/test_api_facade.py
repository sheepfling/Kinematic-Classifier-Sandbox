from __future__ import annotations

import unittest

from kinematic_classifier_sandbox import api


class ApiFacadeTests(unittest.TestCase):
    def test_api_exports_canonical_entry_points(self) -> None:
        self.assertTrue(hasattr(api, "analyze_corpus_adequacy"))
        self.assertTrue(hasattr(api, "analyze_generic_corpus_exploration"))
        self.assertTrue(hasattr(api, "write_repo_story_artifacts"))
        self.assertTrue(hasattr(api, "write_functional_surface_catalog_artifacts"))
        self.assertTrue(hasattr(api, "analyze_validation_ladder"))
        self.assertGreater(len(api.__all__), 10)


if __name__ == "__main__":
    unittest.main()
