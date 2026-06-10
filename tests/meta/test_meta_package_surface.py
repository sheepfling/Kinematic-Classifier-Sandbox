from __future__ import annotations

import unittest

from kinematic_classifier_sandbox import meta


class MetaPackageSurfaceTests(unittest.TestCase):
    def test_curated_surface_exposes_methodology_entrypoints(self) -> None:
        self.assertTrue(callable(meta.analyze_methodology_compendium))
        self.assertTrue(callable(meta.write_methodology_compendium_artifacts))
        self.assertTrue(callable(meta.analyze_methodology_latex))
        self.assertTrue(callable(meta.write_methodology_latex_artifacts))
        self.assertTrue(callable(meta.write_methodology_section_symbol_audit_artifacts))


if __name__ == "__main__":
    unittest.main()
