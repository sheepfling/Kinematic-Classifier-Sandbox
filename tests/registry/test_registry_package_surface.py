from __future__ import annotations

import unittest

from kinematic_classifier_sandbox import registry


class RegistryPackageSurfaceTests(unittest.TestCase):
    def test_curated_surface_exposes_catalog_and_entrypoints(self) -> None:
        self.assertTrue(registry.METHOD_CATALOG)
        self.assertTrue(callable(registry.method_families))
        self.assertTrue(callable(registry.analyze_algorithm_coverage_matrix))
        self.assertTrue(callable(registry.write_algorithm_coverage_matrix_artifacts))
        self.assertTrue(callable(registry.analyze_corpus_evaluation_gap_matrix))
        self.assertTrue(callable(registry.write_corpus_evaluation_gap_matrix_artifacts))
        self.assertTrue(callable(registry.analyze_exported_surface_coverage))
        self.assertTrue(callable(registry.write_exported_surface_coverage_artifacts))
        self.assertTrue(callable(registry.analyze_formal_math_registry))
        self.assertTrue(callable(registry.write_formal_math_registry_artifacts))
        self.assertTrue(callable(registry.analyze_functional_surface_catalog))
        self.assertTrue(callable(registry.write_functional_surface_catalog_artifacts))
        self.assertTrue(callable(registry.analyze_embedding_baseline_frontier))
        self.assertTrue(callable(registry.write_embedding_baseline_frontier_artifacts))
        self.assertTrue(callable(registry.analyze_method_validation_os))
        self.assertTrue(callable(registry.write_method_validation_os_artifacts))
        self.assertTrue(callable(registry.analyze_strict_equation_audit))
        self.assertTrue(callable(registry.write_strict_equation_audit_artifacts))


if __name__ == "__main__":
    unittest.main()
