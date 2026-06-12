from __future__ import annotations

import unittest

from kinematic_classifier_sandbox import registry


class RegistryPackageSurfaceTests(unittest.TestCase):
    def test_curated_surface_exposes_catalog_and_entrypoints(self) -> None:
        self.assertTrue(registry.METHOD_CATALOG)
        self.assertTrue(callable(registry.method_families))
        self.assertTrue(callable(registry.analyze_algorithm_coverage_matrix))
        self.assertTrue(callable(registry.write_algorithm_coverage_matrix_artifacts))
        self.assertTrue(callable(registry.analyze_classifier_family_scorecard))
        self.assertTrue(callable(registry.write_classifier_family_scorecard_artifacts))
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
        self.assertTrue(callable(registry.analyze_neural_sequence_robustness_frontier))
        self.assertTrue(callable(registry.write_neural_sequence_robustness_frontier_artifacts))
        self.assertTrue(callable(registry.analyze_physics_family_promotion_audit))
        self.assertTrue(callable(registry.write_physics_family_promotion_audit_artifacts))
        self.assertTrue(callable(registry.analyze_drcif_interval_promotion_audit))
        self.assertTrue(callable(registry.write_drcif_interval_promotion_audit_artifacts))
        self.assertTrue(callable(registry.analyze_gsf_multimodal_promotion_audit))
        self.assertTrue(callable(registry.write_gsf_multimodal_promotion_audit_artifacts))
        self.assertTrue(callable(registry.analyze_ukf_nonlinear_promotion_audit))
        self.assertTrue(callable(registry.write_ukf_nonlinear_promotion_audit_artifacts))
        self.assertTrue(callable(registry.analyze_imm_switching_promotion_audit))
        self.assertTrue(callable(registry.write_imm_switching_promotion_audit_artifacts))
        self.assertTrue(callable(registry.analyze_ts2vec_backend_parity))
        self.assertTrue(callable(registry.write_ts2vec_backend_parity_artifacts))
        self.assertTrue(callable(registry.analyze_method_validation_os))
        self.assertTrue(callable(registry.write_method_validation_os_artifacts))
        self.assertTrue(callable(registry.analyze_strict_equation_audit))
        self.assertTrue(callable(registry.write_strict_equation_audit_artifacts))


if __name__ == "__main__":
    unittest.main()
