from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.registry.algorithm_coverage_matrix import (
    ALGORITHM_COVERAGE_MATRIX,
    analyze_algorithm_coverage_matrix,
    write_algorithm_coverage_matrix_artifacts,
)


class AlgorithmCoverageMatrixTests(unittest.TestCase):
    def test_registry_covers_expected_lanes_and_methods(self) -> None:
        lanes = {entry.lane for entry in ALGORITHM_COVERAGE_MATRIX}
        self.assertEqual(
            lanes,
            {
                "calibration_uncertainty",
                "core_physics_probabilistic",
                "learning_evidence",
                "learned_filters",
                "neural_sequence",
                "optimizer_generator",
                "representation_learning",
                "time_series_baselines",
                "tracking_fusion_extension",
            },
        )
        method_ids = {entry.method_id for entry in ALGORITHM_COVERAGE_MATRIX}
        self.assertIn("rocket_family", method_ids)
        self.assertIn("dtw_knn", method_ids)
        self.assertIn("shapelet_family", method_ids)
        self.assertIn("drcif_interval_forests", method_ids)
        self.assertIn("dictionary_tde_family", method_ids)
        self.assertIn("hive_cote", method_ids)
        self.assertIn("changepoint_detection", method_ids)
        self.assertIn("hsmm", method_ids)
        self.assertIn("bocpd", method_ids)
        self.assertIn("ukf_ekf_ckf", method_ids)
        self.assertIn("student_t_robust_kalman", method_ids)
        self.assertIn("lstm_gru", method_ids)
        self.assertIn("patchtst_style_encoder", method_ids)
        self.assertIn("kalmannet_family", method_ids)
        self.assertIn("ts2vec_family", method_ids)
        self.assertIn("ts_tcc_softclt", method_ids)
        self.assertIn("masked_timeseries_autoencoder", method_ids)
        self.assertIn("tabular_feature_ml", method_ids)
        self.assertIn("sequence_ml_baselines", method_ids)
        self.assertIn("unsupervised_discovery", method_ids)
        self.assertIn("calibration_wrappers", method_ids)
        self.assertIn("deep_uncertainty_wrappers", method_ids)
        self.assertIn("ood_detection", method_ids)
        self.assertIn("glmb_pmbm", method_ids)
        self.assertIn("random_doe_search", method_ids)
        self.assertIn("latin_hypercube", method_ids)
        self.assertIn("cem", method_ids)
        self.assertIn("cmaes", method_ids)
        self.assertIn("stateless_rl_policy_search", method_ids)
        self.assertIn("bayesian_optimization", method_ids)
        self.assertIn("mpc_adversarial_generator", method_ids)
        self.assertIn("map_elites", method_ids)
        self.assertIn("ppo", method_ids)
        self.assertIn("sac", method_ids)
        self.assertIn("td3", method_ids)

        hsmm_row = next(entry for entry in ALGORITHM_COVERAGE_MATRIX if entry.method_id == "hsmm")
        bocpd_row = next(entry for entry in ALGORITHM_COVERAGE_MATRIX if entry.method_id == "bocpd")
        shapelet_row = next(entry for entry in ALGORITHM_COVERAGE_MATRIX if entry.method_id == "shapelet_family")
        drcif_row = next(entry for entry in ALGORITHM_COVERAGE_MATRIX if entry.method_id == "drcif_interval_forests")
        dictionary_row = next(entry for entry in ALGORITHM_COVERAGE_MATRIX if entry.method_id == "dictionary_tde_family")
        hive_row = next(entry for entry in ALGORITHM_COVERAGE_MATRIX if entry.method_id == "hive_cote")
        ukf_row = next(entry for entry in ALGORITHM_COVERAGE_MATRIX if entry.method_id == "ukf_ekf_ckf")
        student_t_row = next(
            entry for entry in ALGORITHM_COVERAGE_MATRIX if entry.method_id == "student_t_robust_kalman"
        )
        tcn_row = next(entry for entry in ALGORITHM_COVERAGE_MATRIX if entry.method_id == "tcn_inceptiontime")
        cmaes_row = next(entry for entry in ALGORITHM_COVERAGE_MATRIX if entry.method_id == "cmaes")
        ppo_row = next(entry for entry in ALGORITHM_COVERAGE_MATRIX if entry.method_id == "ppo")
        sac_row = next(entry for entry in ALGORITHM_COVERAGE_MATRIX if entry.method_id == "sac")
        td3_row = next(entry for entry in ALGORITHM_COVERAGE_MATRIX if entry.method_id == "td3")
        ts2vec_row = next(entry for entry in ALGORITHM_COVERAGE_MATRIX if entry.method_id == "ts2vec_family")
        self.assertEqual(hsmm_row.status, "implemented")
        self.assertEqual(bocpd_row.status, "implemented")
        self.assertEqual(ukf_row.status, "implemented")
        self.assertEqual(student_t_row.status, "implemented")
        self.assertEqual(shapelet_row.status, "witness_supported")
        self.assertEqual(drcif_row.status, "benchmark_candidate")
        self.assertEqual(dictionary_row.status, "benchmark_candidate")
        self.assertEqual(hive_row.status, "benchmark_candidate")
        self.assertEqual(tcn_row.status, "witness_supported")
        self.assertEqual(cmaes_row.status, "implemented")
        self.assertEqual(ppo_row.status, "implemented")
        self.assertEqual(sac_row.status, "implemented")
        self.assertEqual(td3_row.status, "implemented")
        self.assertTrue(ts2vec_row.online)
        self.assertEqual(ts2vec_row.status, "witness_supported")

    def test_summary_counts_are_nonempty(self) -> None:
        result = analyze_algorithm_coverage_matrix()
        self.assertGreater(result.summary["entry_count"], 0)
        self.assertGreater(result.summary["lane_count"], 0)
        self.assertIn("implemented", result.summary["statuses"])
        self.assertIn("core_physics_probabilistic", result.summary["lanes"])

    def test_artifacts_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_algorithm_coverage_matrix_artifacts(Path(temp_dir))
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.matrix_path.exists())
            self.assertTrue(artifacts.inventory_path.exists())
            self.assertTrue(artifacts.plot_path.exists())


if __name__ == "__main__":
    unittest.main()
