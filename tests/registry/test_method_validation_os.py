from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.registry.method_validation_os import (
    analyze_method_validation_os,
    write_method_validation_os_artifacts,
)


class MethodValidationOSTests(unittest.TestCase):
    def test_analysis_exposes_method_and_witness_registry(self) -> None:
        result = analyze_method_validation_os()
        self.assertGreater(len(result.method_rows), 10)
        self.assertGreater(len(result.witness_rows), 10)
        self.assertEqual(len(result.family_maturity_rows), 4)
        method_ids = {row.method_id for row in result.method_rows}
        witness_ids = {row.witness_id for row in result.witness_rows}
        family_ids = {str(row["family_id"]) for row in result.family_maturity_rows}
        self.assertIn("particle_filter", method_ids)
        self.assertIn("rbpf", method_ids)
        self.assertIn("ukf", method_ids)
        self.assertIn("gaussian_sum_filter", method_ids)
        self.assertIn("tcn", method_ids)
        self.assertIn("drcif_interval_forests", method_ids)
        self.assertIn("dictionary_tde_family", method_ids)
        self.assertIn("temperature_scaling", method_ids)
        self.assertIn("tabular_feature_ml", method_ids)
        self.assertIn("sequence_ml_baselines", method_ids)
        self.assertIn("unsupervised_discovery", method_ids)
        self.assertIn("cmaes", method_ids)
        self.assertIn("map_elites", method_ids)
        self.assertIn("kalmannet", method_ids)
        self.assertIn("differentiable_pf", method_ids)
        self.assertIn("abs_range_multimodal_1d", witness_ids)
        self.assertIn("latent_maneuver_onset_duration", witness_ids)
        self.assertIn("neural_sequence_vs_physics_frontier", witness_ids)
        self.assertIn("neural_sequence_robustness_frontier", witness_ids)
        self.assertIn("tsc_archive_baseline_frontier", witness_ids)
        self.assertIn("archive_family_promotion_audit", witness_ids)
        self.assertIn("continuous_generator_frontier", witness_ids)
        self.assertIn("coverage_archive_diversity_frontier", witness_ids)
        self.assertIn("sequential_control_generator_frontier", witness_ids)
        self.assertIn("sequential_offpolicy_control_frontier", witness_ids)
        self.assertIn("embedding_baseline_frontier", witness_ids)
        self.assertIn("archive_backend_diagnosis", witness_ids)
        self.assertIn("learned_model_mismatch", witness_ids)
        self.assertEqual(
            family_ids,
            {
                "interpretable_kinematic_classifiers",
                "physics_aware_inference_classifiers",
                "generic_time_series_benchmark_classifiers",
                "learned_sequence_and_embedding_classifiers",
            },
        )
        pf_row = next(row for row in result.method_rows if row.method_id == "particle_filter")
        rbpf_row = next(row for row in result.method_rows if row.method_id == "rbpf")
        gsf_row = next(row for row in result.method_rows if row.method_id == "gaussian_sum_filter")
        ukf_row = next(row for row in result.method_rows if row.method_id == "ukf")
        student_t_row = next(row for row in result.method_rows if row.method_id == "student_t_kalman")
        hsmm_row = next(row for row in result.method_rows if row.method_id == "hsmm")
        bocpd_row = next(row for row in result.method_rows if row.method_id == "bocpd")
        shapelet_row = next(row for row in result.method_rows if row.method_id == "shapelet")
        minirocket_row = next(row for row in result.method_rows if row.method_id == "minirocket_family")
        boosted_row = next(row for row in result.method_rows if row.method_id == "gradient_boosted_features")
        tcn_row = next(row for row in result.method_rows if row.method_id == "tcn")
        inception_row = next(row for row in result.method_rows if row.method_id == "inceptiontime")
        hive_row = next(row for row in result.method_rows if row.method_id == "hive_cote")
        drcif_row = next(row for row in result.method_rows if row.method_id == "drcif_interval_forests")
        dictionary_row = next(row for row in result.method_rows if row.method_id == "dictionary_tde_family")
        temperature_row = next(row for row in result.method_rows if row.method_id == "temperature_scaling")
        conformal_row = next(row for row in result.method_rows if row.method_id == "conformal_wrapper")
        tabular_ml_row = next(row for row in result.method_rows if row.method_id == "tabular_feature_ml")
        sequence_ml_row = next(row for row in result.method_rows if row.method_id == "sequence_ml_baselines")
        unsupervised_row = next(row for row in result.method_rows if row.method_id == "unsupervised_discovery")
        cmaes_row = next(row for row in result.method_rows if row.method_id == "cmaes")
        map_elites_row = next(row for row in result.method_rows if row.method_id == "map_elites")
        sac_td3_row = next(row for row in result.method_rows if row.method_id == "sac_td3")
        ts2vec_row = next(row for row in result.method_rows if row.method_id == "ts2vec")
        kalmannet_row = next(row for row in result.method_rows if row.method_id == "kalmannet")
        differentiable_pf_row = next(row for row in result.method_rows if row.method_id == "differentiable_pf")
        self.assertEqual(pf_row.current_status, "study_justified")
        self.assertEqual(rbpf_row.current_status, "study_justified")
        self.assertEqual(rbpf_row.current_failure_status, "")
        self.assertEqual(gsf_row.current_status, "study_justified")
        self.assertEqual(gsf_row.current_failure_status, "")
        self.assertEqual(ukf_row.current_status, "study_justified")
        self.assertEqual(ukf_row.current_failure_status, "")
        self.assertEqual(student_t_row.current_status, "witness_supported")
        imm_row = next(row for row in result.method_rows if row.method_id == "imm")
        hmm_row = next(row for row in result.method_rows if row.method_id == "hmm_transition")
        kalman_bank_row = next(row for row in result.method_rows if row.method_id == "kalman_bank")
        self.assertEqual(imm_row.current_status, "study_justified")
        self.assertEqual(imm_row.current_failure_status, "")
        self.assertEqual(hmm_row.current_status, "witness_supported")
        self.assertEqual(hmm_row.current_failure_status, "")
        self.assertEqual(kalman_bank_row.current_status, "witness_supported")
        self.assertEqual(kalman_bank_row.current_failure_status, "")
        self.assertEqual(hsmm_row.current_status, "witness_supported")
        self.assertEqual(bocpd_row.current_status, "witness_supported")
        self.assertEqual(shapelet_row.current_status, "witness_supported")
        self.assertEqual(minirocket_row.current_status, "witness_supported")
        self.assertEqual(boosted_row.current_status, "witness_supported")
        self.assertEqual(tcn_row.current_status, "witness_supported")
        self.assertEqual(inception_row.current_status, "witness_supported")
        self.assertEqual(hive_row.current_status, "witness_supported")
        self.assertEqual(drcif_row.current_status, "trace_validated")
        self.assertEqual(dictionary_row.current_status, "witness_supported")
        self.assertEqual(temperature_row.current_status, "witness_supported")
        self.assertEqual(conformal_row.current_status, "witness_supported")
        self.assertEqual(tabular_ml_row.current_status, "witness_supported")
        self.assertEqual(sequence_ml_row.current_status, "researched")
        self.assertEqual(unsupervised_row.current_status, "researched")
        self.assertEqual(cmaes_row.current_status, "witness_supported")
        self.assertEqual(map_elites_row.current_status, "witness_supported")
        self.assertEqual(sac_td3_row.current_status, "implemented")
        self.assertEqual(sac_td3_row.current_failure_status, "insufficient_evidence")
        self.assertEqual(ts2vec_row.current_status, "witness_supported")
        self.assertEqual(ts2vec_row.current_failure_status, "insufficient_evidence")
        self.assertEqual(kalmannet_row.current_status, "researched")
        self.assertEqual(kalmannet_row.current_failure_status, "missing")
        self.assertEqual(differentiable_pf_row.current_status, "researched")
        self.assertEqual(differentiable_pf_row.current_failure_status, "missing")
        physics_family_row = next(row for row in result.family_maturity_rows if row["family_id"] == "physics_aware_inference_classifiers")
        generic_tsc_family_row = next(row for row in result.family_maturity_rows if row["family_id"] == "generic_time_series_benchmark_classifiers")
        self.assertEqual(physics_family_row["implemented"], "yes")
        self.assertEqual(physics_family_row["integrated"], "yes")
        self.assertEqual(physics_family_row["proven"], "yes")
        self.assertGreaterEqual(int(physics_family_row["proven_method_count"]), 10)
        self.assertIn("witness-backed", str(physics_family_row["primary_blocker"]))
        self.assertEqual(generic_tsc_family_row["implemented"], "yes")
        self.assertEqual(generic_tsc_family_row["integrated"], "yes")
        self.assertEqual(generic_tsc_family_row["proven"], "yes")
        self.assertIn("DrCIF", str(generic_tsc_family_row["primary_blocker"]))

    def test_artifacts_write_status_and_coverage_matrices(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_method_validation_os_artifacts(temp_dir)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "method_validation_os_v1")
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.method_specs_path.exists())
            self.assertTrue(artifacts.witness_specs_path.exists())
            self.assertTrue(artifacts.promotion_status_matrix_path.exists())
            self.assertTrue(artifacts.witness_coverage_matrix_path.exists())
            self.assertTrue(artifacts.lane_summary_path.exists())
            self.assertTrue(artifacts.family_maturity_matrix_path.exists())
            self.assertTrue(artifacts.promotion_status_plot_path.exists())
            self.assertTrue(artifacts.witness_coverage_plot_path.exists())

            summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
            self.assertGreater(summary["method_count"], 10)
            self.assertGreater(summary["witness_count"], 10)

            with artifacts.promotion_status_matrix_path.open(encoding="utf-8", newline="") as handle:
                status_rows = list(csv.DictReader(handle))
            with artifacts.witness_coverage_matrix_path.open(encoding="utf-8", newline="") as handle:
                coverage_rows = list(csv.DictReader(handle))
            with artifacts.family_maturity_matrix_path.open(encoding="utf-8", newline="") as handle:
                family_rows = list(csv.DictReader(handle))
            self.assertTrue(any(row["method_id"] == "particle_filter" for row in status_rows))
            self.assertTrue(any(row["method_id"] == "rbpf" for row in status_rows))
            self.assertTrue(any(row["method_id"] == "tabular_feature_ml" for row in status_rows))
            self.assertTrue(any(row["family_id"] == "physics_aware_inference_classifiers" for row in family_rows))
            pf_row = next(row for row in status_rows if row["method_id"] == "particle_filter")
            self.assertEqual(pf_row["study_justified"], "yes")
            gsf_row = next(row for row in status_rows if row["method_id"] == "gaussian_sum_filter")
            self.assertEqual(gsf_row["study_justified"], "yes")
            ukf_row = next(row for row in status_rows if row["method_id"] == "ukf")
            self.assertEqual(ukf_row["study_justified"], "yes")
            imm_row = next(row for row in status_rows if row["method_id"] == "imm")
            self.assertEqual(imm_row["study_justified"], "yes")
            rbpf_row = next(row for row in status_rows if row["method_id"] == "rbpf")
            self.assertEqual(rbpf_row["study_justified"], "yes")
            hmm_row = next(row for row in status_rows if row["method_id"] == "hmm_transition")
            self.assertEqual(hmm_row["witness_supported"], "yes")
            kalman_bank_row = next(row for row in status_rows if row["method_id"] == "kalman_bank")
            self.assertEqual(kalman_bank_row["witness_supported"], "yes")
            drcif_row = next(row for row in status_rows if row["method_id"] == "drcif_interval_forests")
            self.assertEqual(drcif_row["trace_validated"], "yes")
            self.assertEqual(drcif_row["witness_supported"], "no")
            physics_row = next(row for row in family_rows if row["family_id"] == "physics_aware_inference_classifiers")
            self.assertEqual(physics_row["implemented"], "yes")
            self.assertEqual(physics_row["integrated"], "yes")
            self.assertEqual(physics_row["proven"], "yes")
            generic_row = next(row for row in family_rows if row["family_id"] == "generic_time_series_benchmark_classifiers")
            self.assertEqual(generic_row["implemented"], "yes")
            self.assertEqual(generic_row["integrated"], "yes")
            self.assertEqual(generic_row["proven"], "yes")
            learned_row = next(row for row in family_rows if row["family_id"] == "learned_sequence_and_embedding_classifiers")
            self.assertEqual(learned_row["implemented"], "yes")
            self.assertEqual(learned_row["integrated"], "yes")
            self.assertEqual(learned_row["proven"], "yes")
            self.assertTrue(learned_row["next_high_signal_move"])
            self.assertTrue(
                any(
                    row["witness_id"] == "abs_range_multimodal_1d"
                    and row["method_id"] == "particle_filter"
                    and row["designed_to_justify"] == "yes"
                    for row in coverage_rows
                )
            )


if __name__ == "__main__":
    unittest.main()
