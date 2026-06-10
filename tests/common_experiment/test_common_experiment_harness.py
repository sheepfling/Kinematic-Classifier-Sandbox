from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.common_experiment.artifact_io import (
    write_common_experiment_artifacts,
)
from kinematic_classifier_sandbox.common_experiment.config import (
    list_common_studies,
    load_common_experiment_config,
    resolve_common_study_adapter,
)
from kinematic_classifier_sandbox.common_experiment.reporting import render_common_experiment_report
from kinematic_classifier_sandbox.common_experiment.runner import analyze_common_experiment


class CommonExperimentHarnessTests(unittest.TestCase):
    def test_common_experiment_resolves_config_and_study_adapter(self) -> None:
        config = load_common_experiment_config()
        adapter = resolve_common_study_adapter(config)
        self.assertEqual(config.experiment_name, "common_1d_classifier_study")
        self.assertEqual(config.study_adapter_id, "common_1d_classifier_study")
        self.assertEqual(config.output_dir_name, "common_1d_classifier_study")
        self.assertEqual(config.dataset_generator_id, "trajectory_generator_v1")
        self.assertIn(("stationary", "constant_velocity"), config.declared_class_pairs)
        self.assertEqual(config.output_filenames["report_path"], "common_experiment_report.md")
        self.assertEqual(adapter.study_id, "common_1d_classifier_study")
        self.assertTrue(any(study.study_id == adapter.study_id for study in list_common_studies()))
        self.assertTrue(any(study.study_id == "common_1d_boundary_study" for study in list_common_studies()))

    def test_boundary_study_config_and_execution_path(self) -> None:
        root = Path(__file__).resolve().parents[2]
        config_path = root / "experiments" / "common_1d_boundary_study" / "common_experiment_config.yaml"
        config = load_common_experiment_config(config_path)
        adapter = resolve_common_study_adapter(config)

        self.assertEqual(config.experiment_name, "common_1d_boundary_study")
        self.assertEqual(config.study_adapter_id, "common_1d_boundary_study")
        self.assertEqual(adapter.study_id, "common_1d_boundary_study")
        self.assertEqual(
            config.declared_class_pairs,
            (
                ("constant_acceleration", "maneuver"),
                ("constant_velocity", "braking"),
                ("maneuver", "bounded_acceleration"),
            ),
        )

        result = analyze_common_experiment(config_path=config_path, seed=5, trajectories_per_case=2)
        self.assertEqual(result.summary.experiment_name, "common_1d_boundary_study")
        self.assertEqual(result.summary.study_adapter_id, "common_1d_boundary_study")
        self.assertEqual(
            result.summary.executable_class_pairs,
            (
                "constant_acceleration_vs_maneuver",
                "constant_velocity_vs_braking",
                "maneuver_vs_bounded_acceleration",
            ),
        )

    def test_common_experiment_artifacts_are_generated(self) -> None:
        result = analyze_common_experiment(seed=7, trajectories_per_case=4)
        report = render_common_experiment_report(result)

        self.assertIn("Common Experiment Harness", report)
        self.assertEqual(result.summary.experiment_name, "common_1d_classifier_study")
        self.assertEqual(result.summary.study_adapter_id, "common_1d_classifier_study")
        self.assertEqual(
            result.summary.executable_class_pairs,
            (
                "stationary_vs_constant_velocity",
                "constant_velocity_vs_constant_acceleration",
                "constant_acceleration_vs_maneuver",
                "constant_velocity_vs_braking",
                "maneuver_vs_bounded_acceleration",
            ),
        )
        self.assertEqual(len(result.comparison.rows), 9)
        self.assertTrue(result.pair_prediction_rows)
        self.assertTrue(result.feature_rows)
        self.assertTrue(result.covariate_rows)
        self.assertTrue(result.identifiability_rows)
        self.assertTrue(result.oracle_rows)
        self.assertTrue(result.irregular_window_rows)
        self.assertTrue(any("feature_set_id" in row for row in result.oracle_rows))
        self.assertTrue(any(bool(row["is_best_feature_set"]) for row in result.oracle_rows))

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_common_experiment_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "common_1d_classifier_study")
            self.assertTrue(artifacts.config_path.exists())
            self.assertTrue(artifacts.dataset_manifest_path.exists())
            self.assertTrue(artifacts.class_definitions_path.exists())
            self.assertTrue(artifacts.feature_manifest_path.exists())
            self.assertTrue(artifacts.feature_sets_path.exists())
            self.assertTrue(artifacts.class_pair_manifest_path.exists())
            self.assertTrue(artifacts.classifier_manifest_path.exists())
            self.assertTrue(artifacts.sensor_regimes_path.exists())
            self.assertTrue(artifacts.predictions_path.exists())
            self.assertTrue(artifacts.posterior_history_path.exists())
            self.assertTrue(artifacts.likelihood_history_path.exists())
            self.assertTrue(artifacts.feature_matrix_path.exists())
            self.assertTrue(artifacts.metrics_by_classifier_path.exists())
            self.assertTrue(artifacts.metrics_by_sensor_regime_path.exists())
            self.assertTrue(artifacts.metrics_by_classifier_and_feature_set_path.exists())
            self.assertTrue(artifacts.metrics_by_class_pair_path.exists())
            self.assertTrue(artifacts.prior_sensitivity_by_class_pair_path.exists())
            self.assertTrue(artifacts.feature_set_comparison_path.exists())
            self.assertTrue(artifacts.irregular_window_comparison_path.exists())
            self.assertTrue(artifacts.class_pair_duration_study_path.exists())
            self.assertTrue(artifacts.class_pair_scenario_study_path.exists())
            self.assertTrue(artifacts.covariate_leakage_audit_path.exists())
            self.assertTrue(artifacts.feature_excitation_matrix_path.exists())
            self.assertTrue(artifacts.identifiability_matrix_path.exists())
            self.assertTrue(artifacts.oracle_classifier_results_path.exists())
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.canonical_report_path.exists())
            self.assertTrue(artifacts.plots_dir.exists())

            prediction_header = artifacts.predictions_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("run_id", prediction_header)
            self.assertIn("classifier_id", prediction_header)
            self.assertIn("sensor_regime_id", prediction_header)
            self.assertIn("measurement_dim", prediction_header)
            self.assertIn("class_pair_id", prediction_header)
            self.assertIn("scenario_family", prediction_header)
            self.assertIn("dataset_tier", prediction_header)
            self.assertIn("posterior_class_a", prediction_header)

            likelihood_header = artifacts.likelihood_history_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("score_type", likelihood_header)
            self.assertIn("class_a", likelihood_header)
            self.assertIn("log_likelihood_class_a", likelihood_header)

            feature_header = artifacts.feature_matrix_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("feature_set_id", feature_header)
            self.assertIn("curvature_proxy", feature_header)

            feature_set_header = artifacts.feature_set_comparison_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("feature_set_id", feature_set_header)
            self.assertIn("min_pair_accuracy", feature_set_header)

            irregular_window_header = artifacts.irregular_window_comparison_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("window_definition", irregular_window_header)
            self.assertIn("cross_window_prediction_disagreement_rate", irregular_window_header)

            sensor_regime_header = artifacts.metrics_by_sensor_regime_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("sensor_regime_id", sensor_regime_header)
            self.assertIn("same_sensor_fairness_bucket", sensor_regime_header)

            duration_header = artifacts.class_pair_duration_study_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("prefix_accuracy", duration_header)
            self.assertIn("posterior_margin", duration_header)

            scenario_header = artifacts.class_pair_scenario_study_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("scenario_family", scenario_header)
            self.assertIn("overall_accuracy", scenario_header)

            covariate_header = artifacts.covariate_leakage_audit_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("dataset_tier", covariate_header)
            self.assertIn("true_class", covariate_header)
            self.assertIn("max_covariate_delta_ratio", covariate_header)

            excitation_header = artifacts.feature_excitation_matrix_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("dataset_tier", excitation_header)
            self.assertIn("scenario_family", excitation_header)
            self.assertIn("position_range_mean_abs", excitation_header)

            identifiability_header = artifacts.identifiability_matrix_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("mean_standardized_feature_distance", identifiability_header)
            self.assertIn("identifiability_status", identifiability_header)

            oracle_header = artifacts.oracle_classifier_results_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("feature_set_id", oracle_header)
            self.assertIn("best_feature_set_for_pair", oracle_header)
            self.assertIn("is_best_feature_set", oracle_header)

            report_text = artifacts.report_path.read_text(encoding="utf-8")
            canonical_report_text = artifacts.canonical_report_path.read_text(encoding="utf-8")
            self.assertIn("Milestone 10", report_text)
            self.assertIn("Milestone 11", report_text)
            self.assertIn("Covariate Leakage Audit", report_text)
            self.assertIn("Feature Excitation Matrix", report_text)
            self.assertIn("Identifiability Matrix", report_text)
            self.assertIn("Oracle Separability Baseline", report_text)
            self.assertIn("Irregular Window Comparison", report_text)
            self.assertIn("Study adapter", report_text)
            self.assertIn("Metrics By Sensor Regime", report_text)
            self.assertIn("Feature-Set Study", report_text)
            self.assertEqual(report_text, canonical_report_text)

            self.assertTrue((artifacts.plots_dir / "overview" / "dataset_balance.png").exists())
            self.assertTrue((artifacts.plots_dir / "posteriors" / "posterior_example.png").exists())
            self.assertTrue((artifacts.plots_dir / "likelihoods" / "likelihood_example.png").exists())
            self.assertTrue((artifacts.plots_dir / "feature_space" / "identifiability_summary.png").exists())
            self.assertTrue((artifacts.plots_dir / "pca" / "feature_pca_snapshot.png").exists())

    def test_common_experiment_runs_from_config_alias(self) -> None:
        root = Path(__file__).resolve().parents[2]
        source_config = root / "experiments" / "common_1d_classifier_study" / "common_experiment_config.yaml"
        config_text = source_config.read_text(encoding="utf-8")
        config_text = config_text.replace("name: common_1d_classifier_study", "name: common_1d_smoke_alias", 1)
        config_text = config_text.replace("output_dir: artifacts/common_1d_classifier_study", "output_dir: artifacts/common_1d_smoke_alias", 1)
        config_text = config_text.replace("predictions_path: unified_predictions.csv", "predictions_path: smoke_predictions.csv", 1)
        config_text = config_text.replace("report_path: common_experiment_report.md", "report_path: smoke_report.md", 1)
        config_text = config_text.replace("feature_set_comparison_path: feature_set_comparison.csv", "feature_set_comparison_path: smoke_feature_sets.csv", 1)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = temp_path / "common_experiment_config.yaml"
            config_path.write_text(config_text, encoding="utf-8")

            result = analyze_common_experiment(config_path=config_path, seed=11, trajectories_per_case=2)
            artifacts = write_common_experiment_artifacts(temp_path, result=result)

            self.assertEqual(result.summary.experiment_name, "common_1d_smoke_alias")
            self.assertEqual(result.summary.study_adapter_id, "common_1d_classifier_study")
            self.assertEqual(artifacts.run_dir, temp_path / "common_1d_smoke_alias")
            self.assertEqual(artifacts.predictions_path.name, "smoke_predictions.csv")
            self.assertEqual(artifacts.report_path.name, "smoke_report.md")
            self.assertEqual(artifacts.feature_set_comparison_path.name, "smoke_feature_sets.csv")
            self.assertTrue(artifacts.predictions_path.exists())
            self.assertTrue(artifacts.report_path.exists())


if __name__ == "__main__":
    unittest.main()
