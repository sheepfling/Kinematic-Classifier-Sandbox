from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.api import (
    BaseFeatureComputationContext,
    analyze_feature_datasets,
    FeatureComputationContext,
    OneDimensionalFeatureComputationContext,
    load_feature_registry,
    load_feature_set_manifest,
    resolve_feature_names,
    write_feature_analysis_artifacts,
)


class FeatureAnalysisTests(unittest.TestCase):
    def test_feature_context_split_preserves_backward_compatible_alias(self) -> None:
        self.assertTrue(issubclass(OneDimensionalFeatureComputationContext, BaseFeatureComputationContext))
        self.assertIs(FeatureComputationContext, OneDimensionalFeatureComputationContext)

    def test_feature_registry_carries_metadata_and_extractors(self) -> None:
        registry = load_feature_registry()
        speed_range = registry["speed_range"]

        self.assertEqual(speed_range.name, "speed_range")
        self.assertEqual(speed_range.group, "finite_difference_velocity")
        self.assertEqual(speed_range.role, "kinematics")
        self.assertTrue(speed_range.description)
        self.assertEqual(speed_range.history_behavior, "windowed")
        self.assertTrue(speed_range.geometry_assumption)
        self.assertTrue(speed_range.dimensional_transfer)
        self.assertTrue(speed_range.dependency_tags)
        self.assertTrue(speed_range.sensitivity_tags)
        self.assertEqual(len(speed_range.default_excitation_thresholds), 3)
        self.assertLess(
            speed_range.default_excitation_thresholds[0],
            speed_range.default_excitation_thresholds[1],
        )
        self.assertLess(
            speed_range.default_excitation_thresholds[1],
            speed_range.default_excitation_thresholds[2],
        )
        self.assertTrue(callable(speed_range.extractor))

    def test_feature_set_manifest_resolves_named_subsets(self) -> None:
        manifest = load_feature_set_manifest()
        feature_names = resolve_feature_names(feature_set="shape_window", manifest=manifest)

        self.assertEqual(
            feature_names,
            (
                "velocity_sign_changes",
                "acceleration_sign_changes",
                "monotonicity",
                "linear_fit_residual",
                "quadratic_fit_residual",
            ),
        )

    def test_feature_names_can_be_selected_by_tags(self) -> None:
        sampling_features = resolve_feature_names(required_tags=("sampling",))
        outlier_features = resolve_feature_names(required_tags=("outlier_sensitive",))
        vector_timing_features = resolve_feature_names(required_tags=("vector_compatible", "timing"))

        self.assertIn("mean_dt", sampling_features)
        self.assertIn("outlier_score", outlier_features)
        self.assertIn("sampling_irregularity", vector_timing_features)

    def test_feature_analysis_reports_excitation_and_separability(self) -> None:
        result = analyze_feature_datasets(seed=7, trajectories_per_class=5)

        self.assertGreater(len(result.feature_rows), 0)
        self.assertIn("position_range", result.feature_rows[0].feature_values)
        self.assertIn("position_range", result.summary.top_features)
        self.assertGreater(result.summary.excitation_totals["position_range"]["strong"], 0)
        self.assertEqual(result.summary.feature_set_name, "all_engineered")

        pairwise_lookup = {
            (row["class_a"], row["class_b"]): row["pairwise_auc"]
            for row in result.pairwise_rows
        }
        self.assertGreater(pairwise_lookup[("constant_acceleration", "maneuver")], 0.95)
        self.assertLess(min(pairwise_lookup.values()), 0.85)

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_feature_analysis_artifacts(temp_dir, seed=7, trajectories_per_class=5)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "feature_analysis_v1")
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.feature_matrix_path.exists())
            self.assertTrue(artifacts.feature_summary_path.exists())
            self.assertTrue(artifacts.feature_excitation_path.exists())
            self.assertTrue(artifacts.feature_excitation_summary_path.exists())
            self.assertTrue(artifacts.feature_separation_scores_path.exists())
            self.assertTrue(artifacts.identifiability_matrix_path.exists())
            self.assertTrue(artifacts.pairwise_distance_matrix_path.exists())
            self.assertTrue(artifacts.pairwise_overlap_matrix_path.exists())
            self.assertTrue(artifacts.pairwise_auc_matrix_path.exists())
            self.assertTrue(artifacts.plot_excitation_png_path.exists())
            self.assertTrue(artifacts.plot_distance_png_path.exists())
            self.assertTrue(artifacts.plot_overlap_png_path.exists())
            self.assertTrue(artifacts.plot_scatter_png_path.exists())
            self.assertTrue(artifacts.plot_confusability_png_path.exists())
            self.assertTrue(artifacts.plot_ranking_png_path.exists())
            report = artifacts.report_path.read_text(encoding="utf-8")
            self.assertIn("Feature Excitation", report)
            self.assertIn("Pairwise Separability", report)

    def test_feature_analysis_can_run_with_named_feature_set(self) -> None:
        result = analyze_feature_datasets(seed=7, trajectories_per_class=5, feature_set="shape_window")

        self.assertEqual(result.summary.feature_set_name, "shape_window")
        self.assertEqual(
            result.summary.feature_names,
            (
                "velocity_sign_changes",
                "acceleration_sign_changes",
                "monotonicity",
                "linear_fit_residual",
                "quadratic_fit_residual",
            ),
        )
        self.assertTrue(all(row["feature"] in result.summary.feature_names for row in result.summary_rows))
        self.assertTrue(set(result.summary.feature_names).issubset(set(result.feature_rows[0].feature_values)))

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_feature_analysis_artifacts(
                temp_dir,
                seed=7,
                trajectories_per_class=5,
                feature_set="shape_window",
            )
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "feature_analysis_shape_window_v1")
            report = artifacts.report_path.read_text(encoding="utf-8")
            self.assertIn("Feature set: shape_window", report)


if __name__ == "__main__":
    unittest.main()
