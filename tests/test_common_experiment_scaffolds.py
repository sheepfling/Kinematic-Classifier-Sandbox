from __future__ import annotations

import json
import unittest
from pathlib import Path

from kinematic_classifier_sandbox import (
    default_dataset_tiers,
    default_trajectory_class_definitions,
)


class CommonExperimentScaffoldTests(unittest.TestCase):
    def test_common_experiment_scaffolds_exist_and_reference_known_classes(self) -> None:
        root = Path(__file__).resolve().parents[1]
        scaffold_dir = root / "experiments" / "common_1d_classifier_study"
        boundary_dir = root / "experiments" / "common_1d_boundary_study"

        config_path = scaffold_dir / "common_experiment_config.yaml"
        feature_sets_path = scaffold_dir / "feature_sets.json"
        class_pairs_path = scaffold_dir / "class_pair_manifest.json"
        classifier_manifest_path = scaffold_dir / "classifier_manifest.json"
        boundary_config_path = boundary_dir / "common_experiment_config.yaml"
        boundary_pairs_path = boundary_dir / "class_pair_manifest.json"

        self.assertTrue(config_path.exists())
        self.assertTrue(feature_sets_path.exists())
        self.assertTrue(class_pairs_path.exists())
        self.assertTrue(classifier_manifest_path.exists())
        self.assertTrue(boundary_config_path.exists())
        self.assertTrue(boundary_pairs_path.exists())

        config_text = config_path.read_text(encoding="utf-8")
        self.assertIn("name: common_1d_classifier_study", config_text)
        self.assertIn("study_adapter_id: common_1d_classifier_study", config_text)
        self.assertIn("generator: trajectory_generator_v1", config_text)
        self.assertIn("metrics_by_sensor_regime.csv", config_text)
        self.assertIn("metrics_by_classifier_and_feature_set.csv", config_text)
        self.assertIn("feature_set_comparison.csv", config_text)
        self.assertIn("irregular_window_comparison.csv", config_text)
        self.assertIn("class_pair_duration_study.csv", config_text)
        self.assertIn("covariate_leakage_audit.csv", config_text)
        self.assertIn("identifiability_matrix.csv", config_text)

        feature_sets = json.loads(feature_sets_path.read_text(encoding="utf-8"))
        self.assertEqual(
            set(feature_sets),
            {
                "instantaneous",
                "raw_extrema",
                "robust_extrema",
                "shape_window",
                "model_residuals",
                "all_engineered",
            },
        )
        self.assertEqual(feature_sets["instantaneous"]["history_behavior"], "instantaneous")
        self.assertIn("includes", feature_sets["all_engineered"])
        self.assertIn("features", feature_sets["shape_window"])
        self.assertIn("features", feature_sets["model_residuals"])
        self.assertIn("position_range", feature_sets["instantaneous"]["features"])
        self.assertIn("outlier_score", feature_sets["model_residuals"]["features"])

        known_classes = {definition.name for definition in default_trajectory_class_definitions()}
        pair_manifest = json.loads(class_pairs_path.read_text(encoding="utf-8"))
        self.assertEqual(len(pair_manifest["class_pairs"]), 5)
        for entry in pair_manifest["class_pairs"]:
            left, right = entry["pair"]
            self.assertIn(left, known_classes)
            self.assertIn(right, known_classes)
            self.assertNotEqual(left, right)
            self.assertTrue(entry["primary_separators"])

        tier_names = {tier.name for tier in default_dataset_tiers()}
        for tier_name in ("easy_v1", "boundary_v1", "adversarial_v1", "stress_v1", "realistic_v1"):
            self.assertIn(tier_name, tier_names)
            self.assertIn(f"- {tier_name}", config_text)

        classifier_manifest = json.loads(classifier_manifest_path.read_text(encoding="utf-8"))
        classifier_ids = [entry["id"] for entry in classifier_manifest["classifiers"]]
        self.assertEqual(
            classifier_ids,
            [
                "pointwise",
                "windowed_raw_extrema",
                "windowed_robust_extrema",
                "windowed_shape_features",
                "bayes_accumulator",
                "kalman_bank",
            ],
        )
        produced_fields = {field for entry in classifier_manifest["classifiers"] for field in entry["produces"]}
        self.assertIn("posterior_history", produced_fields)
        self.assertIn("log_likelihoods", produced_fields)

        boundary_config_text = boundary_config_path.read_text(encoding="utf-8")
        self.assertIn("name: common_1d_boundary_study", boundary_config_text)
        self.assertIn("study_adapter_id: common_1d_boundary_study", boundary_config_text)
        self.assertIn("manifest_path: experiments/common_1d_boundary_study/class_pair_manifest.json", boundary_config_text)

        boundary_pair_manifest = json.loads(boundary_pairs_path.read_text(encoding="utf-8"))
        self.assertEqual(len(boundary_pair_manifest["class_pairs"]), 3)
        boundary_pairs = {tuple(entry["pair"]) for entry in boundary_pair_manifest["class_pairs"]}
        self.assertEqual(
            boundary_pairs,
            {
                ("constant_acceleration", "maneuver"),
                ("constant_velocity", "braking"),
                ("maneuver", "bounded_acceleration"),
            },
        )


if __name__ == "__main__":
    unittest.main()
