from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import yaml

from kinematic_classifier_sandbox.corpus.exploration.generic_corpus_exploration import (
    DEFAULT_GENERIC_CORPUS_EXPLORATION_WEIGHTS,
    analyze_generic_corpus_exploration,
)
from kinematic_classifier_sandbox.corpus.policy import (
    DEFAULT_CORPUS_POLICY_PATH,
    load_corpus_policy_spec,
    score_corpus_autodevelopment_candidate,
    score_corpus_gym_reward,
    score_qd_archive_elite,
    score_study_candidate_monte_carlo,
    score_study_candidate_static,
    validate_corpus_policy_spec,
    write_default_policy_artifacts,
)


class CorpusPolicyTests(unittest.TestCase):
    def test_default_policy_validates_and_normalizes(self) -> None:
        policy = load_corpus_policy_spec()
        self.assertEqual(policy.policy_id, "default_corpus_policy_v1")
        self.assertAlmostEqual(sum(policy.generic_explorer_weights.values()), 1.0)
        self.assertAlmostEqual(sum(policy.corpus_positive_weights.values()), 1.0)
        self.assertAlmostEqual(sum(policy.corpus_penalty_weights.values()), 1.0)

    def test_default_policy_reproduces_current_generic_explorer_weights(self) -> None:
        policy = load_corpus_policy_spec(DEFAULT_CORPUS_POLICY_PATH)
        result_from_default = analyze_generic_corpus_exploration(seed=7)
        from_policy = analyze_generic_corpus_exploration(
            seed=7,
            weights=DEFAULT_GENERIC_CORPUS_EXPLORATION_WEIGHTS.__class__(
                validity=policy.generic_explorer_weights["validity"],
                coverage_novelty=policy.generic_explorer_weights["coverage_novelty"],
                boundary=policy.generic_explorer_weights["boundary_score"],
                stress=policy.generic_explorer_weights["classifier_stress"],
                environment=policy.generic_explorer_weights["environment_score"],
                provenance=policy.generic_explorer_weights["provenance_completeness"],
            ),
        )
        self.assertEqual(
            result_from_default.selected_corpus_manifest["selected_rows"],
            from_policy.selected_corpus_manifest["selected_rows"],
        )

    def test_negative_weights_fail_validation(self) -> None:
        policy = load_corpus_policy_spec()
        invalid = replace(policy, generic_explorer_weights={**policy.generic_explorer_weights, "validity": -0.1})
        with self.assertRaises(ValueError):
            validate_corpus_policy_spec(invalid)

    def test_missing_weights_fail_loading(self) -> None:
        payload = yaml.safe_load(DEFAULT_CORPUS_POLICY_PATH.read_text(encoding="utf-8"))
        del payload["generic_explorer"]["weights"]["validity"]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad_policy.yaml"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_corpus_policy_spec(path)

    def test_schema_and_default_artifacts_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            schema, default = write_default_policy_artifacts(temp_dir)
            self.assertTrue(schema.exists())
            self.assertTrue(default.exists())
            self.assertIn("CorpusPolicySpec", schema.read_text(encoding="utf-8"))

    def test_default_policy_reproduces_remaining_scoring_literals(self) -> None:
        policy = load_corpus_policy_spec()
        self.assertAlmostEqual(
            score_corpus_gym_reward(
                policy,
                class_validity=0.7,
                feature_excitation=0.2,
                coverage_gain=0.3,
                boundary_closeness=0.4,
                classifier_stress=0.5,
                prior_sensitivity=0.6,
                leakage_penalty=0.1,
                physical_invalidity_penalty=0.2,
            ),
            0.22 * 0.7 + 0.14 * 0.2 + 0.14 * 0.3 + 0.14 * 0.4 + 0.14 * 0.5 + 0.12 * 0.6 - 0.10 * 0.1 - 0.14 * 0.2,
        )
        self.assertAlmostEqual(
            score_qd_archive_elite(
                policy,
                validity_score=0.7,
                acceleration_range_pressure=0.2,
                classifier_stress=0.3,
                mean_margin_pressure=0.4,
            ),
            0.30 * 0.7 + 0.25 * 0.2 + 0.25 * 0.3 + 0.20 * 0.4,
        )
        self.assertAlmostEqual(
            score_corpus_autodevelopment_candidate(
                policy,
                balance_score=0.1,
                boundary_coverage_score=0.2,
                feature_excitation_score=0.3,
                difficulty_diversity_score=0.4,
                leakage_penalty=0.05,
                triviality_penalty=0.06,
                degeneracy_penalty=0.07,
            ),
            0.1 + 0.2 + 0.3 + 0.4 - 0.05 - 0.06 - 0.07,
        )
        self.assertAlmostEqual(
            score_study_candidate_static(
                policy,
                feature_class_compatibility=0.8,
                expected_separability=0.7,
                classifier_assumption_fit=0.6,
                corpus_coverage=0.5,
                dimensional_transfer=0.4,
                implementation_readiness=0.3,
                feature_dependency_risk=0.2,
                cumulative_double_counting_risk=0.1,
                prior_sensitivity_risk=0.05,
            ),
            0.18 * 0.8 + 0.18 * 0.7 + 0.14 * 0.6 + 0.14 * 0.5 + 0.12 * 0.4 + 0.12 * 0.3 + 0.12 * 0.8 - 0.10 * 0.1 - 0.10 * 0.05,
        )
        self.assertAlmostEqual(
            score_study_candidate_monte_carlo(
                policy,
                accuracy=0.8,
                prior_flip_fraction=0.2,
                oracle_gap=0.1,
            ),
            0.60 * 0.8 + 0.25 * 0.8 + 0.15 * 0.9,
        )


if __name__ == "__main__":
    unittest.main()
