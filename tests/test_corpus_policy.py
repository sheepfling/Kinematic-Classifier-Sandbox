from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import yaml

from kinematic_classifier_sandbox.corpus_policy import (
    DEFAULT_CORPUS_POLICY_PATH,
    load_corpus_policy_spec,
    validate_corpus_policy_spec,
    write_default_policy_artifacts,
)
from kinematic_classifier_sandbox.generic_corpus_exploration import (
    DEFAULT_GENERIC_CORPUS_EXPLORATION_WEIGHTS,
    analyze_generic_corpus_exploration,
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


if __name__ == "__main__":
    unittest.main()
