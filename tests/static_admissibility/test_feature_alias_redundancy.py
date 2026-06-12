from __future__ import annotations

import unittest

import pandas

from kinematic_classifier_sandbox.static_admissibility.multi_domain_3d import (
    _classify_alias_relationship,
    _linear_equivalence,
    _threshold_action,
    _threshold_subsumption_row,
)


def _meta(
    *,
    base_quantity: str = "altitude",
    aggregation: str = "min",
    unit: str = "m",
    operator: str = "",
    threshold_value: object = "",
    time_scope: str = "full_window",
    expected_uncertainty: float = 8.0,
    derived_from: str = "min_altitude_m",
    semantic_group: str = "altitude",
) -> dict[str, object]:
    return {
        "base_quantity": base_quantity,
        "aggregation": aggregation,
        "unit": unit,
        "operator": operator,
        "threshold_value": threshold_value,
        "time_scope": time_scope,
        "expected_uncertainty": expected_uncertainty,
        "derived_from": derived_from,
        "semantic_group": semantic_group,
    }


class FeatureAliasRedundancyTests(unittest.TestCase):
    def test_exact_duplicate_features_drop_duplicate(self) -> None:
        alias_type, action = _classify_alias_relationship(
            _meta(),
            _meta(),
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            1.0,
            0.0,
        )
        self.assertEqual(alias_type, "duplicate")
        self.assertEqual(action, "drop_duplicate")

    def test_unit_alias_canonicalizes_unit(self) -> None:
        alias_type, action = _classify_alias_relationship(
            _meta(unit="m"),
            _meta(unit="ft"),
            [100.0, 200.0, 300.0],
            [328.1, 656.2, 984.3],
            0.99,
            0.0,
        )
        self.assertEqual(alias_type, "unit_alias")
        self.assertEqual(action, "canonicalize_unit")

    def test_affine_offset_alias_is_detected(self) -> None:
        slope, intercept, rmse, r2 = _linear_equivalence(
            [300.0, 301.0, 302.0],
            [301.0, 302.0, 303.0],
        )
        self.assertAlmostEqual(slope, 1.0, places=3)
        self.assertAlmostEqual(intercept, 1.0, places=3)
        self.assertLess(rmse, 1.0e-6)
        self.assertGreater(r2, 0.999)
        alias_type, action = _classify_alias_relationship(
            _meta(),
            _meta(),
            [300.0, 301.0, 302.0],
            [301.0, 302.0, 303.0],
            0.99,
            0.0,
        )
        self.assertEqual(alias_type, "offset_alias")
        self.assertEqual(action, "canonicalize_unit")

    def test_threshold_alias_without_boundary_samples_collapses(self) -> None:
        frame = pandas.DataFrame(
            {
                "true_class": ["a", "a", "b", "b"],
                "min_altitude_m": [250.0, 280.0, 320.0, 340.0],
                "min_altitude_ge_300m": [0, 0, 1, 1],
                "min_altitude_ge_301m": [0, 0, 1, 1],
            }
        )
        row = _threshold_subsumption_row(
            frame,
            "min_altitude_ge_300m",
            "min_altitude_ge_301m",
            _meta(operator=">=", threshold_value=300),
            _meta(operator=">=", threshold_value=301),
            0.0,
        )
        self.assertEqual(row["boundary_slice_count"], 0)
        self.assertEqual(row["recommended_action"], "collapse_thresholds")

    def test_threshold_alias_boundary_mix_below_uncertainty_is_candidate_level(self) -> None:
        action, confidence, followup = _threshold_action(
            boundary_count=4,
            decision_redundancy_score=0.03,
            observability_ratio=0.125,
        )
        self.assertEqual(action, "retain_pair_specific_candidate")
        self.assertEqual(confidence, "low_to_medium")
        self.assertEqual(followup, "ablation_or_observability_check")

    def test_threshold_alias_boundary_mix_above_uncertainty_can_retain_pair_specific(self) -> None:
        action, confidence, followup = _threshold_action(
            boundary_count=4,
            decision_redundancy_score=0.03,
            observability_ratio=1.5,
        )
        self.assertEqual(action, "retain_pair_specific")
        self.assertEqual(confidence, "medium")
        self.assertEqual(followup, "ablation")

    def test_decision_redundant_given_neighbor_drops_priority(self) -> None:
        action, _, _ = _threshold_action(
            boundary_count=1,
            decision_redundancy_score=0.0,
            observability_ratio=2.0,
        )
        self.assertEqual(action, "collapse_thresholds")


if __name__ == "__main__":
    unittest.main()
