from __future__ import annotations

import unittest

from kinematic_classifier_sandbox.advanced_filters.contracts import AdvancedFilterStep
from kinematic_classifier_sandbox.advanced_filters.protocols import validate_advanced_filter_step


class AdvancedFilterContractTests(unittest.TestCase):
    def test_valid_step_accepts_posterior_contract(self) -> None:
        step = AdvancedFilterStep(
            trajectory_id="t0",
            time=1.0,
            filter_id="dummy",
            predicted_label="a",
            confidence=0.7,
            posterior_by_label={"a": 0.7, "b": 0.3},
            log_evidence_by_label={"a": -0.1, "b": -1.0},
            diagnostics={"ok": True},
        )
        validate_advanced_filter_step(step)

    def test_invalid_step_rejects_non_normalized_posterior(self) -> None:
        step = AdvancedFilterStep(
            trajectory_id="t0",
            time=1.0,
            filter_id="dummy",
            predicted_label="a",
            confidence=0.7,
            posterior_by_label={"a": 0.7, "b": 0.7},
            log_evidence_by_label={"a": -0.1, "b": -1.0},
            diagnostics={},
        )
        with self.assertRaises(ValueError):
            validate_advanced_filter_step(step)


if __name__ == "__main__":
    unittest.main()
