from __future__ import annotations

from collections import Counter

from kinematic_classifier_sandbox.rung_sufficiency import analyze_rung_sufficiency


def test_failure_diagnosis_separates_feature_and_switching_modes():
    result = analyze_rung_sufficiency(seed=7, trajectories_per_case=6)
    counts = Counter(row["failure_mode"] for row in result.failure_mode_rows)
    assert counts["feature_limited"] > 0
    assert counts["switching_state_failure"] > 0
    assert counts["prior_limited"] > 0


def test_corpus_precondition_rows_include_evaluable_and_blocked_cases():
    result = analyze_rung_sufficiency(seed=7, trajectories_per_case=6)
    assert any(bool(row["can_evaluate_classifier"]) for row in result.corpus_precondition_rows)
    assert any(not bool(row["can_evaluate_classifier"]) for row in result.corpus_precondition_rows)


def test_learnability_surface_exposes_oracle_and_overlap_signals():
    result = analyze_rung_sufficiency(seed=7, trajectories_per_case=6)
    assert any(float(row["pairwise_auc"]) >= 0.70 for row in result.learnability_surface_rows)
    assert any(row["learnability_status"] in {"algorithm_limited", "close_to_limit"} for row in result.learnability_surface_rows)
