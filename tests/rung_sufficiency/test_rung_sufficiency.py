from __future__ import annotations

from kinematic_classifier_sandbox.rung_sufficiency.analysis import analyze_rung_sufficiency
from kinematic_classifier_sandbox.rung_sufficiency.capability_matrix import capability_rows


def test_capability_matrix_orders_rungs():
    rows = capability_rows()
    assert [row["rung_id"] for row in rows] == [
        "pointwise",
        "windowed",
        "sequential_bayes",
        "kalman_bank",
        "transition_matrix",
        "imm",
        "particle_filter",
        "rbpf",
    ]


def test_rung_sufficiency_finds_multiple_decision_types():
    result = analyze_rung_sufficiency(seed=7, trajectories_per_case=6)
    decisions = {row["decision"] for row in result.promotion_rows}
    assert "promote" in decisions
    assert "reject_escalation" in decisions
    assert "defer_advanced" in decisions
    assert "revise_prior" in decisions


def test_switching_witness_promotes_transition_or_imm():
    result = analyze_rung_sufficiency(seed=7, trajectories_per_case=6)
    transition_row = next(
        row
        for row in result.promotion_rows
        if row["study_id"] == "switching_witness_transition_vs_kalman"
    )
    imm_row = next(
        row
        for row in result.promotion_rows
        if row["study_id"] == "switching_witness_imm_vs_transition"
    )
    assert transition_row["decision"] == "promote"
    assert transition_row["candidate_next_rung_id"] == "transition_matrix"
    assert imm_row["candidate_next_rung_id"] == "imm"
    assert imm_row["decision"] in {"defer_advanced", "promote"}
    assert imm_row["decision"] != "corpus_limited"
