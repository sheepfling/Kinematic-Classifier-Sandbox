from __future__ import annotations

from pathlib import Path

from kinematic_classifier_sandbox.rung_sufficiency.analysis import analyze_rung_sufficiency, write_rung_sufficiency_artifacts


def test_promotion_rows_include_stay_and_promote():
    result = analyze_rung_sufficiency(seed=7, trajectories_per_case=6)
    decisions = [row["decision"] for row in result.promotion_rows]
    assert "promote" in decisions
    assert "stay" in decisions


def test_artifact_writer_emits_expected_files(tmp_path: Path):
    artifacts = write_rung_sufficiency_artifacts(tmp_path, seed=7, trajectories_per_case=6)
    assert artifacts.threshold_profile_path.exists()
    assert artifacts.capability_matrix_path.exists()
    assert artifacts.corpus_precondition_path.exists()
    assert artifacts.oracle_gap_path.exists()
    assert artifacts.learnability_surface_path.exists()
    assert artifacts.posterior_quality_path.exists()
    assert artifacts.failure_mode_path.exists()
    assert artifacts.promotion_matrix_path.exists()
    assert artifacts.report_path.exists()
    assert artifacts.score_vs_oracle_plot_path.exists()
    assert artifacts.oracle_gap_plot_path.exists()
    assert artifacts.failure_mode_heatmap_path.exists()
    assert artifacts.promotion_decision_plot_path.exists()
    assert artifacts.posterior_quality_plot_path.exists()


def test_threshold_profile_file_contains_one_row_per_rung(tmp_path: Path):
    artifacts = write_rung_sufficiency_artifacts(tmp_path, seed=7, trajectories_per_case=6)
    rows = artifacts.threshold_profile_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 9
    assert rows[0].startswith("rung_id,")
    assert any(line.startswith("imm,") for line in rows[1:])
