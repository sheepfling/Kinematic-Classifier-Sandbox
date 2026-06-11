from __future__ import annotations

from pathlib import Path

from kinematic_classifier_sandbox.utils.io import write_csv

from .static_feature_class_prior_audit_contracts import (
    StaticFeatureClassPriorAuditArtifacts,
    StaticFeatureClassPriorAuditResult,
)
from .static_feature_class_prior_audit_reporting import (
    render_static_decision_card,
    render_static_feature_class_prior_audit_report,
)


def write_static_feature_class_prior_audit_artifacts(
    output_root: str | Path,
    *,
    result: StaticFeatureClassPriorAuditResult,
) -> StaticFeatureClassPriorAuditArtifacts:
    base_path = Path(output_root)
    run_dir = base_path / "static_feature_class_prior_audit_v1"
    run_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_dir / "static_audit_report.md"
    decision_card_path = run_dir / "static_decision_card.md"
    class_confusability_matrix_path = run_dir / "class_confusability_matrix.csv"
    feature_relevance_table_path = run_dir / "feature_relevance_table.csv"
    feature_redundancy_matrix_path = run_dir / "feature_redundancy_matrix.csv"
    feature_synergy_candidates_path = run_dir / "feature_synergy_candidates.csv"
    prior_pathology_report_path = run_dir / "prior_pathology_report.csv"
    coverage_static_report_path = run_dir / "coverage_static_report.csv"
    leakage_static_report_path = run_dir / "leakage_static_report.csv"

    report_path.write_text(render_static_feature_class_prior_audit_report(result), encoding="utf-8")
    decision_card_path.write_text(render_static_decision_card(result) + "\n", encoding="utf-8")
    write_csv(
        class_confusability_matrix_path,
        [dict(row) for row in result.class_pair_rows],
        [
            "class_a",
            "class_b",
            "pairwise_auc",
            "mahalanobis_distance",
            "jensen_shannon",
            "overlap_coefficient",
            "fisher_ratio",
            "status",
        ],
    )
    write_csv(
        feature_relevance_table_path,
        [dict(row) for row in result.feature_relevance_rows],
        [
            "feature",
            "mi_with_class",
            "max_pairwise_auc",
            "mean_effect_size",
            "worst_pair_overlap",
            "missing_rate",
            "recommended_status",
        ],
    )
    write_csv(
        feature_redundancy_matrix_path,
        [dict(row) for row in result.feature_redundancy_rows],
        ["feature_a", "feature_b", "spearman_corr", "mutual_information", "status"],
    )
    write_csv(
        feature_synergy_candidates_path,
        [dict(row) for row in result.feature_synergy_rows],
        [
            "feature_a",
            "feature_b",
            "joint_mutual_information",
            "best_single_feature_mi",
            "pair_gain",
            "conditional_gain_proxy",
            "status",
        ],
    )
    write_csv(
        prior_pathology_report_path,
        [dict(row) for row in result.prior_pathology_rows],
        [
            "class_a",
            "class_b",
            "prior_odds_log",
            "observed_log_lr_min",
            "observed_log_lr_max",
            "flip_threshold_log_lr",
            "flip_possible",
            "evidence_margin",
            "posterior_collapse_rate",
            "pathology_flag",
        ],
    )
    write_csv(
        coverage_static_report_path,
        [dict(row) for row in result.coverage_rows],
        [
            "class_name",
            "feature",
            "sample_count",
            "occupied_bins",
            "empty_bin_rate",
            "min_value",
            "max_value",
            "status",
        ],
    )
    write_csv(
        leakage_static_report_path,
        [dict(row) for row in result.leakage_rows],
        [
            "feature",
            "provenance_tags",
            "online_available",
            "label_rule_overlap_flag",
            "future_dependency_flag",
            "metadata_leakage_flag",
            "status",
        ],
    )

    return StaticFeatureClassPriorAuditArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        decision_card_path=decision_card_path,
        class_confusability_matrix_path=class_confusability_matrix_path,
        feature_relevance_table_path=feature_relevance_table_path,
        feature_redundancy_matrix_path=feature_redundancy_matrix_path,
        feature_synergy_candidates_path=feature_synergy_candidates_path,
        prior_pathology_report_path=prior_pathology_report_path,
        coverage_static_report_path=coverage_static_report_path,
        leakage_static_report_path=leakage_static_report_path,
    )
