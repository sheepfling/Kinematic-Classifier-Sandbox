from __future__ import annotations

from pathlib import Path

from kinematic_classifier_sandbox.utils.io import write_csv
from kinematic_classifier_sandbox.utils.plotting import _figure_to_png

from .static_feature_class_prior_audit_contracts import (
    StaticFeatureClassPriorAuditArtifacts,
    StaticFeatureClassPriorAuditResult,
)
from .static_feature_class_prior_audit_reporting import (
    render_class_confusability_figure,
    render_feature_redundancy_graph_figure,
    render_feature_relevance_figure,
    render_feature_synergy_map_figure,
    render_prior_flip_thresholds_figure,
    render_prior_pathology_surface_figure,
    render_static_audit_action_router_figure,
    render_static_coverage_feasibility_figure,
    render_static_decision_card,
    render_static_decision_card_figure,
    render_static_feature_class_prior_audit_report,
    render_static_leakage_provenance_figure,
)


def _write_figure(path: Path, figure: object) -> None:
    try:
        path.write_bytes(_figure_to_png(figure))
    finally:
        figure.clf()


def _class_confusability_matrix_rows(result: StaticFeatureClassPriorAuditResult) -> list[dict[str, object]]:
    value_by_pair: dict[tuple[str, str], float] = {}
    for row in result.class_pair_rows:
        class_a = str(row["class_a"])
        class_b = str(row["class_b"])
        value = float(row["overlap_coefficient"])
        value_by_pair[(class_a, class_b)] = value
        value_by_pair[(class_b, class_a)] = value
    rows: list[dict[str, object]] = []
    for class_a in result.class_names:
        matrix_row: dict[str, object] = {"class": class_a}
        for class_b in result.class_names:
            matrix_row[class_b] = 0.0 if class_a == class_b else value_by_pair.get((class_a, class_b), "")
        rows.append(matrix_row)
    return rows


def write_static_feature_class_prior_audit_artifacts(
    output_root: str | Path,
    *,
    result: StaticFeatureClassPriorAuditResult | None = None,
    seed: int = 7,
    trajectories_per_class: int = 5,
    feature_analysis_result: object | None = None,
) -> StaticFeatureClassPriorAuditArtifacts:
    base_path = Path(output_root)
    run_dir = base_path / "static_feature_class_prior_audit_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    if result is None:
        from .static_feature_class_prior_audit import (
            analyze_default_static_feature_class_prior_audit,
        )

        result = analyze_default_static_feature_class_prior_audit(
            seed=seed,
            trajectories_per_class=trajectories_per_class,
            feature_analysis_result=feature_analysis_result,
        )

    report_path = run_dir / "static_audit_report.md"
    decision_card_path = run_dir / "static_decision_card.md"
    decision_card_png_path = run_dir / "02b_static_audit_decision_card.png"
    decision_card_legacy_png_path = run_dir / "static_audit_decision_card.png"
    class_confusability_png_path = run_dir / "02c_class_pair_confusability_matrix.png"
    feature_relevance_png_path = run_dir / "02d_feature_relevance_rank.png"
    feature_redundancy_png_path = run_dir / "02e_feature_redundancy_graph.png"
    feature_synergy_png_path = run_dir / "02f_feature_synergy_map.png"
    prior_pathology_surface_png_path = run_dir / "02g_prior_pathology_surface.png"
    prior_flip_thresholds_png_path = run_dir / "02h_prior_flip_thresholds.png"
    coverage_feasibility_png_path = run_dir / "02i_static_coverage_feasibility.png"
    leakage_provenance_png_path = run_dir / "02j_static_leakage_provenance_audit.png"
    action_router_png_path = run_dir / "02k_static_audit_to_action_router.png"
    class_confusability_matrix_path = run_dir / "class_confusability_matrix.csv"
    class_pair_diagnostics_path = run_dir / "class_pair_diagnostics.csv"
    class_feature_signature_path = run_dir / "class_feature_signature.csv"
    class_observability_path = run_dir / "class_observability.csv"
    feature_relevance_table_path = run_dir / "feature_relevance_table.csv"
    feature_redundancy_matrix_path = run_dir / "feature_redundancy_matrix.csv"
    feature_alias_candidates_path = run_dir / "feature_alias_candidates.csv"
    feature_synergy_candidates_path = run_dir / "feature_synergy_candidates.csv"
    prior_regime_path = run_dir / "prior_regime.csv"
    prior_pathology_report_path = run_dir / "prior_pathology_report.csv"
    prior_selection_balance_path = run_dir / "prior_selection_balance.csv"
    prior_flip_thresholds_path = run_dir / "prior_flip_thresholds.csv"
    resolution_plan_path = run_dir / "static_resolution_plan.csv"
    coverage_static_report_path = run_dir / "coverage_static_report.csv"
    coverage_feasibility_path = run_dir / "static_coverage_feasibility.csv"
    leakage_static_report_path = run_dir / "leakage_static_report.csv"
    leakage_provenance_audit_path = run_dir / "static_leakage_provenance_audit.csv"

    report_path.write_text(render_static_feature_class_prior_audit_report(result), encoding="utf-8")
    decision_card_path.write_text(render_static_decision_card(result) + "\n", encoding="utf-8")
    _write_figure(decision_card_png_path, render_static_decision_card_figure(result))
    decision_card_legacy_png_path.write_bytes(decision_card_png_path.read_bytes())
    _write_figure(class_confusability_png_path, render_class_confusability_figure(result))
    _write_figure(feature_relevance_png_path, render_feature_relevance_figure(result))
    _write_figure(feature_redundancy_png_path, render_feature_redundancy_graph_figure(result))
    _write_figure(feature_synergy_png_path, render_feature_synergy_map_figure(result))
    _write_figure(prior_pathology_surface_png_path, render_prior_pathology_surface_figure(result))
    _write_figure(prior_flip_thresholds_png_path, render_prior_flip_thresholds_figure(result))
    _write_figure(coverage_feasibility_png_path, render_static_coverage_feasibility_figure(result))
    _write_figure(leakage_provenance_png_path, render_static_leakage_provenance_figure(result))
    _write_figure(action_router_png_path, render_static_audit_action_router_figure(result))
    write_csv(
        class_confusability_matrix_path,
        _class_confusability_matrix_rows(result),
        ["class", *result.class_names],
    )
    write_csv(
        class_pair_diagnostics_path,
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
            "exact_shared_vector_count",
            "exact_shared_vector_rate",
            "signature_distance",
            "near_feature_collision",
            "collision_status",
            "expected_signature_distance",
            "expected_signature_collision_status",
        ],
    )
    write_csv(
        class_feature_signature_path,
        [dict(row) for row in result.class_feature_signature_rows],
        [
            "class_name",
            "feature",
            "sample_count",
            "mean",
            "std",
            "min_value",
            "max_value",
            "occupied_bins",
            "expected_mean",
            "expected_std",
            "expected_signature_source",
            "status",
        ],
    )
    write_csv(
        class_observability_path,
        [dict(row) for row in result.class_observability_rows],
        [
            "class_name",
            "sample_count",
            "exact_collision_pairs",
            "near_collision_pairs",
            "exact_collision_count",
            "near_collision_count",
            "expected_exact_signature_pairs",
            "expected_near_signature_pairs",
            "expected_exact_signature_count",
            "expected_near_signature_count",
            "expected_signature_feature_count",
            "expected_signature_coverage",
            "expected_signature_source",
            "selection_status",
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
        feature_alias_candidates_path,
        [dict(row) for row in result.feature_alias_rows],
        [
            "feature_a",
            "feature_b",
            "alias_type",
            "spearman_corr",
            "normalized_rmse",
            "sample_similarity_score",
            "same_semantic_group",
            "same_units",
            "same_aggregation",
            "threshold_gap",
            "recommended_action",
        ],
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
        prior_regime_path,
        [
            {"class_name": class_name, "prior_probability": result.priors[class_name]}
            for class_name in result.class_names
        ],
        ["class_name", "prior_probability"],
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
        prior_selection_balance_path,
        [dict(row) for row in result.prior_selection_rows],
        [
            "class_name",
            "prior_probability",
            "sample_count",
            "proxy_selection_count",
            "proxy_selection_rate",
            "true_class_selection_rate",
            "mean_proxy_posterior",
            "selection_gap_to_prior",
            "selection_to_prior_ratio",
            "proxy_model",
            "status",
        ],
    )
    write_csv(
        prior_flip_thresholds_path,
        [
            {
                "class_a": row["class_a"],
                "class_b": row["class_b"],
                "required_log_lr": row["flip_threshold_log_lr"],
                "observed_log_lr_min": row["observed_log_lr_min"],
                "observed_log_lr_max": row["observed_log_lr_max"],
                "flip_possible": row["flip_possible"],
                "pathology_flag": row["pathology_flag"],
            }
            for row in result.prior_pathology_rows
        ],
        [
            "class_a",
            "class_b",
            "required_log_lr",
            "observed_log_lr_min",
            "observed_log_lr_max",
            "flip_possible",
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
        coverage_feasibility_path,
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
    write_csv(
        leakage_provenance_audit_path,
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
    write_csv(
        resolution_plan_path,
        [dict(row) for row in result.resolution_rows],
        [
            "issue_code",
            "severity",
            "affected_scope",
            "evidence",
            "recommended_action",
            "verification",
            "route",
        ],
    )

    return StaticFeatureClassPriorAuditArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        decision_card_path=decision_card_path,
        decision_card_png_path=decision_card_png_path,
        class_confusability_png_path=class_confusability_png_path,
        feature_relevance_png_path=feature_relevance_png_path,
        feature_redundancy_png_path=feature_redundancy_png_path,
        feature_synergy_png_path=feature_synergy_png_path,
        prior_pathology_surface_png_path=prior_pathology_surface_png_path,
        prior_flip_thresholds_png_path=prior_flip_thresholds_png_path,
        coverage_feasibility_png_path=coverage_feasibility_png_path,
        leakage_provenance_png_path=leakage_provenance_png_path,
        action_router_png_path=action_router_png_path,
        class_confusability_matrix_path=class_confusability_matrix_path,
        class_pair_diagnostics_path=class_pair_diagnostics_path,
        class_feature_signature_path=class_feature_signature_path,
        class_observability_path=class_observability_path,
        feature_relevance_table_path=feature_relevance_table_path,
        feature_redundancy_matrix_path=feature_redundancy_matrix_path,
        feature_alias_candidates_path=feature_alias_candidates_path,
        feature_synergy_candidates_path=feature_synergy_candidates_path,
        prior_regime_path=prior_regime_path,
        prior_pathology_report_path=prior_pathology_report_path,
        prior_selection_balance_path=prior_selection_balance_path,
        prior_flip_thresholds_path=prior_flip_thresholds_path,
        resolution_plan_path=resolution_plan_path,
        coverage_static_report_path=coverage_static_report_path,
        coverage_feasibility_path=coverage_feasibility_path,
        leakage_static_report_path=leakage_static_report_path,
        leakage_provenance_audit_path=leakage_provenance_audit_path,
    )
