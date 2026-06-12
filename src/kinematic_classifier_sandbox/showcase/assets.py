from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

from matplotlib.patches import Rectangle

from kinematic_classifier_sandbox.reports.markdown import MarkdownDocument
from kinematic_classifier_sandbox.utils.io import (
    _copy_file,
    _read_csv,
    _read_json,
    _write_text,
    write_csv,
)
from kinematic_classifier_sandbox.utils.plotting import plt, write_plot
from kinematic_classifier_sandbox.utils.text import markdown_table_preview

from ..inference.kalman_filter_bank import run_kalman_bank_benchmark
from ..story.repo_story import (
    render_proof_gallery as render_repo_story_proof_gallery,
)
from ..validation.study_candidate.generation import write_study_candidate_generation_artifacts
from .contracts import (
    ARTIFACTS_ROOT,
    ROOT,
    SHOWCASE_DOCS_DIR,
    RunCardBody,
    RunCardSpec,
    ShowcaseAdvancedFilterSummary,
    ShowcaseAlgorithmReportData,
    ShowcaseCorpusAdequacySummary,
    ShowcaseCorpusReportData,
    ShowcaseDerivedPlotArtifact,
    ShowcaseDimensionalLiftReportData,
    ShowcaseFeatureReportData,
    ShowcaseFilteringReportData,
    ShowcaseHeadlineSummary,
    ShowcaseManifestEntry,
    ShowcaseOpenRisksData,
    ShowcasePlotDefinition,
    ShowcaseSourceDocSpec,
    ShowcaseTableSpec,
    ShowcaseTopResult,
)

if TYPE_CHECKING:
    from ..methodology.context import MethodologyExecutionContext


def _manifest_entry(
    *,
    kind: str,
    relative_path: str,
    section: str | None = None,
    source_path: str | None = None,
    title: str | None = None,
    plot_id: str | None = None,
    caption: str | None = None,
    interpretation: str | None = None,
    limitations: str | None = None,
) -> dict[str, object]:
    return ShowcaseManifestEntry(
        kind=kind,
        relative_path=relative_path,
        section=section,
        source_path=source_path,
        title=title,
        plot_id=plot_id,
        caption=caption,
        interpretation=interpretation,
        limitations=limitations,
    ).to_dict()


def _plot_manifest_entry(
    definition: ShowcasePlotDefinition | ShowcaseDerivedPlotArtifact,
    *,
    kind: str = "plot",
) -> dict[str, object]:
    relative_path = getattr(definition, "relative_path", None)
    source_path = getattr(definition, "source_path", None)
    if relative_path is None:
        relative_path = f"plots/{definition.filename}"
    if source_path is None:
        source_path = definition.source
    return _manifest_entry(
        kind=kind,
        plot_id=definition.plot_id,
        section=definition.section,
        relative_path=relative_path,
        source_path=source_path,
        caption=definition.caption,
        interpretation=definition.interpretation,
        limitations=definition.limitations,
    )


def _load_advanced_filter_method_rows() -> list[dict[str, object]]:
    comparison_path = ARTIFACTS_ROOT / "advanced_filter_comparison_v1" / "method_comparison.csv"
    if comparison_path.exists():
        return _read_csv(comparison_path)

    evidence_path = ARTIFACTS_ROOT / "advanced_filter_decision_v1" / "advanced_filter_decision_summary.json"
    if not evidence_path.exists():
        return []
    evidence = _read_json(evidence_path)
    return [
        {
            "method_id": "imm_v1",
            "corpus_objective_id": "transition_switching_1d_v1",
            "scenario_family": "switching_1d",
            "failure_case": "static_class_assumption",
            "primary_metric": "post_switch_gain",
            "primary_metric_value": evidence.get("transition_post_switch_gain", ""),
            "runtime_seconds": "",
            "promotion_decision": "promote" if evidence.get("imm_justified") else "defer",
        },
        {
            "method_id": "particle_filter_bank_v1",
            "corpus_objective_id": "pf_abs_range_multimodal_oracle_v1",
            "scenario_family": "abs_range_multimodal_1d",
            "failure_case": "multimodal_posterior_collapse",
            "primary_metric": "mean_oracle_to_pf_kl",
            "primary_metric_value": "",
            "runtime_seconds": "",
            "promotion_decision": "promote" if evidence.get("particle_filter_justified", False) else "defer",
        },
        {
            "method_id": "rbpf_v1",
            "corpus_objective_id": "pf_vs_rbpf_frontier_v1",
            "scenario_family": "latent_maneuver_onset_1d",
            "failure_case": "latent_event_timing",
            "primary_metric": "crossover_status",
            "primary_metric_value": "",
            "runtime_seconds": "",
            "promotion_decision": "promote" if evidence.get("rbpf_justified") else "defer",
        },
    ]

def _table_preview(rows: list[dict[str, str]], columns: list[str], limit: int = 8) -> str:
    return markdown_table_preview(rows=rows, columns=columns, limit=limit)


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _safe_float(value: object) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_optional(value: float | None, digits: int = 3) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def _advanced_filter_summary() -> ShowcaseAdvancedFilterSummary:
    comparison_path = ARTIFACTS_ROOT / "advanced_filter_comparison_v1" / "method_comparison.csv"
    if comparison_path.exists():
        rows = _read_csv(comparison_path)
        by_method = {row["method_id"]: row for row in rows}
        return ShowcaseAdvancedFilterSummary(
            imm_justified=by_method.get("imm_v1", {}).get("promotion_decision") == "promote",
            particle_filter_justified=by_method.get("particle_filter_bank_v1", {}).get("promotion_decision") == "promote",
            rbpf_justified=by_method.get("rbpf_v1", {}).get("promotion_decision") == "promote",
            method_rows=tuple(rows),
            primary_artifact="artifacts/advanced_filter_comparison_v1/method_comparison.csv",
        )
    summary = _read_json(ARTIFACTS_ROOT / "advanced_filter_decision_v1" / "advanced_filter_decision_summary.json")
    return ShowcaseAdvancedFilterSummary(
        imm_justified=bool(summary.get("imm_justified", False)),
        particle_filter_justified=bool(summary.get("particle_filter_justified", False)),
        rbpf_justified=False,
        method_rows=tuple(),
        primary_artifact="artifacts/advanced_filter_decision_v1/advanced_filter_decision_summary.json",
    )


def _headline_summary() -> ShowcaseHeadlineSummary:
    metrics_by_classifier = _read_csv(ARTIFACTS_ROOT / "common_1d_classifier_study" / "metrics_by_classifier.csv")
    best_classifier = max(metrics_by_classifier, key=lambda row: _float(row, "overall_accuracy"))
    common_dataset_rows = _read_csv(ARTIFACTS_ROOT / "common_dataset_comparison_v1" / "method_summary.csv")
    supported_common_dataset_rows = [row for row in common_dataset_rows if row.get("applicability_status", "supported") == "supported"]
    best_common_dataset = max(supported_common_dataset_rows or common_dataset_rows, key=lambda row: _float(row, "overall_accuracy"))
    corpus_summary = _read_json(ARTIFACTS_ROOT / "corpus_adequacy_audit_v1" / "corpus_adequacy_summary.json")
    advanced_summary = _advanced_filter_summary()
    dimension_rows = _read_csv(ARTIFACTS_ROOT / "dimensional_lift_audit" / "module_dimension_status.csv")
    dimension_counts: dict[str, int] = {}
    for row in dimension_rows:
        status = row["dimensional_status"]
        dimension_counts[status] = dimension_counts.get(status, 0) + 1
    corpus_adequacy = ShowcaseCorpusAdequacySummary(
        overall_status=str(corpus_summary["summary"]["overall_status"]),
        feature_status=str(corpus_summary["summary"]["feature_status"]),
        class_pair_status=str(corpus_summary["summary"]["class_pair_status"]),
        covariate_status=str(corpus_summary["summary"]["covariate_status"]),
    )
    return ShowcaseHeadlineSummary(
        best_common_study_classifier=ShowcaseTopResult(
            identifier=best_classifier["classifier_id"],
            overall_accuracy=_float(best_classifier, "overall_accuracy"),
        ),
        best_common_dataset_method=ShowcaseTopResult(
            identifier=best_common_dataset["method_name"],
            overall_accuracy=_float(best_common_dataset, "overall_accuracy"),
        ),
        corpus_adequacy=corpus_adequacy,
        advanced_filters=advanced_summary,
        dimensional_status_counts=dimension_counts,
    )


def _corpus_report_data() -> ShowcaseCorpusReportData:
    corpus_payload = _read_json(ARTIFACTS_ROOT / "corpus_adequacy_audit_v1" / "corpus_adequacy_summary.json")
    return ShowcaseCorpusReportData(
        summary=ShowcaseCorpusAdequacySummary(
            overall_status=str(corpus_payload["summary"]["overall_status"]),
            feature_status=str(corpus_payload["summary"]["feature_status"]),
            class_pair_status=str(corpus_payload["summary"]["class_pair_status"]),
            covariate_status=str(corpus_payload["summary"]["covariate_status"]),
        ),
        class_pair_rows=tuple(_read_csv(ARTIFACTS_ROOT / "corpus_adequacy_audit_v1" / "class_pair_coverage.csv")),
        leakage_rows=tuple(_read_csv(ARTIFACTS_ROOT / "corpus_adequacy_audit_v1" / "covariate_leakage_audit.csv")),
    )


def _algorithm_report_data() -> ShowcaseAlgorithmReportData:
    return ShowcaseAlgorithmReportData(
        metrics_by_classifier=tuple(_read_csv(ARTIFACTS_ROOT / "common_1d_classifier_study" / "metrics_by_classifier.csv")),
        common_dataset_rows=tuple(_read_csv(ARTIFACTS_ROOT / "common_dataset_comparison_v1" / "method_summary.csv")),
    )


def _feature_report_data() -> ShowcaseFeatureReportData:
    return ShowcaseFeatureReportData(
        taxonomy_rows=tuple(_read_json(ARTIFACTS_ROOT / "feature_taxonomy" / "feature_taxonomy.json")),
        identifiability_rows=tuple(_read_csv(ARTIFACTS_ROOT / "common_1d_classifier_study" / "identifiability_matrix.csv")),
        oracle_rows=tuple(_read_csv(ARTIFACTS_ROOT / "common_1d_classifier_study" / "oracle_classifier_results.csv")),
    )


def _filtering_report_data() -> ShowcaseFilteringReportData:
    return ShowcaseFilteringReportData(
        advanced_summary=_advanced_filter_summary(),
        transition_rows=tuple(
            _read_csv(ARTIFACTS_ROOT / "transition_matrix_accumulator_v1" / "transition_matrix_scenario_summary.csv")
        ),
    )


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _compute_binary_prediction_metrics(
    rows: list[dict[str, str]],
    *,
    positive_label_field: str = "class_b",
    posterior_positive_field: str = "posterior_class_b",
    predicted_label_field: str = "predicted_class",
    true_label_field: str = "true_class",
    confidence_field: str = "confidence",
) -> dict[str, float]:
    eps = 1e-12
    n = len(rows)
    if n == 0:
        return {"overall_accuracy": 0.0, "negative_log_likelihood": 0.0, "brier_score": 0.0, "ece": 0.0, "posterior_margin": 0.0}
    correct = 0
    nll_sum = 0.0
    brier_sum = 0.0
    margin_sum = 0.0
    bin_counts = [0] * 10
    bin_confidence = [0.0] * 10
    bin_accuracy = [0.0] * 10
    for row in rows:
        predicted_label = row[predicted_label_field]
        true_label = row[true_label_field]
        confidence = float(row[confidence_field])
        positive_label = row[positive_label_field]
        prob_positive = float(row[posterior_positive_field])
        prob_negative = 1.0 - prob_positive
        prob_true = prob_positive if true_label == positive_label else prob_negative
        correct_flag = 1.0 if predicted_label == true_label else 0.0
        correct += int(correct_flag)
        nll_sum += -math.log(max(prob_true, eps))
        true_positive = 1.0 if true_label == positive_label else 0.0
        brier_sum += (prob_positive - true_positive) ** 2
        margin_sum += abs(prob_positive - prob_negative)
        bin_index = min(int(confidence * 10.0), 9)
        bin_counts[bin_index] += 1
        bin_confidence[bin_index] += confidence
        bin_accuracy[bin_index] += correct_flag
    ece = 0.0
    for count, conf_sum, acc_sum in zip(bin_counts, bin_confidence, bin_accuracy, strict=True):
        if count == 0:
            continue
        mean_conf = conf_sum / count
        mean_acc = acc_sum / count
        ece += (count / n) * abs(mean_conf - mean_acc)
    return {
        "overall_accuracy": correct / n,
        "negative_log_likelihood": nll_sum / n,
        "brier_score": brier_sum / n,
        "ece": ece,
        "posterior_margin": margin_sum / n,
    }


def _common_study_prediction_metrics() -> dict[str, dict[str, float]]:
    rows = _read_csv(ARTIFACTS_ROOT / "common_1d_classifier_study" / "unified_predictions.csv")
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["classifier_id"], []).append(row)
    return {classifier_id: _compute_binary_prediction_metrics(group_rows) for classifier_id, group_rows in grouped.items()}


def _common_dataset_prediction_metrics() -> dict[str, dict[str, float]]:
    rows = _read_csv(ARTIFACTS_ROOT / "common_dataset_comparison_v1" / "method_run_summary.csv")
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["method_name"], []).append(row)
    metrics: dict[str, dict[str, float]] = {}
    for method_name, group_rows in grouped.items():
        metrics[method_name] = _compute_binary_prediction_metrics(
            group_rows,
            positive_label_field="true_class",
            posterior_positive_field="posterior_constant_acceleration",
            predicted_label_field="final_predicted_class",
            true_label_field="true_class",
            confidence_field="final_confidence",
        )
    return metrics


def _confidence_only_metrics(rows: list[dict[str, str]]) -> dict[str, float]:
    n = len(rows)
    if n == 0:
        return {"ece": 0.0}
    bin_counts = [0] * 10
    bin_confidence = [0.0] * 10
    bin_accuracy = [0.0] * 10
    for row in rows:
        confidence = float(row["confidence"])
        correct_flag = 1.0 if row["predicted_mode"] == row["true_mode"] else 0.0
        bin_index = min(int(confidence * 10.0), 9)
        bin_counts[bin_index] += 1
        bin_confidence[bin_index] += confidence
        bin_accuracy[bin_index] += correct_flag
    ece = 0.0
    for count, conf_sum, acc_sum in zip(bin_counts, bin_confidence, bin_accuracy, strict=True):
        if count == 0:
            continue
        ece += (count / n) * abs((conf_sum / count) - (acc_sum / count))
    return {"ece": ece}


def _method_evaluation_row(
    artifact_path: Path,
    method_id: str,
) -> dict[str, str] | None:
    if not artifact_path.exists():
        return None
    rows = _read_csv(artifact_path)
    return next((row for row in rows if row["method_id"] == method_id), None)


def _ladder_complexity(method_id: str) -> tuple[int, str]:
    mapping = {
        "pointwise": (1, "very_low"),
        "windowed": (2, "low"),
        "sequential_bayes": (3, "low"),
        "kalman_bank": (4, "medium"),
        "transition_matrix": (5, "medium"),
        "imm": (6, "high"),
        "particle_filter": (7, "very_high"),
        "rbpf": (8, "very_high"),
    }
    return mapping[method_id]


def _full_ladder_metric_rows() -> list[dict[str, str]]:
    common_dataset_rows = _read_csv(ARTIFACTS_ROOT / "common_dataset_comparison_v1" / "method_summary.csv")
    common_by_name = {row["method_name"]: row for row in common_dataset_rows}
    common_method_summary_path = ARTIFACTS_ROOT / "common_1d_classifier_study" / "method_evaluation_summary.csv"
    common_method_summary_rows = _read_csv(common_method_summary_path) if common_method_summary_path.exists() else []
    common_method_summary_by_id = {row["method_id"]: row for row in common_method_summary_rows}
    common_study_metrics = _common_study_prediction_metrics()
    common_dataset_metrics = _common_dataset_prediction_metrics()
    transition_row = _method_evaluation_row(
        ARTIFACTS_ROOT / "transition_matrix_accumulator_v1" / "method_evaluation_summary.csv",
        "transition_matrix_accumulator",
    )
    imm_row = _method_evaluation_row(
        ARTIFACTS_ROOT / "imm_filter_v1" / "method_evaluation_summary.csv",
        "imm_v1",
    )
    pf_row = _method_evaluation_row(
        ARTIFACTS_ROOT / "particle_filter_v1" / "method_evaluation_summary.csv",
        "particle_filter_bank_v1",
    )
    rbpf_row = _method_evaluation_row(
        ARTIFACTS_ROOT / "rbpf_v1" / "method_evaluation_summary.csv",
        "rbpf_v1",
    )
    transition_rows = _read_csv(ARTIFACTS_ROOT / "transition_matrix_accumulator_v1" / "transition_matrix_scenario_summary.csv")
    transition_accuracy = _mean([float(row["transition_accuracy"]) for row in transition_rows])
    transition_post_switch = _mean([float(row["transition_post_switch_accuracy"]) for row in transition_rows])
    imm_legacy_rows = _read_csv(ARTIFACTS_ROOT / "imm_filter_v1" / "advanced_filter_method_comparison.csv")
    imm_legacy_row = next(row for row in imm_legacy_rows if row["method_id"] == "imm_v1")
    imm_history_rows = _read_csv(ARTIFACTS_ROOT / "imm_filter_v1" / "posterior_history.csv")
    imm_ece = _confidence_only_metrics(imm_history_rows)["ece"]
    pf_legacy_row = _read_csv(ARTIFACTS_ROOT / "particle_filter_v1" / "pf_method_comparison.csv")[0]
    rbpf_legacy_row = _read_csv(ARTIFACTS_ROOT / "rbpf_v1" / "rbpf_method_comparison.csv")[0]

    rows: list[dict[str, str]] = []

    def add_common(
        method_id: str,
        source_method: str,
        source_classifier: str,
        decision: str,
        status: str,
        note: str,
    ) -> None:
        source = common_by_name[source_method]
        metric_source = common_method_summary_by_id.get(source_classifier)
        if metric_source is None:
            legacy_metric_source = common_study_metrics.get(source_classifier, common_dataset_metrics.get(source_method, {}))
            metric_source = {key: legacy_metric_source.get(key, "") for key in ("overall_accuracy", "negative_log_likelihood", "brier_score", "ece", "posterior_margin")}
        score, complexity_label = _ladder_complexity(method_id)
        rows.append(
            {
                "method_id": method_id,
                "source_method": source_method,
                "evaluated": "true",
                "applicable": "true" if source.get("applicability_status", "supported") == "supported" else source.get("applicability_status", ""),
                "overall_accuracy": _format_optional(_safe_float(metric_source.get("overall_accuracy"))),
                "negative_log_likelihood": _format_optional(_safe_float(metric_source.get("negative_log_likelihood"))),
                "brier_score": _format_optional(_safe_float(metric_source.get("brier_score"))),
                "ece": _format_optional(_safe_float(metric_source.get("ece"))),
                "posterior_margin": _format_optional(_safe_float(metric_source.get("posterior_margin"))),
                "post_switch_accuracy": _format_optional(_safe_float(metric_source.get("post_switch_accuracy"))),
                "switch_detection_delay": _format_optional(_safe_float(metric_source.get("switch_detection_delay"))),
                "runtime_seconds": _format_optional(_safe_float(metric_source.get("runtime_seconds"))),
                "complexity_score": str(score),
                "complexity_label": complexity_label,
                "status": status,
                "decision": decision,
                "note": note,
                "source_artifact": "artifacts/common_1d_classifier_study/method_evaluation_summary.csv" if common_method_summary_rows else "artifacts/common_1d_classifier_study/unified_predictions.csv",
            }
        )

    add_common("pointwise", "pointwise", "pointwise", "baseline", "baseline", "Instantaneous-evidence sanity check.")
    add_common("windowed", "windowed_robust", "windowed_robust_extrema", "improves_local_shape", "improves", "Canonical windowed rung uses the robust feature variant.")
    add_common("sequential_bayes", "accumulator", "bayes_accumulator", "strong_baseline", "competitive", "Recursive evidence accumulation over the shared study surface.")
    add_common("kalman_bank", "kalman_bank", "kalman_bank", "useful_for_dynamic_residuals", "competitive", "Model-based residual evidence on the shared study surface.")

    transition_score, transition_label = _ladder_complexity("transition_matrix")
    rows.append(
        {
            "method_id": "transition_matrix",
            "source_method": "transition_matrix_accumulator",
            "evaluated": "true",
            "applicable": "true",
            "overall_accuracy": _format_optional(_safe_float(transition_row.get("overall_accuracy")) if transition_row else transition_accuracy),
            "negative_log_likelihood": _format_optional(_safe_float(transition_row.get("negative_log_likelihood")) if transition_row else None),
            "brier_score": _format_optional(_safe_float(transition_row.get("brier_score")) if transition_row else None),
            "ece": _format_optional(_safe_float(transition_row.get("ece")) if transition_row else None),
            "posterior_margin": _format_optional(_safe_float(transition_row.get("posterior_margin")) if transition_row else None),
            "post_switch_accuracy": _format_optional(_safe_float(transition_row.get("post_switch_accuracy")) if transition_row else transition_post_switch),
            "switch_detection_delay": _format_optional(_safe_float(transition_row.get("switch_detection_delay")) if transition_row else None),
            "runtime_seconds": _format_optional(_safe_float(transition_row.get("runtime_seconds")) if transition_row else None),
            "complexity_score": str(transition_score),
            "complexity_label": transition_label,
            "status": "simplest_sufficient",
            "decision": "simplest_sufficient_for_main_switching_witness",
            "note": "First explicit switching-logic rung and current main-study switching selection.",
            "source_artifact": "artifacts/transition_matrix_accumulator_v1/method_evaluation_summary.csv",
        }
    )

    imm_score, imm_label = _ladder_complexity("imm")
    rows.append(
        {
            "method_id": "imm",
            "source_method": "imm_v1",
            "evaluated": "true",
            "applicable": "true",
            "overall_accuracy": _format_optional(_safe_float(imm_row.get("overall_accuracy")) if imm_row else _safe_float(imm_legacy_row.get("mode_accuracy"))),
            "negative_log_likelihood": _format_optional(_safe_float(imm_row.get("negative_log_likelihood")) if imm_row else _safe_float(imm_legacy_row.get("nll"))),
            "brier_score": _format_optional(_safe_float(imm_row.get("brier_score")) if imm_row else None),
            "ece": _format_optional(_safe_float(imm_row.get("ece")) if imm_row else imm_ece),
            "posterior_margin": _format_optional(_safe_float(imm_row.get("posterior_margin")) if imm_row else None),
            "post_switch_accuracy": _format_optional(_safe_float(imm_row.get("post_switch_accuracy")) if imm_row else _safe_float(imm_legacy_row.get("post_switch_accuracy"))),
            "switch_detection_delay": _format_optional(_safe_float(imm_row.get("switch_detection_delay")) if imm_row else _safe_float(imm_legacy_row.get("switch_detection_delay_median"))),
            "runtime_seconds": _format_optional(_safe_float(imm_row.get("runtime_seconds")) if imm_row else _safe_float(imm_legacy_row.get("runtime_seconds"))),
            "complexity_score": str(imm_score),
            "complexity_label": imm_label,
            "status": "witness_supported",
            "decision": "promoted_for_state_mixing_switch_witness",
            "note": "Evaluated broadly but promoted only on the state-mixing switching witness.",
            "source_artifact": "artifacts/imm_filter_v1/method_evaluation_summary.csv",
        }
    )

    pf_score, pf_label = _ladder_complexity("particle_filter")
    rows.append(
        {
            "method_id": "particle_filter",
            "source_method": "particle_filter_bank_v1",
            "evaluated": "true",
            "applicable": "witness_only",
            "overall_accuracy": _format_optional(_safe_float(pf_row.get("overall_accuracy")) if pf_row else None),
            "negative_log_likelihood": _format_optional(_safe_float(pf_row.get("negative_log_likelihood")) if pf_row else _safe_float(pf_legacy_row.get("nonlinear_drag_nll"))),
            "brier_score": _format_optional(_safe_float(pf_row.get("brier_score")) if pf_row else None),
            "ece": _format_optional(_safe_float(pf_row.get("ece")) if pf_row else None),
            "posterior_margin": _format_optional(_safe_float(pf_row.get("posterior_margin")) if pf_row else None),
            "post_switch_accuracy": "",
            "switch_detection_delay": "",
            "runtime_seconds": _format_optional(_safe_float(pf_row.get("runtime_seconds")) if pf_row else _safe_float(pf_legacy_row.get("runtime_seconds"))),
            "complexity_score": str(pf_score),
            "complexity_label": pf_label,
            "status": "required_showcase",
            "decision": "candidate_diagnostic_for_nonlinear_nongaussian_witness",
            "note": "Metrics are visible, but promotion waits for the named nonlinear/heavy-tail showcase against robust baselines.",
            "source_artifact": "artifacts/particle_filter_v1/method_evaluation_summary.csv",
        }
    )

    rbpf_score, rbpf_label = _ladder_complexity("rbpf")
    rows.append(
        {
            "method_id": "rbpf",
            "source_method": "rbpf_v1",
            "evaluated": "true",
            "applicable": "witness_only",
            "overall_accuracy": _format_optional(_safe_float(rbpf_row.get("overall_accuracy")) if rbpf_row else None),
            "negative_log_likelihood": _format_optional(_safe_float(rbpf_row.get("negative_log_likelihood")) if rbpf_row else None),
            "brier_score": _format_optional(_safe_float(rbpf_row.get("brier_score")) if rbpf_row else None),
            "ece": _format_optional(_safe_float(rbpf_row.get("ece")) if rbpf_row else None),
            "posterior_margin": _format_optional(_safe_float(rbpf_row.get("posterior_margin")) if rbpf_row else None),
            "post_switch_accuracy": _format_optional(_safe_float(rbpf_row.get("post_switch_accuracy")) if rbpf_row else _safe_float(rbpf_legacy_row.get("post_onset_mode_accuracy"))),
            "switch_detection_delay": "",
            "runtime_seconds": _format_optional(_safe_float(rbpf_row.get("runtime_seconds")) if rbpf_row else _safe_float(rbpf_legacy_row.get("runtime_seconds"))),
            "complexity_score": str(rbpf_score),
            "complexity_label": rbpf_label,
            "status": "required_showcase",
            "decision": "candidate_diagnostic_for_latent_event_timing_witness",
            "note": "Metrics are visible, but promotion waits for a run-backed PF/IMM comparison at fixed budget.",
            "source_artifact": "artifacts/rbpf_v1/method_evaluation_summary.csv",
        }
    )
    return rows


def _method_status_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for metric_row in _full_ladder_metric_rows():
        method_id = metric_row["method_id"]
        status = metric_row["status"]
        decision = metric_row["decision"]
        rows.append(
            {
                "method_id": method_id,
                "evaluated": "true",
                "applicable": metric_row["applicable"],
                "competitive": "true" if status in {"baseline", "improves", "competitive", "simplest_sufficient", "witness_supported", "required_showcase"} else "false",
                "simplest_sufficient": "true" if status == "simplest_sufficient" else "false",
                "witness_supported": "true" if status == "witness_supported" else "false",
                "promoted": "true" if decision.startswith("promoted_for") or status == "simplest_sufficient" else "false",
                "deferred": "true" if status == "required_showcase" else "false",
                "not_applicable": "true" if metric_row["applicable"] == "not_applicable" else "false",
                "decision": decision,
                "promoted_for": decision.removeprefix("promoted_for_") if decision.startswith("promoted_for_") else ("main_switching_witness" if status == "simplest_sufficient" else ""),
                "source_artifact": metric_row["source_artifact"],
            }
        )
    return rows


def _method_win_by_regime_rows() -> list[dict[str, str]]:
    rows = [
        {"regime": "local_overlap", "pointwise": "best", "windowed": "competitive", "sequential_bayes": "competitive", "kalman_bank": "fails", "transition_matrix": "not_applicable", "imm": "not_applicable", "particle_filter": "not_applicable", "rbpf": "not_applicable"},
        {"regime": "window_outlier", "pointwise": "fails", "windowed": "best", "sequential_bayes": "competitive", "kalman_bank": "competitive", "transition_matrix": "not_applicable", "imm": "not_applicable", "particle_filter": "not_applicable", "rbpf": "not_applicable"},
        {"regime": "weak_repeated_evidence", "pointwise": "fails", "windowed": "competitive", "sequential_bayes": "best", "kalman_bank": "competitive", "transition_matrix": "not_applicable", "imm": "not_applicable", "particle_filter": "not_applicable", "rbpf": "not_applicable"},
        {"regime": "matched_endpoint_dynamics", "pointwise": "fails", "windowed": "fails", "sequential_bayes": "competitive", "kalman_bank": "best", "transition_matrix": "competitive", "imm": "competitive", "particle_filter": "not_applicable", "rbpf": "not_applicable"},
        {"regime": "label_switching", "pointwise": "fails", "windowed": "fails", "sequential_bayes": "fails", "kalman_bank": "competitive", "transition_matrix": "best", "imm": "competitive", "particle_filter": "not_applicable", "rbpf": "not_applicable"},
        {"regime": "state_mixing_switch", "pointwise": "not_applicable", "windowed": "not_applicable", "sequential_bayes": "fails", "kalman_bank": "competitive", "transition_matrix": "competitive", "imm": "best", "particle_filter": "competitive", "rbpf": "competitive"},
        {"regime": "nonlinear_nongaussian", "pointwise": "not_applicable", "windowed": "not_applicable", "sequential_bayes": "not_applicable", "kalman_bank": "fails", "transition_matrix": "not_applicable", "imm": "fails", "particle_filter": "best", "rbpf": "competitive"},
        {"regime": "latent_event_timing", "pointwise": "not_applicable", "windowed": "not_applicable", "sequential_bayes": "not_applicable", "kalman_bank": "fails", "transition_matrix": "competitive", "imm": "fails", "particle_filter": "competitive", "rbpf": "best"},
    ]
    return rows


def _render_classifier_ladder_detail_tables(tables_dir: Path) -> list[dict[str, object]]:
    common_posterior_rows = _read_csv(ARTIFACTS_ROOT / "common_1d_classifier_study" / "unified_posterior_history.csv")
    posterior_history_rows = [
        {
            "method_id": row["classifier_id"],
            "trajectory_id": row["trajectory_id"],
            "scenario_id": row["scenario_id"],
            "time": row["time"],
            "true_class": row["true_class"],
            "posterior_class_a": row["posterior_class_a"],
            "posterior_class_b": row["posterior_class_b"],
        }
        for row in common_posterior_rows
    ]
    for row in _read_csv(ARTIFACTS_ROOT / "imm_filter_v1" / "posterior_history.csv"):
        posterior_history_rows.append(
            {
                "method_id": "imm",
                "trajectory_id": row["trajectory_id"],
                "scenario_id": row["scenario_name"],
                "time": row["time"],
                "true_class": row["true_mode"],
                "posterior_class_a": "",
                "posterior_class_b": row["probability"],
            }
        )
    for row in _read_csv(ARTIFACTS_ROOT / "transition_matrix_accumulator_v1" / "transition_matrix_posterior_history.csv"):
        posterior_history_rows.append(
            {
                "method_id": "transition_matrix",
                "trajectory_id": row["trajectory_id"],
                "scenario_id": row["scenario_name"],
                "time": row["time"],
                "true_class": row["true_mode"],
                "posterior_class_a": row.get("posterior_stationary", ""),
                "posterior_class_b": row["confidence"],
            }
        )
    for method_id, artifact_dir in (
        ("particle_filter", "particle_filter_v1"),
        ("rbpf", "rbpf_v1"),
    ):
        posterior_artifact_path = ARTIFACTS_ROOT / artifact_dir / "posterior_history.csv"
        if not posterior_artifact_path.exists() and method_id == "rbpf":
            posterior_artifact_path = ARTIFACTS_ROOT / artifact_dir / "latent_mode_posterior.csv"
        for row in _read_csv(posterior_artifact_path):
            posterior_history_rows.append(
                {
                    "method_id": method_id,
                    "trajectory_id": row["trajectory_id"],
                    "scenario_id": artifact_dir,
                    "time": row["time"],
                    "true_class": "",
                    "posterior_class_a": row["label"],
                    "posterior_class_b": row["posterior"],
                }
            )
    posterior_history_path = tables_dir / "posterior_history_by_method.csv"
    write_csv(posterior_history_path, posterior_history_rows, list(posterior_history_rows[0].keys()))

    common_summary_path = ARTIFACTS_ROOT / "common_1d_classifier_study" / "method_evaluation_summary.csv"
    common_summary_rows = _read_csv(common_summary_path) if common_summary_path.exists() else []
    if common_summary_rows:
        calibration_rows = [
            {
                "method_id": row["method_id"],
                "ece": _format_optional(_safe_float(row["ece"])),
                "negative_log_likelihood": _format_optional(_safe_float(row["negative_log_likelihood"])),
                "brier_score": _format_optional(_safe_float(row["brier_score"])),
                "posterior_margin": _format_optional(_safe_float(row["posterior_margin"])),
                "source_artifact": "artifacts/common_1d_classifier_study/method_evaluation_summary.csv",
            }
            for row in common_summary_rows
        ]
    else:
        common_metrics = _common_study_prediction_metrics()
        calibration_rows = [
            {
                "method_id": method_id,
                "ece": _format_optional(metric["ece"]),
                "negative_log_likelihood": _format_optional(metric["negative_log_likelihood"]),
                "brier_score": _format_optional(metric["brier_score"]),
                "posterior_margin": _format_optional(metric["posterior_margin"]),
                "source_artifact": "artifacts/common_1d_classifier_study/unified_predictions.csv",
            }
            for method_id, metric in sorted(common_metrics.items())
        ]
    imm_summary_row = _method_evaluation_row(
        ARTIFACTS_ROOT / "imm_filter_v1" / "method_evaluation_summary.csv",
        "imm_v1",
    )
    calibration_rows.append(
        {
            "method_id": "imm",
            "ece": _format_optional(_safe_float(imm_summary_row.get("ece")) if imm_summary_row else _confidence_only_metrics(_read_csv(ARTIFACTS_ROOT / "imm_filter_v1" / "posterior_history.csv"))["ece"]),
            "negative_log_likelihood": _format_optional(_safe_float(imm_summary_row.get("negative_log_likelihood")) if imm_summary_row else _safe_float(_read_csv(ARTIFACTS_ROOT / "imm_filter_v1" / "switching_detection_metrics.csv")[0].get("mode_nll"))),
            "brier_score": _format_optional(_safe_float(imm_summary_row.get("brier_score")) if imm_summary_row else None),
            "posterior_margin": _format_optional(_safe_float(imm_summary_row.get("posterior_margin")) if imm_summary_row else None),
            "source_artifact": "artifacts/imm_filter_v1/method_evaluation_summary.csv" if imm_summary_row else "artifacts/imm_filter_v1/posterior_history.csv",
        }
    )
    for method_id, artifact_dir, summary_method_id in (
        ("transition_matrix", "transition_matrix_accumulator_v1", "transition_matrix_accumulator"),
        ("particle_filter", "particle_filter_v1", "particle_filter_bank_v1"),
        ("rbpf", "rbpf_v1", "rbpf_v1"),
    ):
        summary_row = _method_evaluation_row(
            ARTIFACTS_ROOT / artifact_dir / "method_evaluation_summary.csv",
            summary_method_id,
        )
        if summary_row is None:
            continue
        calibration_rows.append(
            {
                "method_id": method_id,
                "ece": _format_optional(_safe_float(summary_row.get("ece"))),
                "negative_log_likelihood": _format_optional(_safe_float(summary_row.get("negative_log_likelihood"))),
                "brier_score": _format_optional(_safe_float(summary_row.get("brier_score"))),
                "posterior_margin": _format_optional(_safe_float(summary_row.get("posterior_margin"))),
                "source_artifact": f"artifacts/{artifact_dir}/method_evaluation_summary.csv",
            }
        )
    calibration_path = tables_dir / "calibration_by_method.csv"
    write_csv(calibration_path, calibration_rows, list(calibration_rows[0].keys()))

    common_prediction_rows = _read_csv(ARTIFACTS_ROOT / "common_1d_classifier_study" / "unified_predictions.csv")
    confusion_counts: dict[tuple[str, str, str], int] = {}
    for row in common_prediction_rows:
        key = (row["classifier_id"], row["true_class"], row["predicted_class"])
        confusion_counts[key] = confusion_counts.get(key, 0) + 1
    confusion_rows = [
        {
            "method_id": method_id,
            "true_class": true_class,
            "predicted_class": predicted_class,
            "count": str(count),
        }
        for (method_id, true_class, predicted_class), count in sorted(confusion_counts.items())
    ]
    confusion_path = tables_dir / "confusion_by_method.csv"
    write_csv(confusion_path, confusion_rows, list(confusion_rows[0].keys()))

    runtime_rows = []
    for row in _full_ladder_metric_rows():
        runtime_rows.append(
            {
                "method_id": row["method_id"],
                "runtime_seconds": row["runtime_seconds"],
                "complexity_score": row["complexity_score"],
                "complexity_label": row["complexity_label"],
                "source_artifact": row["source_artifact"],
            }
        )
    runtime_path = tables_dir / "runtime_by_method.csv"
    write_csv(runtime_path, runtime_rows, list(runtime_rows[0].keys()))

    manifest_entries: list[dict[str, object]] = []
    for relative_path in (
        "tables/posterior_history_by_method.csv",
        "tables/calibration_by_method.csv",
        "tables/confusion_by_method.csv",
        "tables/runtime_by_method.csv",
    ):
        manifest_entries.append(
            _manifest_entry(
                kind="table",
                section="algorithm_comparison",
                relative_path=relative_path,
                source_path="derived:Epic2 classifier ladder detail tables",
            )
        )
    return manifest_entries


def _render_classifier_ladder_support_tables(tables_dir: Path) -> list[dict[str, object]]:
    full_ladder_rows = _full_ladder_metric_rows()
    status_rows = _method_status_rows()
    regime_rows = _method_win_by_regime_rows()

    full_ladder_path = tables_dir / "full_ladder_metrics.csv"
    write_csv(full_ladder_path, full_ladder_rows, list(full_ladder_rows[0].keys()))
    status_path = tables_dir / "method_status_table.csv"
    write_csv(status_path, status_rows, list(status_rows[0].keys()))
    regime_path = tables_dir / "method_win_by_regime.csv"
    write_csv(regime_path, regime_rows, list(regime_rows[0].keys()))

    decision_card = MarkdownDocument("Classifier Ladder Decision Card")
    decision_card.bullet_list(
        [
            "simplest_sufficient_rung: `transition_matrix`",
            "posterior_contract: `pass`",
            "full_ladder_evaluated: `true`",
            "promotion_rule: `evaluate all applicable methods, promote selectively`",
            "advanced_algorithm_showcase: `prove escalation path for 3D lift`",
            "advanced_filters: IMM/PF/RBPF remain targeted-applicability witnesses rather than default shared-corpus winners.",
        ]
    )
    decision_card.heading("Advanced Algorithm Showcase", level=2)
    decision_card.bullet_list(
        [
            "purpose: `prove escalation path for 3D lift`",
            "imm: witness=`mode_switching_state_mixing`, claim_status=`witness_supported`, next=`compare switch delay, innovation NLL, state RMSE`",
            "pf: witness=`nonlinear_nongaussian_posterior`, claim_status=`required_showcase`, next=`add nonlinear/heavy-tail witness with robust baseline`",
            "rbpf: witness=`latent_event_timing`, claim_status=`required_showcase`, next=`sample event/mode path and marginalize continuous PVA state`",
        ]
    )
    decision_card_path = tables_dir / "classifier_ladder_decision_card.md"
    _write_text(decision_card_path, decision_card.text())

    classifier_report = MarkdownDocument("Classifier Ladder Report")
    classifier_report.paragraph(
        "This report summarizes the full ladder evaluation on common study surfaces, the rung-sufficiency decision, the advanced algorithm showcase, and the 3D lift bridge."
    )
    classifier_report.heading("Key Outputs", level=2)
    classifier_report.bullet_list(
        [
            "`full_ladder_metrics.csv`",
            "`method_status_table.csv`",
            "`method_win_by_regime.csv`",
            "`classifier_ladder_decision_card.md`",
        ]
    )
    classifier_report_path = tables_dir / "classifier_ladder_report.md"
    _write_text(classifier_report_path, classifier_report.text())

    shine_report = MarkdownDocument("Advanced Filter Shine Report")
    shine_report.bullet_list(
        [
            "Simple witnesses prove the evidence contract; advanced witnesses prove the lift path.",
            "IMM shines on state-mixing switching witnesses when post-switch accuracy and state evidence improve beyond transition-only logic.",
            "PF is a required showcase for nonlinear or non-Gaussian witnesses where sampled posterior shape matters materially.",
            "RBPF is a required showcase for latent event timing witnesses where the sampled discrete path plus conditional state split must beat broader sampling.",
        ]
    )
    shine_report_path = tables_dir / "advanced_filter_shine_report.md"
    _write_text(shine_report_path, shine_report.text())

    witness_card_paths = _write_advanced_showcase_witness_cards(tables_dir)

    manifest_entries: list[dict[str, object]] = []
    for relative_path in (
        "tables/full_ladder_metrics.csv",
        "tables/method_status_table.csv",
        "tables/method_win_by_regime.csv",
        "tables/classifier_ladder_decision_card.md",
        "tables/classifier_ladder_report.md",
        "tables/advanced_filter_shine_report.md",
        *witness_card_paths,
    ):
        manifest_entries.append(
            _manifest_entry(
                kind="table",
                section="algorithm_comparison",
                relative_path=relative_path,
                source_path="derived:Epic2 showcase support tables",
            )
        )
    manifest_entries.extend(_render_classifier_ladder_detail_tables(tables_dir))
    return manifest_entries


def _write_advanced_showcase_witness_cards(tables_dir: Path) -> tuple[str, ...]:
    cards = (
        (
            "imm_mode_switching_state_mixing.md",
            "IMM Mode-Switching State-Mixing Witness",
            (
                "Witness: `mode_switching_state_mixing`",
                "Failure mode: label-level transition logic cannot carry mixed dynamic state through a switch.",
                "Expected help: IMM mode probabilities and mixed state estimates improve switching-state evidence.",
                "Claim status: `witness_supported` when run-backed switch delay, innovation NLL, and state RMSE support it.",
                "Limitation: targeted switching-state applicability, not global superiority.",
            ),
        ),
        (
            "pf_nonlinear_nongaussian_posterior.md",
            "PF Nonlinear Non-Gaussian Posterior Witness",
            (
                "Witness: `nonlinear_nongaussian_posterior`",
                "Failure mode: Gaussian residual summaries misrepresent nonlinear, heavy-tailed, or multimodal uncertainty.",
                "Expected help: sampled posterior representation captures shape that simpler filters collapse.",
                "Claim status: `required_showcase` or candidate diagnostic until the named witness is run-backed against robust baselines.",
                "Limitation: particle cost and degeneracy must be measured before broader promotion.",
            ),
        ),
        (
            "rbpf_latent_event_timing.md",
            "RBPF Latent Event Timing Witness",
            (
                "Witness: `latent_event_timing`",
                "Failure mode: unknown maneuver onset or mode path creates discrete ambiguity with continuous PVA state.",
                "Expected help: sample event or mode path while marginalizing conditional continuous state.",
                "Claim status: `required_showcase` or candidate diagnostic until run-backed PF/IMM comparisons pass.",
                "Limitation: the sampled/marginalized split must prove efficiency at fixed budget.",
            ),
        ),
    )
    relative_paths: list[str] = []
    for filename, title, bullets in cards:
        document = MarkdownDocument(title)
        document.bullet_list(list(bullets))
        path = tables_dir / filename
        _write_text(path, document.text())
        relative_paths.append(f"tables/{filename}")
    return tuple(relative_paths)


def _dimensional_lift_report_data() -> ShowcaseDimensionalLiftReportData:
    return ShowcaseDimensionalLiftReportData(
        dimension_rows=tuple(_read_csv(ARTIFACTS_ROOT / "dimensional_lift_audit" / "module_dimension_status.csv"))
    )


def _open_risks_data() -> ShowcaseOpenRisksData:
    corpus_payload = _read_json(ARTIFACTS_ROOT / "corpus_adequacy_audit_v1" / "corpus_adequacy_summary.json")
    return ShowcaseOpenRisksData(
        corpus_summary=ShowcaseCorpusAdequacySummary(
            overall_status=str(corpus_payload["summary"]["overall_status"]),
            feature_status=str(corpus_payload["summary"]["feature_status"]),
            class_pair_status=str(corpus_payload["summary"]["class_pair_status"]),
            covariate_status=str(corpus_payload["summary"]["covariate_status"]),
        ),
        recommendations=tuple(str(item) for item in corpus_payload.get("recommendations", [])),
    )


def _showcase_plot_definitions() -> list[ShowcasePlotDefinition]:
    return [
        ShowcasePlotDefinition(
            plot_id="prior_to_posterior_single_step",
            source="artifacts/bayesian_walkthroughs/plots/prior_to_posterior_single_step.png",
            filename="prior_to_posterior_single_step.png",
            section="bayesian_updates",
            caption="Single-step prior-to-posterior walkthrough for a representative Bayesian update.",
            interpretation="Use this to verify that the posterior move matches the stated prior and likelihood evidence rather than opaque numerical code paths.",
            limitations="This is one witness trajectory and one update step, not a proof that every model family is calibrated equally well.",
        ),
        ShowcasePlotDefinition(
            plot_id="likelihood_curves_with_feature_value",
            source="artifacts/bayesian_walkthroughs/plots/likelihood_curves_with_feature_value.png",
            filename="likelihood_curves_with_feature_value.png",
            section="bayesian_updates",
            caption="Likelihood curves with the observed feature value marked.",
            interpretation="Use this to see why a given feature value favored one class over another before any posterior accumulation happened.",
            limitations="These curves use the current simplified evidence models and are not a universal likelihood audit across all families.",
        ),
        ShowcasePlotDefinition(
            plot_id="posterior_timeline",
            source="artifacts/bayesian_walkthroughs/plots/posterior_timeline.png",
            filename="posterior_timeline.png",
            section="bayesian_updates",
            caption="Posterior timeline over one representative trajectory.",
            interpretation="Use this to see whether evidence accumulated smoothly or only flipped at the end of the track.",
            limitations="This is a representative trace rather than a full Monte Carlo posterior distribution.",
        ),
        ShowcasePlotDefinition(
            plot_id="log_odds_timeline",
            source="artifacts/bayesian_walkthroughs/plots/log_odds_timeline.png",
            filename="log_odds_timeline.png",
            section="bayesian_updates",
            caption="Log-odds timeline for the same representative Bayesian walkthrough.",
            interpretation="Use this to separate likelihood-driven movement from prior offsets in an additive space.",
            limitations="The current walkthough is binary and simplified; multi-class log-odds surfaces are richer than this single example.",
        ),
        ShowcasePlotDefinition(
            plot_id="prior_sensitivity_curve",
            source="artifacts/bayesian_walkthroughs/plots/prior_sensitivity_curve.png",
            filename="prior_sensitivity_curve.png",
            section="bayesian_updates",
            caption="Posterior outcome versus prior sweep for the walkthrough trajectory.",
            interpretation="Use this to see where prior changes can and cannot move the decision boundary.",
            limitations="This is a local fragility study, not a full prior robustness proof across every study candidate.",
        ),
        ShowcasePlotDefinition(
            plot_id="bayes_factor_timeline",
            source="artifacts/bayesian_walkthroughs/plots/bayes_factor_timeline.png",
            filename="bayes_factor_timeline.png",
            section="bayesian_updates",
            caption="Bayes-factor timeline across the representative trajectory.",
            interpretation="Use this to see when evidence was genuinely informative versus merely cumulative.",
            limitations="Bayes factors here are driven by the current proxy evidence surfaces for some families.",
        ),
        ShowcasePlotDefinition(
            plot_id="classifier_pair_accuracy_heatmap",
            source="artifacts/common_1d_classifier_study/plots/confusion_matrices/classifier_pair_accuracy_heatmap.png",
            filename="classifier_pair_accuracy_heatmap.png",
            section="algorithm_comparison",
            caption="Classifier-by-class-pair heatmap from the common study.",
            interpretation="Use this to see which methods fail on which explicit class boundaries rather than only overall accuracy.",
            limitations="This reflects the current 1D executable subset rather than a full multi-dimensional deployment study.",
        ),
        ShowcasePlotDefinition(
            plot_id="prefix_accuracy_curve",
            source="artifacts/common_1d_classifier_study/plots/monte_carlo/prefix_accuracy_curve.png",
            filename="prefix_accuracy_curve.png",
            section="duration_sensitivity",
            caption="Prefix accuracy versus time within the common study.",
            interpretation="Use this to separate early-horizon ambiguity from late-horizon classifier failure.",
            limitations="The curve is specific to the current synthetic corpus and declared class pairs.",
        ),
        ShowcasePlotDefinition(
            plot_id="prior_sensitivity",
            source="artifacts/common_1d_classifier_study/plots/priors/prior_sensitivity.png",
            filename="prior_sensitivity.png",
            section="prior_fragility",
            caption="Prior-sensitivity view from the common study bundle.",
            interpretation="Use this to identify cases where decisions move under prior changes rather than stronger evidence.",
            limitations="This summarizes the current binary and pairwise prior studies, not arbitrary class-cardinality priors.",
        ),
        ShowcasePlotDefinition(
            plot_id="identifiability_summary",
            source="artifacts/common_1d_classifier_study/plots/feature_space/identifiability_summary.png",
            filename="identifiability_summary.png",
            section="identifiability",
            caption="Common-study identifiability summary by class pair and feature bundle.",
            interpretation="Use this to decide whether a failure is more likely a feature/data limit than a classifier implementation issue.",
            limitations="This summarizes engineered features only; it is not a learned representation audit.",
        ),
        ShowcasePlotDefinition(
            plot_id="static_audit_decision_card",
            source="artifacts/static_feature_class_prior_audit_v1/02b_static_audit_decision_card.png",
            filename="static_audit_decision_card.png",
            section="static_feature_class_prior_audit",
            caption="Static feature/class/prior admissibility decision card.",
            interpretation="Use this to check whether class separability, feature relevance, prior pathology, coverage feasibility, and leakage risk were screened before corpus search.",
            limitations="This is an admissibility gate over the current samples and priors, not a final downstream classifier guarantee.",
        ),
        ShowcasePlotDefinition(
            plot_id="feature_space_confusion_map",
            source="artifacts/feature_analysis_v1/feature_space_confusion_map.png",
            filename="feature_space_confusion_map.png",
            section="feature_confusion",
            caption="Feature-space scatter emphasizing confusing class pairs.",
            interpretation="Use this to see whether classes overlap structurally in feature space.",
            limitations="The projection is intentionally simplified and does not prove global separability by itself.",
        ),
        ShowcasePlotDefinition(
            plot_id="class_confusability_heatmap",
            source="artifacts/feature_analysis_v1/class_confusability_heatmap.png",
            filename="class_confusability_heatmap.png",
            section="feature_confusion",
            caption="Class confusability heatmap from pairwise feature analysis.",
            interpretation="Use this to rank which class pairs most deserve targeted feature or corpus work.",
            limitations="This is feature-space confusability, not end-to-end classifier confusion.",
        ),
        ShowcasePlotDefinition(
            plot_id="pairwise_overlap_heatmap",
            source="artifacts/feature_analysis_v1/pairwise_overlap_heatmap.png",
            filename="pairwise_overlap_heatmap.png",
            section="feature_confusion",
            caption="Pairwise overlap heatmap for the current feature-space class distributions.",
            interpretation="Use this to identify class pairs that remain intrinsically close even before classifier choice.",
            limitations="Overlap estimates depend on the current engineered-feature representation.",
        ),
        ShowcasePlotDefinition(
            plot_id="feature_ranking_summary",
            source="artifacts/feature_analysis_v1/feature_ranking_summary.png",
            filename="feature_ranking_summary.png",
            section="feature_confusion",
            caption="Feature ranking summary from the separability analysis.",
            interpretation="Use this to see which engineered features carry the strongest class evidence on the current corpus.",
            limitations="Ranking is corpus-dependent and does not by itself prove causal feature importance.",
        ),
        ShowcasePlotDefinition(
            plot_id="pc1_pc2_by_class",
            source="artifacts/pca_analysis_v1/pc1_pc2_by_class.png",
            filename="pc1_pc2_by_class.png",
            section="pca",
            caption="PC1/PC2 class scatter for the full engineered feature set.",
            interpretation="Use this as a dimensionality diagnostic to see whether major separation is already available in a low-dimensional projection.",
            limitations="PCA is diagnostic here, not a classifier.",
        ),
        ShowcasePlotDefinition(
            plot_id="covariate_leakage_audit",
            source="artifacts/corpus_adequacy_audit_v1/covariate_leakage_audit.png",
            filename="covariate_leakage_audit.png",
            section="corpus_validity",
            caption="Covariate leakage audit across the generated corpus.",
            interpretation="Use this to see whether duration, sampling, or noise covariates are class-linked strongly enough to corrupt comparisons.",
            limitations="A green result reduces suspicion of leakage; it does not prove the corpus is realistic.",
        ),
        ShowcasePlotDefinition(
            plot_id="class_pair_coverage_heatmap",
            source="artifacts/corpus_adequacy_audit_v1/class_pair_coverage_heatmap.png",
            filename="class_pair_coverage_heatmap.png",
            section="corpus_validity",
            caption="Declared class-pair coverage heatmap for the corpus.",
            interpretation="Use this to identify pairs that are still too easy or under-covered in required tiers.",
            limitations="Coverage is only as good as the current manifest definitions and tier design.",
        ),
        ShowcasePlotDefinition(
            plot_id="kalman_bank_diagnostics",
            source="artifacts/kalman_filter_bank/kalman_bank_diagnostics.png",
            filename="kalman_bank_diagnostics.png",
            section="filtering",
            caption="Kalman bank diagnostic panel.",
            interpretation="Use this to inspect innovation-driven evidence quality and posterior movement for the model-based baseline.",
            limitations="This reflects the current position-only Kalman family, not a fully generalized 3D filter stack.",
        ),
        ShowcasePlotDefinition(
            plot_id="transition_matrix_diagnostics",
            source="artifacts/transition_matrix_accumulator_v1/transition_matrix_diagnostics.png",
            filename="transition_matrix_diagnostics.png",
            section="switching",
            caption="Transition benchmark diagnostics over switching scenarios.",
            interpretation="Use this to judge whether explicit transition structure buys something before considering IMM.",
            limitations="This is a methodology exercise over current synthetic switching cases, not an operational mode tracker.",
        ),
    ]


def _copy_showcase_tables(
    tables_dir: Path,
    *,
    methodology_context: "MethodologyExecutionContext | None" = None,
    artifact_mode: str = "full",
) -> list[dict[str, object]]:
    table_sources = [
        ShowcaseTableSpec("bayesian_step_tables.csv", ARTIFACTS_ROOT / "bayesian_walkthroughs" / "bayesian_step_tables.csv", "bayesian_updates"),
        ShowcaseTableSpec("prior_sweep_examples.csv", ARTIFACTS_ROOT / "bayesian_walkthroughs" / "prior_sweep_examples.csv", "bayesian_updates"),
        ShowcaseTableSpec("feature_contribution_examples.csv", ARTIFACTS_ROOT / "bayesian_walkthroughs" / "feature_contribution_examples.csv", "bayesian_updates"),
        ShowcaseTableSpec("posterior_flip_thresholds.csv", ARTIFACTS_ROOT / "bayesian_walkthroughs" / "posterior_flip_thresholds.csv", "bayesian_updates"),
        ShowcaseTableSpec("metrics_by_classifier.csv", ARTIFACTS_ROOT / "common_1d_classifier_study" / "metrics_by_classifier.csv", "algorithm_comparison"),
        ShowcaseTableSpec("metrics_by_class_pair.csv", ARTIFACTS_ROOT / "common_1d_classifier_study" / "metrics_by_class_pair.csv", "class_pair_study"),
        ShowcaseTableSpec("feature_set_comparison.csv", ARTIFACTS_ROOT / "common_1d_classifier_study" / "feature_set_comparison.csv", "feature_study"),
        ShowcaseTableSpec("prior_sensitivity_by_class_pair.csv", ARTIFACTS_ROOT / "common_1d_classifier_study" / "prior_sensitivity_by_class_pair.csv", "prior_fragility"),
        ShowcaseTableSpec("identifiability_matrix.csv", ARTIFACTS_ROOT / "common_1d_classifier_study" / "identifiability_matrix.csv", "identifiability"),
        ShowcaseTableSpec("oracle_classifier_results.csv", ARTIFACTS_ROOT / "common_1d_classifier_study" / "oracle_classifier_results.csv", "feature_study"),
        ShowcaseTableSpec("static_decision_card.md", ARTIFACTS_ROOT / "static_feature_class_prior_audit_v1" / "static_decision_card.md", "static_feature_class_prior_audit"),
        ShowcaseTableSpec("covariate_leakage_audit.csv", ARTIFACTS_ROOT / "common_1d_classifier_study" / "covariate_leakage_audit.csv", "corpus_validity"),
        ShowcaseTableSpec("feature_excitation_matrix.csv", ARTIFACTS_ROOT / "common_1d_classifier_study" / "feature_excitation_matrix.csv", "corpus_validity"),
        ShowcaseTableSpec("validation_ladder_decisions.csv", ARTIFACTS_ROOT / "validation_ladder" / "validation_ladder_decisions.csv", "study_candidate_validation"),
        ShowcaseTableSpec("validation_ladder_scores.csv", ARTIFACTS_ROOT / "validation_ladder" / "validation_ladder_scores.csv", "study_candidate_validation"),
        ShowcaseTableSpec("feature_evidence_table.csv", ARTIFACTS_ROOT / "study_candidate_generation" / "feature_evidence_table.csv", "study_candidate_validation"),
        ShowcaseTableSpec("prior_sensitivity_explanation_table.csv", ARTIFACTS_ROOT / "study_candidate_generation" / "prior_sensitivity_explanation_table.csv", "study_candidate_validation"),
        ShowcaseTableSpec("method_summary.csv", ARTIFACTS_ROOT / "common_dataset_comparison_v1" / "method_summary.csv", "algorithm_comparison"),
        ShowcaseTableSpec("technique_summary.csv", ARTIFACTS_ROOT / "technique_comparison_v1" / "technique_summary.csv", "algorithm_comparison"),
        ShowcaseTableSpec("class_balance.csv", ARTIFACTS_ROOT / "corpus_adequacy_audit_v1" / "class_balance.csv", "corpus_validity"),
        ShowcaseTableSpec("class_pair_coverage.csv", ARTIFACTS_ROOT / "corpus_adequacy_audit_v1" / "class_pair_coverage.csv", "corpus_validity"),
        ShowcaseTableSpec("feature_set_coverage.csv", ARTIFACTS_ROOT / "corpus_adequacy_audit_v1" / "feature_set_coverage.csv", "corpus_validity"),
        ShowcaseTableSpec("classifier_support.csv", ARTIFACTS_ROOT / "coverage_report_v1" / "classifier_support.csv", "corpus_validity"),
        ShowcaseTableSpec("feature_taxonomy.json", ARTIFACTS_ROOT / "feature_taxonomy" / "feature_taxonomy.json", "feature_taxonomy"),
        ShowcaseTableSpec("feature_separation_scores.csv", ARTIFACTS_ROOT / "feature_analysis_v1" / "feature_separation_scores.csv", "feature_study"),
        ShowcaseTableSpec("pairwise_overlap_matrix.csv", ARTIFACTS_ROOT / "feature_analysis_v1" / "pairwise_overlap_matrix.csv", "feature_study"),
        ShowcaseTableSpec("pairwise_auc_matrix.csv", ARTIFACTS_ROOT / "feature_analysis_v1" / "pairwise_auc_matrix.csv", "feature_study"),
        ShowcaseTableSpec("module_dimension_status.csv", ARTIFACTS_ROOT / "dimensional_lift_audit" / "module_dimension_status.csv", "dimensional_lift"),
        ShowcaseTableSpec("scalar_assumption_inventory.csv", ARTIFACTS_ROOT / "dimensional_lift_audit" / "scalar_assumption_inventory.csv", "dimensional_lift"),
        ShowcaseTableSpec("advanced_filter_decision_summary.json", ARTIFACTS_ROOT / "advanced_filter_decision_v1" / "advanced_filter_decision_summary.json", "advanced_filters"),
        ShowcaseTableSpec("advanced_filter_decision_evidence.json", ARTIFACTS_ROOT / "advanced_filter_decision_v1" / "advanced_filter_decision_evidence.json", "advanced_filters"),
        ShowcaseTableSpec("filter_trace_method_matrix.csv", ARTIFACTS_ROOT / "filter_trace_validation_v1" / "method_trace_matrix.csv", "advanced_filters"),
        ShowcaseTableSpec("filter_trace_requirement_matrix.csv", ARTIFACTS_ROOT / "filter_trace_validation_v1" / "trace_requirement_matrix.csv", "advanced_filters"),
        ShowcaseTableSpec("filter_step_trace_schema.json", ARTIFACTS_ROOT / "filter_trace_validation_v1" / "filter_step_trace_schema.json", "advanced_filters"),
        ShowcaseTableSpec("transition_matrix_scenario_summary.csv", ARTIFACTS_ROOT / "transition_matrix_accumulator_v1" / "transition_matrix_scenario_summary.csv", "advanced_filters"),
        ShowcaseTableSpec("algorithm_ladder_proof.csv", ARTIFACTS_ROOT / "latex" / "algorithm_ladder_proof.csv", "algorithm_comparison"),
        ShowcaseTableSpec("toy_problem_summary.csv", ARTIFACTS_ROOT / "latex" / "toy_problem_summary.csv", "study_candidate_validation"),
    ]
    if any(not spec.source.exists() and "study_candidate_generation" in str(spec.source) for spec in table_sources):
        if methodology_context is not None:
            write_study_candidate_generation_artifacts(
                ARTIFACTS_ROOT,
                result=methodology_context.study_generation_result,
            )
        elif artifact_mode == "full":
            write_study_candidate_generation_artifacts(ARTIFACTS_ROOT)
        else:
            missing = [
                str(spec.source.relative_to(ROOT))
                for spec in table_sources
                if not spec.source.exists() and "study_candidate_generation" in str(spec.source)
            ]
            raise FileNotFoundError(
                "fast showcase mode requires prebuilt study-candidate artifacts: "
                + ", ".join(missing)
            )
    manifest_entries: list[dict[str, object]] = []
    for spec in table_sources:
        destination = tables_dir / spec.filename
        _copy_file(spec.source, destination)
        manifest_entries.append(
            _manifest_entry(
                kind="table",
                section=spec.section,
                relative_path=str(destination.relative_to(tables_dir.parents[0])),
                source_path=str(spec.source.relative_to(ROOT)),
            )
        )
    manifest_entries.extend(_render_classifier_ladder_support_tables(tables_dir))
    return manifest_entries


def _proof_gallery_claims() -> list[dict[str, object]]:
    return [
        {
            "heading": "Claim 1: Bayesian update machinery works",
            "claim": "The repo can show a prior, likelihood, Bayes factor, and posterior move explicitly rather than hiding classifier behavior behind final labels only.",
            "evidence": [
                ("Single-step prior to posterior", "plots/prior_to_posterior_single_step.png", "One witness update that can be checked by hand."),
                ("Likelihood curves", "plots/likelihood_curves_with_feature_value.png", "Shows which class the current observation favored before accumulation."),
                ("Posterior timeline", "plots/posterior_timeline.png", "Shows sequential accumulation on one representative trajectory."),
                ("Log-odds timeline", "plots/log_odds_timeline.png", "Shows additive evidence movement and prior offsets."),
                ("Bayes factor timeline", "plots/bayes_factor_timeline.png", "Shows when evidence was informative versus merely cumulative."),
                ("Prior sweep table", "tables/prior_sweep_examples.csv", "Shows decision fragility under controlled prior changes."),
                ("Prior sensitivity explanation table", "tables/prior_sensitivity_explanation_table.csv", "Summarizes which generated studies remain evidence-dominated versus prior-sensitive."),
            ],
        },
        {
            "heading": "Claim 2: History helps",
            "claim": "Sequential accumulation and history-aware methods outperform purely instantaneous evidence on ambiguity that unfolds over time.",
            "evidence": [
                ("Pointwise vs accumulator posterior timelines", "plots/pointwise_vs_accumulator_posterior_timelines.png", "Shows how recursive accumulation differs from a pointwise baseline on the same hard trajectory."),
                ("Prefix accuracy curve", "plots/prefix_accuracy_curve.png", "Shows improvement as more history is available."),
                ("True-class posterior quantiles", "plots/true_class_posterior_quantiles.png", "Shows how true-class posterior mass concentrates across repeated accumulator runs."),
            ],
        },
        {
            "heading": "Claim 3: Features matter",
            "claim": "Feature-set choice changes separability, oracle performance, and posterior behavior in ways that are visible and auditable.",
            "evidence": [
                ("Feature ablation chart", "plots/feature_ablation_posterior.png", "Shows how the witness posterior changes when one feature is removed."),
                ("Feature separation ranking", "plots/feature_ranking_summary.png", "Visual summary of strongest separating engineered features."),
                ("Feature distribution by class", "plots/feature_distribution_by_class.png", "Shows whether separation comes from location, spread, or both."),
                ("Feature correlation heatmap", "plots/feature_correlation_heatmap.png", "Shows where evidence sources are correlated strongly enough to justify caution."),
                ("Feature evidence table", "tables/feature_evidence_table.csv", "Links feature metadata, double-counting risk, transfer status, and best/worst class-pair evidence."),
                ("Feature contribution examples", "tables/feature_contribution_examples.csv", "Walkthrough-level evidence contributions with caveats on correlation."),
            ],
        },
        {
            "heading": "Claim 4: Class pairs have different difficulty",
            "claim": "Not all class boundaries are equally hard, and the packet can show pairwise difficulty explicitly rather than collapsing them into one leaderboard.",
            "evidence": [
                ("Pairwise confusion heatmap", "plots/classifier_pair_accuracy_heatmap.png", "Visual ranking of pair difficulty across methods."),
                ("Pairwise overlap heatmap", "plots/pairwise_overlap_heatmap.png", "Shows pairwise overlap structure directly."),
                ("Oracle classifier results", "tables/oracle_classifier_results.csv", "Feature-only upper-bound separability by pair."),
                ("Duration sensitivity by class pair", "plots/duration_sensitivity_by_class_pair.png", "Shows which class boundaries need more horizon before they separate cleanly."),
            ],
        },
        {
            "heading": "Claim 5: Corpus quality matters",
            "claim": "Classifier comparisons are only credible when coverage, excitation, and leakage are audited explicitly.",
            "evidence": [
                ("Corpus adequacy scorecard", "plots/corpus_adequacy_scorecard.png", "Shows why one corpus candidate outranked another on the adequacy objective."),
                ("Feature excitation matrix", "tables/feature_excitation_matrix.csv", "Shows whether designed scenarios actually exercise the feature families."),
                ("Covariate leakage audit", "tables/covariate_leakage_audit.csv", "Quantifies duration, sample-count, and noise leakage risks."),
                ("Candidate corpus comparison", "plots/candidate_corpus_comparison.png", "Shows the autodevelopment frontier across candidate corpora."),
            ],
        },
        {
            "heading": "Claim 6: Filtering helps when dynamics matter",
            "claim": "Model-based evidence providers add value on dynamics-sensitive cases, but that value is conditional and scenario-specific rather than universal.",
            "evidence": [
                ("Kalman innovation likelihood timeline", "plots/kalman_innovation_likelihood_timeline.png", "Shows which motion hypothesis the measurements favored step by step."),
                ("Kalman vs windowed comparison", "plots/kalman_vs_windowed_comparison.png", "Shows aggregate tradeoffs between filtering and feature-only families."),
                ("State estimate vs truth", "plots/state_estimate_vs_truth.png", "Shows whether the correct model actually tracks the latent motion state."),
                ("Model posterior over time", "plots/model_posterior_over_time.png", "Shows the true-model posterior evolution on a representative Kalman witness trajectory."),
            ],
        },
        {
            "heading": "Claim 7: Advanced filters are promoted by evidence",
            "claim": "IMM, PF, and RBPF are implemented as evidence providers and promoted only on the failure cases they measurably improve.",
            "evidence": [
                ("Advanced filter decision matrix", "plots/advanced_filter_decision_matrix.png", "Compact view of the current evidence relevant to advanced-filter escalation."),
                ("Transition scenario summary", "tables/transition_matrix_scenario_summary.csv", "Switching benchmark results for transition-aware accumulation."),
                ("Advanced filter method comparison", "tables/advanced_filter_method_comparison.csv", "Promotion status for IMM, PF, and RBPF on targeted witness failures."),
            ],
        },
        {
            "heading": "Claim 8: 3D transition is planned",
            "claim": "The repo distinguishes dimension-agnostic contracts from scalar-specific implementations, so 3D transition work is planned explicitly rather than hand-waved.",
            "evidence": [
                ("Dimension-lift audit chart", "plots/dimension_lift_audit_chart.png", "Summarizes how much of the stack already transfers to higher dimensions."),
                ("Feature transfer matrix", "plots/feature_transfer_matrix.png", "Shows per-feature dimensional-transfer status from the taxonomy."),
                ("Generic-vs-1D-specific layer diagram", "plots/generic_vs_1d_specific_layer_diagram.png", "Communicates what transfers directly versus what still needs rewrite work."),
            ],
        },
    ]


def _render_proof_gallery() -> str:
    return render_repo_story_proof_gallery()


def _copy_showcase_plots(plots_dir: Path) -> list[dict[str, object]]:
    manifest_entries: list[dict[str, object]] = []
    for definition in _showcase_plot_definitions():
        source = ROOT / definition.source
        destination = plots_dir / definition.filename
        _copy_file(source, destination)
        manifest_entries.append(_plot_manifest_entry(definition))
    return manifest_entries


def _render_pointwise_vs_accumulator_posterior_timelines(plots_dir: Path) -> ShowcaseDerivedPlotArtifact:
    posterior_rows = _read_csv(ARTIFACTS_ROOT / "common_1d_classifier_study" / "unified_posterior_history.csv")
    candidates = [
        row
        for row in posterior_rows
        if row["class_pair_id"] == "constant_velocity_vs_constant_acceleration"
        and row["trajectory_id"] == "constant_velocity_vs_constant_acceleration_short_noisy_constant_acceleration_2"
        and row["classifier_id"] in {"pointwise", "bayes_accumulator"}
    ]
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in candidates:
        grouped.setdefault(row["classifier_id"], []).append(row)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    colors = {"pointwise": "#dc2626", "bayes_accumulator": "#2563eb"}
    for classifier_id, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: float(row["time"]))
        ax.plot(
            [float(row["time"]) for row in ordered],
            [float(row["posterior_class_b"]) for row in ordered],
            linewidth=2.2,
            color=colors[classifier_id],
            marker="o",
            label=f"{classifier_id} true-class posterior",
        )
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Pointwise vs Accumulator Posterior Timelines", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("time")
    ax.set_ylabel("true-class posterior")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = plots_dir / "pointwise_vs_accumulator_posterior_timelines.png"
    write_plot(fig, path)
    return ShowcaseDerivedPlotArtifact(
        plot_id="pointwise_vs_accumulator_posterior_timelines",
        section="history_comparison",
        relative_path=str(path.relative_to(plots_dir.parents[0])),
        source_path="derived:common_1d_classifier_study/unified_posterior_history.csv",
        caption="Pointwise versus sequential-accumulator true-class posterior timelines on the same hard trajectory.",
        interpretation="Use this to see how recursive evidence accumulation differs from a pointwise baseline on the same short noisy case.",
        limitations="This is a representative witness trajectory rather than a full posterior-distribution summary.",
    )


def _render_feature_distribution_by_class(plots_dir: Path) -> ShowcaseDerivedPlotArtifact:
    rows = _read_csv(ARTIFACTS_ROOT / "feature_analysis_v1" / "feature_matrix.csv")
    selected_features = ("position_range", "acceleration_variance", "linear_fit_residual")
    classes = sorted({row["true_class"] for row in rows})
    fig, axes = plt.subplots(1, len(selected_features), figsize=(12.0, 4.2), sharey=False)
    if len(selected_features) == 1:
        axes = [axes]
    for ax, feature_name in zip(axes, selected_features):
        data = [
            [float(row[feature_name]) for row in rows if row["true_class"] == class_name]
            for class_name in classes
        ]
        ax.boxplot(data, tick_labels=classes, patch_artist=True)
        ax.set_title(feature_name, fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Feature Distributions by Class", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = plots_dir / "feature_distribution_by_class.png"
    write_plot(fig, path)
    return ShowcaseDerivedPlotArtifact(
        plot_id="feature_distribution_by_class",
        section="feature_confusion",
        relative_path=str(path.relative_to(plots_dir.parents[0])),
        source_path="derived:feature_analysis_v1/feature_matrix.csv",
        caption="Representative feature distributions by class for three engineered features.",
        interpretation="Use this to inspect whether a feature separates classes through location, spread, or both.",
        limitations="The selected features are representative, not exhaustive.",
    )


def _render_feature_correlation_heatmap(plots_dir: Path) -> ShowcaseDerivedPlotArtifact:
    rows = _read_csv(ARTIFACTS_ROOT / "feature_analysis_v1" / "feature_matrix.csv")
    feature_names = (
        "position_range",
        "speed_range",
        "acceleration_range",
        "duration",
        "acceleration_variance",
        "quadratic_fit_residual",
        "linear_fit_residual",
    )
    values = {name: [float(row[name]) for row in rows] for name in feature_names}
    matrix: list[list[float]] = []
    for left in feature_names:
        left_values = values[left]
        left_mean = sum(left_values) / len(left_values)
        left_var = sum((value - left_mean) ** 2 for value in left_values)
        row_values: list[float] = []
        for right in feature_names:
            right_values = values[right]
            right_mean = sum(right_values) / len(right_values)
            right_var = sum((value - right_mean) ** 2 for value in right_values)
            numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left_values, right_values))
            denominator = math.sqrt(max(left_var * right_var, 1e-12))
            row_values.append(numerator / denominator if denominator > 0.0 else 0.0)
        matrix.append(row_values)
    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(feature_names)), feature_names, rotation=35, ha="right")
    ax.set_yticks(range(len(feature_names)), feature_names)
    ax.set_title("Feature Correlation Heatmap", loc="left", fontsize=12, fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = plots_dir / "feature_correlation_heatmap.png"
    write_plot(fig, path)
    return ShowcaseDerivedPlotArtifact(
        plot_id="feature_correlation_heatmap",
        section="feature_confusion",
        relative_path=str(path.relative_to(plots_dir.parents[0])),
        source_path="derived:feature_analysis_v1/feature_matrix.csv",
        caption="Feature correlation heatmap across the common engineered feature matrix.",
        interpretation="Use this to see where evidence sources are correlated strongly enough to justify double-counting caution.",
        limitations="Linear correlation is only one dependence diagnostic.",
    )


def _render_duration_sensitivity_by_class_pair(plots_dir: Path) -> ShowcaseDerivedPlotArtifact:
    rows = _read_csv(ARTIFACTS_ROOT / "common_1d_classifier_study" / "class_pair_duration_study.csv")
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row["classifier_id"] == "bayes_accumulator":
            grouped.setdefault(row["class_pair_id"], []).append(row)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for class_pair_id, pair_rows in sorted(grouped.items()):
        ordered = sorted(pair_rows, key=lambda row: float(row["time"]))
        ax.plot(
            [float(row["time"]) for row in ordered],
            [float(row["prefix_accuracy"]) for row in ordered],
            linewidth=1.8,
            label=class_pair_id,
        )
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Duration Sensitivity by Class Pair", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("time")
    ax.set_ylabel("prefix accuracy")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    path = plots_dir / "duration_sensitivity_by_class_pair.png"
    write_plot(fig, path)
    return ShowcaseDerivedPlotArtifact(
        plot_id="duration_sensitivity_by_class_pair",
        section="duration_sensitivity",
        relative_path=str(path.relative_to(plots_dir.parents[0])),
        source_path="derived:common_1d_classifier_study/class_pair_duration_study.csv",
        caption="Prefix-accuracy duration sensitivity by class pair for the sequential accumulator.",
        interpretation="Use this to see which class boundaries require more horizon before they separate cleanly.",
        limitations="This is one family-specific duration view, not a universal lower bound.",
    )


def _render_corpus_adequacy_scorecard(plots_dir: Path) -> ShowcaseDerivedPlotArtifact:
    rows = _read_csv(ARTIFACTS_ROOT / "corpus_autodevelopment_v1" / "corpus_candidate_scores.csv")
    top_rows = sorted(rows, key=lambda row: float(row["overall_score"]), reverse=True)[:5]
    metrics = ["balance_score", "boundary_coverage_score", "feature_excitation_score", "difficulty_diversity_score"]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    y_positions = list(range(len(top_rows)))
    for metric, color in zip(metrics, ("#2563eb", "#16a34a", "#f59e0b", "#7c3aed")):
        ax.plot([float(row[metric]) for row in top_rows], y_positions, marker="o", linewidth=1.8, label=metric.replace("_", " "))
    ax.set_yticks(y_positions, [row["candidate_id"] for row in top_rows])
    ax.set_xlim(0.0, 1.05)
    ax.set_title("Corpus Adequacy Scorecard", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("component score")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = plots_dir / "corpus_adequacy_scorecard.png"
    write_plot(fig, path)
    return ShowcaseDerivedPlotArtifact(
        plot_id="corpus_adequacy_scorecard",
        section="corpus_validity",
        relative_path=str(path.relative_to(plots_dir.parents[0])),
        source_path="derived:corpus_autodevelopment_v1/corpus_candidate_scores.csv",
        caption="Scorecard of the leading corpus candidates across adequacy components.",
        interpretation="Use this to see why one corpus candidate outranked another rather than only reading the final overall score.",
        limitations="The scorecard reflects the current objective weights and does not prove universal corpus realism.",
    )


def _render_kalman_innovation_likelihood_timeline(plots_dir: Path) -> ShowcaseDerivedPlotArtifact:
    rows = _read_csv(ARTIFACTS_ROOT / "kalman_filter_bank" / "innovation_history.csv")
    selected = [row for row in rows if row["trajectory_id"] == "constant_velocity_regular_0"]
    class_names = ("stationary", "constant_velocity", "constant_acceleration")
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for class_name, color in zip(class_names, ("#dc2626", "#2563eb", "#16a34a")):
        values = []
        for row in selected:
            innovation = float(row[f"innovation_{class_name}"])
            variance = max(float(row[f"innovation_variance_{class_name}"]), 1e-9)
            values.append(-0.5 * ((innovation * innovation) / variance + math.log(variance)))
        ax.plot([float(row["time"]) for row in selected], values, marker="o", linewidth=2.0, color=color, label=class_name)
    ax.set_title("Kalman Innovation Likelihood Timeline", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("time")
    ax.set_ylabel("innovation log-likelihood proxy")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = plots_dir / "kalman_innovation_likelihood_timeline.png"
    write_plot(fig, path)
    return ShowcaseDerivedPlotArtifact(
        plot_id="kalman_innovation_likelihood_timeline",
        section="filtering",
        relative_path=str(path.relative_to(plots_dir.parents[0])),
        source_path="derived:kalman_filter_bank/innovation_history.csv",
        caption="Innovation likelihood timeline for one representative Kalman-bank trajectory.",
        interpretation="Use this to see which motion hypothesis the measurements supported at each step.",
        limitations="This is one representative trajectory and a proxy likelihood view, not a full posterior audit.",
    )


def _render_kalman_state_estimate_vs_truth(plots_dir: Path) -> list[ShowcaseDerivedPlotArtifact]:
    benchmark = run_kalman_bank_benchmark(seed=7, trajectories_per_class=4)
    run = next(item for item in benchmark.runs if item.scenario_name == "constant_velocity_regular" and item.true_class == "constant_velocity")
    trajectory = next(item for item in benchmark.trajectories if item.trajectory_id == run.trajectory_id)
    posterior_steps = list(run.steps)
    state_history_rows = _read_csv(ARTIFACTS_ROOT / "kalman_filter_bank" / "state_estimate_history.csv")
    selected_rows = [row for row in state_history_rows if row["trajectory_id"] == run.trajectory_id and row["model_name"] == run.true_class]
    fig1, ax1 = plt.subplots(figsize=(8.2, 4.4))
    ax1.plot(trajectory.times, trajectory.true_position, color="#111827", linestyle="--", linewidth=2.0, label="true position")
    ax1.plot(
        [trajectory.times[index] for index in range(len(selected_rows))],
        [float(row["position"]) for row in selected_rows],
        color="#2563eb",
        linewidth=2.0,
        marker="o",
        label="estimated position",
    )
    ax1.plot(trajectory.times, trajectory.measurements, color="#9ca3af", alpha=0.7, marker="x", linestyle=":", label="measurement")
    ax1.set_title("Kalman State Estimate vs Truth", loc="left", fontsize=12, fontweight="bold")
    ax1.set_xlabel("time")
    ax1.set_ylabel("position")
    ax1.grid(True, alpha=0.25)
    ax1.legend(frameon=False)
    fig1.tight_layout()
    path1 = plots_dir / "state_estimate_vs_truth.png"
    write_plot(fig1, path1)

    fig2, ax2 = plt.subplots(figsize=(8.2, 4.4))
    ax2.plot(
        trajectory.times,
        [step.posterior_weights[run.true_class] for step in posterior_steps],
        color="#2563eb",
        linewidth=2.2,
        marker="o",
        label=f"{run.true_class} posterior",
    )
    ax2.set_ylim(0.0, 1.0)
    ax2.set_title("Kalman Model Posterior Over Time", loc="left", fontsize=12, fontweight="bold")
    ax2.set_xlabel("time")
    ax2.set_ylabel("posterior")
    ax2.grid(True, alpha=0.25)
    ax2.legend(frameon=False)
    fig2.tight_layout()
    path2 = plots_dir / "model_posterior_over_time.png"
    write_plot(fig2, path2)

    return [
        ShowcaseDerivedPlotArtifact(
            plot_id="state_estimate_vs_truth",
            section="filtering",
            relative_path=str(path1.relative_to(plots_dir.parents[0])),
            source_path="derived:run_kalman_bank_benchmark(seed=7)",
            caption="Representative Kalman state estimate against truth and measurements.",
            interpretation="Use this to see whether the correct model is tracking the latent trajectory rather than only winning a classifier vote.",
            limitations="This is a benchmark-witness trajectory from the native Kalman study, not the full common-study corpus.",
        ),
        ShowcaseDerivedPlotArtifact(
            plot_id="model_posterior_over_time",
            section="filtering",
            relative_path=str(path2.relative_to(plots_dir.parents[0])),
            source_path="derived:run_kalman_bank_benchmark(seed=7)",
            caption="True-model posterior over time for the representative Kalman witness trajectory.",
            interpretation="Use this to inspect whether the model-based posterior improves steadily or only after late evidence arrives.",
            limitations="This is one trajectory and one family, not a universal filtering calibration study.",
        ),
    ]


def _render_kalman_vs_windowed_comparison(plots_dir: Path) -> ShowcaseDerivedPlotArtifact:
    source = ARTIFACTS_ROOT / "common_dataset_comparison_v1" / "common_dataset_metric_heatmap.png"
    destination = plots_dir / "kalman_vs_windowed_comparison.png"
    _copy_file(source, destination)
    return ShowcaseDerivedPlotArtifact(
        plot_id="kalman_vs_windowed_comparison",
        section="filtering",
        relative_path=str(destination.relative_to(plots_dir.parents[0])),
        source_path=str(source.relative_to(ROOT)),
        caption="Shared-dataset metric heatmap highlighting Kalman versus windowed families.",
        interpretation="Use this to compare where model-based evidence changes the ranking relative to feature-only windowed baselines.",
        limitations="This is an aggregate scenario summary rather than a step-by-step filtering trace.",
    )


def _render_advanced_filter_decision_matrix(plots_dir: Path) -> list[dict[str, object]]:
    method_rows = _load_advanced_filter_method_rows()
    source_path = "artifacts/advanced_filter_comparison_v1/method_comparison.csv"
    if not method_rows:
        source_path = "artifacts/advanced_filter_decision_v1/advanced_filter_decision_summary.json"
    rows = []
    for row in method_rows:
        if not row["method_id"] or row["primary_metric_value"] == "":
            continue
        try:
            primary_metric_value = float(row["primary_metric_value"])
        except ValueError:
            continue
        rows.append((row["method_id"], primary_metric_value, row["promotion_decision"]))
    if not rows:
        fallback = {
            "method_id": "imm_v1",
            "primary_metric_value": 0.0,
            "promotion_decision": "defer",
        }
        rows = [(fallback["method_id"], fallback["primary_metric_value"], fallback["promotion_decision"])]
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    colors = ["#15803d" if decision == "promote" else "#b45309" for _, _, decision in rows]
    ax.bar([name for name, _, _ in rows], [value for _, value, _ in rows], color=colors)
    ax.axhline(0.0, color="#111827", linewidth=1.0)
    ax.set_title("Advanced Filter Promotion Matrix", loc="left", fontsize=12, fontweight="bold")
    ax.set_ylabel("primary witness metric")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path1 = plots_dir / "advanced_filter_decision_matrix.png"
    write_plot(fig, path1)

    pf_rbpf_lines = ["method_id,decision,primary_metric,primary_metric_value,note"]
    for row in method_rows:
        pf_rbpf_lines.append(
            f"{row['method_id']},{row['promotion_decision']},{row['primary_metric']},{row['primary_metric_value']},Promoted only for the targeted failure case."
        )
    path2 = plots_dir.parents[0] / "tables" / "pf_rbpf_go_no_go_table.csv"
    _write_text(path2, "\n".join(pf_rbpf_lines) + "\n")
    path3 = plots_dir.parents[0] / "tables" / "advanced_filter_method_comparison.csv"
    write_csv(path3, list(method_rows), list(method_rows[0].keys()) if method_rows else [])
    plot_artifact = ShowcaseDerivedPlotArtifact(
        plot_id="advanced_filter_decision_matrix",
        section="advanced_filters",
        relative_path=str(path1.relative_to(plots_dir.parents[0])),
        source_path=source_path,
        caption="Decision matrix summarizing advanced-filter promotion evidence by witness.",
        interpretation="Use this to see which advanced filters are promoted on targeted failure cases.",
        limitations="The matrix promotes methods for current witnesses, not universal dominance.",
    )
    return [
        _plot_manifest_entry(plot_artifact),
        _manifest_entry(
            kind="table",
            section="advanced_filters",
            relative_path=str(path2.relative_to(plots_dir.parents[0])),
            source_path=source_path,
        ),
        _manifest_entry(
            kind="table",
            section="advanced_filters",
            relative_path=str(path3.relative_to(plots_dir.parents[0])),
            source_path=source_path,
        ),
    ]


def _render_full_ladder_comparison_dashboard(plots_dir: Path) -> ShowcaseDerivedPlotArtifact:
    rows = _full_ladder_metric_rows()
    metric_specs = (
        ("overall_accuracy", "accuracy"),
        ("negative_log_likelihood", "nll"),
        ("ece", "ece"),
        ("post_switch_accuracy", "post_switch"),
        ("runtime_seconds", "runtime"),
    )
    matrix: list[list[float]] = []
    for row in rows:
        matrix.append([
            float("nan") if _safe_float(row.get(field)) is None else float(_safe_float(row.get(field)))
            for field, _label in metric_specs
        ])
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    colormap = plt.get_cmap("YlGnBu").copy()
    colormap.set_bad(color="#e5e7eb")
    image = ax.imshow(matrix, aspect="auto", cmap=colormap)
    ax.set_title("Full Ladder Comparison Dashboard", loc="left", fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(metric_specs)))
    ax.set_xticklabels([label for _field, label in metric_specs], rotation=20, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([row["method_id"] for row in rows])
    for row_index, row in enumerate(rows):
        for col_index, (field, _label) in enumerate(metric_specs):
            value = _safe_float(row.get(field))
            text = "n/a" if value is None else f"{value:.2f}"
            ax.text(col_index, row_index, text, ha="center", va="center", fontsize=8)
        ax.text(len(metric_specs) - 0.15, row_index + 0.32, row["status"], ha="right", va="center", fontsize=7, color="#111827")
    fig.colorbar(image, ax=ax, label="metric value")
    fig.tight_layout()
    path = plots_dir / "07b_full_ladder_comparison_dashboard.png"
    write_plot(fig, path)
    return ShowcaseDerivedPlotArtifact(
        plot_id="07b_full_ladder_comparison_dashboard",
        section="algorithm_comparison",
        relative_path=str(path.relative_to(plots_dir.parents[0])),
        source_path="derived:full_ladder_metrics.csv",
        caption="Side-by-side dashboard for the full evaluated ladder with study-specific status labels.",
        interpretation="Use this to show that all applicable rungs were evaluated while promotion stayed separate from raw execution.",
        limitations="Some metrics remain blank where the current repo artifacts do not yet expose a common scalar for every rung.",
    )


def _render_advanced_filter_sweet_spot_matrix(plots_dir: Path) -> ShowcaseDerivedPlotArtifact:
    methods = ("IMM", "PF", "RBPF")
    regimes = (
        "mode switching",
        "state mixing",
        "nonlinear dynamics",
        "non-Gaussian noise",
        "multimodal posterior",
        "latent event timing",
        "conditional PVA state",
    )
    scores = {
        "IMM": (1.0, 1.0, 0.3, 0.3, 0.4, 0.5, 0.7),
        "PF": (0.5, 0.4, 1.0, 1.0, 1.0, 0.6, 0.4),
        "RBPF": (0.7, 0.7, 0.6, 0.6, 0.8, 1.0, 1.0),
    }
    matrix = [list(scores[method]) for method in methods]
    fig, ax = plt.subplots(figsize=(9.8, 3.8))
    image = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_title("Advanced Filter Sweet-Spot Matrix", loc="left", fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(regimes)))
    ax.set_xticklabels(regimes, rotation=25, ha="right")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    for row_index, method in enumerate(methods):
        for col_index, value in enumerate(scores[method]):
            label = "high" if value >= 0.85 else ("med" if value >= 0.55 else "low")
            ax.text(col_index, row_index, label, ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="fit to regime")
    fig.tight_layout()
    path = plots_dir / "10e_advanced_filter_sweet_spot_matrix.png"
    write_plot(fig, path)
    return ShowcaseDerivedPlotArtifact(
        plot_id="10e_advanced_filter_sweet_spot_matrix",
        section="advanced_filters",
        relative_path=str(path.relative_to(plots_dir.parents[0])),
        source_path="derived:advanced algorithm showcase witness map",
        caption="Sweet-spot matrix for IMM, PF, and RBPF across targeted failure regimes.",
        interpretation="Use this to show targeted applicability: each advanced filter has a different reason to exist.",
        limitations="This is a capability map backed by current witnesses, not a global performance ranking.",
    )


def _render_simple_to_advanced_witness_bridge(plots_dir: Path) -> ShowcaseDerivedPlotArtifact:
    columns = ("Readable 1D witnesses", "Escalation witnesses", "3D lift pressures")
    rows = (
        ("pointwise overlap", "IMM: mode switching", "vector PVA"),
        ("windowed extrema", "PF: nonlinear posterior", "nonlinear geometry"),
        ("sequential history", "PF: non-Gaussian noise", "sensor noise / occlusion"),
        ("Kalman endpoint match", "RBPF: latent event timing", "mode uncertainty"),
        ("transition switching", "RBPF: conditional state", "multimodal hypotheses"),
    )
    colors = ("#dbeafe", "#dcfce7", "#fef3c7")
    fig, ax = plt.subplots(figsize=(10.8, 4.6))
    ax.set_axis_off()
    ax.set_title("Simple Witnesses Prove the Contract; Advanced Witnesses Prove the Lift Path", loc="left", fontsize=12, fontweight="bold")
    x_positions = (0.16, 0.50, 0.84)
    for col_index, (x_pos, column, color) in enumerate(zip(x_positions, columns, colors, strict=True)):
        ax.text(x_pos, 0.88, column, ha="center", va="center", fontsize=10, fontweight="bold", transform=ax.transAxes)
        for row_index, row in enumerate(rows):
            y_pos = 0.73 - row_index * 0.13
            ax.add_patch(
                Rectangle(
                    (x_pos - 0.145, y_pos - 0.045),
                    0.29,
                    0.075,
                    transform=ax.transAxes,
                    facecolor=color,
                    edgecolor="#374151",
                    linewidth=0.8,
                )
            )
            ax.text(x_pos, y_pos - 0.008, row[col_index], ha="center", va="center", fontsize=8.5, transform=ax.transAxes)
    for y_pos in [0.73 - index * 0.13 for index in range(len(rows))]:
        ax.annotate("", xy=(0.355, y_pos - 0.008), xytext=(0.305, y_pos - 0.008), xycoords=ax.transAxes, arrowprops={"arrowstyle": "->", "linewidth": 1.2, "color": "#111827"})
        ax.annotate("", xy=(0.695, y_pos - 0.008), xytext=(0.645, y_pos - 0.008), xycoords=ax.transAxes, arrowprops={"arrowstyle": "->", "linewidth": 1.2, "color": "#111827"})
    ax.text(
        0.5,
        0.05,
        "1D makes evidence/posterior behavior readable. 3D makes nonlinear geometry, switching, and latent maneuver structure routine.",
        ha="center",
        va="center",
        fontsize=9,
        transform=ax.transAxes,
    )
    fig.tight_layout()
    path = plots_dir / "10f_simple_to_advanced_witness_bridge.png"
    write_plot(fig, path)
    return ShowcaseDerivedPlotArtifact(
        plot_id="10f_simple_to_advanced_witness_bridge",
        section="advanced_filters",
        relative_path=str(path.relative_to(plots_dir.parents[0])),
        source_path="derived:Epic 2 witness bridge",
        caption="Bridge from readable 1D evidence-contract witnesses to advanced witnesses and 3D lift pressures.",
        interpretation="Use this slide to explain why simple 1D sufficiency does not remove the need for IMM, PF, and RBPF showcase witnesses.",
        limitations="This is a story bridge; method promotion still depends on run-backed witness evidence.",
    )


def _render_method_win_by_regime_map(plots_dir: Path) -> ShowcaseDerivedPlotArtifact:
    rows = _method_win_by_regime_rows()
    methods = [key for key in rows[0].keys() if key != "regime"]
    score_map = {"fails": 0.0, "not_applicable": 0.25, "competitive": 0.7, "best": 1.0}
    matrix = [[score_map[row[method]] for method in methods] for row in rows]
    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    image = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    ax.set_title("Method Win by Regime Map", loc="left", fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([row["regime"] for row in rows])
    for row_index, row in enumerate(rows):
        for col_index, method in enumerate(methods):
            ax.text(col_index, row_index, row[method], ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="regime result")
    fig.tight_layout()
    path = plots_dir / "10f_method_win_by_regime_map.png"
    write_plot(fig, path)
    return ShowcaseDerivedPlotArtifact(
        plot_id="10f_method_win_by_regime_map",
        section="advanced_filters",
        relative_path=str(path.relative_to(plots_dir.parents[0])),
        source_path="derived:method_win_by_regime.csv",
        caption="Regime-by-method map showing where each rung wins, competes, fails, or is not applicable.",
        interpretation="Use this to answer where methods shine without forcing one global winner.",
        limitations="This first-pass map is curated from current witness and shared-study evidence rather than a fully automated cross-study registry.",
    )


def _render_dimensional_lift_plots(plots_dir: Path) -> list[ShowcaseDerivedPlotArtifact]:
    rows = _read_csv(ARTIFACTS_ROOT / "dimensional_lift_audit" / "module_dimension_status.csv")
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["dimensional_status"]] = counts.get(row["dimensional_status"], 0) + 1
    fig1, ax1 = plt.subplots(figsize=(7.0, 4.2))
    labels = list(counts)
    ax1.bar(labels, [counts[label] for label in labels], color=["#2563eb", "#f59e0b", "#dc2626"][: len(labels)])
    ax1.set_title("Dimension-Lift Audit Chart", loc="left", fontsize=12, fontweight="bold")
    ax1.set_ylabel("module count")
    ax1.tick_params(axis="x", rotation=20)
    ax1.grid(True, axis="y", alpha=0.25)
    fig1.tight_layout()
    path1 = plots_dir / "dimension_lift_audit_chart.png"
    write_plot(fig1, path1)

    taxonomy_rows = _read_json(ARTIFACTS_ROOT / "feature_taxonomy" / "feature_taxonomy.json")
    transfer_modes = ["dimension_agnostic", "adapter_compatible", "rewrite_required"]
    feature_names = [row["name"] for row in taxonomy_rows]
    matrix = [[1.0 if row["dimensional_transfer"] == mode else 0.0 for mode in transfer_modes] for row in taxonomy_rows]
    fig2, ax2 = plt.subplots(figsize=(7.5, max(4.5, 0.18 * len(feature_names))))
    image = ax2.imshow(matrix, cmap="Blues", aspect="auto", vmin=0.0, vmax=1.0)
    ax2.set_xticks(range(len(transfer_modes)), transfer_modes, rotation=20, ha="right")
    ax2.set_yticks(range(len(feature_names)), feature_names)
    ax2.set_title("Feature Transfer Matrix", loc="left", fontsize=12, fontweight="bold")
    fig2.colorbar(image, ax=ax2, fraction=0.046, pad=0.04)
    fig2.tight_layout()
    path2 = plots_dir / "feature_transfer_matrix.png"
    write_plot(fig2, path2)

    fig3, ax3 = plt.subplots(figsize=(9.0, 4.4))
    ax3.axis("off")
    boxes = [
        (0.10, 0.65, "Dimension-agnostic\ncontracts / evaluation"),
        (0.42, 0.65, "Adapter-compatible\nshared corpus / harness"),
        (0.74, 0.65, "1D-specific\nfeature + filter logic"),
    ]
    for x, y, text in boxes:
        ax3.text(x, y, text, ha="center", va="center", fontsize=11, bbox=dict(boxstyle="round,pad=0.4", facecolor="#e5e7eb", edgecolor="#6b7280"))
    ax3.annotate("", xy=(0.34, 0.65), xytext=(0.18, 0.65), arrowprops=dict(arrowstyle="->", linewidth=2.0))
    ax3.annotate("", xy=(0.66, 0.65), xytext=(0.50, 0.65), arrowprops=dict(arrowstyle="->", linewidth=2.0))
    ax3.text(0.26, 0.78, "transfers directly", ha="center", fontsize=9)
    ax3.text(0.58, 0.78, "needs vector adapters", ha="center", fontsize=9)
    ax3.text(0.82, 0.48, "rewrite required for\ntrue 3D dynamics", ha="center", fontsize=9)
    ax3.set_title("Generic vs 1D-Specific Layer Diagram", loc="left", fontsize=12, fontweight="bold")
    path3 = plots_dir / "generic_vs_1d_specific_layer_diagram.png"
    write_plot(fig3, path3)

    return [
        ShowcaseDerivedPlotArtifact(
            plot_id="dimension_lift_audit_chart",
            section="dimensional_lift",
            relative_path=str(path1.relative_to(plots_dir.parents[0])),
            source_path="derived:dimensional_lift_audit/module_dimension_status.csv",
            caption="Count of modules by dimensional-lift status.",
            interpretation="Use this to see how much of the stack already transfers to higher dimensions versus needing adapters or rewrites.",
            limitations="Counts summarize status classes; they do not capture implementation effort exactly.",
        ),
        ShowcaseDerivedPlotArtifact(
            plot_id="feature_transfer_matrix",
            section="dimensional_lift",
            relative_path=str(path2.relative_to(plots_dir.parents[0])),
            source_path="derived:feature_taxonomy/feature_taxonomy.json",
            caption="Per-feature dimensional-transfer status matrix.",
            interpretation="Use this to see which feature families already generalize cleanly and which still embed scalar assumptions.",
            limitations="This reflects the current taxonomy declarations, not yet a vector empirical benchmark.",
        ),
        ShowcaseDerivedPlotArtifact(
            plot_id="generic_vs_1d_specific_layer_diagram",
            section="dimensional_lift",
            relative_path=str(path3.relative_to(plots_dir.parents[0])),
            source_path="derived:dimensional_lift_audit/module_dimension_status.csv",
            caption="Layer diagram separating generic methodology surfaces from 1D-specific implementation layers.",
            interpretation="Use this to communicate what transfers directly to 3D versus what still needs vector adapters or rewrites.",
            limitations="This is a conceptual summary diagram rather than a runtime dependency graph.",
        ),
    ]


def _generate_showcase_derived_plots(plots_dir: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    entries.append(_plot_manifest_entry(_render_pointwise_vs_accumulator_posterior_timelines(plots_dir)))
    quantile_source = ARTIFACTS_ROOT / "monte_carlo_accumulator" / "true_class_posterior_quantiles.png"
    quantile_destination = plots_dir / "true_class_posterior_quantiles.png"
    _copy_file(quantile_source, quantile_destination)
    entries.append(
        _plot_manifest_entry(
            ShowcaseDerivedPlotArtifact(
                plot_id="true_class_posterior_quantiles",
                section="history_comparison",
                relative_path=str(quantile_destination.relative_to(plots_dir.parents[0])),
                source_path=str(quantile_source.relative_to(ROOT)),
                caption="True-class posterior quantiles from the Monte Carlo accumulator benchmark.",
                interpretation="Use this to summarize how quickly true-class posterior mass grows across repeated runs rather than one witness trajectory.",
                limitations="This plot comes from the accumulator benchmark, not every classifier family.",
            )
        )
    )
    ablation_source = ARTIFACTS_ROOT / "bayesian_walkthroughs" / "plots" / "feature_ablation_posterior.png"
    ablation_destination = plots_dir / "feature_ablation_posterior.png"
    _copy_file(ablation_source, ablation_destination)
    entries.append(
        _plot_manifest_entry(
            ShowcaseDerivedPlotArtifact(
                plot_id="feature_ablation_posterior",
                section="feature_confusion",
                relative_path=str(ablation_destination.relative_to(plots_dir.parents[0])),
                source_path=str(ablation_source.relative_to(ROOT)),
                caption="Posterior change after feature ablation on a representative feature-only walkthrough.",
                interpretation="Use this to see which feature contributed most to the final class preference in the witness case.",
                limitations="This remains a feature-only walkthrough, not a universal importance measure.",
            )
        )
    )
    entries.append(_plot_manifest_entry(_render_feature_distribution_by_class(plots_dir)))
    entries.append(_plot_manifest_entry(_render_feature_correlation_heatmap(plots_dir)))
    entries.append(_plot_manifest_entry(_render_duration_sensitivity_by_class_pair(plots_dir)))
    entries.append(_plot_manifest_entry(_render_corpus_adequacy_scorecard(plots_dir)))
    pareto_source = ARTIFACTS_ROOT / "corpus_autodevelopment_v1" / "plots" / "corpus_score_pareto.png"
    pareto_destination = plots_dir / "candidate_corpus_comparison.png"
    _copy_file(pareto_source, pareto_destination)
    entries.append(
        _plot_manifest_entry(
            ShowcaseDerivedPlotArtifact(
                plot_id="candidate_corpus_comparison",
                section="corpus_validity",
                relative_path=str(pareto_destination.relative_to(plots_dir.parents[0])),
                source_path=str(pareto_source.relative_to(ROOT)),
                caption="Pareto-style comparison of candidate corpora under the current adequacy objective.",
                interpretation="Use this to see whether corpus improvement came from one clear winner or from tradeoffs between candidates.",
                limitations="All current candidates still fail the full adequacy gate, so this is a relative comparison rather than a final green corpus.",
            )
        )
    )
    entries.append(_plot_manifest_entry(_render_kalman_innovation_likelihood_timeline(plots_dir)))
    entries.append(_plot_manifest_entry(_render_kalman_vs_windowed_comparison(plots_dir)))
    entries.extend(_plot_manifest_entry(item) for item in _render_kalman_state_estimate_vs_truth(plots_dir))
    entries.append(_plot_manifest_entry(_render_full_ladder_comparison_dashboard(plots_dir)))
    entries.extend(_render_advanced_filter_decision_matrix(plots_dir))
    entries.append(_plot_manifest_entry(_render_advanced_filter_sweet_spot_matrix(plots_dir)))
    entries.append(_plot_manifest_entry(_render_simple_to_advanced_witness_bridge(plots_dir)))
    entries.append(_plot_manifest_entry(_render_method_win_by_regime_map(plots_dir)))
    entries.extend(_plot_manifest_entry(item) for item in _render_dimensional_lift_plots(plots_dir))
    return entries


def _build_run_cards(run_cards_dir: Path) -> list[dict[str, object]]:
    cards = [
        RunCardSpec(
            filename="common_study.md",
            title="Common Study Run Card",
            body=RunCardBody(
                heading="Common Study Run Card",
                bullets=(
                    "Study: `common_1d_classifier_study`",
                    "Purpose: compare classifier families on manifest-declared class pairs using one unified artifact contract",
                    "Primary report: `../reports/03_algorithm_ladder.md`",
                    "Primary source artifact: `../../common_1d_classifier_study/common_experiment_report.md`",
                    "Rerun: `python3 scripts/run_study.py experiments/common_1d_classifier_study/common_experiment_config.yaml`",
                ),
            ),
        ),
        RunCardSpec(
            filename="common_dataset_comparison.md",
            title="Shared-Corpus Comparison Run Card",
            body=RunCardBody(
                heading="Shared-Corpus Comparison Run Card",
                bullets=(
                    "Study: `common_dataset_comparison_v1`",
                    "Purpose: compare methods on the same shared binary dynamics corpus",
                    "Primary report: `../reports/03_algorithm_ladder.md`",
                    "Primary source artifact: `../../common_dataset_comparison_v1/common_dataset_comparison_report.md`",
                ),
            ),
        ),
        RunCardSpec(
            filename="feature_analysis.md",
            title="Feature Analysis Run Card",
            body=RunCardBody(
                heading="Feature Analysis Run Card",
                bullets=(
                    "Study: `feature_analysis_v1`",
                    "Purpose: identify feature excitation, class confusability, and separability limits",
                    "Primary report: `../reports/04_feature_taxonomy.md`",
                    "Primary source artifact: `../../feature_analysis_v1/feature_analysis_report.md`",
                ),
            ),
        ),
        RunCardSpec(
            filename="corpus_adequacy.md",
            title="Corpus Adequacy Run Card",
            body=RunCardBody(
                heading="Corpus Adequacy Run Card",
                bullets=(
                    "Study: `corpus_adequacy_audit_v1`",
                    "Purpose: determine whether the synthetic corpus is credible enough to support classifier comparisons",
                    "Primary report: `../reports/08_results_summary.md`",
                    "Primary source artifact: `../../corpus_adequacy_audit_v1/corpus_adequacy_report.md`",
                ),
            ),
        ),
        RunCardSpec(
            filename="advanced_filter_decision.md",
            title="Advanced Filter Decision Run Card",
            body=RunCardBody(
                heading="Advanced Filter Decision Run Card",
                bullets=(
                    "Study: `advanced_filter_decision_v1`",
                    "Purpose: decide whether IMM, PF, or RBPF is justified by current failure cases",
                    "Primary report: `../reports/05_filtering_taxonomy.md`",
                    "Primary source artifact: `../../advanced_filter_decision_v1/advanced_filter_decision_report.md`",
                ),
            ),
        ),
        RunCardSpec(
            filename="filter_trace_validation.md",
            title="Filter Trace Validation Run Card",
            body=RunCardBody(
                heading="Filter Trace Validation Run Card",
                bullets=(
                    "Study: `filter_trace_validation_v1`",
                    "Purpose: prove that step-level prior, prediction, measurement, likelihood, posterior, and diagnostic traces are emitted consistently",
                    "Primary report: `../reports/05_filtering_taxonomy.md`",
                    "Primary source artifact: `../../filter_trace_validation_v1/filter_trace_validation_report.md`",
                ),
            ),
        ),
        RunCardSpec(
            filename="dimensional_lift.md",
            title="Dimensional Lift Run Card",
            body=RunCardBody(
                heading="Dimensional Lift Run Card",
                bullets=(
                    "Study: `dimensional_lift_audit`",
                    "Purpose: identify which methodology layers already support 3D transition work and which remain scalar-specific",
                    "Primary report: `../reports/09_3d_transition_plan.md`",
                    "Primary source artifact: `../../dimensional_lift_audit/dimensional_lift_audit.md`",
                ),
            ),
        ),
    ]
    manifest_entries: list[dict[str, object]] = []
    for card in cards:
        path = run_cards_dir / card.filename
        report = MarkdownDocument(card.body.heading)
        report.bullet_list([str(item) for item in card.body.bullets])
        _write_text(path, report.text())
        manifest_entries.append(
            _manifest_entry(
                kind="run_card",
                title=card.title,
                relative_path=str(path.relative_to(run_cards_dir.parents[0])),
            )
        )
    return manifest_entries


def _copy_showcase_sources(reports_dir: Path) -> list[dict[str, object]]:
    source_specs = [
        ShowcaseSourceDocSpec(
            source=source,
            relative_destination=str((Path("reports") / "source_docs" / source.name)),
        )
        for source in sorted(SHOWCASE_DOCS_DIR.glob("*.md"))
    ]
    manifest_entries: list[dict[str, object]] = []
    for spec in source_specs:
        destination = reports_dir.parents[0] / spec.relative_destination
        _copy_file(spec.source, destination)
        manifest_entries.append(
            _manifest_entry(
                kind="source_doc",
                relative_path=str(destination.relative_to(reports_dir.parents[0])),
                source_path=str(spec.source.relative_to(ROOT)),
            )
        )
    return manifest_entries
