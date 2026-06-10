from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from statistics import median
from typing import Any

from ..utils.io import write_csv
from ..utils.plotting import plt, write_plot
from .contracts import StudyConfidenceArtifacts, StudyConfidenceResult


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _safe_float(value: object, default: float = 0.0) -> float:
    if value in ("", None):
        return default
    return float(value)


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _geometric_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    clipped = [_clamp(value, 1.0e-6, 1.0) for value in values]
    return math.exp(sum(math.log(value) for value in clipped) / len(clipped))


def _read_csv(path: Path) -> list[dict[str, str]]:
    import csv

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _status_score(status: str) -> float:
    mapping = {
        "pass": 1.0,
        "warn": 0.65,
        "green": 1.0,
        "yellow": 0.65,
        "red": 0.2,
        "fail": 0.1,
        "unknown": 0.5,
    }
    return mapping.get(status, 0.5)


def _confidence_band(score: float) -> str:
    if score >= 0.80:
        return "high"
    if score >= 0.60:
        return "moderate"
    if score >= 0.35:
        return "low"
    return "blocked"


def _study_row_key(row: dict[str, object]) -> tuple[str, str, str]:
    return (
        str(row.get("classifier_id", "")),
        str(row.get("feature_set_id", "")),
        str(row.get("class_pair_id", "")),
    )


def _normalize_accuracy(accuracy: float) -> float:
    return _clamp(accuracy)


def _normalize_gap_to_score(gap: float, *, scale: float) -> float:
    return _clamp(1.0 - (gap / scale))


def _load_corpus_terms(audit_dir: Path) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    gate = _load_json(audit_dir / "corpus_decision_gate.json")
    summary = _load_json(audit_dir / "corpus_adequacy_summary.json")["summary"]
    leakage_rows = _read_csv(audit_dir / "leakage_audit.csv")
    return gate, leakage_rows, summary


def _corpus_confidence_terms(
    gate: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, float]:
    validity = _clamp(float(gate["class_validity"]["valid_target_class_fraction"]))
    ambiguity_penalty = _clamp(1.0 - float(gate["class_validity"]["ambiguous_fraction"]) - float(gate["class_validity"]["invalid_fraction"]))
    q_corpus = _clamp(_safe_float(summary.get("q_corpus"), 0.0))
    class_validity_score = _clamp(_safe_float(summary.get("class_validity_score"), 0.0))
    feature_status = _status_score(str(summary.get("feature_status", "unknown")))
    class_pair_status = _status_score(str(summary.get("class_pair_status", "unknown")))
    class_balance_status = _status_score(str(summary.get("class_balance_status", "unknown")))
    leakage_penalty = _clamp(1.0 - _safe_float(summary.get("leakage_penalty"), 1.0))
    degeneracy_penalty = _clamp(1.0 - _safe_float(summary.get("degeneracy_penalty"), 1.0))
    triviality_penalty = _clamp(1.0 - _safe_float(summary.get("triviality_penalty"), 1.0))
    corpus_gate = 1.0 if bool(gate["overall_pass"]) else 0.10
    corpus_confidence = _geometric_mean(
        [
            validity,
            ambiguity_penalty,
            q_corpus,
            class_validity_score,
            feature_status,
            class_pair_status,
            class_balance_status,
            leakage_penalty,
            degeneracy_penalty,
            triviality_penalty,
        ]
    )
    return {
        "corpus_gate": corpus_gate,
        "corpus_confidence": corpus_confidence,
        "validity": validity,
        "ambiguity_penalty": ambiguity_penalty,
        "q_corpus": q_corpus,
        "class_validity_score": class_validity_score,
        "feature_status": feature_status,
        "class_pair_status": class_pair_status,
        "class_balance_status": class_balance_status,
        "leakage_penalty": leakage_penalty,
        "degeneracy_penalty": degeneracy_penalty,
        "triviality_penalty": triviality_penalty,
    }


def _build_component_rows(
    *,
    classifier_row: dict[str, object],
    corpus_terms: dict[str, float],
    promotion_row: dict[str, str] | None,
    failure_row: dict[str, str] | None,
    prior_rows: list[dict[str, str]],
) -> tuple[list[dict[str, object]], dict[str, float], dict[str, object]]:
    accuracy = _safe_float(classifier_row.get("overall_accuracy"), 0.0)
    mean_confidence = _safe_float(classifier_row.get("mean_confidence"), accuracy)
    oracle_accuracy = _safe_float(promotion_row.get("oracle_accuracy") if promotion_row else None, accuracy)
    oracle_gap = _safe_float(promotion_row.get("oracle_gap") if promotion_row else None, max(0.0, oracle_accuracy - accuracy))
    measured_improvement = _safe_float(promotion_row.get("measured_improvement") if promotion_row else None, 0.0)
    runtime_cost_ratio = _safe_float(promotion_row.get("runtime_cost_ratio") if promotion_row else None, 1.0)
    decision = str(promotion_row.get("decision")) if promotion_row else "defer_advanced"
    failure_mode = str(failure_row.get("failure_mode")) if failure_row else "insufficient_data"
    prior_accuracies = [_safe_float(row.get("accuracy"), accuracy) for row in prior_rows]
    prior_spread = (max(prior_accuracies) - min(prior_accuracies)) if prior_accuracies else 0.0
    prior_stability = _clamp(1.0 - (prior_spread / 0.50))
    calibration_norm = _clamp(1.0 - (abs(mean_confidence - accuracy) / 0.50))
    error_control_norm = _clamp(1.0 - max(mean_confidence - accuracy, 0.0))
    temporal_norm = _clamp(mean_confidence)
    classifier_confidence = _geometric_mean(
        [
            _normalize_accuracy(accuracy),
            calibration_norm,
            error_control_norm,
            temporal_norm,
        ]
    )
    learnability_confidence = _geometric_mean(
        [
            _normalize_accuracy(oracle_accuracy),
            _normalize_gap_to_score(oracle_gap, scale=0.50),
            _clamp((oracle_accuracy - max(oracle_gap, 0.0))),
        ]
    )
    method_agreement = 1.0
    regime_stability = 1.0
    robustness_confidence = _geometric_mean([prior_stability, method_agreement, regime_stability])
    if decision == "stay" and oracle_gap <= 0.08:
        sufficiency_confidence = 1.0
    elif decision == "promote" and measured_improvement >= 0.05:
        sufficiency_confidence = 0.75
    elif decision == "defer_advanced":
        sufficiency_confidence = 0.40
    elif decision == "reject_escalation":
        sufficiency_confidence = 0.20
    elif decision.startswith("revise"):
        sufficiency_confidence = 0.10
    else:
        sufficiency_confidence = 0.30
    if failure_mode in {"corpus_limited", "feature_limited"}:
        sufficiency_confidence = min(sufficiency_confidence, 0.05)
    study_claim_confidence = corpus_terms["corpus_gate"] * _geometric_mean(
        [
            corpus_terms["corpus_confidence"],
            learnability_confidence,
            classifier_confidence,
            robustness_confidence,
            sufficiency_confidence,
        ]
    )
    recommended_action = (
        "revise_corpus"
        if failure_mode == "corpus_limited"
        else "revise_features"
        if failure_mode == "feature_limited"
        else "revise_prior"
        if failure_mode == "prior_limited"
        else "promote"
        if decision == "promote"
        else "stay"
        if decision == "stay"
        else "defer"
    )
    classifier_meta = {
        "oracle_accuracy": oracle_accuracy,
        "oracle_gap": oracle_gap,
        "measured_improvement": measured_improvement,
        "runtime_cost_ratio": runtime_cost_ratio,
        "decision": decision,
        "failure_mode": failure_mode,
        "recommended_action": recommended_action,
        "study_claim_confidence": study_claim_confidence,
        "confidence_band": _confidence_band(study_claim_confidence),
    }
    component_values = {
        "corpus_confidence": corpus_terms["corpus_confidence"],
        "learnability_confidence": learnability_confidence,
        "classifier_confidence": classifier_confidence,
        "robustness_confidence": robustness_confidence,
        "sufficiency_confidence": sufficiency_confidence,
    }
    rows = []
    for component_name, score in component_values.items():
        rows.append(
            {
                "study_id": classifier_row.get("study_id", ""),
                "classifier_id": classifier_row["classifier_id"],
                "rung_id": classifier_row["rung_id"],
                "feature_set_id": classifier_row["feature_set_id"],
                "class_pair_id": classifier_row["class_pair_id"],
                "component_name": component_name,
                "normalized_score": round(score, 6),
                "reference_value": round(corpus_terms["corpus_gate"] if component_name == "corpus_confidence" else 1.0, 6),
                "penalty_direction": "lower_is_worse",
                "rationale": {
                    "corpus_confidence": "Trust in the underlying corpus and declared class/feature support.",
                    "learnability_confidence": "Whether the task looks solvable with the current representation.",
                    "classifier_confidence": "Observed classifier quality and calibration proxy.",
                    "robustness_confidence": "Stability under prior changes and missing-regime fallback.",
                    "sufficiency_confidence": "Whether the current rung is enough or a stronger rung is justified.",
                }[component_name],
            }
        )
    return rows, component_values, classifier_meta


def analyze_study_confidence(workflow_root: str | Path, study: dict[str, Any]) -> StudyConfidenceResult:
    root = Path(workflow_root)
    audit_dir = root / "03_corpus_audit"
    ladder_dir = root / "04_ladder_evaluation"
    gate, _leakage_rows, adequacy_summary = _load_corpus_terms(audit_dir)
    corpus_terms = _corpus_confidence_terms(gate, adequacy_summary)
    method_rows = _read_csv(ladder_dir / "method_metrics.csv")
    promotion_rows = {_study_row_key(row): row for row in _read_csv(ladder_dir / "sufficiency_matrix.csv")}
    failure_rows = {_study_row_key(row): row for row in _read_csv(ladder_dir / "insufficiency_matrix.csv")}
    prior_rows = _read_csv(ladder_dir / "prior_sensitivity_by_method.csv")
    component_rows: list[dict[str, object]] = []
    classifier_rows: list[dict[str, object]] = []

    for method_row in method_rows:
        key = _study_row_key(method_row)
        relevant_prior_rows = [
            row
            for row in prior_rows
            if str(row.get("classifier_id", "")) == str(method_row["classifier_id"])
            and str(row.get("class_pair_id", "")) == str(method_row["class_pair_id"])
        ]
        rows, component_values, meta = _build_component_rows(
            classifier_row={
                **method_row,
                "study_id": str(study["study_id"]),
            },
            corpus_terms=corpus_terms,
            promotion_row=promotion_rows.get(key),
            failure_row=failure_rows.get(key),
            prior_rows=relevant_prior_rows,
        )
        component_rows.extend(rows)
        classifier_rows.append(
            {
                "study_id": str(study["study_id"]),
                "classifier_id": str(method_row["classifier_id"]),
                "rung_id": str(method_row["rung_id"]),
                "feature_set_id": str(method_row["feature_set_id"]),
                "class_pair_id": str(method_row["class_pair_id"]),
                "overall_accuracy": round(_safe_float(method_row["overall_accuracy"]), 6),
                "mean_confidence": round(_safe_float(method_row["mean_confidence"]), 6),
                **{name: round(value, 6) for name, value in component_values.items()},
                "final_confidence": round(meta["study_claim_confidence"], 6),
                "confidence_band": meta["confidence_band"],
                "recommended_action": meta["recommended_action"],
                "decision": meta["decision"],
                "failure_mode": meta["failure_mode"],
                "oracle_accuracy": round(meta["oracle_accuracy"], 6),
                "oracle_gap": round(meta["oracle_gap"], 6),
                "measured_improvement": round(meta["measured_improvement"], 6),
                "runtime_cost_ratio": round(meta["runtime_cost_ratio"], 6),
                "corpus_status": str(gate["overall_status"]),
                "corpus_pass": bool(gate["overall_pass"]),
            }
        )

    final_scores = [float(row["final_confidence"]) for row in classifier_rows]
    selected_row = max(classifier_rows, key=lambda row: float(row["final_confidence"])) if classifier_rows else None
    summary = {
        "study_id": str(study["study_id"]),
        "title": str(study["title"]),
        "corpus_status": str(gate["overall_status"]),
        "corpus_pass": bool(gate["overall_pass"]),
        "aggregate_confidence": min(final_scores) if final_scores else 0.0,
        "best_case_confidence": max(final_scores) if final_scores else 0.0,
        "median_classifier_confidence": median(final_scores) if final_scores else 0.0,
        "confidence_band": _confidence_band(min(final_scores) if final_scores else 0.0),
        "classifier_count": len(classifier_rows),
        "selected_claim": selected_row,
    }
    report_lines = [
        f"# Study Confidence Report: {study['title']}",
        "",
        f"- Study ID: `{study['study_id']}`",
        f"- Corpus status: `{gate['overall_status']}`",
        f"- Aggregate confidence: `{summary['aggregate_confidence']:.3f}`",
        f"- Best-case confidence: `{summary['best_case_confidence']:.3f}`",
        f"- Confidence band: `{summary['confidence_band']}`",
        "",
        "## Corpus Trust",
        "",
        f"- Corpus confidence: `{corpus_terms['corpus_confidence']:.3f}`",
        f"- Corpus gate: `{corpus_terms['corpus_gate']:.2f}`",
        f"- Valid target fraction: `{corpus_terms['validity']:.3f}`",
        f"- Leakage penalty retention: `{corpus_terms['leakage_penalty']:.3f}`",
        "",
        "## Classifier Rows",
        "",
    ]
    for row in sorted(classifier_rows, key=lambda item: float(item["final_confidence"]), reverse=True)[:8]:
        report_lines.append(
            f"- `{row['classifier_id']}` on `{row['class_pair_id']}` / `{row['feature_set_id']}`: "
            f"confidence `{float(row['final_confidence']):.3f}`, action `{row['recommended_action']}`, "
            f"failure `{row['failure_mode']}`, oracle gap `{float(row['oracle_gap']):.3f}`"
        )
    return StudyConfidenceResult(
        component_rows=tuple(component_rows),
        classifier_rows=tuple(classifier_rows),
        summary=summary,
        report_markdown="\n".join(report_lines) + "\n",
    )


def _build_dashboard(result: StudyConfidenceResult, path: Path) -> None:
    rows = list(result.classifier_rows)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, max(4.5, 0.55 * max(len(rows), 1))))
    labels = [f"{row['classifier_id']}\n{row['class_pair_id']}" for row in rows]
    final_scores = [float(row["final_confidence"]) for row in rows]
    corpus_scores = [float(row["corpus_confidence"]) for row in rows]
    classifier_scores = [float(row["classifier_confidence"]) for row in rows]
    sufficiency_scores = [float(row["sufficiency_confidence"]) for row in rows]

    axes[0].barh(labels, final_scores, color="#1d4ed8")
    axes[0].set_title("Final Confidence by Classifier")
    axes[0].set_xlabel("confidence")
    axes[0].set_xlim(0.0, 1.0)
    axes[0].grid(axis="x", alpha=0.25)

    y_positions = list(range(len(labels)))
    axes[1].barh(y_positions, corpus_scores, color="#059669", label="corpus")
    axes[1].barh(y_positions, classifier_scores, color="#f59e0b", alpha=0.75, label="classifier")
    axes[1].barh(y_positions, sufficiency_scores, color="#dc2626", alpha=0.55, label="sufficiency")
    axes[1].set_yticks(y_positions)
    axes[1].set_yticklabels(labels)
    axes[1].set_title("Key Component Scores")
    axes[1].set_xlabel("component score")
    axes[1].set_xlim(0.0, 1.0)
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="x", alpha=0.25)
    fig.tight_layout()
    write_plot(fig, path)


def write_study_confidence_artifacts(
    output_dir: str | Path,
    *,
    workflow_root: str | Path,
    study: dict[str, Any],
) -> StudyConfidenceArtifacts:
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    result = analyze_study_confidence(workflow_root, study)
    components_path = run_dir / "confidence_components.csv"
    classifier_scores_path = run_dir / "confidence_by_classifier.csv"
    summary_path = run_dir / "study_confidence_summary.json"
    report_path = run_dir / "confidence_report.md"
    dashboard_path = run_dir / "confidence_dashboard.png"
    if result.component_rows:
        write_csv(components_path, list(result.component_rows), list(result.component_rows[0].keys()))
    else:
        write_csv(components_path, [], ["study_id", "classifier_id", "rung_id", "feature_set_id", "class_pair_id", "component_name", "normalized_score", "reference_value", "penalty_direction", "rationale"])
    if result.classifier_rows:
        write_csv(classifier_scores_path, list(result.classifier_rows), list(result.classifier_rows[0].keys()))
    else:
        write_csv(classifier_scores_path, [], ["study_id", "classifier_id", "rung_id", "feature_set_id", "class_pair_id", "final_confidence"])
    summary_path.write_text(json.dumps(result.summary, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(result.report_markdown, encoding="utf-8")
    _build_dashboard(result, dashboard_path)
    return StudyConfidenceArtifacts(
        run_dir=run_dir,
        components_path=components_path,
        classifier_scores_path=classifier_scores_path,
        summary_path=summary_path,
        report_path=report_path,
        dashboard_path=dashboard_path,
    )

