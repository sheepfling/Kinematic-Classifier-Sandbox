from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from _bootstrap import bootstrap_repo

ROOT = bootstrap_repo(configure_runtime=True)

from typing import Any

import yaml
from numpy import array, corrcoef, isfinite


from kinematic_classifier_sandbox.analysis.feature_analysis_artifact_io import (
    write_feature_analysis_artifacts,
)
from kinematic_classifier_sandbox.analysis.generated_corpus_features import (
    write_generated_corpus_feature_artifacts,
)
from kinematic_classifier_sandbox.common_experiment_harness import (
    write_common_experiment_artifacts,
)
from kinematic_classifier_sandbox.corpus.adequacy_artifact_io import write_corpus_adequacy_artifacts
from kinematic_classifier_sandbox.corpus.selected_generated_corpus_artifact_io import (
    write_selected_generated_corpus_artifacts,
)
from kinematic_classifier_sandbox.corpus.exploration.candidate_generation_rendering import (
    write_candidate_generation_artifacts,
)
from kinematic_classifier_sandbox.corpus.exploration.generic_corpus_exploration import (
    write_generic_corpus_exploration_weight_sweep_artifacts,
)
from kinematic_classifier_sandbox.corpus.policy_sweep import (
    write_corpus_policy_tuning_artifacts,
)
from kinematic_classifier_sandbox.corpus.autodevelopment import (
    write_corpus_autodevelopment_artifacts,
)
from kinematic_classifier_sandbox.rung_sufficiency.analysis import (
    write_rung_sufficiency_artifacts,
)



PHASES = (
    "00_study_declaration",
    "01_feature_class_analysis",
    "02_corpus_generation",
    "03_corpus_audit",
    "04_ladder_evaluation",
    "04b_confidence",
    "05_report",
)


def load_study_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("study config must be a mapping")
    return payload


def workflow_root(output_dir: str | Path, study: dict[str, Any]) -> Path:
    return Path(output_dir) / str(study["study_id"])


def phase_dir(output_dir: str | Path, study: dict[str, Any], phase_name: str) -> Path:
    path = workflow_root(output_dir, study) / phase_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_workflow_dirs(output_dir: str | Path, study: dict[str, Any]) -> Path:
    root = workflow_root(output_dir, study)
    root.mkdir(parents=True, exist_ok=True)
    for name in PHASES:
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved_fieldnames = fieldnames or (list(rows[0].keys()) if rows else [])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=resolved_fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_feature_manifest() -> dict[str, dict[str, object]]:
    path = ROOT / "experiments" / "common_1d_classifier_study" / "feature_sets.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_classifier_manifest() -> dict[str, dict[str, object]]:
    path = ROOT / "experiments" / "common_1d_classifier_study" / "classifier_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["id"]): dict(row) for row in payload["classifiers"]}


def requested_feature_names(study: dict[str, Any]) -> list[str]:
    manifest = load_feature_manifest()
    names: list[str] = []
    for feature_set_id in study.get("feature_sets", []):
        entry = manifest.get(str(feature_set_id), {})
        for feature_name in entry.get("features", []):
            if feature_name not in names:
                names.append(str(feature_name))
    return names


def requested_classifiers(study: dict[str, Any]) -> set[str]:
    return {str(value) for value in study.get("classifiers", [])}


def requested_feature_sets(study: dict[str, Any]) -> set[str]:
    return {str(value) for value in study.get("feature_sets", [])}


def requested_pairs(study: dict[str, Any]) -> set[str]:
    return {str(value) for value in study.get("class_pairs", [])}


def rung_id_for_classifier(classifier_id: str) -> str:
    mapping = {
        "pointwise": "pointwise",
        "windowed_raw_extrema": "windowed",
        "windowed_robust_extrema": "windowed",
        "windowed_shape_features": "windowed",
        "bayes_accumulator": "sequential_bayes",
        "kalman_bank": "kalman_bank",
        "transition_matrix": "transition_matrix",
        "imm": "imm",
        "particle_filter": "particle_filter",
        "rbpf": "rbpf",
    }
    return mapping.get(classifier_id, classifier_id)


def ensure_declaration_artifacts(study_path: str | Path, output_dir: str | Path) -> Path:
    study = load_study_config(study_path)
    root = ensure_workflow_dirs(output_dir, study)
    declaration_dir = root / "00_study_declaration"
    copy_file(Path(study_path), declaration_dir / "study_candidate.yaml")

    class_rows = [{"class_name": str(name), "role": "study_class", "notes": ""} for name in study.get("classes", [])]
    write_csv(declaration_dir / "class_manifest.csv", class_rows, ["class_name", "role", "notes"])

    feature_manifest = load_feature_manifest()
    feature_rows = []
    for feature_set_id in study.get("feature_sets", []):
        entry = feature_manifest[str(feature_set_id)]
        feature_rows.append(
            {
                "feature_set_id": str(feature_set_id),
                "history_behavior": str(entry.get("history_behavior", "unknown")),
                "notes": str(entry.get("description", "")),
            }
        )
    write_csv(declaration_dir / "feature_manifest.csv", feature_rows, ["feature_set_id", "history_behavior", "notes"])

    prior_rows = [{"prior_id": str(prior_id), "description": ""} for prior_id in study.get("priors", [])]
    write_csv(declaration_dir / "prior_manifest.csv", prior_rows, ["prior_id", "description"])

    write_text(
        declaration_dir / "corpus_objective.yaml",
        yaml.safe_dump(dict(study.get("corpus_objective", {})), sort_keys=False),
    )
    return declaration_dir


def filter_common_rows(
    rows: list[dict[str, str]],
    study: dict[str, Any],
    *,
    include_feature_sets: bool = True,
    include_classifiers: bool = True,
    include_pairs: bool = True,
) -> list[dict[str, str]]:
    classifier_ids = requested_classifiers(study)
    pair_ids = requested_pairs(study)
    feature_sets = requested_feature_sets(study)
    manifest = load_classifier_manifest()
    filtered: list[dict[str, str]] = []
    for row in rows:
        classifier_id = str(row.get("classifier_id", ""))
        class_pair_id = str(row.get("class_pair_id", row.get("class_pair", "")))
        feature_set_id = str(row.get("feature_set_id", manifest.get(classifier_id, {}).get("feature_set_id", manifest.get(classifier_id, {}).get("requires_feature_set", ""))))
        if include_classifiers and classifier_id and classifier_id not in classifier_ids:
            continue
        if include_pairs and class_pair_id and class_pair_id not in pair_ids:
            continue
        if include_feature_sets and feature_set_id and feature_set_id not in feature_sets:
            continue
        filtered.append(row)
    return filtered


def write_feature_redundancy_matrix(feature_matrix_path: Path, study: dict[str, Any], output_path: Path) -> None:
    rows = read_csv(feature_matrix_path)
    feature_names = [name for name in requested_feature_names(study) if rows and name in rows[0]]
    if not feature_names:
        write_csv(output_path, [], ["feature_name"])
        return
    data = array([[float(row[name]) for name in feature_names] for row in rows], dtype=float)
    matrix = corrcoef(data, rowvar=False)
    output_rows: list[dict[str, object]] = []
    for row_index, feature_name in enumerate(feature_names):
        output_row: dict[str, object] = {"feature_name": feature_name}
        for col_index, other_name in enumerate(feature_names):
            value = float(matrix[row_index, col_index]) if isfinite(matrix[row_index, col_index]) else 0.0
            output_row[other_name] = round(value, 6)
        output_rows.append(output_row)
    write_csv(output_path, output_rows, ["feature_name", *feature_names])


def derive_confusion_rows(predictions_path: Path, study: dict[str, Any], output_path: Path) -> None:
    rows = filter_common_rows(read_csv(predictions_path), study, include_feature_sets=False)
    counts: dict[tuple[str, str, str], int] = {}
    for row in rows:
        key = (str(row["classifier_id"]), str(row["true_class"]), str(row["predicted_class"]))
        counts[key] = counts.get(key, 0) + 1
    output_rows = [
        {
            "classifier_id": classifier_id,
            "true_class": true_class,
            "predicted_class": predicted_class,
            "count": count,
        }
        for (classifier_id, true_class, predicted_class), count in sorted(counts.items())
    ]
    write_csv(output_path, output_rows, ["classifier_id", "true_class", "predicted_class", "count"])


def derive_method_metrics(predictions_path: Path, study: dict[str, Any], output_path: Path) -> None:
    rows = filter_common_rows(read_csv(predictions_path), study, include_feature_sets=False)
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    manifest = load_classifier_manifest()
    for row in rows:
        classifier_id = str(row["classifier_id"])
        feature_set_id = str(row.get("feature_set_id") or manifest.get(classifier_id, {}).get("feature_set_id", manifest.get(classifier_id, {}).get("requires_feature_set", "")))
        key = (classifier_id, feature_set_id, str(row["class_pair_id"]))
        grouped.setdefault(key, []).append(row)
    output_rows: list[dict[str, object]] = []
    for (classifier_id, feature_set_id, class_pair_id), group_rows in sorted(grouped.items()):
        num_predictions = len(group_rows)
        accuracy = sum(1 for row in group_rows if row["true_class"] == row["predicted_class"]) / max(num_predictions, 1)
        mean_confidence = sum(float(row.get("confidence", 0.0)) for row in group_rows) / max(num_predictions, 1)
        output_rows.append(
            {
                "classifier_id": classifier_id,
                "rung_id": rung_id_for_classifier(classifier_id),
                "feature_set_id": feature_set_id,
                "class_pair_id": class_pair_id,
                "num_predictions": num_predictions,
                "overall_accuracy": round(accuracy, 6),
                "mean_confidence": round(mean_confidence, 6),
            }
        )
    write_csv(
        output_path,
        output_rows,
        ["classifier_id", "rung_id", "feature_set_id", "class_pair_id", "num_predictions", "overall_accuracy", "mean_confidence"],
    )


def derive_corpus_decision_gate(
    adequacy_summary_path: Path,
    class_validity_scores_path: Path,
    output_path: Path,
) -> None:
    summary = json.loads(adequacy_summary_path.read_text(encoding="utf-8"))
    score_rows = read_csv(class_validity_scores_path)
    total = len(score_rows)
    valid_count = sum(1 for row in score_rows if str(row.get("status", "")) == "valid_target_class")
    ambiguous_count = sum(1 for row in score_rows if str(row.get("status", "")) == "ambiguous")
    relabel_count = sum(1 for row in score_rows if str(row.get("status", "")) == "relabel_candidate")
    invalid_count = sum(1 for row in score_rows if str(row.get("status", "")) == "invalid")
    payload = {
        "overall_status": summary["summary"]["overall_status"],
        "overall_pass": bool(summary["summary"]["overall_pass"]),
        "recommendation_count": int(summary["summary"]["recommendation_count"]),
        "class_validity": {
            "total_rows": total,
            "valid_target_class_fraction": 0.0 if total == 0 else valid_count / total,
            "ambiguous_fraction": 0.0 if total == 0 else ambiguous_count / total,
            "relabel_candidate_fraction": 0.0 if total == 0 else relabel_count / total,
            "invalid_fraction": 0.0 if total == 0 else invalid_count / total,
        },
    }
    write_json(output_path, payload)


def derive_decision_card(
    audit_gate_path: Path,
    sufficiency_matrix_path: Path,
    confidence_summary_path: Path,
    output_path: Path,
    study: dict[str, Any],
) -> None:
    audit_gate = json.loads(audit_gate_path.read_text(encoding="utf-8"))
    confidence_summary = json.loads(confidence_summary_path.read_text(encoding="utf-8"))
    promotion_rows = read_csv(sufficiency_matrix_path)
    promote_rows = [row for row in promotion_rows if str(row.get("decision", "")) == "promote"]
    revise_rows = [row for row in promotion_rows if str(row.get("decision", "")).startswith("revise")]
    if not audit_gate["overall_pass"]:
        decision = "revise"
        rationale = "selected corpus does not pass the corpus adequacy gate"
    elif promote_rows:
        decision = "promote"
        rationale = "at least one study row justifies escalation or current-rung sufficiency with a passing corpus gate"
    elif revise_rows:
        decision = "revise"
        rationale = "the corpus passes, but the rung-sufficiency evaluator still recommends revision"
    else:
        decision = "defer"
        rationale = "no strong promotion row was found and no hard corpus failure was detected"
    lines = [
        f"# Decision Card: {study['title']}",
        "",
        f"- Study ID: `{study['study_id']}`",
        f"- Decision: `{decision}`",
        f"- Rationale: {rationale}",
        f"- Corpus gate status: `{audit_gate['overall_status']}`",
        f"- Corpus gate pass: `{audit_gate['overall_pass']}`",
        f"- Recommendation count: `{audit_gate['recommendation_count']}`",
        f"- Aggregate confidence: `{float(confidence_summary['aggregate_confidence']):.3f}`",
        f"- Best-case confidence: `{float(confidence_summary['best_case_confidence']):.3f}`",
        f"- Confidence band: `{confidence_summary['confidence_band']}`",
    ]
    selected_claim = confidence_summary.get("selected_claim")
    if isinstance(selected_claim, dict):
        lines.extend(
            [
                f"- Selected classifier claim: `{selected_claim.get('classifier_id', 'n/a')}` on `{selected_claim.get('class_pair_id', 'n/a')}` / `{selected_claim.get('feature_set_id', 'n/a')}`",
                f"- Selected claim action: `{selected_claim.get('recommended_action', 'n/a')}`",
                f"- Selected claim failure mode: `{selected_claim.get('failure_mode', 'n/a')}`",
                f"- Selected claim oracle gap: `{float(selected_claim.get('oracle_gap', 0.0)):.3f}`",
            ]
        )
    write_text(output_path, "\n".join(lines) + "\n")


def write_visual_gallery(output_dir: str | Path, study: dict[str, Any]) -> Path:
    root = workflow_root(output_dir, study)
    report_dir = root / "05_report"
    lines = [
        "# Visual Gallery",
        "",
        "## Feature/Class Analysis",
        "",
        f"- [Class Confusability Graph]({root / '01_feature_class_analysis' / 'class_confusability_graph.png'})",
        "",
        "## Corpus Generation",
        "",
        f"- [Generated Candidate Coverage]({root / '02_corpus_generation' / 'generated_candidate_coverage.png'})",
        f"- [Corpus Score Pareto]({root / '02_corpus_generation' / 'corpus_score_pareto.png'})",
        "",
        "## Corpus Audit",
        "",
        f"- [Label Status Distribution]({root / '03_corpus_audit' / 'label_status_distribution.png'})",
        "",
        "## Ladder Evaluation",
        "",
        f"- [Promotion Decision Matrix]({root / '04_ladder_evaluation' / 'promotion_decision_matrix.png'})",
        f"- [Posterior Quality By Rung]({root / '04_ladder_evaluation' / 'posterior_quality_by_rung.png'})",
        "",
        "## Confidence",
        "",
        f"- [Confidence Report]({root / '04b_confidence' / 'confidence_report.md'})",
        f"- [Confidence Dashboard]({root / '04b_confidence' / 'confidence_dashboard.png'})",
    ]
    output_path = report_dir / "visual_gallery.md"
    write_text(output_path, "\n".join(str(line) for line in lines) + "\n")
    return output_path
