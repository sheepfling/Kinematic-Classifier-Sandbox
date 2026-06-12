from __future__ import annotations

import csv
from pathlib import Path

REQUIRED_FILES: tuple[str, ...] = (
    "README.md",
    "decision_card.md",
    "static_audit_report.md",
    "static_audit_decision_card.md",
    "class_confusability_matrix.csv",
    "feature_relevance_table.csv",
    "feature_redundancy_matrix.csv",
    "feature_synergy_candidates.csv",
    "prior_pathology_report.csv",
    "prior_flip_thresholds.csv",
    "static_leakage_provenance_audit.csv",
    "figure_manifest.csv",
    "lane_proof_matrix.md",
    "hero_chart_contact_sheet.png",
    "02b_static_audit_decision_card.png",
    "02c_class_pair_confusability_matrix.png",
    "02e_feature_redundancy_graph.png",
    "02g_prior_pathology_surface.png",
)


def validate_static_admissibility_packet(packet_dir: str | Path) -> list[str]:
    base = Path(packet_dir)
    issues: list[str] = []
    for name in REQUIRED_FILES:
        if not (base / name).exists():
            issues.append(f"missing required packet file: {name}")

    if not issues:
        issues.extend(_validate_prior_regime(base / "prior_regime.csv"))
        issues.extend(_validate_class_matrix(base / "class_confusability_matrix.csv"))
        issues.extend(_validate_feature_tables(base))
        issues.extend(_validate_leakage(base / "static_leakage_provenance_audit.csv"))
        issues.extend(_validate_synergy(base / "feature_synergy_candidates.csv"))
        issues.extend(_validate_decision_consistency(base))
        issues.extend(_validate_figure_manifest(base))
    return issues


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _validate_prior_regime(path: Path) -> list[str]:
    rows = _rows(path)
    total = sum(float(row["prior_probability"]) for row in rows)
    zero_classes = [row["class_name"] for row in rows if float(row["prior_probability"]) <= 0.0]
    issues: list[str] = []
    if abs(total - 1.0) > 1.0e-6:
        issues.append(f"prior probabilities must sum to 1.0, found {total:.6f}")
    if any(float(row["prior_probability"]) < 0.0 for row in rows):
        issues.append("prior probabilities must be nonnegative")
    if zero_classes:
        issues.append(f"prior regime assigns zero mass to declared classes: {', '.join(zero_classes)}")
    return issues


def _validate_class_matrix(path: Path) -> list[str]:
    rows = _rows(path)
    classes = [row["class"] for row in rows]
    issues: list[str] = []
    for row in rows:
        for class_name in classes:
            if class_name not in row:
                issues.append(f"class matrix missing column for {class_name}")
                continue
            mirror = next(other for other in rows if other["class"] == class_name)
            left = float(row[class_name] or 0.0)
            right = float(mirror[row["class"]] or 0.0)
            if abs(left - right) > 1.0e-9:
                issues.append(f"class matrix is not symmetric for {row['class']} and {class_name}")
    return issues


def _validate_feature_tables(base: Path) -> list[str]:
    relevance = _rows(base / "feature_relevance_table.csv")
    redundancy = _rows(base / "feature_redundancy_matrix.csv")
    features = {row["feature"] for row in relevance}
    mentioned = set(features)
    for row in redundancy:
        mentioned.add(row["feature_a"])
        mentioned.add(row["feature_b"])
    missing = sorted(features - mentioned)
    return [f"declared features missing from redundancy output: {', '.join(missing)}"] if missing else []


def _validate_leakage(path: Path) -> list[str]:
    rows = _rows(path)
    required = {"feature", "provenance_tags", "online_available", "label_rule_overlap_flag", "status"}
    issues = []
    for row in rows:
        missing = required - set(row)
        if missing:
            issues.append(f"leakage row missing fields: {', '.join(sorted(missing))}")
        if row.get("online_available", "") == "":
            issues.append(f"leakage row missing online availability for {row.get('feature', '<unknown>')}")
    return issues


def _validate_synergy(path: Path) -> list[str]:
    rows = _rows(path)
    bad = [row for row in rows if row.get("status") not in {"ordinary", "synergy_candidate"}]
    if bad:
        return ["synergy must remain ordinary or synergy_candidate unless ablation-backed"]
    return []


def _validate_decision_consistency(base: Path) -> list[str]:
    decision_text = (base / "decision_card.md").read_text(encoding="utf-8")
    leakage_rows = _rows(base / "static_leakage_provenance_audit.csv")
    has_blocker = any(row["status"] == "blocker" for row in leakage_rows)
    if has_blocker and "promote_to_corpus_explorer" in decision_text:
        return ["static blocker cannot produce promote_to_corpus_explorer"]
    return []


def _validate_figure_manifest(base: Path) -> list[str]:
    rows = _rows(base / "figure_manifest.csv")
    issues = []
    for row in rows:
        figure = base / row["figure_id"]
        source = base / row["source_table"]
        if not figure.exists():
            issues.append(f"manifest figure missing: {row['figure_id']}")
        if not source.exists():
            issues.append(f"manifest source missing: {row['source_table']}")
        if row["figure_id"] == "02f_feature_synergy_map.png" and "candidate" not in row["claim_boundary"]:
            issues.append("synergy figure must keep candidate claim boundary")
    return issues
