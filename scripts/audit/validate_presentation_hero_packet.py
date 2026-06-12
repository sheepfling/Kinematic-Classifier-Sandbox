from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from _bootstrap import bootstrap_repo

REQUIRED_TIERS = {"RUN-BACKED", "ARTIFACT-BACKED", "EXPERIMENTAL-WITNESS", "CANDIDATE-DIAGNOSTIC", "ROADMAP"}
PRIVATE_PATH_RE = re.compile(r"/Users/[^\\s)]+")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _source_parts(source_text: str) -> list[str]:
    return [part.strip() for part in source_text.split(";") if part.strip()]


def _public_text_files(packet_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in packet_dir.rglob("*")
        if path.is_file()
        and "deck_workspaces" not in path.parts
        and path.name != "artifact-build-manifest.json"
        and path.suffix.lower() in {".md", ".csv", ".json", ".yaml", ".yml", ".txt"}
    )


def validate_packet(packet_dir: Path) -> list[str]:
    issues: list[str] = []
    root = packet_dir.parents[1]
    manifest_path = packet_dir / "hero_chart_manifest.csv"
    decision_card_path = packet_dir / "decision_card.md"
    lane_matrix_path = packet_dir / "lane_proof_matrix.csv"
    slide_script_path = packet_dir / "slide_speaker_script.csv"

    for required in [manifest_path, decision_card_path, lane_matrix_path, slide_script_path]:
        if not required.exists():
            issues.append(f"missing required packet file: {required}")

    if issues:
        return issues

    manifest_rows = _read_csv(manifest_path)
    required_manifest_columns = {"chart_id", "role", "path", "evidence_tier", "source_artifact", "claim_boundary"}
    if not manifest_rows:
        issues.append("hero_chart_manifest.csv is empty")
    else:
        missing_columns = required_manifest_columns.difference(manifest_rows[0])
        if missing_columns:
            issues.append(f"hero_chart_manifest.csv missing columns: {sorted(missing_columns)}")

    for row in manifest_rows:
        chart_id = row.get("chart_id", "")
        evidence_tier = row.get("evidence_tier", "")
        if evidence_tier not in REQUIRED_TIERS:
            issues.append(f"{chart_id}: invalid or missing evidence_tier={evidence_tier!r}")
        path_text = row.get("path", "")
        chart_path = Path(path_text)
        if not chart_path.is_absolute():
            chart_path = packet_dir.parents[1] / path_text
        png_companion = chart_path.with_suffix(".png") if chart_path.suffix.lower() == ".svg" else chart_path
        if not chart_path.exists() and not png_companion.exists():
            issues.append(f"{chart_id}: missing chart file {chart_path}")
        if not row.get("source_artifact"):
            issues.append(f"{chart_id}: source_artifact is empty")
        sources = _source_parts(row.get("source_artifact", ""))
        has_run_source = any(source.endswith((".csv", ".json")) for source in sources)
        if evidence_tier == "RUN-BACKED" and not has_run_source:
            issues.append(f"{chart_id}: RUN-BACKED charts must cite a CSV or JSON source artifact")
        for source in sources:
            if "*" in source or not source.endswith((".csv", ".json", ".md")):
                continue
            source_path = root / source
            if not source_path.exists():
                issues.append(f"{chart_id}: missing cited source artifact {source}")
        if not row.get("claim_boundary"):
            issues.append(f"{chart_id}: claim_boundary is empty")

    lane_rows = _read_csv(lane_matrix_path)
    required_lane_columns = {
        "epic",
        "lane",
        "claim",
        "hero_chart",
        "backing_artifact",
        "evidence_tier",
        "validation_check",
        "decision_card_field",
        "limitation",
        "next_work",
        "status",
    }
    if not lane_rows:
        issues.append("lane_proof_matrix.csv is empty")
    else:
        missing_columns = required_lane_columns.difference(lane_rows[0])
        if missing_columns:
            issues.append(f"lane_proof_matrix.csv missing columns: {sorted(missing_columns)}")

    decision_text = decision_card_path.read_text(encoding="utf-8")
    for token in [
        "study_id",
        "run_id",
        "seed",
        "corpus_search_backend_decision",
        "novelty_search_claim_status",
        "static_audit_decision",
        "epic_1_static_admissibility",
        "epic_2_classifier_filter_ladder",
        "epic_3_corpus_evaluation_advanced_exploration",
        "overall_decision",
        "advanced-filter decision",
        "advanced_filter_decisions",
        "final decision",
    ]:
        if token not in decision_text:
            issues.append(f"decision_card.md missing token: {token}")
    for token in ["trace_validated", "witness_supported", "study_justified"]:
        if token not in decision_text:
            issues.append(f"decision_card.md must name advanced-filter status layer: {token}")
    if "witness-specific" not in decision_text.lower():
        issues.append("decision_card.md must explicitly preserve witness-specific advanced-filter scope")
    if "ppo remains experimental" not in decision_text.lower():
        issues.append("decision_card.md must explicitly say PPO remains experimental")
    if "not promoted as a general novelty backend" not in decision_text.lower():
        issues.append("decision_card.md must preserve the general-backend caveat for PPO")

    manifest_by_id = {row["chart_id"]: row for row in manifest_rows}
    if manifest_by_id.get("24_ppo_boundary_shaping_trace", {}).get("evidence_tier") != "EXPERIMENTAL-WITNESS":
        issues.append("24_ppo_boundary_shaping_trace must be EXPERIMENTAL-WITNESS")
    summary_row = manifest_by_id.get("10f_advanced_filter_showcase_summary")
    if summary_row is None:
        issues.append("missing advanced-filter status summary chart in manifest: 10f_advanced_filter_showcase_summary")
    else:
        boundary = summary_row.get("claim_boundary", "")
        source_artifact = summary_row.get("source_artifact", "")
        if "trace_validated" not in boundary or "study_justified" not in boundary:
            issues.append("10f_advanced_filter_showcase_summary must declare the status-layer split in claim_boundary")
        if "filter_trace_validation_v1" not in source_artifact:
            issues.append("10f_advanced_filter_showcase_summary must reference the trace-validation packet")

    for chart_id in {"10b_imm_switching_shine_witness", "10c_pf_nonlinear_nongaussian_shine_witness", "10d_rbpf_latent_event_shine_witness"}:
        row = manifest_by_id.get(chart_id)
        if row is None:
            issues.append(f"missing advanced-filter witness chart in manifest: {chart_id}")
            continue
        boundary = row.get("claim_boundary", "").lower()
        source_artifact = row.get("source_artifact", "").lower()
        if "witness" not in boundary:
            issues.append(f"{chart_id} must declare witness scope in claim_boundary")
        if "baseline" not in source_artifact and "comparison" not in source_artifact:
            issues.append(f"{chart_id} must reference a simpler-rung baseline comparison artifact")

    for chart_id in {"21_search_backend_comparison_frontier", "26_downstream_diagnostic_yield"}:
        if chart_id not in manifest_by_id:
            issues.append(f"missing novelty-search proof chart: {chart_id}")

    for chart_id in {
        "02b_static_audit_decision_card",
        "02c_class_pair_confusability_matrix",
        "02g_prior_pathology_surface",
    }:
        if chart_id not in manifest_by_id:
            issues.append(f"missing static-audit main proof chart: {chart_id}")

    synergy_row = manifest_by_id.get("02f_feature_synergy_map")
    if synergy_row is None:
        issues.append("missing static-audit synergy chart: 02f_feature_synergy_map")
    elif (
        synergy_row.get("evidence_tier") != "CANDIDATE-DIAGNOSTIC"
        and "candidate" not in synergy_row.get("claim_boundary", "").lower()
    ):
        issues.append("02f_feature_synergy_map must remain candidate evidence unless confirmed by ablation")

    static_dir = root / "artifacts/static_feature_class_prior_audit_v1"
    prior_rows = _read_csv(static_dir / "prior_regime.csv") if (static_dir / "prior_regime.csv").exists() else []
    if not prior_rows:
        issues.append("static audit missing prior_regime.csv")
    else:
        prior_total = sum(float(row["prior_probability"]) for row in prior_rows)
        if abs(prior_total - 1.0) > 1.0e-6:
            issues.append(f"static prior regime must sum to one, got {prior_total:.6f}")

    matrix_path = static_dir / "class_confusability_matrix.csv"
    if matrix_path.exists():
        matrix_rows = _read_csv(matrix_path)
        classes = [column for column in matrix_rows[0] if column != "class"] if matrix_rows else []
        row_classes = [row["class"] for row in matrix_rows]
        if set(classes) != set(row_classes):
            issues.append("class_confusability_matrix.csv must cover the same row and column classes")
        value_by_pair = {
            (row["class"], column): float(row[column])
            for row in matrix_rows
            for column in classes
            if row.get(column, "") != ""
        }
        for left in classes:
            for right in classes:
                if (left, right) not in value_by_pair:
                    issues.append(f"class_confusability_matrix.csv missing pair {left}/{right}")
                    continue
                if abs(value_by_pair[(left, right)] - value_by_pair.get((right, left), -999.0)) > 1.0e-6:
                    issues.append(f"class_confusability_matrix.csv not symmetric for {left}/{right}")
    else:
        issues.append("static audit missing class_confusability_matrix.csv")

    relevance_path = static_dir / "feature_relevance_table.csv"
    redundancy_path = static_dir / "feature_redundancy_matrix.csv"
    if relevance_path.exists() and redundancy_path.exists():
        relevance_rows = _read_csv(relevance_path)
        redundancy_rows = _read_csv(redundancy_path)
        features = {row["feature"] for row in relevance_rows}
        seen_pairs = {
            tuple(sorted((row["feature_a"], row["feature_b"])))
            for row in redundancy_rows
            if row.get("feature_a") and row.get("feature_b")
        }
        expected_pair_count = len(features) * (len(features) - 1) // 2
        if len(seen_pairs) != expected_pair_count:
            issues.append(
                "feature_redundancy_matrix.csv must include every declared feature pair "
                f"({len(seen_pairs)} of {expected_pair_count})"
            )
    else:
        issues.append("static audit missing feature relevance or redundancy CSV")

    leakage_path = static_dir / "static_leakage_provenance_audit.csv"
    if leakage_path.exists():
        leakage_rows = _read_csv(leakage_path)
        required_leakage_columns = {
            "feature",
            "provenance_tags",
            "online_available",
            "future_dependency_flag",
            "label_rule_overlap_flag",
            "status",
        }
        if leakage_rows and required_leakage_columns.difference(leakage_rows[0]):
            issues.append("static_leakage_provenance_audit.csv missing provenance or availability columns")
        if any(not row.get("provenance_tags") for row in leakage_rows):
            issues.append("static_leakage_provenance_audit.csv must include provenance for every feature")
    else:
        issues.append("static audit missing static_leakage_provenance_audit.csv")

    for rel_path in [
        "artifacts/packets/static_admissibility_mvp/README.md",
        "artifacts/packets/classifier_ladder_mvp/README.md",
        "artifacts/packets/classifier_ladder_mvp/evidence_capability_ladder.csv",
        "artifacts/packets/classifier_ladder_mvp/method_capability_matrix.csv",
        "artifacts/packets/classifier_ladder_mvp/filter_promotion_criteria.csv",
        "artifacts/packets/classifier_ladder_mvp/advanced_inference_architecture_map.csv",
        "artifacts/packets/classifier_ladder_mvp/figures/06c_capability_ladder.png",
        "artifacts/packets/classifier_ladder_mvp/figures/10g_method_capability_matrix.png",
        "artifacts/packets/classifier_ladder_mvp/figures/10h_advanced_inference_architecture_map.png",
        "artifacts/packets/classifier_ladder_mvp/figures/10i_filter_promotion_criteria.png",
        "artifacts/packets/advanced_algorithm_showcase/README.md",
        "artifacts/packets/advanced_algorithm_showcase/advanced_algorithm_decision_card.md",
        "artifacts/packets/advanced_algorithm_showcase/method_capability_matrix.md",
        "artifacts/packets/advanced_algorithm_showcase/full_ladder_metrics.csv",
        "artifacts/packets/advanced_algorithm_showcase/method_win_by_regime.csv",
        "artifacts/packets/advanced_algorithm_showcase/source_manifest.csv",
        "artifacts/packets/advanced_algorithm_showcase/imm_switching_witness.md",
        "artifacts/packets/advanced_algorithm_showcase/pf_nonlinear_witness.md",
        "artifacts/packets/advanced_algorithm_showcase/rbpf_latent_event_witness.md",
        "artifacts/packets/advanced_algorithm_showcase/cem_hard_case_search.md",
        "artifacts/packets/advanced_algorithm_showcase/ppo_boundary_shaping.md",
        "artifacts/packets/advanced_algorithm_showcase/figures/10f_method_win_by_regime_map.png",
        "artifacts/packets/corpus_exploration_mvp/README.md",
        "artifacts/packets/corpus_explorer_mvp/README.md",
        "artifacts/packets/corpus_explorer_mvp/corpus_explorer_decision_card.md",
        "artifacts/packets/corpus_explorer_mvp/advanced_algorithm_route_matrix.csv",
        "artifacts/packets/corpus_explorer_mvp/advanced_algorithm_route_proof.md",
        "artifacts/packets/anduril_c2_blend/README.md",
        "artifacts/packets/anduril_c2_blend/decision_card.md",
        "artifacts/packets/anduril_c2_blend/lane_proof_matrix.md",
        "artifacts/packets/anduril_c2_blend/claim_boundary.md",
    ]:
        if not (root / rel_path).exists():
            issues.append(f"missing three-epic packet artifact: {rel_path}")

    deck_manifest_path = packet_dir / "deck_manifest.json"
    if deck_manifest_path.exists():
        deck_manifest = json.loads(deck_manifest_path.read_text(encoding="utf-8"))
        main_chart_ids = set(deck_manifest.get("main_chart_ids", []))
        for required_chart in {"06c_capability_ladder", "10h_advanced_inference_architecture_map"}:
            if required_chart not in main_chart_ids:
                issues.append(f"main deck must include advanced architecture chart: {required_chart}")

    capability_path = root / "artifacts/packets/classifier_ladder_mvp/filter_promotion_criteria.csv"
    if capability_path.exists():
        capability_rows = _read_csv(capability_path)
        required_methods = {"imm_v1", "particle_filter_bank_v1", "rbpf_v1"}
        methods = {row.get("method_id", "") for row in capability_rows}
        if not required_methods.issubset(methods):
            issues.append("classifier_ladder_mvp filter_promotion_criteria.csv missing advanced method rows")
        for row in capability_rows:
            if row.get("method_id") in required_methods:
                if row.get("architecturally_exercised") not in {"witness_supported", "prototype_plus_witness"}:
                    issues.append(f"{row.get('method_id')} must be architecturally exercised")
                if row.get("trace_status") != "trace_validated":
                    issues.append(f"{row.get('method_id')} must be trace_validated")

    route_path = root / "artifacts/packets/corpus_explorer_mvp/advanced_algorithm_route_matrix.csv"
    if route_path.exists():
        route_rows = _read_csv(route_path)
        required_routes = {"imm_v1", "particle_filter_bank_v1", "rbpf_v1", "ornstein_uhlenbeck_pf_v1", "ts2vec"}
        route_methods = {row.get("method_id", "") for row in route_rows}
        if not required_routes.issubset(route_methods):
            issues.append("corpus_explorer_mvp advanced route matrix missing required advanced route")
        expected_status = {
            "imm_v1": "trace_validated",
            "particle_filter_bank_v1": "trace_validated",
            "rbpf_v1": "trace_validated",
            "ornstein_uhlenbeck_pf_v1": "trace_validated",
            "ts2vec": "witness_supported",
        }
        for row in route_rows:
            method_id = row.get("method_id", "")
            if method_id in expected_status and row.get("trace_status") != expected_status[method_id]:
                issues.append(f"corpus_explorer_mvp route {method_id} must be {expected_status[method_id]}")
        ts2vec_rows = [row for row in route_rows if row.get("method_id") == "ts2vec"]
        for row in ts2vec_rows:
            if "embedding_baseline_frontier" not in row.get("supporting_artifact", ""):
                issues.append("corpus_explorer_mvp ts2vec route must cite the embedding frontier artifact")

    showcase_decision_path = root / "artifacts/packets/advanced_algorithm_showcase/advanced_algorithm_decision_card.md"
    if showcase_decision_path.exists():
        showcase_text = showcase_decision_path.read_text(encoding="utf-8").lower()
        for token in [
            "claim_a_main_toy_need: not_claimed",
            "claim_b_advanced_method_exercise: passed",
            "a method can pass a shine witness without being generally best",
            "cem/ppo require baseline comparison",
        ]:
            if token not in showcase_text:
                issues.append(f"advanced_algorithm_showcase decision card missing token: {token}")

    showcase_win_path = root / "artifacts/packets/advanced_algorithm_showcase/method_win_by_regime.csv"
    if showcase_win_path.exists():
        showcase_rows = _read_csv(showcase_win_path)
        required_showcase_methods = {"imm_v1", "particle_filter_bank_v1", "rbpf_v1", "cem_open_loop", "ppo_policy"}
        showcase_methods = {row.get("method_id", "") for row in showcase_rows}
        if not required_showcase_methods.issubset(showcase_methods):
            issues.append("advanced_algorithm_showcase method_win_by_regime.csv missing required advanced methods")
        for row in showcase_rows:
            method_id = row.get("method_id", "")
            boundary = row.get("claim_boundary", "").lower()
            if method_id in {"imm_v1", "particle_filter_bank_v1", "rbpf_v1"} and "not a universal" not in boundary:
                issues.append(f"{method_id} must preserve witness-specific, non-universal scope")
            if method_id in {"cem_open_loop", "ppo_policy"} and "baseline comparison" not in boundary:
                issues.append(f"{method_id} must preserve search-backend promotion caveats")

    epic_values = {row.get("epic", "") for row in lane_rows}
    for required_epic in {
        "Epic 1: Static Admissibility",
        "Epic 2: Classifier / Filter Ladder",
        "Epic 3: Corpus Evaluation and Advanced Exploration",
    }:
        if required_epic not in epic_values:
            issues.append(f"lane_proof_matrix.csv missing epic coverage: {required_epic}")

    for text_path in _public_text_files(packet_dir):
        text = text_path.read_text(encoding="utf-8")
        lower = text.lower()
        if "ppo promoted" in lower or "cem promoted" in lower:
            issues.append(f"public packet overclaims CEM/PPO promotion: {text_path}")
        if "pf and rbpf are not promoted" in lower:
            issues.append(f"public packet is stale; PF/RBPF now need witness-specific wording instead: {text_path}")

    for text_path in _public_text_files(packet_dir):
        text = text_path.read_text(encoding="utf-8")
        if PRIVATE_PATH_RE.search(text):
            issues.append(f"private local path leaked into public packet text: {text_path}")
        if "generally proven" in text and "not promoted" not in text:
            issues.append(f"possible overclaim without not-promoted caveat: {text_path}")

    return issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-dir", default="artifacts/presentation_hero_charts_v5")
    args = parser.parse_args()
    root = bootstrap_repo()
    packet_dir = (root / args.packet_dir).resolve()
    issues = validate_packet(packet_dir)
    if issues:
        for issue in issues:
            print(f"FAIL: {issue}")
        return 1
    print(f"PASS: {packet_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
