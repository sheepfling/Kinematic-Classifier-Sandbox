from __future__ import annotations

import csv
import json
from pathlib import Path


REQUIRED_FILES: tuple[str, ...] = (
    "README.md",
    "corpus_explorer_decision_card.md",
    "corpus_objective.yaml",
    "selected_corpus_manifest.csv",
    "corpus_candidate_frontier.csv",
    "corpus_adequacy_report.md",
    "leakage_adequacy_audit.csv",
    "feature_excitation_report.csv",
    "search_backend_comparison.csv",
    "downstream_diagnostic_yield.csv",
    "novelty_to_filter_escalation_report.md",
    "advanced_algorithm_route_matrix.csv",
    "advanced_algorithm_route_proof.md",
    "packet_manifest.json",
)

REQUIRED_FIGURES: tuple[str, ...] = (
    "figures/03_corpus_candidate_frontier.png",
    "figures/18_leakage_adequacy_audit.png",
    "figures/21_search_backend_comparison_frontier.png",
    "figures/26_downstream_diagnostic_yield.png",
    "figures/27_novelty_to_filter_escalation_bridge.png",
)

REQUIRED_ROUTE_METHODS: dict[str, str] = {
    "imm_v1": "trace_validated",
    "particle_filter_bank_v1": "trace_validated",
    "rbpf_v1": "trace_validated",
    "ornstein_uhlenbeck_pf_v1": "trace_validated",
    "ts2vec": "witness_supported",
}


def validate_corpus_explorer_packet(packet_dir: str | Path) -> list[str]:
    base = Path(packet_dir)
    issues: list[str] = []
    for name in REQUIRED_FILES:
        if not (base / name).exists():
            issues.append(f"missing required packet file: {name}")
    for name in REQUIRED_FIGURES:
        if not (base / name).exists():
            issues.append(f"missing required figure: {name}")
    if issues:
        return issues

    issues.extend(_validate_manifest(base / "packet_manifest.json"))
    frontier_rows = _rows(base / "corpus_candidate_frontier.csv")
    selected_rows = _rows(base / "selected_corpus_manifest.csv")
    leakage_rows = _rows(base / "leakage_adequacy_audit.csv")
    feature_rows = _rows(base / "feature_excitation_report.csv")
    backend_rows = _rows(base / "search_backend_comparison.csv")
    yield_rows = _rows(base / "downstream_diagnostic_yield.csv")
    route_rows = _rows(base / "advanced_algorithm_route_matrix.csv")

    issues.extend(_validate_candidate_tables(frontier_rows, selected_rows))
    issues.extend(_validate_leakage(frontier_rows, leakage_rows))
    issues.extend(_validate_feature_report(frontier_rows, feature_rows))
    issues.extend(_validate_backend_comparison(backend_rows))
    issues.extend(_validate_downstream_yield(yield_rows))
    issues.extend(_validate_route_matrix(route_rows))
    issues.extend(_validate_hard_case_cards(base, frontier_rows))
    issues.extend(_validate_public_text(base))
    return issues


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _validate_manifest(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    issues: list[str] = []
    if payload.get("packet_id") != "corpus_explorer_mvp":
        issues.append("packet_manifest.json must declare packet_id corpus_explorer_mvp")
    if payload.get("plan_path") != "docs/plans/PLN-036_corpus_explorer_execution_brief.md":
        issues.append("packet_manifest.json must point at PLN-036")
    goal = payload.get("short_goal_blurb", "")
    if "corpus decision system" not in goal.lower():
        issues.append("packet_manifest.json must carry the short goal blurb")
    validator = payload.get("validator", "")
    if "validate-packet" not in validator or "corpus_explorer_mvp" not in validator:
        issues.append("packet_manifest.json must include the corpus packet validator command")
    return issues


def _validate_candidate_tables(
    frontier_rows: list[dict[str, str]],
    selected_rows: list[dict[str, str]],
) -> list[str]:
    issues: list[str] = []
    if not frontier_rows:
        return ["corpus_candidate_frontier.csv must include candidate rows"]
    selected = [row for row in frontier_rows if row.get("selected") == "True"]
    rejected = [row for row in frontier_rows if row.get("selected") != "True"]
    if not selected:
        issues.append("corpus_candidate_frontier.csv must include selected candidates")
    if not rejected:
        issues.append("corpus_candidate_frontier.csv must include rejected candidates")

    selected_ids = {row["candidate_id"] for row in selected}
    manifest_ids = {row["candidate_id"] for row in selected_rows}
    if selected_ids != manifest_ids:
        issues.append("selected_corpus_manifest.csv must match selected candidate ids")

    for row in selected_rows:
        if row.get("validity_status") != "pass":
            issues.append(f"selected candidate {row.get('candidate_id')} must pass validity")
        if row.get("leakage_status") != "pass":
            issues.append(f"selected candidate {row.get('candidate_id')} must pass leakage")
        if row.get("rejection_reason"):
            issues.append(f"selected candidate {row.get('candidate_id')} must not have a rejection reason")
        if not row.get("target_failure_mode"):
            issues.append(f"selected candidate {row.get('candidate_id')} must name a target failure mode")
        if not row.get("routed_action"):
            issues.append(f"selected candidate {row.get('candidate_id')} must include a routed action")
        if "reject" in row.get("routed_action", "").lower():
            issues.append(f"selected candidate {row.get('candidate_id')} cannot route to rejection")

    for row in rejected:
        if not row.get("rejection_reason"):
            issues.append(f"rejected candidate {row.get('candidate_id')} must include a rejection reason")
    return issues


def _validate_leakage(
    frontier_rows: list[dict[str, str]],
    leakage_rows: list[dict[str, str]],
) -> list[str]:
    issues: list[str] = []
    leakage_by_id = {row["candidate_id"]: row for row in leakage_rows}
    for row in frontier_rows:
        candidate_id = row["candidate_id"]
        leakage = leakage_by_id.get(candidate_id)
        if leakage is None:
            issues.append(f"leakage_adequacy_audit.csv missing candidate {candidate_id}")
            continue
        if row.get("selected") == "True":
            if leakage.get("class_validity") != "pass":
                issues.append(f"selected candidate {candidate_id} must pass class validity in leakage audit")
            if leakage.get("feature_availability") != "pass":
                issues.append(f"selected candidate {candidate_id} must pass feature availability")
            if leakage.get("leakage_status") != "pass":
                issues.append(f"selected candidate {candidate_id} must pass leakage audit")
        elif not leakage.get("rejection_reason"):
            issues.append(f"rejected candidate {candidate_id} must carry rejection reason in leakage audit")
    return issues


def _validate_feature_report(
    frontier_rows: list[dict[str, str]],
    feature_rows: list[dict[str, str]],
) -> list[str]:
    issues: list[str] = []
    feature_by_id = {row["candidate_id"]: row for row in feature_rows}
    for row in frontier_rows:
        candidate_id = row["candidate_id"]
        feature = feature_by_id.get(candidate_id)
        if feature is None:
            issues.append(f"feature_excitation_report.csv missing candidate {candidate_id}")
            continue
        expected = "usable" if row.get("selected") == "True" else "rejected"
        if feature.get("status") != expected:
            issues.append(f"feature report status mismatch for {candidate_id}")
    return issues


def _validate_backend_comparison(rows: list[dict[str, str]]) -> list[str]:
    issues: list[str] = []
    backends = {row.get("backend_id", ""): row for row in rows}
    for backend in ("doe_schedule_bank", "guided_schedule_mutation", "cem_open_loop", "ppo_policy"):
        if backend not in backends:
            issues.append(f"search_backend_comparison.csv missing backend {backend}")
    for backend in ("cem_open_loop", "ppo_policy"):
        row = backends.get(backend)
        if row is None:
            continue
        gate = row.get("promotion_gate", "")
        if "not_promoted" not in gate:
            issues.append(f"{backend} must retain the not-promoted gate")
        if row.get("status") == "promoted":
            issues.append(f"{backend} must not be marked promoted")
        if not row.get("justification"):
            issues.append(f"{backend} must include justification")
    return issues


def _validate_downstream_yield(rows: list[dict[str, str]]) -> list[str]:
    issues: list[str] = []
    triggers = {row.get("decision_trigger", "") for row in rows}
    required = {
        "switching_state_failure -> IMM witness",
        "nonlinear_posterior_candidate -> PF/GSF witness",
        "maneuver ambiguity -> RBPF latent-event witness",
    }
    missing = sorted(required - triggers)
    if missing:
        issues.append(f"downstream_diagnostic_yield.csv missing triggers: {', '.join(missing)}")
    if not any("rejected before ladder influence" in trigger for trigger in triggers):
        issues.append("downstream_diagnostic_yield.csv must include rejected invalid-case evidence")
    return issues


def _validate_route_matrix(rows: list[dict[str, str]]) -> list[str]:
    issues: list[str] = []
    rows_by_method = {row.get("method_id", ""): row for row in rows}
    for method_id, expected_trace in REQUIRED_ROUTE_METHODS.items():
        row = rows_by_method.get(method_id)
        if row is None:
            issues.append(f"advanced_algorithm_route_matrix.csv missing route for {method_id}")
            continue
        if row.get("trace_status") != expected_trace:
            issues.append(f"{method_id} must have trace_status {expected_trace}")
    ts2vec = rows_by_method.get("ts2vec")
    if ts2vec is not None:
        if "embedding_baseline_frontier" not in ts2vec.get("supporting_artifact", ""):
            issues.append("ts2vec route must cite the embedding frontier artifact")
        claim_boundary = ts2vec.get("claim_boundary", "").lower()
        if "proxy witness only" not in claim_boundary and "bounded parity witness" not in claim_boundary:
            issues.append("ts2vec route must preserve a bounded proxy/parity claim boundary")
    return issues


def _validate_hard_case_cards(base: Path, frontier_rows: list[dict[str, str]]) -> list[str]:
    issues: list[str] = []
    for row in frontier_rows:
        card_name = f"{row['candidate_id']}_{row['target_failure_mode']}.md"
        card_path = base / "hard_case_cards" / card_name
        if not card_path.exists():
            issues.append(f"missing hard-case card: {card_name}")
            continue
        text = card_path.read_text(encoding="utf-8")
        if row["routed_action"] not in text:
            issues.append(f"hard-case card {card_name} must include routed action")
        if row.get("selected") != "True" and row.get("rejection_reason", "") not in text:
            issues.append(f"rejected card {card_name} must include rejection reason")
    return issues


def _validate_public_text(base: Path) -> list[str]:
    issues: list[str] = []
    readme_text = (base / "README.md").read_text(encoding="utf-8").lower()
    if "decision endpoint" not in readme_text:
        issues.append("README.md must describe the decision-card endpoint")
    route_text = (base / "advanced_algorithm_route_proof.md").read_text(encoding="utf-8").lower()
    if "not a universal default" not in route_text:
        issues.append("advanced_algorithm_route_proof.md must preserve non-universal claim boundary")
    decision_text = (base / "corpus_explorer_decision_card.md").read_text(encoding="utf-8").lower()
    for token in (
        "status: promote_selected_corpus",
        "experimental_or_run_backed",
        "experimental_witness",
        "future_3d_lift",
    ):
        if token not in decision_text:
            issues.append(f"corpus_explorer_decision_card.md missing token: {token}")
    return issues
