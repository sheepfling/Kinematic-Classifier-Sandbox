from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from _bootstrap import bootstrap_repo


REQUIRED_TIERS = {"RUN-BACKED", "ARTIFACT-BACKED", "EXPERIMENTAL-WITNESS", "CANDIDATE-DIAGNOSTIC", "ROADMAP"}
PRIVATE_PATH_RE = re.compile(r"/Users/[^\\s)]+")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
        if not chart_path.exists():
            issues.append(f"{chart_id}: missing chart file {chart_path}")
        if not row.get("source_artifact"):
            issues.append(f"{chart_id}: source_artifact is empty")
        if not row.get("claim_boundary"):
            issues.append(f"{chart_id}: claim_boundary is empty")

    lane_rows = _read_csv(lane_matrix_path)
    required_lane_columns = {
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
        "advanced-filter decision",
        "advanced_filter_decisions",
        "final decision",
    ]:
        if token not in decision_text:
            issues.append(f"decision_card.md missing token: {token}")
    if "witness-specific" not in decision_text.lower():
        issues.append("decision_card.md must explicitly preserve witness-specific advanced-filter scope")
    if "ppo remains experimental" not in decision_text.lower():
        issues.append("decision_card.md must explicitly say PPO remains experimental")
    if "not promoted as a general novelty backend" not in decision_text.lower():
        issues.append("decision_card.md must preserve the general-backend caveat for PPO")

    manifest_by_id = {row["chart_id"]: row for row in manifest_rows}
    if manifest_by_id.get("24_ppo_boundary_shaping_trace", {}).get("evidence_tier") != "EXPERIMENTAL-WITNESS":
        issues.append("24_ppo_boundary_shaping_trace must be EXPERIMENTAL-WITNESS")

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
    parser.add_argument("--packet-dir", default="artifacts/presentation_hero_charts_v4")
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
