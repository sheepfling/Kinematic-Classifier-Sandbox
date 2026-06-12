from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from kinematic_classifier_sandbox.corpus.validation import validate_corpus_explorer_packet
from kinematic_classifier_sandbox.reports.markdown import MarkdownDocument
from kinematic_classifier_sandbox.static_admissibility.exemplar_suite import (
    write_static_admissibility_exemplar_suite_packet,
)
from kinematic_classifier_sandbox.static_admissibility.validation import (
    validate_static_admissibility_packet,
)
from kinematic_classifier_sandbox.utils.io import _write_json, _write_text, read_csv_rows, write_csv
from kinematic_classifier_sandbox.utils.runtime import repo_root


@dataclass(frozen=True, slots=True)
class V7AndurilC2BlendArtifacts:
    packet_dir: Path
    manifest_path: Path
    decision_card_path: Path
    claim_boundary_path: Path
    epic_summary_path: Path
    hero_chart_manifest_path: Path
    lane_proof_matrix_path: Path
    validation_report_path: Path
    main_deck_path: Path
    appendix_deck_path: Path
    whitepaper_main_path: Path


ROOT = repo_root()
SOURCE_HERO_PACKET = ROOT / "artifacts" / "presentation_hero_charts_v5"
SOURCE_E2_PACKET = ROOT / "artifacts" / "packets" / "classifier_ladder_mvp"
SOURCE_E3_PACKET = ROOT / "artifacts" / "packets" / "corpus_explorer_mvp"
SOURCE_SHOWCASE_TABLES = ROOT / "artifacts" / "showcase" / "tables"
SOURCE_SHOWCASE_PLOTS = ROOT / "artifacts" / "showcase" / "plots"
SOURCE_CEILING = ROOT / "artifacts" / "classifier_family_scorecard_v1" / "ceiling_efficiency.csv"
SOURCE_ROCKET = ROOT / "artifacts" / "packets" / "advanced_algorithm_showcase"

INTEGRATED_MAIN_CHARTS: tuple[dict[str, str], ...] = (
    {
        "chart_id": "02m_static_exemplar_decision_surface",
        "epic": "Epic 1",
        "role": "main",
        "filename": "02m_static_exemplar_decision_surface.png",
        "source_artifact": "epic_packets/01_static_admissibility_gate/source_artifacts/exemplar_route_matrix.csv",
        "evidence_tier": "RUN-BACKED",
        "validation_check": "six file-backed exemplar routes match expected status",
        "limitation": "curated exemplar suite; not an exhaustive study universe",
        "next_action": "add more bundle families only when they change routing behavior",
    },
    {
        "chart_id": "02c_class_pair_confusability_matrix",
        "epic": "Epic 1",
        "role": "main",
        "filename": "02c_class_pair_confusability_matrix.png",
        "source_artifact": "artifacts/static_feature_class_prior_audit_v1/class_confusability_matrix.csv",
        "evidence_tier": "RUN-BACKED",
        "validation_check": "pairwise matrix symmetric and complete",
        "limitation": "feature-surface confusability, not final dynamic classifier confusion",
        "next_action": "route hard pairs into corpus and ladder witnesses",
    },
    {
        "chart_id": "02g_prior_pathology_surface",
        "epic": "Epic 1",
        "role": "main",
        "filename": "02g_prior_pathology_surface.png",
        "source_artifact": "artifacts/static_feature_class_prior_audit_v1/prior_pathology_report.csv",
        "evidence_tier": "RUN-BACKED",
        "validation_check": "prior regime sums to one and thresholds are finite",
        "limitation": "proxy prior-domination warning surface",
        "next_action": "carry flagged regimes into prior-sweep studies only",
    },
    {
        "chart_id": "06b_evidence_contract_spine",
        "epic": "Epic 2",
        "role": "main",
        "filename": "06b_evidence_contract_spine.png",
        "source_artifact": "epic_packets/02_evidence_construction_ladder/evidence_contract.json",
        "evidence_tier": "ARTIFACT-BACKED",
        "validation_check": "every evaluated method emits the shared posterior contract",
        "limitation": "contract diagram, not a leaderboard",
        "next_action": "keep extending methods through the same contract rather than special-casing outputs",
    },
    {
        "chart_id": "06_posterior_timeline_witness",
        "epic": "Epic 2",
        "role": "main",
        "filename": "06_posterior_timeline_witness.png",
        "source_artifact": "artifacts/common_1d_classifier_study/unified_posterior_history.csv",
        "evidence_tier": "RUN-BACKED",
        "validation_check": "posterior rows sum to one",
        "limitation": "representative witness rather than full regime atlas",
        "next_action": "expand witness families where posterior behavior is materially different",
    },
    {
        "chart_id": "07b_full_ladder_comparison_dashboard",
        "epic": "Epic 2",
        "role": "main",
        "filename": "07b_full_ladder_comparison_dashboard.png",
        "source_artifact": "artifacts/showcase/tables/full_ladder_metrics.csv",
        "evidence_tier": "RUN-BACKED",
        "validation_check": "full ladder metrics table nonempty",
        "limitation": "current study surface and witness bundle only",
        "next_action": "add more 3D-lift witnesses without changing metrics contract",
    },
    {
        "chart_id": "07d_static_ceiling_capture_by_method",
        "epic": "Epic 2",
        "role": "main",
        "filename": "07d_static_ceiling_capture_by_method.png",
        "source_artifact": "epic_packets/02_evidence_construction_ladder/static_ceiling_capture.csv",
        "evidence_tier": "RUN-BACKED",
        "validation_check": "ceiling rows cite Epic 1 proxy source and method ids",
        "limitation": "ceiling capture is a diagnostic bridge, not a theorem",
        "next_action": "replace proxy rows with richer oracle alignment where available",
    },
    {
        "chart_id": "07_rung_sufficiency_map",
        "epic": "Epic 2",
        "role": "main",
        "filename": "07_rung_sufficiency_map.png",
        "source_artifact": "artifacts/rung_sufficiency/rung_promotion_matrix.csv",
        "evidence_tier": "RUN-BACKED",
        "validation_check": "rung decisions present with simpler-rung comparisons",
        "limitation": "aggregate rung summary over current witnesses",
        "next_action": "keep the ladder broad but only promote methods per witness",
    },
    {
        "chart_id": "10e_advanced_filter_sweet_spot_matrix",
        "epic": "Epic 2",
        "role": "main",
        "filename": "10e_advanced_filter_sweet_spot_matrix.png",
        "source_artifact": "artifacts/advanced_filter_comparison_v1/advanced_method_gate_matrix.csv",
        "evidence_tier": "ARTIFACT-BACKED",
        "validation_check": "advanced-method claims cite named failure modes and baselines",
        "limitation": "sweet-spot map, not universal method promotion",
        "next_action": "add more shine witnesses before any broader promotion claim",
    },
    {
        "chart_id": "10f_method_win_by_regime_map",
        "epic": "Epic 2",
        "role": "main",
        "filename": "10f_method_win_by_regime_map.png",
        "source_artifact": "artifacts/packets/advanced_algorithm_showcase/method_win_by_regime.csv",
        "evidence_tier": "ARTIFACT-BACKED",
        "validation_check": "advanced methods appear only with regime-scoped wins",
        "limitation": "regime win is not global superiority",
        "next_action": "turn regime wins into 3D-relevant witness families",
    },
    {
        "chart_id": "22b_scenario_design_coverage_map",
        "epic": "Epic 3",
        "role": "main",
        "filename": "22b_scenario_design_coverage_map.png",
        "source_artifact": "epic_packets/03_corpus_explorer_design_engine/scenario_design_baselines.csv",
        "evidence_tier": "RUN-BACKED",
        "validation_check": "scenario design baselines cover declared families and tiers",
        "limitation": "1D scenario-space design map only",
        "next_action": "lift the same design contract into vector-valued and vehicle-specific corpora",
    },
    {
        "chart_id": "03_corpus_candidate_frontier",
        "epic": "Epic 3",
        "role": "main",
        "filename": "03_corpus_candidate_frontier.png",
        "source_artifact": "artifacts/packets/corpus_explorer_mvp/corpus_candidate_frontier.csv",
        "evidence_tier": "RUN-BACKED",
        "validation_check": "selected corpus manifest matches selected frontier rows",
        "limitation": "current frontier over declared 1D search objective",
        "next_action": "add richer objectives, not ad hoc generators",
    },
    {
        "chart_id": "21_search_backend_comparison_frontier",
        "epic": "Epic 3",
        "role": "main",
        "filename": "21_search_backend_comparison_frontier.png",
        "source_artifact": "artifacts/packets/corpus_explorer_mvp/search_backend_comparison.csv",
        "evidence_tier": "RUN-BACKED",
        "validation_check": "baseline comparison and promotion gates present for CEM/PPO",
        "limitation": "current witness budget and seed count only",
        "next_action": "expand seed stability and downstream-yield sweeps before broader search claims",
    },
    {
        "chart_id": "26_downstream_diagnostic_yield",
        "epic": "Epic 3",
        "role": "main",
        "filename": "26_downstream_diagnostic_yield.png",
        "source_artifact": "artifacts/packets/corpus_explorer_mvp/downstream_diagnostic_yield.csv",
        "evidence_tier": "RUN-BACKED",
        "validation_check": "yield rows cite ladder/filter actions and rejected invalid cases",
        "limitation": "diagnostic-yield table is current-packet specific",
        "next_action": "use downstream yield as the default exploration value metric",
    },
    {
        "chart_id": "28_failure_region_atlas",
        "epic": "Epic 3",
        "role": "main",
        "filename": "28_failure_region_atlas.png",
        "source_artifact": "epic_packets/03_corpus_explorer_design_engine/failure_region_atlas.csv",
        "evidence_tier": "RUN-BACKED",
        "validation_check": "every failure region cites validity, leakage, and routed action",
        "limitation": "atlas bins are current behavior descriptors only",
        "next_action": "stabilize descriptor choices before comparing 1D and 3D exploration surfaces",
    },
)


def write_v7_anduril_c2_blend_packet(
    output_dir: str | Path = ROOT / "artifacts" / "validation_packets" / "v7_anduril_c2_blend",
) -> V7AndurilC2BlendArtifacts:
    packet_dir = Path(output_dir)
    epic_root = packet_dir / "epic_packets"
    whitepaper_dir = packet_dir / "whitepaper"
    figures_dir = packet_dir / "figures"
    for path in (packet_dir, epic_root, whitepaper_dir / "sections", whitepaper_dir / "figures", figures_dir):
        path.mkdir(parents=True, exist_ok=True)

    epic1_dir = epic_root / "01_static_admissibility_gate"
    epic2_dir = epic_root / "02_evidence_construction_ladder"
    epic3_dir = epic_root / "03_corpus_explorer_design_engine"

    epic1_packet = write_static_admissibility_exemplar_suite_packet(epic1_dir)
    _augment_epic1_packet(epic1_dir, epic1_packet)
    _build_epic2_packet(epic2_dir)
    _build_epic3_packet(epic3_dir)
    hero_rows = _build_integrated_figures(packet_dir, figures_dir)

    _copy_if_exists(SOURCE_HERO_PACKET / "main_deck.pptx", packet_dir / "main_deck.pptx")
    _copy_if_exists(SOURCE_HERO_PACKET / "appendix_deck.pptx", packet_dir / "appendix_deck.pptx")
    _copy_if_exists(SOURCE_HERO_PACKET / "main_deck_contact_sheet.png", packet_dir / "main_deck_contact_sheet.png")
    _copy_if_exists(
        SOURCE_HERO_PACKET / "appendix_deck_contact_sheet.png",
        packet_dir / "appendix_deck_contact_sheet.png",
    )

    manifest_path = packet_dir / "packet_manifest.yaml"
    decision_card_path = packet_dir / "integrated_decision_card.md"
    claim_boundary_path = packet_dir / "integrated_claim_boundary.md"
    epic_summary_path = packet_dir / "integrated_epic_summary.md"
    hero_chart_manifest_path = packet_dir / "integrated_hero_chart_manifest.csv"
    lane_proof_matrix_path = packet_dir / "integrated_lane_proof_matrix.md"
    validation_report_path = packet_dir / "validation_report.md"
    whitepaper_main_path = whitepaper_dir / "main.tex"

    write_csv(hero_chart_manifest_path, hero_rows, list(hero_rows[0].keys()))
    _write_text(packet_dir / "README.md", _render_root_readme())
    _write_text(decision_card_path, _render_integrated_decision_card())
    _write_text(claim_boundary_path, _render_integrated_claim_boundary())
    _write_text(epic_summary_path, _render_integrated_epic_summary())
    _write_text(lane_proof_matrix_path, _render_integrated_lane_proof_matrix())
    _write_text(packet_dir / "integrated_lane_proof_matrix.csv", _lane_matrix_csv_text())
    _write_text(packet_dir / "integrated_claim_boundary.md", _render_integrated_claim_boundary())
    _write_text(packet_dir / "integrated_decision_card.md", _render_integrated_decision_card())
    _write_text(packet_dir / "integrated_epic_summary.md", _render_integrated_epic_summary())
    _write_text(packet_dir / "integrated_lane_proof_matrix.md", _render_integrated_lane_proof_matrix())
    _write_text(packet_dir / "integrated_decision_card.yaml", _render_integrated_decision_card_yaml())
    _write_text(packet_dir / "integrated_hero_chart_manifest.md", _render_chart_manifest_notes(hero_rows))
    _write_text(manifest_path, _render_packet_manifest())
    _write_whitepaper(whitepaper_dir)

    issues = validate_v7_anduril_c2_blend_packet(packet_dir)
    validation_lines = ["# Validation Report", ""]
    if issues:
        validation_lines.extend(f"- FAIL: {issue}" for issue in issues)
    else:
        validation_lines.append("- PASS: V7 integrated packet surfaces validated.")
    _write_text(validation_report_path, "\n".join(validation_lines) + "\n")

    return V7AndurilC2BlendArtifacts(
        packet_dir=packet_dir,
        manifest_path=manifest_path,
        decision_card_path=decision_card_path,
        claim_boundary_path=claim_boundary_path,
        epic_summary_path=epic_summary_path,
        hero_chart_manifest_path=hero_chart_manifest_path,
        lane_proof_matrix_path=lane_proof_matrix_path,
        validation_report_path=validation_report_path,
        main_deck_path=packet_dir / "main_deck.pptx",
        appendix_deck_path=packet_dir / "appendix_deck.pptx",
        whitepaper_main_path=whitepaper_main_path,
    )


def validate_v7_anduril_c2_blend_packet(packet_dir: str | Path) -> list[str]:
    base = Path(packet_dir)
    issues: list[str] = []
    required = (
        "README.md",
        "packet_manifest.yaml",
        "integrated_decision_card.md",
        "integrated_claim_boundary.md",
        "integrated_epic_summary.md",
        "integrated_hero_chart_manifest.csv",
        "integrated_lane_proof_matrix.md",
        "main_deck.pptx",
        "appendix_deck.pptx",
        "whitepaper/main.tex",
        "epic_packets/01_static_admissibility_gate/decision_card.md",
        "epic_packets/02_evidence_construction_ladder/static_ceiling_capture.csv",
        "epic_packets/03_corpus_explorer_design_engine/hard_case_route_ledger.csv",
    )
    for rel_path in required:
        if not (base / rel_path).exists():
            issues.append(f"missing required V7 artifact: {rel_path}")
    if issues:
        return issues

    issues.extend(
        f"epic1: {issue}"
        for issue in validate_static_admissibility_packet(
            base / "epic_packets" / "01_static_admissibility_gate",
            repo_root=ROOT,
        )
    )
    issues.extend(
        f"epic3: {issue}"
        for issue in validate_corpus_explorer_packet(base / "epic_packets" / "03_corpus_explorer_design_engine")
    )

    route_rows = read_csv_rows(
        base / "epic_packets" / "01_static_admissibility_gate" / "source_artifacts" / "exemplar_route_matrix.csv"
    )
    by_exemplar = {row["exemplar_id"]: row for row in route_rows}
    leakage = by_exemplar.get("leakage_blocker_family")
    if leakage is None or leakage.get("actual_route") != "reject":
        issues.append("Epic 1 leakage_blocker_family must route to reject")
    coverage = by_exemplar.get("coverage_thin_cells_family")
    if coverage is None or coverage.get("actual_route") != "promote_to_corpus_explorer":
        issues.append("Epic 1 coverage_thin_cells_family must route to corpus work rather than algorithm escalation")

    hero_rows = read_csv_rows(base / "integrated_hero_chart_manifest.csv")
    required_columns = {
        "chart_id",
        "epic",
        "role",
        "path",
        "evidence_tier",
        "source_artifact",
        "validation_check",
        "limitation",
        "next_action",
    }
    if not hero_rows:
        issues.append("integrated_hero_chart_manifest.csv must not be empty")
    else:
        missing = required_columns.difference(hero_rows[0])
        if missing:
            issues.append(f"integrated_hero_chart_manifest.csv missing columns: {sorted(missing)}")
    for row in hero_rows:
        for column in required_columns:
            if not row.get(column):
                issues.append(f"{row.get('chart_id', '<unknown>')} missing manifest field `{column}`")
        if not (base / row["path"]).exists():
            issues.append(f"{row['chart_id']} references missing chart file {row['path']}")

    synergy_rows = read_csv_rows(
        base / "epic_packets" / "01_static_admissibility_gate" / "hero_chart_manifest.csv"
    )
    synergy_row = next((row for row in synergy_rows if row.get("chart_id") == "02f_feature_synergy_map"), None)
    if synergy_row is not None and "candidate" not in synergy_row.get("claim_boundary", "").lower():
        issues.append("Epic 1 synergy surface must remain candidate evidence without ablation-backed proof")

    e2_status = read_csv_rows(base / "epic_packets" / "02_evidence_construction_ladder" / "method_status_table.csv")
    for method_id in ("particle_filter", "rbpf"):
        row = next((item for item in e2_status if item.get("method_id") == method_id), None)
        if row is None:
            issues.append(f"Epic 2 method_status_table missing {method_id}")
            continue
        if row.get("decision", "").startswith("promoted") and row.get("applicable") == "witness_only":
            issues.append(f"Epic 2 cannot broadly promote {method_id} while it remains witness_only")

    backend_rows = read_csv_rows(base / "epic_packets" / "03_corpus_explorer_design_engine" / "search_backend_comparison.csv")
    for backend_id in ("cem_open_loop", "ppo_policy"):
        row = next((item for item in backend_rows if item.get("backend_id") == backend_id), None)
        if row is None:
            issues.append(f"Epic 3 search backend comparison missing {backend_id}")
            continue
        gate = row.get("promotion_gate", "")
        if "baseline" not in gate or "ablation" not in gate or "seed_stability" not in gate or "downstream_yield" not in gate:
            issues.append(f"Epic 3 {backend_id} gate must preserve baseline, ablation, seed stability, and downstream yield requirements")

    selected_rows = read_csv_rows(base / "epic_packets" / "03_corpus_explorer_design_engine" / "selected_corpus_manifest.csv")
    for row in selected_rows:
        if row.get("validity_status") != "pass" or row.get("leakage_status") != "pass":
            issues.append(f"Epic 3 selected candidate {row.get('candidate_id')} must be valid and non-leaky")

    decision_text = (base / "integrated_decision_card.md").read_text(encoding="utf-8")
    for token in (
        "epic_1_static_admissibility_gate",
        "epic_2_evidence_construction_ladder",
        "epic_3_corpus_explorer_design_engine",
        "presentable_methodology_workbench",
        "general PF/RBPF promotion without run-backed shine witnesses",
        "general CEM/PPO superiority without baseline, ablation, seed stability, and downstream-yield evidence",
    ):
        if token not in decision_text:
            issues.append(f"integrated_decision_card.md missing token: {token}")

    return issues


def _augment_epic1_packet(epic1_dir: Path, packet) -> None:
    study_examples_dir = epic1_dir / "study_bundle_examples"
    study_examples_dir.mkdir(parents=True, exist_ok=True)
    source_bundles_dir = epic1_dir / "source_bundles"
    for source_dir in sorted(path for path in source_bundles_dir.iterdir() if path.is_dir()):
        shutil.copytree(source_dir, study_examples_dir / source_dir.name, dirs_exist_ok=True)
    _write_json(
        epic1_dir / "static_bundle_schema.json",
        {
            "type": "object",
            "required": ["static_audit_bundle.yaml", "samples.csv", "feature_schema.csv", "class_schema.csv"],
            "properties": {
                "static_audit_bundle.yaml": {"description": "Study declaration and prior regime"},
                "samples.csv": {"description": "Labeled feature rows"},
                "feature_schema.csv": {"description": "Feature provenance and online/leakage flags"},
                "class_schema.csv": {"description": "Declared class surface"},
            },
        },
    )
    route_rows = read_csv_rows(packet.route_matrix_path)
    write_csv(epic1_dir / "route_ledger.csv", route_rows, list(route_rows[0].keys()))
    _copy_if_exists(epic1_dir / "figures" / "02a_static_exemplar_suite_routing_matrix.png", epic1_dir / "figures" / "02m_static_exemplar_decision_surface.png")
    doc = MarkdownDocument("Epic 1 Exemplar Decision Surface")
    doc.paragraph(
        "This surface treats the static admissibility gate as a decision system over six file-backed study bundle families. The proof obligation is not raw accuracy; it is whether each family routes to the correct next action before corpus search or classifier escalation."
    )
    doc.heading("Expected Families", level=2)
    doc.table(
        ["Exemplar", "Expected", "Observed", "Validator"],
        [
            (
                row["exemplar_id"],
                row["expected_route"],
                row["actual_route"],
                row["validator_result"],
            )
            for row in route_rows
        ],
    )
    _write_text(epic1_dir / "exemplar_suite_decision_surface.md", doc.text() + "\n")
    _write_text(epic1_dir / "static_audit_decision_card.md", (epic1_dir / "decision_card.md").read_text(encoding="utf-8"))


def _build_epic2_packet(epic2_dir: Path) -> None:
    shutil.copytree(SOURCE_E2_PACKET, epic2_dir, dirs_exist_ok=True)
    figures_dir = epic2_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for name in (
        "posterior_history_by_method.csv",
        "full_ladder_metrics.csv",
        "method_status_table.csv",
        "calibration_by_method.csv",
        "confusion_by_method.csv",
        "runtime_by_method.csv",
        "classifier_ladder_decision_card.md",
        "method_win_by_regime.csv",
    ):
        _copy_if_exists(SOURCE_SHOWCASE_TABLES / name, epic2_dir / name)

    ceiling_rows = _build_static_ceiling_capture_rows()
    write_csv(epic2_dir / "static_ceiling_capture.csv", ceiling_rows, list(ceiling_rows[0].keys()))
    _write_json(epic2_dir / "evidence_contract.json", _evidence_contract_schema())
    _write_text(epic2_dir / "rung_sufficiency_report.md", _render_epic2_rung_report())
    _write_text(epic2_dir / "advanced_state_space_justification_report.md", _render_epic2_advanced_report())
    _write_text(epic2_dir / "witness_cards.md", _render_epic2_witness_cards())
    _write_text(epic2_dir / "claim_boundary.md", _render_epic2_claim_boundary())
    _write_text(epic2_dir / "validation_report.md", _render_epic2_validation_report())
    _write_text(epic2_dir / "README.md", _render_epic2_readme())
    _write_text(epic2_dir / "packet_manifest.yaml", _render_epic2_manifest())

    _copy_if_exists(SOURCE_HERO_PACKET / "figures" / "06_posterior_timeline_witness.png", figures_dir / "06_posterior_timeline_witness.png")
    _copy_if_exists(SOURCE_SHOWCASE_PLOTS / "07b_full_ladder_comparison_dashboard.png", figures_dir / "07b_full_ladder_comparison_dashboard.png")
    _copy_if_exists(SOURCE_HERO_PACKET / "figures" / "07_rung_sufficiency_map.png", figures_dir / "07_rung_sufficiency_map.png")
    _copy_if_exists(SOURCE_HERO_PACKET / "figures" / "10e_advanced_filter_sweet_spot_matrix.png", figures_dir / "10e_advanced_filter_sweet_spot_matrix.png")
    _copy_if_exists(SOURCE_HERO_PACKET / "figures" / "10f_method_win_by_regime_map.png", figures_dir / "10f_method_win_by_regime_map.png")
    _copy_if_exists(SOURCE_HERO_PACKET / "figures" / "12_1d_to_3d_pva_lift_map.png", figures_dir / "10g_1d_to_3d_state_space_escalation_map.png")
    _render_evidence_contract_spine(figures_dir / "06b_evidence_contract_spine.png")
    _render_static_ceiling_capture(figures_dir / "07d_static_ceiling_capture_by_method.png", ceiling_rows)


def _build_epic3_packet(epic3_dir: Path) -> None:
    shutil.copytree(SOURCE_E3_PACKET, epic3_dir, dirs_exist_ok=True)
    figures_dir = epic3_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    frontier_rows = read_csv_rows(epic3_dir / "corpus_candidate_frontier.csv")
    route_rows = read_csv_rows(epic3_dir / "advanced_algorithm_route_matrix.csv")
    baseline_rows = _scenario_design_baseline_rows(frontier_rows)
    illumination_rows = _illumination_rows(frontier_rows)
    atlas_rows = _failure_region_rows(frontier_rows, route_rows)
    ledger_rows = _route_ledger_rows(frontier_rows)

    write_csv(epic3_dir / "scenario_design_baselines.csv", baseline_rows, list(baseline_rows[0].keys()))
    write_csv(epic3_dir / "illumination_map.csv", illumination_rows, list(illumination_rows[0].keys()))
    write_csv(epic3_dir / "failure_region_atlas.csv", atlas_rows, list(atlas_rows[0].keys()))
    write_csv(epic3_dir / "hard_case_route_ledger.csv", ledger_rows, list(ledger_rows[0].keys()))
    _write_text(epic3_dir / "scenario_design_baseline_report.md", _render_epic3_design_report())
    _write_text(epic3_dir / "corpus_coverage_report.md", _render_epic3_coverage_report())
    _write_text(epic3_dir / "claim_boundary.md", _render_epic3_claim_boundary())
    _write_text(epic3_dir / "validation_report.md", _render_epic3_validation_report())
    _write_text(epic3_dir / "packet_manifest.yaml", _render_epic3_manifest())

    selected_dir = epic3_dir / "hard_case_cards"
    rejected_dir = epic3_dir / "rejected_case_cards"
    rejected_dir.mkdir(parents=True, exist_ok=True)
    for row in frontier_rows:
        card_name = f"{row['candidate_id']}_{row['target_failure_mode']}.md"
        source = selected_dir / card_name
        if source.exists() and row.get("selected") != "True":
            shutil.copy2(source, rejected_dir / card_name)

    _copy_if_exists(SOURCE_HERO_PACKET / "figures" / "03_corpus_candidate_frontier.png", figures_dir / "03_corpus_candidate_frontier.png")
    _copy_if_exists(SOURCE_HERO_PACKET / "figures" / "21_search_backend_comparison_frontier.png", figures_dir / "21_search_backend_comparison_frontier.png")
    _copy_if_exists(SOURCE_HERO_PACKET / "figures" / "26_downstream_diagnostic_yield.png", figures_dir / "26_downstream_diagnostic_yield.png")
    _copy_if_exists(SOURCE_HERO_PACKET / "figures" / "27_novelty_to_filter_escalation_bridge.png", figures_dir / "27_novelty_to_filter_escalation_bridge.png")
    _copy_if_exists(SOURCE_HERO_PACKET / "figures" / "18_leakage_adequacy_audit.png", figures_dir / "18_leakage_adequacy_audit.png")
    _render_scenario_design_coverage(figures_dir / "22b_scenario_design_coverage_map.png", baseline_rows)
    _render_failure_region_atlas(figures_dir / "28_failure_region_atlas.png", atlas_rows)
    _render_illumination_map(figures_dir / "29_illumination_map.png", illumination_rows)


def _build_integrated_figures(packet_dir: Path, figures_dir: Path) -> list[dict[str, str]]:
    hero_rows: list[dict[str, str]] = []
    integrated_sources = {
        "02m_static_exemplar_decision_surface": packet_dir
        / "epic_packets"
        / "01_static_admissibility_gate"
        / "figures"
        / "02m_static_exemplar_decision_surface.png",
        "02c_class_pair_confusability_matrix": SOURCE_HERO_PACKET / "figures" / "02c_class_pair_confusability_matrix.png",
        "02g_prior_pathology_surface": SOURCE_HERO_PACKET / "figures" / "02g_prior_pathology_surface.png",
        "06b_evidence_contract_spine": packet_dir
        / "epic_packets"
        / "02_evidence_construction_ladder"
        / "figures"
        / "06b_evidence_contract_spine.png",
        "06_posterior_timeline_witness": packet_dir
        / "epic_packets"
        / "02_evidence_construction_ladder"
        / "figures"
        / "06_posterior_timeline_witness.png",
        "07b_full_ladder_comparison_dashboard": packet_dir
        / "epic_packets"
        / "02_evidence_construction_ladder"
        / "figures"
        / "07b_full_ladder_comparison_dashboard.png",
        "07d_static_ceiling_capture_by_method": packet_dir
        / "epic_packets"
        / "02_evidence_construction_ladder"
        / "figures"
        / "07d_static_ceiling_capture_by_method.png",
        "07_rung_sufficiency_map": packet_dir
        / "epic_packets"
        / "02_evidence_construction_ladder"
        / "figures"
        / "07_rung_sufficiency_map.png",
        "10e_advanced_filter_sweet_spot_matrix": packet_dir
        / "epic_packets"
        / "02_evidence_construction_ladder"
        / "figures"
        / "10e_advanced_filter_sweet_spot_matrix.png",
        "10f_method_win_by_regime_map": packet_dir
        / "epic_packets"
        / "02_evidence_construction_ladder"
        / "figures"
        / "10f_method_win_by_regime_map.png",
        "22b_scenario_design_coverage_map": packet_dir
        / "epic_packets"
        / "03_corpus_explorer_design_engine"
        / "figures"
        / "22b_scenario_design_coverage_map.png",
        "03_corpus_candidate_frontier": packet_dir
        / "epic_packets"
        / "03_corpus_explorer_design_engine"
        / "figures"
        / "03_corpus_candidate_frontier.png",
        "21_search_backend_comparison_frontier": packet_dir
        / "epic_packets"
        / "03_corpus_explorer_design_engine"
        / "figures"
        / "21_search_backend_comparison_frontier.png",
        "26_downstream_diagnostic_yield": packet_dir
        / "epic_packets"
        / "03_corpus_explorer_design_engine"
        / "figures"
        / "26_downstream_diagnostic_yield.png",
        "28_failure_region_atlas": packet_dir
        / "epic_packets"
        / "03_corpus_explorer_design_engine"
        / "figures"
        / "28_failure_region_atlas.png",
    }
    for spec in INTEGRATED_MAIN_CHARTS:
        source = integrated_sources[spec["chart_id"]]
        target = figures_dir / spec["filename"]
        _copy_if_exists(source, target)
        hero_rows.append(
            {
                "chart_id": spec["chart_id"],
                "epic": spec["epic"],
                "role": spec["role"],
                "path": str(target.relative_to(packet_dir)),
                "evidence_tier": spec["evidence_tier"],
                "source_artifact": spec["source_artifact"],
                "validation_check": spec["validation_check"],
                "limitation": spec["limitation"],
                "next_action": spec["next_action"],
            }
        )
    return hero_rows


def _build_static_ceiling_capture_rows() -> list[dict[str, str]]:
    source_rows = read_csv_rows(SOURCE_CEILING)
    rows: list[dict[str, str]] = []
    for row in source_rows:
        rows.append(
            {
                "method_id": row["method_id"],
                "display_name": row["display_name"],
                "public_family": row["public_family"],
                "fraction_of_proxy_captured_capped": row.get("mean_fraction_of_proxy_captured_capped", ""),
                "mean_classifier_accuracy": row.get("mean_classifier_accuracy", ""),
                "mean_epic1_oracle_proxy": row.get("mean_epic1_oracle_proxy", ""),
                "ceiling_status": row.get("ceiling_status", ""),
                "source_static_audit": "artifacts/static_feature_class_prior_audit_v1",
                "evidence_note": row.get("evidence_note", ""),
            }
        )
    return rows


def _scenario_design_baseline_rows(frontier_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    counts: dict[tuple[str, str], int] = {}
    for row in frontier_rows:
        key = (row["scenario_family"], row["class_pair_target"])
        counts[key] = counts.get(key, 0) + 1
    rows: list[dict[str, str]] = []
    for (scenario_family, class_pair_target), count in sorted(counts.items()):
        rows.append(
            {
                "scenario_family": scenario_family,
                "class_pair_target": class_pair_target,
                "design_baseline": "grid_then_guided_search",
                "candidate_count": str(count),
                "selected_count": str(
                    sum(
                        1
                        for row in frontier_rows
                        if row["scenario_family"] == scenario_family
                        and row["class_pair_target"] == class_pair_target
                        and row.get("selected") == "True"
                    )
                ),
            }
        )
    return rows


def _illumination_rows(frontier_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in frontier_rows:
        boundary = float(row.get("boundary_stress_score") or 0.0)
        excitation = float(row.get("feature_excitation_score") or 0.0)
        rows.append(
            {
                "candidate_id": row["candidate_id"],
                "scenario_family": row["scenario_family"],
                "class_pair_target": row["class_pair_target"],
                "boundary_bin": _score_bin(boundary),
                "excitation_bin": _score_bin(excitation),
                "selected": row.get("selected", ""),
            }
        )
    return rows


def _failure_region_rows(
    frontier_rows: list[dict[str, str]],
    route_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    route_lookup = {row["failure_mode"]: row for row in route_rows}
    rows: list[dict[str, str]] = []
    for row in frontier_rows:
        route = route_lookup.get(row["target_failure_mode"], {})
        rows.append(
            {
                "candidate_id": row["candidate_id"],
                "scenario_family": row["scenario_family"],
                "class_pair_target": row["class_pair_target"],
                "target_failure_mode": row["target_failure_mode"],
                "validity_status": row["validity_status"],
                "leakage_status": row["leakage_status"],
                "route_status": route.get("route_status", "candidate"),
                "advanced_algorithm": route.get("advanced_algorithm", "reject_candidate"),
                "routed_action": row["routed_action"],
                "selected": row["selected"],
            }
        )
    return rows


def _route_ledger_rows(frontier_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "candidate_id": row["candidate_id"],
            "selected": row["selected"],
            "validity_status": row["validity_status"],
            "leakage_status": row["leakage_status"],
            "target_failure_mode": row["target_failure_mode"],
            "routed_action": row["routed_action"],
            "rejection_reason": row["rejection_reason"],
        }
        for row in frontier_rows
    ]


def _render_evidence_contract_spine(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.axis("off")
    labels = ["features", "evidence", "posterior", "metrics", "decision card"]
    xs = [0.08, 0.28, 0.48, 0.68, 0.88]
    for x, label in zip(xs, labels):
        ax.text(
            x,
            0.55,
            label,
            ha="center",
            va="center",
            fontsize=13,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "#F4F6F7", "edgecolor": "#AEB6BF"},
        )
    for start, end in zip(xs[:-1], xs[1:]):
        ax.annotate("", xy=(end - 0.06, 0.55), xytext=(start + 0.06, 0.55), arrowprops={"arrowstyle": "->", "lw": 2.0})
    ax.text(0.5, 0.86, "Shared Evidence / Posterior Contract", ha="center", fontsize=16, weight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _render_static_ceiling_capture(path: Path, rows: list[dict[str, str]]) -> None:
    filtered = [row for row in rows if row["fraction_of_proxy_captured_capped"]]
    labels = [row["display_name"] for row in filtered]
    values = [float(row["fraction_of_proxy_captured_capped"]) for row in filtered]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(labels, values, color="#2E86AB")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("capped fraction")
    ax.set_title("Static Ceiling Capture by Method", loc="left", fontsize=16, weight="bold")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _render_scenario_design_coverage(path: Path, rows: list[dict[str, str]]) -> None:
    labels = [f"{row['scenario_family']}\n{row['class_pair_target']}" for row in rows]
    values = [int(row["candidate_count"]) for row in rows]
    selected = [int(row["selected_count"]) for row in rows]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(labels, values, color="#D5DBDB", label="candidates")
    ax.bar(labels, selected, color="#1B998B", label="selected")
    ax.set_title("Scenario Design Coverage Map", loc="left", fontsize=16, weight="bold")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _render_failure_region_atlas(path: Path, rows: list[dict[str, str]]) -> None:
    labels = [row["target_failure_mode"] for row in rows]
    selected = [1 if row["selected"] == "True" else 0 for row in rows]
    validity = [1 if row["validity_status"] == "pass" else 0 for row in rows]
    leakage = [1 if row["leakage_status"] == "pass" else 0 for row in rows]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = range(len(rows))
    ax.scatter(x, selected, label="selected", color="#2E7D32", s=100)
    ax.scatter(x, validity, label="valid", color="#2E86AB", marker="s", s=60)
    ax.scatter(x, leakage, label="non-leaky", color="#E67E22", marker="^", s=60)
    ax.set_xticks(list(x), labels, rotation=25, ha="right")
    ax.set_yticks([0, 1], ["no", "yes"])
    ax.set_title("Failure Region Atlas", loc="left", fontsize=16, weight="bold")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _render_illumination_map(path: Path, rows: list[dict[str, str]]) -> None:
    xs = [{"low": 0, "mid": 1, "high": 2}[row["boundary_bin"]] for row in rows]
    ys = [{"low": 0, "mid": 1, "high": 2}[row["excitation_bin"]] for row in rows]
    colors = ["#1B998B" if row["selected"] == "True" else "#AEB6BF" for row in rows]
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.scatter(xs, ys, c=colors, s=120)
    ax.set_xticks([0, 1, 2], ["low", "mid", "high"])
    ax.set_yticks([0, 1, 2], ["low", "mid", "high"])
    ax.set_xlabel("boundary stress bin")
    ax.set_ylabel("feature excitation bin")
    ax.set_title("Behavior-Space Illumination", loc="left", fontsize=16, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _score_bin(value: float) -> str:
    if value < 0.34:
        return "low"
    if value < 0.67:
        return "mid"
    return "high"


def _copy_if_exists(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _render_root_readme() -> str:
    return "\n".join(
        [
            "# V7 Anduril/C2 Blend",
            "",
            "This packet treats the repository as one methodology workbench with three proof systems:",
            "",
            "1. Epic 1: reusable static admissibility gate over portable feature/class/prior bundles.",
            "2. Epic 2: evidence-construction ladder that measures how admissible signal becomes posterior certainty.",
            "3. Epic 3: corpus explorer and experimental-design engine that discovers valid hard cases and routes them back into study actions.",
            "",
            "The packet is intentionally architectural and run-backed. It reuses the existing Epic 1 exemplar-suite generator, the current classifier-ladder and advanced-algorithm packets, the corpus explorer packet, and the presentation hero bundle, then reorganizes them into the stronger V7 final-story shape.",
        ]
    ) + "\n"


def _render_integrated_decision_card() -> str:
    return _render_integrated_decision_card_yaml().replace("integrated_decision_card:\n", "# Integrated Decision Card\n\n")


def _render_integrated_decision_card_yaml() -> str:
    return "\n".join(
        [
            "integrated_decision_card:",
            "  packet_id: v7_anduril_c2_blend",
            "  study_id: methodology_workbench_story_v1",
            "  run_id: v7_packet_rework",
            "  evidence_tier: mixed",
            "  epic_1_static_admissibility_gate:",
            "    status: route_surface_validated",
            "    exemplar_suite_status: six_file_backed_families_checked",
            "    portable_bundle_status: schema_and_examples_present",
            "    strongest_evidence:",
            "      - static exemplar decision surface",
            "      - class confusability matrix",
            "      - prior pathology surface",
            "      - leakage provenance audit",
            "    blockers: []",
            "    warnings:",
            "      - candidate synergy remains candidate until ablation-backed evidence exists",
            "    next_actions:",
            "      - route thin coverage and hard pairs into corpus objectives",
            "  epic_2_evidence_construction_ladder:",
            "    status: full_ladder_run_selective_promotion",
            "    evaluated_methods:",
            "      - pointwise",
            "      - windowed",
            "      - sequential_bayes",
            "      - kalman_bank",
            "      - transition_matrix",
            "      - imm",
            "      - particle_filter",
            "      - rbpf",
            "    simplest_sufficient_rung: transition_matrix_for_main_switching_witness",
            "    static_ceiling_capture_summary: proxy_bridge_present_for_named methods and explicitly incomplete elsewhere",
            "    advanced_state_space_status:",
            "      imm: witness_supported_and_methodologically_promoted_on_switching_witness",
            "      pf: witness_specific_only_not_generally_promoted",
            "      rbpf: witness_specific_only_not_generally_promoted",
            "    next_actions:",
            "      - add more shine witnesses before any broader PF/RBPF promotion claim",
            "  epic_3_corpus_explorer_design_engine:",
            "    status: corpus_objectives_to_route_ledger_operational",
            "    selected_corpus_id: corpus_explorer_mvp_selected_surface",
            "    design_baseline_status: present",
            "    illumination_status: present",
            "    search_backend_status:",
            "      qd: run_backed",
            "      cem: gated_not_promoted",
            "      ppo: gated_not_promoted",
            "    hard_case_routes:",
            "      - transition_switching_delay -> IMM witness",
            "      - nonlinear_posterior_candidate -> PF/GSF frontier",
            "      - maneuver_vs_oscillatory_confusion -> RBPF latent-event witness",
            "    next_actions:",
            "      - expand seeds and downstream-yield studies before broader search claims",
            "  overall_decision:",
            "    status: presentable_methodology_workbench",
            "    strongest_claim: architectural_integration_with_validated_epic_packets",
            "    not_claimed:",
            "      - operational deployment performance",
            "      - general superiority of one classifier family",
            "      - general PF/RBPF promotion without run-backed shine witnesses",
            "      - general CEM/PPO superiority without baseline, ablation, seed stability, and downstream-yield evidence",
            "    next_work:",
            "      - complete a V7-native deck renderer instead of inheriting the current hero deck",
            "      - add 3D-lift witness packets that preserve the same contracts",
        ]
    ) + "\n"


def _render_integrated_claim_boundary() -> str:
    return "\n".join(
        [
            "# Integrated Claim Boundary",
            "",
            "The strongest V7 claim is architectural and methodological: the repository now has a packet shape that shows how static admissibility, evidence construction, and corpus exploration fit together as one decision workbench.",
            "",
            "This packet does not claim deployment readiness, universal advanced-filter superiority, or general CEM/PPO superiority. It also does not claim that the inherited V5 hero deck is already a V7-native slide renderer. The deck is copied forward as a presentable inherited surface while the new packet organizes the stronger proof obligations around it.",
        ]
    ) + "\n"


def _render_integrated_epic_summary() -> str:
    return "\n".join(
        [
            "# Integrated Epic Summary",
            "",
            "Epic 1 proves that a feature/class/prior study bundle can be screened before heavier work, and that the route surface is stable over six file-backed study families.",
            "",
            "Epic 2 proves that multiple evidence-construction families can be evaluated through the same posterior contract, compared against the same metrics, and interpreted against an Epic 1 ceiling proxy rather than only by raw accuracy.",
            "",
            "Epic 3 proves that static warnings and ladder failures can be turned into corpus objectives, explored by multiple backends, rejected when invalid or leaky, and routed into concrete study actions through a hard-case ledger.",
        ]
    ) + "\n"


def _render_integrated_lane_proof_matrix() -> str:
    return "\n".join(
        [
            "# Integrated Lane Proof Matrix",
            "",
            "## Epic 1",
            "",
            "- claim: the static gate can route portable study bundles before corpus or ladder work",
            "- backing packet: `epic_packets/01_static_admissibility_gate/`",
            "- key proof: six file-backed exemplar families and the exemplar decision surface",
            "- limitation: synergy remains candidate evidence until downstream ablation confirms it",
            "",
            "## Epic 2",
            "",
            "- claim: evaluated methods share a posterior contract and can be judged against both ladder metrics and an Epic 1 ceiling proxy",
            "- backing packet: `epic_packets/02_evidence_construction_ladder/`",
            "- key proof: full ladder dashboard, static ceiling capture, rung sufficiency, and advanced state-space sweet spots",
            "- limitation: PF/RBPF remain witness-specific rather than broadly promoted",
            "",
            "## Epic 3",
            "",
            "- claim: corpus exploration is a design-and-routing engine, not only a generator",
            "- backing packet: `epic_packets/03_corpus_explorer_design_engine/`",
            "- key proof: scenario design baselines, corpus frontier, backend comparison, failure-region atlas, and hard-case route ledger",
            "- limitation: CEM/PPO remain gated by baseline comparison, ablation, seed stability, and downstream-yield evidence",
        ]
    ) + "\n"


def _lane_matrix_csv_text() -> str:
    rows = [
        ["epic", "claim", "backing_packet", "proof", "limitation", "next_work"],
        [
            "Epic 1",
            "route portable study bundles before heavier work",
            "epic_packets/01_static_admissibility_gate",
            "six file-backed exemplars and route ledger",
            "synergy remains candidate",
            "add more families only when routing changes",
        ],
        [
            "Epic 2",
            "compare methods under shared posterior contract",
            "epic_packets/02_evidence_construction_ladder",
            "full ladder metrics and static ceiling capture",
            "PF/RBPF still witness-specific",
            "expand shine witnesses and 3D-lift baselines",
        ],
        [
            "Epic 3",
            "discover and route valid hard cases",
            "epic_packets/03_corpus_explorer_design_engine",
            "backend comparison, atlas, and route ledger",
            "CEM/PPO still gated",
            "increase seed stability and downstream-yield breadth",
        ],
    ]
    return "\n".join(",".join(row) for row in rows) + "\n"


def _render_chart_manifest_notes(rows: list[dict[str, str]]) -> str:
    doc = MarkdownDocument("Integrated Hero Chart Manifest")
    doc.paragraph(
        "Every main-deck chart carries its source artifact, validation check, limitation, and next action so the packet can be read as a decision surface rather than only a gallery."
    )
    doc.table(
        ["Chart", "Epic", "Evidence", "Validation", "Limitation"],
        [
            (
                row["chart_id"],
                row["epic"],
                row["evidence_tier"],
                row["validation_check"],
                row["limitation"],
            )
            for row in rows
        ],
    )
    return doc.text() + "\n"


def _render_packet_manifest() -> str:
    return "\n".join(
        [
            "packet_id: v7_anduril_c2_blend",
            "title: V7 final packet rework",
            "source_deck: artifacts/presentation_hero_charts_v5",
            "epic_packets:",
            "  - epic_packets/01_static_admissibility_gate",
            "  - epic_packets/02_evidence_construction_ladder",
            "  - epic_packets/03_corpus_explorer_design_engine",
            "validator: PYTHONPATH=src python3 -m kinematic_classifier_sandbox validate-packet artifacts/validation_packets/v7_anduril_c2_blend --profile v7_anduril_c2_blend",
        ]
    ) + "\n"


def _write_whitepaper(whitepaper_dir: Path) -> None:
    sections_dir = whitepaper_dir / "sections"
    _write_text(
        whitepaper_dir / "main.tex",
        "\n".join(
            [
                "\\documentclass{article}",
                "\\usepackage{graphicx}",
                "\\begin{document}",
                "\\input{sections/01_introduction.tex}",
                "\\input{sections/02_epic1_static_admissibility_gate.tex}",
                "\\input{sections/03_epic2_evidence_construction_ladder.tex}",
                "\\input{sections/04_epic3_corpus_explorer_design_engine.tex}",
                "\\input{sections/05_integrated_decision_card.tex}",
                "\\input{sections/06_claim_boundaries_and_next_work.tex}",
                "\\end{document}",
            ]
        )
        + "\n",
    )
    section_text = {
        "01_introduction.tex": "\\section{Introduction}\nThis whitepaper presents the repository as a kinematic-classification methodology workbench organized around three proof systems.\n",
        "02_epic1_static_admissibility_gate.tex": "\\section{Reusable Static Admissibility Gate}\nEpic 1 proves that portable feature/class/prior study bundles can be routed before corpus search or classifier escalation.\n",
        "03_epic2_evidence_construction_ladder.tex": "\\section{Evidence-Construction Ladder}\nEpic 2 measures how admissible signal becomes posterior certainty and where advanced state-space methods are justified.\n",
        "04_epic3_corpus_explorer_design_engine.tex": "\\section{Corpus Explorer and Experimental-Design Engine}\nEpic 3 turns static warnings and ladder failures into corpus objectives, valid hard cases, and routed study actions.\n",
        "05_integrated_decision_card.tex": "\\section{Integrated Decision Card}\nThe integrated decision card records the status, strongest evidence, and next work for each epic.\n",
        "06_claim_boundaries_and_next_work.tex": "\\section{Claim Boundaries and Next Work}\nThe packet is methodological and architectural. It does not claim deployment readiness or universal method superiority.\n",
    }
    for filename, text in section_text.items():
        _write_text(sections_dir / filename, text)
    _write_text(whitepaper_dir / "references.bib", "% V7 whitepaper references placeholder\n")


def _evidence_contract_schema() -> dict[str, object]:
    return {
        "required_fields": [
            "method_id",
            "trajectory_id",
            "scenario_id",
            "time",
            "true_class",
            "posterior_class_a",
            "posterior_class_b",
        ],
        "purpose": "Shared evidence/posterior contract for method-comparable ladder outputs.",
    }


def _render_epic2_rung_report() -> str:
    return (
        "# Rung Sufficiency Report\n\n"
        "Run the full ladder. Promote selectively. The report surface in this packet is deliberately broader than any one study result: it shows which rung is simplest sufficient for a witness, which methods are merely evaluated, and which advanced methods are still scoped to named shine regimes.\n"
    )


def _render_epic2_advanced_report() -> str:
    return (
        "# Advanced State-Space Justification Report\n\n"
        "IMM, PF, and RBPF appear here as methodologically important rungs on the route to 3D. Their inclusion means the workbench can host, compare, and diagnose them. It does not mean every advanced method is broadly promoted beyond its current witnesses.\n"
    )


def _render_epic2_witness_cards() -> str:
    return "\n".join(
        [
            "# Witness Cards",
            "",
            "- IMM: switching/state-mixing witness where transition logic alone is insufficient.",
            "- PF: nonlinear/non-Gaussian or multimodal posterior witness where Gaussian assumptions collapse.",
            "- RBPF: latent-event witness where sampled latent structure and conditional continuous state should be separated.",
        ]
    ) + "\n"


def _render_epic2_claim_boundary() -> str:
    return (
        "# Epic 2 Claim Boundary\n\n"
        "Epic 2 is about posterior-certainty conversion and evidence-construction families. A method can be evaluated and methodologically promoted on the ladder without being claimed as the default for all future systems.\n"
    )


def _render_epic2_validation_report() -> str:
    return (
        "# Epic 2 Validation Report\n\n"
        "- shared posterior contract file present\n"
        "- full ladder metrics, calibration, confusion, runtime, and status tables present\n"
        "- static ceiling capture bridge present and explicitly diagnostic\n"
        "- advanced methods remain witness-scoped where broader promotion is not justified\n"
    )


def _render_epic2_readme() -> str:
    return (
        "# Epic 2: Evidence-Construction Ladder\n\n"
        "This packet reframes the classifier/filter story as signal-to-posterior conversion. It combines the existing ladder packet, showcase tables, and a static-ceiling bridge so the work can be read as a methodology proof rather than a one-off classifier comparison.\n"
    )


def _render_epic2_manifest() -> str:
    return (
        "packet_id: 02_evidence_construction_ladder\n"
        "source_packets:\n"
        "  - artifacts/packets/classifier_ladder_mvp\n"
        "  - artifacts/showcase/tables\n"
        "  - artifacts/classifier_family_scorecard_v1/ceiling_efficiency.csv\n"
    )


def _render_epic3_design_report() -> str:
    return (
        "# Scenario Design Baseline Report\n\n"
        "Epic 3 begins with declared scenario-space baselines so the search story does not collapse into whichever backend happened to stumble into a witness first. The current baseline layer is still 1D-oriented, but it is explicit and therefore liftable.\n"
    )


def _render_epic3_coverage_report() -> str:
    return (
        "# Corpus Coverage Report\n\n"
        "The selected corpus manifest, the frontier table, the illumination map, and the failure-region atlas are read together. The corpus engine is valuable only if it covers the declared design space, rejects invalid or leaky cases, and yields better downstream study actions.\n"
    )


def _render_epic3_claim_boundary() -> str:
    return (
        "# Epic 3 Claim Boundary\n\n"
        "Epic 3 proves that corpus exploration can be governed as experimental design. It does not prove that any one backend, including CEM or PPO, is yet the generally superior exploration policy.\n"
    )


def _render_epic3_validation_report() -> str:
    return (
        "# Epic 3 Validation Report\n\n"
        "- corpus objective present\n"
        "- selected corpus manifest and frontier present\n"
        "- every selected hard case stays valid and non-leaky\n"
        "- CEM/PPO retain baseline, ablation, seed stability, and downstream-yield gates\n"
        "- hard-case route ledger and failure-region atlas present\n"
    )


def _render_epic3_manifest() -> str:
    return (
        "packet_id: 03_corpus_explorer_design_engine\n"
        "source_packet: artifacts/packets/corpus_explorer_mvp\n"
        "validator: corpus_explorer_mvp validator plus V7 atlas/ledger checks\n"
    )
