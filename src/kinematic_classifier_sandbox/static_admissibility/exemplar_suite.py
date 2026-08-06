from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from matplotlib import patches
from matplotlib.colors import ListedColormap
from numpy import array

from kinematic_classifier_sandbox.reports.markdown import MarkdownDocument
from kinematic_classifier_sandbox.utils.io import _copy_file, _write_text, write_csv
from kinematic_classifier_sandbox.utils.plotting import plt
from kinematic_classifier_sandbox.utils.runtime import repo_root

from .io import build_static_admissibility_result, write_static_admissibility_packet
from .schemas import load_static_admissibility_config
from .validation import validate_static_admissibility_packet

DEFAULT_SUITE_MANIFEST = (
    repo_root() / "experiments" / "static_admissibility" / "epic1_exemplar_suite.yaml"
)


def _repo_relative_text(path: Path) -> str:
    try:
        return str(path.relative_to(repo_root()))
    except ValueError:
        return str(path)


@dataclass(frozen=True, slots=True)
class StaticAdmissibilityExemplarSuitePacket:
    packet_dir: Path
    readme_path: Path
    quickstart_path: Path
    packet_manifest_path: Path
    decision_card_path: Path
    validation_report_path: Path
    claim_boundary_path: Path
    hero_chart_manifest_path: Path
    lane_proof_matrix_path: Path
    automated_brief_path: Path
    executive_brief_path: Path
    source_manifest_path: Path
    route_matrix_path: Path
    fingerprint_scores_path: Path
    card_manifest_path: Path


def write_static_admissibility_exemplar_suite_packet(
    output_dir: str | Path,
    *,
    suite_manifest_path: str | Path = DEFAULT_SUITE_MANIFEST,
) -> StaticAdmissibilityExemplarSuitePacket:
    packet_dir = Path(output_dir)
    figures_dir = packet_dir / "figures"
    source_artifacts_dir = packet_dir / "source_artifacts"
    source_bundles_dir = packet_dir / "source_bundles"
    exemplar_cards_dir = packet_dir / "exemplar_cards"
    latex_dir = packet_dir / "latex"
    source_runs_dir = packet_dir / "source_runs"
    for path in (
        packet_dir,
        figures_dir,
        source_artifacts_dir,
        source_bundles_dir,
        exemplar_cards_dir,
        latex_dir,
        source_runs_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    suite_manifest_path = Path(suite_manifest_path)
    manifest_payload = yaml.safe_load(suite_manifest_path.read_text(encoding="utf-8")) or {}
    exemplars = tuple(manifest_payload.get("epic1_static_admissibility_exemplars", ()))
    if not exemplars:
        raise ValueError("Epic 1 exemplar suite manifest does not declare any exemplars")

    suite_rows: list[dict[str, object]] = []
    route_rows: list[dict[str, object]] = []
    fingerprint_rows: list[dict[str, object]] = []
    card_rows: list[dict[str, object]] = []
    executive_rows: list[dict[str, object]] = []
    validation_lines = ["# Validation Report", ""]

    for exemplar in exemplars:
        exemplar_id = str(exemplar["exemplar_id"])
        config_path = suite_manifest_path.parent / str(exemplar["config"])
        config = load_static_admissibility_config(config_path)
        result = build_static_admissibility_result(config)
        source_packet = write_static_admissibility_packet(
            source_runs_dir / exemplar_id,
            config=config,
            result=result,
        )
        issues = validate_static_admissibility_packet(source_packet.packet_dir, repo_root=repo_root())
        actual_route = str(result.static_decision["status"])
        expected_route = str(exemplar["expected_status"])
        validator_status = "pass" if not issues and expected_route == actual_route else "fail"
        primary_diagnostic, secondary_diagnostic = _diagnostic_labels(exemplar_id)
        suite_rows.append(
            {
                "exemplar_id": exemplar_id,
                "bundle_path": str(config_path),
                "purpose": str(exemplar.get("matrix_constitutes", "")),
                "input_signature": str(exemplar.get("input_signature", "")),
                "expected_route": expected_route,
                "actual_route": actual_route,
                "validator_status": validator_status,
                "primary_diagnostic": primary_diagnostic,
                "secondary_diagnostic": secondary_diagnostic,
                "resolution_codes": "|".join(
                    str(code) for code in result.static_decision.get("resolution_codes", ())
                ),
                "recommended_resolution": _first_next_action(result),
            }
        )
        route_rows.append(
            {
                "exemplar_id": exemplar_id,
                "class_separability_status": _lane_status(result, "class separability"),
                "feature_relevance_status": _lane_status(result, "feature relevance"),
                "redundancy_status": _lane_status(result, "feature redundancy"),
                "synergy_status": _lane_status(result, "feature synergy"),
                "prior_pathology_status": _lane_status(result, "prior pathology"),
                "prior_selection_balance_status": _lane_status(result, "prior selection balance"),
                "coverage_status": _lane_status(result, "coverage feasibility"),
                "leakage_status": _lane_status(result, "leakage risk"),
                "expected_route": expected_route,
                "actual_route": actual_route,
                "validator_result": validator_status,
            }
        )
        fingerprint_rows.append(
            {
                "exemplar_id": exemplar_id,
                "confusability_score": _confusability_score(result),
                "prior_pathology_score": _prior_pathology_score(result),
                "redundancy_score": _redundancy_score(result),
                "synergy_candidate_score": _synergy_score(result),
                "coverage_thinness_score": _coverage_score(result),
                "leakage_risk_score": _leakage_score(result),
                "prior_selection_skew_score": _prior_selection_score(result),
                "decisionability_score": _decisionability_score(actual_route),
            }
        )
        _copy_bundle_sources(source_bundles_dir / exemplar_id, config_path, config)
        _copy_exemplar_source_artifacts(source_artifacts_dir / exemplar_id, source_packet.packet_dir)
        card_md_path = exemplar_cards_dir / f"{exemplar_id}.md"
        card_png_path = figures_dir / _card_filename(exemplar_id)
        _write_text(card_md_path, _render_exemplar_card_markdown(exemplar, result, validator_status))
        _render_exemplar_card_figure(
            card_png_path,
            exemplar=exemplar,
            result=result,
            validator_status=validator_status,
        )
        card_rows.append(
            {
                "exemplar_id": exemplar_id,
                "card_png": str(card_png_path.relative_to(packet_dir)),
                "card_md": str(card_md_path.relative_to(packet_dir)),
                "source_bundle": str((source_bundles_dir / exemplar_id).relative_to(packet_dir)),
                "source_artifacts": str((source_artifacts_dir / exemplar_id).relative_to(packet_dir)),
                "claim": "Epic 1 routes this study family before corpus search or classifier escalation.",
                "limitation": "Static admissibility is an early gate; it does not prove downstream dynamic performance.",
                "next_action": _first_next_action(result),
                "resolution_codes": "|".join(
                    str(code) for code in result.static_decision.get("resolution_codes", ())
                ),
            }
        )
        executive_rows.append(
            _executive_evidence_row(
                exemplar_id=exemplar_id,
                result=result,
                validator_status=validator_status,
            )
        )
        validation_lines.append(
            f"- `{exemplar_id}`: expected `{expected_route}`, observed `{actual_route}`, validator `{validator_status}`"
        )
        for issue in issues:
            validation_lines.append(f"  - issue: {issue}")

    source_manifest_path = source_artifacts_dir / "exemplar_suite_manifest.csv"
    route_matrix_path = source_artifacts_dir / "exemplar_route_matrix.csv"
    fingerprint_scores_path = source_artifacts_dir / "exemplar_fingerprint_scores.csv"
    card_manifest_path = source_artifacts_dir / "exemplar_card_manifest.csv"
    write_csv(source_manifest_path, suite_rows, list(suite_rows[0].keys()))
    write_csv(route_matrix_path, route_rows, list(route_rows[0].keys()))
    write_csv(fingerprint_scores_path, fingerprint_rows, list(fingerprint_rows[0].keys()))
    write_csv(card_manifest_path, card_rows, list(card_rows[0].keys()))

    _render_bundle_ingestion_spine(figures_dir / "02a_static_bundle_ingestion_spine.png")
    _render_routing_matrix(figures_dir / "02a_static_exemplar_suite_routing_matrix.png", route_rows)
    _render_suite_decision_card(figures_dir / "02b_static_audit_decision_card.png", suite_rows)
    _render_fingerprint_strip(figures_dir / "02m_static_exemplar_fingerprint_strip.png", fingerprint_rows, route_rows)
    _render_action_router(figures_dir / "02k_static_audit_to_action_router.png")

    readme_path = packet_dir / "README.md"
    quickstart_path = packet_dir / "quickstart.md"
    packet_manifest_path = packet_dir / "packet_manifest.yaml"
    decision_card_path = packet_dir / "decision_card.md"
    validation_report_path = packet_dir / "validation_report.md"
    claim_boundary_path = packet_dir / "claim_boundary.md"
    hero_chart_manifest_path = packet_dir / "hero_chart_manifest.csv"
    lane_proof_matrix_path = packet_dir / "lane_proof_matrix.md"
    automated_brief_path = packet_dir / "automated_brief.md"
    executive_brief_path = packet_dir / "executive_brief.md"

    _write_text(readme_path, _render_suite_readme())
    _write_text(quickstart_path, _render_suite_quickstart())
    _write_text(decision_card_path, _render_suite_decision_card_markdown(suite_rows, route_rows))
    _write_text(validation_report_path, "\n".join(validation_lines) + "\n")
    _write_text(claim_boundary_path, _render_claim_boundary())
    _write_text(lane_proof_matrix_path, _render_suite_lane_proof_matrix())
    _write_text(automated_brief_path, _render_suite_automated_brief(suite_rows, route_rows))
    _write_text(
        executive_brief_path,
        _render_suite_executive_brief(suite_rows, route_rows, executive_rows),
    )
    _write_text(
        latex_dir / "static_admissibility_exemplar_suite.tex",
        _render_suite_latex(route_rows),
    )
    hero_rows = _hero_chart_manifest_rows()
    write_csv(hero_chart_manifest_path, hero_rows, list(hero_rows[0].keys()))
    packet_manifest_path.write_text(
        yaml.safe_dump(
            {
                "packet_id": "01_static_admissibility",
                "claim": "The exemplar suite validates the Epic 1 routing surface over file-backed study bundles.",
                "suite_manifest": _repo_relative_text(suite_manifest_path),
                "source_tables": [
                    str(source_manifest_path.relative_to(packet_dir)),
                    str(route_matrix_path.relative_to(packet_dir)),
                    str(fingerprint_scores_path.relative_to(packet_dir)),
                    str(card_manifest_path.relative_to(packet_dir)),
                ],
                "executive_brief": str(executive_brief_path.relative_to(packet_dir)),
                "figure_manifest": str(hero_chart_manifest_path.relative_to(packet_dir)),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return StaticAdmissibilityExemplarSuitePacket(
        packet_dir=packet_dir,
        readme_path=readme_path,
        quickstart_path=quickstart_path,
        packet_manifest_path=packet_manifest_path,
        decision_card_path=decision_card_path,
        validation_report_path=validation_report_path,
        claim_boundary_path=claim_boundary_path,
        hero_chart_manifest_path=hero_chart_manifest_path,
        lane_proof_matrix_path=lane_proof_matrix_path,
        automated_brief_path=automated_brief_path,
        executive_brief_path=executive_brief_path,
        source_manifest_path=source_manifest_path,
        route_matrix_path=route_matrix_path,
        fingerprint_scores_path=fingerprint_scores_path,
        card_manifest_path=card_manifest_path,
    )


def _lane_status(result, lane: str) -> str:
    row = next(item for item in result.decision_card_rows if str(item["lane"]) == lane)
    status = str(row["status"])
    if status in {"pass", "promote", "promote_to_corpus_explorer"}:
        return "pass"
    if status in {"warning", "warn"}:
        return "warn"
    if status == "candidate":
        return "candidate"
    if status in {"blocker", "reject", "revise_class_set", "revise_prior", "revise_feature_set"}:
        return "block"
    return "not_applicable"


def _confusability_score(result) -> float:
    return max(0.0, 1.0 - min(float(row["pairwise_auc"]) for row in result.class_pair_rows))


def _prior_pathology_score(result) -> float:
    rows = tuple(result.prior_pathology_rows)
    blockers = sum(1 for row in rows if row["pathology_flag"] == "prior_domination")
    return blockers / max(len(rows), 1)


def _redundancy_score(result) -> float:
    rows = tuple(result.feature_redundancy_rows)
    flagged = sum(1 for row in rows if row["status"] == "high_redundancy")
    return flagged / max(len(rows), 1)


def _synergy_score(result) -> float:
    rows = tuple(result.feature_synergy_rows)
    flagged = sum(1 for row in rows if row["status"] == "synergy_candidate")
    return flagged / max(len(rows), 1)


def _coverage_score(result) -> float:
    rows = tuple(result.coverage_rows)
    flagged = sum(1 for row in rows if row["status"] == "low_count")
    return flagged / max(len(rows), 1)


def _prior_selection_score(result) -> float:
    rows = tuple(result.prior_selection_rows)
    flagged = sum(
        1
        for row in rows
        if row["status"]
        in {"never_selected_on_observed_surface", "rarely_selected", "underselected_for_own_samples"}
    )
    return flagged / max(len(rows), 1)


def _leakage_score(result) -> float:
    rows = tuple(result.leakage_rows)
    flagged = sum(1 for row in rows if row["status"] == "blocker")
    return flagged / max(len(rows), 1)


def _decisionability_score(actual_route: str) -> float:
    if actual_route == "promote_to_corpus_explorer":
        return 1.0
    if actual_route in {"revise_class_set", "revise_prior", "revise_feature_set"}:
        return 0.5
    return 0.0


def _diagnostic_labels(exemplar_id: str) -> tuple[str, str]:
    mapping = {
        "promote_separable_family": ("class separability", "feature relevance"),
        "class_overlap_boundary_family": ("class overlap boundary", "confusability"),
        "prior_domination_family": ("prior pathology", "flip thresholds"),
        "future_class_surface_family": ("future class pruning", "expected signature collision"),
        "redundancy_synergy_family": ("feature redundancy", "candidate synergy"),
        "coverage_thin_cells_family": ("coverage feasibility", "decisionability"),
        "leakage_blocker_family": ("leakage provenance", "hard gate"),
    }
    return mapping.get(exemplar_id, ("static admissibility", "routing"))


def _card_filename(exemplar_id: str) -> str:
    return f"E1_card_{exemplar_id}.png"


def _copy_bundle_sources(destination: Path, config_path: Path, config) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    _copy_file(config_path, destination / "static_audit_bundle.yaml")
    if config.input_bundle is None:
        return
    _copy_file(config.input_bundle.sample_table_path, destination / "samples.csv")
    if config.input_bundle.feature_schema_path is not None:
        _copy_file(config.input_bundle.feature_schema_path, destination / "feature_schema.csv")
    if config.input_bundle.class_schema_path is not None:
        _copy_file(config.input_bundle.class_schema_path, destination / "class_schema.csv")
    if config.input_bundle.class_feature_signature_path is not None:
        _copy_file(
            config.input_bundle.class_feature_signature_path,
            destination / "class_feature_signature.csv",
        )


def _copy_exemplar_source_artifacts(destination: Path, source_packet_dir: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in (
        "class_confusability_matrix.csv",
        "class_pair_diagnostics.csv",
        "class_feature_signature.csv",
        "class_observability.csv",
        "feature_relevance_table.csv",
        "feature_redundancy_matrix.csv",
        "feature_alias_candidates.csv",
        "feature_synergy_candidates.csv",
        "prior_pathology_report.csv",
        "prior_selection_balance.csv",
        "prior_flip_thresholds.csv",
        "static_resolution_plan.csv",
        "static_coverage_feasibility.csv",
        "static_leakage_provenance_audit.csv",
        "02c_class_pair_confusability_matrix.png",
        "02d_feature_relevance_rank.png",
        "02e_feature_redundancy_graph.png",
        "02f_feature_synergy_map.png",
        "02g_prior_pathology_surface.png",
        "02h_prior_flip_thresholds.png",
        "02i_static_coverage_feasibility.png",
        "02j_static_leakage_provenance_audit.png",
    ):
        source = source_packet_dir / name
        if source.exists():
            _copy_file(source, destination / name)


def _render_bundle_ingestion_spine(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")
    boxes = [
        (0.04, 0.55, 0.18, 0.22, "static_audit_bundle.yaml"),
        (0.04, 0.18, 0.18, 0.22, "samples.csv\nfeature_schema.csv\nclass_schema.csv"),
        (0.34, 0.36, 0.22, 0.28, "StaticAuditSample /\nFeatureSchema /\nClassSchema"),
        (0.64, 0.36, 0.22, 0.28, "Static Feature /\nClass / Prior Audit"),
        (0.90, 0.55, 0.08, 0.22, "decision\ncard"),
        (0.90, 0.18, 0.08, 0.22, "figures /\nsource /\nvalidation"),
    ]
    for x, y, w, h, label in boxes:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", facecolor="#F7F9FB", edgecolor="#355C7D", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11, fontweight="bold")
    for start, end in [((0.22, 0.66), (0.34, 0.50)), ((0.22, 0.29), (0.34, 0.50)), ((0.56, 0.50), (0.64, 0.50)), ((0.86, 0.57), (0.90, 0.66)), ((0.86, 0.43), (0.90, 0.29))]:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "linewidth": 2.0, "color": "#355C7D"})
    ax.set_title("Epic 1 now accepts portable feature/class/prior study bundles", loc="left", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _render_routing_matrix(path: Path, route_rows: list[dict[str, object]]) -> None:
    columns = [
        "class_separability_status",
        "feature_relevance_status",
        "redundancy_status",
        "synergy_status",
        "prior_pathology_status",
        "prior_selection_balance_status",
        "coverage_status",
        "leakage_status",
    ]
    status_map = {"not_applicable": 0, "pass": 1, "warn": 2, "candidate": 3, "block": 4}
    matrix = array([[status_map[str(row[column])] for column in columns] for row in route_rows], dtype=float)
    cmap = ListedColormap(["#E5E7EB", "#B8E0D2", "#F6D186", "#9CC2FF", "#F28482"])
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=4)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(
        ["class sep", "feature rel", "redundancy", "synergy", "prior", "selection", "coverage", "leakage"],
        rotation=20,
        ha="right",
    )
    ax.set_yticks(range(len(route_rows)))
    ax.set_yticklabels([str(row["exemplar_id"]) for row in route_rows])
    for row_index, row in enumerate(route_rows):
        for col_index, column in enumerate(columns):
            ax.text(col_index, row_index, str(row[column])[:4], ha="center", va="center", fontsize=8, fontweight="bold")
        ax.text(len(columns) + 0.1, row_index, str(row["expected_route"]), va="center", fontsize=9)
        ax.text(len(columns) + 2.3, row_index, str(row["actual_route"]), va="center", fontsize=9)
        ax.text(len(columns) + 4.4, row_index, str(row["validator_result"]), va="center", fontsize=9, fontweight="bold")
    ax.text(len(columns) + 0.1, -0.9, "expected route", fontsize=9, fontweight="bold")
    ax.text(len(columns) + 2.3, -0.9, "actual route", fontsize=9, fontweight="bold")
    ax.text(len(columns) + 4.4, -0.9, "validator", fontsize=9, fontweight="bold")
    ax.set_xlim(-0.5, len(columns) + 5.5)
    ax.set_title("Each exemplar teaches one static-admissibility route", loc="left", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _render_suite_decision_card(path: Path, suite_rows: list[dict[str, object]]) -> None:
    route_counts: dict[str, int] = {}
    validator_pass = 0
    for row in suite_rows:
        route = str(row["actual_route"])
        route_counts[route] = route_counts.get(route, 0) + 1
        if row["validator_status"] == "pass":
            validator_pass += 1
    lines = [
        ("bundle ingestion", "PASS", "file-backed suite manifest", "portable input path"),
        ("route correctness", "PASS" if validator_pass == len(suite_rows) else "WARN", f"{validator_pass}/{len(suite_rows)} validators", "check expected routes"),
        ("promote family", str(route_counts.get("promote_to_corpus_explorer", 0)), "admissible bundles", "promote to corpus explorer"),
        ("revise class family", str(route_counts.get("revise_class_set", 0)), "overlap bundles", "tighten class definitions"),
        ("revise prior family", str(route_counts.get("revise_prior", 0)), "prior-dominated bundles", "sweep priors"),
        ("reject family", str(route_counts.get("reject", 0)), "leakage bundles", "block study"),
    ]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")
    ax.set_title("The static audit decides before classifiers run", loc="left", fontsize=15, fontweight="bold")
    col_x = [0.02, 0.36, 0.54, 0.78]
    headers = ["row", "status", "worst case", "next action"]
    for index, header in enumerate(headers):
        ax.text(col_x[index], 0.92, header, fontsize=11, fontweight="bold")
    y = 0.82
    for lane, status, worst, action in lines:
        ax.text(col_x[0], y, lane, fontsize=11)
        ax.text(col_x[1], y, status, fontsize=11, fontweight="bold")
        ax.text(col_x[2], y, worst, fontsize=11)
        ax.text(col_x[3], y, action, fontsize=11)
        y -= 0.12
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _render_fingerprint_strip(path: Path, fingerprint_rows: list[dict[str, object]], route_rows: list[dict[str, object]]) -> None:
    columns = [
        "confusability_score",
        "prior_pathology_score",
        "redundancy_score",
        "synergy_candidate_score",
        "coverage_thinness_score",
        "leakage_risk_score",
        "prior_selection_skew_score",
        "decisionability_score",
    ]
    matrix = array([[float(row[column]) for column in columns] for row in fingerprint_rows], dtype=float)
    fig, ax = plt.subplots(figsize=(15, 5))
    heat = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(
        ["confusability", "prior", "redundancy", "synergy", "coverage", "leakage", "selection", "decision"],
        rotation=20,
        ha="right",
    )
    ax.set_yticks(range(len(fingerprint_rows)))
    ax.set_yticklabels([str(row["exemplar_id"]) for row in fingerprint_rows])
    for row_index, row in enumerate(fingerprint_rows):
        for col_index, column in enumerate(columns):
            ax.text(col_index, row_index, f"{float(row[column]):.2f}", ha="center", va="center", fontsize=8)
        ax.text(len(columns) + 0.2, row_index, str(route_rows[row_index]["actual_route"]), va="center", fontsize=9)
    ax.text(len(columns) + 0.2, -0.8, "route", fontsize=9, fontweight="bold")
    ax.set_xlim(-0.5, len(columns) + 2.2)
    ax.set_title("Each static exemplar has a distinct diagnostic fingerprint", loc="left", fontsize=15, fontweight="bold")
    fig.colorbar(heat, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _render_action_router(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")
    routes = [
        ("class overlap", "revise class set"),
        ("feature blindness", "revise feature set"),
        ("redundancy", "cluster/drop/regularize"),
        ("candidate synergy", "ablation TODO"),
        ("prior domination", "revise prior / sweep prior"),
        ("thin coverage", "Corpus Explorer objective"),
        ("leakage", "block study"),
        ("clean admissible", "promote to Corpus Explorer / ladder"),
    ]
    y = 0.88
    for left, right in routes:
        left_rect = patches.FancyBboxPatch((0.06, y - 0.06), 0.30, 0.09, boxstyle="round,pad=0.02", facecolor="#F7F9FB", edgecolor="#355C7D")
        right_rect = patches.FancyBboxPatch((0.62, y - 0.06), 0.30, 0.09, boxstyle="round,pad=0.02", facecolor="#FFF7E6", edgecolor="#C06C84")
        ax.add_patch(left_rect)
        ax.add_patch(right_rect)
        ax.text(0.21, y - 0.015, left, ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(0.77, y - 0.015, right, ha="center", va="center", fontsize=11, fontweight="bold")
        ax.annotate("", xy=(0.62, y - 0.015), xytext=(0.36, y - 0.015), arrowprops={"arrowstyle": "->", "linewidth": 1.8, "color": "#355C7D"})
        y -= 0.10
    ax.set_title("Static findings route to actions before compute-heavy work", loc="left", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _render_exemplar_card_markdown(exemplar: dict[str, object], result, validator_status: str) -> str:
    doc = MarkdownDocument(f"Epic 1 Exemplar: {exemplar['exemplar_id']}")
    doc.paragraph("Single exemplar card for the static admissibility teaching/proof atlas.")
    doc.bullet_list(
        [
            f"study_bundle: `{exemplar['config']}`",
            f"input signature: {exemplar.get('input_signature', '')}",
            f"primary diagnostic: {_diagnostic_labels(str(exemplar['exemplar_id']))[0]}",
            f"secondary diagnostic: {_diagnostic_labels(str(exemplar['exemplar_id']))[1]}",
            f"expected route: `{exemplar['expected_status']}`",
            f"actual route: `{result.static_decision['status']}`",
            f"validator result: `{validator_status}`",
            f"resolution codes: `{', '.join(str(code) for code in result.static_decision.get('resolution_codes', ()))}`",
            "limitation: static screen only; not a downstream classifier benchmark",
            f"next action: {_first_next_action(result)}",
        ]
    )
    return doc.text() + "\n"


def _render_exemplar_card_figure(path: Path, *, exemplar: dict[str, object], result, validator_status: str) -> None:
    fig = plt.figure(figsize=(14, 8))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.2, 0.9])
    ax_left = fig.add_subplot(grid[0, 0])
    ax_center = fig.add_subplot(grid[0, 1])
    ax_right = fig.add_subplot(grid[0, 2])
    for ax in (ax_left, ax_center, ax_right):
        ax.axis("off")
    ax_left.text(0.0, 0.95, str(exemplar["exemplar_id"]), fontsize=15, fontweight="bold", va="top")
    ax_left.text(
        0.0,
        0.80,
        "\n".join(
            [
                f"bundle: {exemplar['config']}",
                f"signature: {exemplar.get('input_signature', '')}",
                f"expected: {exemplar['expected_status']}",
                f"actual: {result.static_decision['status']}",
            ]
        ),
        fontsize=11,
        va="top",
    )
    rows = list(result.decision_card_rows)
    labels = [str(row["lane"]) for row in rows]
    values = [1.0 if str(row["status"]) == "pass" else 0.6 if str(row["status"]) in {"warning", "candidate"} else 0.2 for row in rows]
    colors = ["#5ABF90" if value == 1.0 else "#F6D186" if value == 0.6 else "#F28482" for value in values]
    ax_center.barh(range(len(labels)), values, color=colors)
    ax_center.set_yticks(range(len(labels)))
    ax_center.set_yticklabels(labels, fontsize=9)
    ax_center.set_xlim(0.0, 1.05)
    ax_center.set_title("Diagnostic profile", fontsize=12, fontweight="bold")
    ax_center.grid(True, axis="x", alpha=0.25)
    ax_right.text(0.0, 0.95, "Audit route", fontsize=14, fontweight="bold", va="top")
    ax_right.text(
        0.0,
        0.78,
        "\n".join(
            [
                f"primary: {_diagnostic_labels(str(exemplar['exemplar_id']))[0]}",
                f"secondary: {_diagnostic_labels(str(exemplar['exemplar_id']))[1]}",
                f"validator: {validator_status}",
                f"next: {_first_next_action(result)}",
            ]
        ),
        fontsize=11,
        va="top",
    )
    fig.suptitle("Epic 1 exemplar card", fontsize=16, fontweight="bold", x=0.05, ha="left")
    fig.subplots_adjust(left=0.03, right=0.98, top=0.90, bottom=0.06, wspace=0.18)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _first_next_action(result) -> str:
    recommendations = tuple(
        str(row["recommended_action"])
        for row in result.resolution_rows
        if row["severity"] != "info"
    )
    if recommendations:
        return recommendations[0]
    next_work = tuple(result.static_decision.get("next_work", ()))
    return str(next_work[0]) if next_work else "promote"


def _executive_evidence_row(
    *,
    exemplar_id: str,
    result,
    validator_status: str,
) -> dict[str, object]:
    resolution_codes = tuple(
        str(code) for code in result.static_decision.get("resolution_codes", ())
    )
    hard_pairs = tuple(row for row in result.class_pair_rows if row["status"] == "hard")
    expected_collisions = tuple(
        row
        for row in result.class_pair_rows
        if row["expected_signature_collision_status"] == "expected_exact_signature_collision"
    )
    unobserved = tuple(
        row for row in result.class_observability_rows if row["status"] == "unobserved_class"
    )
    rare_selection = tuple(
        row
        for row in result.prior_selection_rows
        if row["status"] in {"never_selected_on_observed_surface", "rarely_selected"}
    )
    redundant = tuple(
        row for row in result.feature_redundancy_rows if row["status"] == "high_redundancy"
    )
    synergy = tuple(
        row
        for row in result.feature_synergy_rows
        if row["status"] == "synergy_candidate"
    )
    thin = tuple(row for row in result.coverage_rows if row["status"] == "low_count")
    leakage = tuple(row for row in result.leakage_rows if row["status"] == "blocker")

    if "LEAKAGE_BLOCKER" in resolution_codes:
        affected = ", ".join(str(row["feature"]) for row in leakage)
        finding = f"{len(leakage)} leakage blocker(s): {affected}"
        prevents = "Invalid or future-dependent evidence from reaching classifier work"
    elif expected_collisions or unobserved:
        collision_pairs = ", ".join(
            f"{row['class_a']} vs {row['class_b']}" for row in expected_collisions
        )
        unobserved_classes = ", ".join(str(row["class_name"]) for row in unobserved)
        details = []
        if collision_pairs:
            details.append(f"expected collision: {collision_pairs}")
        if unobserved_classes:
            details.append(f"unobserved: {unobserved_classes}")
        finding = "; ".join(details)
        prevents = "Future classes that cannot be justified or selected from the current surface"
    elif hard_pairs:
        hardest = min(hard_pairs, key=lambda row: float(row["pairwise_auc"]))
        finding = (
            f"hard pair {hardest['class_a']} vs {hardest['class_b']} "
            f"(pairwise AUC {float(hardest['pairwise_auc']):.3f})"
        )
        prevents = "Non-decisionable class boundaries from expanding the search and label space"
    elif "PRIOR_DOMINATION" in resolution_codes or rare_selection:
        rare_details = ", ".join(
            f"{row['class_name']} own-selection {float(row['true_class_selection_rate']):.3f}"
            for row in rare_selection
        )
        finding = f"prior/evidence imbalance; {rare_details or 'prior domination detected'}"
        prevents = "Downstream work on prior regimes where retained classes are effectively never selected"
    elif redundant:
        redundant_pairs = ", ".join(
            f"{row['feature_a']} vs {row['feature_b']}" for row in redundant
        )
        finding = (
            f"{len(redundant)} redundant feature pair(s): {redundant_pairs}; "
            f"{len(synergy)} joint-evidence candidate pair(s)"
        )
        prevents = "Duplicate dimensions and unconfirmed interactions from inflating the feature search"
    elif thin:
        finding = f"{len(thin)} class-feature cells have low witness counts"
        prevents = "Broad corpus search over under-witnessed regions that should become targeted objectives"
    else:
        finding = "No hard blocker; the declared surface is admissible for the next lane"
        prevents = "Premature rejection while preserving explicit warnings for follow-up"

    primary_diagnostic, _ = _diagnostic_labels(exemplar_id)
    return {
        "exemplar_id": exemplar_id,
        "diagnostic": primary_diagnostic,
        "finding": finding,
        "prevents": prevents,
        "route": str(result.static_decision["status"]),
        "resolution_codes": "|".join(resolution_codes) or "CLEAN_ADMISSIBLE_SURFACE",
        "recommended_resolution": _first_next_action(result),
        "validator_status": validator_status,
    }


def _render_suite_readme() -> str:
    return "\n".join(
        [
            "# Epic 1 Static Admissibility Validation Packet",
            "",
            "This packet is the exemplar atlas for Epic 1. It uses seven file-backed study bundles to exercise the static-admissibility routing surface.",
            "",
            "## Main deck figures",
            "",
            "- `figures/02a_static_bundle_ingestion_spine.png`",
            "- `figures/02a_static_exemplar_suite_routing_matrix.png`",
            "- `figures/02b_static_audit_decision_card.png`",
            "- `figures/02m_static_exemplar_fingerprint_strip.png`",
            "",
            "## Appendix figures",
            "",
            "- `figures/E1_card_*.png`",
            "- `figures/02k_static_audit_to_action_router.png`",
            "",
            "## Programmatic recommendations",
            "",
            "Each source run includes `static_resolution_plan.csv` and `prior_selection_balance.csv`; exemplar cards and manifests quote their generated issue codes and first recommended action.",
            "",
            "## Executive showcase",
            "",
            "Start with `executive_brief.md` for a lead-facing explanation of what the tool eliminates from feature, class, prior, and corpus-search space before classifier work.",
            "",
        ]
    ) + "\n"


def _render_suite_quickstart() -> str:
    return "\n".join(
        [
            "# Quickstart",
            "",
            "Build the Epic 1 validation packet:",
            "",
            "```bash",
            "PYTHONPATH=src python3 -m kinematic_classifier_sandbox run-static-audit-suite \\",
            "  experiments/static_admissibility/epic1_exemplar_suite.yaml \\",
            "  --output-dir artifacts/validation_packets/01_static_admissibility",
            "```",
            "",
            "Validate it:",
            "",
            "```bash",
            "PYTHONPATH=src python3 -m kinematic_classifier_sandbox validate-packet \\",
            "  artifacts/validation_packets/01_static_admissibility",
            "```",
            "",
            "This packet is a static feature/class/prior validation atlas, not a classifier benchmark.",
        ]
    ) + "\n"


def _render_suite_decision_card_markdown(
    suite_rows: list[dict[str, object]],
    route_rows: list[dict[str, object]],
) -> str:
    route_counts: dict[str, int] = {}
    for row in suite_rows:
        route = str(row["actual_route"])
        route_counts[route] = route_counts.get(route, 0) + 1
    doc = MarkdownDocument("Epic 1 Exemplar Suite Decision Card")
    doc.bullet_list(
        [
            f"exemplar count: `{len(suite_rows)}`",
            f"promote routes: `{route_counts.get('promote_to_corpus_explorer', 0)}`",
            f"class revision routes: `{route_counts.get('revise_class_set', 0)}`",
            f"prior revision routes: `{route_counts.get('revise_prior', 0)}`",
            f"reject routes: `{route_counts.get('reject', 0)}`",
            f"route-matrix rows: `{len(route_rows)}`",
        ]
    )
    return doc.text() + "\n"


def _render_suite_automated_brief(
    suite_rows: list[dict[str, object]],
    route_rows: list[dict[str, object]],
) -> str:
    lines = [
        "# Epic 1 Static Admissibility Validation Packet",
        "",
        "This packet validates the Epic 1 routing surface over portable, file-backed study bundles.",
        "",
        "It is designed to answer one question:",
        "",
        "Can a proposed feature/class/prior study be screened before corpus search or classifier work?",
        "",
        "## Exemplar families",
        "",
    ]
    lines.extend(
        f"- `{row['exemplar_id']}` -> expected `{row['expected_route']}` / actual `{row['actual_route']}`"
        for row in suite_rows
    )
    lines.extend(["", "## Route coverage", ""])
    lines.extend(
        f"- `{row['exemplar_id']}`: class `{row['class_separability_status']}`, prior `{row['prior_pathology_status']}`, selection `{row['prior_selection_balance_status']}`, leakage `{row['leakage_status']}`"
        for row in route_rows
    )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "- This is a static admissibility validation atlas.",
            "- It does not prove downstream classifier or filter performance.",
            "- Candidate synergy remains candidate until ablation-backed.",
            "- Resolution recommendations are generated from the diagnostic tables; exemplar metadata does not prescribe the fix.",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_suite_executive_brief(
    suite_rows: list[dict[str, object]],
    route_rows: list[dict[str, object]],
    evidence_rows: list[dict[str, object]],
) -> str:
    route_counts: dict[str, int] = {}
    issue_counts: dict[str, int] = {}
    for row in suite_rows:
        route = str(row["actual_route"])
        route_counts[route] = route_counts.get(route, 0) + 1
        for code in str(row["resolution_codes"]).split("|"):
            if code:
                issue_counts[code] = issue_counts.get(code, 0) + 1

    lines = [
        "# Static Admissibility — Executive Showcase",
        "",
        "## The pitch",
        "",
        "Static Admissibility is a quick-turn gate for a proposed feature/class/prior study. It runs the same declared bundle through evidence checks before we spend corpus-search, classifier-training, or reinforcement-learning effort.",
        "",
        "The practical question is: **what can we eliminate or repair before the expensive problem begins?**",
        "",
        "## What it screens upfront",
        "",
        "| surface | automatic questions | upfront value |",
        "| --- | --- | --- |",
        "| Feature space | Are features leaky, unavailable, weak, duplicated, dependent, or only jointly useful? | Remove invalid or redundant dimensions and turn synergy into an explicit ablation question. |",
        "| Class space | Are class pairs confusable, exactly colliding, unobserved, or not selectable from the current surface? | Merge, split, redefine, or prune classes before broad corpus exploration. |",
        "| Prior space | Can declared priors overwhelm the available evidence, or cause a class to be selected almost never? | Rebalance priors, add witness evidence, or remove classes that are not decisionable under the regime. |",
        "| Corpus-search space | Are class-feature regions thinly covered? | Send targeted coverage objectives to Corpus Explorer instead of searching blindly. |",
        "",
        "## The generated evidence atlas",
        "",
        f"This suite contains `{len(suite_rows)}` file-backed examples: `{route_counts.get('promote_to_corpus_explorer', 0)}` promotion route(s), `{route_counts.get('revise_class_set', 0)}` class-set revision route(s), `{route_counts.get('revise_prior', 0)}` prior revision route(s), and `{route_counts.get('reject', 0)}` hard rejection route(s).",
        "",
        "| example | static finding | what it eliminates or changes | generated route |",
        "| --- | --- | --- | --- |",
    ]
    for row in evidence_rows:
        lines.append(
            f"| `{row['exemplar_id']}` | {row['finding']} | {row['prevents']} | `{row['route']}` |"
        )

    lines.extend(
        [
            "",
            "## Why the prior is part of the same audit",
            "",
            "The prior is not a separate spreadsheet review. The declared prior regime is applied to the same feature/class surface, so the report can show when a class looks separable in principle but is still selected rarely or never under the proposed operating assumptions.",
            "",
            "The prior-selection table is a Gaussian feature/prior proxy—not a deployed classifier confusion matrix. Its purpose is to expose a bad regime early and make the next action explicit.",
            "",
            "## A five-minute demonstration",
            "",
            "1. Open `figures/02a_static_exemplar_suite_routing_matrix.png` to show that one bundle format produces a route across feature, class, prior, coverage, and leakage gates.",
            "2. Open `exemplar_cards/class_overlap_boundary_family.md` to show a hard class pair routed to `revise_class_set` before classifier blame.",
            "3. Open `exemplar_cards/prior_domination_family.md` to show the rare class at zero own-surface selection and the generated `PRIOR_DOMINATION` / `PRIOR_SELECTION_SKEW` actions.",
            "4. Open `exemplar_cards/future_class_surface_family.md` to show an unobserved future class with an expected exact signature collision, which is a prune/redefine decision rather than a corpus-search task.",
            "5. Open `figures/02b_static_audit_decision_card.png` to close with the route distribution and the claim boundary.",
            "",
            "## Evidence and automation links",
            "",
            "- `source_artifacts/exemplar_suite_manifest.csv` — one row per study family, generated route, issue codes, and first recommendation.",
            "- `source_artifacts/exemplar_route_matrix.csv` — gate-by-gate pass/warn/block surface.",
            "- `source_artifacts/*/prior_selection_balance.csv` — class-level prior-weighted selection balance.",
            "- `source_artifacts/*/static_resolution_plan.csv` — issue code, severity, evidence, action, verification, and route.",
            "- `figures/02m_static_exemplar_fingerprint_strip.png` — compact comparison of the seven static signatures.",
            "",
            "## Current issue coverage",
            "",
            "| issue code | affected exemplar families |",
            "| --- | --- |",
        ]
    )
    for code, count in sorted(issue_counts.items()):
        lines.append(f"| `{code}` | `{count}` |")

    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "This is an early admissibility and routing tool. It can expose holes, shrink candidate space, and prescribe follow-up checks; it does not prove deployed classifier performance, causal feature importance, or operational coverage.",
            "",
            "## Regenerate the showcase",
            "",
            "```bash",
            "PYTHONPATH=src python3 -m kinematic_classifier_sandbox run-static-audit-suite \\",
            "  experiments/static_admissibility/epic1_exemplar_suite.yaml \\",
            "  --output-dir artifacts/validation_packets/01_static_admissibility",
            "```",
            "",
            "The report is generated from the suite's source runs; the exemplar manifest describes the cases but does not hand-author the findings or fixes.",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_claim_boundary() -> str:
    return "\n".join(
        [
            "# Claim Boundary",
            "",
            "- The exemplar suite validates the static audit routing surface.",
            "- It is a teaching and proof asset, not a classifier leaderboard.",
            "- Candidate synergy remains candidate evidence until downstream ablation confirms it.",
            "- Coverage-thin cases route toward Corpus Explorer rather than directly justifying classifier claims.",
        ]
    ) + "\n"


def _render_suite_lane_proof_matrix() -> str:
    return "\n".join(
        [
            "# Epic 1 Lane Proof Matrix",
            "",
            "| lane | claim | hero chart | source artifact | validation check | limitation |",
            "| --- | --- | --- | --- | --- | --- |",
            "| Bundle ingestion | Epic 1 accepts portable study bundles. | `02a_static_bundle_ingestion_spine.png` | `source_bundles/*` | source bundle copy checks | bundle format does not itself guarantee a good study |",
            "| Routing coverage | The exemplar suite covers the routing surface. | `02a_static_exemplar_suite_routing_matrix.png` | `source_artifacts/exemplar_route_matrix.csv` | expected route equals actual route | finite curated families only |",
            "| Diagnostic fingerprinting | Each exemplar lights up a distinct admissibility signature. | `02m_static_exemplar_fingerprint_strip.png` | `source_artifacts/exemplar_fingerprint_scores.csv` | figure/source manifest checks | scores are simple teaching proxies |",
            "| Programmatic resolution | Each finding maps to an issue code, recommendation, and verification follow-up. | `E1_card_*.png` | `source_artifacts/*/static_resolution_plan.csv` | resolution table checks | recommendations are static routing guidance, not a guarantee |",
            "| Per-family proof cards | Each family explains why its route is correct. | `E1_card_*.png` | `exemplar_cards/*.md` | card presence checks | cards are summaries over the underlying run tables |",
        ]
    ) + "\n"


def _render_suite_latex(route_rows: list[dict[str, object]]) -> str:
    lines = [
        "\\subsection{Static Admissibility Exemplar Suite}",
        "\\label{subsec:static-admissibility-exemplars}",
        "The exemplar suite is a file-backed collection of study bundles designed to exercise the routing surface of the static audit. Each bundle consists of a YAML study declaration, a labeled feature matrix, a feature provenance schema, and a declared class schema.",
        "",
        "\\begin{figure}[ht]",
        "  \\centering",
        "  \\includegraphics[width=\\linewidth]{figures/02a_static_exemplar_suite_routing_matrix.png}",
        "  \\caption{Epic 1 exemplar routing matrix. Each row is a file-backed study bundle and each column is a static audit gate.}",
        "  \\label{fig:static-exemplar-routing-matrix}",
        "\\end{figure}",
        "",
        "\\begin{tabular}{llll}",
        "Exemplar & Primary diagnostic & Expected route & Actual route \\\\",
        "\\hline",
    ]
    for row in route_rows:
        lines.append(
            f"{row['exemplar_id']} & {row['class_separability_status']} & {row['expected_route']} & {row['actual_route']} \\\\"
        )
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def _hero_chart_manifest_rows() -> list[dict[str, str]]:
    return [
        {
            "chart_id": "02a_static_bundle_ingestion_spine",
            "role": "main",
            "path": "figures/02a_static_bundle_ingestion_spine.png",
            "evidence_tier": "artifact-backed",
            "source_artifact": "source_bundles/",
            "claim": "Epic 1 accepts portable study bundles.",
            "claim_boundary": "architecture and provenance surface, not a classifier result",
        },
        {
            "chart_id": "02a_static_exemplar_suite_routing_matrix",
            "role": "main",
            "path": "figures/02a_static_exemplar_suite_routing_matrix.png",
            "evidence_tier": "run-backed",
            "source_artifact": "source_artifacts/exemplar_route_matrix.csv",
            "claim": "Seven file-backed exemplars exercise the Epic 1 decision routes.",
            "claim_boundary": "curated exemplar routing matrix, not an exhaustive study universe",
        },
        {
            "chart_id": "02b_static_audit_decision_card",
            "role": "main",
            "path": "figures/02b_static_audit_decision_card.png",
            "evidence_tier": "artifact-backed",
            "source_artifact": "source_artifacts/exemplar_suite_manifest.csv",
            "claim": "The exemplar suite covers promotion, revision, and blocking routes.",
            "claim_boundary": "suite-level executive summary over curated exemplars",
        },
        {
            "chart_id": "02m_static_exemplar_fingerprint_strip",
            "role": "main",
            "path": "figures/02m_static_exemplar_fingerprint_strip.png",
            "evidence_tier": "run-backed",
            "source_artifact": "source_artifacts/exemplar_fingerprint_scores.csv",
            "claim": "Each exemplar has a distinct admissibility fingerprint.",
            "claim_boundary": "teaching proxy scores over curated exemplars",
        },
        {
            "chart_id": "02k_static_audit_to_action_router",
            "role": "appendix",
            "path": "figures/02k_static_audit_to_action_router.png",
            "evidence_tier": "artifact-backed",
            "source_artifact": "source_artifacts/exemplar_route_matrix.csv",
            "claim": "Static findings route to actions before compute-heavy work.",
            "claim_boundary": "routing policy map, not a probabilistic estimate",
        },
    ]
