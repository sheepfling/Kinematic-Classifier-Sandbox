from __future__ import annotations

import shutil
from pathlib import Path

from kinematic_classifier_sandbox.analysis.static_feature_class_prior_audit import (
    analyze_default_static_feature_class_prior_audit,
)
from kinematic_classifier_sandbox.analysis.static_feature_class_prior_audit_artifact_io import (
    write_static_feature_class_prior_audit_artifacts,
)
from kinematic_classifier_sandbox.analysis.static_feature_class_prior_audit_contracts import (
    StaticFeatureClassPriorAuditResult,
)
from kinematic_classifier_sandbox.utils.io import write_csv
from kinematic_classifier_sandbox.utils.plotting import plt

from .schemas import StaticAdmissibilityConfig, StaticAdmissibilityPacket

MAIN_FIGURES: tuple[str, ...] = (
    "02b_static_audit_decision_card.png",
    "02c_class_pair_confusability_matrix.png",
    "02g_prior_pathology_surface.png",
    "02e_feature_redundancy_graph.png",
)

APPENDIX_FIGURES: tuple[str, ...] = (
    "02d_feature_relevance_rank.png",
    "02f_feature_synergy_map.png",
    "02h_prior_flip_thresholds.png",
    "02i_static_coverage_feasibility.png",
    "02j_static_leakage_provenance_audit.png",
    "02k_static_audit_to_action_router.png",
)

TABLE_NAMES: tuple[str, ...] = (
    "class_confusability_matrix.csv",
    "feature_relevance_table.csv",
    "feature_redundancy_matrix.csv",
    "feature_synergy_candidates.csv",
    "prior_pathology_report.csv",
    "prior_flip_thresholds.csv",
    "static_leakage_provenance_audit.csv",
    "coverage_static_report.csv",
    "static_coverage_feasibility.csv",
    "prior_regime.csv",
)


def build_static_admissibility_result(
    config: StaticAdmissibilityConfig,
) -> StaticFeatureClassPriorAuditResult:
    return analyze_default_static_feature_class_prior_audit(
        seed=config.seed,
        trajectories_per_class=config.trajectories_per_class,
        priors=config.priors,
    )


def write_static_admissibility_packet(
    output_dir: str | Path,
    *,
    config: StaticAdmissibilityConfig | None = None,
    result: StaticFeatureClassPriorAuditResult | None = None,
) -> StaticAdmissibilityPacket:
    packet_dir = Path(output_dir)
    packet_dir.mkdir(parents=True, exist_ok=True)
    config = config or StaticAdmissibilityConfig()
    result = result or build_static_admissibility_result(config)

    source_artifacts = write_static_feature_class_prior_audit_artifacts(
        packet_dir,
        result=result,
        seed=config.seed,
        trajectories_per_class=config.trajectories_per_class,
    )
    source_dir = source_artifacts.run_dir

    copies = {
        "static_audit_report.md": "static_audit_report.md",
        "static_decision_card.md": "static_audit_decision_card.md",
        **{name: name for name in MAIN_FIGURES},
        **{name: name for name in APPENDIX_FIGURES},
        **{name: name for name in TABLE_NAMES},
    }
    for source_name, target_name in copies.items():
        source_path = source_dir / source_name
        if source_path.exists():
            shutil.copyfile(source_path, packet_dir / target_name)

    decision_card_path = packet_dir / "decision_card.md"
    readme_path = packet_dir / "README.md"
    figure_manifest_path = packet_dir / "figure_manifest.csv"
    lane_proof_matrix_path = packet_dir / "lane_proof_matrix.md"
    contact_sheet_path = packet_dir / "hero_chart_contact_sheet.png"

    decision_card_path.write_text(_render_packet_decision_card(config, result), encoding="utf-8")
    readme_path.write_text(_render_packet_readme(config, result), encoding="utf-8")
    write_csv(figure_manifest_path, _figure_manifest_rows(), ["figure_id", "role", "source_table", "claim", "claim_boundary"])
    lane_proof_matrix_path.write_text(_render_lane_proof_matrix(), encoding="utf-8")
    _write_contact_sheet(packet_dir, contact_sheet_path)
    return StaticAdmissibilityPacket(
        packet_dir=packet_dir,
        readme_path=readme_path,
        decision_card_path=decision_card_path,
        static_audit_report_path=packet_dir / "static_audit_report.md",
        static_audit_decision_card_path=packet_dir / "static_audit_decision_card.md",
        figure_manifest_path=figure_manifest_path,
        lane_proof_matrix_path=lane_proof_matrix_path,
        contact_sheet_path=contact_sheet_path,
    )


def export_static_admissibility_packet(run_dir: str | Path, output_dir: str | Path) -> StaticAdmissibilityPacket:
    run_path = Path(run_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for path in run_path.iterdir():
        if path.is_file():
            shutil.copyfile(path, output_path / path.name)
    return StaticAdmissibilityPacket(
        packet_dir=output_path,
        readme_path=output_path / "README.md",
        decision_card_path=output_path / "decision_card.md",
        static_audit_report_path=output_path / "static_audit_report.md",
        static_audit_decision_card_path=output_path / "static_audit_decision_card.md",
        figure_manifest_path=output_path / "figure_manifest.csv",
        lane_proof_matrix_path=output_path / "lane_proof_matrix.md",
        contact_sheet_path=output_path / "hero_chart_contact_sheet.png",
    )


def _render_packet_decision_card(
    config: StaticAdmissibilityConfig,
    result: StaticFeatureClassPriorAuditResult,
) -> str:
    decision = result.static_decision
    lines = [
        "# Static Admissibility Decision Card",
        "",
        f"- study_id: `{config.study_id}`",
        f"- seed: `{config.seed}`",
        f"- trajectories_per_class: `{config.trajectories_per_class}`",
        f"- static_audit_decision: `{decision['status']}`",
        f"- adequacy_label: `{decision['adequacy_label']}`",
        f"- class_count: `{len(result.class_names)}`",
        f"- feature_count: `{len(result.feature_names)}`",
        f"- leakage: `{'blocker' if any(row['status'] == 'blocker' for row in result.leakage_rows) else 'pass'}`",
        "",
        "## Warnings",
        "",
    ]
    warnings = tuple(decision.get("warnings", ()))
    lines.extend(f"- {warning}" for warning in warnings) if warnings else lines.append("- none")
    lines.extend(["", "## Blockers", ""])
    blockers = tuple(decision.get("blockers", ()))
    lines.extend(f"- {blocker}" for blocker in blockers) if blockers else lines.append("- none")
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {item}" for item in tuple(decision.get("next_work", ())))
    return "\n".join(lines) + "\n"


def _render_packet_readme(
    config: StaticAdmissibilityConfig,
    result: StaticFeatureClassPriorAuditResult,
) -> str:
    return "\n".join(
        [
            "# Static Admissibility MVP Packet",
            "",
            "This packet answers whether a feature/class/prior setup is meaningful enough to send to Corpus Explorer or classifier/filter evaluation.",
            "",
            f"- study_id: `{config.study_id}`",
            f"- decision: `{result.static_decision['status']}`",
            "- claim boundary: static admissibility is an early gate, not a final classifier benchmark.",
            "",
            "## Main Hero Charts",
            "",
            *[f"- `{name}`" for name in MAIN_FIGURES],
            "",
            "## Proof Tables",
            "",
            *[f"- `{name}`" for name in TABLE_NAMES],
            "",
            "## Regeneration",
            "",
            "```bash",
            "PYTHONPATH=src python3 -m kinematic_classifier_sandbox run-static-audit experiments/static_admissibility/common_1d_static_audit.yaml --output-dir artifacts/packets/static_admissibility_mvp",
            "PYTHONPATH=src python3 -m kinematic_classifier_sandbox validate-packet artifacts/packets/static_admissibility_mvp",
            "```",
            "",
        ]
    )


def _figure_manifest_rows() -> list[dict[str, str]]:
    source_by_figure = {
        "02b_static_audit_decision_card.png": "static_audit_decision_card.md",
        "02c_class_pair_confusability_matrix.png": "class_confusability_matrix.csv",
        "02e_feature_redundancy_graph.png": "feature_redundancy_matrix.csv",
        "02g_prior_pathology_surface.png": "prior_pathology_report.csv",
        "02d_feature_relevance_rank.png": "feature_relevance_table.csv",
        "02f_feature_synergy_map.png": "feature_synergy_candidates.csv",
        "02h_prior_flip_thresholds.png": "prior_flip_thresholds.csv",
        "02i_static_coverage_feasibility.png": "static_coverage_feasibility.csv",
        "02j_static_leakage_provenance_audit.png": "static_leakage_provenance_audit.csv",
        "02k_static_audit_to_action_router.png": "static_audit_decision_card.md",
    }
    rows = []
    for figure in (*MAIN_FIGURES, *APPENDIX_FIGURES):
        rows.append(
            {
                "figure_id": figure,
                "role": "main" if figure in MAIN_FIGURES else "appendix",
                "source_table": source_by_figure[figure],
                "claim": "static feature/class/prior admissibility is audited before corpus search",
                "claim_boundary": "static screen only; synergy remains candidate until ablation-backed",
            }
        )
    return rows


def _render_lane_proof_matrix() -> str:
    return "\n".join(
        [
            "# Static Admissibility Lane Proof Matrix",
            "",
            "| lane | claim | hero chart | source artifact | validation check | decision card field | limitation | next work | status |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            "| Static admissibility | Study candidates are screened by feature/class/prior admissibility before corpus search. | `02b_static_audit_decision_card.png` | `decision_card.md` | packet validator | `static_audit_decision` | Static checks are sample-backed proxies. | Bind static warnings into Corpus Explorer objectives. | MVP |",
            "| Class confusability | Hard class pairs are visible before algorithm blame. | `02c_class_pair_confusability_matrix.png` | `class_confusability_matrix.csv` | symmetric matrix check | `class_separability` | Pairwise overlap can miss multiclass interaction. | Add oracle overlay. | MVP |",
            "| Prior pathology | Priors can dominate before classifier work starts. | `02g_prior_pathology_surface.png` | `prior_pathology_report.csv` | prior sum and flip threshold checks | `prior_pathology` | Gaussian proxy only. | Add prior sweep objectives. | MVP |",
            "| Redundancy and synergy | Redundant clusters and candidate feature synergy are flagged before method selection. | `02e_feature_redundancy_graph.png` | `feature_redundancy_matrix.csv`; `feature_synergy_candidates.csv` | feature coverage and synergy status checks | `feature_redundancy`; `feature_synergy` | Synergy is candidate until ablation-backed. | Validate synergy candidates downstream. | MVP |",
            "",
        ]
    )


def _write_contact_sheet(packet_dir: Path, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, figure_name in zip(axes.ravel(), MAIN_FIGURES):
        image = plt.imread(packet_dir / figure_name)
        ax.imshow(image)
        ax.set_title(figure_name, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

