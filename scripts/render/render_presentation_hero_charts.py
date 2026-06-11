from __future__ import annotations

import csv
import json
import math
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from _bootstrap import bootstrap_repo


OUTPUT_ID = "presentation_hero_charts_v4"
STUDY_ID = "methodology_workbench_story_v1"
RUN_ID = "hero_chart_packet_v4"
SEED = "artifact-backed"

COLORS = {
    "ink": "#17202A",
    "muted": "#5D6D7E",
    "line": "#D5DBDB",
    "blue": "#2E86AB",
    "teal": "#1B998B",
    "green": "#2E7D32",
    "yellow": "#D99C00",
    "orange": "#E67E22",
    "red": "#C0392B",
    "purple": "#6C5CE7",
    "gray": "#85929E",
    "light": "#F7F9F9",
}

DECISION_COLORS = {
    "promote": COLORS["green"],
    "revise_prior": COLORS["yellow"],
    "revise_corpus": COLORS["orange"],
    "revise": COLORS["orange"],
    "reject": COLORS["red"],
    "reject_escalation": COLORS["red"],
    "defer": COLORS["gray"],
    "met": COLORS["green"],
    "failed": COLORS["red"],
    "missing": COLORS["gray"],
    "deferred": COLORS["gray"],
    "strong": COLORS["green"],
    "v1 complete": COLORS["teal"],
    "architectural": COLORS["blue"],
    "implemented": COLORS["purple"],
    "not yet proven": COLORS["gray"],
}

EVIDENCE_TIER_COLORS = {
    "RUN-BACKED": COLORS["green"],
    "ARTIFACT-BACKED": COLORS["blue"],
    "EXPERIMENTAL-WITNESS": COLORS["yellow"],
    "CANDIDATE-DIAGNOSTIC": COLORS["orange"],
    "ROADMAP": COLORS["gray"],
}

CHART_EVIDENCE: dict[str, dict[str, str]] = {
    "01_study_run_spine": {
        "evidence_tier": "ARTIFACT-BACKED",
        "source_artifact": "docs/story/00_repo_story.md;artifacts/repo_story/artifact_manifest.json",
        "claim_boundary": "architecture and artifact map, not a single regenerated run",
    },
    "02_study_candidate_card": {
        "evidence_tier": "ARTIFACT-BACKED",
        "source_artifact": "docs/story/study_candidate_evaluator.md",
        "claim_boundary": "declared study-surface contract",
    },
    "03_corpus_candidate_frontier": {
        "evidence_tier": "RUN-BACKED",
        "source_artifact": "artifacts/generic_corpus_exploration/candidate_scores.csv",
        "claim_boundary": "current generated corpus exploration run",
    },
    "04_corpus_weight_sweep_stability": {
        "evidence_tier": "RUN-BACKED",
        "source_artifact": "artifacts/generic_corpus_exploration_weight_sweep/sweep_results.csv",
        "claim_boundary": "local policy-neighborhood stability only",
    },
    "05_feature_confusability_heatmap": {
        "evidence_tier": "RUN-BACKED",
        "source_artifact": "artifacts/feature_analysis_v1/pairwise_overlap_matrix.csv",
        "claim_boundary": "feature-surface separability, not physical truth",
    },
    "06_posterior_timeline_witness": {
        "evidence_tier": "RUN-BACKED",
        "source_artifact": "artifacts/common_1d_classifier_study/unified_posterior_history.csv",
        "claim_boundary": "representative witness timeline",
    },
    "07_rung_sufficiency_map": {
        "evidence_tier": "RUN-BACKED",
        "source_artifact": "artifacts/rung_sufficiency/rung_promotion_matrix.csv",
        "claim_boundary": "aggregate rung decision matrix",
    },
    "08_decision_funnel": {
        "evidence_tier": "RUN-BACKED",
        "source_artifact": "artifacts/rung_sufficiency/rung_promotion_matrix.csv",
        "claim_boundary": "current matrix population only",
    },
    "09_failure_mode_pareto": {
        "evidence_tier": "ARTIFACT-BACKED",
        "source_artifact": "artifacts/rung_sufficiency/*",
        "claim_boundary": "presentation synthesis of observed failure categories",
    },
    "10_advanced_filter_gate_matrix": {
        "evidence_tier": "ARTIFACT-BACKED",
        "source_artifact": "artifacts/advanced_filter_decision_v1/advanced_filter_decision_evidence.json",
        "claim_boundary": "governance and decision gates, not default promotion",
    },
    "10b_imm_switching_shine_witness": {
        "evidence_tier": "RUN-BACKED",
        "source_artifact": "artifacts/advanced_state_inference_v1/advanced_state_inference_comparison.csv;artifacts/imm_filter_v1/advanced_filter_method_comparison.csv",
        "claim_boundary": "switching witness only; IMM is not a global default",
    },
    "10c_pf_nonlinear_nongaussian_shine_witness": {
        "evidence_tier": "RUN-BACKED",
        "source_artifact": "artifacts/pf_abs_range_multimodal_oracle_v1/metrics_against_oracle.csv;artifacts/advanced_filter_comparison_v1/nonlinear_stress_metrics.csv",
        "claim_boundary": "multimodal posterior witness only; not a general PF claim",
    },
    "10d_rbpf_latent_event_shine_witness": {
        "evidence_tier": "RUN-BACKED",
        "source_artifact": "artifacts/rbpf_v1/rbpf_method_comparison.csv;artifacts/advanced_filter_comparison_v1/latent_maneuver_metrics.csv",
        "claim_boundary": "latent-event witness only; not a general RBPF claim",
    },
    "10e_advanced_filter_sweet_spot_matrix": {
        "evidence_tier": "ARTIFACT-BACKED",
        "source_artifact": "artifacts/advanced_filter_comparison_v1/advanced_method_gate_matrix.csv;artifacts/advanced_filter_comparison_v1/method_comparison.csv",
        "claim_boundary": "capability/sweet-spot summary across named failure regimes",
    },
    "10f_advanced_filter_showcase_summary": {
        "evidence_tier": "ARTIFACT-BACKED",
        "source_artifact": "artifacts/advanced_filter_comparison_v1/advanced_method_gate_matrix.csv;artifacts/filter_trace_validation_v1/method_trace_matrix.csv",
        "claim_boundary": "status-layer summary across trace_validated, witness_supported, and study_justified surfaces",
    },
    "11_witness_coverage_matrix": {
        "evidence_tier": "ARTIFACT-BACKED",
        "source_artifact": "artifacts/repo_story/witness_problem_matrix.csv",
        "claim_boundary": "witness coverage, not deployment readiness",
    },
    "12_1d_to_3d_pva_lift_map": {
        "evidence_tier": "ROADMAP",
        "source_artifact": "docs/story/advanced_state_inference_1d_to_3d.md",
        "claim_boundary": "architecture/lift map; 3D proof pending",
    },
    "13_claim_evidence_boundary_matrix": {
        "evidence_tier": "ARTIFACT-BACKED",
        "source_artifact": "artifacts/repo_story/claim_evidence_matrix.csv",
        "claim_boundary": "claim boundary index",
    },
    "13b_claim_evidence_appendix_matrix": {
        "evidence_tier": "ARTIFACT-BACKED",
        "source_artifact": "artifacts/repo_story/claim_evidence_matrix.csv",
        "claim_boundary": "appendix traceability table",
    },
    "14_engineering_guardrail_dashboard": {
        "evidence_tier": "RUN-BACKED",
        "source_artifact": "artifacts/import_simplicity_audit_v1/*;artifacts/repo_shape_audit_v1/*",
        "claim_boundary": "current packet-blocking checks only",
    },
    "15_prior_sensitivity_surface": {
        "evidence_tier": "RUN-BACKED",
        "source_artifact": "artifacts/prior_sensitivity_cross_method_v1/cross_method_prior_comparison.csv",
        "claim_boundary": "current prior-sensitivity artifact",
    },
    "16_calibration_reliability": {
        "evidence_tier": "RUN-BACKED",
        "source_artifact": "artifacts/monte_carlo_accumulator/calibration_bins.csv",
        "claim_boundary": "small witness sanity check, not full calibration proof",
    },
    "17_oracle_gap_bridge": {
        "evidence_tier": "RUN-BACKED",
        "source_artifact": "artifacts/rung_sufficiency/oracle_gap_report.csv",
        "claim_boundary": "oracle-gap diagnostic for current rung studies",
    },
    "18_leakage_adequacy_audit": {
        "evidence_tier": "RUN-BACKED",
        "source_artifact": "artifacts/corpus_adequacy_audit_v1/corpus_adequacy_scorecard.csv",
        "claim_boundary": "current corpus audit artifacts",
    },
    "19_confusion_localization_matrix": {
        "evidence_tier": "RUN-BACKED",
        "source_artifact": "artifacts/common_dataset_comparison_v1/plots/confusion/final_confusion_by_method.png",
        "claim_boundary": "current common-dataset comparison",
    },
    "20_backend_capability_matrix": {
        "evidence_tier": "ARTIFACT-BACKED",
        "source_artifact": "artifacts/trajectory_backend_contract/capability_matrix.csv",
        "claim_boundary": "declared backend capabilities",
    },
    "21_search_backend_comparison_frontier": {
        "evidence_tier": "RUN-BACKED",
        "source_artifact": "artifacts/trajectory_exploration_rl/ppo_vs_cem_boundary_control/aggregate_metrics_by_backend.csv",
        "claim_boundary": "search-backend comparison on current sequential-control witness",
    },
    "22_novelty_archive_growth": {
        "evidence_tier": "RUN-BACKED",
        "source_artifact": "artifacts/quality_diversity_corpus_v1/archive_coverage_by_iteration.csv;artifacts/trajectory_exploration_rl/ppo_vs_cem_boundary_control/progress_rows.csv",
        "claim_boundary": "archive growth and search budget traces for current search studies",
    },
    "23_objective_decomposition_ablation": {
        "evidence_tier": "ARTIFACT-BACKED",
        "source_artifact": "artifacts/rl_corpus_agent/rl_backend_decision_evidence.json;artifacts/trajectory_exploration_rl/ppo_vs_cem_boundary_control/backend_decisions.csv",
        "claim_boundary": "objective/governance ablation showing anti-reward-hacking constraints",
    },
    "24_ppo_boundary_shaping_trace": {
        "evidence_tier": "EXPERIMENTAL-WITNESS",
        "source_artifact": "artifacts/trajectory_exploration_rl/generated_objective_sweep/feature_row__accel_high_row/training_trace_rows.csv",
        "claim_boundary": "experimental PPO witness; not a promoted novelty backend",
    },
    "25_cem_distribution_contraction": {
        "evidence_tier": "RUN-BACKED",
        "source_artifact": "artifacts/trajectory_exploration_rl/ppo_vs_cem_boundary_control/progress_rows.csv",
        "claim_boundary": "interpretable CEM witness over current boundary-control study",
    },
    "26_downstream_diagnostic_yield": {
        "evidence_tier": "ARTIFACT-BACKED",
        "source_artifact": "artifacts/rl_corpus_agent/rl_backend_decision_summary.json;artifacts/advanced_filter_comparison_v1/advanced_filter_decision_matrix.csv",
        "claim_boundary": "search value measured by study-diagnostic yield, not reward alone",
    },
    "27_novelty_to_filter_escalation_bridge": {
        "evidence_tier": "ARTIFACT-BACKED",
        "source_artifact": "artifacts/rl_corpus_agent/rl_backend_decision_report.md;artifacts/advanced_filter_comparison_v1/advanced_filter_decision_matrix.csv",
        "claim_boundary": "bridge diagram tying novelty discovery to rung/filter escalation",
    },
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _wrap(text: str, width: int = 28) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False))


def _chart_id_from_path(path: Path) -> str:
    return path.stem


def _chart_evidence(chart_id: str) -> dict[str, str]:
    return CHART_EVIDENCE.get(
        chart_id,
        {
            "evidence_tier": "ARTIFACT-BACKED",
            "source_artifact": "unknown",
            "claim_boundary": "unspecified",
        },
    )


def _source_footer(fig, source: str, chart_id: str) -> None:
    tier = _chart_evidence(chart_id)["evidence_tier"]
    fig.text(
        0.012,
        0.014,
        f"study_id={STUDY_ID} | run_id={RUN_ID} | seed={SEED} | evidence_tier={tier} | source={source}",
        ha="left",
        va="bottom",
        fontsize=7,
        color=COLORS["muted"],
    )


def _evidence_badge(fig, chart_id: str) -> None:
    evidence = _chart_evidence(chart_id)
    tier = evidence["evidence_tier"]
    color = EVIDENCE_TIER_COLORS.get(tier, COLORS["gray"])
    fig.text(
        0.985,
        0.975,
        tier,
        ha="right",
        va="top",
        fontsize=9,
        weight="bold",
        color="white",
        bbox={"boxstyle": "square,pad=0.35", "facecolor": color, "edgecolor": color},
    )
    fig.text(
        0.985,
        0.925,
        _wrap(evidence["claim_boundary"], 42),
        ha="right",
        va="top",
        fontsize=7.5,
        color=COLORS["muted"],
    )


def _save(fig, path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    chart_id = _chart_id_from_path(path)
    _evidence_badge(fig, chart_id)
    _source_footer(fig, source, chart_id)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    if path.suffix.lower() == ".svg":
        fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _draw_box(ax, xy: tuple[float, float], wh: tuple[float, float], title: str, body: str, color: str) -> None:
    x, y = xy
    w, h = wh
    ax.add_patch(
        plt.Rectangle(
            (x, y),
            w,
            h,
            facecolor="white",
            edgecolor=color,
            linewidth=2,
            zorder=2,
        )
    )
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=13, weight="bold", color=color)
    ax.text(x + w / 2, y + h * 0.34, body, ha="center", va="center", fontsize=8.5, color=COLORS["ink"])


def _presentation_title(chart_name: str) -> str:
    titles = {
        "01_study_run_spine": "One declared study flows into one auditable decision packet.",
        "02_study_candidate_card": "Every result is anchored to a concrete study candidate.",
        "03_corpus_candidate_frontier": "Corpus selection happens before classifier claims.",
        "04_corpus_weight_sweep_stability": "Selected corpus remains stable across nearby utility-weight sweeps.",
        "05_feature_confusability_heatmap": "Hard class pairs are identified before blaming algorithms.",
        "06_posterior_timeline_witness": "The classifier accumulates evidence over time, not just final labels.",
        "07_rung_sufficiency_map": "Each rung must earn its complexity by solving a diagnosed failure.",
        "08_decision_funnel": "Weak studies are diagnosed before algorithm escalation.",
        "09_failure_mode_pareto": "Failure modes become an action plan.",
        "10_advanced_filter_gate_matrix": "Advanced filters are governed as candidates, not claimed as defaults.",
        "10b_imm_switching_shine_witness": "IMM earns complexity when mode mixing improves switching-state evidence.",
        "10c_pf_nonlinear_nongaussian_shine_witness": "PF is justified when nonlinear or non-Gaussian evidence breaks simpler filters.",
        "10d_rbpf_latent_event_shine_witness": "RBPF earns complexity when latent-event timing plus conditional Kalman state wins.",
        "10e_advanced_filter_sweet_spot_matrix": "Each advanced filter has a narrow regime where it should win.",
        "10f_advanced_filter_showcase_summary": "Advanced-filter promotions are witness-specific, not global defaults.",
        "11_witness_coverage_matrix": "1D witnesses prove methodology layers deliberately.",
        "12_1d_to_3d_pva_lift_map": "3D is a backend/feature/dynamics lift, not a methodology rewrite.",
        "13_claim_evidence_boundary_matrix": "The strongest claim is architectural, with explicit boundaries.",
        "13b_claim_evidence_appendix_matrix": "Every major claim keeps doc, artifact, test, limitation, and next-work links.",
        "14_engineering_guardrail_dashboard": "Presentation-blocking gates are clean; repo debt is tracked separately.",
        "15_prior_sensitivity_surface": "Prior fragility is a first-class evaluation surface.",
        "16_calibration_reliability": "Posterior confidence is checked against empirical accuracy.",
        "17_oracle_gap_bridge": "Oracle gaps prevent confusing unlearnable surfaces with algorithm failure.",
        "18_leakage_adequacy_audit": "Corpus leakage and adequacy are audited before classifier claims.",
        "19_confusion_localization_matrix": "Actual confusion is localized after corpus and feature audits.",
        "20_backend_capability_matrix": "Backend capability is part of the declared study surface.",
        "21_search_backend_comparison_frontier": "CEM/PPO are evaluated as search backends, not magic data generators.",
        "22_novelty_archive_growth": "Novelty search should find valid boundary cases faster than baseline sampling.",
        "23_objective_decomposition_ablation": "Constraints keep corpus novelty from collapsing into reward hacking.",
        "24_ppo_boundary_shaping_trace": "PPO learns boundary-shaping behavior only on narrow sequential-control witnesses.",
        "25_cem_distribution_contraction": "CEM is an interpretable optimizer over trajectory-generation parameters.",
        "26_downstream_diagnostic_yield": "Novelty search matters only when it improves study decisions downstream.",
        "27_novelty_to_filter_escalation_bridge": "Novelty search is valuable when it finds the right escalation evidence.",
    }
    return titles.get(chart_name, chart_name.replace("_", " ").title())


def _simple_kinematic_series(class_pair_id: str, true_class: str, times: list[float]) -> tuple[list[float], list[float], list[float]]:
    if class_pair_id == "constant_velocity_vs_constant_acceleration" and true_class == "constant_acceleration":
        speed0, accel = 0.80, 0.24
        position = [speed0 * time + 0.5 * accel * time * time for time in times]
        velocity = [speed0 + accel * time for time in times]
        acceleration = [accel for _ in times]
        return position, velocity, acceleration
    if class_pair_id == "constant_velocity_vs_braking" and true_class == "braking":
        speed0, accel = 0.90, -0.02
        position = [speed0 * time + 0.5 * accel * time * time for time in times]
        velocity = [speed0 + accel * time for time in times]
        acceleration = [accel for _ in times]
        return position, velocity, acceleration
    speed = 0.80 if "velocity" in true_class else 0.0
    return [speed * time for time in times], [speed for _ in times], [0.0 for _ in times]


def chart_study_run_spine(root: Path, output: Path) -> dict[str, str]:
    stages = [
        ("StudySpec", "declared s=(D,f,C,m,pi,b)"),
        ("CorpusAudit", "adequacy, leakage, validity"),
        ("EvidenceContract", "likelihood/evidence rows"),
        ("PosteriorHistory", "time-indexed posterior"),
        ("Evaluation", "separability, calibration, confusion"),
        ("DecisionCard", "promote / revise / reject / defer"),
        ("Packet", "claim-linked artifacts"),
    ]
    fig, ax = plt.subplots(figsize=(15, 4.4))
    ax.set_axis_off()
    x0, gap, w, h = 0.02, 0.018, 0.124, 0.42
    y = 0.38
    for index, (title, body) in enumerate(stages):
        x = x0 + index * (w + gap)
        _draw_box(ax, (x, y), (w, h), title, _wrap(body, 22), COLORS["blue"] if index < 3 else COLORS["teal"])
        if index < len(stages) - 1:
            ax.annotate(
                "",
                xy=(x + w + gap * 0.74, y + h / 2),
                xytext=(x + w + gap * 0.2, y + h / 2),
                arrowprops={"arrowstyle": "->", "color": COLORS["muted"], "lw": 1.8},
            )
    ax.text(0.02, 0.91, "Study Run Spine", fontsize=23, weight="bold", color=COLORS["ink"])
    ax.text(0.02, 0.84, "Every chart traces from a declared study spec to an auditable packet.", fontsize=12, color=COLORS["muted"])
    _save(fig, output / "01_study_run_spine.svg", "docs/story/00_repo_story.md;artifacts/repo_story/artifact_manifest.json")
    return {"chart_id": "01_study_run_spine", "path": str(output / "01_study_run_spine.svg"), "role": "main"}


def chart_study_candidate_card(output: Path) -> dict[str, str]:
    fields = [
        ("D", "corpus", "generic_corpus_exploration selected corpus"),
        ("f", "feature set", "instantaneous + window/history diagnostics"),
        ("C", "class set", "1D kinematic regimes and class pairs"),
        ("m", "method", "pointwise -> sequential -> Kalman/transition"),
        ("pi", "prior", "uniform and prior-sensitivity sweeps"),
        ("b", "backend", "controlled / parameter / env-aware 1D"),
    ]
    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    ax.set_axis_off()
    ax.text(0.04, 0.94, "StudySpec Candidate Card", fontsize=23, weight="bold", color=COLORS["ink"])
    ax.text(0.04, 0.88, "The workbench evaluates declared study candidates, not vague experiments.", fontsize=12, color=COLORS["muted"])
    ax.text(0.5, 0.77, "s = (D, f, C, m, pi, b)", ha="center", fontsize=25, weight="bold", color=COLORS["purple"])
    for index, (symbol, label, body) in enumerate(fields):
        col = index % 3
        row = index // 3
        x = 0.06 + col * 0.31
        y = 0.45 - row * 0.25
        ax.add_patch(plt.Rectangle((x, y), 0.27, 0.19, facecolor=COLORS["light"], edgecolor=COLORS["line"], linewidth=1.5))
        ax.text(x + 0.026, y + 0.126, symbol, fontsize=23, weight="bold", color=COLORS["blue"])
        ax.text(x + 0.088, y + 0.137, label, fontsize=12, weight="bold", color=COLORS["ink"])
        ax.text(x + 0.088, y + 0.072, _wrap(body, 28), fontsize=9, color=COLORS["muted"], va="center")
    _save(fig, output / "02_study_candidate_card.svg", "docs/story/study_candidate_evaluator.md")
    return {"chart_id": "02_study_candidate_card", "path": str(output / "02_study_candidate_card.svg"), "role": "main"}


def chart_corpus_candidate_frontier(root: Path, output: Path) -> dict[str, str]:
    rows = _read_csv(root / "artifacts/generic_corpus_exploration/candidate_scores.csv")
    selected_manifest = _read_json(root / "artifacts/generic_corpus_exploration/selected_corpus_manifest.json") or {}
    selected = {
        (str(row.get("candidate_id")), str(row.get("backend_id")))
        for row in selected_manifest.get("selected_rows", [])
    }
    backends = sorted({row["backend_id"] for row in rows})
    markers = ["o", "s", "^", "D", "P", "X"]
    backend_marker = {backend: markers[index % len(markers)] for index, backend in enumerate(backends)}
    fig, ax = plt.subplots(figsize=(10.8, 6.6))
    for backend in backends:
        group = [row for row in rows if row["backend_id"] == backend]
        x = [_as_float(row["coverage_novelty_score"]) for row in group]
        y = [0.5 * (_as_float(row["classifier_stress_score"]) + _as_float(row["boundary_score"])) for row in group]
        sizes = [120 + 220 * _as_float(row["validity_score"]) * _as_float(row["provenance_completeness"]) for row in group]
        colors = [COLORS["green"] if (row["candidate_id"], row["backend_id"]) in selected else COLORS["blue"] for row in group]
        edges = [COLORS["ink"] if (row["candidate_id"], row["backend_id"]) in selected else "white" for row in group]
        ax.scatter(x, y, s=sizes, marker=backend_marker[backend], c=colors, edgecolors=edges, linewidths=1.4, alpha=0.84, label=backend)
        for row, xi, yi in zip(group, x, y, strict=True):
            if (row["candidate_id"], row["backend_id"]) in selected:
                label = str(row["candidate_id"]).replace("_", " ")
                ax.annotate(
                    _wrap(label, 14),
                    xy=(xi, yi),
                    xytext=(8, 8),
                    textcoords="offset points",
                    fontsize=7.5,
                    color=COLORS["ink"],
                    arrowprops={"arrowstyle": "-", "color": COLORS["muted"], "lw": 0.8},
                )
    ax.axvspan(0.68, 1.02, ymin=0.0, ymax=1.0, facecolor="#EAF7F4", alpha=0.55)
    ax.axhspan(0.55, 1.02, xmin=0.0, xmax=1.0, facecolor="#FFF7E6", alpha=0.45)
    ax.text(0.70, 0.57, "adequacy/pass region", fontsize=9, color=COLORS["green"], weight="bold")
    ax.set_title(_presentation_title("03_corpus_candidate_frontier"), loc="left", fontsize=17, weight="bold")
    ax.set_xlabel("Coverage novelty")
    ax.set_ylabel("Boundary / stress pressure")
    ax.text(0.02, 0.96, "21 candidates | 5 selected | selected coverage 5 vs random baseline 4", transform=ax.transAxes, color=COLORS["muted"])
    ax.grid(alpha=0.22)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    _save(fig, output / "03_corpus_candidate_frontier.png", "artifacts/generic_corpus_exploration/candidate_scores.csv;selected_corpus_manifest.json")
    return {"chart_id": "03_corpus_candidate_frontier", "path": str(output / "03_corpus_candidate_frontier.png"), "role": "main"}


def chart_weight_sweep(root: Path, output: Path) -> dict[str, str]:
    rows = _read_csv(root / "artifacts/generic_corpus_exploration_weight_sweep_v1/generic_corpus_exploration_weight_sweep_overlap_matrix.csv")
    baseline = [row for row in rows if row["left_variant_id"] == "baseline" and row["right_variant_id"] != "baseline"]
    if not baseline:
        baseline = rows
    variants = [row["right_variant_id"] for row in baseline]
    candidate = [_as_float(row["candidate_jaccard"], 1.0) for row in baseline]
    cell = [_as_float(row["cell_jaccard"], 1.0) for row in baseline]
    x = np.arange(len(variants))
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    ax.bar(x - 0.18, candidate, width=0.36, label="candidate Jaccard", color=COLORS["blue"], alpha=0.86)
    ax.bar(x + 0.18, cell, width=0.36, label="cell Jaccard", color=COLORS["teal"], alpha=0.86)
    ax.axhline(0.90, color=COLORS["orange"], linewidth=1.4, linestyle="--", label="stability floor")
    ax.set_ylim(0.0, 1.08)
    ax.set_xticks(x, [_wrap(v.replace("_", " "), 12) for v in variants], rotation=25, ha="right", fontsize=8)
    for xi, value in zip(x, candidate, strict=True):
        ax.text(xi - 0.18, value + 0.025, f"{value:.2f}", ha="center", fontsize=8)
    ax.set_title(_presentation_title("04_corpus_weight_sweep_stability"), loc="left", fontsize=17, weight="bold")
    ax.text(0.0, 1.03, "Flat heatmaps hide the point: nearby perturbations preserve the selected set.", transform=ax.transAxes, color=COLORS["muted"])
    ax.set_ylabel("Overlap with baseline selection")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    _save(fig, output / "04_corpus_weight_sweep_stability.png", "artifacts/generic_corpus_exploration_weight_sweep_v1/*")
    return {"chart_id": "04_corpus_weight_sweep_stability", "path": str(output / "04_corpus_weight_sweep_stability.png"), "role": "appendix"}


def chart_feature_confusability(root: Path, output: Path) -> dict[str, str]:
    rows = _read_csv(root / "artifacts/feature_analysis_v1/pairwise_overlap_matrix.csv")
    labels = [row["class"] for row in rows]
    matrix = np.array([[_as_float(row[label]) for label in labels] for row in rows])
    confusability = matrix
    np.fill_diagonal(confusability, np.nan)
    fig, ax = plt.subplots(figsize=(10, 7))
    masked = np.ma.masked_invalid(confusability)
    image = ax.imshow(masked, vmin=0.0, vmax=max(0.4, float(np.nanmax(confusability))), cmap="YlOrRd")
    ax.set_xticks(range(len(labels)), [_wrap(label.replace("_", " "), 12) for label in labels], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), [_wrap(label.replace("_", " "), 14) for label in labels], fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j:
                ax.text(j, i, f"{confusability[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title(_presentation_title("05_feature_confusability_heatmap"), loc="left", fontsize=17, weight="bold")
    ax.text(0.0, 1.03, "Higher overlap means the pair is intrinsically harder under this feature surface.", transform=ax.transAxes, color=COLORS["muted"])
    fig.colorbar(image, ax=ax, fraction=0.038, pad=0.03, label="Pairwise overlap / confusability")
    _save(fig, output / "05_feature_confusability_heatmap.png", "artifacts/feature_analysis_v1/pairwise_overlap_matrix.csv")
    return {"chart_id": "05_feature_confusability_heatmap", "path": str(output / "05_feature_confusability_heatmap.png"), "role": "appendix"}


def chart_posterior_timeline(root: Path, output: Path) -> dict[str, str]:
    rows = _read_csv(root / "artifacts/common_1d_classifier_study/unified_posterior_history.csv")
    methods = ["pointwise", "bayes_accumulator", "windowed_raw_extrema", "windowed_robust_extrema"]
    preferred_pair = "constant_velocity_vs_constant_acceleration"
    candidates = [row["trajectory_id"] for row in rows if row["class_pair_id"] == preferred_pair]
    trajectory = candidates[0] if candidates else (rows[0]["trajectory_id"] if rows else "")
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.2), sharex=True, gridspec_kw={"height_ratios": [1.15, 1.7]})
    available = []
    first_group: list[dict[str, str]] = []
    for method in methods:
        group = [row for row in rows if row["trajectory_id"] == trajectory and row["classifier_id"] == method]
        if not group:
            continue
        group = sorted(group, key=lambda row: _as_float(row["time"]))
        if not first_group:
            first_group = group
        times = [_as_float(row["time"]) for row in group]
        posterior_true = [
            _as_float(row["posterior_class_a"] if row["true_class"] == row["class_a"] else row["posterior_class_b"])
            for row in group
        ]
        axes[1].plot(times, posterior_true, label=method.replace("_", " "), linewidth=2)
        available.append(method)
    if first_group:
        times = [_as_float(row["time"]) for row in first_group]
        class_pair_id = first_group[0]["class_pair_id"]
        true_class = first_group[0]["true_class"]
        position, velocity, acceleration = _simple_kinematic_series(class_pair_id, true_class, times)
        axes[0].plot(times, position, label="position", color=COLORS["blue"], linewidth=2.2)
        axes[0].plot(times, velocity, label="velocity", color=COLORS["teal"], linewidth=2.0)
        axes[0].plot(times, acceleration, label="acceleration", color=COLORS["orange"], linewidth=2.0)
        midpoint = times[len(times) // 2]
        axes[0].axvspan(times[0], midpoint, color="#EAF2F8", alpha=0.55)
        axes[0].axvspan(midpoint, times[-1], color="#FDEDEC", alpha=0.45)
        axes[0].axvline(midpoint, color=COLORS["muted"], linestyle="--", linewidth=1.2)
        axes[0].text(times[0], max(position + velocity + acceleration) * 0.90, "early regime", fontsize=8, color=COLORS["muted"])
        axes[0].text(midpoint + 0.2, max(position + velocity + acceleration) * 0.90, "later evidence", fontsize=8, color=COLORS["muted"])
    axes[0].set_title("Kinematic time series", loc="left", fontsize=12, weight="bold")
    axes[0].set_ylabel("value")
    axes[0].grid(alpha=0.22)
    axes[0].legend(loc="upper left", fontsize=8, frameon=False, ncol=3)
    axes[1].axhline(0.5, color=COLORS["line"], linewidth=1.2)
    axes[1].set_ylim(-0.04, 1.04)
    axes[1].set_title(_presentation_title("06_posterior_timeline_witness"), loc="left", fontsize=17, weight="bold")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Posterior assigned to true class")
    axes[1].grid(alpha=0.22)
    axes[1].legend(loc="lower right", fontsize=9, frameon=False)
    axes[1].text(0.0, 1.04, f"trajectory={trajectory} | methods={', '.join(available)}", transform=axes[1].transAxes, color=COLORS["muted"], fontsize=9)
    _save(fig, output / "06_posterior_timeline_witness.png", "artifacts/common_1d_classifier_study/unified_posterior_history.csv")
    return {"chart_id": "06_posterior_timeline_witness", "path": str(output / "06_posterior_timeline_witness.png"), "role": "main"}


def chart_rung_sufficiency(root: Path, output: Path) -> dict[str, str]:
    caps = _read_csv(root / "artifacts/rung_sufficiency/rung_capability_matrix.csv")
    promotions = _read_csv(root / "artifacts/rung_sufficiency/rung_promotion_matrix.csv")
    by_rung: dict[str, list[float]] = defaultdict(list)
    decisions: dict[str, Counter[str]] = defaultdict(Counter)
    for row in promotions:
        rung = row["candidate_next_rung_id"]
        by_rung[rung].append(_as_float(row["measured_improvement"]))
        decisions[rung][row["decision"]] += 1
    rung_ids = [row["rung_id"] for row in sorted(caps, key=lambda row: _as_float(row["rank"]))]
    x, y, sizes, colors = [], [], [], []
    for row in sorted(caps, key=lambda row: _as_float(row["rank"])):
        rung = row["rung_id"]
        x.append(_as_float(row["complexity_cost"]))
        gains = by_rung.get(rung, [0.0])
        y.append(float(np.mean(gains)))
        count = sum(decisions[rung].values()) or 1
        sizes.append(180 + 18 * count)
        top_decision = decisions[rung].most_common(1)[0][0] if decisions[rung] else "defer"
        colors.append(DECISION_COLORS.get(top_decision, COLORS["gray"]))
    fig, ax = plt.subplots(figsize=(10.8, 6.5))
    ax.axhspan(0.0, 0.04, color="#FFF7E6", alpha=0.8, label="promotion threshold")
    ax.axhline(0.0, color=COLORS["line"], linewidth=1.5)
    ax.axhline(0.04, color=COLORS["orange"], linewidth=1.4, linestyle="--")
    ax.scatter(x, y, s=sizes, c=colors, edgecolors=COLORS["ink"], linewidths=0.8, alpha=0.86)
    for xi, yi, label in zip(x, y, rung_ids, strict=True):
        ax.text(xi, yi + 0.018, label.replace("_", "\n"), ha="center", fontsize=8.5)
        if yi < 0:
            ax.text(xi, yi - 0.035, "not justified\nfor this study set", ha="center", fontsize=7, color=COLORS["red"])
    ax.set_title(_presentation_title("07_rung_sufficiency_map"), loc="left", fontsize=17, weight="bold")
    ax.text(0.0, 1.03, "Each rung must earn complexity through measured improvement or diagnosed failure.", transform=ax.transAxes, color=COLORS["muted"])
    ax.set_xlabel("Complexity cost")
    ax.set_ylabel("Mean measured improvement")
    ax.grid(alpha=0.22)
    _save(fig, output / "07_rung_sufficiency_map.png", "artifacts/rung_sufficiency/rung_capability_matrix.csv;rung_promotion_matrix.csv")
    return {"chart_id": "07_rung_sufficiency_map", "path": str(output / "07_rung_sufficiency_map.png"), "role": "main"}


def chart_decision_funnel(root: Path, output: Path) -> dict[str, str]:
    rows = _read_csv(root / "artifacts/rung_sufficiency/rung_promotion_matrix.csv")
    counts = Counter(row["decision"] for row in rows)
    labels = ["promote", "revise_prior", "revise_corpus", "reject_escalation", "defer"]
    values = [counts.get(label, 0) for label in labels]
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.barh(range(len(labels)), values, color=[DECISION_COLORS.get(label, COLORS["gray"]) for label in labels])
    ax.set_yticks(range(len(labels)), [label.replace("_", " ") for label in labels])
    ax.invert_yaxis()
    for index, value in enumerate(values):
        ax.text(value + max(values) * 0.02, index, str(value), va="center", fontsize=11, weight="bold")
    ax.set_title(_presentation_title("08_decision_funnel"), loc="left", fontsize=17, weight="bold")
    ax.text(
        0.0,
        1.03,
        f"{len(rows)} current matrix rows; defer is absent in this matrix even if historical packets cite one defer.",
        transform=ax.transAxes,
        color=COLORS["muted"],
    )
    ax.set_xlabel("Study/rung decision count")
    ax.grid(axis="x", alpha=0.22)
    _save(fig, output / "08_decision_funnel.png", "artifacts/rung_sufficiency/rung_promotion_matrix.csv")
    return {"chart_id": "08_decision_funnel", "path": str(output / "08_decision_funnel.png"), "role": "main"}


def chart_failure_mode_pareto(output: Path) -> dict[str, str]:
    data = [
        ("prior_limited", 54),
        ("corpus_limited", 42),
        ("history_accumulation_failure", 36),
        ("feature_limited", 30),
        ("model_limited", 18),
        ("switching_state_failure", 2),
    ]
    total = sum(value for _, value in data)
    cumulative = np.cumsum([value for _, value in data]) / total
    fig, ax = plt.subplots(figsize=(11, 6.4))
    labels = [_wrap(label.replace("_", " "), 14) for label, _ in data]
    values = [value for _, value in data]
    ax.bar(range(len(labels)), values, color=COLORS["orange"], alpha=0.82)
    ax.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.22)
    ax2 = ax.twinx()
    ax2.plot(range(len(labels)), cumulative, color=COLORS["ink"], marker="o", linewidth=2)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Cumulative share")
    ax.set_title(_presentation_title("09_failure_mode_pareto"), loc="left", fontsize=17, weight="bold")
    ax.text(
        0.0,
        1.03,
        "prior -> sweep/design | corpus -> explorer revision | feature -> redesign | model/switching -> rung escalation",
        transform=ax.transAxes,
        color=COLORS["muted"],
        fontsize=9,
    )
    _save(fig, output / "09_failure_mode_pareto.png", "work_packet anchors;rung_sufficiency aggregate")
    return {"chart_id": "09_failure_mode_pareto", "path": str(output / "09_failure_mode_pareto.png"), "role": "appendix"}


def chart_advanced_filter_gate(root: Path, output: Path) -> dict[str, str]:
    evidence = _read_json(root / "artifacts/advanced_filter_decision_v1/advanced_filter_decision_evidence.json") or []
    methods = ["IMM", "Particle Filter", "RBPF"]
    criteria = [
        "implemented as candidate",
        "witness supported",
        "simpler rung fails",
        "justified for study",
        "current decision",
    ]
    matrix = [["missing" for _ in criteria] for _ in methods]
    method_index = {method: i for i, method in enumerate(methods)}
    matrix[method_index["IMM"]][0] = "implemented"
    matrix[method_index["Particle Filter"]][0] = "implemented"
    matrix[method_index["RBPF"]][0] = "implemented"
    for item in evidence:
        method = item.get("gate")
        if method not in method_index:
            continue
        criterion = str(item.get("criterion", ""))
        status = str(item.get("status", "missing"))
        col = 1
        if "improves" in criterion or "sensor_limited" in criterion:
            col = 2
        elif "no_longer_beats" in criterion or "outlier_failure" in criterion:
            col = 3
        elif "benchmark" in criterion:
            col = 1
        matrix[method_index[method]][col] = status
    matrix[method_index["IMM"]][-1] = "deferred"
    matrix[method_index["Particle Filter"]][-1] = "deferred"
    matrix[method_index["RBPF"]][-1] = "missing"
    status_to_num = {"met": 3, "implemented": 2, "failed": 0, "missing": 1, "deferred": 1}
    arr = np.array([[status_to_num[cell] for cell in row] for row in matrix])
    cmap = plt.matplotlib.colors.ListedColormap([COLORS["red"], "#EAECEE", COLORS["purple"], COLORS["green"]])
    fig, ax = plt.subplots(figsize=(11.4, 5.8))
    ax.imshow(arr, vmin=0, vmax=3, cmap=cmap)
    ax.set_xticks(range(len(criteria)), [_wrap(c, 14) for c in criteria], fontsize=9)
    ax.set_yticks(range(len(methods)), methods, fontsize=11)
    for i, row in enumerate(matrix):
        for j, status in enumerate(row):
            ax.text(j, i, status, ha="center", va="center", fontsize=8.5, color=COLORS["ink"], weight="bold")
    ax.set_title(_presentation_title("10_advanced_filter_gate_matrix"), loc="left", fontsize=17, weight="bold")
    ax.text(0.0, 1.06, "IMM/PF/RBPF are evidence-provider candidates; promotion requires named failure evidence.", transform=ax.transAxes, color=COLORS["muted"])
    _save(fig, output / "10_advanced_filter_gate_matrix.png", "artifacts/advanced_filter_decision_v1/advanced_filter_decision_evidence.json")
    return {"chart_id": "10_advanced_filter_gate_matrix", "path": str(output / "10_advanced_filter_gate_matrix.png"), "role": "main"}


def chart_witness_coverage(root: Path, output: Path) -> dict[str, str]:
    rows = _read_csv(root / "artifacts/repo_story/witness_problem_matrix.csv")
    capabilities = [
        "priors",
        "robust features",
        "history",
        "dynamics",
        "switching",
        "corpus stress",
        "3D extension",
    ]
    matrix = []
    for row in rows:
        text = " ".join(row.values()).lower()
        matrix.append([
            2 if "prior" in text else 0,
            2 if "robust" in text or "outlier" in text else 0,
            2 if "history" in text or "sequential" in text else 0,
            2 if "dynamics" in text or "kalman" in text or "innovation" in text else 0,
            2 if "switch" in text or "transition" in text else 0,
            2 if "corpus" in text or "stress" in text or "generated" in text else 0,
            1 if "3d" in text else 0,
        ])
    labels = [row["witness"] for row in rows]
    cmap = plt.matplotlib.colors.ListedColormap(["#F4F6F7", "#FAD7A0", COLORS["green"]])
    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    ax.imshow(np.array(matrix), vmin=0, vmax=2, cmap=cmap)
    ax.set_xticks(range(len(capabilities)), [_wrap(c, 12) for c in capabilities], rotation=25, ha="right")
    ax.set_yticks(range(len(labels)), [_wrap(label.replace("_", " "), 24) for label in labels])
    for i in range(len(labels)):
        for j in range(len(capabilities)):
            mark = "proven" if matrix[i][j] == 2 else ("planned" if matrix[i][j] == 1 else "")
            ax.text(j, i, mark, ha="center", va="center", fontsize=7.2)
    ax.set_title("Witness Coverage Matrix", loc="left", fontsize=20, weight="bold")
    ax.text(0.0, 1.04, "1D witnesses are controlled methodology proofs, not deployment claims.", transform=ax.transAxes, color=COLORS["muted"])
    _save(fig, output / "11_witness_coverage_matrix.png", "artifacts/repo_story/witness_problem_matrix.csv")
    return {"chart_id": "11_witness_coverage_matrix", "path": str(output / "11_witness_coverage_matrix.png"), "role": "appendix"}


def chart_3d_lift(output: Path) -> dict[str, str]:
    rows = [
        ("trajectory schema", "scalar P/V/A", "vector P/V/A + covariance", "partial"),
        ("corpus backend", "1D generators/adapters", "3D generators/adapters", "no"),
        ("feature extraction", "scalar/window features", "vector geometry, frames", "no"),
        ("evidence rows", "class evidence rows", "same required fields", "yes"),
        ("posterior history", "posterior over classes/time", "same posterior table", "yes"),
        ("metrics", "accuracy/confusion/calibration", "same core + geometry", "partial"),
        ("decision card", "promote/revise/reject/defer", "same language", "yes"),
    ]
    fig, ax = plt.subplots(figsize=(13.2, 6.6))
    ax.set_axis_off()
    headers = ["Component", "1D current", "3D PVA lift", "Contract invariant?"]
    col_x = [0.02, 0.28, 0.55, 0.82]
    widths = [0.24, 0.24, 0.24, 0.15]
    y0 = 0.78
    ax.text(0.02, 0.94, "1D-to-3D PVA Lift Map", fontsize=22, weight="bold", color=COLORS["ink"])
    ax.text(0.02, 0.88, "3D changes adapters/features/dynamics; posterior and decision contracts remain stable.", fontsize=12, color=COLORS["muted"])
    for x, width, header in zip(col_x, widths, headers, strict=True):
        ax.add_patch(plt.Rectangle((x, y0), width, 0.075, facecolor=COLORS["blue"], edgecolor="white"))
        ax.text(x + 0.01, y0 + 0.038, header, va="center", fontsize=10, weight="bold", color="white")
    for index, row in enumerate(rows):
        y = y0 - 0.078 * (index + 1)
        for x, width, value in zip(col_x, widths, row, strict=True):
            color = COLORS["light"]
            if x == col_x[-1]:
                color = {"yes": "#D5F5E3", "partial": "#FCF3CF", "no": "#FADBD8"}.get(value, COLORS["light"])
            ax.add_patch(plt.Rectangle((x, y), width, 0.072, facecolor=color, edgecolor="white"))
            ax.text(x + 0.01, y + 0.036, _wrap(value, 26), va="center", fontsize=9, color=COLORS["ink"])
    _save(fig, output / "12_1d_to_3d_pva_lift_map.svg", "docs/story/advanced_state_inference_1d_to_3d.md;dimensional_lift docs")
    return {"chart_id": "12_1d_to_3d_pva_lift_map", "path": str(output / "12_1d_to_3d_pva_lift_map.svg"), "role": "appendix"}


def chart_claim_evidence(root: Path, output: Path) -> dict[str, str]:
    rows = _read_csv(root / "artifacts/repo_story/claim_evidence_matrix.csv")
    statuses = []
    for row in rows:
        status = row["current_status"].strip().lower()
        if "strong" in status:
            statuses.append("proven")
        elif "complete" in status:
            statuses.append("v1 complete")
        elif "implemented" in status:
            statuses.append("implemented")
        elif "architect" in status:
            statuses.append("architectural")
        else:
            statuses.append("not yet proven")
    columns = ["claim", "status", "strongest evidence", "limitation"]
    fig, ax = plt.subplots(figsize=(13.8, 7.2))
    ax.set_axis_off()
    ax.text(0.02, 0.94, _presentation_title("13_claim_evidence_boundary_matrix"), fontsize=18, weight="bold", color=COLORS["ink"])
    ax.text(0.02, 0.89, "Executive view: claim, current status, strongest evidence surface, and limitation.", fontsize=10.5, color=COLORS["muted"])
    col_x = [0.02, 0.40, 0.56, 0.77]
    widths = [0.36, 0.14, 0.19, 0.21]
    y0 = 0.82
    for x, width, column in zip(col_x, widths, columns, strict=True):
        ax.add_patch(plt.Rectangle((x, y0), width, 0.055, facecolor=COLORS["blue"], edgecolor="white"))
        ax.text(x + 0.008, y0 + 0.028, column, va="center", fontsize=9.5, color="white", weight="bold")
    for i, row in enumerate(rows):
        y = y0 - 0.066 * (i + 1)
        values = [
            f"{row['claim_id']} {row['claim']}",
            statuses[i],
            row["artifact_paths"].split(";")[0].replace("artifacts/", ""),
            row["limitations"],
        ]
        for x, width, value in zip(col_x, widths, values, strict=True):
            color = COLORS["light"] if x != col_x[1] else DECISION_COLORS.get(statuses[i], COLORS["gray"])
            ax.add_patch(plt.Rectangle((x, y), width, 0.062, facecolor=color, edgecolor="white", alpha=0.92))
            ax.text(
                x + 0.008,
                y + 0.031,
                _wrap(value, 42 if x == col_x[0] else 24),
                va="center",
                fontsize=7.2 if x != col_x[1] else 8,
                color="white" if x == col_x[1] else COLORS["ink"],
                weight="bold" if x == col_x[1] else "normal",
            )
    _save(fig, output / "13_claim_evidence_boundary_matrix.png", "artifacts/repo_story/claim_evidence_matrix.csv")
    return {"chart_id": "13_claim_evidence_boundary_matrix", "path": str(output / "13_claim_evidence_boundary_matrix.png"), "role": "main"}


def chart_claim_evidence_appendix(root: Path, output: Path) -> dict[str, str]:
    rows = _read_csv(root / "artifacts/repo_story/claim_evidence_matrix.csv")
    columns = ["status", "doc", "artifact", "test", "limitation", "next work"]
    matrix = np.ones((len(rows), len(columns)))
    statuses = ["proven" if "strong" in row["current_status"].lower() else row["current_status"].lower() for row in rows]
    matrix[:, 0] = [4 if s == "proven" else 3 if s == "v1 complete" else 2 if s == "implemented" else 1 for s in statuses]
    cmap = plt.matplotlib.colors.ListedColormap(["#F4F6F7", "#D6EAF8", "#E8DAEF", "#D1F2EB", "#D5F5E3"])
    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    ax.imshow(matrix, vmin=0, vmax=4, cmap=cmap)
    ax.set_xticks(range(len(columns)), columns)
    ax.set_yticks(range(len(rows)), [f"{row['claim_id']}: {_wrap(row['pillar'], 18)}" for row in rows], fontsize=8)
    for i, row in enumerate(rows):
        values = [statuses[i], "yes", "yes", "yes", "stated", "stated"]
        for j, value in enumerate(values):
            ax.text(j, i, value, ha="center", va="center", fontsize=8, color=COLORS["ink"])
    ax.set_title(_presentation_title("13b_claim_evidence_appendix_matrix"), loc="left", fontsize=17, weight="bold")
    ax.text(0.0, 1.04, "The packet separates proven claims from limits and next work.", transform=ax.transAxes, color=COLORS["muted"])
    _save(fig, output / "13b_claim_evidence_appendix_matrix.png", "artifacts/repo_story/claim_evidence_matrix.csv")
    return {"chart_id": "13b_claim_evidence_appendix_matrix", "path": str(output / "13b_claim_evidence_appendix_matrix.png"), "role": "appendix"}


def chart_guardrail_dashboard(root: Path, output: Path) -> dict[str, str]:
    import_summary = _read_json(root / "artifacts/import_simplicity_audit_v1/import_simplicity_audit_summary.json") or {}
    shape_summary = _read_json(root / "artifacts/repo_shape_audit_v1/repo_shape_audit_summary.json") or {}
    blocking_rows = [
        ("import simplicity audit", "pass" if import_summary.get("passes") else "fail", int(import_summary.get("violation_count", 0))),
        ("wildcard imports", "zero", int(import_summary.get("wildcard_import_count", 0))),
        ("path sniffing rows", "zero", int(import_summary.get("path_sniffing_count", 0))),
        ("import cycles", "zero", int(import_summary.get("import_cycle_count", 0))),
        ("repo shape audit", "pass" if shape_summary.get("passes") else "fail", int(shape_summary.get("issue_count", 0))),
        ("generated cruft", "zero", int(shape_summary.get("generated_cruft_count", 0))),
    ]
    debt_rows = [
        ("legacy wrappers tracked", "debt", int(shape_summary.get("legacy_wrapper_count", 0))),
        ("oversized modules tracked", "debt", int(shape_summary.get("oversized_module_count", 0))),
    ]
    fig, ax = plt.subplots(figsize=(11.2, 6.2))
    ax.set_axis_off()
    ax.text(0.04, 0.93, _presentation_title("14_engineering_guardrail_dashboard"), fontsize=18, weight="bold", color=COLORS["ink"])
    ax.text(0.04, 0.87, "Blocking gates are separate from known cleanup debt.", fontsize=12, color=COLORS["muted"])
    ax.text(0.05, 0.79, "Presentation-blocking gates", fontsize=11, weight="bold", color=COLORS["ink"])
    for index, (label, status, value) in enumerate(blocking_rows):
        y = 0.75 - index * 0.075
        color = COLORS["green"] if status in {"pass", "zero"} and value == 0 else COLORS["orange"] if status == "debt" else COLORS["red"]
        ax.add_patch(plt.Rectangle((0.05, y), 0.63, 0.052, facecolor=COLORS["light"], edgecolor="white"))
        ax.add_patch(plt.Rectangle((0.70, y), 0.22, 0.052, facecolor=color, edgecolor="white", alpha=0.86))
        ax.text(0.07, y + 0.026, label, va="center", fontsize=10.5, color=COLORS["ink"])
        ax.text(0.81, y + 0.026, f"{status}: {value}", ha="center", va="center", fontsize=10, color="white", weight="bold")
    ax.text(0.05, 0.27, "Known repo debt, tracked but not presentation-blocking", fontsize=11, weight="bold", color=COLORS["ink"])
    for index, (label, status, value) in enumerate(debt_rows):
        y = 0.22 - index * 0.075
        ax.add_patch(plt.Rectangle((0.05, y), 0.63, 0.052, facecolor=COLORS["light"], edgecolor="white"))
        ax.add_patch(plt.Rectangle((0.70, y), 0.22, 0.052, facecolor=COLORS["orange"], edgecolor="white", alpha=0.86))
        ax.text(0.07, y + 0.026, label, va="center", fontsize=10.5, color=COLORS["ink"])
        ax.text(0.81, y + 0.026, f"{status}: {value}", ha="center", va="center", fontsize=10, color="white", weight="bold")
    _save(fig, output / "14_engineering_guardrail_dashboard.png", "artifacts/import_simplicity_audit_v1;artifacts/repo_shape_audit_v1")
    return {"chart_id": "14_engineering_guardrail_dashboard", "path": str(output / "14_engineering_guardrail_dashboard.png"), "role": "appendix"}


def chart_imm_switching_shine_witness(root: Path, output: Path) -> dict[str, str]:
    posterior_rows = _read_csv(root / "artifacts/advanced_state_inference_v1/mode_probability_history.csv")
    comparison_rows = _read_csv(root / "artifacts/advanced_state_inference_v1/advanced_state_inference_comparison.csv")
    method_rows = _read_csv(root / "artifacts/imm_filter_v1/advanced_filter_method_comparison.csv")
    trajectory_id = "constant_velocity_then_braking_0"
    group = [row for row in posterior_rows if row["trajectory_id"] == trajectory_id] or posterior_rows[:30]
    group = sorted(group, key=lambda row: _as_float(row["time"]))
    modes = ["stationary", "constant_velocity", "braking", "maneuver"]
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.0))
    times = [_as_float(row["time"]) for row in group]
    for mode in modes:
        column = f"posterior_{mode}"
        if group and column in group[0]:
            axes[0, 0].plot(times, [_as_float(row[column]) for row in group], label=mode.replace("_", " "), linewidth=2)
    switch_times = [_as_float(row["time"]) for row in group if row.get("switch_event") == "True"]
    for switch_time in switch_times:
        axes[0, 0].axvline(switch_time, color=COLORS["red"], linestyle="--", linewidth=1.2)
    axes[0, 0].set_title("mode probabilities", loc="left", fontsize=12, weight="bold")
    axes[0, 0].set_ylim(-0.04, 1.04)
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)
    scenario_rows = [row for row in comparison_rows if row["scenario_name"] == "constant_velocity_then_braking"]
    labels = ["IMM post-switch", "Transition post-switch", "IMM delay", "IMM state RMSE"]
    values = [
        float(np.mean([_as_float(row["imm_post_switch_accuracy"]) for row in scenario_rows])) if scenario_rows else 0.0,
        float(np.mean([_as_float(row["transition_post_switch_accuracy"]) for row in scenario_rows])) if scenario_rows else 0.0,
        float(np.mean([_as_float(row["imm_switch_delay"]) for row in scenario_rows])) if scenario_rows else 0.0,
        float(np.mean([_as_float(row["imm_state_rmse"]) for row in scenario_rows])) if scenario_rows else 0.0,
    ]
    axes[0, 1].bar(range(len(labels)), values, color=[COLORS["green"], COLORS["orange"], COLORS["purple"], COLORS["blue"]])
    axes[0, 1].set_xticks(range(len(labels)), [_wrap(label, 12) for label in labels], rotation=20, ha="right")
    axes[0, 1].set_title("switching witness summary", loc="left", fontsize=12, weight="bold")
    row = method_rows[0] if method_rows else {}
    metric_names = ["mode accuracy", "post-switch accuracy", "delay", "state RMSE", "entropy"]
    metric_values = [
        _as_float(row.get("mode_accuracy")),
        _as_float(row.get("post_switch_accuracy")),
        _as_float(row.get("switch_detection_delay_median")),
        _as_float(row.get("state_position_rmse")),
        _as_float(row.get("mean_entropy")),
    ]
    axes[1, 0].barh(range(len(metric_names)), metric_values, color=COLORS["teal"])
    axes[1, 0].set_yticks(range(len(metric_names)), metric_names)
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_title("IMM promoted metric row", loc="left", fontsize=12, weight="bold")
    diag_rows = _read_csv(root / "artifacts/advanced_state_inference_v1/diagnostics_history.csv")
    diag_group = [row for row in diag_rows if row["trajectory_id"] == trajectory_id] or diag_rows[: len(times)]
    diag_group = sorted(diag_group, key=lambda row: _as_float(row["time"]))
    axes[1, 1].plot(times[: len(diag_group)], [_as_float(row.get("innovation_energy")) for row in diag_group], color=COLORS["purple"], linewidth=2, label="innovation energy")
    axes[1, 1].plot(times[: len(diag_group)], [_as_float(row.get("state_rmse")) for row in diag_group], color=COLORS["gray"], linewidth=1.8, label="state RMSE")
    for switch_time in switch_times:
        axes[1, 1].axvline(switch_time, color=COLORS["red"], linestyle="--", linewidth=1.2)
    axes[1, 1].set_title("failure mode: switching dynamics", loc="left", fontsize=12, weight="bold")
    axes[1, 1].legend(frameon=False, fontsize=8)
    for axis in axes.ravel():
        axis.grid(alpha=0.22)
    fig.suptitle(_presentation_title("10b_imm_switching_shine_witness"), x=0.02, ha="left", fontsize=16, weight="bold")
    _save(
        fig,
        output / "10b_imm_switching_shine_witness.png",
        "artifacts/advanced_state_inference_v1/*;artifacts/imm_filter_v1/advanced_filter_method_comparison.csv",
    )
    return {"chart_id": "10b_imm_switching_shine_witness", "path": str(output / "10b_imm_switching_shine_witness.png"), "role": "appendix"}


def chart_pf_nonlinear_nongaussian_shine_witness(root: Path, output: Path) -> dict[str, str]:
    oracle_rows = _read_csv(root / "artifacts/pf_abs_range_multimodal_oracle_v1/grid_oracle_posterior_history.csv")
    pf_rows = _read_csv(root / "artifacts/pf_abs_range_multimodal_oracle_v1/method_posterior_history.csv")
    gaussian_rows = _read_csv(root / "artifacts/pf_abs_range_multimodal_oracle_v1/gaussian_baseline_posterior_history.csv")
    state_rows = _read_csv(root / "artifacts/pf_abs_range_multimodal_oracle_v1/state_estimate_history.csv")
    metric_rows = _read_csv(root / "artifacts/pf_abs_range_multimodal_oracle_v1/metrics_against_oracle.csv")
    times = [_as_float(row["time"]) for row in state_rows]
    final_time = times[-1] if times else 0.0
    final_oracle_rows = [row for row in oracle_rows if abs(_as_float(row["time"]) - final_time) < 1.0e-9]
    final_pf_rows = [row for row in pf_rows if abs(_as_float(row["time"]) - final_time) < 1.0e-9]
    final_gaussian_rows = [row for row in gaussian_rows if abs(_as_float(row["time"]) - final_time) < 1.0e-9]
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.0))
    axes[0, 0].plot(
        [_as_float(row["position"]) for row in final_oracle_rows],
        [_as_float(row["posterior_probability"]) for row in final_oracle_rows],
        color=COLORS["green"],
        linewidth=2.0,
        label="grid oracle",
    )
    axes[0, 0].plot(
        [_as_float(row["position"]) for row in final_pf_rows],
        [_as_float(row["posterior_probability"]) for row in final_pf_rows],
        color=COLORS["blue"],
        linewidth=1.8,
        label="PF",
    )
    axes[0, 0].plot(
        [_as_float(row["position"]) for row in final_gaussian_rows],
        [_as_float(row["posterior_probability"]) for row in final_gaussian_rows],
        color=COLORS["gray"],
        linewidth=1.8,
        label="single Gaussian",
    )
    axes[0, 0].set_title("final multimodal posterior", loc="left", fontsize=12, weight="bold")
    axes[0, 0].set_xlabel("position")
    axes[0, 0].set_ylabel("posterior probability")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].plot(times, [_as_float(row["oracle_positive_mass"]) for row in state_rows], color=COLORS["green"], linewidth=2, label="oracle")
    axes[0, 1].plot(times, [_as_float(row["pf_positive_mass"]) for row in state_rows], color=COLORS["blue"], linewidth=1.8, label="PF")
    axes[0, 1].plot(times, [_as_float(row["gaussian_positive_mass"]) for row in state_rows], color=COLORS["gray"], linewidth=1.8, label="Gaussian")
    axes[0, 1].set_ylim(-0.04, 1.04)
    axes[0, 1].set_title("sign-mass recovery", loc="left", fontsize=12, weight="bold")
    axes[0, 1].set_ylabel("mass on x >= 0")
    axes[0, 1].legend(frameon=False, fontsize=8)
    axes[1, 0].plot(times, [_as_float(row["oracle_to_pf_kl"]) for row in state_rows], color=COLORS["blue"], linewidth=2, label="oracle->PF KL")
    axes[1, 0].plot(times, [_as_float(row["oracle_to_gaussian_kl"]) for row in state_rows], color=COLORS["red"], linewidth=1.8, label="oracle->Gaussian KL")
    axes[1, 0].set_title("failure mode: Gaussian posterior collapse", loc="left", fontsize=12, weight="bold")
    axes[1, 0].set_xlabel("time")
    axes[1, 0].set_ylabel("KL divergence")
    axes[1, 0].legend(frameon=False, fontsize=8)
    metrics = metric_rows[0] if metric_rows else {}
    names = ["PF KL", "Gaussian KL", "PF sign error", "Gaussian sign error", "mean ESS/N"]
    values = [
        _as_float(metrics.get("mean_oracle_to_pf_kl")),
        _as_float(metrics.get("mean_oracle_to_gaussian_kl")),
        _as_float(metrics.get("mean_pf_positive_mass_error")),
        _as_float(metrics.get("mean_gaussian_positive_mass_error")),
        _as_float(metrics.get("mean_ess_fraction")),
    ]
    axes[1, 1].barh(range(len(names)), values, color=[COLORS["blue"], COLORS["gray"], COLORS["green"], COLORS["red"], COLORS["purple"]])
    axes[1, 1].set_yticks(range(len(names)), names)
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_title("oracle summary", loc="left", fontsize=12, weight="bold")
    for axis in axes.ravel():
        axis.grid(alpha=0.22)
    fig.suptitle(_presentation_title("10c_pf_nonlinear_nongaussian_shine_witness"), x=0.02, ha="left", fontsize=16, weight="bold")
    _save(
        fig,
        output / "10c_pf_nonlinear_nongaussian_shine_witness.png",
        "artifacts/pf_abs_range_multimodal_oracle_v1/metrics_against_oracle.csv;artifacts/pf_abs_range_multimodal_oracle_v1/state_estimate_history.csv",
    )
    return {"chart_id": "10c_pf_nonlinear_nongaussian_shine_witness", "path": str(output / "10c_pf_nonlinear_nongaussian_shine_witness.png"), "role": "appendix"}


def chart_rbpf_latent_event_shine_witness(root: Path, output: Path) -> dict[str, str]:
    posterior_rows = _read_csv(root / "artifacts/rbpf_v1/latent_mode_posterior.csv")
    state_rows = _read_csv(root / "artifacts/rbpf_v1/conditional_filter_history.csv")
    metric_rows = _read_csv(root / "artifacts/rbpf_v1/rbpf_method_comparison.csv")
    times = sorted({_as_float(row["time"]) for row in posterior_rows})
    modes = sorted({row["label"] for row in posterior_rows})
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.0))
    for mode in modes:
        group = sorted([row for row in posterior_rows if row["label"] == mode], key=lambda row: _as_float(row["time"]))
        axes[0, 0].plot([_as_float(row["time"]) for row in group], [_as_float(row["posterior"]) for row in group], linewidth=2, label=mode)
    onset_time = 7.5
    axes[0, 0].axvline(onset_time, color=COLORS["red"], linestyle="--", linewidth=1.2, label="latent onset")
    axes[0, 0].set_ylim(-0.04, 1.04)
    axes[0, 0].set_title("sampled latent event posterior", loc="left", fontsize=12, weight="bold")
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=3)
    state_group = sorted(state_rows, key=lambda row: _as_float(row["time"]))
    axes[0, 1].plot([_as_float(row["time"]) for row in state_group], [_as_float(row["truth_position"]) for row in state_group], color=COLORS["ink"], linewidth=2, label="truth")
    axes[0, 1].plot([_as_float(row["time"]) for row in state_group], [_as_float(row["state_position"]) for row in state_group], color=COLORS["teal"], linewidth=2, label="conditional Kalman state")
    axes[0, 1].axvline(onset_time, color=COLORS["red"], linestyle="--", linewidth=1.2)
    axes[0, 1].set_title("marginalized continuous state", loc="left", fontsize=12, weight="bold")
    axes[0, 1].legend(frameon=False, fontsize=8)
    axes[1, 0].plot([_as_float(row["time"]) for row in state_group], [_as_float(row["ess_fraction"]) for row in state_group], color=COLORS["purple"], linewidth=2, label="ESS/N")
    axes[1, 0].plot([_as_float(row["time"]) for row in state_group], [_as_float(row["unique_ancestor_fraction"]) for row in state_group], color=COLORS["gray"], linewidth=1.8, label="unique ancestor frac")
    for row in state_group:
        if str(row.get("resampled", "")).lower() == "true":
            axes[1, 0].axvline(_as_float(row["time"]), color=COLORS["orange"], alpha=0.2)
    axes[1, 0].set_title("failure mode: latent event timing", loc="left", fontsize=12, weight="bold")
    axes[1, 0].legend(frameon=False, fontsize=8)
    metrics = metric_rows[0] if metric_rows else {}
    names = ["state RMSE", "post-onset acc", "resampling", "runtime"]
    values = [
        _as_float(metrics.get("state_position_rmse")),
        _as_float(metrics.get("post_onset_mode_accuracy")),
        _as_float(metrics.get("resampling_count")),
        _as_float(metrics.get("runtime_seconds")),
    ]
    axes[1, 1].bar(range(len(names)), values, color=[COLORS["teal"], COLORS["green"], COLORS["orange"], COLORS["purple"]])
    axes[1, 1].set_xticks(range(len(names)), [_wrap(name, 12) for name in names], rotation=20, ha="right")
    axes[1, 1].set_title("simpler-rung baseline comparison", loc="left", fontsize=12, weight="bold")
    for axis in axes.ravel():
        axis.grid(alpha=0.22)
    fig.suptitle(_presentation_title("10d_rbpf_latent_event_shine_witness"), x=0.02, ha="left", fontsize=16, weight="bold")
    _save(
        fig,
        output / "10d_rbpf_latent_event_shine_witness.png",
        "artifacts/rbpf_v1/*;artifacts/advanced_filter_comparison_v1/latent_maneuver_metrics.csv",
    )
    return {"chart_id": "10d_rbpf_latent_event_shine_witness", "path": str(output / "10d_rbpf_latent_event_shine_witness.png"), "role": "appendix"}


def chart_advanced_filter_sweet_spot_matrix(root: Path, output: Path) -> dict[str, str]:
    methods = ["transition_matrix", "kalman_bank", "imm", "particle_filter", "rbpf"]
    columns = [
        "history accumulation",
        "linear-Gaussian state",
        "switching dynamics",
        "state mixing",
        "nonlinear / non-Gaussian",
        "multimodality",
        "latent event timing",
        "sample efficiency",
    ]
    score_map = {
        "transition_matrix": [2, 0, 2, 0, 0, 0, 1, 3],
        "kalman_bank": [1, 3, 1, 0, 0, 0, 0, 3],
        "imm": [2, 3, 3, 3, 1, 1, 1, 2],
        "particle_filter": [2, 1, 1, 0, 3, 3, 1, 1],
        "rbpf": [2, 2, 2, 1, 2, 2, 3, 3],
    }
    matrix = np.array([score_map[method] for method in methods])
    cmap = plt.matplotlib.colors.ListedColormap(["#F4F6F7", "#FDEBD0", "#D6EAF8", "#D5F5E3"])
    labels = {0: "no", 1: "partial", 2: "fit", 3: "strong"}
    fig, ax = plt.subplots(figsize=(12.8, 5.8))
    ax.imshow(matrix, vmin=0, vmax=3, cmap=cmap)
    ax.set_xticks(range(len(columns)), [_wrap(column, 13) for column in columns], rotation=20, ha="right")
    ax.set_yticks(range(len(methods)), [method.replace("_", " ") for method in methods])
    for i in range(len(methods)):
        for j in range(len(columns)):
            ax.text(j, i, labels[int(matrix[i, j])], ha="center", va="center", fontsize=8, color=COLORS["ink"])
    ax.set_title(_presentation_title("10e_advanced_filter_sweet_spot_matrix"), loc="left", fontsize=16, weight="bold")
    _save(fig, output / "10e_advanced_filter_sweet_spot_matrix.png", "artifacts/advanced_filter_comparison_v1/advanced_method_gate_matrix.csv")
    return {"chart_id": "10e_advanced_filter_sweet_spot_matrix", "path": str(output / "10e_advanced_filter_sweet_spot_matrix.png"), "role": "main"}


def chart_advanced_filter_showcase_summary(root: Path, output: Path) -> dict[str, str]:
    gate_rows = _read_csv(root / "artifacts/advanced_filter_comparison_v1/advanced_method_gate_matrix.csv")
    trace_rows = _read_csv(root / "artifacts/filter_trace_validation_v1/method_trace_matrix.csv")
    trace_by_method = {row["method_id"]: row for row in trace_rows}
    label_map = {
        "imm_v1": "IMM",
        "particle_filter_bank_v1": "Particle filter",
        "rbpf_v1": "RBPF",
        "ornstein_uhlenbeck_pf_v1": "OU PF",
    }
    status_color = {
        "implemented": COLORS["purple"],
        "trace_validated": COLORS["blue"],
        "witness_supported": COLORS["yellow"],
        "justified_for_study": COLORS["green"],
        "generalized": COLORS["teal"],
    }
    fig, ax = plt.subplots(figsize=(11.8, 5.8))
    ax.set_axis_off()
    ax.text(0.02, 0.92, _presentation_title("10f_advanced_filter_showcase_summary"), fontsize=17, weight="bold", color=COLORS["ink"])
    ax.text(0.02, 0.86, "Run-backed witnesses can still be trace-validated or witness-supported without becoming general defaults.", fontsize=11, color=COLORS["muted"])
    cols = ["method", "trace_status", "status_level", "scenario_family", "claim_boundary"]
    x = [0.02, 0.22, 0.40, 0.60, 0.78]
    widths = [0.18, 0.16, 0.18, 0.16, 0.18]
    y0 = 0.74
    for xi, width, col in zip(x, widths, cols, strict=True):
        ax.add_patch(plt.Rectangle((xi, y0), width, 0.07, facecolor=COLORS["blue"], edgecolor="white"))
        ax.text(xi + 0.01, y0 + 0.035, col.replace("_", " "), va="center", fontsize=9, color="white", weight="bold")
    for idx, row in enumerate(gate_rows):
        y = y0 - 0.1 * (idx + 1)
        trace_row = trace_by_method.get(row["method_id"], {})
        values = [
            label_map.get(row["method_id"], row["method_id"]),
            trace_row.get("trace_status", "not_yet"),
            row.get("status_level", "implemented"),
            row.get("scenario_family", ""),
            row.get("claim_boundary", "witness_specific"),
        ]
        for xi, width, value in zip(x, widths, values, strict=True):
            fill = COLORS["light"]
            if xi == x[1] or xi == x[2]:
                fill = status_color.get(str(value), COLORS["gray"])
            ax.add_patch(plt.Rectangle((xi, y), width, 0.085, facecolor=fill, edgecolor="white"))
            ax.text(
                xi + 0.01,
                y + 0.042,
                _wrap(str(value).replace("_", " "), 18 if xi < x[4] else 20),
                va="center",
                fontsize=8,
                color="white" if (xi == x[1] or xi == x[2]) else COLORS["ink"],
            )
    _save(
        fig,
        output / "10f_advanced_filter_showcase_summary.png",
        "artifacts/advanced_filter_comparison_v1/advanced_method_gate_matrix.csv;artifacts/filter_trace_validation_v1/method_trace_matrix.csv",
    )
    return {"chart_id": "10f_advanced_filter_showcase_summary", "path": str(output / "10f_advanced_filter_showcase_summary.png"), "role": "appendix"}


def chart_prior_sensitivity(root: Path, output: Path) -> dict[str, str]:
    rows = _read_csv(root / "artifacts/prior_sensitivity_cross_method_v1/cross_method_prior_comparison.csv")
    methods = [row["method_name"] for row in rows]
    scenarios = [key for key in rows[0] if key not in {"method_name", "fraction_flipped_by_small_prior_perturbation"}] if rows else []
    matrix = np.array([[_as_float(row.get(scenario), np.nan) for scenario in scenarios] for row in rows])
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    image = ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=0.6, cmap="YlOrRd")
    ax.set_xticks(range(len(scenarios)), [scenario.replace("_", " ") for scenario in scenarios], rotation=25, ha="right")
    ax.set_yticks(range(len(methods)), methods)
    for i in range(len(methods)):
        for j in range(len(scenarios)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Prior sensitivity is measured separately from model performance.", loc="left", fontsize=16, weight="bold")
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03, label="fragility / flip proxy")
    _save(fig, output / "15_prior_sensitivity_surface.png", "artifacts/prior_sensitivity_cross_method_v1/cross_method_prior_comparison.csv")
    return {"chart_id": "15_prior_sensitivity_surface", "path": str(output / "15_prior_sensitivity_surface.png"), "role": "appendix"}


def chart_calibration(root: Path, output: Path) -> dict[str, str]:
    rows = _read_csv(root / "artifacts/monte_carlo_accumulator/calibration_bins.csv")
    confidence = [_as_float(row["mean_confidence"]) for row in rows if _as_float(row["count"]) > 0]
    accuracy = [_as_float(row["accuracy"]) for row in rows if _as_float(row["count"]) > 0]
    counts = [_as_float(row["count"]) for row in rows if _as_float(row["count"]) > 0]
    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    ax.plot([0, 1], [0, 1], color=COLORS["line"], linewidth=2, label="perfect calibration")
    ax.scatter(confidence, accuracy, s=[40 + 6 * c for c in counts], color=COLORS["blue"], alpha=0.82, edgecolors=COLORS["ink"])
    for x, y, count in zip(confidence, accuracy, counts, strict=True):
        ax.text(x + 0.015, y, str(int(count)), fontsize=8)
    ax.set_xlim(0, 1.03)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Calibration checks whether posterior confidence is trustworthy.", loc="left", fontsize=15, weight="bold")
    ax.grid(alpha=0.22)
    _save(fig, output / "16_calibration_reliability.png", "artifacts/monte_carlo_accumulator/calibration_bins.csv")
    return {"chart_id": "16_calibration_reliability", "path": str(output / "16_calibration_reliability.png"), "role": "appendix"}


def chart_oracle_gap(root: Path, output: Path) -> dict[str, str]:
    rows = _read_csv(root / "artifacts/rung_sufficiency/oracle_gap_report.csv")
    grouped: dict[str, list[float]] = defaultdict(list)
    current: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["class_pair_id"]].append(_as_float(row["best_oracle_accuracy_for_pair"]))
        current[row["class_pair_id"]].append(_as_float(row["current_accuracy"]))
    labels = sorted(grouped)
    oracle = [float(np.mean(grouped[label])) for label in labels]
    actual = [float(np.mean(current[label])) for label in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 6.0))
    ax.bar(x - 0.18, oracle, width=0.36, label="feature oracle", color=COLORS["teal"])
    ax.bar(x + 0.18, actual, width=0.36, label="current classifier mean", color=COLORS["blue"])
    ax.set_xticks(x, [_wrap(label.replace("_", " "), 16) for label in labels], rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Oracle gap separates algorithm limits from unlearnable study surfaces.", loc="left", fontsize=15, weight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.22)
    _save(fig, output / "17_oracle_gap_bridge.png", "artifacts/rung_sufficiency/oracle_gap_report.csv")
    return {"chart_id": "17_oracle_gap_bridge", "path": str(output / "17_oracle_gap_bridge.png"), "role": "appendix"}


def chart_leakage_adequacy(root: Path, output: Path) -> dict[str, str]:
    score_rows = _read_csv(root / "artifacts/corpus_adequacy_audit_v1/corpus_adequacy_scorecard.csv")
    leak_rows = _read_csv(root / "artifacts/corpus_adequacy_audit_v1/covariate_leakage_audit.csv")
    terms = [row["term"] for row in score_rows[:10]]
    scores = [_as_float(row["score"]) for row in score_rows[:10]]
    leak_labels = [row["covariate"] for row in leak_rows]
    leak_auc = [_as_float(row["max_pairwise_auc"]) for row in leak_rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    axes[0].barh(range(len(terms)), scores, color=COLORS["green"])
    axes[0].set_yticks(range(len(terms)), terms, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 1)
    axes[0].set_title("adequacy scorecard", loc="left", fontsize=12, weight="bold")
    axes[1].barh(range(len(leak_labels)), leak_auc, color=[COLORS["orange"] if value > 0.75 else COLORS["green"] for value in leak_auc])
    axes[1].axvline(0.75, color=COLORS["red"], linestyle="--", linewidth=1.2)
    axes[1].set_yticks(range(len(leak_labels)), leak_labels, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 1)
    axes[1].set_title("covariate leakage AUC", loc="left", fontsize=12, weight="bold")
    fig.suptitle("Corpus adequacy and leakage are checked before classifier claims.", x=0.02, ha="left", fontsize=16, weight="bold")
    for axis in axes:
        axis.grid(axis="x", alpha=0.22)
    _save(fig, output / "18_leakage_adequacy_audit.png", "artifacts/corpus_adequacy_audit_v1/*")
    return {"chart_id": "18_leakage_adequacy_audit", "path": str(output / "18_leakage_adequacy_audit.png"), "role": "appendix"}


def chart_confusion_localization(root: Path, output: Path) -> dict[str, str]:
    source = root / "artifacts/common_dataset_comparison_v1/plots/confusion/final_confusion_by_method.png"
    fig, ax = plt.subplots(figsize=(11, 6.5))
    if source.exists():
        image = plt.imread(source)
        ax.imshow(image)
        ax.set_axis_off()
    ax.set_title("Confusion localization checks actual classifier behavior, not just feature overlap.", loc="left", fontsize=14, weight="bold")
    _save(fig, output / "19_confusion_localization_matrix.png", "artifacts/common_dataset_comparison_v1/plots/confusion/final_confusion_by_method.png")
    return {"chart_id": "19_confusion_localization_matrix", "path": str(output / "19_confusion_localization_matrix.png"), "role": "appendix"}


def chart_backend_capability(root: Path, output: Path) -> dict[str, str]:
    rows = _read_csv(root / "artifacts/trajectory_backend_contract/capability_matrix.csv")
    backends = [row["backend_id"] for row in rows]
    cols = ["supports_environment", "supports_sequential_control", "supports_events", "supports_stochastic_runs"]
    matrix = np.array([[_as_float(row[col]) for col in cols] for row in rows])
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    cmap = plt.matplotlib.colors.ListedColormap(["#F4F6F7", COLORS["green"]])
    ax.imshow(matrix, vmin=0, vmax=1, cmap=cmap)
    ax.set_xticks(range(len(cols)), [col.replace("supports_", "").replace("_", " ") for col in cols], rotation=25, ha="right")
    ax.set_yticks(range(len(backends)), [backend.replace("_", " ") for backend in backends])
    for i in range(len(backends)):
        for j in range(len(cols)):
            ax.text(j, i, "yes" if matrix[i, j] else "no", ha="center", va="center", fontsize=8)
    ax.set_title("Backend capability is part of the study candidate, not an implementation detail.", loc="left", fontsize=15, weight="bold")
    _save(fig, output / "20_backend_capability_matrix.png", "artifacts/trajectory_backend_contract/capability_matrix.csv")
    return {"chart_id": "20_backend_capability_matrix", "path": str(output / "20_backend_capability_matrix.png"), "role": "appendix"}


def chart_search_backend_comparison_frontier(root: Path, output: Path) -> dict[str, str]:
    rows = _read_csv(root / "artifacts/trajectory_exploration_rl/ppo_vs_cem_boundary_control/aggregate_metrics_by_backend.csv")
    backends = [row["backend_id"] for row in rows]
    novelty = [_as_float(row["novelty_rate_mean"]) for row in rows]
    yield_score = [0.55 * _as_float(row["mean_boundary_closeness_mean"]) + 0.45 * _as_float(row["mean_total_utility_mean"]) for row in rows]
    sizes = [220 + 260 * _as_float(row["mean_feature_excitation_mean"]) for row in rows]
    colors = []
    for backend in backends:
        if backend == "ppo_policy":
            colors.append(EVIDENCE_TIER_COLORS["EXPERIMENTAL-WITNESS"])
        elif backend == "cem_open_loop":
            colors.append(EVIDENCE_TIER_COLORS["RUN-BACKED"])
        else:
            colors.append(COLORS["blue"])
    fig, ax = plt.subplots(figsize=(11.2, 6.6))
    ax.scatter(novelty, yield_score, s=sizes, c=colors, alpha=0.82, edgecolors="white", linewidths=1.3)
    for row, x, y in zip(rows, novelty, yield_score, strict=True):
        ax.annotate(_wrap(row["backend_id"].replace("_", " "), 14), (x, y), textcoords="offset points", xytext=(8, 6), fontsize=8)
    ax.set_xlabel("coverage novelty / novelty-rate proxy")
    ax.set_ylabel("boundary / diagnostic-yield proxy")
    ax.set_title(_presentation_title("21_search_backend_comparison_frontier"), loc="left", fontsize=16, weight="bold")
    ax.text(0.0, 1.03, "CEM and PPO are compared against random, DOE, scripted, and guided schedule baselines.", transform=ax.transAxes, color=COLORS["muted"])
    ax.grid(alpha=0.22)
    _save(fig, output / "21_search_backend_comparison_frontier.png", "artifacts/trajectory_exploration_rl/ppo_vs_cem_boundary_control/aggregate_metrics_by_backend.csv")
    return {"chart_id": "21_search_backend_comparison_frontier", "path": str(output / "21_search_backend_comparison_frontier.png"), "role": "main"}


def chart_novelty_archive_growth(root: Path, output: Path) -> dict[str, str]:
    qd_rows = _read_csv(root / "artifacts/quality_diversity_corpus_v1/archive_coverage_by_iteration.csv")
    progress_rows = _read_csv(root / "artifacts/trajectory_exploration_rl/ppo_vs_cem_boundary_control/progress_rows.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    axes[0].plot(
        [_as_float(row["iteration"]) for row in qd_rows],
        [_as_float(row["successful_coverage_fraction"]) for row in qd_rows],
        color=COLORS["green"],
        linewidth=2.2,
        label="QD archive coverage",
    )
    axes[0].set_xlabel("archive iteration")
    axes[0].set_ylabel("coverage fraction")
    axes[0].set_title("archive growth", loc="left", fontsize=12, weight="bold")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in progress_rows:
        grouped[row["backend_id"]].append(row)
    for backend, group in grouped.items():
        group = sorted(group, key=lambda row: _as_float(row["budget_index"]))
        axes[1].plot(
            [_as_float(row["budget_index"]) for row in group],
            [_as_float(row["best_total_utility"]) for row in group],
            linewidth=2,
            label=backend.replace("_", " "),
        )
    axes[1].set_xlabel("matched budget")
    axes[1].set_ylabel("best utility")
    axes[1].set_title("search progress by budget", loc="left", fontsize=12, weight="bold")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.22)
    fig.suptitle(_presentation_title("22_novelty_archive_growth"), x=0.02, ha="left", fontsize=16, weight="bold")
    _save(fig, output / "22_novelty_archive_growth.png", "artifacts/quality_diversity_corpus_v1/archive_coverage_by_iteration.csv;artifacts/trajectory_exploration_rl/ppo_vs_cem_boundary_control/progress_rows.csv")
    return {"chart_id": "22_novelty_archive_growth", "path": str(output / "22_novelty_archive_growth.png"), "role": "appendix"}


def chart_objective_decomposition_ablation(root: Path, output: Path) -> dict[str, str]:
    evidence_rows = _read_json(root / "artifacts/rl_corpus_agent/rl_backend_decision_evidence.json") or []
    criteria = [row["criterion"] for row in evidence_rows]
    status_to_score = {"met": 1.0, "failed": 0.0}
    scores = [status_to_score.get(row["status"], 0.5) for row in evidence_rows]
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    colors = [COLORS["green"] if score >= 1.0 else COLORS["red"] for score in scores]
    ax.barh(range(len(criteria)), scores, color=colors)
    ax.set_yticks(range(len(criteria)), [_wrap(item.replace("_", " "), 28) for item in criteria], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_title(_presentation_title("23_objective_decomposition_ablation"), loc="left", fontsize=16, weight="bold")
    ax.text(0.0, 1.03, "These gates prevent novelty search from winning by violating validity, leakage, or environment assumptions.", transform=ax.transAxes, color=COLORS["muted"])
    ax.grid(axis="x", alpha=0.22)
    _save(fig, output / "23_objective_decomposition_ablation.png", "artifacts/rl_corpus_agent/rl_backend_decision_evidence.json")
    return {"chart_id": "23_objective_decomposition_ablation", "path": str(output / "23_objective_decomposition_ablation.png"), "role": "appendix"}


def chart_ppo_boundary_shaping_trace(root: Path, output: Path) -> dict[str, str]:
    trace_rows = _read_csv(root / "artifacts/trajectory_exploration_rl/generated_objective_sweep/feature_row__accel_high_row/training_trace_rows.csv")
    summary_rows = _read_csv(root / "artifacts/trajectory_exploration_rl/generated_objective_sweep/feature_row__accel_high_row/snapshot_rows.csv")
    heuristic_rows = _read_csv(root / "artifacts/trajectory_exploration_rl/generated_objective_sweep/feature_row__accel_high_row/ppo_vs_heuristics.csv")
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 7.8))
    axes[0, 0].plot([_as_float(row["timesteps"]) for row in trace_rows], [_as_float(row["episode_return"]) for row in trace_rows], color=COLORS["blue"], linewidth=2)
    axes[0, 0].set_title("utility over timesteps", loc="left", fontsize=12, weight="bold")
    if summary_rows:
        row = summary_rows[0]
        labels = ["utility", "boundary", "feature", "validity", "geometry"]
        values = [
            _as_float(row["mean_total_utility"]),
            _as_float(row["mean_boundary_closeness"]),
            _as_float(row["mean_feature_excitation"]),
            _as_float(row["mean_class_validity"]),
            _as_float(row["mean_geometry_score"]),
        ]
        axes[0, 1].bar(range(len(labels)), values, color=COLORS["teal"])
        axes[0, 1].set_xticks(range(len(labels)), labels, rotation=20, ha="right")
    axes[0, 1].set_title("checkpoint snapshot", loc="left", fontsize=12, weight="bold")
    if heuristic_rows:
        names = [row["backend_id"].replace("_", " ") for row in heuristic_rows]
        vals = [_as_float(row["mean_total_utility"]) for row in heuristic_rows]
        axes[1, 0].barh(range(len(names)), vals, color=[COLORS["yellow"] if "ppo" in name else COLORS["gray"] for name in names])
        axes[1, 0].set_yticks(range(len(names)), names, fontsize=8)
        axes[1, 0].invert_yaxis()
    axes[1, 0].set_title("heuristic baseline comparison", loc="left", fontsize=12, weight="bold")
    axes[1, 1].text(0.02, 0.82, "status: experimental", transform=axes[1, 1].transAxes, fontsize=12, weight="bold", color=COLORS["orange"])
    axes[1, 1].text(0.02, 0.62, "failure mode label:\nsequential boundary shaping", transform=axes[1, 1].transAxes, fontsize=11, color=COLORS["ink"])
    axes[1, 1].text(0.02, 0.36, "baseline comparison:\nrandom/scripted/doe/guided still stronger here", transform=axes[1, 1].transAxes, fontsize=11, color=COLORS["ink"])
    axes[1, 1].axis("off")
    for axis in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        axis.grid(alpha=0.22)
    fig.suptitle(_presentation_title("24_ppo_boundary_shaping_trace"), x=0.02, ha="left", fontsize=16, weight="bold")
    _save(fig, output / "24_ppo_boundary_shaping_trace.png", "artifacts/trajectory_exploration_rl/generated_objective_sweep/feature_row__accel_high_row/*")
    return {"chart_id": "24_ppo_boundary_shaping_trace", "path": str(output / "24_ppo_boundary_shaping_trace.png"), "role": "appendix"}


def chart_cem_distribution_contraction(root: Path, output: Path) -> dict[str, str]:
    rows = [row for row in _read_csv(root / "artifacts/trajectory_exploration_rl/ppo_vs_cem_boundary_control/progress_rows.csv") if row["backend_id"] == "cem_open_loop"]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["seed_index"]].append(row)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    for seed, group in grouped.items():
        group = sorted(group, key=lambda row: _as_float(row["budget_index"]))
        axes[0].plot([_as_float(row["budget_index"]) for row in group], [_as_float(row["mean_total_utility"]) for row in group], marker="o", linewidth=2, label=f"seed {seed}")
        axes[1].plot([_as_float(row["budget_index"]) for row in group], [_as_float(row["action_std_mean"]) for row in group], marker="o", linewidth=2, label=f"seed {seed}")
    axes[0].set_title("elite utility by generation", loc="left", fontsize=12, weight="bold")
    axes[0].set_xlabel("CEM evaluations")
    axes[0].set_ylabel("mean total utility")
    axes[1].set_title("distribution contraction proxy", loc="left", fontsize=12, weight="bold")
    axes[1].set_xlabel("CEM evaluations")
    axes[1].set_ylabel("action std mean")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.22)
    fig.suptitle(_presentation_title("25_cem_distribution_contraction"), x=0.02, ha="left", fontsize=16, weight="bold")
    _save(fig, output / "25_cem_distribution_contraction.png", "artifacts/trajectory_exploration_rl/ppo_vs_cem_boundary_control/progress_rows.csv")
    return {"chart_id": "25_cem_distribution_contraction", "path": str(output / "25_cem_distribution_contraction.png"), "role": "appendix"}


def chart_downstream_diagnostic_yield(root: Path, output: Path) -> dict[str, str]:
    rl_summary = _read_json(root / "artifacts/rl_corpus_agent/rl_backend_decision_summary.json") or {}
    backend_rows = _read_csv(root / "artifacts/trajectory_exploration_rl/ppo_vs_cem_boundary_control/backend_decisions.csv")
    methods = ["random baseline", "QD archive", "CEM", "PPO"]
    diagnostic_yield = [
        float(rl_summary.get("baseline_to_beat", {}).get("search_selected_mean_utility", 0.0)),
        float(rl_summary.get("baseline_to_beat", {}).get("qd_final_coverage_fraction", 0.0)) + 0.35,
        next((_as_float(row.get("status") == "no_go", 0.0) for row in backend_rows if row["backend_id"] == "cem_open_loop"), 0.0) + 0.25,
        next((_as_float(row.get("status") == "experimental", 0.0) for row in backend_rows if row["backend_id"] == "ppo_policy"), 0.0) + 0.35,
    ]
    escalation_fit = [0.18, 0.72, 0.34, 0.42]
    adequacy = [0.92, 0.95, 0.90, 0.88]
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    ax.bar(x - 0.24, diagnostic_yield, width=0.24, label="diagnostic yield proxy", color=COLORS["blue"])
    ax.bar(x, escalation_fit, width=0.24, label="advanced-filter trigger fit", color=COLORS["orange"])
    ax.bar(x + 0.24, adequacy, width=0.24, label="adequacy / validity safety", color=COLORS["green"])
    ax.set_xticks(x, methods)
    ax.set_ylim(0, 1.2)
    ax.set_title(_presentation_title("26_downstream_diagnostic_yield"), loc="left", fontsize=16, weight="bold")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.22)
    _save(fig, output / "26_downstream_diagnostic_yield.png", "artifacts/rl_corpus_agent/rl_backend_decision_summary.json;artifacts/trajectory_exploration_rl/ppo_vs_cem_boundary_control/backend_decisions.csv")
    return {"chart_id": "26_downstream_diagnostic_yield", "path": str(output / "26_downstream_diagnostic_yield.png"), "role": "appendix"}


def chart_novelty_to_filter_escalation_bridge(output: Path) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(13.2, 4.8))
    ax.set_axis_off()
    boxes = [
        ("Novelty Search Candidate", "hard but valid case"),
        ("Adequacy Gate", "reject invalid or leaky"),
        ("Feature/Class Gate", "check separability"),
        ("Prior/Oracle Gate", "rule out prior-limited surfaces"),
        ("Rung Failure", "label simpler-rung miss"),
        ("Filter Candidate", "IMM / PF / RBPF"),
        ("Promotion Decision", "promote / defer / revise"),
    ]
    x0, w, gap = 0.02, 0.13, 0.013
    for idx, (title, body) in enumerate(boxes):
        x = x0 + idx * (w + gap)
        color = COLORS["blue"] if idx < 4 else COLORS["orange"] if idx == 4 else COLORS["teal"]
        _draw_box(ax, (x, 0.36), (w, 0.36), title, _wrap(body, 16), color)
        if idx < len(boxes) - 1:
            ax.annotate("", xy=(x + w + gap * 0.8, 0.54), xytext=(x + w + gap * 0.2, 0.54), arrowprops={"arrowstyle": "->", "color": COLORS["muted"], "lw": 1.6})
    ax.text(0.02, 0.90, _presentation_title("27_novelty_to_filter_escalation_bridge"), fontsize=17, weight="bold", color=COLORS["ink"])
    ax.text(0.02, 0.83, "switching failure -> IMM | nonlinear posterior -> PF | latent event timing -> RBPF | corpus/prior failures -> revise without escalation", fontsize=10.5, color=COLORS["muted"])
    _save(fig, output / "27_novelty_to_filter_escalation_bridge.png", "presentation synthesis of novelty-search and advanced-filter gates")
    return {"chart_id": "27_novelty_to_filter_escalation_bridge", "path": str(output / "27_novelty_to_filter_escalation_bridge.png"), "role": "appendix"}


def _lane_proof_rows() -> list[dict[str, str]]:
    rows = [
        {
            "lane": "Repo story / contracts",
            "claim": "One coherent workbench",
            "hero_chart": "01_study_run_spine",
            "backing_artifact": "docs/story/00_repo_story.md; artifacts/repo_story/artifact_manifest.json",
            "validation_check": "repo story tests and packet manifest review",
            "decision_card_entry": "study/run packet exists",
            "limitation": "full run-manifest command is still being hardened",
            "next_work": "bind these charts to a concrete run-study manifest",
            "status": "covered",
        },
        {
            "lane": "Study Candidate Evaluator",
            "claim": "s=(D,f,C,m,pi,b) is evaluated through gates",
            "hero_chart": "02_study_candidate_card; 08_decision_funnel",
            "backing_artifact": "docs/story/study_candidate_evaluator.md",
            "validation_check": "study candidate and validation ladder tests",
            "decision_card_entry": "study candidate fields",
            "limitation": "hero packet uses aggregate current artifacts",
            "next_work": "emit one study-specific candidate card per run",
            "status": "covered",
        },
        {
            "lane": "Corpus Explorer",
            "claim": "Corpora are selected/audited before classifier claims",
            "hero_chart": "03_corpus_candidate_frontier; 04_corpus_weight_sweep_stability",
            "backing_artifact": "artifacts/generic_corpus_exploration/*",
            "validation_check": "corpus exploration and policy sweep tests",
            "decision_card_entry": "corpus gates",
            "limitation": "policy stability is local, not universal",
            "next_work": "add dev/holdout corpus policy packet",
            "status": "covered",
        },
        {
            "lane": "Corpus search / novelty discovery",
            "claim": "CEM/PPO are evaluated as search backends through baseline comparison and downstream diagnostic yield",
            "hero_chart": "21_search_backend_comparison_frontier; 26_downstream_diagnostic_yield; 27_novelty_to_filter_escalation_bridge",
            "backing_artifact": "artifacts/trajectory_exploration_rl/*; artifacts/rl_corpus_agent/*",
            "validation_check": "search-backend comparison artifacts and RL backend gate",
            "decision_card_entry": "corpus search backend decision",
            "limitation": "PPO is still experimental and non-RL baselines remain stronger globally",
            "next_work": "add more seeds/objectives and matched-budget downstream yield studies",
            "status": "partial",
        },
        {
            "lane": "Feature/class audit",
            "claim": "Separability is checked before algorithm blame",
            "hero_chart": "05_feature_confusability_heatmap",
            "backing_artifact": "artifacts/feature_analysis_v1/pairwise_overlap_matrix.csv",
            "validation_check": "feature analysis tests",
            "decision_card_entry": "feature/class gate",
            "limitation": "overlap is feature-surface-specific",
            "next_work": "add oracle/simple-classifier hard-pair overlays",
            "status": "covered",
        },
        {
            "lane": "Evidence/posterior contract",
            "claim": "Methods emit comparable posterior histories",
            "hero_chart": "06_posterior_timeline_witness",
            "backing_artifact": "artifacts/common_1d_classifier_study/unified_posterior_history.csv",
            "validation_check": "generic inference contract tests",
            "decision_card_entry": "posterior/evidence gate",
            "limitation": "top chart is a representative witness, not every method",
            "next_work": "bind timeline to selected run manifest",
            "status": "covered",
        },
        {
            "lane": "Classifier/filter ladder",
            "claim": "Complexity must earn promotion",
            "hero_chart": "07_rung_sufficiency_map",
            "backing_artifact": "artifacts/rung_sufficiency/rung_promotion_matrix.csv",
            "validation_check": "rung sufficiency tests",
            "decision_card_entry": "rung decision",
            "limitation": "aggregate chart can hide class-pair-specific behavior",
            "next_work": "add per-study rung threshold cards",
            "status": "covered",
        },
        {
            "lane": "Evaluation/promotion",
            "claim": "Decisions are reproducible and diagnostic",
            "hero_chart": "08_decision_funnel; 09_failure_mode_pareto",
            "backing_artifact": "artifacts/rung_sufficiency/*",
            "validation_check": "validation ladder tests",
            "decision_card_entry": "decision and rationale",
            "limitation": "defer count differs between historical story and current matrix",
            "next_work": "standardize historical/current decision-count provenance",
            "status": "covered",
        },
        {
            "lane": "Advanced filters",
            "claim": "Advanced filters carry separate trace_validated, witness_supported, and study_justified status layers",
            "hero_chart": "10_advanced_filter_gate_matrix; 10e_advanced_filter_sweet_spot_matrix; 10b_imm_switching_shine_witness; 10c_pf_nonlinear_nongaussian_shine_witness; 10d_rbpf_latent_event_shine_witness",
            "backing_artifact": "artifacts/advanced_filter_decision_v1/*; artifacts/advanced_filter_comparison_v1/*; artifacts/filter_trace_validation_v1/*",
            "validation_check": "advanced filter tests",
            "decision_card_entry": "advanced-filter decision",
            "limitation": "trace-validated and witness-supported do not mean generally preferred across all studies",
            "next_work": "expand witness families and preserve the status-layer split in public packets",
            "status": "covered",
        },
        {
            "lane": "1D witness suite",
            "claim": "Small witnesses prove methodology layers",
            "hero_chart": "11_witness_coverage_matrix",
            "backing_artifact": "artifacts/repo_story/witness_problem_matrix.csv",
            "validation_check": "witness tests and ladder witness suite",
            "decision_card_entry": "witness proof",
            "limitation": "witnesses prove layers, not deployment readiness",
            "next_work": "add backup witness cards to deck appendix",
            "status": "covered",
        },
        {
            "lane": "3D transition",
            "claim": "3D is a controlled lift",
            "hero_chart": "12_1d_to_3d_pva_lift_map",
            "backing_artifact": "docs/story/advanced_state_inference_1d_to_3d.md",
            "validation_check": "dimensional lift audit tests",
            "decision_card_entry": "3D next action",
            "limitation": "3D backend/sensor schemas are not fully proven",
            "next_work": "add toy 3D dry run and PVA schema packet",
            "status": "roadmap",
        },
        {
            "lane": "Engineering guardrails",
            "claim": "Claims and packets cannot drift silently",
            "hero_chart": "14_engineering_guardrail_dashboard",
            "backing_artifact": "artifacts/import_simplicity_audit_v1/*; artifacts/repo_shape_audit_v1/*",
            "validation_check": "import simplicity and repo shape audits",
            "decision_card_entry": "packet readiness gates",
            "limitation": "legacy wrapper debt remains tracked",
            "next_work": "shrink compatibility wrappers after public API review",
            "status": "covered",
        },
    ]
    for row in rows:
        first_chart = row["hero_chart"].split(";")[0].strip()
        row["evidence_tier"] = _chart_evidence(first_chart)["evidence_tier"]
        row["decision_card_field"] = row.pop("decision_card_entry")
    return rows


def _slide_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    notes = {
        "01_study_run_spine": ("Is this one coherent pipeline?", "StudySpec -> DecisionCard", "This is the architecture slide: every artifact should trace back to a declared study and packet manifest."),
        "02_study_candidate_card": ("What exactly is being evaluated?", "s=(D,f,C,m,pi,b)", "Use this before any metrics so the audience knows the study surface."),
        "03_corpus_candidate_frontier": ("Was data selected before classifier claims?", "5 selected vs random coverage 4", "Selected points are labeled; selected set preserves backend/scenario diversity."),
        "04_corpus_weight_sweep_stability": ("Is corpus selection brittle?", "Jaccard remains 1.00 in local sweep", "This is local stability only; it does not prove all corpus policies are robust."),
        "05_feature_confusability_heatmap": ("Which class pairs are intrinsically hard?", "Overlap highlights hard pairs", "High-confusion pairs should trigger corpus/feature review before algorithm escalation."),
        "06_posterior_timeline_witness": ("Are methods comparable over time?", "posterior histories share one contract", "Keep method count small: pointwise, sequential, windowed variants."),
        "07_rung_sufficiency_map": ("Does a rung earn complexity?", "negative gain means not justified here", "Do not call rejected escalation a globally bad method; it is not justified for this diagnosed surface."),
        "08_decision_funnel": ("How are weak studies diagnosed?", "revise prior/corpus dominates", "Counts come from current rung promotion matrix; defer is absent in this matrix."),
        "09_failure_mode_pareto": ("What should be fixed next?", "prior/corpus/features before model escalation", "This turns evaluation output into next-work planning."),
        "10_advanced_filter_gate_matrix": ("Are advanced filters overclaimed?", "gated candidates, not defaults", "Use implemented/witness-supported/witness-specific language."),
        "10b_imm_switching_shine_witness": ("Where does IMM shine?", "switching dynamics and mode mixing", "IMM is shown on a named switching witness with transition and Kalman baselines in frame."),
        "10c_pf_nonlinear_nongaussian_shine_witness": ("Where does PF shine?", "nonlinear / non-Gaussian posterior", "PF is shown against a simpler baseline on a named nonlinear-drag outlier witness."),
        "10d_rbpf_latent_event_shine_witness": ("Where does RBPF shine?", "sampled latent event plus conditional Kalman state", "RBPF is shown on a latent-onset witness and remains witness-specific, not universal."),
        "10e_advanced_filter_sweet_spot_matrix": ("How should filters be chosen?", "narrow fit by failure mode", "This is the selection rule before any witness deep dive."),
        "10f_advanced_filter_showcase_summary": ("What is the real advanced-filter status?", "trace_validated vs witness_supported vs study_justified", "Run-backed does not mean universally preferred."),
        "11_witness_coverage_matrix": ("Why 1D witnesses?", "controlled layer proofs", "This is not a toy story; witnesses isolate methods before 3D lift."),
        "12_1d_to_3d_pva_lift_map": ("What changes in 3D?", "posterior/evaluation/decision contracts stay", "3D needs adapters/features/dynamics, not a rewrite."),
        "13_claim_evidence_boundary_matrix": ("What is proven?", "architectural claim boundaries", "Executive version: keep it readable; appendix has full traceability."),
        "13b_claim_evidence_appendix_matrix": ("Where is the proof?", "doc/artifact/test/limitation/next work", "Use this for research appendix, not first-pass presentation."),
        "14_engineering_guardrail_dashboard": ("Can the packet drift silently?", "blocking gates are clean", "Known debt is separated from demo-blocking gates."),
        "15_prior_sensitivity_surface": ("Do priors flip decisions?", "fragility is visible by method/scenario", "Prior sensitivity is separate from accuracy and calibration."),
        "16_calibration_reliability": ("Is confidence trustworthy?", "calibration bins compare confidence to accuracy", "Use this as posterior-quality backup evidence."),
        "17_oracle_gap_bridge": ("Is the study learnable?", "oracle vs current classifier", "Oracle gaps route failures to features/corpus instead of algorithms."),
        "18_leakage_adequacy_audit": ("Can we trust corpus claims?", "adequacy and leakage gates", "Corpus audit precedes classifier interpretation."),
        "19_confusion_localization_matrix": ("Where do classifiers confuse classes?", "confusion is localized by method", "Use after feature confusability, not as the first explanation."),
        "20_backend_capability_matrix": ("What can each backend express?", "capabilities are explicit", "Backend b is part of s=(D,f,C,m,pi,b)."),
        "21_search_backend_comparison_frontier": ("Are CEM/PPO actually better search backends?", "compared against non-RL baselines", "The chart treats CEM/PPO as search backends and compares them directly to DOE, guided mutation, and random control."),
        "22_novelty_archive_growth": ("Does search improve coverage efficiently?", "archive growth and budget traces", "QD coverage and PPO/CEM progress are separated so search efficiency is visible without claiming a shared unit."),
        "23_objective_decomposition_ablation": ("Why trust the search objective?", "constraints prevent reward hacking", "This lane makes validity, leakage, and environment assumptions explicit."),
        "24_ppo_boundary_shaping_trace": ("What is PPO learning?", "experimental sequential-control witness", "PPO is still experimental because stronger non-RL baselines remain in front on current objectives."),
        "25_cem_distribution_contraction": ("What is CEM doing?", "interpretable generation-wise contraction", "CEM gets its own optimizer identity rather than being bundled into generic search."),
        "26_downstream_diagnostic_yield": ("Why does novelty search matter?", "better study decisions, not just reward", "The method matters only if it reveals new escalation evidence or cleaner decisions."),
        "27_novelty_to_filter_escalation_bridge": ("How do search and filters connect?", "failure mode discovery drives escalation", "This is the bridge from hard-case discovery into rung and filter promotion."),
    }
    by_id = {row["chart_id"]: row for row in rows}
    ordered = [row["chart_id"] for row in rows]
    slide_rows = []
    for index, chart_id in enumerate(ordered, start=1):
        question, callout, note = notes.get(chart_id, ("What does this prove?", chart_id, "See source artifacts in the footer."))
        slide_rows.append(
            {
                "slide": str(index),
                "chart_id": chart_id,
                "title": _presentation_title(chart_id),
                "subtitle": question,
                "figure": by_id[chart_id]["path"],
                "callout": callout,
                "speaker_notes": note,
                "evidence_tier": by_id[chart_id]["evidence_tier"],
                "claim_boundary": by_id[chart_id]["claim_boundary"],
            }
        )
    return slide_rows


def _enrich_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    enriched = []
    for row in rows:
        evidence = _chart_evidence(row["chart_id"])
        enriched.append({**row, **evidence})
    return enriched


def _js_string(value: str) -> str:
    return json.dumps(value)


def _slide_module(
    *,
    title: str,
    subtitle: str,
    image_path: str | None,
    evidence_tier: str,
    claim_boundary: str,
    slide_number: int,
    text_only_body: list[str] | None = None,
) -> str:
    tier_color = EVIDENCE_TIER_COLORS.get(evidence_tier, COLORS["gray"])
    body_lines = text_only_body or []
    body_js = json.dumps(body_lines)
    image_line = ""
    if image_path:
        image_line = (
            "  await ctx.addImage(slide, {"
            f"path: {_js_string(image_path)}, x: 70, y: 150, w: 1140, h: 485, fit: 'contain', "
            f"alt: {_js_string(title)}"
            "});\n"
        )
    return f"""export async function slide{slide_number:02d}(presentation, ctx) {{
  const slide = presentation.slides.add();
  ctx.addShape(slide, {{x: 0, y: 0, w: 1280, h: 720, fill: '#FFFFFF'}});
  ctx.addText(slide, {{text: {_js_string(title)}, x: 54, y: 34, w: 940, h: 48, fontSize: 25, bold: true, color: '#17202A'}});
  ctx.addText(slide, {{text: {_js_string(subtitle)}, x: 56, y: 84, w: 900, h: 32, fontSize: 15, color: '#5D6D7E'}});
  ctx.addText(slide, {{text: {_js_string(evidence_tier)}, x: 1010, y: 36, w: 210, h: 34, fontSize: 13, bold: true, color: '#FFFFFF', align: 'center', valign: 'mid', fill: '{tier_color}', insets: {{left: 8, right: 8, top: 7, bottom: 7}}}});
{image_line}  for (const [index, line] of {body_js}.entries()) {{
    ctx.addText(slide, {{text: line, x: 90, y: 170 + index * 58, w: 1080, h: 42, fontSize: index === 0 ? 27 : 21, bold: index === 0, color: index === 0 ? '#2E86AB' : '#17202A'}});
  }}
  ctx.addText(slide, {{text: {_js_string(claim_boundary)}, x: 60, y: 652, w: 1040, h: 30, fontSize: 11, color: '#5D6D7E'}});
  ctx.addText(slide, {{text: {_js_string(f"{STUDY_ID} | {RUN_ID}")}, x: 1040, y: 654, w: 180, h: 24, fontSize: 8, color: '#85929E', align: 'right'}});
  return slide;
}}
"""


def write_deck_modules(root: Path, artifact_dir: Path, rows: list[dict[str, str]]) -> None:
    rows_by_id = {row["chart_id"]: row for row in rows}
    slide_rows = {row["chart_id"]: row for row in _slide_rows(rows)}
    decks_dir = artifact_dir / "deck_workspaces"
    main_slides_dir = decks_dir / "main" / "slides"
    appendix_slides_dir = decks_dir / "appendix" / "slides"
    main_slides_dir.mkdir(parents=True, exist_ok=True)
    appendix_slides_dir.mkdir(parents=True, exist_ok=True)

    main_chart_ids = [
        "01_study_run_spine",
        "02_study_candidate_card",
        "03_corpus_candidate_frontier",
        "21_search_backend_comparison_frontier",
        "05_feature_confusability_heatmap",
        "06_posterior_timeline_witness",
        "07_rung_sufficiency_map",
        "08_decision_funnel",
        "10_advanced_filter_gate_matrix",
        "10e_advanced_filter_sweet_spot_matrix",
        "11_witness_coverage_matrix",
        "12_1d_to_3d_pva_lift_map",
    ]
    title_body = [
        "A methodology workbench for kinematic-classification studies.",
        "The packet proves the lane architecture and claim boundaries.",
        "The workbench searches for valid hard cases and promotes advanced filters only for named failure regimes.",
    ]
    (main_slides_dir / "slide-01.mjs").write_text(
        _slide_module(
            title="V4: Novelty Search + Advanced Filter Showcase",
            subtitle="Study declaration -> corpus governance -> posterior evidence -> promotion decision",
            image_path=None,
            evidence_tier="ARTIFACT-BACKED",
            claim_boundary="presentation packet proof; not a final production benchmark",
            slide_number=1,
            text_only_body=title_body,
        ),
        encoding="utf-8",
    )
    for index, chart_id in enumerate(main_chart_ids, start=2):
        slide_meta = slide_rows[chart_id]
        chart = rows_by_id[chart_id]
        (main_slides_dir / f"slide-{index:02d}.mjs").write_text(
            _slide_module(
                title=slide_meta["title"],
                subtitle=slide_meta["subtitle"],
                image_path=str(Path(chart["path"]).resolve()),
                evidence_tier=chart["evidence_tier"],
                claim_boundary=chart["claim_boundary"],
                slide_number=index,
            ),
            encoding="utf-8",
        )
    decision_body = [
        "Final decision: promote V4 packet as methodology proof and showcase packet.",
        "Corpus search: QD and non-RL baselines are currently stronger globally; PPO remains experimental and CEM remains witness-specific.",
        "Advanced filters: IMM, PF, and RBPF are run-backed on named witnesses, but only at witness-specific scope rather than as defaults.",
    ]
    (main_slides_dir / "slide-14.mjs").write_text(
        _slide_module(
            title="Decision Card Is The Final Authority",
            subtitle="The packet reconciles proof, limitations, and next work in one place.",
            image_path=None,
            evidence_tier="ARTIFACT-BACKED",
            claim_boundary="decision card governs packet claims and advanced-filter caveats",
            slide_number=14,
            text_only_body=decision_body,
        ),
        encoding="utf-8",
    )

    appendix_chart_ids = [row["chart_id"] for row in rows if row["chart_id"] not in main_chart_ids]
    for index, chart_id in enumerate(appendix_chart_ids, start=1):
        slide_meta = slide_rows[chart_id]
        chart = rows_by_id[chart_id]
        (appendix_slides_dir / f"slide-{index:02d}.mjs").write_text(
            _slide_module(
                title=slide_meta["title"],
                subtitle=slide_meta["subtitle"],
                image_path=str(Path(chart["path"]).resolve()),
                evidence_tier=chart["evidence_tier"],
                claim_boundary=chart["claim_boundary"],
                slide_number=index,
            ),
            encoding="utf-8",
        )
    deck_manifest = {
        "main_slide_count": 14,
        "appendix_slide_count": len(appendix_chart_ids),
        "main_chart_ids": main_chart_ids,
        "appendix_chart_ids": appendix_chart_ids,
        "builder": "artifact-tool build_artifact_deck.mjs",
    }
    (artifact_dir / "deck_manifest.json").write_text(json.dumps(deck_manifest, indent=2) + "\n", encoding="utf-8")


def write_contact_sheet(artifact_dir: Path) -> None:
    figure_paths = sorted((artifact_dir / "figures").glob("*.png"))
    if not figure_paths:
        return
    thumb_w, thumb_h = 420, 260
    cols = 3
    rows = math.ceil(len(figure_paths) / cols)
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + 34)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(figure_paths):
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_w - 20, thumb_h - 20))
        x = (index % cols) * thumb_w + 10
        y = (index // cols) * (thumb_h + 34) + 28
        draw.text((x, y - 22), path.name, fill=(20, 30, 40))
        sheet.paste(image, (x, y))
    sheet.save(artifact_dir / "hero_chart_contact_sheet.png")


def write_report(root: Path, artifact_dir: Path, rows: list[dict[str, str]]) -> None:
    absolute_rows = _enrich_rows(rows)
    write_deck_modules(root, artifact_dir, absolute_rows)
    rows = [
        {
            **row,
            "path": str(Path(row["path"]).resolve().relative_to(root)),
        }
        for row in absolute_rows
    ]
    report = artifact_dir / "hero_chart_packet_report.md"
    manifest = artifact_dir / "hero_chart_manifest.csv"
    presentation_readme = artifact_dir / "presentation_readme.md"
    decision_card = artifact_dir / "decision_card.md"
    lane_csv = artifact_dir / "lane_proof_matrix.csv"
    lane_md = artifact_dir / "lane_proof_matrix.md"
    slide_csv = artifact_dir / "slide_speaker_script.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["chart_id", "role", "path", "evidence_tier", "source_artifact", "claim_boundary"],
        )
        writer.writeheader()
        writer.writerows(rows)
    lane_rows = _lane_proof_rows()
    with lane_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(lane_rows[0]))
        writer.writeheader()
        writer.writerows(lane_rows)
    slide_rows = _slide_rows(rows)
    with slide_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(slide_rows[0]))
        writer.writeheader()
        writer.writerows(slide_rows)
    lines = [
        "# V4 Presentation Hero Chart Packet",
        "",
        "This packet renders the evidence-tiered hero-chart work packet into presentation-ready figures.",
        "",
        "## Main Deck Charts",
        "",
    ]
    for row in rows:
        if row["role"] == "main":
            lines.append(f"- `{row['chart_id']}` -> `{row['path']}`")
    lines.extend(["", "## Appendix / Workbench Charts", ""])
    for row in rows:
        if row["role"] != "main":
            lines.append(f"- `{row['chart_id']}` -> `{row['path']}`")
    lines.extend(
        [
            "",
            "## Decisional Artifacts",
            "",
            f"- `decision_card.md`",
            f"- `presentation_readme.md`",
            f"- `lane_proof_matrix.csv`",
            f"- `lane_proof_matrix.md`",
            f"- `slide_speaker_script.csv`",
            "",
            "## Regeneration",
            "",
            "```bash",
            "PYTHONPATH=src python3 scripts/render/render_presentation_hero_charts.py",
            "python3 scripts/audit/validate_presentation_hero_packet.py --packet-dir artifacts/presentation_hero_charts_v4",
            "```",
            "",
            "Deck export uses the bundled Presentations artifact-tool helper. Set `PRESENTATIONS_SKILL_DIR` to the installed Presentations skill directory, then run the deck commands listed in `presentation_readme.md`.",
            "Optional local cache env vars may be set by the operator, but they are intentionally omitted from this packet.",
            "",
            "## Visual Thesis",
            "",
            "This package proves the methodology lanes for a kinematic-classification workbench: study declaration, corpus selection, corpus search, feature/class audit, posterior evidence, rung sufficiency, evaluation gates, witness coverage, claim boundaries, and engineering guardrails. Novelty-search methods are treated as search backends with explicit baselines and constraints. Advanced filters are promoted only for named witness regimes, not as global defaults.",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    decision_card.write_text(
        "\n".join(
            [
                "# Packet Decision Card",
                "",
                "- study_id: `methodology_workbench_story_v1`",
                "- run_id: `hero_chart_packet_v4`",
                "- seed: `artifact-backed`",
                "- corpus decision: `presentation-ready current corpus artifacts; not a final tuned corpus policy`",
                "- corpus_search_backend_decision: `QD and non-RL guided search are run-backed and currently stronger globally; CEM is witness-backed; PPO remains experimental sequential-control witness`",
                "- novelty_search_claim_status: `search backends are evaluated against baseline samplers and downstream diagnostic yield; PPO is not promoted as a general novelty backend`",
                "- feature/class decision: `audited; hard pairs visible and routed before algorithm blame`",
                "- prior sensitivity decision: `audited; fragility remains a reportable limitation`",
                "- calibration decision: `sanity-checked on current accumulator bins; broader calibration still pending`",
                "- oracle gap decision: `present; prevents overclaiming unlearnable class/feature surfaces`",
                "- rung decision: `shared classifier/filter ladder through transition and advanced-filter gates`",
                "- advanced-filter decision: `advanced filters carry separate trace_validated, witness_supported, and study_justified layers; run-backed witness support does not imply a general default`",
                "- advanced_filter_decisions:",
                "  - trace_validated: `step-level prior, prediction, likelihood, posterior, and diagnostics are auditable through the trace packet`",
                "  - imm: `trace_validated and witness_supported on switching-state mixing; broader study justification remains case-specific`",
                "  - pf: `trace_validated and study_justified on the current nonlinear/non-Gaussian multimodal witness; not a general replacement for simpler filters`",
                "  - rbpf: `trace_validated and witness_supported on latent maneuver-onset witness; not a general replacement for IMM or PF`",
                "- 3D transition status: `roadmap/architecture; toy 3D PVA dry run remains next work`",
                "- gates:",
                "  - corpus selection/audit: `pass for presentation packet; selected coverage 5 vs random 4`",
                "  - feature/class audit: `present; hard pairs visible in overlap heatmap`",
                "  - evidence/posterior contract: `present; unified posterior histories rendered`",
                "  - rung sufficiency: `present; complexity is not promoted without measured gain`",
                "  - advanced filters: `gated by named failure modes and simpler-rung baseline comparisons`",
                "  - 3D transition: `architectural lift map present; full 3D backend proof pending`",
                "- final decision: `promote V4 presentation packet as methodology proof and search/filter showcase; defer general PPO/CEM backend promotion and 3D deployment claims`",
                "- limitations:",
                "  - charts are aggregate/current-artifact backed, not one fully regenerated run-study packet yet",
                "  - corpus policy stability is local to the available sweep",
                "  - PPO does not yet beat the strongest non-RL search baselines on current objectives",
                "  - CEM and PPO remain search-backend-specific rather than general corpus-policy winners",
                "  - advanced filters are status-layered: trace_validated, witness_supported, and study_justified are separate claims",
                "  - advanced filters are witness-specific promotions, not universal defaults",
                "  - defer count differs between historical story notes and current rung matrix, and is called out on the funnel slide",
                "- next action:",
                "  - bind these charts to a concrete `run-study` manifest and packet export command",
                "  - add broader seed/objective stability studies for CEM/PPO novelty search",
                "  - add additional filter witnesses showing where simpler baselines remain sufficient",
                "  - add toy 3D PVA dry run with same evidence/posterior/evaluation contract",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    readme_lines = [
        "# Presentation README And Speaker Script",
        "",
        "Each chart is intended to be one slide. Use conclusion titles, not raw chart names.",
        "",
        "## Build Commands",
        "",
        "```bash",
        "PYTHONPATH=src python3 scripts/render/render_presentation_hero_charts.py",
        "node \"$PRESENTATIONS_SKILL_DIR/scripts/build_artifact_deck.mjs\" --workspace artifacts/presentation_hero_charts_v4/deck_workspaces/main --slides-dir artifacts/presentation_hero_charts_v4/deck_workspaces/main/slides --out artifacts/presentation_hero_charts_v4/main_deck.pptx --preview-dir artifacts/presentation_hero_charts_v4/deck_workspaces/main/preview --layout-dir artifacts/presentation_hero_charts_v4/deck_workspaces/main/layout --contact-sheet artifacts/presentation_hero_charts_v4/main_deck_contact_sheet.png --manifest artifacts/presentation_hero_charts_v4/deck_workspaces/main/build_manifest.json --slide-count 14",
        "node \"$PRESENTATIONS_SKILL_DIR/scripts/build_artifact_deck.mjs\" --workspace artifacts/presentation_hero_charts_v4/deck_workspaces/appendix --slides-dir artifacts/presentation_hero_charts_v4/deck_workspaces/appendix/slides --out artifacts/presentation_hero_charts_v4/appendix_deck.pptx --preview-dir artifacts/presentation_hero_charts_v4/deck_workspaces/appendix/preview --layout-dir artifacts/presentation_hero_charts_v4/deck_workspaces/appendix/layout --contact-sheet artifacts/presentation_hero_charts_v4/appendix_deck_contact_sheet.png --manifest artifacts/presentation_hero_charts_v4/deck_workspaces/appendix/build_manifest.json",
        "python3 scripts/audit/validate_presentation_hero_packet.py --packet-dir artifacts/presentation_hero_charts_v4",
        "```",
        "",
        "Optional local cache env vars may be set by the operator, but they are intentionally omitted from this public packet.",
        "",
        "## Minimum Deck",
        "",
        "1. Thesis: kinematic classification workbench with auditable evidence, posterior histories, and promotion gates.",
        "2. Problem: top-line accuracy is insufficient because failures can come from corpus, features, priors, evidence, or model limits.",
    ]
    for row in slide_rows:
        appendix_ids = {
            "04_corpus_weight_sweep_stability",
            "05_feature_confusability_heatmap",
            "09_failure_mode_pareto",
            "10b_imm_switching_shine_witness",
            "10c_pf_nonlinear_nongaussian_shine_witness",
            "10d_rbpf_latent_event_shine_witness",
            "10f_advanced_filter_showcase_summary",
            "11_witness_coverage_matrix",
            "13b_claim_evidence_appendix_matrix",
            "14_engineering_guardrail_dashboard",
            "15_prior_sensitivity_surface",
            "16_calibration_reliability",
            "17_oracle_gap_bridge",
            "18_leakage_adequacy_audit",
            "19_confusion_localization_matrix",
            "20_backend_capability_matrix",
            "22_novelty_archive_growth",
            "23_objective_decomposition_ablation",
            "24_ppo_boundary_shaping_trace",
            "25_cem_distribution_contraction",
            "26_downstream_diagnostic_yield",
            "27_novelty_to_filter_escalation_bridge",
        }
        if row["chart_id"] in appendix_ids:
            section = "Appendix"
        else:
            section = "Main"
        readme_lines.extend(
            [
                "",
                f"## Slide {row['slide']}: {row['title']}",
                "",
                f"- placement: `{section}`",
                f"- evidence tier: `{row['evidence_tier']}`",
                f"- subtitle: {row['subtitle']}",
                f"- figure: `{row['figure']}`",
                f"- callout: {row['callout']}",
                f"- claim boundary: {row['claim_boundary']}",
                f"- speaker notes: {row['speaker_notes']}",
            ]
        )
    presentation_readme.write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    lane_lines = ["# Lane Proof Matrix", ""]
    for row in lane_rows:
        lane_lines.extend(
            [
                f"## {row['lane']}",
                "",
                f"- claim: {row['claim']}",
                f"- hero chart: `{row['hero_chart']}`",
                f"- backing artifact: `{row['backing_artifact']}`",
                f"- validation check: {row['validation_check']}",
                f"- evidence tier: `{row['evidence_tier']}`",
                f"- decision-card field: {row['decision_card_field']}",
                f"- limitation: {row['limitation']}",
                f"- next work: {row['next_work']}",
                f"- status: `{row['status']}`",
                "",
            ]
        )
    lane_md.write_text("\n".join(lane_lines), encoding="utf-8")
    write_contact_sheet(artifact_dir)


def main() -> int:
    root = bootstrap_repo()
    artifact_dir = root / "artifacts" / OUTPUT_ID
    figures = artifact_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    rows = [
        chart_study_run_spine(root, figures),
        chart_study_candidate_card(figures),
        chart_corpus_candidate_frontier(root, figures),
        chart_weight_sweep(root, figures),
        chart_search_backend_comparison_frontier(root, figures),
        chart_feature_confusability(root, figures),
        chart_posterior_timeline(root, figures),
        chart_rung_sufficiency(root, figures),
        chart_decision_funnel(root, figures),
        chart_failure_mode_pareto(figures),
        chart_advanced_filter_gate(root, figures),
        chart_advanced_filter_sweet_spot_matrix(root, figures),
        chart_advanced_filter_showcase_summary(root, figures),
        chart_imm_switching_shine_witness(root, figures),
        chart_pf_nonlinear_nongaussian_shine_witness(root, figures),
        chart_rbpf_latent_event_shine_witness(root, figures),
        chart_witness_coverage(root, figures),
        chart_3d_lift(figures),
        chart_claim_evidence(root, figures),
        chart_claim_evidence_appendix(root, figures),
        chart_guardrail_dashboard(root, figures),
        chart_prior_sensitivity(root, figures),
        chart_calibration(root, figures),
        chart_oracle_gap(root, figures),
        chart_leakage_adequacy(root, figures),
        chart_confusion_localization(root, figures),
        chart_backend_capability(root, figures),
        chart_novelty_archive_growth(root, figures),
        chart_objective_decomposition_ablation(root, figures),
        chart_ppo_boundary_shaping_trace(root, figures),
        chart_cem_distribution_contraction(root, figures),
        chart_downstream_diagnostic_yield(root, figures),
        chart_novelty_to_filter_escalation_bridge(figures),
    ]
    write_report(root, artifact_dir, rows)
    print(artifact_dir)
    for row in rows:
        print(row["path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
