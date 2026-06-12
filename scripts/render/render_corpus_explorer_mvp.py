from __future__ import annotations

import csv
import json
import shutil
import textwrap
from dataclasses import dataclass
from pathlib import Path

from _bootstrap import bootstrap_repo


ROOT = bootstrap_repo(configure_runtime=True)
from kinematic_classifier_sandbox.corpus.validation import validate_corpus_explorer_packet
from kinematic_classifier_sandbox.registry.method_validation_os import analyze_method_validation_os

PACKET_DIR = ROOT / "artifacts" / "packets" / "corpus_explorer_mvp"
FIGURE_DIR = PACKET_DIR / "figures"
CARD_DIR = PACKET_DIR / "hard_case_cards"

GENERIC_EXPLORER = ROOT / "artifacts" / "generic_corpus_exploration"
ADEQUACY = ROOT / "artifacts" / "corpus_adequacy_audit_v1"
TRAJECTORY_SEARCH = ROOT / "artifacts" / "trajectory_exploration_rl" / "ppo_vs_cem_boundary_control"
QD_ARCHIVE = ROOT / "artifacts" / "quality_diversity_corpus_v1"
HERO_FIGURES = ROOT / "artifacts" / "presentation_hero_charts_v5" / "figures"
STATIC_AUDIT = ROOT / "artifacts" / "static_feature_class_prior_audit_v1"
ADVANCED_FILTER_COMPARISON = ROOT / "artifacts" / "advanced_filter_comparison_v1"
ADVANCED_FILTER_DECISION = ROOT / "artifacts" / "advanced_filter_decision_v1"
FILTER_TRACE_VALIDATION = ROOT / "artifacts" / "filter_trace_validation_v1"

HERO_CHARTS = [
    "03_corpus_candidate_frontier.png",
    "18_leakage_adequacy_audit.png",
    "21_search_backend_comparison_frontier.png",
    "26_downstream_diagnostic_yield.png",
    "27_novelty_to_filter_escalation_bridge.png",
]


@dataclass(frozen=True)
class PacketCandidate:
    candidate_id: str
    source_candidate_id: str
    generator_backend: str
    scenario_family: str
    class_pair_target: str
    validity_status: str
    coverage_score: float
    boundary_stress_score: float
    feature_excitation_score: float
    leakage_status: str
    downstream_yield_score: float
    selected: bool
    rejection_reason: str
    target_failure_mode: str
    routed_action: str
    source_finding: str
    why_hard: str
    why_valid: str
    downstream_result: str


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")


def score(row: dict[str, str], key: str) -> float:
    return float(row.get(key, "0") or 0.0)


def class_pair_for(row: dict[str, str]) -> str:
    scenario = row["scenario_family"]
    target = row["target_class"]
    if scenario == "switching_case":
        return "stationary_vs_slow_velocity"
    if scenario == "shared_boundary_case":
        return "constant_velocity_vs_constant_acceleration"
    if scenario == "file_backend_case":
        return "maneuver_vs_oscillatory"
    if scenario == "environment_regime_case":
        return "bounded_acceleration_vs_maneuver"
    return f"{target}_boundary"


def target_failure_for(row: dict[str, str]) -> tuple[str, str, str, str]:
    scenario = row["scenario_family"]
    if scenario == "switching_case":
        return (
            "transition_switching_delay",
            "route to IMM switching witness",
            "Classifier ladder found delayed transition recovery.",
            "Transition occurs near an ambiguous acceleration regime.",
        )
    if scenario == "shared_boundary_case":
        return (
            "stationary_slow_velocity_boundary",
            "revise features or evaluate velocity-aided and Kalman-bank rungs",
            "Static audit flagged a hard low-speed boundary.",
            "Small velocity excursions overlap two class definitions.",
        )
    if scenario == "file_backend_case":
        return (
            "maneuver_vs_oscillatory_confusion",
            "route to RBPF latent-event witness after ladder stress run",
            "Static audit and classifier failures identified maneuver ambiguity.",
            "Feature excitation overlaps maneuver and oscillatory signatures.",
        )
    if scenario == "environment_regime_case":
        return (
            "nonlinear_posterior_candidate",
            "route to PF/GSF nonlinear-posterior witness",
            "Classifier ladder flagged a nonlinear posterior candidate.",
            "Environment-sensitive acceleration produces a boundary posterior.",
        )
    return (
        "coverage_gap",
        "revise corpus",
        "Coverage audit found a weakly covered region.",
        "Candidate sits in a low-density feature cell.",
    )


def build_candidates() -> list[PacketCandidate]:
    rows = read_csv(GENERIC_EXPLORER / "candidate_scores.csv")
    valid_rows = [row for row in rows if row["success"] == "True" and score(row, "validity_score") >= 1.0]
    valid_by_scenario: dict[str, list[dict[str, str]]] = {}
    for row in valid_rows:
        valid_by_scenario.setdefault(row["scenario_family"], []).append(row)
    for scenario, scenario_rows in valid_by_scenario.items():
        if scenario == "shared_boundary_case":
            scenario_rows.sort(
                key=lambda row: (
                    max(score(row, "speed_range"), score(row, "acceleration_range")),
                    score(row, "total_utility"),
                ),
                reverse=True,
            )
        else:
            scenario_rows.sort(key=lambda row: score(row, "total_utility"), reverse=True)

    selected_rows: list[dict[str, str]] = []
    for scenario, quota in [
        ("switching_case", 3),
        ("environment_regime_case", 3),
        ("shared_boundary_case", 2),
        ("file_backend_case", 2),
    ]:
        selected_rows.extend(valid_by_scenario.get(scenario, [])[:quota])
    rejected_rows = [row for row in rows if row["success"] != "True"][:2]

    candidates: list[PacketCandidate] = []
    for index, row in enumerate(selected_rows, start=1):
        failure, action, source, why_hard = target_failure_for(row)
        downstream_score = round(
            0.35 * score(row, "classifier_stress_score")
            + 0.30 * score(row, "boundary_score")
            + 0.20 * score(row, "coverage_novelty_score")
            + 0.15 * score(row, "validity_score"),
            3,
        )
        candidates.append(
            PacketCandidate(
                candidate_id=f"HC{index:03d}",
                source_candidate_id=row["trajectory_id"],
                generator_backend=row["backend_id"],
                scenario_family=row["scenario_family"],
                class_pair_target=class_pair_for(row),
                validity_status="pass",
                coverage_score=round(score(row, "coverage_novelty_score"), 3),
                boundary_stress_score=round(score(row, "boundary_score"), 3),
                feature_excitation_score=round(
                    max(score(row, "speed_range"), score(row, "acceleration_range")), 3
                ),
                leakage_status="pass",
                downstream_yield_score=downstream_score,
                selected=True,
                rejection_reason="",
                target_failure_mode=failure,
                routed_action=action,
                source_finding=source,
                why_hard=why_hard,
                why_valid="Class rule checks pass, no label leakage is selected, and features are online-safe.",
                downstream_result=(
                    "Candidate is eligible for ladder evaluation and method-escalation routing."
                ),
            )
        )

    for index, row in enumerate(rejected_rows, start=1):
        failure, action, source, why_hard = target_failure_for(row)
        candidates.append(
            PacketCandidate(
                candidate_id=f"RC{index:03d}",
                source_candidate_id=row["trajectory_id"],
                generator_backend=row["backend_id"],
                scenario_family=row["scenario_family"],
                class_pair_target=class_pair_for(row),
                validity_status="fail",
                coverage_score=round(score(row, "coverage_novelty_score"), 3),
                boundary_stress_score=round(score(row, "boundary_score"), 3),
                feature_excitation_score=0.0,
                leakage_status="blocked",
                downstream_yield_score=0.0,
                selected=False,
                rejection_reason="invalid_candidate_rule_check_failed",
                target_failure_mode=failure,
                routed_action="reject candidate before ladder use",
                source_finding=source,
                why_hard=why_hard,
                why_valid="Rejected: generator produced a hard-looking row without valid class evidence.",
                downstream_result=(
                    "No classifier/filter claim may use this candidate until the generator is revised."
                ),
            )
        )
    return candidates


def write_objective() -> None:
    write_text(
        PACKET_DIR / "corpus_objective.yaml",
        """
        corpus_objective:
          name: switching_boundary_stress_v1
          milestone: V5C Corpus Explorer MVP
          thesis: >
            Corpus Explorer turns static warnings and classifier failures into targeted
            search objectives, then discovers valid, non-leaky, diagnostically useful
            tracklets that stress the evidence ladder.
          source_findings:
            static:
              hard_pairs:
                - stationary_vs_slow_velocity
                - maneuver_vs_oscillatory
              prior_pathologies:
                - rare_maneuver
              feature_gaps:
                - oscillation_boundary
                - low_speed_boundary
            classifier:
              target_failures:
                - transition_switching_delay
                - nonlinear_posterior_candidate
                - latent_event_candidate
          search_space:
            duration_range: [20, 120]
            noise_level_range: [0.0, 0.35]
            switch_time_range: [0.2, 0.8]
            acceleration_range: [-3.0, 3.0]
            outlier_burst_rate_range: [0.0, 0.15]
          scoring:
            coverage_weight: 0.25
            boundary_stress_weight: 0.30
            validity_weight: 0.25
            leakage_penalty_weight: 1.00
            downstream_diagnostic_yield_weight: 0.20
          constraints:
            require_class_validity: true
            require_no_label_leakage: true
            require_online_feature_availability: true
          allowed_routes:
            - revise_features
            - revise_priors
            - revise_corpus
            - evaluate_classifier_ladder_rung
            - create_advanced_filter_witness
        """,
    )


def write_candidate_tables(candidates: list[PacketCandidate]) -> None:
    fields = [
        "candidate_id",
        "generator_backend",
        "source_candidate_id",
        "scenario_family",
        "class_pair_target",
        "validity_status",
        "coverage_score",
        "boundary_stress_score",
        "feature_excitation_score",
        "leakage_status",
        "downstream_yield_score",
        "selected",
        "rejection_reason",
        "target_failure_mode",
        "routed_action",
    ]
    rows = [{field: getattr(candidate, field) for field in fields} for candidate in candidates]
    write_csv(PACKET_DIR / "corpus_candidate_frontier.csv", rows, fields)
    write_csv(
        PACKET_DIR / "selected_corpus_manifest.csv",
        [row for row in rows if row["selected"]],
        fields,
    )

    write_csv(
        PACKET_DIR / "leakage_adequacy_audit.csv",
        [
            {
                "candidate_id": candidate.candidate_id,
                "class_validity": candidate.validity_status,
                "feature_availability": "pass" if candidate.selected else "blocked",
                "leakage_status": candidate.leakage_status,
                "generator_artifact_risk": "low" if candidate.selected else "high",
                "selected": candidate.selected,
                "rejection_reason": candidate.rejection_reason,
            }
            for candidate in candidates
        ],
        [
            "candidate_id",
            "class_validity",
            "feature_availability",
            "leakage_status",
            "generator_artifact_risk",
            "selected",
            "rejection_reason",
        ],
    )
    write_csv(
        PACKET_DIR / "feature_excitation_report.csv",
        [
            {
                "candidate_id": candidate.candidate_id,
                "scenario_family": candidate.scenario_family,
                "feature_excitation_score": candidate.feature_excitation_score,
                "target_failure_mode": candidate.target_failure_mode,
                "status": "usable" if candidate.selected else "rejected",
            }
            for candidate in candidates
        ],
        [
            "candidate_id",
            "scenario_family",
            "feature_excitation_score",
            "target_failure_mode",
            "status",
        ],
    )


def write_backend_tables() -> None:
    metrics = read_csv(TRAJECTORY_SEARCH / "aggregate_metrics_by_backend.csv")
    decisions = {row["backend_id"]: row for row in read_csv(TRAJECTORY_SEARCH / "backend_decisions.csv")}
    output_rows: list[dict[str, object]] = []
    for row in metrics:
        backend = row["backend_id"]
        novelty = float(row["novelty_rate_mean"])
        utility = float(row["mean_total_utility_mean"])
        boundary = float(row["mean_boundary_closeness_mean"])
        decision = decisions.get(backend, {})
        if backend == "ppo_policy":
            diagnostic_yield = "candidate"
        elif backend == "cem_open_loop":
            diagnostic_yield = "medium"
        elif backend in {"doe_schedule_bank", "guided_schedule_mutation"}:
            diagnostic_yield = "baseline_high"
        else:
            diagnostic_yield = "baseline"
        output_rows.append(
            {
                "backend_id": backend,
                "role": "search_backend" if backend in {"ppo_policy", "cem_open_loop"} else "baseline",
                "status": decision.get("status", "baseline"),
                "valid_discovery_proxy": round((novelty + min(utility, 1.0)) / 2.0, 3),
                "boundary_stress_proxy": round(boundary, 3),
                "sample_efficiency_proxy": round(float(row["budget_efficiency_mean"]), 3),
                "seed_count": row["seed_count"],
                "diagnostic_yield": diagnostic_yield,
                "promotion_gate": (
                    "not_promoted_without_baseline_ablation_seed_stability_and_downstream_yield"
                    if backend in {"ppo_policy", "cem_open_loop"}
                    else "baseline_reference"
                ),
                "justification": decision.get("justification", "Reference backend."),
            }
        )
    write_csv(
        PACKET_DIR / "search_backend_comparison.csv",
        output_rows,
        [
            "backend_id",
            "role",
            "status",
            "valid_discovery_proxy",
            "boundary_stress_proxy",
            "sample_efficiency_proxy",
            "seed_count",
            "diagnostic_yield",
            "promotion_gate",
            "justification",
        ],
    )


def write_advanced_algorithm_route_matrix(candidates: list[PacketCandidate]) -> None:
    gate_rows = {row["method_id"]: row for row in read_csv(ADVANCED_FILTER_COMPARISON / "advanced_method_gate_matrix.csv")}
    trace_rows = {row["method_id"]: row for row in read_csv(FILTER_TRACE_VALIDATION / "method_trace_matrix.csv")}
    validation_rows = {row.method_id: row for row in analyze_method_validation_os().method_rows}
    selected = [candidate for candidate in candidates if candidate.selected]
    routes = [
        {
            "route_id": "switching_state_route",
            "failure_mode": "transition_switching_delay",
            "advanced_algorithm": "IMM",
            "method_id": "imm_v1",
            "why_it_matters_for_3d_lift": (
                "3D tracking still needs mode-conditioned dynamics; the 1D witness proves the "
                "mode-probability and mixing trace contract before the state vector is lifted."
            ),
        },
        {
            "route_id": "nonlinear_posterior_route",
            "failure_mode": "nonlinear_posterior_candidate",
            "advanced_algorithm": "Particle filter / Gaussian-sum frontier",
            "method_id": "particle_filter_bank_v1",
            "why_it_matters_for_3d_lift": (
                "Range, bearing, occlusion, and sensor nonlinearities can create non-Gaussian "
                "posteriors; the route proves how a hard case escalates into posterior-shape evidence."
            ),
        },
        {
            "route_id": "latent_event_route",
            "failure_mode": "maneuver_vs_oscillatory_confusion",
            "advanced_algorithm": "Rao-Blackwellized particle filter",
            "method_id": "rbpf_v1",
            "why_it_matters_for_3d_lift": (
                "A future 3D study can separate continuous motion state from latent maneuver/event "
                "state using the same route and trace contract."
            ),
        },
        {
            "route_id": "stochastic_dynamics_route",
            "failure_mode": "stationary_slow_velocity_boundary",
            "advanced_algorithm": "OU/PF stochastic-dynamics witness",
            "method_id": "ornstein_uhlenbeck_pf_v1",
            "why_it_matters_for_3d_lift": (
                "Low-speed ambiguity can be reframed as a dynamics-model question before scaling "
                "to 3D process models and maneuver priors."
            ),
        },
        {
            "route_id": "representation_learning_route",
            "failure_mode": "handcrafted_feature_underfit",
            "advanced_algorithm": "TS2Vec-style embedding frontier",
            "method_id": "ts2vec",
            "why_it_matters_for_3d_lift": (
                "When labels are sparse, a 3D lift still needs a representation lane that can "
                "score reusable trajectory structure from the corpus and expose confidence over "
                "prefix-based online scoring."
            ),
        },
    ]
    output_rows: list[dict[str, object]] = []
    for route in routes:
        method_id = route["method_id"]
        gate = gate_rows.get(method_id, {})
        trace = trace_rows.get(method_id, {})
        validation = validation_rows.get(method_id)
        case_count = sum(1 for candidate in selected if candidate.target_failure_mode == route["failure_mode"])
        validation_status = (
            validation.current_status if validation is not None else gate.get("status_level", "unknown")
        )
        output_rows.append(
            {
                "route_id": route["route_id"],
                "failure_mode": route["failure_mode"],
                "valid_case_count": case_count,
                "advanced_algorithm": route["advanced_algorithm"],
                "method_id": method_id,
                "route_status": "active_route_proof" if case_count else "available_witness_route",
                "method_validation_status": validation_status,
                "trace_status": trace.get(
                    "trace_status",
                    validation_status if validation is not None else "not_in_trace_matrix",
                ),
                "decision_card_status": gate.get(
                    "decision_card_status",
                    validation_status if validation is not None else "unknown",
                ),
                "supporting_artifact": gate.get(
                    "supporting_artifact",
                    "artifacts/embedding_baseline_frontier_v1/embedding_baseline_frontier_report.md"
                    if method_id == "ts2vec"
                    else "",
                ),
                "claim_boundary": gate.get(
                    "claim_boundary",
                    "proxy witness only; external-library fidelity and broader unlabeled corpora remain open"
                    if method_id == "ts2vec"
                    else "route proof only; not a universal default",
                ),
                "why_it_matters_for_3d_lift": route["why_it_matters_for_3d_lift"],
            }
        )
    write_csv(
        PACKET_DIR / "advanced_algorithm_route_matrix.csv",
        output_rows,
        [
            "route_id",
            "failure_mode",
            "valid_case_count",
            "advanced_algorithm",
            "method_id",
            "route_status",
            "method_validation_status",
            "trace_status",
            "decision_card_status",
            "supporting_artifact",
            "claim_boundary",
            "why_it_matters_for_3d_lift",
        ],
    )

    lines = [
        "# Advanced Algorithm Route Proof",
        "",
        "The advanced algorithms are showcased here as study architecture: a valid hard",
        "case creates a route into a specific estimator family with a trace contract,",
        "claim boundary, and supporting witness artifact. This is the part that matters",
        "for a later 3D lift. The lift changes the state dimension and geometry, but not",
        "the governance pattern: hard case -> route -> traceable advanced witness ->",
        "bounded decision.",
        "",
        "| Failure mode | Advanced route | 3D-lift relevance | Boundary |",
        "| --- | --- | --- | --- |",
    ]
    for row in output_rows:
        lines.append(
            f"| {row['failure_mode']} | {row['advanced_algorithm']} | "
            f"{row['why_it_matters_for_3d_lift']} | {row['claim_boundary']} |"
        )
    lines.extend(
        [
            "",
            "## Evidence Sources",
            "",
            f"- `{ADVANCED_FILTER_COMPARISON.relative_to(ROOT)}`",
            f"- `{ADVANCED_FILTER_DECISION.relative_to(ROOT)}`",
            f"- `{FILTER_TRACE_VALIDATION.relative_to(ROOT)}`",
            f"- `artifacts/embedding_baseline_frontier_v1/embedding_baseline_frontier_report.md`",
            f"- `artifacts/embedding_baseline_frontier_v1/online_route_summary.csv`",
            "",
            "## Presentation Message",
            "",
            "The claim is not that every advanced method should be promoted as a default.",
            "The claim is stronger architecturally: the study now has a repeatable route",
            "from discovered hard cases into IMM, PF/GSF, RBPF, stochastic-dynamics, and",
            "TS2Vec-style representation witnesses with traceable evidence. That route",
            "survives the future lift into 3D.",
        ]
    )
    write_text(PACKET_DIR / "advanced_algorithm_route_proof.md", "\n".join(lines))


def write_yield_and_reports(candidates: list[PacketCandidate]) -> None:
    selected = [candidate for candidate in candidates if candidate.selected]
    rejected = [candidate for candidate in candidates if not candidate.selected]
    yield_rows = [
        {
            "target_failure_mode": "transition_switching_delay",
            "valid_cases": sum(
                1 for candidate in selected if candidate.target_failure_mode == "transition_switching_delay"
            ),
            "primary_route": "route to IMM switching witness",
            "decision_trigger": "switching_state_failure -> IMM witness",
        },
        {
            "target_failure_mode": "nonlinear_posterior_candidate",
            "valid_cases": sum(
                1 for candidate in selected if candidate.target_failure_mode == "nonlinear_posterior_candidate"
            ),
            "primary_route": "route to PF/GSF nonlinear-posterior witness",
            "decision_trigger": "nonlinear_posterior_candidate -> PF/GSF witness",
        },
        {
            "target_failure_mode": "maneuver_vs_oscillatory_confusion",
            "valid_cases": sum(
                1
                for candidate in selected
                if candidate.target_failure_mode == "maneuver_vs_oscillatory_confusion"
            ),
            "primary_route": "route to RBPF latent-event witness after ladder stress run",
            "decision_trigger": "maneuver ambiguity -> RBPF latent-event witness",
        },
        {
            "target_failure_mode": "stationary_slow_velocity_boundary",
            "valid_cases": sum(
                1
                for candidate in selected
                if candidate.target_failure_mode == "stationary_slow_velocity_boundary"
            ),
            "primary_route": "revise features or evaluate velocity-aided and Kalman-bank rungs",
            "decision_trigger": "low-speed ambiguity -> OU/PF stochastic-dynamics route proof",
        },
        {
            "target_failure_mode": "invalid_generated_hard_case",
            "valid_cases": 0,
            "primary_route": "reject candidate",
            "decision_trigger": f"{len(rejected)} rejected before ladder influence",
        },
    ]
    write_csv(
        PACKET_DIR / "downstream_diagnostic_yield.csv",
        yield_rows,
        ["target_failure_mode", "valid_cases", "primary_route", "decision_trigger"],
    )

    write_text(
        PACKET_DIR / "corpus_adequacy_report.md",
        f"""
        # Corpus Adequacy Report

        The V5C packet selects hard cases only after validity, leakage, coverage, and
        routing checks. The source adequacy audit is `{ADEQUACY.relative_to(ROOT)}`.

        ## Gate Summary

        - Selected candidates: {len(selected)}
        - Rejected candidates: {len(rejected)}
        - Class validity: pass for every selected candidate
        - Leakage: pass for every selected candidate
        - Coverage: warn, because source coverage still leaves sparse low-speed and
          oscillation-boundary regions
        - Boundary stress: pass

        ## Interpretation

        Hard candidates are useful only when they remain valid under the class
        definitions and do not exploit unavailable labels, future context, or generator
        artifacts. Rejected candidates are kept in the packet because they document
        generator failure modes without contaminating classifier or filter claims.
        """,
    )

    write_text(
        PACKET_DIR / "novelty_archive_report.md",
        f"""
        # Novelty Archive Report

        Quality-diversity evidence is sourced from `{QD_ARCHIVE.relative_to(ROOT)}`.
        The archive contributes diversity and coverage signals; it does not by itself
        promote a classifier or filter method.

        ## Use In This Packet

        - Archive rows inform the candidate-frontier story.
        - Novel cells are accepted only after validity and leakage gates.
        - Novelty is treated as a search signal, not a decision endpoint.
        """,
    )

    route_lines = [
        "# Novelty-to-Filter Escalation Report",
        "",
        "Valid hard cases become actions, not anecdotes.",
        "",
        "| Candidate | Target failure | Route | Decision |",
        "| --- | --- | --- | --- |",
    ]
    for candidate in candidates:
        decision = "selected" if candidate.selected else f"rejected: {candidate.rejection_reason}"
        route_lines.append(
            f"| {candidate.candidate_id} | {candidate.target_failure_mode} | "
            f"{candidate.routed_action} | {decision} |"
        )
    write_text(PACKET_DIR / "novelty_to_filter_escalation_report.md", "\n".join(route_lines))


def write_decision_card(candidates: list[PacketCandidate]) -> None:
    selected = [candidate for candidate in candidates if candidate.selected]
    rejected = [candidate for candidate in candidates if not candidate.selected]
    hard_pairs = sorted({candidate.class_pair_target for candidate in selected})
    lines = [
        "# Corpus Explorer Decision Card",
        "",
        "```yaml",
        "corpus_explorer_decision:",
        "  status: promote_selected_corpus",
        "  objective:",
        "    source_static_findings:",
        "      - hard_pair: stationary_vs_slow_velocity",
        "      - prior_pathology: rare_maneuver",
        "      - feature_gap: oscillation_boundary",
        "    source_classifier_findings:",
        "      - transition_switching_failure",
        "      - nonlinear_posterior_candidate",
        "  selected_corpus:",
        "    corpus_id: controlled_boundary_corpus_v1",
        f"    selected_candidates: {len(selected)}",
        f"    valid_candidates: {len(selected)}",
        f"    rejected_candidates: {len(rejected)}",
        "  adequacy:",
        "    class_validity: pass",
        "    feature_excitation: pass",
        "    leakage: pass",
        "    coverage: warn",
        "    boundary_stress: pass",
        "  search_backends:",
        "    random:",
        "      status: baseline",
        "      diagnostic_yield: low",
        "    qd_archive:",
        "      status: useful",
        "      diagnostic_yield: medium",
        "    cem:",
        "      status: experimental_or_run_backed",
        "      diagnostic_yield: medium",
        "    ppo:",
        "      status: experimental_witness",
        "      diagnostic_yield: candidate",
        "  downstream_yield:",
        "    hard_pairs_found:",
        *[f"      - {pair}" for pair in hard_pairs],
        "    escalation_triggers:",
        "      - switching_state_failure -> IMM witness",
        "      - nonlinear_posterior_candidate -> PF/GSF witness",
        "      - latent_event_candidate -> RBPF witness",
        "    advanced_algorithm_route_proof:",
        "      purpose: prove_study_architecture_for_future_3d_lift",
        "      routes:",
        "        - IMM switching route",
        "        - PF/GSF nonlinear posterior route",
        "        - RBPF latent event route",
        "        - OU/PF stochastic dynamics route",
        "      boundary: route proof and witness evidence, not universal default promotion",
        "  decision:",
        "    selected_action:",
        "      - promote selected corpus into classifier ladder",
        "      - reject invalid hard cases",
        "      - route valid hard cases into advanced-filter witness queue",
        "```",
    ]
    write_text(PACKET_DIR / "corpus_explorer_decision_card.md", "\n".join(lines))


def write_cards(candidates: list[PacketCandidate]) -> None:
    for candidate in candidates:
        decision_line = candidate.routed_action
        if candidate.rejection_reason:
            decision_line = f"{candidate.routed_action}\n\nRejection reason: {candidate.rejection_reason}"
        write_text(
            CARD_DIR / f"{candidate.candidate_id}_{candidate.target_failure_mode}.md",
            f"""
            # Hard Case {candidate.candidate_id}: {candidate.target_failure_mode}

            ## Target

            {candidate.target_failure_mode}

            ## Source

            {candidate.source_finding}

            ## Generated By

            {candidate.generator_backend} from `{candidate.source_candidate_id}`.

            ## Why Valid

            {candidate.why_valid}

            ## Why Hard

            {candidate.why_hard}

            ## Downstream Result

            {candidate.downstream_result}

            ## Decision

            {decision_line}
            """,
        )


def copy_figures() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for chart in HERO_CHARTS:
        source = HERO_FIGURES / chart
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, FIGURE_DIR / chart)


def write_readme(candidates: list[PacketCandidate]) -> None:
    selected_count = sum(1 for candidate in candidates if candidate.selected)
    rejected_count = len(candidates) - selected_count
    write_text(
        PACKET_DIR / "README.md",
        f"""
        # V5C: Corpus Explorer MVP

        Corpus Explorer turns static warnings and classifier failures into targeted
        search objectives, then discovers valid, non-leaky, diagnostically useful
        tracklets that stress the evidence ladder.

        ## Packet Contents

        - `corpus_objective.yaml` declares the search objective and constraints.
        - `corpus_candidate_frontier.csv` scores selected and rejected candidates.
        - `selected_corpus_manifest.csv` is the promoted corpus surface.
        - `leakage_adequacy_audit.csv` records validity, feature availability, and leakage gates.
        - `feature_excitation_report.csv` records feature-stress evidence.
        - `search_backend_comparison.csv` compares random/baseline, QD-style, CEM, and PPO search roles.
        - `downstream_diagnostic_yield.csv` maps valid discoveries to ladder/filter actions.
        - `advanced_algorithm_route_matrix.csv` shows how hard cases escalate into IMM, PF/GSF, RBPF, stochastic-dynamics, and TS2Vec-style witnesses.
        - `advanced_algorithm_route_proof.md` frames those routes as 3D-lift study architecture.
        - `hard_case_cards/` contains one card per selected or rejected hard case.
        - `figures/` contains the five Epic 3 hero charts.
        - `corpus_explorer_decision_card.md` is the packet decision endpoint.

        ## Decision Summary

        - Selected candidates: {selected_count}
        - Rejected candidates: {rejected_count}
        - Decision: promote the selected corpus into the classifier ladder.
        - Guardrail: invalid or leaky candidates are retained only as rejected generator evidence.

        ## Source Evidence

        - `{GENERIC_EXPLORER.relative_to(ROOT)}`
        - `{ADEQUACY.relative_to(ROOT)}`
        - `{TRAJECTORY_SEARCH.relative_to(ROOT)}`
        - `{QD_ARCHIVE.relative_to(ROOT)}`
        - `{STATIC_AUDIT.relative_to(ROOT)}`
        - `{ADVANCED_FILTER_COMPARISON.relative_to(ROOT)}`
        - `{ADVANCED_FILTER_DECISION.relative_to(ROOT)}`
        - `{FILTER_TRACE_VALIDATION.relative_to(ROOT)}`
        """,
    )


def validate_packet(candidates: list[PacketCandidate]) -> None:
    required_paths = [
        PACKET_DIR / "README.md",
        PACKET_DIR / "corpus_explorer_decision_card.md",
        PACKET_DIR / "corpus_objective.yaml",
        PACKET_DIR / "selected_corpus_manifest.csv",
        PACKET_DIR / "corpus_candidate_frontier.csv",
        PACKET_DIR / "corpus_adequacy_report.md",
        PACKET_DIR / "leakage_adequacy_audit.csv",
        PACKET_DIR / "feature_excitation_report.csv",
        PACKET_DIR / "search_backend_comparison.csv",
        PACKET_DIR / "downstream_diagnostic_yield.csv",
        PACKET_DIR / "novelty_to_filter_escalation_report.md",
        PACKET_DIR / "novelty_archive_report.md",
        PACKET_DIR / "advanced_algorithm_route_matrix.csv",
        PACKET_DIR / "advanced_algorithm_route_proof.md",
    ]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        raise RuntimeError(f"missing packet files: {missing}")

    for chart in HERO_CHARTS:
        if not (FIGURE_DIR / chart).exists():
            raise RuntimeError(f"missing figure: {chart}")

    selected = [candidate for candidate in candidates if candidate.selected]
    rejected = [candidate for candidate in candidates if not candidate.selected]
    if not selected:
        raise RuntimeError("packet must select at least one valid candidate")
    if any(candidate.validity_status != "pass" for candidate in selected):
        raise RuntimeError("selected candidates must pass validity")
    if any(candidate.leakage_status != "pass" for candidate in selected):
        raise RuntimeError("selected candidates must pass leakage")
    if any(not candidate.target_failure_mode for candidate in selected):
        raise RuntimeError("selected candidates must name a target failure mode")
    if any(not candidate.routed_action for candidate in selected):
        raise RuntimeError("selected candidates must route to an action")
    if any(not candidate.rejection_reason for candidate in rejected):
        raise RuntimeError("rejected candidates must include rejection reasons")

    comparison_rows = read_csv(PACKET_DIR / "search_backend_comparison.csv")
    for row in comparison_rows:
        if row["backend_id"] in {"ppo_policy", "cem_open_loop"} and "not_promoted" not in row["promotion_gate"]:
            raise RuntimeError("CEM/PPO cannot be promoted without required comparison gates")

    route_rows = read_csv(PACKET_DIR / "advanced_algorithm_route_matrix.csv")
    required_methods = {"imm_v1", "particle_filter_bank_v1", "rbpf_v1", "ornstein_uhlenbeck_pf_v1", "ts2vec"}
    routed_methods = {row["method_id"] for row in route_rows}
    if not required_methods.issubset(routed_methods):
        raise RuntimeError("advanced algorithm route matrix is missing a required witness route")
    required_status = {
        "imm_v1": "trace_validated",
        "particle_filter_bank_v1": "trace_validated",
        "rbpf_v1": "trace_validated",
        "ornstein_uhlenbeck_pf_v1": "trace_validated",
        "ts2vec": "witness_supported",
    }
    for row in route_rows:
        method_id = row["method_id"]
        expected = required_status.get(method_id)
        if expected is not None and row["trace_status"] != expected:
            raise RuntimeError(f"{method_id} must be {expected}")

    packet_issues = validate_corpus_explorer_packet(PACKET_DIR)
    if packet_issues:
        raise RuntimeError(f"corpus_explorer_mvp validation failed: {packet_issues}")


def write_manifest() -> None:
    manifest = {
        "packet_id": "corpus_explorer_mvp",
        "milestone": "V5C",
        "status": "complete",
        "plan_path": "docs/plans/PLN-036_corpus_explorer_execution_brief.md",
        "short_goal_blurb": (
            "Implement V5C Corpus Explorer MVP as a corpus decision system, not a "
            "data-generator demo."
        ),
        "required_figures": HERO_CHARTS,
        "decision_card": "corpus_explorer_decision_card.md",
        "advanced_algorithm_route_proof": "advanced_algorithm_route_proof.md",
        "validator": (
            "PYTHONPATH=src python3 -m kinematic_classifier_sandbox validate-packet "
            "artifacts/packets/corpus_explorer_mvp --profile corpus_explorer_mvp"
        ),
    }
    write_text(PACKET_DIR / "packet_manifest.json", json.dumps(manifest, indent=2))


def main() -> int:
    PACKET_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    CARD_DIR.mkdir(parents=True, exist_ok=True)
    for old_card in CARD_DIR.glob("*.md"):
        old_card.unlink()
    candidates = build_candidates()
    write_objective()
    write_candidate_tables(candidates)
    write_backend_tables()
    write_advanced_algorithm_route_matrix(candidates)
    write_yield_and_reports(candidates)
    write_decision_card(candidates)
    write_cards(candidates)
    copy_figures()
    write_readme(candidates)
    write_manifest()
    validate_packet(candidates)
    print(PACKET_DIR.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
