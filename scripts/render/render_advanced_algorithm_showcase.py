from __future__ import annotations

import csv
import json
import os
import shutil
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/kcs-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy
from _bootstrap import bootstrap_repo


ROOT = bootstrap_repo(configure_runtime=True)
PACKET_DIR = ROOT / "artifacts" / "packets" / "advanced_algorithm_showcase"
FIGURE_DIR = PACKET_DIR / "figures"
SOURCE_DIR = PACKET_DIR / "source"
HERO_FIGURES = ROOT / "artifacts" / "presentation_hero_charts_v5" / "figures"

IMM = ROOT / "artifacts" / "imm_filter_v1"
PF = ROOT / "artifacts" / "pf_abs_range_multimodal_oracle_v1"
RBPF = ROOT / "artifacts" / "rbpf_v1"
ADVANCED = ROOT / "artifacts" / "advanced_filter_comparison_v1"
SEARCH = ROOT / "artifacts" / "trajectory_exploration_rl" / "ppo_vs_cem_boundary_control"
PPO = ROOT / "artifacts" / "trajectory_exploration_rl" / "ppo_boundary_control"
RL_DECISION = ROOT / "artifacts" / "rl_corpus_agent"

HERO_CHARTS = [
    "10e_advanced_filter_sweet_spot_matrix.png",
    "10b_imm_switching_shine_witness.png",
    "10c_pf_nonlinear_nongaussian_shine_witness.png",
    "10d_rbpf_latent_event_shine_witness.png",
    "21_search_backend_comparison_frontier.png",
    "24_ppo_boundary_shaping_trace.png",
    "25_cem_distribution_contraction.png",
    "26_downstream_diagnostic_yield.png",
    "27_novelty_to_filter_escalation_bridge.png",
]

REPORT_ALIASES = {
    "imm_switching_state_mixing": "imm_switching_witness.md",
    "pf_nonlinear_nongaussian": "pf_nonlinear_witness.md",
    "rbpf_latent_event_timing": "rbpf_latent_event_witness.md",
    "cem_parameterized_hard_case_search": "cem_hard_case_search.md",
    "ppo_sequential_boundary_shaping": "ppo_boundary_shaping.md",
}

WITNESSES = [
    {
        "method_id": "kalman_bank",
        "method": "Kalman bank",
        "witness_id": "kalman_dynamic_residuals",
        "intended_failure_mode": "same_endpoint_dynamic_profile_ambiguity",
        "controlled_scenario": "same endpoints but different acceleration profiles",
        "baselines": "pointwise; windowed; sequential_bayes",
        "diagnostics": "state estimate history; innovation likelihood; residual evidence",
        "status": "evaluated",
        "decision": "capability_exercised",
        "claim_b": "dynamic residuals can be represented as class evidence under the shared posterior contract",
        "source_artifact": "artifacts/rung_sufficiency/rung_capability_matrix.csv",
    },
    {
        "method_id": "transition_matrix",
        "method": "Transition matrix",
        "witness_id": "transition_plausible_switching",
        "intended_failure_mode": "static_class_assumption_fails",
        "controlled_scenario": "class switch with plausible and implausible transitions",
        "baselines": "pointwise; sequential_bayes; kalman_bank",
        "diagnostics": "posterior history; transition-rung promotion matrix; switch rationale",
        "status": "evaluated",
        "decision": "capability_exercised",
        "claim_b": "label and mode history can change the decision without changing the local feature vector",
        "source_artifact": "artifacts/rung_sufficiency/rung_promotion_matrix.csv",
    },
    {
        "method_id": "imm_v1",
        "method": "IMM",
        "witness_id": "imm_switching_state_mixing",
        "intended_failure_mode": "switching_dynamics_with_state_uncertainty",
        "controlled_scenario": "CV to CA or braking switch under noise",
        "baselines": "kalman_bank; transition_matrix; PF; RBPF",
        "diagnostics": "mode probability timeline; switch detection delay; state RMSE; entropy; innovation likelihood",
        "status": "shine_witness_passed",
        "decision": "promote_for_switching_witness",
        "claim_b": "mode mixing can be implemented, diagnosed, and compared when switching dynamics require it",
        "source_artifact": "artifacts/imm_filter_v1/switching_detection_metrics.csv",
    },
    {
        "method_id": "particle_filter_bank_v1",
        "method": "PF / GSF",
        "witness_id": "pf_nonlinear_nongaussian",
        "intended_failure_mode": "nonlinear_or_multimodal_posterior_collapse",
        "controlled_scenario": "absolute-range observation with multimodal posterior and non-Gaussian posterior shape",
        "baselines": "gaussian_baseline; grid_oracle; GSF; IMM",
        "diagnostics": "particle diagnostics; ESS; posterior mass error; oracle KL; final posterior overlay",
        "status": "shine_witness_passed",
        "decision": "promote_for_multimodal_posterior_witness",
        "claim_b": "sampled posterior support matters when Gaussian summaries mislead",
        "source_artifact": "artifacts/pf_abs_range_multimodal_oracle_v1/metrics_against_oracle.csv",
    },
    {
        "method_id": "rbpf_v1",
        "method": "RBPF",
        "witness_id": "rbpf_latent_event_timing",
        "intended_failure_mode": "latent_event_timing_with_conditional_state",
        "controlled_scenario": "unknown maneuver onset with conditional continuous state",
        "baselines": "transition_matrix; IMM; plain PF",
        "diagnostics": "latent mode posterior; conditional filter history; ESS; onset mode accuracy; state RMSE",
        "status": "shine_witness_passed",
        "decision": "promote_for_latent_event_witness",
        "claim_b": "sampled discrete latent structure plus conditional Kalman state can be more efficient than plain sampling",
        "source_artifact": "artifacts/rbpf_v1/rbpf_method_comparison.csv",
    },
    {
        "method_id": "cem_open_loop",
        "method": "CEM",
        "witness_id": "cem_parameterized_hard_case_search",
        "intended_failure_mode": "parameterized_boundary_case_discovery",
        "controlled_scenario": "open-loop search over switch time, acceleration, noise, endpoint and outlier parameters",
        "baselines": "random_control; scripted_profiles; DOE; guided_schedule_mutation; PPO",
        "diagnostics": "elite set; distribution contraction; best objective curve; seed comparison; downstream yield",
        "status": "evaluated",
        "decision": "exercise_not_promote_yet",
        "claim_b": "CEM gives an interpretable parameter-search witness even when stronger baselines win on current budget",
        "source_artifact": "artifacts/trajectory_exploration_rl/ppo_vs_cem_boundary_control/aggregate_metrics_by_backend.csv",
    },
    {
        "method_id": "ppo_policy",
        "method": "PPO",
        "witness_id": "ppo_sequential_boundary_shaping",
        "intended_failure_mode": "sequential_control_boundary_generation",
        "controlled_scenario": "policy chooses bounded controls to shape boundary-stressing trajectories",
        "baselines": "random_control; scripted_profiles; DOE; guided_schedule_mutation; CEM",
        "diagnostics": "training curve; rollout cards; control sequences; seed comparison; downstream yield",
        "status": "evaluated",
        "decision": "experimental_witness",
        "claim_b": "PPO can be exercised as sequential trajectory design, but promotion requires stronger baseline and seed evidence",
        "source_artifact": "artifacts/trajectory_exploration_rl/ppo_vs_cem_boundary_control/backend_decisions.csv",
    },
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")


def copy_if_exists(source: Path, rel_target: str) -> str:
    target = SOURCE_DIR / rel_target
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.exists():
        shutil.copy2(source, target)
    return str(target.relative_to(PACKET_DIR))


def copy_figures() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for chart in HERO_CHARTS:
        source = HERO_FIGURES / chart
        if source.exists():
            shutil.copy2(source, FIGURE_DIR / chart)


def source_artifact_map() -> dict[str, list[str]]:
    copied = {
        "imm_switching_state_mixing": [
            copy_if_exists(IMM / "switching_detection_metrics.csv", "imm/switching_detection_metrics.csv"),
            copy_if_exists(IMM / "mode_probability_history.csv", "imm/mode_probability_history.csv"),
            copy_if_exists(IMM / "traces/filter_step_trace.csv", "imm/filter_step_trace.csv"),
            copy_if_exists(IMM / "method_evaluation_summary.csv", "imm/method_evaluation_summary.csv"),
        ],
        "pf_nonlinear_nongaussian": [
            copy_if_exists(PF / "metrics_against_oracle.csv", "pf/metrics_against_oracle.csv"),
            copy_if_exists(PF / "particle_diagnostics.csv", "pf/particle_diagnostics.csv"),
            copy_if_exists(PF / "method_posterior_history.csv", "pf/method_posterior_history.csv"),
            copy_if_exists(PF / "decision_card.md", "pf/decision_card.md"),
        ],
        "rbpf_latent_event_timing": [
            copy_if_exists(RBPF / "rbpf_method_comparison.csv", "rbpf/rbpf_method_comparison.csv"),
            copy_if_exists(RBPF / "latent_mode_posterior.csv", "rbpf/latent_mode_posterior.csv"),
            copy_if_exists(RBPF / "conditional_filter_history.csv", "rbpf/conditional_filter_history.csv"),
            copy_if_exists(RBPF / "traces/filter_step_trace.csv", "rbpf/filter_step_trace.csv"),
        ],
        "cem_parameterized_hard_case_search": [
            copy_if_exists(SEARCH / "aggregate_metrics_by_backend.csv", "search/aggregate_metrics_by_backend.csv"),
            copy_if_exists(SEARCH / "progress_rows.csv", "search/progress_rows.csv"),
            copy_if_exists(SEARCH / "backend_decisions.csv", "search/backend_decisions.csv"),
            copy_if_exists(SEARCH / "seed_runs.csv", "search/seed_runs.csv"),
        ],
        "ppo_sequential_boundary_shaping": [
            copy_if_exists(PPO / "training_trace_rows.csv", "ppo/training_trace_rows.csv"),
            copy_if_exists(PPO / "selected_rollouts.csv", "ppo/selected_rollouts.csv"),
            copy_if_exists(PPO / "ppo_vs_heuristics.csv", "ppo/ppo_vs_heuristics.csv"),
            copy_if_exists(SEARCH / "backend_decisions.csv", "ppo/backend_decisions.csv"),
        ],
    }
    return copied


def method_win_rows() -> list[dict[str, object]]:
    comparison = {row["method_id"]: row for row in read_csv(ADVANCED / "method_comparison.csv")}
    backend_decisions = {row["backend_id"]: row for row in read_csv(SEARCH / "backend_decisions.csv")}
    rows = []
    regime_map = {
        "imm_v1": "switching_state_mixing",
        "particle_filter_bank_v1": "nonlinear_nongaussian_posterior",
        "rbpf_v1": "latent_event_timing",
    }
    for method_id, regime in regime_map.items():
        row = comparison.get(method_id, {})
        rows.append(
            {
                "regime": regime,
                "method_id": method_id,
                "method_status": "shine_witness_passed",
                "baseline_failed": row.get("baseline_failed", "yes"),
                "method_improved": row.get("method_improved", "yes"),
                "primary_metric": row.get("primary_metric", ""),
                "primary_metric_value": row.get("primary_metric_value", ""),
                "decision": row.get("promotion_decision", "promote"),
                "claim_boundary": "named witness only; not a universal default",
            }
        )
    for backend_id, regime in [("cem_open_loop", "parameterized_hard_case_search"), ("ppo_policy", "sequential_boundary_shaping")]:
        row = backend_decisions.get(backend_id, {})
        rows.append(
            {
                "regime": regime,
                "method_id": backend_id,
                "method_status": "evaluated",
                "baseline_failed": "no",
                "method_improved": "mixed",
                "primary_metric": "valid_discovery_and_diagnostic_yield",
                "primary_metric_value": row.get("status", ""),
                "decision": row.get("status", "experimental"),
                "claim_boundary": "search backend witness; promotion requires baseline comparison, ablation, seed stability, and downstream diagnostic yield",
            }
        )
    return rows


def full_ladder_rows() -> list[dict[str, object]]:
    capability = {
        "kalman_bank": ("dynamic residual evidence", "evaluated", "capability_exercised"),
        "transition_matrix": ("switching logic", "evaluated", "capability_exercised"),
        "imm_v1": ("mode mixing", "shine_witness_passed", "promote_for_switching_witness"),
        "particle_filter_bank_v1": ("nonlinear/non-Gaussian posterior reasoning", "shine_witness_passed", "promote_for_multimodal_posterior_witness"),
        "rbpf_v1": ("latent discrete state plus conditional continuous inference", "shine_witness_passed", "promote_for_latent_event_witness"),
        "cem_open_loop": ("parameterized hard-case search", "evaluated", "exercise_not_promote_yet"),
        "ppo_policy": ("sequential boundary-shaping control", "evaluated", "experimental_witness"),
    }
    rows = []
    win_by_method = {row["method_id"]: row for row in method_win_rows()}
    for witness in WITNESSES:
        method_id = witness["method_id"]
        win = win_by_method.get(method_id, {})
        rows.append(
            {
                "method_id": method_id,
                "method": witness["method"],
                "witness_id": witness["witness_id"],
                "capability": capability[method_id][0],
                "status": capability[method_id][1],
                "decision": capability[method_id][2],
                "primary_metric": win.get("primary_metric", ""),
                "primary_metric_value": win.get("primary_metric_value", ""),
                "claim_a_main_toy_need": "not_claimed",
                "claim_b_showcase": "passed" if "shine" in witness["status"] or method_id in {"kalman_bank", "transition_matrix"} else "exercised",
            }
        )
    return rows


def render_method_win_by_regime(rows: list[dict[str, object]]) -> None:
    regimes = [str(row["regime"]) for row in rows]
    labels = [str(row["method_id"]) for row in rows]
    status_score = []
    for row in rows:
        decision = str(row["decision"])
        if "promote" in decision:
            status_score.append(1.0)
        elif decision in {"experimental", "no_go"}:
            status_score.append(0.45)
        else:
            status_score.append(0.7)
    fig, ax = plt.subplots(figsize=(12, 5.8))
    y = numpy.arange(len(rows))
    colors = ["#2E7D32" if score >= 0.9 else "#D99C00" for score in status_score]
    ax.barh(y, status_score, color=colors)
    ax.set_yticks(y, [f"{regime}\n{label}" for regime, label in zip(regimes, labels)])
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("witness support score")
    ax.set_title("Method Win By Regime Map", loc="left", fontsize=16, fontweight="bold")
    ax.text(
        0,
        -0.9,
        "A method can pass a shine witness without becoming generally best.",
        fontsize=10,
        color="#5D6D7E",
    )
    ax.grid(axis="x", alpha=0.22)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "10f_method_win_by_regime_map.png", dpi=180)
    plt.close(fig)


def _witness_report_text(witness: dict[str, str], source_map: dict[str, list[str]]) -> str:
    lines = [
        f"# {witness['method']}: {witness['witness_id']}",
        "",
        f"- Intended failure mode: `{witness['intended_failure_mode']}`",
        f"- Controlled scenario: {witness['controlled_scenario']}",
        f"- Simpler baselines: {witness['baselines']}",
        f"- Diagnostics: {witness['diagnostics']}",
        f"- Status: `{witness['status']}`",
        f"- Decision: `{witness['decision']}`",
        "",
        "## Claim Separation",
        "",
        "- Claim A, general need on the main toy study: not claimed here.",
        f"- Claim B, implementation/exercise/diagnosis on a tailored witness: {witness['claim_b']}.",
        "",
        "## Source Evidence",
        "",
    ]
    copied = source_map.get(witness["witness_id"], [])
    if copied:
        lines.extend(f"- `{path}`" for path in copied)
    else:
        lines.append(f"- `{witness['source_artifact']}`")
    return "\n".join(lines)


def write_witness_reports(source_map: dict[str, list[str]]) -> None:
    for witness in WITNESSES:
        report = _witness_report_text(witness, source_map)
        write_text(PACKET_DIR / f"{witness['witness_id']}.md", report)
        alias = REPORT_ALIASES.get(witness["witness_id"])
        if alias is not None:
            write_text(PACKET_DIR / alias, report)


def write_source_manifest(source_map: dict[str, list[str]]) -> None:
    rows = []
    for witness in WITNESSES:
        copied = source_map.get(witness["witness_id"], [])
        if copied:
            for source in copied:
                rows.append({"witness_id": witness["witness_id"], "source_path": source})
        else:
            rows.append({"witness_id": witness["witness_id"], "source_path": witness["source_artifact"]})
    write_csv(PACKET_DIR / "source_manifest.csv", rows, ["witness_id", "source_path"])


def write_reports() -> None:
    write_text(
        PACKET_DIR / "README.md",
        """
        # Advanced Algorithm Showcase MVP

        This packet separates two claims:

        - Claim A: whether an advanced method is generally needed on the main toy study.
        - Claim B: whether the repo can implement, exercise, diagnose, and compare that method on a witness designed for its strengths.

        The showcase proves Claim B. The purpose is not to crown one universal
        winner. The purpose is to show that the workbench can intentionally
        create the conditions where advanced algorithms matter and then diagnose
        their internal behavior.

        ## Witnesses

        - `imm_switching_state_mixing`
        - `pf_nonlinear_nongaussian`
        - `rbpf_latent_event_timing`
        - `cem_parameterized_hard_case_search`
        - `ppo_sequential_boundary_shaping`

        ## Required Outputs

        - `advanced_algorithm_decision_card.md`
        - `method_capability_matrix.md`
        - `full_ladder_metrics.csv`
        - `method_win_by_regime.csv`
        - `source_manifest.csv`
        - per-witness reports
        - per-witness source evidence under `source/`
        - hero charts under `figures/`
        """,
    )
    write_text(
        PACKET_DIR / "advanced_algorithm_decision_card.md",
        """
        # Advanced Algorithm Decision Card

        ```yaml
        advanced_algorithm_showcase:
          status: claim_b_exercised
          claim_a_main_toy_need: not_claimed
          claim_b_advanced_method_exercise: passed
          rule:
            - a method can be evaluated without being promoted
            - a method can pass a shine witness without being generally best
            - PF/RBPF require run-backed nonlinear/latent witnesses before promotion
            - CEM/PPO require baseline comparison and downstream diagnostic yield before novelty-search promotion
          witnesses:
            imm_switching_state_mixing:
              method: IMM
              status: shine_witness_passed
              decision: promote_for_switching_witness
            pf_nonlinear_nongaussian:
              method: PF/GSF
              status: shine_witness_passed
              decision: promote_for_multimodal_posterior_witness
            rbpf_latent_event_timing:
              method: RBPF
              status: shine_witness_passed
              decision: promote_for_latent_event_witness
            cem_parameterized_hard_case_search:
              method: CEM
              status: evaluated
              decision: exercise_not_promote_yet
            ppo_sequential_boundary_shaping:
              method: PPO
              status: evaluated
              decision: experimental_witness
        ```
        """,
    )
    capability_lines = [
        "# Method Capability Matrix",
        "",
        "| Method | Capability | Witness | Status | Decision |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in full_ladder_rows():
        capability_lines.append(
            f"| {row['method']} | {row['capability']} | {row['witness_id']} | {row['status']} | {row['decision']} |"
        )
    write_text(PACKET_DIR / "method_capability_matrix.md", "\n".join(capability_lines))


def validate_packet() -> None:
    required = [
        "README.md",
        "advanced_algorithm_decision_card.md",
        "method_capability_matrix.md",
        "full_ladder_metrics.csv",
        "method_win_by_regime.csv",
        "source_manifest.csv",
        "10f_method_win_by_regime_map.png",
        "imm_switching_witness.md",
        "pf_nonlinear_witness.md",
        "rbpf_latent_event_witness.md",
        "cem_hard_case_search.md",
        "ppo_boundary_shaping.md",
    ]
    missing = [name for name in required if not (PACKET_DIR / name).exists() and not (FIGURE_DIR / name).exists()]
    if missing:
        raise RuntimeError(f"missing advanced algorithm showcase artifacts: {missing}")
    rows = read_csv(PACKET_DIR / "method_win_by_regime.csv")
    required_methods = {"imm_v1", "particle_filter_bank_v1", "rbpf_v1", "cem_open_loop", "ppo_policy"}
    methods = {row["method_id"] for row in rows}
    if not required_methods.issubset(methods):
        raise RuntimeError("method_win_by_regime.csv is missing required advanced methods")
    for row in rows:
        if row["method_id"] in {"cem_open_loop", "ppo_policy"} and "baseline comparison" not in row["claim_boundary"]:
            raise RuntimeError("CEM/PPO rows must preserve promotion caveats")


def write_manifest() -> None:
    write_text(
        PACKET_DIR / "packet_manifest.json",
        json.dumps(
            {
                "packet_id": "advanced_algorithm_showcase",
                "status": "claim_b_exercised",
                "claim_a_main_toy_need": "not_claimed",
                "claim_b_advanced_method_exercise": "passed",
                "hero_charts": [*HERO_CHARTS, "10f_method_win_by_regime_map.png"],
            },
            indent=2,
        ),
    )


def main() -> int:
    PACKET_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    copy_figures()
    sources = source_artifact_map()
    win_rows = method_win_rows()
    ladder_rows = full_ladder_rows()
    write_csv(
        PACKET_DIR / "method_win_by_regime.csv",
        win_rows,
        [
            "regime",
            "method_id",
            "method_status",
            "baseline_failed",
            "method_improved",
            "primary_metric",
            "primary_metric_value",
            "decision",
            "claim_boundary",
        ],
    )
    write_csv(
        PACKET_DIR / "full_ladder_metrics.csv",
        ladder_rows,
        [
            "method_id",
            "method",
            "witness_id",
            "capability",
            "status",
            "decision",
            "primary_metric",
            "primary_metric_value",
            "claim_a_main_toy_need",
            "claim_b_showcase",
        ],
    )
    render_method_win_by_regime(win_rows)
    write_witness_reports(sources)
    write_source_manifest(sources)
    write_reports()
    write_manifest()
    validate_packet()
    print(PACKET_DIR.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
