from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

import numpy as np

from ...utils.io import _write_json, _write_text, write_csv
from ...utils.plotting import plt, write_plot
from .contracts import TrajectoryExplorationObjective
from .ppo_boundary_control import (
    SequentialPpoConfig,
    SequentialPpoResult,
    _baseline_sequences,
    _control_signature_distance,
    _evaluate_sequence_family,
    _selected_rows,
    analyze_sequential_ppo_boundary_control,
)
from .objective_generation import generate_trajectory_exploration_objective_suite
from .sequential_gym import (
    SequentialBoundaryControlConfig,
    SequentialRolloutSummary,
    default_boundary_control_objective,
    evaluate_control_sequence,
)


@dataclass(frozen=True, slots=True)
class SequentialCemConfig:
    iterations: int = 10
    population_size: int = 24
    elite_fraction: float = 0.25
    init_std: float = 0.55
    min_std: float = 0.08
    smoothing: float = 0.75
    eval_seed_start: int = 400


@dataclass(frozen=True, slots=True)
class SequentialComparisonArtifacts:
    run_dir: Path
    config_path: Path
    artifact_manifest_path: Path
    backend_metrics_path: Path
    aggregate_backend_metrics_path: Path
    backend_decisions_path: Path
    seed_runs_path: Path
    evaluation_rows_path: Path
    selected_rollouts_path: Path
    control_sequences_path: Path
    progress_rows_path: Path
    strengths_limits_path: Path
    report_path: Path
    progress_plot_path: Path
    backend_metrics_plot_path: Path
    control_gallery_path: Path


@dataclass(frozen=True, slots=True)
class SequentialComparisonResult:
    config_payload: dict[str, object]
    backend_metrics_rows: tuple[dict[str, object], ...]
    aggregate_backend_metrics_rows: tuple[dict[str, object], ...]
    backend_decision_rows: tuple[dict[str, object], ...]
    seed_run_rows: tuple[dict[str, object], ...]
    evaluation_rows: tuple[dict[str, object], ...]
    selected_rollouts: tuple[dict[str, object], ...]
    control_rows: tuple[dict[str, object], ...]
    progress_rows: tuple[dict[str, object], ...]
    strengths_limits_rows: tuple[dict[str, object], ...]
    ppo_gallery_rollouts: tuple[SequentialRolloutSummary, ...]
    cem_gallery_rollouts: tuple[SequentialRolloutSummary, ...]
    artifact_manifest: dict[str, object]
    report_markdown: str
    artifacts: SequentialComparisonArtifacts | None = None


@dataclass(frozen=True, slots=True)
class SequentialObjectiveSweepArtifacts:
    run_dir: Path
    config_path: Path
    artifact_manifest_path: Path
    objective_summary_path: Path
    backend_summary_path: Path
    decision_summary_path: Path
    objective_backend_matrix_path: Path
    report_path: Path
    objective_backend_heatmap_path: Path


@dataclass(frozen=True, slots=True)
class SequentialObjectiveSweepResult:
    config_payload: dict[str, object]
    objective_summary_rows: tuple[dict[str, object], ...]
    backend_summary_rows: tuple[dict[str, object], ...]
    decision_summary_rows: tuple[dict[str, object], ...]
    objective_backend_matrix_rows: tuple[dict[str, object], ...]
    artifact_manifest: dict[str, object]
    report_markdown: str
    artifacts: SequentialObjectiveSweepArtifacts | None = None


def _ppo_config_for_seed(config: SequentialPpoConfig, *, seed: int) -> SequentialPpoConfig:
    return SequentialPpoConfig(
        total_timesteps=config.total_timesteps,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        eval_episodes=config.eval_episodes,
        train_seed=seed,
        eval_seed_start=seed + 200,
        hidden_sizes=config.hidden_sizes,
        progress_eval_episodes=config.progress_eval_episodes,
        checkpoint_interval_timesteps=config.checkpoint_interval_timesteps,
        snapshot_interval_timesteps=config.snapshot_interval_timesteps,
        resume_if_possible=False,
    )


def _cem_config_for_seed(config: SequentialCemConfig, *, seed: int) -> SequentialCemConfig:
    return SequentialCemConfig(
        iterations=config.iterations,
        population_size=config.population_size,
        elite_fraction=config.elite_fraction,
        init_std=config.init_std,
        min_std=config.min_std,
        smoothing=config.smoothing,
        eval_seed_start=seed + 400,
    )


def _evaluate_cem_sequences(
    *,
    config: SequentialBoundaryControlConfig,
    objective: TrajectoryExplorationObjective,
    cem_config: SequentialCemConfig,
    seed_index: int,
) -> tuple[list[SequentialRolloutSummary], list[dict[str, object]]]:
    rng = np.random.default_rng(cem_config.eval_seed_start)
    horizon = config.episode_horizon
    mean_vector = np.zeros(horizon, dtype=float)
    std_vector = np.full(horizon, cem_config.init_std, dtype=float)
    selected: list[SequentialRolloutSummary] = []
    trace_rows: list[dict[str, object]] = []
    eval_counter = 0
    for iteration in range(cem_config.iterations):
        batch: list[SequentialRolloutSummary] = []
        for population_index in range(cem_config.population_size):
            sampled = rng.normal(mean_vector, std_vector, size=horizon)
            sequence = tuple(float(value) for value in np.clip(sampled, -1.0, 1.0))
            summary = evaluate_control_sequence(
                sequence,
                objective=objective,
                config=config,
                seed=cem_config.eval_seed_start + eval_counter,
                proposal_id=f"cem_seed{seed_index}_iter{iteration}_cand{population_index}",
                backend_id="cem_open_loop",
                iteration=iteration,
                candidate_index=population_index,
            )
            batch.append(summary)
            eval_counter += 1
        ranked = sorted(batch, key=lambda summary: summary.evaluation.total_utility, reverse=True)
        elite_count = max(1, int(cem_config.population_size * cem_config.elite_fraction))
        elites = ranked[:elite_count]
        elite_array = np.asarray([summary.control_sequence for summary in elites], dtype=float)
        elite_mean = elite_array.mean(axis=0)
        elite_std = elite_array.std(axis=0)
        mean_vector = cem_config.smoothing * mean_vector + (1.0 - cem_config.smoothing) * elite_mean
        std_vector = np.maximum(cem_config.min_std, cem_config.smoothing * std_vector + (1.0 - cem_config.smoothing) * elite_std)
        selected.extend(elites[: min(4, len(elites))])
        trace_rows.append(
            {
                "seed_index": seed_index,
                "backend_id": "cem_open_loop",
                "progress_kind": "cem_iteration",
                "iteration": iteration,
                "budget_index": (iteration + 1) * cem_config.population_size,
                "mean_total_utility": mean(summary.evaluation.total_utility for summary in batch),
                "best_total_utility": elites[0].evaluation.total_utility,
                "mean_boundary_closeness": mean(summary.evaluation.boundary_closeness for summary in batch),
                "mean_feature_excitation": mean(summary.evaluation.feature_excitation for summary in batch),
                "mean_prior_sensitivity": mean(summary.evaluation.prior_sensitivity for summary in batch),
                "action_std_mean": float(std_vector.mean()),
            }
        )
    unique_selected: dict[str, SequentialRolloutSummary] = {}
    for summary in selected:
        unique_selected.setdefault(summary.proposal.proposal_id, summary)
    return list(unique_selected.values()), trace_rows


def _rollout_rows(
    summaries: list[SequentialRolloutSummary],
    *,
    backend_family: str,
    novelty_reference: list[tuple[float, ...]],
    seed_index: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    evaluation_rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []
    for summary in summaries:
        row = summary.evaluation.as_row()
        row["backend_family"] = backend_family
        row["seed_index"] = seed_index
        row["novelty_vs_scripted"] = min(
            (_control_signature_distance(summary.control_sequence, sequence) for sequence in novelty_reference),
            default=0.0,
        )
        evaluation_rows.append(row)
        for step_index, control in enumerate(summary.control_sequence):
            control_rows.append(
                {
                    "seed_index": seed_index,
                    "proposal_id": summary.proposal.proposal_id,
                    "backend_id": backend_family,
                    "step_index": step_index,
                    "control": control,
                }
            )
    return evaluation_rows, control_rows


def _backend_metrics(backend_id: str, rows: list[dict[str, object]], *, budget: int, seed_index: int) -> dict[str, object]:
    if not rows:
        return {
            "seed_index": seed_index,
            "backend_id": backend_id,
            "count": 0,
            "mean_total_utility": 0.0,
            "best_total_utility": 0.0,
            "mean_boundary_closeness": 0.0,
            "mean_feature_excitation": 0.0,
            "mean_prior_sensitivity": 0.0,
            "mean_geometry_score": 0.0,
            "novelty_rate": 0.0,
            "budget_efficiency": 0.0,
        }
    return {
        "seed_index": seed_index,
        "backend_id": backend_id,
        "count": len(rows),
        "mean_total_utility": mean(float(row["total_utility"]) for row in rows),
        "best_total_utility": max(float(row["total_utility"]) for row in rows),
        "mean_boundary_closeness": mean(float(row["boundary_closeness"]) for row in rows),
        "mean_feature_excitation": mean(float(row["feature_excitation"]) for row in rows),
        "mean_prior_sensitivity": mean(float(row["prior_sensitivity"]) for row in rows),
        "mean_geometry_score": mean(float(row["geometry_score"]) for row in rows),
        "novelty_rate": mean(1.0 if float(row.get("novelty_vs_scripted", 0.0)) > 0.18 else 0.0 for row in rows),
        "budget_efficiency": max(float(row["total_utility"]) for row in rows) / max(budget, 1),
    }


def _aggregate_backend_metrics(seed_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in seed_rows:
        grouped.setdefault(str(row["backend_id"]), []).append(row)
    aggregate_rows: list[dict[str, object]] = []
    for backend_id, rows in grouped.items():
        aggregate_rows.append(
            {
                "backend_id": backend_id,
                "seed_count": len(rows),
                "mean_total_utility_mean": mean(float(row["mean_total_utility"]) for row in rows),
                "mean_total_utility_std": pstdev(float(row["mean_total_utility"]) for row in rows) if len(rows) > 1 else 0.0,
                "best_total_utility_mean": mean(float(row["best_total_utility"]) for row in rows),
                "mean_boundary_closeness_mean": mean(float(row["mean_boundary_closeness"]) for row in rows),
                "mean_feature_excitation_mean": mean(float(row["mean_feature_excitation"]) for row in rows),
                "mean_prior_sensitivity_mean": mean(float(row["mean_prior_sensitivity"]) for row in rows),
                "mean_geometry_score_mean": mean(float(row["mean_geometry_score"]) for row in rows),
                "novelty_rate_mean": mean(float(row["novelty_rate"]) for row in rows),
                "budget_efficiency_mean": mean(float(row["budget_efficiency"]) for row in rows),
            }
        )
    return sorted(aggregate_rows, key=lambda row: float(row["mean_total_utility_mean"]), reverse=True)


def _backend_decisions(aggregate_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_backend = {str(row["backend_id"]): row for row in aggregate_rows}
    ppo = by_backend.get("ppo_policy")
    cem = by_backend.get("cem_open_loop")
    doe = by_backend.get("doe_schedule_bank")
    guided = by_backend.get("guided_schedule_mutation")
    rows: list[dict[str, object]] = []
    if ppo is not None and cem is not None:
        ppo_go = float(ppo["mean_total_utility_mean"]) > float(cem["mean_total_utility_mean"]) + 0.02
        ppo_promote = ppo_go and (
            doe is None or float(ppo["mean_total_utility_mean"]) >= float(doe["mean_total_utility_mean"]) - 0.02
        )
        rows.append(
            {
                "backend_id": "ppo_policy",
                "status": "promote" if ppo_promote else ("experimental" if ppo_go else "no_go"),
                "justification": (
                    "Beats CEM and is close to or above open-loop baselines."
                    if ppo_promote
                    else "Beats CEM but not the strongest open-loop baselines yet."
                    if ppo_go
                    else "Does not beat CEM at matched witness settings."
                ),
            }
        )
        cem_go = float(cem["mean_total_utility_mean"]) >= float(ppo["mean_total_utility_mean"])
        rows.append(
            {
                "backend_id": "cem_open_loop",
                "status": "promote" if cem_go and guided is None else ("experimental" if cem_go else "no_go"),
                "justification": (
                    "Competitive with PPO and appropriate for short-horizon open-loop search."
                    if cem_go
                    else "Below PPO on this witness."
                ),
            }
        )
    for backend_id in ("random_control", "scripted_profiles", "doe_schedule_bank", "guided_schedule_mutation"):
        if backend_id in by_backend:
            rows.append(
                {
                    "backend_id": backend_id,
                    "status": "baseline",
                    "justification": "Reference family for sanity and open-loop ceiling checks.",
                }
            )
    return rows


def _strengths_limits_rows(
    aggregate_rows: list[dict[str, object]],
    *,
    ppo_budget: int,
    cem_budget: int,
) -> list[dict[str, object]]:
    by_backend = {str(row["backend_id"]): row for row in aggregate_rows}
    rows = [
        {
            "backend_id": "ppo_policy",
            "strength": "policy reuse across episodes",
            "limit": "higher training cost and rollout-collection overhead",
            "better_when": "sequential timing and reusable control logic matter",
            "evidence": "beats_cem" if float(by_backend["ppo_policy"]["mean_total_utility_mean"]) > float(by_backend["cem_open_loop"]["mean_total_utility_mean"]) else "below_cem",
            "budget": ppo_budget,
        },
        {
            "backend_id": "cem_open_loop",
            "strength": "strong open-loop search on short horizons",
            "limit": "does not learn a reusable feedback policy",
            "better_when": "short-horizon schedule search is enough",
            "evidence": "beats_ppo" if float(by_backend["cem_open_loop"]["mean_total_utility_mean"]) >= float(by_backend["ppo_policy"]["mean_total_utility_mean"]) else "below_ppo",
            "budget": cem_budget,
        },
    ]
    for backend_id in ("random_control", "scripted_profiles", "doe_schedule_bank", "guided_schedule_mutation"):
        if backend_id in by_backend:
            rows.append(
                {
                    "backend_id": backend_id,
                    "strength": "cheap baseline",
                    "limit": "limited adaptation to target geometry",
                    "better_when": "you need a sanity floor or interpretable hand-authored schedules",
                    "evidence": "baseline",
                    "budget": int(by_backend[backend_id]["seed_count"]),
                }
            )
    return rows


def _render_progress_plot(progress_rows: list[dict[str, object]]):
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.8))
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in progress_rows:
        grouped.setdefault(str(row["backend_id"]), []).append(row)
    for backend_id, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: (int(row["seed_index"]), float(row["budget_index"])))
        axes[0].plot(
            [float(row["budget_index"]) for row in ordered],
            [float(row["best_total_utility"]) for row in ordered],
            label=backend_id,
        )
        axes[1].plot(
            [float(row["budget_index"]) for row in ordered],
            [float(row["mean_boundary_closeness"]) for row in ordered],
            label=backend_id,
        )
    axes[0].set_title("PPO vs CEM utility progress", loc="left", fontweight="bold")
    axes[0].set_ylabel("Best total utility")
    axes[1].set_title("Boundary-closeness progress", loc="left", fontweight="bold")
    axes[1].set_xlabel("Budget index")
    axes[1].set_ylabel("Mean boundary closeness")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _render_backend_metrics_plot(aggregate_rows: list[dict[str, object]]):
    ordered = sorted(aggregate_rows, key=lambda row: float(row["mean_total_utility_mean"]), reverse=True)
    names = [str(row["backend_id"]) for row in ordered]
    means = [float(row["mean_total_utility_mean"]) for row in ordered]
    stds = [float(row["mean_total_utility_std"]) for row in ordered]
    boundaries = [float(row["mean_boundary_closeness_mean"]) for row in ordered]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4))
    axes[0].bar(names, means, yerr=stds, capsize=4)
    axes[0].set_title("Mean total utility across seeds", loc="left", fontweight="bold")
    axes[1].bar(names, boundaries)
    axes[1].set_title("Mean boundary closeness across seeds", loc="left", fontweight="bold")
    for axis in axes:
        axis.tick_params(axis="x", rotation=30)
        axis.grid(alpha=0.20, axis="y")
    fig.tight_layout()
    return fig


def _render_control_gallery(ppo_summaries: list[SequentialRolloutSummary], cem_summaries: list[SequentialRolloutSummary]):
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.5))
    for summary in sorted(ppo_summaries, key=lambda item: item.evaluation.total_utility, reverse=True)[:3]:
        axes[0].plot(range(len(summary.control_sequence)), summary.control_sequence, label=summary.proposal.proposal_id)
    for summary in sorted(cem_summaries, key=lambda item: item.evaluation.total_utility, reverse=True)[:3]:
        axes[1].plot(range(len(summary.control_sequence)), summary.control_sequence, label=summary.proposal.proposal_id)
    axes[0].set_title("Top PPO control sequences", loc="left", fontweight="bold")
    axes[1].set_title("Top CEM control sequences", loc="left", fontweight="bold")
    for axis in axes:
        axis.set_xlabel("Step")
        axis.set_ylabel("Control")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _artifact_manifest(run_dir: Path) -> dict[str, object]:
    return {
        "run_dir": str(run_dir),
        "reports": ["report.md"],
        "tables": [
            "metrics_by_backend.csv",
            "aggregate_metrics_by_backend.csv",
            "backend_decisions.csv",
            "seed_runs.csv",
            "evaluation_rows.csv",
            "selected_rollouts.csv",
            "control_sequences.csv",
            "progress_rows.csv",
            "strengths_and_limits.csv",
        ],
        "plots": [
            "progress_comparison.png",
            "backend_metrics.png",
            "control_gallery.png",
        ],
    }


def _comparison_report(
    *,
    objective: TrajectoryExplorationObjective,
    aggregate_rows: list[dict[str, object]],
    decision_rows: list[dict[str, object]],
    seed_count: int,
    ppo_budget: int,
    cem_budget: int,
) -> str:
    by_backend = {str(row["backend_id"]): row for row in aggregate_rows}
    ppo = by_backend["ppo_policy"]
    cem = by_backend["cem_open_loop"]
    winner = "ppo_policy" if float(ppo["mean_total_utility_mean"]) > float(cem["mean_total_utility_mean"]) else "cem_open_loop"
    overall_best = max(aggregate_rows, key=lambda row: float(row["mean_total_utility_mean"]))
    return "\n".join(
        [
            "# Sequential PPO vs CEM Comparison Report",
            "",
            f"- objective id: `{objective.objective_id}`",
            f"- seed count: `{seed_count}`",
            f"- winner by aggregate mean total utility: `{winner}`",
            f"- PPO mean total utility: `{float(ppo['mean_total_utility_mean']):.3f} +/- {float(ppo['mean_total_utility_std']):.3f}`",
            f"- CEM mean total utility: `{float(cem['mean_total_utility_mean']):.3f} +/- {float(cem['mean_total_utility_std']):.3f}`",
            f"- best overall backend in this run: `{overall_best['backend_id']}` at `{float(overall_best['mean_total_utility_mean']):.3f}`",
            f"- PPO budget (timesteps completed): `{ppo_budget}`",
            f"- CEM budget (evaluations): `{cem_budget}`",
            "",
            "## Interpretation",
            "",
            "- PPO is the better pathway when a reusable sequential policy matters, or when the next step is richer vehicle dynamics and feedback control.",
            "- CEM is the better pathway when the control problem can be treated as short-horizon open-loop schedule search and you want a simpler optimizer baseline.",
            "- The key comparison is not only final utility but also boundary closeness, feature excitation, novelty rate, and budget efficiency.",
            "- If scripted or DOE families outrun both PPO and CEM, the current witness is still dominated by open-loop schedule search and PPO should remain a pathway proof rather than a promoted default.",
            "",
            "## Backend Decisions",
            "",
            *[f"- `{row['backend_id']}`: `{row['status']}` — {row['justification']}" for row in decision_rows],
        ]
    )


def analyze_sequential_ppo_vs_cem_comparison(
    *,
    config: SequentialBoundaryControlConfig | None = None,
    ppo_config: SequentialPpoConfig | None = None,
    cem_config: SequentialCemConfig | None = None,
    objective: TrajectoryExplorationObjective | None = None,
    ppo_run_dir: str | Path | None = None,
    seed_count: int = 1,
    base_seed: int = 7,
) -> SequentialComparisonResult:
    resolved_config = config or SequentialBoundaryControlConfig()
    resolved_ppo = ppo_config or SequentialPpoConfig()
    resolved_cem = cem_config or SequentialCemConfig()
    resolved_objective = objective or default_boundary_control_objective()
    resolved_seed_count = max(1, seed_count)

    all_backend_metrics_rows: list[dict[str, object]] = []
    all_evaluation_rows: list[dict[str, object]] = []
    all_control_rows: list[dict[str, object]] = []
    all_progress_rows: list[dict[str, object]] = []
    ppo_gallery_rollouts: list[SequentialRolloutSummary] = []
    cem_gallery_rollouts: list[SequentialRolloutSummary] = []
    seed_run_rows: list[dict[str, object]] = []
    ppo_budget = 0
    cem_budget = resolved_cem.iterations * resolved_cem.population_size

    for seed_index in range(resolved_seed_count):
        seed = base_seed + seed_index * 17
        ppo_run_seed_dir = None if ppo_run_dir is None else Path(ppo_run_dir) / f"seed_{seed_index}"
        ppo_result = analyze_sequential_ppo_boundary_control(
            config=resolved_config,
            ppo_config=_ppo_config_for_seed(resolved_ppo, seed=seed),
            objective=resolved_objective,
            run_dir=ppo_run_seed_dir,
        )
        ppo_budget = max(ppo_budget, int(ppo_result.training_summary.get("timesteps_completed", 0)))
        scripted_sequences = [
            sequence
            for _, sequence in _baseline_sequences(resolved_config, seed + 200, objective=resolved_objective)["scripted_profiles"]
        ]
        cem_summaries, cem_progress_rows = _evaluate_cem_sequences(
            config=resolved_config,
            objective=resolved_objective,
            cem_config=_cem_config_for_seed(resolved_cem, seed=seed),
            seed_index=seed_index,
        )
        cem_eval_rows, cem_control_rows = _rollout_rows(
            cem_summaries,
            backend_family="cem_open_loop",
            novelty_reference=scripted_sequences,
            seed_index=seed_index,
        )
        baseline_bank = _baseline_sequences(resolved_config, seed + 200, objective=resolved_objective)
        baseline_results = {
            family_id: _evaluate_sequence_family(
                family_id,
                entries,
                objective=resolved_objective,
                config=resolved_config,
                eval_seed_start=seed + 300 + family_index * 100,
            )
            for family_index, (family_id, entries) in enumerate(baseline_bank.items())
        }

        evaluation_rows = list(ppo_result.evaluation_rows) + cem_eval_rows
        control_rows = list(ppo_result.control_sequences) + cem_control_rows
        for row in evaluation_rows:
            row["seed_index"] = seed_index
        for row in control_rows:
            row["seed_index"] = seed_index
        for family_id, summaries in baseline_results.items():
            rows, controls = _rollout_rows(
                summaries,
                backend_family=family_id,
                novelty_reference=scripted_sequences,
                seed_index=seed_index,
            )
            evaluation_rows.extend(rows)
            control_rows.extend(controls)

        by_family: dict[str, list[dict[str, object]]] = {}
        for row in evaluation_rows:
            by_family.setdefault(str(row["backend_family"]), []).append(row)
        backend_rows = [
            _backend_metrics("ppo_policy", by_family.get("ppo", []), budget=max(int(ppo_result.training_summary.get("timesteps_completed", 0)), 1), seed_index=seed_index),
            _backend_metrics("cem_open_loop", by_family.get("cem_open_loop", []), budget=cem_budget, seed_index=seed_index),
            _backend_metrics("random_control", by_family.get("random_control", []), budget=len(by_family.get("random_control", [])), seed_index=seed_index),
            _backend_metrics("scripted_profiles", by_family.get("scripted_profiles", []), budget=len(by_family.get("scripted_profiles", [])), seed_index=seed_index),
            _backend_metrics("doe_schedule_bank", by_family.get("doe_schedule_bank", []), budget=len(by_family.get("doe_schedule_bank", [])), seed_index=seed_index),
            _backend_metrics("guided_schedule_mutation", by_family.get("guided_schedule_mutation", []), budget=len(by_family.get("guided_schedule_mutation", [])), seed_index=seed_index),
        ]
        all_backend_metrics_rows.extend(backend_rows)
        all_evaluation_rows.extend(evaluation_rows)
        all_control_rows.extend(control_rows)
        all_progress_rows.extend(
            [
                {
                    "seed_index": seed_index,
                    "backend_id": "ppo_policy",
                    "progress_kind": "ppo_snapshot",
                    "iteration": index,
                    "budget_index": float(row["timesteps"]),
                    "mean_total_utility": float(row["mean_total_utility"]),
                    "best_total_utility": float(row["mean_total_utility"]),
                    "mean_boundary_closeness": float(row["mean_boundary_closeness"]),
                    "mean_feature_excitation": float(row["mean_feature_excitation"]),
                    "mean_prior_sensitivity": float(row.get("mean_prior_sensitivity", 0.0)),
                    "action_std_mean": 0.0,
                }
                for index, row in enumerate(ppo_result.snapshot_rows)
            ]
        )
        all_progress_rows.extend(cem_progress_rows)
        ppo_gallery_rollouts.extend(ppo_result.gallery_rollouts)
        cem_gallery_rollouts.extend(sorted(cem_summaries, key=lambda summary: summary.evaluation.total_utility, reverse=True)[:2])
        seed_run_rows.append(
            {
                "seed_index": seed_index,
                "seed": seed,
                "objective_id": resolved_objective.objective_id,
                "ppo_timesteps_completed": int(ppo_result.training_summary.get("timesteps_completed", 0)),
                "cem_evaluations": cem_budget,
                "ppo_beats_cem": next(row for row in backend_rows if row["backend_id"] == "ppo_policy")["mean_total_utility"]
                > next(row for row in backend_rows if row["backend_id"] == "cem_open_loop")["mean_total_utility"],
            }
        )

    aggregate_rows = _aggregate_backend_metrics(all_backend_metrics_rows)
    decision_rows = _backend_decisions(aggregate_rows)
    strengths_limits_rows = _strengths_limits_rows(aggregate_rows, ppo_budget=ppo_budget, cem_budget=cem_budget)
    selected_rows = _selected_rows(all_evaluation_rows)
    report = _comparison_report(
        objective=resolved_objective,
        aggregate_rows=aggregate_rows,
        decision_rows=decision_rows,
        seed_count=resolved_seed_count,
        ppo_budget=ppo_budget,
        cem_budget=cem_budget,
    )
    manifest = _artifact_manifest(
        Path(ppo_run_dir) if ppo_run_dir is not None else Path("trajectory_exploration_rl") / "ppo_vs_cem_boundary_control"
    )
    return SequentialComparisonResult(
        config_payload={
            "objective_id": resolved_objective.objective_id,
            "seed_count": resolved_seed_count,
            "base_seed": base_seed,
            "environment": {
                "episode_horizon": resolved_config.episode_horizon,
                "dt": resolved_config.dt,
                "acceleration_limit": resolved_config.acceleration_limit,
            },
            "ppo": {
                "total_timesteps": resolved_ppo.total_timesteps,
                "n_steps": resolved_ppo.n_steps,
                "batch_size": resolved_ppo.batch_size,
                "eval_episodes": resolved_ppo.eval_episodes,
            },
            "cem": {
                "iterations": resolved_cem.iterations,
                "population_size": resolved_cem.population_size,
                "elite_fraction": resolved_cem.elite_fraction,
                "init_std": resolved_cem.init_std,
            },
        },
        backend_metrics_rows=tuple(all_backend_metrics_rows),
        aggregate_backend_metrics_rows=tuple(aggregate_rows),
        backend_decision_rows=tuple(decision_rows),
        seed_run_rows=tuple(seed_run_rows),
        evaluation_rows=tuple(all_evaluation_rows),
        selected_rollouts=selected_rows,
        control_rows=tuple(all_control_rows),
        progress_rows=tuple(all_progress_rows),
        strengths_limits_rows=tuple(strengths_limits_rows),
        ppo_gallery_rollouts=tuple(sorted(ppo_gallery_rollouts, key=lambda summary: summary.evaluation.total_utility, reverse=True)[:4]),
        cem_gallery_rollouts=tuple(sorted(cem_gallery_rollouts, key=lambda summary: summary.evaluation.total_utility, reverse=True)[:4]),
        artifact_manifest=manifest,
        report_markdown=report,
    )


def write_sequential_ppo_vs_cem_comparison_artifacts(
    output_dir: str | Path,
    *,
    config: SequentialBoundaryControlConfig | None = None,
    ppo_config: SequentialPpoConfig | None = None,
    cem_config: SequentialCemConfig | None = None,
    objective: TrajectoryExplorationObjective | None = None,
    run_name: str = "ppo_vs_cem_boundary_control",
    seed_count: int = 1,
    base_seed: int = 7,
) -> SequentialComparisonResult:
    root = Path(output_dir)
    run_dir = root / "trajectory_exploration_rl" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ppo_cache_dir = run_dir / "_ppo_cache"
    ppo_cache_dir.mkdir(parents=True, exist_ok=True)
    result = analyze_sequential_ppo_vs_cem_comparison(
        config=config,
        ppo_config=ppo_config,
        cem_config=cem_config,
        objective=objective,
        ppo_run_dir=ppo_cache_dir,
        seed_count=seed_count,
        base_seed=base_seed,
    )
    config_path = run_dir / "comparison_config.json"
    artifact_manifest_path = run_dir / "artifact_manifest.json"
    backend_metrics_path = run_dir / "metrics_by_backend.csv"
    aggregate_backend_metrics_path = run_dir / "aggregate_metrics_by_backend.csv"
    backend_decisions_path = run_dir / "backend_decisions.csv"
    seed_runs_path = run_dir / "seed_runs.csv"
    evaluation_rows_path = run_dir / "evaluation_rows.csv"
    selected_rollouts_path = run_dir / "selected_rollouts.csv"
    control_sequences_path = run_dir / "control_sequences.csv"
    progress_rows_path = run_dir / "progress_rows.csv"
    strengths_limits_path = run_dir / "strengths_and_limits.csv"
    report_path = run_dir / "report.md"
    progress_plot_path = run_dir / "progress_comparison.png"
    backend_metrics_plot_path = run_dir / "backend_metrics.png"
    control_gallery_path = run_dir / "control_gallery.png"

    manifest = {
        **result.artifact_manifest,
        "run_dir": str(run_dir),
        "objective_id": result.config_payload["objective_id"],
        "seed_count": result.config_payload["seed_count"],
    }
    _write_json(config_path, result.config_payload)
    _write_json(artifact_manifest_path, manifest)
    write_csv(backend_metrics_path, list(result.backend_metrics_rows), list(result.backend_metrics_rows[0].keys()))
    write_csv(aggregate_backend_metrics_path, list(result.aggregate_backend_metrics_rows), list(result.aggregate_backend_metrics_rows[0].keys()))
    write_csv(backend_decisions_path, list(result.backend_decision_rows), list(result.backend_decision_rows[0].keys()))
    write_csv(seed_runs_path, list(result.seed_run_rows), list(result.seed_run_rows[0].keys()))
    write_csv(evaluation_rows_path, list(result.evaluation_rows), list(result.evaluation_rows[0].keys()))
    write_csv(selected_rollouts_path, list(result.selected_rollouts), list(result.selected_rollouts[0].keys()))
    write_csv(control_sequences_path, list(result.control_rows), list(result.control_rows[0].keys()))
    write_csv(progress_rows_path, list(result.progress_rows), list(result.progress_rows[0].keys()))
    write_csv(strengths_limits_path, list(result.strengths_limits_rows), list(result.strengths_limits_rows[0].keys()))
    _write_text(report_path, result.report_markdown)
    write_plot(_render_progress_plot(list(result.progress_rows)), progress_plot_path)
    write_plot(_render_backend_metrics_plot(list(result.aggregate_backend_metrics_rows)), backend_metrics_plot_path)
    write_plot(_render_control_gallery(list(result.ppo_gallery_rollouts), list(result.cem_gallery_rollouts)), control_gallery_path)

    artifacts = SequentialComparisonArtifacts(
        run_dir=run_dir,
        config_path=config_path,
        artifact_manifest_path=artifact_manifest_path,
        backend_metrics_path=backend_metrics_path,
        aggregate_backend_metrics_path=aggregate_backend_metrics_path,
        backend_decisions_path=backend_decisions_path,
        seed_runs_path=seed_runs_path,
        evaluation_rows_path=evaluation_rows_path,
        selected_rollouts_path=selected_rollouts_path,
        control_sequences_path=control_sequences_path,
        progress_rows_path=progress_rows_path,
        strengths_limits_path=strengths_limits_path,
        report_path=report_path,
        progress_plot_path=progress_plot_path,
        backend_metrics_plot_path=backend_metrics_plot_path,
        control_gallery_path=control_gallery_path,
    )
    return SequentialComparisonResult(
        config_payload=result.config_payload,
        backend_metrics_rows=result.backend_metrics_rows,
        aggregate_backend_metrics_rows=result.aggregate_backend_metrics_rows,
        backend_decision_rows=result.backend_decision_rows,
        seed_run_rows=result.seed_run_rows,
        evaluation_rows=result.evaluation_rows,
        selected_rollouts=result.selected_rollouts,
        control_rows=result.control_rows,
        progress_rows=result.progress_rows,
        strengths_limits_rows=result.strengths_limits_rows,
        ppo_gallery_rollouts=result.ppo_gallery_rollouts,
        cem_gallery_rollouts=result.cem_gallery_rollouts,
        artifact_manifest=manifest,
        report_markdown=result.report_markdown,
        artifacts=artifacts,
    )


def _objective_backend_matrix_rows(results: list[SequentialComparisonResult]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in results:
        objective_id = str(result.config_payload["objective_id"])
        for row in result.aggregate_backend_metrics_rows:
            rows.append(
                {
                    "objective_id": objective_id,
                    "backend_id": row["backend_id"],
                    "mean_total_utility": row["mean_total_utility_mean"],
                    "mean_boundary_closeness": row["mean_boundary_closeness_mean"],
                    "mean_feature_excitation": row["mean_feature_excitation_mean"],
                    "mean_geometry_score": row["mean_geometry_score_mean"],
                    "novelty_rate": row["novelty_rate_mean"],
                }
            )
    return rows


def _objective_summary_rows(results: list[SequentialComparisonResult], objective_lookup: dict[str, TrajectoryExplorationObjective]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in results:
        objective_id = str(result.config_payload["objective_id"])
        objective = objective_lookup[objective_id]
        by_backend = {str(row["backend_id"]): row for row in result.aggregate_backend_metrics_rows}
        ppo = by_backend["ppo_policy"]
        cem = by_backend["cem_open_loop"]
        best = max(result.aggregate_backend_metrics_rows, key=lambda row: float(row["mean_total_utility_mean"]))
        ppo_decision = next(row for row in result.backend_decision_rows if row["backend_id"] == "ppo_policy")
        rows.append(
            {
                "objective_id": objective_id,
                "mode": objective.mode,
                "geometry_target": objective.geometry_target,
                "generation_scope": objective.backend_constraints.get("generation_scope", ""),
                "target_class": objective.target.class_name or "",
                "target_class_pair": " vs ".join(objective.target.class_pair) if objective.target.class_pair else "",
                "best_backend": best["backend_id"],
                "best_mean_total_utility": best["mean_total_utility_mean"],
                "ppo_mean_total_utility": ppo["mean_total_utility_mean"],
                "cem_mean_total_utility": cem["mean_total_utility_mean"],
                "ppo_minus_cem": float(ppo["mean_total_utility_mean"]) - float(cem["mean_total_utility_mean"]),
                "ppo_status": ppo_decision["status"],
                "run_dir": result.artifact_manifest["run_dir"],
            }
        )
    return rows


def _backend_summary_rows(objective_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    backends = sorted({str(row["best_backend"]) for row in objective_rows} | {"ppo_policy", "cem_open_loop"})
    rows: list[dict[str, object]] = []
    for backend_id in backends:
        rows.append(
            {
                "backend_id": backend_id,
                "objective_win_count": sum(1 for row in objective_rows if row["best_backend"] == backend_id),
                "objective_count": len(objective_rows),
                "win_fraction": sum(1 for row in objective_rows if row["best_backend"] == backend_id) / max(len(objective_rows), 1),
            }
        )
    return rows


def _decision_summary_rows(objective_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    statuses = sorted({str(row["ppo_status"]) for row in objective_rows})
    return [
        {
            "backend_id": "ppo_policy",
            "status": status,
            "objective_count": sum(1 for row in objective_rows if row["ppo_status"] == status),
        }
        for status in statuses
    ]


def _render_objective_backend_heatmap(rows: list[dict[str, object]]):
    objectives = list(dict.fromkeys(str(row["objective_id"]) for row in rows))
    backends = list(dict.fromkeys(str(row["backend_id"]) for row in rows))
    values = np.zeros((len(objectives), len(backends)), dtype=float)
    for row in rows:
        values[objectives.index(str(row["objective_id"])), backends.index(str(row["backend_id"]))] = float(row["mean_total_utility"])
    fig, ax = plt.subplots(figsize=(max(8.0, 0.72 * len(backends) + 5.0), max(4.8, 0.42 * len(objectives) + 2.0)))
    image = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(backends)), backends, rotation=30, ha="right")
    ax.set_yticks(range(len(objectives)), objectives)
    ax.set_title("Backend utility by generated objective", loc="left", fontweight="bold")
    ax.set_xlabel("Backend")
    ax.set_ylabel("Generated objective")
    fig.colorbar(image, ax=ax, label="Mean total utility")
    fig.tight_layout()
    return fig


def _objective_sweep_report(
    *,
    objective_summary_rows: list[dict[str, object]],
    backend_summary_rows: list[dict[str, object]],
    seed_count: int,
) -> str:
    ppo_experimental = sum(1 for row in objective_summary_rows if row["ppo_status"] == "experimental")
    ppo_promoted = sum(1 for row in objective_summary_rows if row["ppo_status"] == "promote")
    best_backend = max(backend_summary_rows, key=lambda row: float(row["win_fraction"]))
    return "\n".join(
        [
            "# Sequential Feature/Class Space PPO vs CEM Sweep",
            "",
            f"- objective count: `{len(objective_summary_rows)}`",
            f"- seed count per objective: `{seed_count}`",
            f"- top backend by objective wins: `{best_backend['backend_id']}`",
            f"- PPO promoted objectives: `{ppo_promoted}`",
            f"- PPO experimental objectives: `{ppo_experimental}`",
            "",
            "## Interpretation",
            "",
            "- This study expands the sequential-control comparison across generated feature cells, feature rows, class-pair regions, and novelty zones.",
            "- Each objective is evaluated through the same PPO/CEM/backend schemas; this top-level report summarizes the larger feature/class-space map.",
            "- PPO should move beyond pathway proof only on objective families where it beats CEM and approaches the strongest open-loop schedule baselines.",
        ]
    )


def analyze_sequential_objective_sweep_comparison(
    *,
    config: SequentialBoundaryControlConfig | None = None,
    ppo_config: SequentialPpoConfig | None = None,
    cem_config: SequentialCemConfig | None = None,
    objective_ids: tuple[str, ...] | None = None,
    objective_limit: int | None = None,
    seed_count: int = 1,
    base_seed: int = 7,
    ppo_run_dir: str | Path | None = None,
) -> SequentialObjectiveSweepResult:
    suite = generate_trajectory_exploration_objective_suite()
    selected = [
        objective
        for objective in suite.objectives
        if objective_ids is None or objective.objective_id in objective_ids
    ]
    if objective_limit is not None:
        selected = selected[: max(0, objective_limit)]
    if not selected:
        raise ValueError("no generated trajectory objectives selected")
    results: list[SequentialComparisonResult] = []
    for objective_index, objective in enumerate(selected):
        objective_run_dir = None if ppo_run_dir is None else Path(ppo_run_dir) / objective.objective_id
        results.append(
            analyze_sequential_ppo_vs_cem_comparison(
                config=config,
                ppo_config=ppo_config,
                cem_config=cem_config,
                objective=objective,
                ppo_run_dir=objective_run_dir,
                seed_count=seed_count,
                base_seed=base_seed + objective_index * 101,
            )
        )
    objective_lookup = {objective.objective_id: objective for objective in selected}
    objective_rows = _objective_summary_rows(results, objective_lookup)
    backend_rows = _backend_summary_rows(objective_rows)
    decision_rows = _decision_summary_rows(objective_rows)
    matrix_rows = _objective_backend_matrix_rows(results)
    report = _objective_sweep_report(
        objective_summary_rows=objective_rows,
        backend_summary_rows=backend_rows,
        seed_count=seed_count,
    )
    manifest = {
        "objective_count": len(selected),
        "objective_ids": [objective.objective_id for objective in selected],
        "tables": [
            "objective_summary.csv",
            "backend_summary.csv",
            "decision_summary.csv",
            "objective_backend_matrix.csv",
        ],
        "plots": ["objective_backend_heatmap.png"],
        "reports": ["report.md"],
    }
    return SequentialObjectiveSweepResult(
        config_payload={
            "source_spec_id": suite.spec.spec_id,
            "objective_count": len(selected),
            "objective_ids": [objective.objective_id for objective in selected],
            "seed_count": seed_count,
            "base_seed": base_seed,
        },
        objective_summary_rows=tuple(objective_rows),
        backend_summary_rows=tuple(backend_rows),
        decision_summary_rows=tuple(decision_rows),
        objective_backend_matrix_rows=tuple(matrix_rows),
        artifact_manifest=manifest,
        report_markdown=report,
    )


def write_sequential_objective_sweep_comparison_artifacts(
    output_dir: str | Path,
    *,
    config: SequentialBoundaryControlConfig | None = None,
    ppo_config: SequentialPpoConfig | None = None,
    cem_config: SequentialCemConfig | None = None,
    objective_ids: tuple[str, ...] | None = None,
    objective_limit: int | None = None,
    seed_count: int = 1,
    base_seed: int = 7,
) -> SequentialObjectiveSweepResult:
    root = Path(output_dir)
    run_dir = root / "trajectory_exploration_rl" / "ppo_vs_cem_objective_sweep"
    run_dir.mkdir(parents=True, exist_ok=True)
    ppo_cache_dir = run_dir / "_ppo_cache"
    ppo_cache_dir.mkdir(parents=True, exist_ok=True)
    result = analyze_sequential_objective_sweep_comparison(
        config=config,
        ppo_config=ppo_config,
        cem_config=cem_config,
        objective_ids=objective_ids,
        objective_limit=objective_limit,
        seed_count=seed_count,
        base_seed=base_seed,
        ppo_run_dir=ppo_cache_dir,
    )
    config_path = run_dir / "comparison_config.json"
    artifact_manifest_path = run_dir / "artifact_manifest.json"
    objective_summary_path = run_dir / "objective_summary.csv"
    backend_summary_path = run_dir / "backend_summary.csv"
    decision_summary_path = run_dir / "decision_summary.csv"
    objective_backend_matrix_path = run_dir / "objective_backend_matrix.csv"
    report_path = run_dir / "report.md"
    objective_backend_heatmap_path = run_dir / "objective_backend_heatmap.png"
    manifest = {**result.artifact_manifest, "run_dir": str(run_dir)}

    _write_json(config_path, result.config_payload)
    _write_json(artifact_manifest_path, manifest)
    write_csv(objective_summary_path, list(result.objective_summary_rows), list(result.objective_summary_rows[0].keys()))
    write_csv(backend_summary_path, list(result.backend_summary_rows), list(result.backend_summary_rows[0].keys()))
    write_csv(decision_summary_path, list(result.decision_summary_rows), list(result.decision_summary_rows[0].keys()))
    write_csv(objective_backend_matrix_path, list(result.objective_backend_matrix_rows), list(result.objective_backend_matrix_rows[0].keys()))
    _write_text(report_path, result.report_markdown)
    write_plot(_render_objective_backend_heatmap(list(result.objective_backend_matrix_rows)), objective_backend_heatmap_path)

    artifacts = SequentialObjectiveSweepArtifacts(
        run_dir=run_dir,
        config_path=config_path,
        artifact_manifest_path=artifact_manifest_path,
        objective_summary_path=objective_summary_path,
        backend_summary_path=backend_summary_path,
        decision_summary_path=decision_summary_path,
        objective_backend_matrix_path=objective_backend_matrix_path,
        report_path=report_path,
        objective_backend_heatmap_path=objective_backend_heatmap_path,
    )
    return SequentialObjectiveSweepResult(
        config_payload=result.config_payload,
        objective_summary_rows=result.objective_summary_rows,
        backend_summary_rows=result.backend_summary_rows,
        decision_summary_rows=result.decision_summary_rows,
        objective_backend_matrix_rows=result.objective_backend_matrix_rows,
        artifact_manifest=manifest,
        report_markdown=result.report_markdown,
        artifacts=artifacts,
    )
