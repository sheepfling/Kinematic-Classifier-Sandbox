from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean

try:
    from stable_baselines3 import SAC, TD3
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
except Exception:  # pragma: no cover - exercised only when the optional dependency is missing
    SAC = None
    TD3 = None
    Monitor = None
    DummyVecEnv = None

from kinematic_classifier_sandbox.corpus.trajectory_exploration.objective_generation import (
    generated_trajectory_exploration_objectives,
)
from kinematic_classifier_sandbox.corpus.trajectory_exploration.comparison_surface import write_comparison_summary_csv
from kinematic_classifier_sandbox.corpus.trajectory_exploration.ppo_boundary_control import (
    SequentialBoundaryControlConfig,
    SequentialPpoConfig,
    _evaluate_policy_rollouts,
    analyze_sequential_ppo_boundary_control,
    has_stable_baselines3_support,
)
from kinematic_classifier_sandbox.corpus.trajectory_exploration.sequential_gym import SequentialTrajectoryGym
from kinematic_classifier_sandbox.corpus.trajectory_exploration.contracts import TrajectoryExplorationObjective
from kinematic_classifier_sandbox.utils.io import write_csv
from kinematic_classifier_sandbox.utils.plotting import _figure_to_png
from kinematic_classifier_sandbox.utils.plotting import plt


FOCUS_OBJECTIVE_IDS = (
    "feature_cell__acceleration_range_high_monotonicity_low",
    "feature_row__accel_high_row",
    "class_pair__cv_vs_ca",
    "posterior_target__cv_ca_50_50",
    "novelty_region__novel_cv_ca_boundary_zone",
)


@dataclass(frozen=True, slots=True)
class SequentialOffPolicyFrontierRow:
    objective_id: str
    objective_mode: str
    ppo_mean_total_utility: float
    sac_mean_total_utility: float
    td3_mean_total_utility: float
    random_mean_total_utility: float
    scripted_mean_total_utility: float
    doe_mean_total_utility: float
    guided_mean_total_utility: float
    ppo_sample_efficiency: float
    sac_sample_efficiency: float
    td3_sample_efficiency: float
    best_policy_backend: str
    best_baseline_backend: str
    best_policy_minus_best_baseline: float
    best_sample_efficiency_backend: str
    promotion_decision: str


@dataclass(frozen=True, slots=True)
class SequentialOffPolicyBudgetSweepRow:
    objective_id: str
    objective_mode: str
    budget_timesteps: int
    ppo_mean_total_utility: float
    sac_mean_total_utility: float
    td3_mean_total_utility: float
    best_policy_backend: str
    best_baseline_backend: str
    best_policy_minus_best_baseline: float
    promotion_decision: str


@dataclass(frozen=True, slots=True)
class SequentialOffPolicySeedSweepRow:
    seed: int
    objective_count: int
    budget_count: int
    mean_ppo_total_utility: float
    mean_sac_total_utility: float
    mean_td3_total_utility: float
    mean_best_policy_minus_best_baseline: float
    best_policy_backend: str
    promotion_decision: str


@dataclass(frozen=True, slots=True)
class SequentialOffPolicyFrontierResult:
    frontier_rows: tuple[SequentialOffPolicyFrontierRow, ...]
    budget_sweep_rows: tuple[SequentialOffPolicyBudgetSweepRow, ...]
    seed_sweep_rows: tuple[SequentialOffPolicySeedSweepRow, ...]
    metrics: dict[str, float | int | str]


@dataclass(frozen=True, slots=True)
class SequentialOffPolicyFrontierArtifacts:
    run_dir: Path
    frontier_summary_path: Path
    budget_sweep_path: Path
    seed_sweep_path: Path
    metrics_path: Path
    report_path: Path
    decision_card_path: Path
    plot_paths: tuple[Path, ...]


def _selected_objectives() -> tuple[TrajectoryExplorationObjective, ...]:
    objective_lookup = {objective.objective_id: objective for objective in generated_trajectory_exploration_objectives()}
    return tuple(
        objective_lookup[objective_id]
        for objective_id in FOCUS_OBJECTIVE_IDS
        if objective_id in objective_lookup
    )


def has_sequential_offpolicy_support() -> bool:
    return (
        has_stable_baselines3_support()
        and SAC is not None
        and TD3 is not None
        and Monitor is not None
        and DummyVecEnv is not None
    )


def _make_env_factory(objective: TrajectoryExplorationObjective, config: SequentialBoundaryControlConfig, seed: int):
    def _factory():
        return Monitor(SequentialTrajectoryGym(objective=objective, config=config, seed=seed))

    return _factory


def _train_offpolicy_model(
    algorithm_id: str,
    objective: TrajectoryExplorationObjective,
    *,
    config: SequentialBoundaryControlConfig,
    train_seed: int,
    total_timesteps: int,
    hidden_sizes: tuple[int, int],
):
    env = DummyVecEnv([_make_env_factory(objective, config, train_seed)])
    common_kwargs = {
        "seed": train_seed,
        "verbose": 0,
        "policy_kwargs": {"net_arch": list(hidden_sizes)},
    }
    if algorithm_id == "sac":
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=2048,
            learning_starts=32,
            batch_size=32,
            train_freq=1,
            gradient_steps=1,
            gamma=0.98,
            tau=0.02,
            ent_coef="auto",
            **common_kwargs,
        )
    elif algorithm_id == "td3":
        model = TD3(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=2048,
            learning_starts=32,
            batch_size=32,
            train_freq=1,
            gradient_steps=1,
            gamma=0.98,
            tau=0.02,
            policy_delay=2,
            target_policy_noise=0.10,
            target_noise_clip=0.20,
            **common_kwargs,
        )
    else:
        raise ValueError(f"unsupported off-policy algorithm: {algorithm_id}")
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    return model


def _analyze_single_seed_frontier(
    *,
    seed: int,
    objectives: tuple[TrajectoryExplorationObjective, ...],
    resolved_config: SequentialBoundaryControlConfig,
    resolved_ppo: SequentialPpoConfig,
    budget_sweep_timesteps: tuple[int, ...],
    eval_episodes: int,
) -> tuple[tuple[SequentialOffPolicyFrontierRow, ...], tuple[SequentialOffPolicyBudgetSweepRow, ...], SequentialOffPolicySeedSweepRow]:
    budget_sweep_rows: list[SequentialOffPolicyBudgetSweepRow] = []
    frontier_by_objective: dict[str, SequentialOffPolicyFrontierRow] = {}
    max_budget_timesteps = max(budget_sweep_timesteps) if budget_sweep_timesteps else 0

    for budget_index, budget_timesteps in enumerate(budget_sweep_timesteps):
        budget_ppo = replace(
            resolved_ppo,
            total_timesteps=budget_timesteps,
            eval_episodes=eval_episodes,
            train_seed=seed + budget_index * 101,
            eval_seed_start=seed + 200 + budget_index * 17,
        )
        for objective_index, objective in enumerate(objectives):
            ppo_result = analyze_sequential_ppo_boundary_control(
                config=resolved_config,
                ppo_config=budget_ppo,
                objective=objective,
            )
            comparison = {str(row["backend_id"]): row for row in ppo_result.ppo_vs_heuristics_rows}
            ppo_row = comparison["ppo_policy"]
            random_row = comparison["random_control"]
            scripted_row = comparison["scripted_profiles"]
            doe_row = comparison["doe_schedule_bank"]
            guided_row = comparison["guided_schedule_mutation"]

            sac_model = _train_offpolicy_model(
                "sac",
                objective,
                config=resolved_config,
                train_seed=seed + budget_index * 101 + objective_index * 19 + 1,
                total_timesteps=budget_timesteps,
                hidden_sizes=resolved_ppo.hidden_sizes,
            )
            sac_summary = _evaluate_policy_rollouts(
                sac_model,
                config=resolved_config,
                ppo_config=budget_ppo,
                objective=objective,
                episode_count=eval_episodes,
            )
            td3_model = _train_offpolicy_model(
                "td3",
                objective,
                config=resolved_config,
                train_seed=seed + budget_index * 101 + objective_index * 19 + 7,
                total_timesteps=budget_timesteps,
                hidden_sizes=resolved_ppo.hidden_sizes,
            )
            td3_summary = _evaluate_policy_rollouts(
                td3_model,
                config=resolved_config,
                ppo_config=budget_ppo,
                objective=objective,
                episode_count=eval_episodes,
            )

            ppo_mean = float(ppo_row["mean_total_utility"])
            sac_mean = mean(summary.evaluation.total_utility for summary in sac_summary)
            td3_mean = mean(summary.evaluation.total_utility for summary in td3_summary)
            best_policy_backend, best_policy_mean = max(
                (("ppo_policy", ppo_mean), ("sac", sac_mean), ("td3", td3_mean)),
                key=lambda item: item[1],
            )
            best_baseline_backend, best_baseline_mean = max(
                (
                    ("random_control", float(random_row["mean_total_utility"])),
                    ("scripted_profiles", float(scripted_row["mean_total_utility"])),
                    ("doe_schedule_bank", float(doe_row["mean_total_utility"])),
                    ("guided_schedule_mutation", float(guided_row["mean_total_utility"])),
                ),
                key=lambda item: item[1],
            )
            budget_sweep_rows.append(
                SequentialOffPolicyBudgetSweepRow(
                    objective_id=objective.objective_id,
                    objective_mode=objective.mode,
                    budget_timesteps=budget_timesteps,
                    ppo_mean_total_utility=ppo_mean,
                    sac_mean_total_utility=sac_mean,
                    td3_mean_total_utility=td3_mean,
                    best_policy_backend=best_policy_backend,
                    best_baseline_backend=best_baseline_backend,
                    best_policy_minus_best_baseline=best_policy_mean - best_baseline_mean,
                    promotion_decision=(
                        "promote_offpolicy_sequential_frontier"
                        if best_policy_backend in {"sac", "td3"} and best_policy_mean > best_baseline_mean
                        else "revise_sequential_offpolicy_frontier"
                    ),
                )
            )
            if budget_timesteps == max_budget_timesteps:
                frontier_by_objective[objective.objective_id] = SequentialOffPolicyFrontierRow(
                    objective_id=objective.objective_id,
                    objective_mode=objective.mode,
                    ppo_mean_total_utility=ppo_mean,
                    sac_mean_total_utility=sac_mean,
                    td3_mean_total_utility=td3_mean,
                    random_mean_total_utility=float(random_row["mean_total_utility"]),
                    scripted_mean_total_utility=float(scripted_row["mean_total_utility"]),
                    doe_mean_total_utility=float(doe_row["mean_total_utility"]),
                    guided_mean_total_utility=float(guided_row["mean_total_utility"]),
                    ppo_sample_efficiency=ppo_mean / max(budget_timesteps, 1),
                    sac_sample_efficiency=sac_mean / max(budget_timesteps, 1),
                    td3_sample_efficiency=td3_mean / max(budget_timesteps, 1),
                    best_policy_backend=best_policy_backend,
                    best_baseline_backend=best_baseline_backend,
                    best_policy_minus_best_baseline=best_policy_mean - best_baseline_mean,
                    best_sample_efficiency_backend=max(
                        (
                            ("ppo_policy", ppo_mean / max(budget_timesteps, 1)),
                            ("sac", sac_mean / max(budget_timesteps, 1)),
                            ("td3", td3_mean / max(budget_timesteps, 1)),
                        ),
                        key=lambda item: item[1],
                    )[0],
                    promotion_decision=(
                        "promote_offpolicy_sequential_frontier"
                        if best_policy_backend in {"sac", "td3"} and best_policy_mean > best_baseline_mean
                        else "revise_sequential_offpolicy_frontier"
                    ),
                )

    frontier_rows = [frontier_by_objective[objective.objective_id] for objective in objectives if objective.objective_id in frontier_by_objective]
    mean_ppo = mean(row.ppo_mean_total_utility for row in frontier_rows)
    mean_sac = mean(row.sac_mean_total_utility for row in frontier_rows)
    mean_td3 = mean(row.td3_mean_total_utility for row in frontier_rows)
    mean_best_policy_delta = mean(row.best_policy_minus_best_baseline for row in frontier_rows)
    best_policy_backend = max((("ppo_policy", mean_ppo), ("sac", mean_sac), ("td3", mean_td3)), key=lambda item: item[1])[0]
    seed_summary = SequentialOffPolicySeedSweepRow(
        seed=seed,
        objective_count=len(frontier_rows),
        budget_count=len(budget_sweep_timesteps),
        mean_ppo_total_utility=mean_ppo,
        mean_sac_total_utility=mean_sac,
        mean_td3_total_utility=mean_td3,
        mean_best_policy_minus_best_baseline=mean_best_policy_delta,
        best_policy_backend=best_policy_backend,
        promotion_decision=(
            "promote_offpolicy_sequential_frontier"
            if mean_best_policy_delta > 0.0 and max(mean_sac, mean_td3) >= mean_ppo
            else "revise_sequential_offpolicy_frontier"
        ),
    )
    return tuple(frontier_rows), tuple(budget_sweep_rows), seed_summary


def _render_frontier_bars(result: SequentialOffPolicyFrontierResult):
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    labels = [row.objective_id for row in result.frontier_rows]
    x = list(range(len(labels)))
    width = 0.14
    ppo = [row.ppo_mean_total_utility for row in result.frontier_rows]
    sac = [row.sac_mean_total_utility for row in result.frontier_rows]
    td3 = [row.td3_mean_total_utility for row in result.frontier_rows]
    random = [row.random_mean_total_utility for row in result.frontier_rows]
    scripted = [row.scripted_mean_total_utility for row in result.frontier_rows]
    ax.bar([value - 2 * width for value in x], ppo, width=width, label="ppo_policy", color="#2563eb")
    ax.bar([value - width for value in x], sac, width=width, label="sac", color="#be123c")
    ax.bar(x, td3, width=width, label="td3", color="#7c3aed")
    ax.bar([value + width for value in x], random, width=width, label="random_control", color="#9ca3af")
    ax.bar([value + 2 * width for value in x], scripted, width=width, label="scripted_profiles", color="#16a34a")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("mean total utility")
    ax.set_title("Sequential Off-Policy Frontier", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncols=2)
    fig.tight_layout()
    return fig


def _render_efficiency_bars(result: SequentialOffPolicyFrontierResult):
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    labels = [row.objective_id for row in result.frontier_rows]
    x = list(range(len(labels)))
    width = 0.24
    ppo = [row.ppo_sample_efficiency for row in result.frontier_rows]
    sac = [row.sac_sample_efficiency for row in result.frontier_rows]
    td3 = [row.td3_sample_efficiency for row in result.frontier_rows]
    ax.bar([value - width for value in x], ppo, width=width, label="ppo", color="#2563eb")
    ax.bar(x, sac, width=width, label="sac", color="#be123c")
    ax.bar([value + width for value in x], td3, width=width, label="td3", color="#7c3aed")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("utility per timestep")
    ax.set_title("Sequential Sample Efficiency", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def analyze_sequential_offpolicy_control_frontier(
    *,
    seed: int = 1409,
    config: SequentialBoundaryControlConfig | None = None,
    ppo_config: SequentialPpoConfig | None = None,
    seed_sweep: tuple[int, ...] | None = None,
    budget_sweep_timesteps: tuple[int, ...] = (32, 64),
    eval_episodes: int = 1,
) -> SequentialOffPolicyFrontierResult:
    resolved_seed_sweep = seed_sweep or (seed, seed + 1)
    if not has_sequential_offpolicy_support():
        metrics: dict[str, float | int | str] = {
            "study_id": "sequential_offpolicy_control_frontier_v1",
            "seed": seed,
            "objective_count": 0,
            "budget_count": 0,
            "seed_count": 0,
            "status": "dependency_missing",
            "promotion_decision": "revise_sequential_offpolicy_frontier",
        }
        return SequentialOffPolicyFrontierResult(frontier_rows=(), budget_sweep_rows=(), seed_sweep_rows=(), metrics=metrics)

    objectives = _selected_objectives()
    resolved_config = config or SequentialBoundaryControlConfig(episode_horizon=8)
    resolved_ppo = ppo_config or SequentialPpoConfig(
        total_timesteps=max(budget_sweep_timesteps) if budget_sweep_timesteps else 32,
        eval_episodes=eval_episodes,
        train_seed=seed,
        eval_seed_start=seed + 200,
        hidden_sizes=(32, 32),
    )

    seed_sweep_rows: list[SequentialOffPolicySeedSweepRow] = []
    frontier_rows: tuple[SequentialOffPolicyFrontierRow, ...] = ()
    budget_sweep_rows: tuple[SequentialOffPolicyBudgetSweepRow, ...] = ()
    for index, sweep_seed in enumerate(resolved_seed_sweep):
        seed_frontier_rows, seed_budget_rows, seed_summary = _analyze_single_seed_frontier(
            seed=sweep_seed,
            objectives=objectives,
            resolved_config=resolved_config,
            resolved_ppo=resolved_ppo,
            budget_sweep_timesteps=budget_sweep_timesteps,
            eval_episodes=eval_episodes,
        )
        seed_sweep_rows.append(seed_summary)
        if index == 0:
            frontier_rows = seed_frontier_rows
            budget_sweep_rows = seed_budget_rows

    mean_ppo = mean(row.mean_ppo_total_utility for row in seed_sweep_rows)
    mean_sac = mean(row.mean_sac_total_utility for row in seed_sweep_rows)
    mean_td3 = mean(row.mean_td3_total_utility for row in seed_sweep_rows)
    mean_best_policy_delta = mean(row.mean_best_policy_minus_best_baseline for row in seed_sweep_rows)
    mean_ppo_efficiency = mean(row.ppo_sample_efficiency for row in frontier_rows)
    mean_sac_efficiency = mean(row.sac_sample_efficiency for row in frontier_rows)
    mean_td3_efficiency = mean(row.td3_sample_efficiency for row in frontier_rows)
    seed_promotion_rate = sum(row.promotion_decision == "promote_offpolicy_sequential_frontier" for row in seed_sweep_rows) / max(len(seed_sweep_rows), 1)
    seed_best_policy_counts = Counter(row.best_policy_backend for row in seed_sweep_rows)
    seed_best_policy_backend = max(seed_best_policy_counts.items(), key=lambda item: item[1])[0]
    metrics = {
        "study_id": "sequential_offpolicy_control_frontier_v1",
        "seed": seed,
        "objective_count": len(frontier_rows),
        "budget_count": len(budget_sweep_timesteps),
        "seed_count": len(seed_sweep_rows),
        "objective_ids": ", ".join(row.objective_id for row in frontier_rows),
        "budget_timesteps": ",".join(str(value) for value in budget_sweep_timesteps),
        "max_budget_timesteps": max(budget_sweep_timesteps) if budget_sweep_timesteps else 0,
        "seed_values": ",".join(str(row.seed) for row in seed_sweep_rows),
        "eval_episodes": eval_episodes,
        "mean_ppo_total_utility": mean_ppo,
        "mean_sac_total_utility": mean_sac,
        "mean_td3_total_utility": mean_td3,
        "mean_ppo_sample_efficiency": mean_ppo_efficiency,
        "mean_sac_sample_efficiency": mean_sac_efficiency,
        "mean_td3_sample_efficiency": mean_td3_efficiency,
        "mean_best_policy_minus_best_baseline": mean_best_policy_delta,
        "best_policy_backend": max((("ppo_policy", mean_ppo), ("sac", mean_sac), ("td3", mean_td3)), key=lambda item: item[1])[0],
        "seed_best_policy_backend": seed_best_policy_backend,
        "seed_promotion_rate": seed_promotion_rate,
        "promotion_decision": (
            "promote_offpolicy_sequential_frontier"
            if mean_best_policy_delta > 0.0 and max(mean_sac, mean_td3) >= mean_ppo and seed_promotion_rate >= 0.5
            else "revise_sequential_offpolicy_frontier"
        ),
        "status": "implemented",
    }
    return SequentialOffPolicyFrontierResult(
        frontier_rows=tuple(frontier_rows),
        budget_sweep_rows=tuple(budget_sweep_rows),
        seed_sweep_rows=tuple(seed_sweep_rows),
        metrics=metrics,
    )


def write_sequential_offpolicy_control_frontier_artifacts(
    output_dir: str | Path,
    *,
    result: SequentialOffPolicyFrontierResult | None = None,
    seed: int = 1409,
    config: SequentialBoundaryControlConfig | None = None,
    ppo_config: SequentialPpoConfig | None = None,
    seed_sweep: tuple[int, ...] | None = None,
    budget_sweep_timesteps: tuple[int, ...] = (32, 64),
    eval_episodes: int = 1,
) -> SequentialOffPolicyFrontierArtifacts:
    payload = result or analyze_sequential_offpolicy_control_frontier(
        seed=seed,
        config=config,
        ppo_config=ppo_config,
        seed_sweep=seed_sweep,
        budget_sweep_timesteps=budget_sweep_timesteps,
        eval_episodes=eval_episodes,
    )
    run_dir = Path(output_dir) / "sequential_offpolicy_control_frontier_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    frontier_summary_path = run_dir / "frontier_summary.csv"
    budget_sweep_path = run_dir / "budget_sweep_summary.csv"
    seed_sweep_path = run_dir / "seed_sweep_summary.csv"
    metrics_path = run_dir / "metrics.csv"
    report_path = run_dir / "sequential_offpolicy_control_frontier_report.md"
    decision_card_path = run_dir / "decision_card.md"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    frontier_plot_path = plots_dir / "frontier_bars.png"
    efficiency_plot_path = plots_dir / "efficiency_bars.png"

    write_csv(frontier_summary_path, [asdict(row) for row in payload.frontier_rows], list(SequentialOffPolicyFrontierRow.__dataclass_fields__.keys()))
    write_comparison_summary_csv(run_dir, [asdict(row) for row in payload.frontier_rows], filename="summary.csv")
    write_csv(budget_sweep_path, [asdict(row) for row in payload.budget_sweep_rows], list(SequentialOffPolicyBudgetSweepRow.__dataclass_fields__.keys()))
    write_csv(seed_sweep_path, [asdict(row) for row in payload.seed_sweep_rows], list(SequentialOffPolicySeedSweepRow.__dataclass_fields__.keys()))
    write_csv(metrics_path, [payload.metrics], list(payload.metrics.keys()))

    report_lines = [
        "# Sequential Off-Policy Control Frontier",
        "",
        "- Study: `sequential_offpolicy_control_frontier_v1`",
        "- Compared backends: `ppo_policy`, `sac`, `td3`, `random_control`, `scripted_profiles`, `doe_schedule_bank`, `guided_schedule_mutation`",
        "",
        f"- objective count: `{payload.metrics['objective_count']}`",
        f"- budget sweep: `{payload.metrics['budget_timesteps']}`",
        f"- seed sweep: `{payload.metrics.get('seed_values', '')}`",
        f"- seed promotion rate: `{float(payload.metrics.get('seed_promotion_rate', 0.0)):.2f}`",
        f"- mean PPO utility: `{float(payload.metrics['mean_ppo_total_utility']):.4f}`",
        f"- mean SAC utility: `{float(payload.metrics['mean_sac_total_utility']):.4f}`",
        f"- mean TD3 utility: `{float(payload.metrics['mean_td3_total_utility']):.4f}`",
        f"- mean best-policy-minus-best-baseline: `{float(payload.metrics['mean_best_policy_minus_best_baseline']):.4f}`",
        f"- decision: `{payload.metrics['promotion_decision']}`",
        "",
        "This packet is the first explicit SAC/TD3 smoke run on the sequential-control witness surface.",
        "It is intentionally small-budget and is meant to expose sample-efficiency comparisons rather than claim a final policy winner.",
        "The current version adds a narrow seed sweep so stability is visible alongside the per-objective frontier.",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    decision_lines = [
        "# Decision Card",
        "",
        "- Candidate methods: `sac`, `td3`",
        "- Adjacent comparator: `ppo_policy`",
        f"- best policy backend: `{payload.metrics['best_policy_backend']}`",
        f"- seed best-policy backend: `{payload.metrics.get('seed_best_policy_backend', '')}`",
        f"- seed promotion rate: `{float(payload.metrics.get('seed_promotion_rate', 0.0)):.2f}`",
        f"- decision: `{payload.metrics['promotion_decision']}`",
    ]
    decision_card_path.write_text("\n".join(decision_lines) + "\n", encoding="utf-8")

    frontier_plot_path.write_bytes(_figure_to_png(_render_frontier_bars(payload)))
    efficiency_plot_path.write_bytes(_figure_to_png(_render_efficiency_bars(payload)))

    return SequentialOffPolicyFrontierArtifacts(
        run_dir=run_dir,
        frontier_summary_path=frontier_summary_path,
        budget_sweep_path=budget_sweep_path,
        seed_sweep_path=seed_sweep_path,
        metrics_path=metrics_path,
        report_path=report_path,
        decision_card_path=decision_card_path,
        plot_paths=(frontier_plot_path, efficiency_plot_path),
    )


__all__ = [
    "SequentialOffPolicyFrontierArtifacts",
    "SequentialOffPolicyBudgetSweepRow",
    "SequentialOffPolicyFrontierResult",
    "SequentialOffPolicyFrontierRow",
    "SequentialOffPolicySeedSweepRow",
    "analyze_sequential_offpolicy_control_frontier",
    "has_sequential_offpolicy_support",
    "write_sequential_offpolicy_control_frontier_artifacts",
]
