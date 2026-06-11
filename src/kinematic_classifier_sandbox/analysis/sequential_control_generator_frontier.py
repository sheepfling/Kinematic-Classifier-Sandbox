from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from kinematic_classifier_sandbox.corpus.trajectory_exploration.objective_generation import (
    generated_trajectory_exploration_objectives,
)
from kinematic_classifier_sandbox.corpus.trajectory_exploration.ppo_boundary_control import (
    SequentialBoundaryControlConfig,
    SequentialPpoConfig,
    analyze_sequential_ppo_boundary_control,
    has_stable_baselines3_support,
)
from kinematic_classifier_sandbox.corpus.trajectory_exploration.contracts import (
    TrajectoryExplorationObjective,
)
from kinematic_classifier_sandbox.utils.io import write_csv
from kinematic_classifier_sandbox.utils.plotting import _figure_to_png
from kinematic_classifier_sandbox.utils.plotting import plt


FOCUS_OBJECTIVE_IDS = (
    "feature_row__accel_high_row",
    "class_pair__cv_vs_ca",
    "posterior_target__cv_ca_50_50",
    "novelty_region__novel_cv_ca_boundary_zone",
)


@dataclass(frozen=True, slots=True)
class SequentialControlFrontierRow:
    objective_id: str
    objective_mode: str
    ppo_mean_total_utility: float
    random_mean_total_utility: float
    scripted_mean_total_utility: float
    doe_mean_total_utility: float
    guided_mean_total_utility: float
    ppo_minus_random: float
    ppo_minus_scripted: float
    ppo_minus_doe: float
    ppo_minus_guided: float
    best_baseline_backend: str
    promotion_decision: str


@dataclass(frozen=True, slots=True)
class SequentialControlFrontierResult:
    frontier_rows: tuple[SequentialControlFrontierRow, ...]
    metrics: dict[str, float | int | str]


@dataclass(frozen=True, slots=True)
class SequentialControlFrontierArtifacts:
    run_dir: Path
    frontier_summary_path: Path
    metrics_path: Path
    report_path: Path
    decision_card_path: Path
    plot_paths: tuple[Path, ...]


def _selected_objectives() -> tuple[TrajectoryExplorationObjective, ...]:
    objective_lookup = {objective.objective_id: objective for objective in generated_trajectory_exploration_objectives()}
    return tuple(objective_lookup[objective_id] for objective_id in FOCUS_OBJECTIVE_IDS if objective_id in objective_lookup)


def _render_frontier_bars(result: SequentialControlFrontierResult):
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    labels = [row.objective_id for row in result.frontier_rows]
    x = list(range(len(labels)))
    width = 0.18
    ppo = [row.ppo_mean_total_utility for row in result.frontier_rows]
    random = [row.random_mean_total_utility for row in result.frontier_rows]
    scripted = [row.scripted_mean_total_utility for row in result.frontier_rows]
    doe = [row.doe_mean_total_utility for row in result.frontier_rows]
    guided = [row.guided_mean_total_utility for row in result.frontier_rows]
    ax.bar([value - 2 * width for value in x], ppo, width=width, label="ppo_policy", color="#2563eb")
    ax.bar([value - width for value in x], random, width=width, label="random_control", color="#9ca3af")
    ax.bar(x, scripted, width=width, label="scripted_profiles", color="#16a34a")
    ax.bar([value + width for value in x], doe, width=width, label="doe_schedule_bank", color="#0f766e")
    ax.bar([value + 2 * width for value in x], guided, width=width, label="guided_schedule_mutation", color="#7c3aed")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("mean total utility")
    ax.set_title("Sequential Control Generator Frontier", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncols=2)
    fig.tight_layout()
    return fig


def _render_delta_bars(result: SequentialControlFrontierResult):
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    labels = [row.objective_id for row in result.frontier_rows]
    x = list(range(len(labels)))
    ppo_minus_baseline = [row.ppo_mean_total_utility - max(row.random_mean_total_utility, row.scripted_mean_total_utility, row.doe_mean_total_utility, row.guided_mean_total_utility) for row in result.frontier_rows]
    ppo_minus_random = [row.ppo_minus_random for row in result.frontier_rows]
    width = 0.32
    ax.bar([value - width / 2 for value in x], ppo_minus_baseline, width=width, label="ppo - best baseline", color="#2563eb")
    ax.bar([value + width / 2 for value in x], ppo_minus_random, width=width, label="ppo - random", color="#7c3aed")
    ax.axhline(0.0, color="#6b7280", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("utility delta")
    ax.set_title("Sequential Control Deltas", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def analyze_sequential_control_generator_frontier(
    *,
    seed: int = 1309,
    config: SequentialBoundaryControlConfig | None = None,
    ppo_config: SequentialPpoConfig | None = None,
) -> SequentialControlFrontierResult:
    objectives = _selected_objectives()
    if not objectives or not has_stable_baselines3_support():
        metrics: dict[str, float | int | str] = {
            "study_id": "sequential_control_generator_frontier_v1",
            "seed": seed,
            "objective_count": len(objectives),
            "status": "dependency_missing",
            "promotion_decision": "revise_sequential_control_proxy",
        }
        return SequentialControlFrontierResult(frontier_rows=(), metrics=metrics)

    resolved_config = config or SequentialBoundaryControlConfig(episode_horizon=8)
    resolved_ppo = ppo_config or SequentialPpoConfig(total_timesteps=256, n_steps=32, batch_size=32, eval_episodes=4)

    frontier_rows: list[SequentialControlFrontierRow] = []
    for objective in objectives:
        result = analyze_sequential_ppo_boundary_control(
            config=resolved_config,
            ppo_config=resolved_ppo,
            objective=objective,
        )
        comparison = {str(row["backend_id"]): row for row in result.ppo_vs_heuristics_rows}
        ppo_row = comparison["ppo_policy"]
        random_row = comparison["random_control"]
        scripted_row = comparison["scripted_profiles"]
        doe_row = comparison["doe_schedule_bank"]
        guided_row = comparison["guided_schedule_mutation"]
        baseline_rows = {
            "random_control": random_row,
            "scripted_profiles": scripted_row,
            "doe_schedule_bank": doe_row,
            "guided_schedule_mutation": guided_row,
        }
        best_baseline_backend, best_baseline_row = max(
            baseline_rows.items(),
            key=lambda item: float(item[1]["mean_total_utility"]),
        )
        ppo_minus_best_baseline = float(ppo_row["mean_total_utility"]) - float(best_baseline_row["mean_total_utility"])
        frontier_rows.append(
            SequentialControlFrontierRow(
                objective_id=objective.objective_id,
                objective_mode=objective.mode,
                ppo_mean_total_utility=float(ppo_row["mean_total_utility"]),
                random_mean_total_utility=float(random_row["mean_total_utility"]),
                scripted_mean_total_utility=float(scripted_row["mean_total_utility"]),
                doe_mean_total_utility=float(doe_row["mean_total_utility"]),
                guided_mean_total_utility=float(guided_row["mean_total_utility"]),
                ppo_minus_random=float(ppo_row["mean_total_utility"]) - float(random_row["mean_total_utility"]),
                ppo_minus_scripted=float(ppo_row["mean_total_utility"]) - float(scripted_row["mean_total_utility"]),
                ppo_minus_doe=float(ppo_row["mean_total_utility"]) - float(doe_row["mean_total_utility"]),
                ppo_minus_guided=float(ppo_row["mean_total_utility"]) - float(guided_row["mean_total_utility"]),
                best_baseline_backend=best_baseline_backend,
                promotion_decision=(
                    "promote_ppo_proxy_for_sequential_control_frontier"
                    if ppo_minus_best_baseline > 0.0 and bool(result.training_summary.get("beats_random_control", False))
                    else "revise_sequential_control_proxy"
                ),
            )
        )

    mean_ppo = mean(row.ppo_mean_total_utility for row in frontier_rows)
    mean_random = mean(row.random_mean_total_utility for row in frontier_rows)
    mean_scripted = mean(row.scripted_mean_total_utility for row in frontier_rows)
    mean_doe = mean(row.doe_mean_total_utility for row in frontier_rows)
    mean_guided = mean(row.guided_mean_total_utility for row in frontier_rows)
    mean_best_baseline_delta = mean(
        row.ppo_mean_total_utility
        - max(row.random_mean_total_utility, row.scripted_mean_total_utility, row.doe_mean_total_utility, row.guided_mean_total_utility)
        for row in frontier_rows
    )
    metrics = {
        "study_id": "sequential_control_generator_frontier_v1",
        "seed": seed,
        "objective_count": len(frontier_rows),
        "objective_ids": ", ".join(row.objective_id for row in frontier_rows),
        "mean_ppo_total_utility": mean_ppo,
        "mean_random_total_utility": mean_random,
        "mean_scripted_total_utility": mean_scripted,
        "mean_doe_total_utility": mean_doe,
        "mean_guided_total_utility": mean_guided,
        "mean_ppo_minus_random": mean(row.ppo_minus_random for row in frontier_rows),
        "mean_ppo_minus_scripted": mean(row.ppo_minus_scripted for row in frontier_rows),
        "mean_ppo_minus_doe": mean(row.ppo_minus_doe for row in frontier_rows),
        "mean_ppo_minus_guided": mean(row.ppo_minus_guided for row in frontier_rows),
        "mean_ppo_minus_best_baseline": mean_best_baseline_delta,
        "promotion_decision": (
            "promote_ppo_proxy_for_sequential_control_frontier"
            if mean_best_baseline_delta > 0.0 and mean_ppo > mean_random
            else "revise_sequential_control_proxy"
        ),
        "status": "experimental",
    }
    return SequentialControlFrontierResult(frontier_rows=tuple(frontier_rows), metrics=metrics)


def write_sequential_control_generator_frontier_artifacts(
    output_dir: str | Path,
    *,
    result: SequentialControlFrontierResult | None = None,
    seed: int = 1309,
    config: SequentialBoundaryControlConfig | None = None,
    ppo_config: SequentialPpoConfig | None = None,
) -> SequentialControlFrontierArtifacts:
    payload = result or analyze_sequential_control_generator_frontier(seed=seed, config=config, ppo_config=ppo_config)
    run_dir = Path(output_dir) / "sequential_control_generator_frontier_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    frontier_summary_path = run_dir / "frontier_summary.csv"
    metrics_path = run_dir / "metrics.csv"
    report_path = run_dir / "sequential_control_generator_frontier_report.md"
    decision_card_path = run_dir / "decision_card.md"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    frontier_plot_path = plots_dir / "frontier_bars.png"
    delta_plot_path = plots_dir / "delta_bars.png"

    write_csv(frontier_summary_path, [asdict(row) for row in payload.frontier_rows], list(SequentialControlFrontierRow.__dataclass_fields__.keys()))
    write_csv(metrics_path, [payload.metrics], list(payload.metrics.keys()))

    report_lines = [
        "# Sequential Control Generator Frontier",
        "",
        "- Study: `sequential_control_generator_frontier_v1`",
        "- Compared backends: `ppo_policy`, `random_control`, `scripted_profiles`, `doe_schedule_bank`, `guided_schedule_mutation`",
        "",
        f"- objective count: `{payload.metrics['objective_count']}`",
        f"- mean ppo total utility: `{float(payload.metrics['mean_ppo_total_utility']):.4f}`",
        f"- mean random total utility: `{float(payload.metrics['mean_random_total_utility']):.4f}`",
        f"- mean scripted total utility: `{float(payload.metrics['mean_scripted_total_utility']):.4f}`",
        f"- mean best-baseline delta: `{float(payload.metrics['mean_ppo_minus_best_baseline']):.4f}`",
        f"- decision: `{payload.metrics['promotion_decision']}`",
        "",
        "This packet keeps the sequential-control generator lane explicit using the current PPO proxy and baseline control families.",
        "It is a proxy frontier packet, not a claim that SAC or TD3 has already been trained in the repo.",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    decision_lines = [
        "# Decision Card",
        "",
        "- Candidate method: `ppo_policy` proxy for sequential control",
        "- Adjacent future methods: `sac`, `td3`",
        f"- mean ppo minus best baseline: `{float(payload.metrics['mean_ppo_minus_best_baseline']):.4f}`",
        f"- decision: `{payload.metrics['promotion_decision']}`",
    ]
    decision_card_path.write_text("\n".join(decision_lines) + "\n", encoding="utf-8")

    frontier_plot_path.write_bytes(_figure_to_png(_render_frontier_bars(payload)))
    delta_plot_path.write_bytes(_figure_to_png(_render_delta_bars(payload)))

    return SequentialControlFrontierArtifacts(
        run_dir=run_dir,
        frontier_summary_path=frontier_summary_path,
        metrics_path=metrics_path,
        report_path=report_path,
        decision_card_path=decision_card_path,
        plot_paths=(frontier_plot_path, delta_plot_path),
    )


__all__ = [
    "SequentialControlFrontierArtifacts",
    "SequentialControlFrontierResult",
    "SequentialControlFrontierRow",
    "analyze_sequential_control_generator_frontier",
    "write_sequential_control_generator_frontier_artifacts",
]
