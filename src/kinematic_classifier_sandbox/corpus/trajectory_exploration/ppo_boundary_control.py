from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean

import numpy

from ...utils.io import _write_json, _write_text, write_csv
from ...utils.plotting import plt, write_plot
from .comparison_surface import write_comparison_summary_csv, write_decision_card
from .contracts import TrajectoryExplorationObjective
from .objective_generation import generate_trajectory_exploration_objective_suite
from .sequential_control_specs import (
    default_air_vehicle_control_problem_spec,
    default_three_dimensional_point_mass_problem_spec,
)
from .sequential_gym import (
    SequentialBoundaryControlConfig,
    SequentialRolloutSummary,
    SequentialTrajectoryGym,
    default_boundary_control_objective,
    doe_schedule_bank,
    evaluate_control_sequence,
    guided_mutation_bank,
    scripted_control_profiles,
    sequential_environment_contract,
)

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    _SB3_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised by optional dependency test
    PPO = None
    BaseCallback = object
    Monitor = None
    DummyVecEnv = None
    _SB3_IMPORT_ERROR = exc


def has_stable_baselines3_support() -> bool:
    return PPO is not None and DummyVecEnv is not None and Monitor is not None


@dataclass(frozen=True, slots=True)
class SequentialPpoConfig:
    total_timesteps: int = 1024
    n_steps: int = 64
    batch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.98
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    eval_episodes: int = 8
    train_seed: int = 7
    eval_seed_start: int = 200
    hidden_sizes: tuple[int, int] = (64, 64)
    progress_eval_episodes: int = 4
    checkpoint_interval_timesteps: int = 256
    snapshot_interval_timesteps: int = 256
    resume_if_possible: bool = True


@dataclass(frozen=True, slots=True)
class SequentialPpoArtifacts:
    run_dir: Path
    checkpoints_dir: Path
    environment_contract_path: Path
    control_problem_contract_path: Path
    transition_report_path: Path
    training_config_path: Path
    checkpoint_manifest_path: Path
    training_summary_path: Path
    evaluation_rows_path: Path
    selected_rollouts_path: Path
    control_sequences_path: Path
    training_trace_rows_path: Path
    snapshot_rows_path: Path
    training_curve_path: Path
    rollout_gallery_path: Path
    utility_progress_path: Path
    feature_progress_path: Path
    class_space_progress_path: Path
    ppo_vs_heuristics_path: Path
    report_path: Path
    rl_algorithm_decision_report_path: Path


@dataclass(frozen=True, slots=True)
class SequentialPpoSweepArtifacts:
    run_dir: Path
    manifest_path: Path
    summary_rows_path: Path
    report_path: Path


def _transition_report_markdown(config: SequentialBoundaryControlConfig, objective: TrajectoryExplorationObjective) -> str:
    current = config.control_problem
    point_mass_3d = default_three_dimensional_point_mass_problem_spec()
    air_vehicle = default_air_vehicle_control_problem_spec()
    return "\n".join(
        [
            "# Sequential Control Transition Report",
            "",
            f"- current problem id: `{current.problem_id}`",
            f"- current vehicle family: `{current.vehicle_family}`",
            f"- current objective id: `{objective.objective_id}`",
            "",
            "## 3D Point-Mass Lift",
            "",
            *[f"- {step}" for step in point_mass_3d.transition_path],
            "",
            "## Aerodynamic Vehicle Lift",
            "",
            *[f"- {step}" for step in air_vehicle.transition_path],
            "",
            "## Adapter Requirements",
            "",
            *[f"- {item}" for item in point_mass_3d.adapter_requirements],
            *[f"- {item}" for item in air_vehicle.adapter_requirements],
        ]
    )


@dataclass(frozen=True, slots=True)
class SequentialPpoResult:
    training_summary: dict[str, object]
    checkpoint_manifest: dict[str, object]
    evaluation_rows: tuple[dict[str, object], ...]
    selected_rollouts: tuple[dict[str, object], ...]
    control_sequences: tuple[dict[str, object], ...]
    ppo_vs_heuristics_rows: tuple[dict[str, object], ...]
    training_curve_rows: tuple[dict[str, float], ...]
    snapshot_rows: tuple[dict[str, float], ...]
    gallery_rollouts: tuple[SequentialRolloutSummary, ...]
    report_markdown: str
    rl_algorithm_decision_report_markdown: str
    artifacts: SequentialPpoArtifacts | None = None


class _TrainingTraceCallback(BaseCallback):
    def __init__(
        self,
        *,
        run_dir: Path | None,
        config: SequentialBoundaryControlConfig,
        ppo_config: SequentialPpoConfig,
        objective: TrajectoryExplorationObjective,
    ) -> None:
        super().__init__()
        self._run_dir = run_dir
        self._config = config
        self._ppo_config = ppo_config
        self._objective = objective
        self.episode_rows: list[dict[str, float]] = []
        self.snapshot_rows: list[dict[str, float]] = []
        self._last_checkpoint_timestep = 0
        self._last_snapshot_timestep = 0

    def _trace_path(self) -> Path | None:
        return None if self._run_dir is None else self._run_dir / "training_trace_rows.csv"

    def _snapshot_path(self) -> Path | None:
        return None if self._run_dir is None else self._run_dir / "snapshot_rows.csv"

    def _checkpoint_dir(self) -> Path | None:
        return None if self._run_dir is None else self._run_dir / "checkpoints"

    def _checkpoint_manifest_path(self) -> Path | None:
        return None if self._run_dir is None else self._run_dir / "checkpoint_manifest.json"

    def _persist_training_rows(self) -> None:
        trace_path = self._trace_path()
        if trace_path is None or not self.episode_rows:
            return
        write_csv(trace_path, list(self.episode_rows), list(self.episode_rows[0].keys()))

    def _persist_snapshot_rows(self) -> None:
        snapshot_path = self._snapshot_path()
        if snapshot_path is None or not self.snapshot_rows:
            return
        write_csv(snapshot_path, list(self.snapshot_rows), list(self.snapshot_rows[0].keys()))

    def _persist_checkpoint(self) -> None:
        checkpoint_dir = self._checkpoint_dir()
        manifest_path = self._checkpoint_manifest_path()
        if checkpoint_dir is None or manifest_path is None:
            return
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"ppo_step_{self.num_timesteps}.zip"
        self.model.save(checkpoint_path)
        _write_json(
            manifest_path,
            {
                "status": "in_progress",
                "latest_checkpoint_path": str(checkpoint_path),
                "timesteps_completed": int(self.num_timesteps),
                "target_total_timesteps": int(self._ppo_config.total_timesteps),
                "checkpoint_interval_timesteps": int(self._ppo_config.checkpoint_interval_timesteps),
                "snapshot_interval_timesteps": int(self._ppo_config.snapshot_interval_timesteps),
            },
        )

    def _record_snapshot(self, model=None) -> None:
        resolved_model = self.model if model is None else model
        summaries = _evaluate_policy_rollouts(
            resolved_model,
            config=self._config,
            ppo_config=self._ppo_config,
            objective=self._objective,
            episode_count=self._ppo_config.progress_eval_episodes,
        )
        snapshot_row = {
            "timesteps": float(self.num_timesteps),
            "mean_total_utility": mean(summary.evaluation.total_utility for summary in summaries),
            "mean_boundary_closeness": mean(summary.evaluation.boundary_closeness for summary in summaries),
            "mean_feature_excitation": mean(summary.evaluation.feature_excitation for summary in summaries),
            "mean_class_validity": mean(summary.evaluation.class_validity for summary in summaries),
            "mean_prior_sensitivity": mean(summary.evaluation.prior_sensitivity for summary in summaries),
            "mean_geometry_score": mean(summary.evaluation.geometry_score for summary in summaries),
            "feature_target_hit_fraction": mean(
                1.0 if summary.evaluation.feature_excitation >= 0.75 else 0.0 for summary in summaries
            ),
            "boundary_hit_fraction": mean(
                1.0 if summary.evaluation.boundary_closeness >= 0.60 else 0.0 for summary in summaries
            ),
        }
        self.snapshot_rows.append(snapshot_row)
        self._persist_snapshot_rows()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            episode = info.get("episode")
            if not episode:
                continue
            entropy_proxy = float(numpy.exp(self.model.policy.log_std.detach().cpu().numpy()).mean()) if hasattr(self.model.policy, "log_std") else 0.0
            self.episode_rows.append(
                {
                    "timesteps": float(self.num_timesteps),
                    "episode_return": float(episode["r"]),
                    "episode_length": float(episode["l"]),
                    "entropy_proxy": entropy_proxy,
                }
            )
        if infos:
            self._persist_training_rows()
        if self._run_dir is not None and self.num_timesteps - self._last_snapshot_timestep >= self._ppo_config.snapshot_interval_timesteps:
            self._record_snapshot()
            self._last_snapshot_timestep = int(self.num_timesteps)
        if self._run_dir is not None and self.num_timesteps - self._last_checkpoint_timestep >= self._ppo_config.checkpoint_interval_timesteps:
            self._persist_checkpoint()
            self._last_checkpoint_timestep = int(self.num_timesteps)
        return True


def _control_signature_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    shared = min(len(left), len(right))
    if shared == 0:
        return 0.0
    return sum(abs(left[index] - right[index]) for index in range(shared)) / shared


def _selected_rows(rows: list[dict[str, object]]) -> tuple[dict[str, object], ...]:
    ranked = sorted(rows, key=lambda row: float(row["total_utility"]), reverse=True)
    count = min(6, max(1, len(ranked) // 4))
    selected_ids = {str(row["proposal_id"]) for row in ranked[:count]}
    return tuple({**row, "selected": row["proposal_id"] in selected_ids} for row in ranked[:count])


def _baseline_sequences(
    config: SequentialBoundaryControlConfig,
    eval_seed: int,
    *,
    objective: TrajectoryExplorationObjective,
) -> dict[str, list[tuple[str, tuple[float, ...]]]]:
    scripted = list(scripted_control_profiles(config).items())
    doe_sequences = list(enumerate(doe_schedule_bank(config)))
    doe_scored = [
        (
            f"doe_{index}",
            sequence,
            evaluate_control_sequence(
                sequence,
                objective=objective,
                config=config,
                seed=eval_seed,
                proposal_id=f"doe_probe_{index}",
                backend_id="doe_schedule_bank",
            ).evaluation.total_utility,
        )
        for index, sequence in doe_sequences
    ]
    doe_best = sorted(doe_scored, key=lambda row: row[2], reverse=True)[:4]
    guided: list[tuple[str, tuple[float, ...]]] = []
    for base_index, (_, sequence, _) in enumerate(doe_best):
        for mutant_index, mutant in enumerate(guided_mutation_bank(sequence)[:6]):
            guided.append((f"guided_{base_index}_{mutant_index}", mutant))
    rng = numpy.random.default_rng(eval_seed + 99)
    random_sequences = [
        (f"random_{index}", tuple(float(value) for value in rng.uniform(-1.0, 1.0, size=config.episode_horizon)))
        for index in range(12)
    ]
    return {
        "scripted_profiles": scripted,
        "doe_schedule_bank": [(name, sequence) for name, sequence, _ in doe_best],
        "guided_schedule_mutation": guided,
        "random_control": random_sequences,
    }


def _evaluate_sequence_family(
    family_id: str,
    entries: list[tuple[str, tuple[float, ...]]],
    *,
    objective: TrajectoryExplorationObjective,
    config: SequentialBoundaryControlConfig,
    eval_seed_start: int,
) -> list[SequentialRolloutSummary]:
    return [
        evaluate_control_sequence(
            sequence,
            objective=objective,
            config=config,
            seed=eval_seed_start + index,
            proposal_id=name,
            backend_id=family_id,
            candidate_index=index,
        )
        for index, (name, sequence) in enumerate(entries)
    ]


def _evaluate_policy_rollouts(
    model,
    *,
    config: SequentialBoundaryControlConfig,
    ppo_config: SequentialPpoConfig,
    objective: TrajectoryExplorationObjective,
    episode_count: int | None = None,
) -> list[SequentialRolloutSummary]:
    rows: list[SequentialRolloutSummary] = []
    resolved_episode_count = ppo_config.eval_episodes if episode_count is None else episode_count
    for episode_index in range(resolved_episode_count):
        env = SequentialTrajectoryGym(objective=objective, config=config, seed=ppo_config.eval_seed_start + episode_index)
        obs, _ = env.reset(seed=ppo_config.eval_seed_start + episode_index)
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
        summary = env.last_summary
        assert summary is not None
        rows.append(summary)
    return rows


def _render_training_curve(rows: list[dict[str, float]]):
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    if rows:
        ax.plot([row["timesteps"] for row in rows], [row["episode_return"] for row in rows], label="episode return")
        ax.plot([row["timesteps"] for row in rows], [row["entropy_proxy"] for row in rows], label="entropy proxy")
        ax.legend()
    ax.set_title("PPO training trace", loc="left", fontweight="bold")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def _render_snapshot_metric(
    rows: list[dict[str, float]],
    *,
    y_keys: tuple[str, ...],
    title: str,
    ylabel: str,
):
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    if rows:
        x_values = [row["timesteps"] for row in rows]
        for key in y_keys:
            ax.plot(x_values, [row[key] for row in rows], label=key.replace("_", " "))
        ax.legend(fontsize=8)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def _render_rollout_gallery(rows: list[SequentialRolloutSummary]):
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.5))
    if rows:
        for summary in rows[:4]:
            axes[0].plot(summary.times, summary.positions, label=summary.proposal.proposal_id)
            axes[1].plot(range(len(summary.control_sequence)), summary.control_sequence, label=summary.proposal.proposal_id)
        axes[0].legend(fontsize=8)
        axes[1].legend(fontsize=8)
    axes[0].set_title("Selected PPO positions", loc="left", fontweight="bold")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Position")
    axes[1].set_title("Selected PPO control commands", loc="left", fontweight="bold")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Control")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def _checkpoint_manifest(run_dir: Path, ppo_config: SequentialPpoConfig) -> dict[str, object]:
    manifest_path = run_dir / "checkpoint_manifest.json"
    if manifest_path.exists():
        return {
            **asdict(ppo_config),
            **_read_json_compat(manifest_path),
        }
    return {
        "status": "not_started",
        "timesteps_completed": 0,
        "target_total_timesteps": int(ppo_config.total_timesteps),
        "latest_checkpoint_path": None,
        "resume_used": False,
        "checkpoint_interval_timesteps": int(ppo_config.checkpoint_interval_timesteps),
        "snapshot_interval_timesteps": int(ppo_config.snapshot_interval_timesteps),
    }


def _read_json_compat(path: Path) -> dict[str, object]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _config_for_objective(
    config: SequentialBoundaryControlConfig,
    objective: TrajectoryExplorationObjective,
) -> SequentialBoundaryControlConfig:
    target = objective.target
    class_name = target.class_name
    if class_name is None and target.class_pair is not None:
        class_name = target.class_pair[1]
    return replace(
        config,
        tier_name=target.target_tier or config.tier_name,
        class_name=class_name or config.class_name,
    )


def _algorithm_decision_report(training_summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "# RL Algorithm Decision Report",
            "",
            "- `PPO`: `go` for the first sequential-control witness because the simulator is cheap, the state is low-dimensional, and timing matters.",
            "- `Dreamer`: `no-go for now` because there is no sample-efficiency or latent-dynamics bottleneck yet.",
            "- `SAC/TD3`: `defer` unless PPO shows action-smoothness or stability failures.",
            "- `CEM`: `recommended adjacent comparator` for the next non-RL sequential benchmark.",
            "",
            "## Current Witness",
            "",
            f"- objective id: `{training_summary.get('objective_id', '')}`",
            "- environment: `sequential_boundary_control`",
            f"- status: `{training_summary['status']}`",
            f"- resume used: `{training_summary.get('resume_used', False)}`",
            f"- beats random control: `{training_summary.get('beats_random_control', False)}`",
            f"- beats scripted mean: `{training_summary.get('beats_scripted_mean', False)}`",
            "- promotion gate: PPO remains experimental until broader matched-budget comparisons clear.",
        ]
    )


def _report_markdown(training_summary: dict[str, object], rows: list[dict[str, object]], selected_rows: tuple[dict[str, object], ...]) -> str:
    return "\n".join(
        [
            "# PPO Boundary Control Report",
            "",
            "This run trains an SB3 PPO policy on a true sequential 1D acceleration-control witness and compares it to sequential heuristic baselines.",
            "",
            f"- status: `{training_summary['status']}`",
            f"- objective id: `{training_summary.get('objective_id', '')}`",
            f"- control problem id: `{training_summary.get('control_problem_id', '')}`",
            f"- vehicle family: `{training_summary.get('vehicle_family', '')}`",
            f"- PPO mean total utility: `{training_summary.get('ppo_mean_total_utility', 0.0):.3f}`",
            f"- PPO best total utility: `{training_summary.get('ppo_best_total_utility', 0.0):.3f}`",
            f"- timesteps completed: `{training_summary.get('timesteps_completed', 0)}` / `{training_summary.get('target_total_timesteps', 0)}`",
            f"- latest checkpoint: `{training_summary.get('latest_checkpoint_name', 'none')}`",
            f"- beats random control: `{training_summary.get('beats_random_control', False)}`",
            f"- beats scripted mean: `{training_summary.get('beats_scripted_mean', False)}`",
            f"- final mean boundary closeness: `{training_summary.get('final_mean_boundary_closeness', 0.0):.3f}`",
            f"- final mean feature excitation: `{training_summary.get('final_mean_feature_excitation', 0.0):.3f}`",
            f"- final boundary-hit fraction: `{training_summary.get('final_boundary_hit_fraction', 0.0):.3f}`",
            f"- final feature-hit fraction: `{training_summary.get('final_feature_target_hit_fraction', 0.0):.3f}`",
            f"- selected rollouts: `{len(selected_rows)}`",
        ]
    )


def analyze_sequential_ppo_boundary_control(
    *,
    config: SequentialBoundaryControlConfig | None = None,
    ppo_config: SequentialPpoConfig | None = None,
    objective: TrajectoryExplorationObjective | None = None,
    run_dir: str | Path | None = None,
) -> SequentialPpoResult:
    base_config = config or SequentialBoundaryControlConfig()
    resolved_ppo = ppo_config or SequentialPpoConfig()
    resolved_objective = objective or default_boundary_control_objective()
    resolved_config = _config_for_objective(base_config, resolved_objective)
    contract = sequential_environment_contract(resolved_config)
    baseline_bank = _baseline_sequences(resolved_config, resolved_ppo.eval_seed_start, objective=resolved_objective)
    resolved_run_dir = None if run_dir is None else Path(run_dir)
    manifest = (
        {
            "status": "not_run_dependency_missing",
            "timesteps_completed": 0,
            "target_total_timesteps": int(resolved_ppo.total_timesteps),
            "latest_checkpoint_path": None,
            "resume_used": False,
            "latest_checkpoint_name": "none",
        }
        if resolved_run_dir is None
        else _checkpoint_manifest(resolved_run_dir, resolved_ppo)
    )

    if not has_stable_baselines3_support():
        summary = {
            "status": "not_run_dependency_missing",
            "dependency_error": repr(_SB3_IMPORT_ERROR),
            "environment_contract": contract["environment_id"],
            "witness_task": contract["witness_task"],
            "objective_id": resolved_objective.objective_id,
            "timesteps_completed": 0,
            "target_total_timesteps": int(resolved_ppo.total_timesteps),
            "resume_used": False,
            "latest_checkpoint_name": "none",
        }
        report = _report_markdown(summary, [], ())
        decision = _algorithm_decision_report(summary)
        return SequentialPpoResult(
            training_summary=summary,
            checkpoint_manifest=manifest,
            evaluation_rows=(),
            selected_rollouts=(),
            control_sequences=(),
            ppo_vs_heuristics_rows=(),
            training_curve_rows=(),
            snapshot_rows=(),
            gallery_rollouts=(),
            report_markdown=report,
            rl_algorithm_decision_report_markdown=decision,
        )

    callback = _TrainingTraceCallback(
        run_dir=resolved_run_dir,
        config=resolved_config,
        ppo_config=resolved_ppo,
        objective=resolved_objective,
    )

    def _make_env():
        return Monitor(SequentialTrajectoryGym(objective=resolved_objective, config=resolved_config, seed=resolved_ppo.train_seed))

    vec_env = DummyVecEnv([_make_env])
    latest_checkpoint_path = manifest.get("latest_checkpoint_path")
    timesteps_completed = int(manifest.get("timesteps_completed", 0))
    resume_used = bool(resolved_ppo.resume_if_possible and latest_checkpoint_path and Path(str(latest_checkpoint_path)).exists())
    if resume_used:
        model = PPO.load(str(latest_checkpoint_path), env=vec_env)
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            seed=resolved_ppo.train_seed,
            verbose=0,
            n_steps=resolved_ppo.n_steps,
            batch_size=resolved_ppo.batch_size,
            learning_rate=resolved_ppo.learning_rate,
            gamma=resolved_ppo.gamma,
            gae_lambda=resolved_ppo.gae_lambda,
            clip_range=resolved_ppo.clip_range,
            ent_coef=resolved_ppo.ent_coef,
            vf_coef=resolved_ppo.vf_coef,
            policy_kwargs={"net_arch": list(resolved_ppo.hidden_sizes)},
        )
        timesteps_completed = 0
    remaining_timesteps = max(0, resolved_ppo.total_timesteps - timesteps_completed)
    if remaining_timesteps > 0:
        model.learn(total_timesteps=remaining_timesteps, callback=callback, progress_bar=False, reset_num_timesteps=not resume_used)
    if resolved_run_dir is not None:
        final_model_path = resolved_run_dir / "checkpoints" / "ppo_final_model.zip"
        final_model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(final_model_path)
        timesteps_completed = max(int(model.num_timesteps), int(manifest.get("timesteps_completed", 0)))
        manifest = {
            "status": "completed",
            "timesteps_completed": timesteps_completed,
            "target_total_timesteps": int(resolved_ppo.total_timesteps),
            "latest_checkpoint_path": str(final_model_path),
            "latest_checkpoint_name": final_model_path.name,
            "resume_used": resume_used,
            "checkpoint_interval_timesteps": int(resolved_ppo.checkpoint_interval_timesteps),
            "snapshot_interval_timesteps": int(resolved_ppo.snapshot_interval_timesteps),
        }
        _write_json(resolved_run_dir / "checkpoint_manifest.json", manifest)
    else:
        manifest = {
            "status": "completed",
            "timesteps_completed": int(model.num_timesteps),
            "target_total_timesteps": int(resolved_ppo.total_timesteps),
            "latest_checkpoint_path": None,
            "latest_checkpoint_name": "none",
            "resume_used": resume_used,
        }

    if not callback.snapshot_rows:
        callback._record_snapshot(model=model)

    ppo_summaries = _evaluate_policy_rollouts(model, config=resolved_config, ppo_config=resolved_ppo, objective=resolved_objective)
    baseline_results = {
        family_id: _evaluate_sequence_family(
            family_id,
            entries,
            objective=resolved_objective,
            config=resolved_config,
            eval_seed_start=resolved_ppo.eval_seed_start + (family_index + 1) * 100,
        )
        for family_index, (family_id, entries) in enumerate(baseline_bank.items())
    }
    scripted_sequences = [sequence for _, sequence in baseline_bank["scripted_profiles"]]

    evaluation_rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []
    ppo_utilities = [summary.evaluation.total_utility for summary in ppo_summaries]
    for summary in ppo_summaries:
        row = summary.evaluation.as_row()
        row["backend_family"] = "ppo"
        row["novelty_vs_scripted"] = min((_control_signature_distance(summary.control_sequence, sequence) for sequence in scripted_sequences), default=0.0)
        evaluation_rows.append(row)
        for step_index, control in enumerate(summary.control_sequence):
            control_rows.append(
                {
                    "proposal_id": summary.proposal.proposal_id,
                    "backend_id": "ppo_policy",
                    "step_index": step_index,
                    "control": control,
                }
            )
    comparison_rows: list[dict[str, object]] = []
    for family_id, summaries in baseline_results.items():
        utilities = [summary.evaluation.total_utility for summary in summaries]
        comparison_rows.append(
            {
                "backend_id": family_id,
                "mean_total_utility": mean(utilities),
                "best_total_utility": max(utilities),
                "count": len(utilities),
            }
        )
        for summary in summaries:
            row = summary.evaluation.as_row()
            row["backend_family"] = family_id
            row["novelty_vs_scripted"] = 0.0
            evaluation_rows.append(row)
            for step_index, control in enumerate(summary.control_sequence):
                control_rows.append(
                    {
                        "proposal_id": summary.proposal.proposal_id,
                        "backend_id": family_id,
                        "step_index": step_index,
                        "control": control,
                    }
                )
    scripted_mean = next(row["mean_total_utility"] for row in comparison_rows if row["backend_id"] == "scripted_profiles")
    random_mean = next(row["mean_total_utility"] for row in comparison_rows if row["backend_id"] == "random_control")
    ppo_row = {
        "backend_id": "ppo_policy",
        "mean_total_utility": mean(ppo_utilities),
        "best_total_utility": max(ppo_utilities),
        "count": len(ppo_utilities),
    }
    comparison_rows.insert(0, ppo_row)
    selected_rows = _selected_rows(evaluation_rows)
    final_snapshot = callback.snapshot_rows[-1] if callback.snapshot_rows else {
        "mean_boundary_closeness": mean(summary.evaluation.boundary_closeness for summary in ppo_summaries),
        "mean_feature_excitation": mean(summary.evaluation.feature_excitation for summary in ppo_summaries),
        "feature_target_hit_fraction": mean(1.0 if summary.evaluation.feature_excitation >= 0.75 else 0.0 for summary in ppo_summaries),
        "boundary_hit_fraction": mean(1.0 if summary.evaluation.boundary_closeness >= 0.60 else 0.0 for summary in ppo_summaries),
    }
    training_summary = {
        "status": "experimental",
        "witness_task": contract["witness_task"],
        "objective_id": resolved_objective.objective_id,
        "control_problem_id": resolved_config.control_problem.problem_id,
        "vehicle_family": resolved_config.control_problem.vehicle_family,
        "control_channel_names": [channel.name for channel in resolved_config.control_problem.control_channels],
        "ppo_mean_total_utility": ppo_row["mean_total_utility"],
        "ppo_best_total_utility": ppo_row["best_total_utility"],
        "random_mean_total_utility": random_mean,
        "scripted_mean_total_utility": scripted_mean,
        "beats_random_control": ppo_row["mean_total_utility"] > random_mean,
        "beats_scripted_mean": ppo_row["mean_total_utility"] > scripted_mean,
        "novel_rollout_count": sum(1 for row in evaluation_rows if row["backend_family"] == "ppo" and float(row["novelty_vs_scripted"]) > 0.18),
        "training_trace_rows": len(callback.episode_rows),
        "snapshot_rows": len(callback.snapshot_rows),
        "timesteps_completed": int(manifest["timesteps_completed"]),
        "target_total_timesteps": int(manifest["target_total_timesteps"]),
        "resume_used": bool(manifest["resume_used"]),
        "latest_checkpoint_name": str(manifest["latest_checkpoint_name"]),
        "final_mean_boundary_closeness": float(final_snapshot["mean_boundary_closeness"]),
        "final_mean_feature_excitation": float(final_snapshot["mean_feature_excitation"]),
        "final_feature_target_hit_fraction": float(final_snapshot["feature_target_hit_fraction"]),
        "final_boundary_hit_fraction": float(final_snapshot["boundary_hit_fraction"]),
    }
    report = _report_markdown(training_summary, evaluation_rows, selected_rows)
    decision = _algorithm_decision_report(training_summary)
    return SequentialPpoResult(
        training_summary=training_summary,
        checkpoint_manifest=manifest,
        evaluation_rows=tuple(evaluation_rows),
        selected_rollouts=selected_rows,
        control_sequences=tuple(control_rows),
        ppo_vs_heuristics_rows=tuple(comparison_rows),
        training_curve_rows=tuple(callback.episode_rows),
        snapshot_rows=tuple(callback.snapshot_rows),
        gallery_rollouts=tuple(ppo_summaries[:4]),
        report_markdown=report,
        rl_algorithm_decision_report_markdown=decision,
    )


def write_sequential_ppo_boundary_control_artifacts(
    output_dir: str | Path,
    *,
    config: SequentialBoundaryControlConfig | None = None,
    ppo_config: SequentialPpoConfig | None = None,
    objective: TrajectoryExplorationObjective | None = None,
    run_name: str = "ppo_boundary_control",
) -> SequentialPpoResult:
    resolved_config = config or SequentialBoundaryControlConfig()
    resolved_ppo = ppo_config or SequentialPpoConfig()
    root = Path(output_dir)
    resolved_run_name = run_name
    if objective is not None and run_name == "ppo_boundary_control" and objective.objective_id != default_boundary_control_objective().objective_id:
        resolved_run_name = f"ppo_boundary_control/{objective.objective_id}"
    run_dir = root / "trajectory_exploration_rl" / resolved_run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    rl_dir = root / "trajectory_exploration_rl"
    rl_dir.mkdir(parents=True, exist_ok=True)

    result = analyze_sequential_ppo_boundary_control(
        config=resolved_config,
        ppo_config=resolved_ppo,
        objective=objective,
        run_dir=run_dir,
    )
    environment_contract_path = run_dir / "environment_contract.json"
    control_problem_contract_path = run_dir / "control_problem_contract.json"
    transition_report_path = run_dir / "three_d_transition_report.md"
    training_config_path = run_dir / "training_config.json"
    checkpoint_manifest_path = run_dir / "checkpoint_manifest.json"
    training_summary_path = run_dir / "training_summary.json"
    evaluation_rows_path = run_dir / "evaluation_rows.csv"
    selected_rollouts_path = run_dir / "selected_rollouts.csv"
    control_sequences_path = run_dir / "control_sequences.csv"
    training_trace_rows_path = run_dir / "training_trace_rows.csv"
    snapshot_rows_path = run_dir / "snapshot_rows.csv"
    training_curve_path = run_dir / "ppo_training_curve.png"
    rollout_gallery_path = run_dir / "rollout_gallery.png"
    utility_progress_path = run_dir / "utility_progress.png"
    feature_progress_path = run_dir / "feature_progress.png"
    class_space_progress_path = run_dir / "class_space_progress.png"
    ppo_vs_heuristics_path = run_dir / "ppo_vs_heuristics.csv"
    summary_path = run_dir / "summary.csv"
    report_path = run_dir / "report.md"
    rl_algorithm_decision_report_path = rl_dir / "rl_algorithm_decision_report.md"

    _write_json(environment_contract_path, sequential_environment_contract(resolved_config))
    _write_json(control_problem_contract_path, resolved_config.control_problem.as_payload())
    _write_text(transition_report_path, _transition_report_markdown(resolved_config, objective or default_boundary_control_objective()))
    _write_json(training_config_path, {"environment": asdict(resolved_config), "ppo": asdict(resolved_ppo)})
    _write_json(checkpoint_manifest_path, result.checkpoint_manifest)
    _write_json(training_summary_path, result.training_summary)
    if result.evaluation_rows:
        write_csv(evaluation_rows_path, list(result.evaluation_rows), list(result.evaluation_rows[0].keys()))
    else:
        evaluation_rows_path.write_text("proposal_id\n", encoding="utf-8")
    if result.selected_rollouts:
        write_csv(selected_rollouts_path, list(result.selected_rollouts), list(result.selected_rollouts[0].keys()))
    else:
        selected_rollouts_path.write_text("proposal_id\n", encoding="utf-8")
    if result.control_sequences:
        write_csv(control_sequences_path, list(result.control_sequences), list(result.control_sequences[0].keys()))
    else:
        control_sequences_path.write_text("proposal_id\n", encoding="utf-8")
    if result.training_curve_rows:
        write_csv(training_trace_rows_path, list(result.training_curve_rows), list(result.training_curve_rows[0].keys()))
    else:
        training_trace_rows_path.write_text("timesteps,episode_return,episode_length,entropy_proxy\n", encoding="utf-8")
    if result.snapshot_rows:
        write_csv(snapshot_rows_path, list(result.snapshot_rows), list(result.snapshot_rows[0].keys()))
    else:
        snapshot_rows_path.write_text("timesteps,mean_total_utility\n", encoding="utf-8")
    if result.ppo_vs_heuristics_rows:
        write_csv(ppo_vs_heuristics_path, list(result.ppo_vs_heuristics_rows), list(result.ppo_vs_heuristics_rows[0].keys()))
        write_comparison_summary_csv(run_dir, result.ppo_vs_heuristics_rows, filename="summary.csv")
    else:
        ppo_vs_heuristics_path.write_text("backend_id\n", encoding="utf-8")
        summary_path.write_text("backend_id\n", encoding="utf-8")
    _write_text(report_path, result.report_markdown)
    _write_text(rl_algorithm_decision_report_path, result.rl_algorithm_decision_report_markdown)
    write_decision_card(run_dir, result.rl_algorithm_decision_report_markdown)
    write_plot(_render_training_curve(list(result.training_curve_rows)), training_curve_path)
    write_plot(_render_rollout_gallery(list(result.gallery_rollouts)), rollout_gallery_path)
    write_plot(
        _render_snapshot_metric(
            list(result.snapshot_rows),
            y_keys=("mean_total_utility",),
            title="PPO utility progress",
            ylabel="Mean total utility",
        ),
        utility_progress_path,
    )
    write_plot(
        _render_snapshot_metric(
            list(result.snapshot_rows),
            y_keys=("mean_feature_excitation", "feature_target_hit_fraction"),
            title="Feature-space exploration progress",
            ylabel="Feature objective value",
        ),
        feature_progress_path,
    )
    write_plot(
        _render_snapshot_metric(
            list(result.snapshot_rows),
            y_keys=("mean_boundary_closeness", "boundary_hit_fraction"),
            title="Class-space boundary exploration progress",
            ylabel="Boundary objective value",
        ),
        class_space_progress_path,
    )

    artifacts = SequentialPpoArtifacts(
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        environment_contract_path=environment_contract_path,
        control_problem_contract_path=control_problem_contract_path,
        transition_report_path=transition_report_path,
        training_config_path=training_config_path,
        checkpoint_manifest_path=checkpoint_manifest_path,
        training_summary_path=training_summary_path,
        evaluation_rows_path=evaluation_rows_path,
        selected_rollouts_path=selected_rollouts_path,
        control_sequences_path=control_sequences_path,
        training_trace_rows_path=training_trace_rows_path,
        snapshot_rows_path=snapshot_rows_path,
        training_curve_path=training_curve_path,
        rollout_gallery_path=rollout_gallery_path,
        utility_progress_path=utility_progress_path,
        feature_progress_path=feature_progress_path,
        class_space_progress_path=class_space_progress_path,
        ppo_vs_heuristics_path=ppo_vs_heuristics_path,
        report_path=report_path,
        rl_algorithm_decision_report_path=rl_algorithm_decision_report_path,
    )
    return SequentialPpoResult(
        training_summary=result.training_summary,
        checkpoint_manifest=result.checkpoint_manifest,
        evaluation_rows=result.evaluation_rows,
        selected_rollouts=result.selected_rollouts,
        control_sequences=result.control_sequences,
        ppo_vs_heuristics_rows=result.ppo_vs_heuristics_rows,
        training_curve_rows=result.training_curve_rows,
        snapshot_rows=result.snapshot_rows,
        gallery_rollouts=result.gallery_rollouts,
        report_markdown=result.report_markdown,
        rl_algorithm_decision_report_markdown=result.rl_algorithm_decision_report_markdown,
        artifacts=artifacts,
    )


def write_generated_trajectory_objective_ppo_sweep_artifacts(
    output_dir: str | Path,
    *,
    config: SequentialBoundaryControlConfig | None = None,
    ppo_config: SequentialPpoConfig | None = None,
    objective_ids: tuple[str, ...] | None = None,
) -> SequentialPpoSweepArtifacts:
    root = Path(output_dir)
    run_dir = root / "trajectory_exploration_rl" / "generated_objective_sweep"
    run_dir.mkdir(parents=True, exist_ok=True)
    suite = generate_trajectory_exploration_objective_suite()
    selected_objectives = tuple(
        objective
        for objective in suite.objectives
        if objective_ids is None or objective.objective_id in objective_ids
    )
    summary_rows: list[dict[str, object]] = []
    for objective in selected_objectives:
        result = write_sequential_ppo_boundary_control_artifacts(
            root,
            config=config,
            ppo_config=ppo_config,
            objective=objective,
            run_name=f"generated_objective_sweep/{objective.objective_id}",
        )
        summary_rows.append(
            {
                "objective_id": objective.objective_id,
                "mode": objective.mode,
                "geometry_target": objective.geometry_target,
                "target_id": objective.target.target_id,
                "target_type": objective.target.target_type,
                "ppo_mean_total_utility": result.training_summary["ppo_mean_total_utility"],
                "ppo_best_total_utility": result.training_summary["ppo_best_total_utility"],
                "beats_random_control": result.training_summary["beats_random_control"],
                "beats_scripted_mean": result.training_summary["beats_scripted_mean"],
                "final_mean_boundary_closeness": result.training_summary["final_mean_boundary_closeness"],
                "final_mean_feature_excitation": result.training_summary["final_mean_feature_excitation"],
                "latest_checkpoint_name": result.training_summary["latest_checkpoint_name"],
                "run_dir": str(root / "trajectory_exploration_rl" / "generated_objective_sweep" / objective.objective_id),
            }
        )
    manifest = {
        "objective_count": len(selected_objectives),
        "objective_ids": [objective.objective_id for objective in selected_objectives],
        "source_spec_id": suite.spec.spec_id,
    }
    report = "\n".join(
        [
            "# Generated Objective PPO Sweep",
            "",
            f"- source spec id: `{suite.spec.spec_id}`",
            f"- objective count: `{len(selected_objectives)}`",
            "",
            "This bundle runs the PPO sequential-control witness against mechanically generated objective regions so feature-space and class-space targeting can be compared without hand-authored objectives.",
        ]
    )
    manifest_path = run_dir / "manifest.json"
    summary_rows_path = run_dir / "summary_rows.csv"
    summary_path = run_dir / "summary.csv"
    report_path = run_dir / "report.md"
    _write_json(manifest_path, manifest)
    if summary_rows:
        write_csv(summary_rows_path, summary_rows, list(summary_rows[0].keys()))
        write_comparison_summary_csv(run_dir, summary_rows, filename="summary.csv")
    else:
        summary_rows_path.write_text("objective_id\n", encoding="utf-8")
        summary_path.write_text("objective_id\n", encoding="utf-8")
    _write_text(report_path, report)
    write_decision_card(
        run_dir,
        "\n".join(
            [
                "# Decision Card",
                "",
                f"- source spec id: `{suite.spec.spec_id}`",
                f"- objective count: `{len(selected_objectives)}`",
                "- report: `report.md`",
            ]
        ),
    )
    return SequentialPpoSweepArtifacts(
        run_dir=run_dir,
        manifest_path=manifest_path,
        summary_rows_path=summary_rows_path,
        report_path=report_path,
    )
