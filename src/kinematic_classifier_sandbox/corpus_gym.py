from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import random

from .feature_analysis import _one_dimensional_feature_context_from_trajectory
from .trajectory_generator import (
    DatasetTierDefinition,
    GeneratedTrajectoryDataset,
    TrajectoryArtifact,
    _class_by_name,
    _generate_states,
    _generate_times,
    _inject_measurement_noise,
    _make_trajectory,
    _sample_parameters,
    _sample_steps_and_dt,
    _tier_by_name,
)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass(frozen=True, slots=True)
class CorpusGymTarget:
    target_id: str
    target_type: str
    description: str
    class_name: str | None = None
    class_pair: tuple[str, str] | None = None
    feature_constraints: dict[str, dict[str, float]] | None = None
    target_tier: str | None = None
    target_failure_mode: str | None = None
    target_prior_sensitivity: str | None = None
    target_switching_pattern: str | None = None


@dataclass(frozen=True, slots=True)
class CorpusGymAction:
    seed: int
    tier_name: str
    duration_scale: float = 1.0
    measurement_scale: float = 1.0
    irregularity_scale: float = 1.0
    outlier_scale: float = 1.0
    step_scale: float = 1.0


@dataclass(frozen=True, slots=True)
class CorpusGymReward:
    class_validity: float
    feature_excitation: float
    coverage_gain: float
    boundary_closeness: float
    classifier_stress: float
    prior_sensitivity: float
    leakage_penalty: float
    physical_invalidity_penalty: float
    total_utility: float


@dataclass(frozen=True, slots=True)
class CorpusGymEpisode:
    target: CorpusGymTarget
    action: CorpusGymAction
    trajectory: TrajectoryArtifact
    diagnostics: dict[str, object]
    reward: CorpusGymReward


@dataclass(frozen=True, slots=True)
class CorpusGymContractResult:
    environment_contract: dict[str, object]
    example_targets: tuple[dict[str, object], ...]
    example_episode: CorpusGymEpisode
    report_markdown: str


@dataclass(frozen=True, slots=True)
class CorpusGymArtifacts:
    run_dir: Path
    environment_contract_path: Path
    example_targets_path: Path
    report_path: Path


def default_corpus_gym_targets() -> tuple[CorpusGymTarget, ...]:
    return (
        CorpusGymTarget(
            target_id="target_class_constant_acceleration",
            target_type="target_class",
            description="Generate valid constant-acceleration examples for nominal corpus growth.",
            class_name="constant_acceleration",
            target_tier="realistic_v1",
        ),
        CorpusGymTarget(
            target_id="target_class_pair_cv_vs_ca_boundary",
            target_type="target_class_pair",
            description="Search for short-horizon boundary cases between constant velocity and constant acceleration.",
            class_pair=("constant_velocity", "constant_acceleration"),
            target_tier="boundary_v1",
        ),
        CorpusGymTarget(
            target_id="target_feature_cell_high_accel_low_monotonicity",
            target_type="target_feature_cell",
            description="Find trajectories with high acceleration range and reduced monotonicity.",
            class_name="maneuver",
            feature_constraints={
                "acceleration_range": {"min": 0.60},
                "monotonicity": {"max": 0.92},
            },
            target_tier="adversarial_v1",
        ),
        CorpusGymTarget(
            target_id="target_difficulty_short_noisy_boundary",
            target_type="target_difficulty",
            description="Generate short noisy boundary trajectories for hard class-pair comparisons.",
            class_name="constant_velocity",
            target_tier="stress_v1",
        ),
        CorpusGymTarget(
            target_id="target_failure_raw_extrema_outlier",
            target_type="target_failure_mode",
            description="Find trajectories likely to trigger raw-extrema weakness under outlier corruption.",
            class_name="constant_velocity",
            target_failure_mode="raw_extrema_failure",
            target_tier="adversarial_v1",
        ),
        CorpusGymTarget(
            target_id="target_prior_sensitivity_small_flip",
            target_type="target_prior_sensitivity",
            description="Find boundary cases where small prior changes are likely to matter.",
            class_pair=("constant_velocity", "braking"),
            target_prior_sensitivity="high",
            target_tier="boundary_v1",
        ),
        CorpusGymTarget(
            target_id="target_switching_velocity_to_braking",
            target_type="target_switching_pattern",
            description="Specify a future switching search target for velocity-to-braking trajectories.",
            class_pair=("constant_velocity", "braking"),
            target_switching_pattern="constant_velocity_to_braking",
            target_tier="boundary_v1",
        ),
    )


def _scaled_tier_definition(action: CorpusGymAction) -> DatasetTierDefinition:
    base = _tier_by_name(action.tier_name)
    scaled_steps = (
        max(4, int(round(base.steps_range[0] * action.step_scale))),
        max(5, int(round(base.steps_range[1] * action.step_scale))),
    )
    if scaled_steps[0] >= scaled_steps[1]:
        scaled_steps = (scaled_steps[0], scaled_steps[0] + 1)
    return replace(
        base,
        steps_range=scaled_steps,
        measurement_std_range=(
            max(0.001, base.measurement_std_range[0] * action.measurement_scale),
            max(0.002, base.measurement_std_range[1] * action.measurement_scale),
        ),
        outlier_probability=_clamp(base.outlier_probability * action.outlier_scale, 0.0, 0.35),
        irregular_sampling_strength=_clamp(base.irregular_sampling_strength * action.irregularity_scale, 0.0, 1.0),
    )


def _feature_value_matches(value: float, constraints: dict[str, float]) -> float:
    score = 1.0
    if "min" in constraints:
        threshold = float(constraints["min"])
        score *= 1.0 if value >= threshold else _clamp(value / max(threshold, 1e-6), 0.0, 1.0)
    if "max" in constraints:
        threshold = float(constraints["max"])
        if value <= threshold:
            score *= 1.0
        else:
            score *= _clamp(threshold / max(value, 1e-6), 0.0, 1.0)
    return score


def _trajectory_dataset_for_context(trajectory: TrajectoryArtifact, tier_name: str, seed: int) -> GeneratedTrajectoryDataset:
    class_definition = _class_by_name(trajectory.true_class)
    tier_definition = _tier_by_name(tier_name)
    return GeneratedTrajectoryDataset(
        tier=tier_name,
        seed=seed,
        class_definitions=(class_definition,),
        tier_definition=tier_definition,
        trajectories=(trajectory,),
    )


def _class_validity_score(trajectory: TrajectoryArtifact, tier_name: str) -> float:
    dataset = _trajectory_dataset_for_context(trajectory, tier_name, trajectory.seed)
    context = _one_dimensional_feature_context_from_trajectory(dataset, trajectory)
    velocities = list(trajectory.true_velocity or ())
    accelerations = list(trajectory.true_acceleration or ())
    if trajectory.true_class == "stationary":
        return _clamp(1.0 - abs(context.speed_range) / 0.4, 0.0, 1.0)
    if trajectory.true_class == "constant_velocity":
        return _clamp(1.0 - context.acceleration_variance / 0.10, 0.0, 1.0)
    if trajectory.true_class == "constant_acceleration":
        return _clamp(1.0 - context.acceleration_variance / 0.02, 0.0, 1.0)
    if trajectory.true_class == "braking":
        if not velocities or not accelerations:
            return 0.0
        slowdown = 1.0 if velocities[-1] <= velocities[0] else 0.0
        negative_support = sum(1 for value in accelerations if value <= 0.0) / max(len(accelerations), 1)
        return 0.5 * slowdown + 0.5 * negative_support
    if trajectory.true_class == "maneuver":
        return _clamp(context.acceleration_sign_changes / 2.0, 0.0, 1.0)
    if trajectory.true_class == "oscillatory":
        return _clamp(context.velocity_sign_changes / 3.0, 0.0, 1.0)
    if trajectory.true_class == "bounded_acceleration":
        accel_limit = float(trajectory.generator_parameters.get("accel_limit", 1.0))
        max_abs = max((abs(value) for value in accelerations), default=0.0)
        return 1.0 if max_abs <= accel_limit + 1e-6 else _clamp(accel_limit / max_abs, 0.0, 1.0)
    return 0.5


def _feature_excitation_score(target: CorpusGymTarget, trajectory: TrajectoryArtifact, tier_name: str) -> float:
    if not target.feature_constraints:
        return 0.0
    dataset = _trajectory_dataset_for_context(trajectory, tier_name, trajectory.seed)
    context = _one_dimensional_feature_context_from_trajectory(dataset, trajectory)
    values = {
        "duration": context.duration,
        "position_range": context.position_range,
        "speed_range": context.speed_range,
        "acceleration_variance": context.acceleration_variance,
        "acceleration_range": context.acceleration_range,
        "velocity_sign_changes": float(context.velocity_sign_changes),
        "acceleration_sign_changes": float(context.acceleration_sign_changes),
        "monotonicity": context.monotonicity,
        "linear_fit_residual": context.linear_fit_residual,
        "quadratic_fit_residual": context.quadratic_fit_residual,
        "outlier_score": context.outlier_score,
        "sampling_irregularity": context.sampling_irregularity,
    }
    scores = [
        _feature_value_matches(values.get(feature_name, 0.0), constraints)
        for feature_name, constraints in sorted(target.feature_constraints.items())
    ]
    return sum(scores) / max(len(scores), 1)


def _boundary_closeness_score(target: CorpusGymTarget, trajectory: TrajectoryArtifact, tier_name: str) -> float:
    dataset = _trajectory_dataset_for_context(trajectory, tier_name, trajectory.seed)
    context = _one_dimensional_feature_context_from_trajectory(dataset, trajectory)
    pair = target.class_pair
    if pair == ("constant_velocity", "constant_acceleration"):
        return _clamp((1.0 - min(context.acceleration_range / 1.0, 1.0)) * min(2.2 / max(context.duration, 1e-6), 1.0), 0.0, 1.0)
    if pair == ("constant_velocity", "braking"):
        end_speed = abs((trajectory.true_velocity or (0.0,))[-1])
        return _clamp((1.0 - min(context.acceleration_range / 1.4, 1.0)) * min((end_speed + 0.2) / 1.2, 1.0), 0.0, 1.0)
    if pair == ("constant_acceleration", "maneuver"):
        return _clamp((1.0 - min(context.acceleration_sign_changes / 2.0, 1.0)) * (1.0 - min(context.acceleration_variance / 0.2, 1.0)), 0.0, 1.0)
    return 0.0


def _classifier_stress_score(target: CorpusGymTarget, trajectory: TrajectoryArtifact, tier_name: str) -> float:
    boundary = _boundary_closeness_score(target, trajectory, tier_name)
    dataset = _trajectory_dataset_for_context(trajectory, tier_name, trajectory.seed)
    context = _one_dimensional_feature_context_from_trajectory(dataset, trajectory)
    outlier_factor = _clamp(context.outlier_score / 6.0, 0.0, 1.0)
    irregularity_factor = _clamp(context.sampling_irregularity / 0.6, 0.0, 1.0)
    if target.target_failure_mode == "raw_extrema_failure":
        return _clamp(0.55 * outlier_factor + 0.25 * irregularity_factor + 0.20 * boundary, 0.0, 1.0)
    return _clamp(0.60 * boundary + 0.20 * outlier_factor + 0.20 * irregularity_factor, 0.0, 1.0)


def _prior_sensitivity_score(target: CorpusGymTarget, trajectory: TrajectoryArtifact, tier_name: str) -> float:
    dataset = _trajectory_dataset_for_context(trajectory, tier_name, trajectory.seed)
    context = _one_dimensional_feature_context_from_trajectory(dataset, trajectory)
    boundary = _boundary_closeness_score(target, trajectory, tier_name)
    duration_factor = _clamp(2.0 / max(context.duration, 1e-6), 0.0, 1.0)
    if target.target_prior_sensitivity == "high":
        return _clamp(0.60 * boundary + 0.40 * duration_factor, 0.0, 1.0)
    return 0.20 * boundary


def _coverage_gain_score(target: CorpusGymTarget, action: CorpusGymAction, trajectory: TrajectoryArtifact, tier_name: str) -> float:
    tier_match = 1.0 if (target.target_tier is None or target.target_tier == tier_name) else 0.35
    class_match = 1.0 if (target.class_name is None or target.class_name == trajectory.true_class) else 0.25
    novelty = _clamp(
        0.25 * min(action.measurement_scale, 1.5)
        + 0.25 * min(action.irregularity_scale, 1.5)
        + 0.25 * min(action.outlier_scale, 1.5)
        + 0.25 * min(action.step_scale, 1.5),
        0.0,
        1.0,
    )
    return 0.40 * tier_match + 0.30 * class_match + 0.30 * novelty


def _leakage_penalty(action: CorpusGymAction, trajectory: TrajectoryArtifact, tier_name: str) -> float:
    dataset = _trajectory_dataset_for_context(trajectory, tier_name, trajectory.seed)
    context = _one_dimensional_feature_context_from_trajectory(dataset, trajectory)
    duration_risk = _clamp(context.duration / 24.0, 0.0, 1.0)
    sample_risk = _clamp(len(trajectory.times) / 36.0, 0.0, 1.0)
    noise_risk = _clamp(float(trajectory.measurement_std or 0.0) / 0.20, 0.0, 1.0)
    return 0.30 * duration_risk + 0.30 * sample_risk + 0.40 * noise_risk


def _physical_invalidity_penalty(trajectory: TrajectoryArtifact) -> float:
    times = list(trajectory.times)
    if len(times) < 2:
        return 1.0
    non_increasing = any(times[index] <= times[index - 1] for index in range(1, len(times)))
    if non_increasing:
        return 1.0
    accelerations = [abs(value) for value in (trajectory.true_acceleration or ())]
    accel_penalty = _clamp(max(accelerations, default=0.0) / 5.0, 0.0, 1.0)
    return 0.0 if accel_penalty <= 0.8 else accel_penalty


def _reward_from_components(
    *,
    class_validity: float,
    feature_excitation: float,
    coverage_gain: float,
    boundary_closeness: float,
    classifier_stress: float,
    prior_sensitivity: float,
    leakage_penalty: float,
    physical_invalidity_penalty: float,
) -> CorpusGymReward:
    total_utility = _clamp(
        0.22 * class_validity
        + 0.14 * feature_excitation
        + 0.14 * coverage_gain
        + 0.14 * boundary_closeness
        + 0.14 * classifier_stress
        + 0.12 * prior_sensitivity
        - 0.10 * leakage_penalty
        - 0.14 * physical_invalidity_penalty,
        0.0,
        1.0,
    )
    return CorpusGymReward(
        class_validity=class_validity,
        feature_excitation=feature_excitation,
        coverage_gain=coverage_gain,
        boundary_closeness=boundary_closeness,
        classifier_stress=classifier_stress,
        prior_sensitivity=prior_sensitivity,
        leakage_penalty=leakage_penalty,
        physical_invalidity_penalty=physical_invalidity_penalty,
        total_utility=total_utility,
    )


def _simulate_trajectory(target: CorpusGymTarget, action: CorpusGymAction) -> TrajectoryArtifact:
    if target.class_name is not None:
        class_name = target.class_name
    elif target.class_pair is not None:
        class_name = target.class_pair[0]
    else:
        class_name = "constant_velocity"
    class_definition = _class_by_name(class_name)
    tier_definition = _scaled_tier_definition(action)
    local_rng = random.Random(action.seed)
    steps, dt, measurement_std = _sample_steps_and_dt(local_rng, class_definition, tier_definition)
    params = _sample_parameters(local_rng, class_definition, tier_definition)
    times = _generate_times(
        local_rng,
        steps,
        dt,
        tier_definition.irregular_sampling_strength + class_definition.irregular_sampling_strength,
    )
    positions_true, velocities_true, accelerations_true = _generate_states(class_definition, times, params)
    measurements, outlier_indices = _inject_measurement_noise(
        local_rng,
        positions_true,
        measurement_std,
        tier_definition.outlier_probability + class_definition.outlier_probability,
    )
    scenario_id = f"{target.target_id}_{class_name}_{action.seed}"
    return _make_trajectory(
        class_definition=class_definition,
        tier_definition=tier_definition,
        steps=steps,
        dt=dt,
        measurement_std=measurement_std,
        seed=action.seed,
        scenario_id=scenario_id,
        params=params,
        times=times,
        positions_true=positions_true,
        velocities_true=velocities_true,
        accelerations_true=accelerations_true,
        measurements=measurements,
        outlier_indices=outlier_indices,
    )


class CorpusGymEnvironment:
    def __init__(self) -> None:
        self._target: CorpusGymTarget | None = None
        self._action_history: list[CorpusGymAction] = []
        self._last_episode: CorpusGymEpisode | None = None

    def reset(self, target: CorpusGymTarget) -> dict[str, object]:
        self._target = target
        self._action_history = []
        self._last_episode = None
        return {
            "target_id": target.target_id,
            "target_type": target.target_type,
            "step_index": 0,
            "done": False,
        }

    def simulate(self, parameterization: CorpusGymAction) -> CorpusGymEpisode:
        if self._target is None:
            raise RuntimeError("reset(target) must be called before simulate(parameterization)")
        trajectory = _simulate_trajectory(self._target, parameterization)
        reward = self.score(trajectory, action=parameterization)
        diagnostics = self.render_diagnostics(trajectory, reward=reward)
        episode = CorpusGymEpisode(
            target=self._target,
            action=parameterization,
            trajectory=trajectory,
            diagnostics=diagnostics,
            reward=reward,
        )
        self._last_episode = episode
        return episode

    def step(self, action: CorpusGymAction) -> dict[str, object]:
        self._action_history.append(action)
        episode = self.simulate(action)
        return {
            "done": True,
            "trajectory_id": episode.trajectory.trajectory_id,
            "reward": asdict(episode.reward),
            "diagnostics": episode.diagnostics,
            "action_history_length": len(self._action_history),
        }

    def trajectory(self) -> TrajectoryArtifact | None:
        return self._last_episode.trajectory if self._last_episode is not None else None

    def score(self, trajectory: TrajectoryArtifact, *, action: CorpusGymAction | None = None) -> CorpusGymReward:
        if self._target is None:
            raise RuntimeError("reset(target) must be called before score(trajectory)")
        tier_name = action.tier_name if action is not None else str(trajectory.generator_parameters.get("tier", "realistic_v1"))
        class_validity = _class_validity_score(trajectory, tier_name)
        feature_excitation = _feature_excitation_score(self._target, trajectory, tier_name)
        coverage_gain = _coverage_gain_score(self._target, action or CorpusGymAction(seed=trajectory.seed, tier_name=tier_name), trajectory, tier_name)
        boundary_closeness = _boundary_closeness_score(self._target, trajectory, tier_name)
        classifier_stress = _classifier_stress_score(self._target, trajectory, tier_name)
        prior_sensitivity = _prior_sensitivity_score(self._target, trajectory, tier_name)
        leakage_penalty = _leakage_penalty(action or CorpusGymAction(seed=trajectory.seed, tier_name=tier_name), trajectory, tier_name)
        physical_invalidity_penalty = _physical_invalidity_penalty(trajectory)
        return _reward_from_components(
            class_validity=class_validity,
            feature_excitation=feature_excitation,
            coverage_gain=coverage_gain,
            boundary_closeness=boundary_closeness,
            classifier_stress=classifier_stress,
            prior_sensitivity=prior_sensitivity,
            leakage_penalty=leakage_penalty,
            physical_invalidity_penalty=physical_invalidity_penalty,
        )

    def render_diagnostics(self, trajectory: TrajectoryArtifact, *, reward: CorpusGymReward | None = None) -> dict[str, object]:
        tier_name = str(trajectory.generator_parameters.get("tier", "realistic_v1"))
        dataset = _trajectory_dataset_for_context(trajectory, tier_name, trajectory.seed)
        context = _one_dimensional_feature_context_from_trajectory(dataset, trajectory)
        diagnostics = {
            "trajectory_id": trajectory.trajectory_id,
            "true_class": trajectory.true_class,
            "tier": tier_name,
            "duration": context.duration,
            "position_range": context.position_range,
            "speed_range": context.speed_range,
            "acceleration_range": context.acceleration_range,
            "acceleration_variance": context.acceleration_variance,
            "monotonicity": context.monotonicity,
            "outlier_score": context.outlier_score,
            "sampling_irregularity": context.sampling_irregularity,
            "num_samples": len(trajectory.times),
        }
        if reward is not None:
            diagnostics["reward"] = asdict(reward)
        return diagnostics


def _environment_contract() -> dict[str, object]:
    return {
        "environment_id": "corpus_gym_v1",
        "purpose": "Target-conditioned trajectory and corpus search contract layered over the existing 1D motion generator.",
        "interfaces": {
            "reset": {"input": "CorpusGymTarget", "output": ["target_id", "target_type", "step_index", "done"]},
            "step": {"input": "CorpusGymAction", "output": ["done", "trajectory_id", "reward", "diagnostics", "action_history_length"]},
            "simulate": {"input": "CorpusGymAction", "output": "CorpusGymEpisode"},
            "trajectory": {"input": None, "output": "TrajectoryArtifact | None"},
            "score": {"input": "TrajectoryArtifact", "output": "CorpusGymReward"},
            "render_diagnostics": {"input": "TrajectoryArtifact", "output": "dict"},
        },
        "target_types": [
            "target_class",
            "target_class_pair",
            "target_feature_cell",
            "target_difficulty",
            "target_failure_mode",
            "target_prior_sensitivity",
            "target_switching_pattern",
        ],
        "reward_components": [
            "class_validity",
            "feature_excitation",
            "coverage_gain",
            "boundary_closeness",
            "classifier_stress",
            "prior_sensitivity",
            "leakage_penalty",
            "physical_invalidity_penalty",
            "total_utility",
        ],
        "backend_policy": {
            "deep_rl_default": False,
            "recommended_progression": [
                "random_or_doe_search",
                "rejection_or_scoring_search",
                "quality_diversity_archive",
                "adaptive_stress_search",
                "rl_backend_decision",
            ],
        },
    }


def analyze_corpus_gym_contract() -> CorpusGymContractResult:
    environment = CorpusGymEnvironment()
    targets = default_corpus_gym_targets()
    example_target = targets[2]
    environment.reset(example_target)
    example_episode = environment.simulate(
        CorpusGymAction(
            seed=71,
            tier_name=example_target.target_tier or "adversarial_v1",
            duration_scale=0.92,
            measurement_scale=1.10,
            irregularity_scale=1.15,
            outlier_scale=1.05,
            step_scale=0.95,
        )
    )
    report_markdown = "\n".join(
        [
            "# CorpusGym Environment Contract",
            "",
            "This artifact defines the first search-facing environment contract for trajectory and corpus synthesis.",
            "",
            "## Core Design",
            "",
            "- The environment wraps the existing 1D generator instead of replacing it.",
            "- One episode currently generates one reproducible trajectory from a target plus action parameterization.",
            "- Reward is decomposed into validity, excitation, coverage, boundary, stress, prior, leakage, and invalidity terms.",
            "- RL is intentionally not the default backend. The contract is designed so search, QD, and stress backends can plug in first.",
            "",
            "## Example Target",
            "",
            f"- target_id: `{example_episode.target.target_id}`",
            f"- target_type: `{example_episode.target.target_type}`",
            f"- generated_class: `{example_episode.trajectory.true_class}`",
            f"- generated_tier: `{example_episode.trajectory.generator_parameters.get('tier')}`",
            f"- total_utility: `{example_episode.reward.total_utility:.3f}`",
            "",
            "## Reward Breakdown",
            "",
            f"- class_validity: `{example_episode.reward.class_validity:.3f}`",
            f"- feature_excitation: `{example_episode.reward.feature_excitation:.3f}`",
            f"- coverage_gain: `{example_episode.reward.coverage_gain:.3f}`",
            f"- boundary_closeness: `{example_episode.reward.boundary_closeness:.3f}`",
            f"- classifier_stress: `{example_episode.reward.classifier_stress:.3f}`",
            f"- prior_sensitivity: `{example_episode.reward.prior_sensitivity:.3f}`",
            f"- leakage_penalty: `{example_episode.reward.leakage_penalty:.3f}`",
            f"- physical_invalidity_penalty: `{example_episode.reward.physical_invalidity_penalty:.3f}`",
            "",
            "## Current Limits",
            "",
            "- This M25 version is contract-first and uses heuristic reward components rather than a full classifier-backed search objective.",
            "- Episodes are single-trajectory generations for now; sequential control and archive logic belong to later milestones.",
            "- Switching-pattern targets are represented in the target vocabulary now, but will need dedicated simulator support in later milestones.",
        ]
    )
    return CorpusGymContractResult(
        environment_contract=_environment_contract(),
        example_targets=tuple(asdict(target) for target in targets),
        example_episode=example_episode,
        report_markdown=report_markdown,
    )


def write_corpus_gym_artifacts(
    output_dir: str | Path,
    *,
    result: CorpusGymContractResult | None = None,
) -> CorpusGymArtifacts:
    contract = result or analyze_corpus_gym_contract()
    run_dir = Path(output_dir) / "corpus_gym"
    run_dir.mkdir(parents=True, exist_ok=True)
    environment_contract_path = run_dir / "environment_contract.json"
    example_targets_path = run_dir / "example_targets.json"
    report_path = run_dir / "corpus_gym_report.md"
    environment_contract_path.write_text(json.dumps(contract.environment_contract, indent=2), encoding="utf-8")
    example_targets_path.write_text(json.dumps({"targets": list(contract.example_targets)}, indent=2), encoding="utf-8")
    report_path.write_text(contract.report_markdown, encoding="utf-8")
    return CorpusGymArtifacts(
        run_dir=run_dir,
        environment_contract_path=environment_contract_path,
        example_targets_path=example_targets_path,
        report_path=report_path,
    )
