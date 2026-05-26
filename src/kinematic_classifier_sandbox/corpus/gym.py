from __future__ import annotations

from dataclasses import asdict

from ..analysis.feature_analysis import _one_dimensional_feature_context_from_trajectory
from .gym_rendering import render_corpus_gym_numeric_walkthrough_markdown, write_corpus_gym_artifacts
from .gym_types import (
    CorpusGymAction,
    CorpusGymArtifacts,
    CorpusGymContractResult,
    CorpusGymEpisode,
    CorpusGymReward,
    CorpusGymTarget,
)
from .gym_utils import (
    _boundary_closeness_score,
    _class_validity_score,
    _coverage_gain_score,
    _feature_excitation_score,
    _leakage_penalty,
    _physical_invalidity_penalty,
    _prior_sensitivity_score,
    _reward_from_components,
    _trajectory_dataset_for_context,
    _simulate_trajectory,
    _classifier_stress_score,
)
from .policy import CorpusPolicySpec, load_corpus_policy_spec


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


class CorpusGymEnvironment:
    def __init__(self, policy: CorpusPolicySpec | None = None) -> None:
        self._policy = policy or load_corpus_policy_spec()
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

    def trajectory(self) -> object | None:
        return self._last_episode.trajectory if self._last_episode is not None else None

    def score(self, trajectory, *, action: CorpusGymAction | None = None) -> CorpusGymReward:
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
            policy=self._policy,
            class_validity=class_validity,
            feature_excitation=feature_excitation,
            coverage_gain=coverage_gain,
            boundary_closeness=boundary_closeness,
            classifier_stress=classifier_stress,
            prior_sensitivity=prior_sensitivity,
            leakage_penalty=leakage_penalty,
            physical_invalidity_penalty=physical_invalidity_penalty,
        )

    def render_diagnostics(self, trajectory, *, reward: CorpusGymReward | None = None) -> dict[str, object]:
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
    policy = load_corpus_policy_spec()
    return {
        "environment_id": "corpus_gym_v1",
        "policy_id": policy.policy_id,
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
        "reward_weights": policy.corpus_gym_weights,
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
