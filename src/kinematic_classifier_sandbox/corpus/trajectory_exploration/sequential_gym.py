from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import mean

import gymnasium as gym
import numpy
from gymnasium import spaces

from ...analysis.feature_analysis import _one_dimensional_feature_context_from_trajectory
from ...trajectory_generator import (
    GeneratedTrajectoryDataset,
    TrajectoryArtifact,
    _class_by_name,
    _make_manual_trajectory,
    _tier_by_name,
)
from ...utils.math import _clamp
from ..gym_types import CorpusGymAction, CorpusGymTarget
from ..gym_utils import (
    _boundary_closeness_score,
    _class_validity_score,
    _coverage_gain_score,
    _feature_excitation_score,
    _leakage_penalty,
    _physical_invalidity_penalty,
    _prior_sensitivity_score,
    _reward_from_components,
)
from .contracts import (
    TrajectoryExplorationEvaluation,
    TrajectoryExplorationObjective,
    TrajectoryExplorationProposal,
)
from .objective_scoring import (
    posterior_target_spec_from_payload,
    score_posterior_target_distribution,
)
from .sequential_control_specs import (
    SequentialControlProblemSpec,
    default_air_vehicle_control_problem_spec,
    default_one_dimensional_acceleration_problem_spec,
    default_three_dimensional_point_mass_problem_spec,
)


@dataclass(frozen=True, slots=True)
class SequentialBoundaryControlConfig:
    episode_horizon: int = 16
    dt: float = 0.25
    acceleration_limit: float = 1.25
    process_noise_std: float = 0.0
    measurement_std: float = 0.04
    tier_name: str = "boundary_v1"
    class_name: str = "constant_acceleration"
    scenario_family: str = "sequential_boundary_control"
    boundary_pair: tuple[str, str] = ("constant_velocity", "constant_acceleration")
    position_limit: float = 18.0
    velocity_limit: float = 4.0
    nontrivial_speed_floor: float = 0.18
    control_problem: SequentialControlProblemSpec = field(default_factory=default_one_dimensional_acceleration_problem_spec)


@dataclass(frozen=True, slots=True)
class SequentialRolloutSummary:
    proposal: TrajectoryExplorationProposal
    evaluation: TrajectoryExplorationEvaluation
    trajectory_id: str
    control_sequence: tuple[float, ...]
    times: tuple[float, ...]
    positions: tuple[float, ...]
    velocities: tuple[float, ...]
    accelerations: tuple[float, ...]
    measurements: tuple[float, ...]
    reward_trace: tuple[float, ...]
    reward_components: dict[str, float]


def default_boundary_control_objective() -> TrajectoryExplorationObjective:
    target = CorpusGymTarget(
        target_id="boundary_shaping_cv_vs_ca",
        target_type="class_pair_boundary",
        description="Sequential 1D acceleration control for the CV/CA boundary witness.",
        class_name="constant_acceleration",
        class_pair=("constant_velocity", "constant_acceleration"),
        target_tier="boundary_v1",
        feature_constraints={
            "acceleration_range": {"min": 0.15, "max": 1.05},
            "acceleration_variance": {"min": 0.01, "max": 0.30},
            "duration": {"min": 2.0, "max": 4.5},
        },
        target_prior_sensitivity="high",
    )
    return TrajectoryExplorationObjective(
        objective_id="boundary_shaping_cv_vs_ca",
        mode="sequential_control",
        geometry_target="create_ambiguous_boundary_trajectories",
        description=target.description,
        target=target,
        reward_weights={
            "coverage_gain": 0.15,
            "feature_excitation": 0.20,
            "boundary_closeness": 0.35,
            "class_validity": 0.15,
            "prior_sensitivity": 0.15,
        },
        thresholds={"min_total_utility": 0.42, "min_boundary_closeness": 0.58},
        evaluation_budget=64,
        backend_constraints={"action_mode": "sequential_control", "control_dim": 1},
        classifier_family="boundary_witness",
    )


def _trajectory_dataset(trajectory: TrajectoryArtifact, tier_name: str) -> GeneratedTrajectoryDataset:
    return GeneratedTrajectoryDataset(
        tier=tier_name,
        seed=trajectory.seed,
        class_definitions=(_class_by_name(trajectory.true_class),),
        tier_definition=_tier_by_name(tier_name),
        trajectories=(trajectory,),
    )


def _build_trajectory(
    *,
    config: SequentialBoundaryControlConfig,
    proposal_id: str,
    seed: int,
    positions: tuple[float, ...],
    velocities: tuple[float, ...],
    accelerations: tuple[float, ...],
    measurements: tuple[float, ...],
) -> TrajectoryArtifact:
    times = tuple(index * config.dt for index in range(len(measurements)))
    return _make_manual_trajectory(
        trajectory_id=proposal_id,
        true_class=config.class_name,
        tier=config.tier_name,
        scenario_family=config.scenario_family,
        measurements=measurements,
        times=times,
        true_position=positions,
        true_velocity=velocities,
        true_acceleration=accelerations,
        measurement_std=config.measurement_std,
        outlier_indices=[],
        seed=seed,
        generator_parameters={
            "dt": config.dt,
            "episode_horizon": config.episode_horizon,
            "acceleration_limit": config.acceleration_limit,
        },
    )


def evaluate_boundary_control_rollout(
    objective: TrajectoryExplorationObjective,
    proposal: TrajectoryExplorationProposal,
    trajectory: TrajectoryArtifact,
    *,
    rollout_return: float,
) -> TrajectoryExplorationEvaluation:
    tier_name = objective.target.target_tier or "boundary_v1"
    class_validity = _class_validity_score(trajectory, tier_name)
    feature_excitation = _feature_excitation_score(objective.target, trajectory, tier_name)
    coverage_gain = _coverage_gain_score(objective.target, proposal.action, trajectory, tier_name)
    boundary_closeness = _boundary_closeness_score(objective.target, trajectory, tier_name)
    prior_sensitivity = _prior_sensitivity_score(objective.target, trajectory, tier_name)
    leakage_penalty = _leakage_penalty(proposal.action, trajectory, tier_name)
    physical_invalidity_penalty = _physical_invalidity_penalty(trajectory)
    reward = _reward_from_components(
        class_validity=class_validity,
        feature_excitation=feature_excitation,
        coverage_gain=coverage_gain,
        boundary_closeness=boundary_closeness,
        classifier_stress=boundary_closeness,
        prior_sensitivity=prior_sensitivity,
        leakage_penalty=leakage_penalty,
        physical_invalidity_penalty=physical_invalidity_penalty,
    )
    context = _one_dimensional_feature_context_from_trajectory(_trajectory_dataset(trajectory, tier_name), trajectory)
    geometry_score = mean((boundary_closeness, feature_excitation, prior_sensitivity))
    diagnostics = {
        "action_mode": "sequential_control",
        "control_sequence_length": len(proposal.control_sequence or ()),
        "rollout_return": rollout_return,
        "duration": context.duration,
        "acceleration_range": context.acceleration_range,
        "acceleration_variance": context.acceleration_variance,
        "sampling_irregularity": context.sampling_irregularity,
        "measurement_scale": proposal.action.measurement_scale,
        "duration_scale": proposal.action.duration_scale,
        "irregularity_scale": proposal.action.irregularity_scale,
        "outlier_scale": proposal.action.outlier_scale,
        "step_scale": proposal.action.step_scale,
    }
    posterior_target_distribution = objective.backend_constraints.get("posterior_target_distribution")
    total_utility = 0.5 * reward.total_utility + 0.5 * rollout_return
    if isinstance(posterior_target_distribution, dict):
        posterior_spec = posterior_target_spec_from_payload(
            objective_id=objective.objective_id,
            target_distribution={str(key): float(value) for key, value in posterior_target_distribution.items()},
            evidence_provider_id=str(objective.backend_constraints.get("evidence_provider_id", "class_similarity_proxy_v1")),
        )
        posterior_score = score_posterior_target_distribution(
            posterior_spec,
            trajectory,
            action=proposal.action,
            candidate_id=proposal.proposal_id,
            backend_id=proposal.backend_id,
            tier_name=tier_name,
        )
        geometry_score = posterior_score.score
        boundary_closeness = posterior_score.score
        diagnostics.update(posterior_score.primary_terms)
        diagnostics.update(posterior_score.penalties)
        diagnostics.update(posterior_score.metadata)
        diagnostics["posterior_target_passed_constraints"] = posterior_score.passed_constraints
        total_utility = _clamp(
            0.55 * posterior_score.score
            + 0.25 * class_validity
            + 0.10 * feature_excitation
            + 0.10 * prior_sensitivity
            - 0.15 * leakage_penalty
            - 0.15 * physical_invalidity_penalty,
            0.0,
            1.0,
        )
    return TrajectoryExplorationEvaluation(
        proposal_id=proposal.proposal_id,
        backend_id=proposal.backend_id,
        objective_id=objective.objective_id,
        iteration=proposal.iteration,
        candidate_index=proposal.candidate_index,
        target_id=objective.target.target_id,
        trajectory_id=trajectory.trajectory_id,
        true_class=trajectory.true_class,
        total_utility=total_utility,
        class_validity=class_validity,
        feature_excitation=feature_excitation,
        coverage_gain=coverage_gain,
        boundary_closeness=boundary_closeness,
        classifier_stress=boundary_closeness,
        prior_sensitivity=prior_sensitivity,
        leakage_penalty=leakage_penalty,
        physical_invalidity_penalty=physical_invalidity_penalty,
        feature_cell_coverage_gain=feature_excitation,
        class_pair_overlap_reduction=1.0 - boundary_closeness,
        pairwise_auc_gain=geometry_score,
        pca_margin_gain=geometry_score,
        confusion_witness_score=boundary_closeness,
        feature_dependency_stress=context.acceleration_variance,
        prior_flip_witness_score=prior_sensitivity,
        geometry_score=geometry_score,
        diagnostics=diagnostics,
    )


class SequentialTrajectoryGym(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        objective: TrajectoryExplorationObjective | None = None,
        config: SequentialBoundaryControlConfig | None = None,
        *,
        seed: int = 7,
    ) -> None:
        super().__init__()
        self.objective = objective or default_boundary_control_objective()
        self.config = config or SequentialBoundaryControlConfig()
        self._episode_seed = seed
        self._rng = numpy.random.default_rng(seed)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(9,), dtype=numpy.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=numpy.float32)
        self._step_index = 0
        self._position = 0.0
        self._velocity = 0.0
        self._acceleration = 0.0
        self._positions: list[float] = []
        self._velocities: list[float] = []
        self._accelerations: list[float] = []
        self._measurements: list[float] = []
        self._controls: list[float] = []
        self._rewards: list[float] = []
        self._last_summary: SequentialRolloutSummary | None = None
        self._last_partial_utility = 0.0

    @property
    def last_summary(self) -> SequentialRolloutSummary | None:
        return self._last_summary

    def _target_descriptor(self) -> float:
        pair = self.objective.target.class_pair or ("constant_velocity", "constant_acceleration")
        return 1.0 if pair == ("constant_velocity", "constant_acceleration") else 0.0

    def _partial_indicators(self) -> tuple[float, float]:
        if len(self._accelerations) < 2:
            return 0.0, 0.0
        accel_range = max(self._accelerations) - min(self._accelerations)
        speed_span = max(abs(value) for value in self._velocities) if self._velocities else 0.0
        return _clamp(accel_range / max(self.config.acceleration_limit, 1e-6), 0.0, 1.0), _clamp(speed_span / max(self.config.velocity_limit, 1e-6), 0.0, 1.0)

    def _observation(self) -> numpy.ndarray:
        horizon_fraction = 1.0 - (self._step_index / max(self.config.episode_horizon, 1))
        accel_indicator, speed_indicator = self._partial_indicators()
        reward_mean = float(mean(self._rewards)) if self._rewards else 0.0
        return numpy.asarray(
            [
                self._step_index / max(self.config.episode_horizon, 1),
                self._position / max(self.config.position_limit, 1e-6),
                self._velocity / max(self.config.velocity_limit, 1e-6),
                self._acceleration / max(self.config.acceleration_limit, 1e-6),
                horizon_fraction,
                self._target_descriptor(),
                accel_indicator,
                speed_indicator,
                reward_mean,
            ],
            dtype=numpy.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict[str, object] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._episode_seed = int(seed)
            self._rng = numpy.random.default_rng(seed)
        self._step_index = 0
        self._position = float(self._rng.uniform(-1.0, 1.0))
        self._velocity = float(self._rng.uniform(-0.45, 0.45))
        self._acceleration = float(self._rng.uniform(-0.10, 0.10))
        self._positions = [self._position]
        self._velocities = [self._velocity]
        self._accelerations = [self._acceleration]
        self._measurements = [self._position + float(self._rng.normal(0.0, self.config.measurement_std))]
        self._controls = []
        self._rewards = []
        self._last_summary = None
        self._last_partial_utility = 0.0
        return self._observation(), {"target_id": self.objective.target.target_id}

    def _integrate(self, jerk_command: float) -> None:
        self._acceleration = _clamp(jerk_command, -1.0, 1.0) * self.config.acceleration_limit
        if self.config.process_noise_std > 0.0:
            self._acceleration += float(self._rng.normal(0.0, self.config.process_noise_std))
            self._acceleration = _clamp(self._acceleration, -self.config.acceleration_limit, self.config.acceleration_limit)
        self._velocity = _clamp(self._velocity + self._acceleration * self.config.dt, -self.config.velocity_limit, self.config.velocity_limit)
        self._position = _clamp(self._position + self._velocity * self.config.dt, -self.config.position_limit, self.config.position_limit)
        self._positions.append(self._position)
        self._velocities.append(self._velocity)
        self._accelerations.append(self._acceleration)
        self._measurements.append(self._position + float(self._rng.normal(0.0, self.config.measurement_std)))

    def _current_partial_utility(self) -> float:
        summary = self.evaluate_current_rollout(
            proposal_id=f"partial_rollout_{self._episode_seed}",
            backend_id="sequential_env_partial",
            iteration=0,
            candidate_index=0,
            rollout_return=sum(self._rewards),
        )
        return summary.evaluation.total_utility

    def step(self, action: numpy.ndarray):
        jerk_command = _clamp(float(numpy.asarray(action, dtype=float).reshape(-1)[0]), -1.0, 1.0)
        self._controls.append(jerk_command)
        self._integrate(jerk_command)
        self._step_index += 1
        current_utility = self._current_partial_utility()
        step_reward = current_utility - self._last_partial_utility
        self._last_partial_utility = current_utility
        terminated = self._step_index >= self.config.episode_horizon
        truncated = False
        if terminated:
            rollout_return = sum(self._rewards) + step_reward
            summary = self.evaluate_current_rollout(
                proposal_id=f"env_rollout_{self._episode_seed}",
                backend_id="sequential_env",
                iteration=0,
                candidate_index=0,
                rollout_return=rollout_return,
            )
            self._last_summary = summary
            step_reward += 2.5 * summary.evaluation.total_utility
        self._rewards.append(step_reward)
        info = {"target_id": self.objective.target.target_id, "step_reward": step_reward, "control": jerk_command}
        if terminated and self._last_summary is not None:
            info["evaluation"] = self._last_summary.evaluation.as_row()
        return self._observation(), float(step_reward), terminated, truncated, info

    def evaluate_current_rollout(
        self,
        *,
        proposal_id: str,
        backend_id: str,
        iteration: int,
        candidate_index: int,
        rollout_return: float | None = None,
    ) -> SequentialRolloutSummary:
        proposal = TrajectoryExplorationProposal(
            proposal_id=proposal_id,
            backend_id=backend_id,
            iteration=iteration,
            candidate_index=candidate_index,
            action=CorpusGymAction(
                seed=self._episode_seed,
                tier_name=self.config.tier_name,
                duration_scale=1.0,
                measurement_scale=max(0.5, self.config.measurement_std / 0.04),
                irregularity_scale=1.0,
                outlier_scale=1.0,
                step_scale=1.0,
                metadata={"mean_abs_control": mean(abs(value) for value in self._controls) if self._controls else 0.0},
            ),
            control_sequence=tuple(self._controls),
            metadata={"action_mode": "sequential_control"},
        )
        trajectory = _build_trajectory(
            config=self.config,
            proposal_id=proposal_id,
            seed=self._episode_seed,
            positions=tuple(self._positions),
            velocities=tuple(self._velocities),
            accelerations=tuple(self._accelerations),
            measurements=tuple(self._measurements),
        )
        evaluation = evaluate_boundary_control_rollout(
            self.objective,
            proposal,
            trajectory,
            rollout_return=sum(self._rewards) if rollout_return is None else rollout_return,
        )
        return SequentialRolloutSummary(
            proposal=proposal,
            evaluation=evaluation,
            trajectory_id=trajectory.trajectory_id,
            control_sequence=tuple(self._controls),
            times=tuple(trajectory.times),
            positions=tuple(self._positions),
            velocities=tuple(self._velocities),
            accelerations=tuple(self._accelerations),
            measurements=tuple(self._measurements),
            reward_trace=tuple(self._rewards),
            reward_components={
                "total_utility": evaluation.total_utility,
                "boundary_closeness": evaluation.boundary_closeness,
                "feature_excitation": evaluation.feature_excitation,
                "class_validity": evaluation.class_validity,
                "prior_sensitivity": evaluation.prior_sensitivity,
                "leakage_penalty": evaluation.leakage_penalty,
                "physical_invalidity_penalty": evaluation.physical_invalidity_penalty,
            },
        )


def evaluate_control_sequence(
    control_sequence: tuple[float, ...],
    *,
    objective: TrajectoryExplorationObjective | None = None,
    config: SequentialBoundaryControlConfig | None = None,
    seed: int = 7,
    proposal_id: str = "control_sequence",
    backend_id: str = "sequential_baseline",
    iteration: int = 0,
    candidate_index: int = 0,
) -> SequentialRolloutSummary:
    environment = SequentialTrajectoryGym(objective=objective, config=config, seed=seed)
    environment.reset(seed=seed)
    for value in control_sequence:
        _, _, terminated, truncated, _ = environment.step(numpy.asarray([value], dtype=numpy.float32))
        if terminated or truncated:
            break
    return environment.evaluate_current_rollout(
        proposal_id=proposal_id,
        backend_id=backend_id,
        iteration=iteration,
        candidate_index=candidate_index,
    )


def scripted_control_profiles(config: SequentialBoundaryControlConfig | None = None) -> dict[str, tuple[float, ...]]:
    resolved = config or SequentialBoundaryControlConfig()
    horizon = resolved.episode_horizon
    third = max(1, horizon // 3)
    return {
        "constant_positive": tuple(0.55 for _ in range(horizon)),
        "constant_negative": tuple(-0.55 for _ in range(horizon)),
        "bang_bang": tuple(0.9 if index < horizon // 2 else -0.9 for index in range(horizon)),
        "pulse": tuple(0.0 if index < third else 0.85 if index < 2 * third else -0.35 for index in range(horizon)),
        "ramp": tuple(-0.7 + 1.4 * index / max(horizon - 1, 1) for index in range(horizon)),
        "triangular": tuple(1.0 - 4.0 * abs((index / max(horizon - 1, 1)) - 0.5) for index in range(horizon)),
    }


def doe_schedule_bank(config: SequentialBoundaryControlConfig | None = None) -> tuple[tuple[float, ...], ...]:
    resolved = config or SequentialBoundaryControlConfig()
    horizon = resolved.episode_horizon
    onset_values = (3, max(4, horizon // 2), max(5, horizon - 4))
    amplitudes = (-0.8, -0.35, 0.35, 0.8)
    durations = (3, 5, 7)
    bank: list[tuple[float, ...]] = []
    for onset in onset_values:
        for amplitude in amplitudes:
            for duration in durations:
                schedule = [0.0] * horizon
                for index in range(onset, min(horizon, onset + duration)):
                    schedule[index] = amplitude
                bank.append(tuple(schedule))
    return tuple(bank)


def guided_mutation_bank(base_sequence: tuple[float, ...], *, amplitude: float = 0.18) -> tuple[tuple[float, ...], ...]:
    mutants: list[tuple[float, ...]] = []
    for pivot in range(len(base_sequence)):
        raised = list(base_sequence)
        raised[pivot] = _clamp(raised[pivot] + amplitude, -1.0, 1.0)
        mutants.append(tuple(raised))
        lowered = list(base_sequence)
        lowered[pivot] = _clamp(lowered[pivot] - amplitude, -1.0, 1.0)
        mutants.append(tuple(lowered))
    return tuple(mutants)


def sequential_environment_contract(config: SequentialBoundaryControlConfig | None = None) -> dict[str, object]:
    resolved = config or SequentialBoundaryControlConfig()
    return {
        "environment_id": "sequential_boundary_control",
        "control_problem": resolved.control_problem.as_payload(),
        "three_d_point_mass_path": default_three_dimensional_point_mass_problem_spec().as_payload(),
        "air_vehicle_path": default_air_vehicle_control_problem_spec().as_payload(),
        "state_fields": [
            "step_fraction",
            "position",
            "velocity",
            "acceleration",
            "remaining_horizon_fraction",
            "target_descriptor",
            "partial_acceleration_range",
            "partial_speed_range",
            "mean_reward_so_far",
        ],
        "action_fields": ["normalized_acceleration_command"],
        "episode_horizon": resolved.episode_horizon,
        "dt": resolved.dt,
        "acceleration_limit": resolved.acceleration_limit,
        "measurement_std": resolved.measurement_std,
        "witness_task": "boundary_shaping_cv_vs_ca",
        "sequential_mode": True,
        "config": asdict(resolved),
    }
