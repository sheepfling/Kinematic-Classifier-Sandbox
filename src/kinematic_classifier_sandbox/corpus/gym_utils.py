from __future__ import annotations

import random
from dataclasses import replace

from ..analysis.feature_analysis import _one_dimensional_feature_context_from_trajectory
from ..trajectory_generator import (
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
from ..utils.math import _clamp
from .gym_types import CorpusGymAction, CorpusGymReward, CorpusGymTarget
from .policy import CorpusPolicySpec, load_corpus_policy_spec, score_corpus_gym_reward


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
    policy: CorpusPolicySpec | None = None,
    class_validity: float,
    feature_excitation: float,
    coverage_gain: float,
    boundary_closeness: float,
    classifier_stress: float,
    prior_sensitivity: float,
    leakage_penalty: float,
    physical_invalidity_penalty: float,
) -> CorpusGymReward:
    resolved_policy = policy or load_corpus_policy_spec()
    total_utility = score_corpus_gym_reward(
        resolved_policy,
        class_validity=class_validity,
        feature_excitation=feature_excitation,
        coverage_gain=coverage_gain,
        boundary_closeness=boundary_closeness,
        classifier_stress=classifier_stress,
        prior_sensitivity=prior_sensitivity,
        leakage_penalty=leakage_penalty,
        physical_invalidity_penalty=physical_invalidity_penalty,
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
    trajectory = _make_trajectory(
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
    generator_parameters = dict(trajectory.generator_parameters)
    generator_parameters.update(action.metadata)
    generator_parameters["corpus_gym_target_id"] = target.target_id
    generator_parameters["corpus_gym_target_type"] = target.target_type
    generator_parameters["tier"] = tier_definition.name
    return replace(trajectory, generator_parameters=generator_parameters)
